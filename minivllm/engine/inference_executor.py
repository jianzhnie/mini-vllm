"""Inference Executor module for handling model inference execution.

This module provides the InferenceExecutor class which handles:
- KV cache management and allocation
- CUDA Graph optimization for efficient decode
- Token sampling and generation
- Batch execution and optimization
- Performance monitoring and metrics collection
- Input preparation for both prefill and decode phases
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from minivllm.config import Config
from minivllm.engine.sequence import Sequence
from minivllm.sampling.sampler import Sampler
from minivllm.utils.context import reset_context, set_context
from minivllm.utils.device import (
    empty_cache,
    get_current_device,
    mem_get_info,
    memory_stats,
    move_tensor_to_device,
    reset_peak_memory_stats,
    should_use_pin_memory,
    supports_cuda_graph,
    synchronize,
)
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ['InferenceExecutor']


class InferenceExecutor:
    """Handles model inference execution and optimization.

    This class is responsible for the actual execution of model inference,
    including KV cache management, CUDA graph optimization, and efficient
    batch processing.

    Attributes:
        config: Engine configuration.
        model: The loaded language model.
        device: Current device for execution.
        sampler: Token sampler for generation.
        kv_cache: Pre-allocated tensor for KV cache.
        block_size: Number of tokens per cache block.
        enforce_eager: Whether to skip device graph optimization.
        graphs: Dictionary of captured device graphs for different batch sizes.
        graph_vars: Shared tensors for device graphs.
    """

    def __init__(self, config: Config, model: Any) -> None:
        """Initialize the inference executor.

        Args:
            config: Engine configuration.
            model: Loaded language model.
        """
        self.config = config
        self.model = model
        self.device = get_current_device()
        self.sampler: Optional[Sampler] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.block_size = config.kvcache_block_size

        # Extract model config
        hf_config = config.hf_config
        self.num_kv_heads = getattr(
            hf_config, 'num_key_value_heads',
            getattr(hf_config, 'num_attention_heads', 1))
        self.head_dim = getattr(
            hf_config, 'head_dim',
            hf_config.hidden_size // hf_config.num_attention_heads)
        self.num_layers = hf_config.num_hidden_layers

        # Use model's actual dtype if model is available, otherwise resolve from config
        if hasattr(model, 'parameters'):
            try:
                first_param = next(model.parameters())
                self.dtype = first_param.dtype
            except StopIteration:
                self.dtype = self._resolve_dtype()
        else:
            self.dtype = self._resolve_dtype()

        self._init_attributes()

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve the correct dtype for model inference.

        Returns:
            The torch dtype to use for the model.
        """
        dtype_str = str(self.config.dtype).lower()

        # Standard dtype mappings
        dtype_map = {
            'float32': torch.float32,
            'fp32': torch.float32,
            'float': torch.float32,
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'auto': getattr(self.config.hf_config, 'torch_dtype',
                            torch.float16),
        }

        if dtype_str in dtype_map:
            return dtype_map[dtype_str]

        # Fallback: try to get from torch directly
        try:
            return getattr(torch, dtype_str)
        except AttributeError:
            logger.warning(
                f"Unknown dtype '{dtype_str}', defaulting to float32")
            return torch.float32

    def _init_attributes(self) -> None:
        """Initialize remaining attributes after dtype is resolved."""
        self.enforce_eager = self.config.enforce_eager
        self.graphs: Dict[int, Any] = {}
        self.graph_vars: Dict[str, torch.Tensor] = {}
        self.graph_bs: List[int] = []
        self.graph_pool: Optional[Any] = None

        # Performance metrics
        self.total_tokens_generated = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.inference_count = 0

        logger.debug(f'InferenceExecutor initialized on device {self.device}')

    def initialize(self, max_num_batched_tokens: int,
                   max_num_seqs: int) -> None:
        """Initialize the executor with cache and sampler.

        Args:
            max_num_batched_tokens: Maximum tokens in a batch.
            max_num_seqs: Maximum sequences in a batch.
        """
        self._allocate_kv_cache()
        self._initialize_sampler()
        self._optimize_model()
        self._warmup_model(max_num_batched_tokens, max_num_seqs)

        logger.info('Inference executor initialized successfully')

    def _allocate_kv_cache(self) -> None:
        """Allocate KV cache memory based on configuration."""
        try:
            num_blocks = self.config.num_kvcache_blocks

            # Auto-calculate num_blocks if not set
            if num_blocks <= 0:
                num_blocks = self._calculate_num_kv_blocks()
                self.config.num_kvcache_blocks = num_blocks

            # Calculate KV cache shape: (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
            kv_cache_shape = (2, self.num_layers, num_blocks, self.block_size,
                              self.num_kv_heads, self.head_dim)

            # Allocate contiguous KV cache tensor
            self.kv_cache = torch.empty(kv_cache_shape,
                                        device=self.device,
                                        dtype=self.dtype)

            # Assign cache slices to model layers
            self._assign_kv_cache_to_layers()

            cache_size_gb = self.kv_cache.numel() * self.kv_cache.element_size(
            ) / (1024**3)
            logger.info(f'Allocated KV cache: {cache_size_gb:.2f} GB '
                        f'({num_blocks} blocks x {self.block_size} tokens)')

        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f'OOM during KV cache allocation. '
                f'Try reducing device_memory_utilization (current: {self.config.device_memory_utilization}) '
                f'or max_num_seqs. Error: {e}') from e
        except Exception as e:
            raise RuntimeError(f'Failed to allocate KV cache: {e}') from e

    def _calculate_num_kv_blocks(self) -> int:
        """Calculate number of KV cache blocks based on available memory.

        Returns:
            Number of KV cache blocks to allocate.
        """
        config = self.config

        # Get device memory information
        try:
            free, total = mem_get_info(self.device)
        except Exception as e:
            logger.warning(f'Failed to get memory info: {e}, using defaults')
            free, total = 4 * 1024**3, 8 * 1024**3  # Default 4GB free, 8GB total

        used = total - free
        stats = memory_stats(self.device)
        peak = stats.get('allocated_bytes.all.peak', 0)
        current = stats.get('allocated_bytes.all.current', 0)

        # Calculate KV cache requirements per block
        block_bytes = (2 * self.num_layers * self.block_size *
                       self.num_kv_heads * self.head_dim * self.dtype.itemsize)

        # Calculate available memory for KV cache
        if self.device.type == 'cpu':
            available_memory = int(free * config.device_memory_utilization)
        else:
            available_memory = int(total * config.device_memory_utilization -
                                   used - peak + current)

        if available_memory <= 0:
            raise RuntimeError(
                f'Insufficient memory for KV cache allocation. '
                f'Device: {self.device.type}, Available: {available_memory} bytes, '
                f'Block size: {block_bytes} bytes')

        # Calculate max blocks per sequence and total
        max_blocks_per_seq = ((config.max_model_len + self.block_size - 1) //
                              self.block_size)
        max_total_blocks = int(config.max_num_seqs * max_blocks_per_seq)

        # Calculate number of blocks from available memory
        num_blocks = int(available_memory // block_bytes)

        # Cap at maximum needed
        if num_blocks > max_total_blocks:
            num_blocks = max_total_blocks

        if num_blocks <= 0:
            raise ValueError(
                f'Calculated KV cache blocks ({num_blocks}) must be > 0. '
                f'Available memory: {available_memory} bytes, '
                f'Block size: {block_bytes} bytes')

        logger.debug(f'Calculated KV cache blocks: {num_blocks} '
                     f'({available_memory / 1024**3:.2f} GB available, '
                     f'{block_bytes} bytes/block)')

        return num_blocks

    def _assign_kv_cache_to_layers(self) -> None:
        """Assign KV cache slices to model attention layers."""
        if self.kv_cache is None:
            return

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                # Assign K and V cache for this layer
                module.k_cache = self.kv_cache[0, layer_id]  # K cache
                module.v_cache = self.kv_cache[1, layer_id]  # V cache
                layer_id += 1

        if layer_id == 0:
            logger.warning(
                'No attention layers found with k_cache/v_cache attributes. '
                'KV cache may not be properly used.')
        else:
            logger.debug(f'Assigned KV cache to {layer_id} attention layers')

    def _initialize_sampler(self) -> None:
        """Initialize the token sampler."""
        self.sampler = Sampler()
        self.sampler = self.sampler.to(self.device)
        logger.debug('Token sampler initialized')

    def _optimize_model(self) -> None:
        """Apply model optimizations."""
        # Set model to evaluation mode
        self.model.eval()

        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

        # Disable gradient checkpointing if present
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

        # Enable inference mode optimizations
        if hasattr(torch, 'inference_mode'):
            logger.debug('Model optimizations applied')

    def _warmup_model(self, max_num_batched_tokens: int,
                      max_num_seqs: int) -> None:
        """Warmup model by running inference on dummy data.

        Args:
            max_num_batched_tokens: Maximum tokens in a batch.
            max_num_seqs: Maximum sequences in a batch.
        """
        empty_cache()
        reset_peak_memory_stats(self.device)

        # Calculate warmup batch size
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max(max_model_len // 2, 1),
                       max_num_seqs, 2)

        # Create dummy sequences for warmup
        dummy_sequences = [
            Sequence([0] * min(16, max_model_len)) for _ in range(num_seqs)
        ]

        # Warmup prefill
        with torch.inference_mode():
            self.execute_batch(dummy_sequences, prefill=True)

        # Warmup decode
        for seq in dummy_sequences:
            seq.append_token(0)

        with torch.inference_mode():
            self.execute_batch(dummy_sequences, prefill=False)

        empty_cache()
        logger.debug(f'Model warmup completed with {num_seqs} sequences')

    def execute_batch(self,
                      sequences: List[Sequence],
                      prefill: bool = True) -> Tuple[torch.Tensor, List[int]]:
        """Execute inference for a batch of sequences.

        Args:
            sequences: List of sequences to process.
            prefill: Whether this is a prefill step.

        Returns:
            Tuple of (logits, next_tokens).
        """
        if not sequences:
            return torch.empty(0, device=self.device), []

        # Use CUDA events for timing if available
        use_cuda_events = self.device.type == 'cuda'
        if use_cuda_events:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        else:
            import time
            start_time = time.perf_counter()

        try:
            # Prepare batch inputs
            input_ids, positions = self._prepare_batch_input(
                sequences, prefill)

            # Execute model
            logits = self._execute_model(input_ids, positions, prefill)

            # Sample next tokens (only need last token logits for prefill)
            if prefill and logits.size(0) > len(sequences):
                # Get logits for last token of each sequence
                seq_lengths = [len(seq) for seq in sequences]
                last_indices = torch.cumsum(
                    torch.tensor(seq_lengths, device=logits.device), dim=0) - 1
                logits = logits[last_indices]

            next_tokens = self._sample_tokens(logits, sequences)

            # Update metrics
            self._update_metrics(sequences, prefill)

            if use_cuda_events:
                end_time.record()
                torch.cuda.synchronize()
            self.inference_count += 1
            return logits, next_tokens

        except Exception as e:
            logger.error(f'Batch execution failed: {e}')
            raise
        finally:
            reset_context()

    def _prepare_batch_input(
            self, sequences: List[Sequence],
            prefill: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors for the batch.

        Args:
            sequences: List of sequences.
            prefill: Whether this is prefill or decode.

        Returns:
            Tuple of (input_ids, positions) tensors.
        """
        if prefill:
            return self._prepare_prefill_input(sequences)
        else:
            return self._prepare_decode_input(sequences)

    def _prepare_prefill_input(
            self,
            sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors for prefill phase.

        Args:
            sequences: List of sequences in prefill phase.

        Returns:
            Tuple of (input_ids, positions) tensors.
        """
        input_ids: List[int] = []
        positions: List[int] = []
        cum_seqlens_q: List[int] = [0]
        cum_seqlens_k: List[int] = [0]
        max_seqlen_q: int = 0
        max_seqlen_k: int = 0
        slot_mapping: List[int] = []

        for seq in sequences:
            seqlen = len(seq)
            num_cached = seq.num_cached_tokens

            # Only process uncached tokens
            input_ids.extend(seq.token_ids[num_cached:])
            positions.extend(range(num_cached, seqlen))

            seqlen_q = seqlen - num_cached
            seqlen_k = seqlen
            cum_seqlens_q.append(cum_seqlens_q[-1] + seqlen_q)
            cum_seqlens_k.append(cum_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # Slot mapping for KV cache placement
            if seq.block_table:
                for i in range(seq.num_cached_blocks, seq.num_blocks):
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens
                    slot_mapping.extend(range(start, end))

        # Convert to tensors
        pin_mem = should_use_pin_memory(self.device)

        input_ids_tensor = torch.tensor(input_ids,
                                        dtype=torch.long,
                                        pin_memory=pin_mem)
        input_ids_tensor = move_tensor_to_device(input_ids_tensor,
                                                 self.device,
                                                 non_blocking=True)

        positions_tensor = torch.tensor(positions,
                                        dtype=torch.long,
                                        pin_memory=pin_mem)
        positions_tensor = move_tensor_to_device(positions_tensor,
                                                 self.device,
                                                 non_blocking=True)

        # Prepare context tensors
        cum_seqlens_q_tensor = torch.tensor(cum_seqlens_q,
                                            dtype=torch.int32,
                                            pin_memory=pin_mem)
        cum_seqlens_q_tensor = move_tensor_to_device(cum_seqlens_q_tensor,
                                                     self.device,
                                                     non_blocking=True)

        cum_seqlens_k_tensor = torch.tensor(cum_seqlens_k,
                                            dtype=torch.int32,
                                            pin_memory=pin_mem)
        cum_seqlens_k_tensor = move_tensor_to_device(cum_seqlens_k_tensor,
                                                     self.device,
                                                     non_blocking=True)

        slot_mapping_tensor = None
        if slot_mapping:
            slot_mapping_tensor = torch.tensor(slot_mapping,
                                               dtype=torch.int32,
                                               pin_memory=pin_mem)
            slot_mapping_tensor = move_tensor_to_device(slot_mapping_tensor,
                                                        self.device,
                                                        non_blocking=True)

        # Prepare block tables for prefix caching
        block_tables = None
        if any(seq.block_table for seq in sequences):
            block_tables = self._prepare_block_tables(sequences)

        # Set context for attention kernels
        set_context(
            is_prefill=True,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cum_seqlens_q=cum_seqlens_q_tensor,
            cum_seqlens_k=cum_seqlens_k_tensor,
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=block_tables,
        )

        return input_ids_tensor, positions_tensor

    def _prepare_decode_input(
            self,
            sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors for decode phase.

        Args:
            sequences: List of sequences in decode phase.

        Returns:
            Tuple of (input_ids, positions) tensors.
        """
        # For decode, only process the last token of each sequence
        input_ids = [seq.last_token for seq in sequences]
        positions = [len(seq) - 1 for seq in sequences]
        context_lens = [len(seq) for seq in sequences]

        # Handle empty block tables during warmup
        slot_mapping = []
        has_empty_block_table = False
        for seq in sequences:
            if seq.block_table:
                slot = (seq.block_table[-1] * self.block_size +
                        seq.last_block_num_tokens - 1)
                slot_mapping.append(slot)
            else:
                # No block assigned yet - use a dummy slot (will be ignored)
                slot_mapping.append(0)
                has_empty_block_table = True

        if has_empty_block_table:
            logger.warning(
                'Some sequences have empty block tables during decode. '
                'This is expected during warmup but may indicate an issue during actual inference.'
            )

        pin_mem = should_use_pin_memory(self.device)

        input_ids_tensor = torch.tensor(input_ids,
                                        dtype=torch.long,
                                        pin_memory=pin_mem)
        input_ids_tensor = move_tensor_to_device(input_ids_tensor,
                                                 self.device,
                                                 non_blocking=True)

        positions_tensor = torch.tensor(positions,
                                        dtype=torch.long,
                                        pin_memory=pin_mem)
        positions_tensor = move_tensor_to_device(positions_tensor,
                                                 self.device,
                                                 non_blocking=True)

        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.int32,
                                           pin_memory=pin_mem)
        slot_mapping_tensor = move_tensor_to_device(slot_mapping_tensor,
                                                    self.device,
                                                    non_blocking=True)

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int32,
                                           pin_memory=pin_mem)
        context_lens_tensor = move_tensor_to_device(context_lens_tensor,
                                                    self.device,
                                                    non_blocking=True)

        block_tables = self._prepare_block_tables(sequences)

        # Set context for attention kernels
        set_context(
            is_prefill=False,
            max_seqlen_q=0,
            max_seqlen_k=0,
            cum_seqlens_q=None,
            cum_seqlens_k=None,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
        )

        return input_ids_tensor, positions_tensor

    def _prepare_block_tables(self, sequences: List[Sequence]) -> torch.Tensor:
        """Prepare block tables for sequences.

        Args:
            sequences: List of sequences.

        Returns:
            A 2D integer tensor of shape (num_seqs, max_blocks).
        """
        if not sequences:
            return torch.empty((0, 0), dtype=torch.int32, device=self.device)

        max_blocks = max(len(seq.block_table) for seq in sequences)
        if max_blocks == 0:
            return torch.empty((len(sequences), 0),
                               dtype=torch.int32,
                               device=self.device)

        # Pad block tables to same length
        block_tables = [
            seq.block_table + [-1] * (max_blocks - len(seq.block_table))
            for seq in sequences
        ]

        pin_mem = should_use_pin_memory(self.device)
        block_tables_tensor = torch.tensor(block_tables,
                                           dtype=torch.int32,
                                           pin_memory=pin_mem)
        return move_tensor_to_device(block_tables_tensor,
                                     self.device,
                                     non_blocking=True)

    def _execute_model(self, input_ids: torch.Tensor, positions: torch.Tensor,
                       is_prefill: bool) -> torch.Tensor:
        """Execute model inference.

        Args:
            input_ids: Input token IDs tensor.
            positions: Token positions tensor.
            is_prefill: Whether this is prefill phase.

        Returns:
            Model logits tensor.
        """
        batch_size = input_ids.size(0)

        # Use CUDA graph for decode if available and enabled
        if (not is_prefill and not self.enforce_eager
                and supports_cuda_graph() and batch_size in self.graphs):
            return self._execute_with_cuda_graph(input_ids, positions)

        # Standard eager execution
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, positions=positions)
            if isinstance(output, tuple):
                output = output[0]
            return self.model.compute_logits(output)

    def _execute_with_cuda_graph(self, input_ids: torch.Tensor,
                                 positions: torch.Tensor) -> torch.Tensor:
        """Execute using captured CUDA graph.

        Args:
            input_ids: Input token IDs tensor.
            positions: Token positions tensor.

        Returns:
            Model logits tensor.
        """
        batch_size = input_ids.size(0)

        # Find the appropriate graph (smallest >= batch_size)
        graph_bs = next((bs for bs in self.graph_bs if bs >= batch_size), None)
        if graph_bs is None or graph_bs not in self.graphs:
            # Fallback to eager execution
            with torch.inference_mode():
                output = self.model(input_ids=input_ids, positions=positions)
                return self.model.compute_logits(output)

        graph = self.graphs[graph_bs]
        vars_dict = self.graph_vars

        # Update graph input tensors
        vars_dict['input_ids'][:batch_size] = input_ids
        vars_dict['positions'][:batch_size] = positions

        # Update context information from current context
        from minivllm.utils.context import get_context
        ctx = get_context()

        if ctx.slot_mapping is not None:
            vars_dict['slot_mapping'].fill_(-1)
            vars_dict['slot_mapping'][:batch_size] = ctx.slot_mapping

        if ctx.context_lens is not None:
            vars_dict['context_lens'].zero_()
            vars_dict['context_lens'][:batch_size] = ctx.context_lens

        if ctx.block_tables is not None:
            max_blocks = ctx.block_tables.size(1)
            vars_dict[
                'block_tables'][:batch_size, :max_blocks] = ctx.block_tables

        # Replay the captured graph
        graph.replay()

        # Return output for the actual batch size
        return self.model.compute_logits(vars_dict['outputs'][:batch_size])

    def _sample_tokens(self, logits: torch.Tensor,
                       sequences: List[Sequence]) -> List[int]:
        """Sample next tokens from logits.

        Args:
            logits: Model output logits.
            sequences: Input sequences for sampling parameters.

        Returns:
            List of sampled token IDs.
        """
        if self.sampler is None:
            return []

        # Extract sampling parameters
        device = logits.device
        temperatures = torch.tensor([seq.temperature for seq in sequences],
                                    device=device,
                                    dtype=torch.float32)
        top_ps = torch.tensor([seq.top_p for seq in sequences],
                              device=device,
                              dtype=torch.float32)
        top_ks = torch.tensor([seq.top_k for seq in sequences],
                              device=device,
                              dtype=torch.int64)
        min_ps = torch.tensor([seq.min_p for seq in sequences],
                              device=device,
                              dtype=torch.float32)

        # Sample tokens
        next_tokens = self.sampler(logits,
                                   temperatures=temperatures,
                                   top_ps=top_ps,
                                   top_ks=top_ks,
                                   min_ps=min_ps)

        return next_tokens.tolist()

    def _update_metrics(self, sequences: List[Sequence],
                        prefill: bool) -> None:
        """Update performance metrics.

        Args:
            sequences: Processed sequences.
            prefill: Whether this was prefill or decode.
        """
        if prefill:
            tokens = sum(len(seq) - seq.num_cached_tokens for seq in sequences)
            self.total_prefill_tokens += tokens
        else:
            self.total_decode_tokens += len(sequences)

        self.total_tokens_generated += len(sequences)

    def capture_device_graphs(self, max_batch_size: int) -> None:
        """Capture device graphs for efficient decode phase execution.

        Captures graphs for batch sizes: 1, 2, 4, 8, 16, 32, ...

        Args:
            max_batch_size: Maximum batch size to capture.
        """
        if self.enforce_eager or not supports_cuda_graph():
            logger.debug('Skipping device graph capture')
            return

        logger.info('Capturing device graphs for efficient decode...')

        try:
            # Determine batch sizes to capture
            self.graph_bs = [1, 2, 4, 8]
            self.graph_bs.extend(range(16, min(max_batch_size, 512) + 1, 16))

            max_blocks = (self.config.max_model_len + self.block_size -
                          1) // self.block_size

            # Pre-allocate tensors for graphs
            max_bs = self.graph_bs[-1]
            self.graph_vars = {
                'input_ids':
                torch.zeros(max_bs, dtype=torch.long, device=self.device),
                'positions':
                torch.zeros(max_bs, dtype=torch.long, device=self.device),
                'slot_mapping':
                torch.full((max_bs, ),
                           -1,
                           dtype=torch.int32,
                           device=self.device),
                'context_lens':
                torch.zeros(max_bs, dtype=torch.int32, device=self.device),
                'block_tables':
                torch.zeros(max_bs,
                            max_blocks,
                            dtype=torch.int32,
                            device=self.device),
                'outputs':
                torch.zeros(max_bs,
                            self.config.hf_config.hidden_size,
                            device=self.device),
            }

            # Capture graphs in reverse order (larger first for memory efficiency)
            for bs in reversed(self.graph_bs):
                self._capture_graph_for_batch_size(bs)

            logger.info(
                f'Captured {len(self.graphs)} device graphs for batch sizes: '
                f'{self.graph_bs[:4]}...{self.graph_bs[-1]}')

        except Exception as e:
            logger.warning(f'Failed to capture device graphs: {e}')
            self.graphs.clear()
            self.graph_vars.clear()

    def _capture_graph_for_batch_size(self, batch_size: int) -> None:
        """Capture CUDA graph for a specific batch size.

        Args:
            batch_size: Batch size to capture for.
        """
        try:
            # Get slices of pre-allocated tensors
            input_ids = self.graph_vars['input_ids'][:batch_size]
            positions = self.graph_vars['positions'][:batch_size]
            slot_mapping = self.graph_vars['slot_mapping'][:batch_size]
            context_lens = self.graph_vars['context_lens'][:batch_size]
            block_tables = self.graph_vars['block_tables'][:batch_size]
            outputs = self.graph_vars['outputs'][:batch_size]

            # Set context for graph capture
            set_context(is_prefill=False,
                        slot_mapping=slot_mapping,
                        context_lens=context_lens,
                        block_tables=block_tables)

            # Warmup
            outputs.copy_(
                self.model(input_ids=input_ids, positions=positions)[0])

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                outputs.copy_(
                    self.model(input_ids=input_ids, positions=positions)[0])

            # Create graph pool on first capture for memory efficiency
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[batch_size] = graph
            synchronize(self.device)
            reset_context()

        except Exception as e:
            logger.warning(
                f'Failed to capture graph for batch size {batch_size}: {e}')
            reset_context()

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics.

        Returns:
            Dictionary with performance metrics.
        """
        metrics = {
            'total_tokens_generated': self.total_tokens_generated,
            'total_prefill_tokens': self.total_prefill_tokens,
            'total_decode_tokens': self.total_decode_tokens,
            'inference_count': self.inference_count,
        }

        # Add memory statistics
        stats = memory_stats(self.device)
        metrics.update({
            'memory_allocated_mb':
            stats.get('allocated', 0) / 1024**2,
            'memory_reserved_mb':
            stats.get('reserved', 0) / 1024**2,
            'memory_max_allocated_mb':
            stats.get('max_allocated', 0) / 1024**2,
        })

        return metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.total_tokens_generated = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.inference_count = 0
        reset_peak_memory_stats(self.device)

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.debug('Cleaning up InferenceExecutor...')

        # Clear device graphs
        self.graphs.clear()
        self.graph_vars.clear()
        self.graph_pool = None

        # Clear KV cache
        if self.kv_cache is not None:
            del self.kv_cache
            self.kv_cache = None

        # Clear model and sampler references
        self.model = None
        self.sampler = None

        # Clear cache
        empty_cache()

        logger.debug('InferenceExecutor cleanup completed')
