"""Inference Executor module for handling model inference execution.

This module provides the InferenceExecutor class which handles:
- KV cache management and allocation
- CUDA Graph optimization
- Token sampling and generation
- Batch execution and optimization
- Performance monitoring and metrics collection
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
    memory_stats,
    move_tensor_to_device,
    reset_peak_memory_stats,
    should_use_pin_memory,
    supports_cuda_graph,
)
from minivllm.utils.logger_utils import get_logger
from minivllm.utils.memory_pool import get_memory_pool

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
        sampler: Token sampler for generation.
        kv_cache: Pre-allocated tensor for KV cache.
        block_size: Number of tokens per cache block.
        enforce_eager: Whether to skip device graph optimization.
        graphs: Dictionary of captured device graphs for different batch sizes.
        graph_vars: Shared tensors for device graphs.
        memory_pool: Memory pool for efficient allocation.
        performance_monitor: Performance monitoring system.
    """

    def __init__(self, config: Config, model: Any) -> None:
        """Initialize the inference executor.

        Args:
            config: Engine configuration.
            model: Loaded language model.
        """
        self.config = config
        self.model = model
        self.sampler: Optional[Sampler] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.block_size = config.kvcache_block_size
        if config.hf_config:
            self.num_kv_heads = getattr(
                config.hf_config, 'num_key_value_heads',
                getattr(config.hf_config, 'num_attention_heads', 1))
        else:
            self.num_kv_heads = 1
        self.enforce_eager = config.enforce_eager
        self.graphs: Dict[int, Any] = {}
        self.graph_vars: Dict[str, torch.Tensor] = {}

        # Memory pool for efficient allocation
        device = get_current_device()
        self.memory_pool = get_memory_pool(
            device=device,
            max_pool_size_mb=config.device_memory_utilization * 1024,
            block_size_mb=64)

        # Performance metrics (legacy compatibility)
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0

        logger.debug(f'InferenceExecutor initialized on device {device}')

    def initialize(self, max_num_batched_tokens: int,
                   max_num_seqs: int) -> None:
        """Initialize the executor with cache and sampler.

        Args:
            max_num_batched_tokens: Maximum tokens in a batch.
            max_num_seqs: Maximum sequences in a batch.
        """
        self._allocate_kv_cache(max_num_batched_tokens, max_num_seqs)
        self._initialize_sampler()
        self._optimize_model()

        total_size = sum(t.numel() * t.element_size() for pair in self.kv_cache
                         for t in pair)
        logger.info(f'Inference executor initialized. KV cache size: '
                    f'{total_size / 1024**2:.2f} MB')

    def _allocate_kv_cache(self, max_num_batched_tokens: int,
                           max_num_seqs: int) -> None:
        """Allocate KV cache memory pool.

        Args:
            max_num_batched_tokens: Maximum number of tokens in a batch.
            max_num_seqs: Maximum number of sequences in a batch.
        """
        try:
            # Resolve dtype
            dtype = self.config.dtype
            if isinstance(dtype, str):
                dtype_map = {
                    'float16': torch.float16,
                    'float32': torch.float32,
                    'bfloat16': torch.bfloat16,
                }
                dtype = dtype_map.get(dtype, torch.float16)

            kv_cache_shape = (self.config.num_kvcache_blocks, self.block_size,
                              self.num_kv_heads,
                              self.config.hf_config.hidden_size //
                              self.config.hf_config.num_attention_heads)

            # Ensure shape contains only integers
            kv_cache_shape = tuple(int(x) for x in kv_cache_shape)

            # Allocate KV cache for each layer
            num_layers = self.config.hf_config.num_hidden_layers
            self.kv_cache = []
            for _ in range(num_layers):
                k_cache = self.memory_pool.allocate(kv_cache_shape,
                                                    dtype=dtype)
                v_cache = self.memory_pool.allocate(kv_cache_shape,
                                                    dtype=dtype)
                self.kv_cache.append((k_cache, v_cache))

        except Exception as e:
            raise RuntimeError(f'Failed to allocate KV cache: {e}')

    def _initialize_sampler(self) -> None:
        """Initialize the token sampler."""
        self.sampler = Sampler()
        logger.debug('Token sampler initialized')

    def _optimize_model(self) -> None:
        """Apply model optimizations."""
        # Set model to evaluation mode
        self.model.eval()

        # Enable attention optimizations if available
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

        # Apply any additional optimizations
        self._apply_kv_cache_optimization()

        logger.debug('Model optimizations applied')

    def _apply_kv_cache_optimization(self) -> None:
        """Apply KV cache optimizations to the model."""
        if hasattr(self.model, 'set_kv_cache'):
            self.model.set_kv_cache(self.kv_cache, self.block_size)
        elif hasattr(self.model, 'kv_cache'):
            self.model.kv_cache = self.kv_cache
            self.model.block_size = self.block_size

    def _prepare_block_tables(self, sequences: List[Sequence]) -> torch.Tensor:
        """Prepare block tables for sequences.

        Args:
            sequences: List of sequences to prepare block tables for.

        Returns:
            A 2D integer tensor on device of shape (len(seqs), max_block_len)
            containing block table IDs with -1 padding.
        """
        device = get_current_device()
        max_len = max(len(seq.block_table) for seq in sequences)
        if max_len == 0:
            return torch.empty((len(sequences), 0),
                               dtype=torch.int32,
                               device=device)

        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in sequences
        ]

        block_tables_tensor = torch.tensor(
            block_tables,
            dtype=torch.int32,
            pin_memory=should_use_pin_memory(device))

        return move_tensor_to_device(block_tables_tensor,
                                     device,
                                     non_blocking=True)

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
            return torch.empty(0), []

        try:
            # Prepare batch inputs
            input_ids, positions, seq_lengths = self._prepare_batch_input(
                sequences, prefill)

            try:
                # Execute model
                if prefill:
                    logits = self._execute_prefill(input_ids, positions,
                                                   seq_lengths)
                    # For prefill, we only need logits for the last token of each sequence
                    if logits.dim() == 2 and logits.size(0) > len(sequences):
                        # Calculate indices of last tokens
                        last_token_indices = torch.cumsum(torch.tensor(
                            seq_lengths, device=logits.device),
                                                          dim=0) - 1
                        logits = logits[last_token_indices]
                else:
                    logits = self._execute_decode(input_ids, positions,
                                                  seq_lengths)

                # Sample next tokens
                next_tokens = self._sample_tokens(logits, sequences)

                # Update legacy metrics
                self.total_tokens_generated += len(next_tokens)

                return logits, next_tokens

            except Exception as e:
                logger.error(f'Batch execution failed: {e}')
                raise
        finally:
            reset_context()

    def _prepare_batch_input(self,
                             sequences: List[Sequence],
                             prefill: bool = False
                             ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Prepare batch inputs from sequences.

        Args:
            sequences: List of sequences.
            prefill: Whether this is a prefill step.

        Returns:
            Tuple of (input_ids, positions, seq_lengths).
        """
        device = get_current_device()

        batch_input_ids = []
        batch_positions = []
        seq_lengths = []

        # Context related lists
        slot_mapping = []
        context_lens = []
        cum_seqlens_q = [0]
        cum_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq in sequences:
            if len(seq) == 0:
                continue

            seq_len = len(seq)
            seq_lengths.append(seq_len)

            if prefill:
                # For prefill, process all tokens
                batch_input_ids.extend(seq.token_ids)
                batch_positions.extend(range(seq_len))

                # Context logic for prefill
                seqlen_q = seq_len
                seqlen_k = seq_len
                cum_seqlens_q.append(cum_seqlens_q[-1] + seqlen_q)
                cum_seqlens_k.append(cum_seqlens_k[-1] + seqlen_k)
                max_seqlen_q = max(seqlen_q, max_seqlen_q)
                max_seqlen_k = max(seqlen_k, max_seqlen_k)

                if seq.block_table:
                    # Map tokens to slots if block table exists
                    for i in range(seq.num_blocks):
                        start = seq.block_table[i] * self.block_size
                        if i != seq.num_blocks - 1:
                            end = start + self.block_size
                        else:
                            end = start + seq.last_block_num_tokens
                        slot_mapping.extend(list(range(start, end)))
            else:
                # For decode, process only the last token
                batch_input_ids.append(seq.token_ids[-1])
                batch_positions.append(seq_len - 1)

                # Context logic for decode
                context_lens.append(seq_len)
                if seq.block_table:
                    # Map last token to its slot
                    slot_mapping.append(seq.block_table[-1] * self.block_size +
                                        seq.last_block_num_tokens - 1)

        input_ids = torch.tensor(batch_input_ids,
                                 dtype=torch.long,
                                 device=device)
        positions = torch.tensor(batch_positions,
                                 dtype=torch.long,
                                 device=device)

        # Set context
        if prefill:
            cum_seqlens_q_tensor = torch.tensor(cum_seqlens_q,
                                                dtype=torch.int32,
                                                device=device)
            cum_seqlens_k_tensor = torch.tensor(cum_seqlens_k,
                                                dtype=torch.int32,
                                                device=device)
            slot_mapping_tensor = torch.tensor(
                slot_mapping, dtype=torch.int32,
                device=device) if slot_mapping else None

            # Prepare block tables if needed for prefill (e.g. prefix caching)
            block_tables = None
            if any(seq.block_table for seq in sequences):
                block_tables = self._prepare_block_tables(sequences)

            set_context(True, max_seqlen_q, max_seqlen_k, cum_seqlens_q_tensor,
                        cum_seqlens_k_tensor, slot_mapping_tensor, None,
                        block_tables)
        else:
            context_lens_tensor = torch.tensor(context_lens,
                                               dtype=torch.int32,
                                               device=device)
            slot_mapping_tensor = torch.tensor(
                slot_mapping, dtype=torch.int32,
                device=device) if slot_mapping else None
            block_tables = self._prepare_block_tables(sequences)

            set_context(False,
                        slot_mapping=slot_mapping_tensor,
                        context_lens=context_lens_tensor,
                        block_tables=block_tables)

        return input_ids, positions, seq_lengths

    def _execute_prefill(self, input_ids: torch.Tensor,
                         positions: torch.Tensor,
                         seq_lengths: List[int]) -> torch.Tensor:
        """Execute prefill step for the batch.

        Args:
            input_ids: Input token IDs.
            positions: Token positions.
            seq_lengths: Sequence lengths.

        Returns:
            Model logits.
        """
        with torch.no_grad():
            # Use standard execution for prefill
            return self.model(input_ids=input_ids, positions=positions)[0]

    def _execute_decode(self, input_ids: torch.Tensor, positions: torch.Tensor,
                        seq_lengths: List[int]) -> torch.Tensor:
        """Execute decode step for the batch.

        Args:
            input_ids: Input token IDs.
            positions: Token positions.
            seq_lengths: Sequence lengths.

        Returns:
            Model logits.
        """
        batch_size = len(input_ids)

        # Try to use CUDA graph if available
        if (not self.enforce_eager and supports_cuda_graph()
                and batch_size in self.graphs):
            return self._execute_with_graph(input_ids, positions)
        else:
            return self._execute_eager(input_ids, positions)

    def _execute_with_graph(self, input_ids: torch.Tensor,
                            positions: torch.Tensor) -> torch.Tensor:
        """Execute using captured CUDA graph.

        Args:
            input_ids: Input token IDs.
            positions: Token positions.

        Returns:
            Model logits.
        """
        batch_size = len(input_ids)
        graph = self.graphs[batch_size]

        # Update graph variables
        # Note: Must use the specific tensors for this batch size
        self.graph_vars[f'input_ids_{batch_size}'].copy_(input_ids)
        self.graph_vars[f'positions_{batch_size}'].copy_(positions)

        # Replay graph
        graph.replay()

        return self.graph_vars[f'logits_{batch_size}']

    def _execute_eager(self, input_ids: torch.Tensor,
                       positions: torch.Tensor) -> torch.Tensor:
        """Execute eagerly without CUDA graph.

        Args:
            input_ids: Input token IDs.
            positions: Token positions.

        Returns:
            Model logits.
        """
        with torch.no_grad():
            return self.model(input_ids=input_ids, positions=positions)[0]

    def _sample_tokens(self, logits: torch.Tensor,
                       sequences: List[Sequence]) -> List[int]:
        """Sample next tokens from logits.

        Args:
            logits: Model output logits.
            sequences: Input sequences for sampling parameters.

        Returns:
            List of sampled token IDs.
        """
        if not self.sampler:
            return []

        # Extract sampling parameters
        device = logits.device
        temperatures = torch.tensor(
            [seq.sampling_params.temperature for seq in sequences],
            device=device,
            dtype=torch.float)
        top_ps = torch.tensor([seq.sampling_params.top_p for seq in sequences],
                              device=device,
                              dtype=torch.float)
        top_ks = torch.tensor([seq.sampling_params.top_k for seq in sequences],
                              device=device,
                              dtype=torch.long)
        min_ps = torch.tensor([seq.sampling_params.min_p for seq in sequences],
                              device=device,
                              dtype=torch.float)

        # Sample tokens
        next_tokens = self.sampler(logits,
                                   temperatures=temperatures,
                                   top_ps=top_ps,
                                   top_ks=top_ks,
                                   min_ps=min_ps)

        return next_tokens.tolist()

    def capture_cuda_graphs(self, max_batch_size: int) -> None:
        """Capture CUDA graphs for different batch sizes.

        Args:
            max_batch_size: Maximum batch size to capture.
        """
        if self.enforce_eager or not supports_cuda_graph():
            logger.debug('Skipping CUDA graph capture')
            return

        logger.info('Capturing CUDA graphs...')

        try:
            # Create dummy inputs for graph capture
            for batch_size in range(1, max_batch_size + 1):
                self._capture_graph_for_batch_size(batch_size)

            logger.info(
                f'Captured CUDA graphs for batch sizes 1-{max_batch_size}')

        except Exception as e:
            logger.warning(f'Failed to capture CUDA graphs: {e}')
            self.graphs.clear()
            self.graph_vars.clear()

    def _capture_graph_for_batch_size(self, batch_size: int) -> None:
        """Capture CUDA graph for specific batch size.

        Args:
            batch_size: Batch size to capture.
        """
        # Create dummy inputs
        input_ids = torch.zeros(batch_size,
                                dtype=torch.long,
                                device=get_current_device())
        positions = torch.zeros(batch_size,
                                dtype=torch.long,
                                device=get_current_device())

        # Create graph variables
        self.graph_vars[f'input_ids_{batch_size}'] = input_ids
        self.graph_vars[f'positions_{batch_size}'] = positions

        # Capture graph
        static_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(static_graph):
            logits = self.model(input_ids=input_ids, positions=positions)[0]
            self.graph_vars[f'logits_{batch_size}'] = logits

        self.graphs[batch_size] = static_graph

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics.

        Returns:
            Dictionary with performance metrics.
        """
        # Legacy metrics for backward compatibility
        legacy_metrics = {
            'total_tokens_generated':
            self.total_tokens_generated,
            'total_inference_time':
            self.total_inference_time,
            'tokens_per_second': (self.total_tokens_generated /
                                  max(self.total_inference_time, 0.001)
                                  if self.total_inference_time > 0 else 0),
        }

        # Add memory statistics
        device = get_current_device()
        if device.type == 'cuda':
            memory_stats_dict = memory_stats()
            legacy_metrics.update({
                'gpu_memory_allocated_mb':
                memory_stats_dict.get('allocated', 0) / 1024**2,
                'gpu_memory_cached_mb':
                memory_stats_dict.get('cached', 0) / 1024**2,
                'gpu_memory_max_allocated_mb':
                memory_stats_dict.get('max_allocated', 0) / 1024**2,
            })

        return legacy_metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        reset_peak_memory_stats()

    def cleanup(self) -> None:
        """Clean up resources."""
        # Clear CUDA graphs
        self.graphs.clear()
        self.graph_vars.clear()

        # Return KV cache to memory pool
        if self.kv_cache is not None:
            for k_cache, v_cache in self.kv_cache:
                self.memory_pool.deallocate(k_cache)
                self.memory_pool.deallocate(v_cache)
            self.kv_cache = None

        # Clear model reference
        self.model = None
        self.sampler = None

        # Cleanup memory pool
        if hasattr(self, 'memory_pool') and self.memory_pool:
            self.memory_pool.cleanup()

        # Clear cache
        empty_cache()

        logger.debug('Inference executor cleanup completed')
