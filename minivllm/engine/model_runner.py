"""Model Runner module for executing model inference.

This module provides the ModelRunner class which handles model loading,
KV cache management, device graph capture for efficient batching (CUDA Graph
on CUDA devices), and distributed tensor parallelism using PyTorch's distributed
backend. Supports multiple device types including CUDA, NPU, XPU, etc.
"""

import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from minivllm.config import Config
from minivllm.engine.sequence import Sequence
from minivllm.models.qwen3 import Qwen3ForCausalLM
from minivllm.sampling.sampler import Sampler
from minivllm.utils.context import Context, get_context, reset_context, set_context
from minivllm.utils.device import (
    empty_cache,
    get_current_device,
    get_distributed_backend,
    mem_get_info,
    memory_stats,
    move_tensor_to_device,
    reset_peak_memory_stats,
    set_device,
    should_use_pin_memory,
    supports_cuda_graph,
    synchronize,
    validate_device,
)
from minivllm.utils.loader import load_model
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ['ModelRunner']


class ModelRunner:
    """Executes model inference with distributed tensor parallelism
    and device graph optimization (CUDA Graph on CUDA devices).

    The ModelRunner handles:
    - Loading pre-trained language models
    - Allocating and managing KV cache memory
    - Batching sequences efficiently
    - Capturing and replaying device graphs for decode phase (CUDA Graph on CUDA)
    - Distributed inference across multiple devices using tensor parallelism
    - Token sampling based on model logits
    - Multi-device support (CUDA, NPU, XPU, etc.)

    For single-device inference (tensor_parallel_size=1), the ModelRunner runs
    in the main process. For multi-device inference, worker processes are spawned
    that communicate via shared memory.

    Attributes:
        config: Engine configuration.
        rank: Device rank in distributed setup (0 = main process).
        world_size: Total number of devices.
        device: Current device (torch.device).
        device_name: Device name string ('cuda', 'npu', etc.).
        model: The loaded language model.
        tokenizer: Tokenizer for text processing.
        sampler: Token sampler for generation.
        kv_cache: Pre-allocated tensor for KV cache.
        block_size: Number of tokens per cache block.
        enforce_eager: Whether to skip device graph optimization.
        graphs: Dictionary of captured device graphs for different batch sizes.
        graph_vars: Shared tensors for device graphs.
        shm: Shared memory for IPC with worker processes.
    """

    def __init__(self, config: Config, rank: int,
                 event: Union[Event, List[Event]]) -> None:
        """Initialize the model runner.

        This method:
        1. Sets up distributed communication if needed
        2. Loads the model and tokenizer
        3. Allocates KV cache memory
        4. Optionally captures CUDA graphs for efficient inference
        5. Sets up IPC for multi-GPU scenarios

        Args:
            config: Engine configuration with model path and settings.
            rank: Device rank (0 = main device in distributed setup).
            event: Synchronization event(s) for distributed coordination.
                If single Event, used for worker process synchronization.
                If List[Event], multiple events for synchronizing with
                worker processes.

        Raises:
            RuntimeError: If distributed initialization fails or model loading
                fails.
            ValueError: If configuration is invalid.

        Note:
            For multi-device inference, worker processes (rank > 0) will block
            in the __init__ loop waiting for commands from the main process.
            Device graph optimization (CUDA Graph) is only supported on CUDA devices.
        """
        self.config: Config = config
        hf_config: Any = config.hf_config
        # Normalize torch_dtype to a torch.dtype
        try:
            if isinstance(hf_config.torch_dtype, str):
                dtype_opt = getattr(torch, hf_config.torch_dtype, None)
                hf_config.torch_dtype = dtype_opt or torch.bfloat16
        except Exception:
            hf_config.torch_dtype = torch.bfloat16
        self.block_size: int = config.kvcache_block_size
        self.enforce_eager: bool = config.enforce_eager
        self.world_size: int = config.tensor_parallel_size
        self.rank: int = rank
        self.event: Union[Event, List[Event]] = event

        # Optional attributes that may be set later when model/sampler
        # are available. Initialize to None to avoid AttributeError in
        # environments where model loading is skipped (tests, stubs).

        # Load model based on HuggingFace config
        self.model = Qwen3ForCausalLM(hf_config)
        self.sampler = Sampler()
        load_model(self.model, config.model)

        self.kv_cache: Optional[torch.Tensor] = None
        self.share_memory: Optional[SharedMemory] = None
        # Use Any type for graphs to support different device graph types
        self.graphs: Optional[Dict[int, Any]] = None
        self.graph_vars: Optional[Dict[str, torch.Tensor]] = None
        self.graph_bs: Optional[List[int]] = None
        self.graph_pool: Optional[Any] = None

        # Get current device for this rank
        self.device: torch.device = get_current_device()
        self.device_name: str = self.device.type
        self.device_index: int = self.device.index if self.device.index is not None else 0

        # Validate device is available
        validate_device(self.device)
        logger.info(
            f'Rank {self.rank}: Using device {self.device} ({self.device_name})'
        )

        # Initialize distributed communication
        if self.world_size > 1:
            backend: str = get_distributed_backend()
            logger.info(
                f'Rank {self.rank}: Initializing distributed backend {backend}'
            )
            dist.init_process_group(backend=backend,
                                    world_size=self.world_size,
                                    rank=rank)

        # Set device and dtype
        set_device(self.device)
        default_dtype: torch.dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        # torch.set_default_device is available in PyTorch 2.0+
        # It's device-agnostic and works with all device types
        try:
            torch.set_default_device(self.device)
        except AttributeError:
            # Fallback for older PyTorch versions
            logger.debug(
                'torch.set_default_device not available, using device context manager'
            )

        # Move model and sampler to device
        self.model = self.model.to(self.device)
        self.sampler = self.sampler.to(self.device)

        # Allocate KV cache and warmup
        self.warmup_model()
        self.allocate_kv_cache()

        # Capture device graphs for efficient decode (optional)
        # Only CUDA devices support graph optimization currently
        if not self.enforce_eager and supports_cuda_graph():
            self.capture_device_graph()

        # Reset default device and dtype to avoid affecting other code
        try:
            torch.set_default_device('cpu')
        except AttributeError:
            # Fallback for older PyTorch versions
            pass
        torch.set_default_dtype(default_dtype)

        # Setup IPC for distributed inference
        if self.world_size > 1:
            if rank == 0:
                # Main process: create shared memory
                self.share_memory = SharedMemory(name='minivllm',
                                                 create=True,
                                                 size=2**20)
                dist.barrier()
            else:
                # Worker process: wait for main process, then enter
                # command loop
                dist.barrier()
                self.share_memory = SharedMemory(name='minivllm')
                self.loop()

    def exit(self) -> None:
        """Cleanup model runner resources.

        This method:
        1. Closes shared memory (if distributed)
        2. Deallocates CUDA graphs
        3. Synchronizes CUDA operations
        4. Destroys distributed process group

        Called by main process to gracefully shutdown inference.

        Note:
            This method uses ordered cleanup to respect resource dependencies.
            Errors in individual steps are logged but don't prevent subsequent cleanup.
        """
        import warnings

        errors: List[str] = []

        try:
            # Step 1: Close shared memory for distributed inference (highest priority)
            if self.world_size > 1 and hasattr(self, 'share_memory'):
                try:
                    if self.share_memory is not None:
                        logger.debug(
                            f'Rank {self.rank}: Closing shared memory...')
                        self.share_memory.close()

                        # Synchronize all processes before unlinking
                        if dist.is_initialized():
                            dist.barrier()

                        # Only rank 0 unlinks the shared memory
                        if self.rank == 0:
                            logger.debug('Rank 0: Unlinking shared memory')
                            self.share_memory.unlink()
                except Exception as e:
                    errors.append(
                        f'Failed to cleanup shared memory (rank {self.rank}): {e}'
                    )
                    logger.error(f'Shared memory cleanup error: {e}',
                                 exc_info=True)

            # Step 2: Cleanup device graphs (must be done before device sync)
            if not self.enforce_eager and supports_cuda_graph():
                try:
                    if hasattr(self, 'graphs') and self.graphs is not None:
                        logger.debug(
                            f'Rank {self.rank}: Cleaning up {len(self.graphs)} device graphs'
                        )
                        self.graphs.clear()

                    if hasattr(self,
                               'graph_pool') and self.graph_pool is not None:
                        del self.graph_pool
                        self.graph_pool = None

                    if hasattr(self,
                               'graph_vars') and self.graph_vars is not None:
                        self.graph_vars.clear()
                except Exception as e:
                    errors.append(
                        f'Failed to cleanup device graphs (rank {self.rank}): {e}'
                    )
                    logger.error(f'Device graphs cleanup error: {e}',
                                 exc_info=True)

            # Step 3: Synchronize device operations (critical for proper cleanup)
            try:
                logger.debug(f'Rank {self.rank}: Synchronizing device...')
                synchronize(self.device)
            except Exception as e:
                errors.append(
                    f'Failed to synchronize device (rank {self.rank}): {e}')
                logger.error(f'Device synchronization error: {e}',
                             exc_info=True)

            # Step 4: Destroy distributed process group (lowest priority)
            if self.world_size > 1:
                try:
                    if dist.is_initialized():
                        logger.debug(
                            f'Rank {self.rank}: Destroying process group...')
                        dist.destroy_process_group()
                except Exception as e:
                    errors.append(
                        f'Failed to destroy process group (rank {self.rank}): {e}'
                    )
                    logger.error(f'Process group destruction error: {e}',
                                 exc_info=True)

        except Exception as e:
            # Catch-all for unexpected errors
            errors.append(
                f'Unexpected error during model runner cleanup (rank {self.rank}): {e}'
            )
            logger.critical(f'Unexpected error in ModelRunner.exit(): {e}',
                            exc_info=True)

        finally:
            # Log any errors that occurred
            if errors:
                warnings.warn(
                    f'Errors during model runner cleanup: {"; ".join(errors)}',
                    RuntimeWarning)
                logger.warning(
                    f'ModelRunner cleanup completed with errors: {errors}')
            else:
                logger.info(
                    f'Rank {self.rank}: ModelRunner cleanup completed successfully'
                )

    def loop(self) -> None:
        """Main event loop for worker processes.

        Worker processes (rank > 0) wait for commands from the main process
        via shared memory, execute them, and continue until receiving 'exit'.
        The loop reads commands via read_share_memory and dispatches them
        to the appropriate handler method.

        Raises:
            Exception: Any exception raised by the dispatched method will
                propagate up.
        """
        while True:
            method_name: str
            args: Tuple[Any, ...]
            method_name, args = self.read_share_memory()
            self.call(method_name, *args)
            if method_name == 'exit':
                break

    def read_share_memory(self) -> Tuple[str, Tuple[Any, ...]]:
        """Read command from shared memory.

        Waits for signal from main process, then reads pickled command
        data from shared memory. Used only by worker processes.

        Returns:
            A tuple containing:
            - method_name: Name of the method to call
            - args: Arguments for the method call as a tuple

        Raises:
            AssertionError: If called from a non-worker process.
            Exception: If unpickling data fails.

        Note:
            Only valid for worker processes (rank > 0 and world_size > 1).
        """
        assert self.world_size > 1 and self.rank > 0, (
            'read_share_memory can only be called from worker processes')
        self.event.wait()  # type: ignore[union-attr]
        if self.share_memory is None or self.share_memory.buf is None:
            raise RuntimeError('Shared memory not initialized')
        n: int = int.from_bytes(self.share_memory.buf[0:4], 'little')
        method_name, *args = pickle.loads(self.share_memory.buf[4:n + 4])
        self.event.clear()  # type: ignore[union-attr]
        return method_name, tuple(args)

    def write_share_memory(self, method_name: str, *args: Any) -> None:
        """Write command to shared memory for worker processes.

        Serializes the method name and arguments, writes them to shared
        memory, and signals worker processes via events.

        Args:
            method_name: Name of method to call on workers.
            *args: Arguments to pass to the method.

        Raises:
            AssertionError: If called from a non-main process.

        Note:
            Only valid for main process (rank == 0 and world_size > 1).
        """
        assert self.world_size > 1 and self.rank == 0, (
            'write_share_memory can only be called from main process')
        data: bytes = pickle.dumps([method_name, *args])
        n: int = len(data)
        if self.share_memory is None or self.share_memory.buf is None:
            raise RuntimeError('Shared memory not initialized')
        self.share_memory.buf[0:4] = n.to_bytes(4, 'little')
        self.share_memory.buf[4:n + 4] = data
        for event in self.event:  # type: ignore[union-attr]
            event.set()

    def call(self, method_name: str, *args: Any) -> Any:
        """Call a method, handling distributed IPC if needed.

        For main process: sends command to workers if distributed,
        then executes locally. For worker processes: executes command
        received via IPC.

        Args:
            method_name: Name of method to call.
            *args: Arguments for the method.

        Returns:
            Return value of the called method.

        Raises:
            AttributeError: If method_name does not correspond to a valid
                method on the ModelRunner instance.
            Exception: Any exception raised by the called method will be
                re-raised.
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_share_memory(method_name, *args)
        method: Optional[Callable[..., Any]] = getattr(self, method_name, None)
        if method is None:
            raise AttributeError(f"ModelRunner has no method '{method_name}'")
        return method(*args)

    def warmup_model(self) -> None:
        """Warmup model by running inference on dummy data."""
        empty_cache()
        reset_peak_memory_stats(self.device)
        max_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_batched_tokens // max_model_len,
                       self.config.max_num_seqs)
        sequences: List[Sequence] = [
            Sequence([0] * max_model_len) for _ in range(num_seqs)
        ]
        self.run(sequences, True)
        empty_cache()

    def allocate_kv_cache(self) -> None:
        """Allocate KV cache memory based on device memory availability.

        This method calculates the optimal number of KV cache blocks based on:
        - Available device memory
        - Memory utilization configuration
        - Model architecture (number of layers, heads, etc.)

        Raises:
            RuntimeError: If insufficient memory is available for KV cache allocation.
            ValueError: If calculated number of cache blocks is invalid.
        """
        config: Config = self.config
        hf_config: Any = config.hf_config

        # Get device memory information
        free: int
        total: int
        try:
            free, total = mem_get_info(self.device)
        except Exception as e:
            logger.error(
                f'Failed to get memory info for device {self.device}: {e}')
            raise RuntimeError(
                f'Cannot allocate KV cache: failed to query device memory. '
                f'Device: {self.device.type}, Error: {e}') from e

        used: int = total - free
        stats: Dict[str, Any] = memory_stats(self.device)
        peak: int = stats.get('allocated_bytes.all.peak', 0)
        current: int = stats.get('allocated_bytes.all.current', 0)

        # Calculate KV cache requirements
        num_kv_heads: int = (hf_config.num_key_value_heads // self.world_size)
        head_dim: int = getattr(
            hf_config, 'head_dim',
            hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes: int = (2 * hf_config.num_hidden_layers * self.block_size *
                            num_kv_heads * head_dim *
                            hf_config.torch_dtype.itemsize)

        # Calculate number of cache blocks that fit in available memory
        # Use device_memory_utilization (backward compatible with gpu_memory_utilization)
        available_memory: int = int(total * config.device_memory_utilization -
                                    used - peak + current)

        if available_memory <= 0:
            raise RuntimeError(
                f'Insufficient memory for KV cache allocation. '
                f'Device: {self.device.type}, Available: {available_memory} bytes, '
                f'Required per block: {block_bytes} bytes. '
                f'Try reducing device_memory_utilization or max_model_len.')

        config.num_kvcache_blocks = int(available_memory // block_bytes)

        if config.num_kvcache_blocks <= 0:
            raise ValueError(
                f'Calculated KV cache blocks ({config.num_kvcache_blocks}) must be > 0. '
                f'Available memory: {available_memory} bytes, '
                f'Block size: {block_bytes} bytes. '
                f'Device: {self.device.type}')

        logger.info(
            f'Allocating KV cache: {config.num_kvcache_blocks} blocks, '
            f'{config.num_kvcache_blocks * block_bytes / (1024**3):.2f} GB, '
            f'device: {self.device.type}')

        # Allocate KV cache tensor
        try:
            self.kv_cache = torch.empty(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                device=self.device,
                dtype=hf_config.torch_dtype,
            )
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f'OOM during KV cache allocation. '
                f'Try reducing device_memory_utilization (current: {config.device_memory_utilization}) '
                f'or max_num_seqs. Error: {e}') from e

        # Assign cache slices to model layers
        layer_id: int = 0
        if self.kv_cache is not None:
            for module in self.model.modules():
                if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1

        if layer_id == 0:
            logger.warning(
                'No attention layers found with k_cache/v_cache attributes. '
                'KV cache may not be properly initialized.')

    def prepare_block_tables(self, sequences: List[Sequence]) -> torch.Tensor:
        """Prepare block tables for sequences.

        Creates a padded 2D tensor where each row is a sequence's block table,
        padded to the maximum block table length with -1s.

        Args:
            seqs: List of sequences to prepare block tables for.

        Returns:
            A 2D integer tensor on GPU of shape (len(seqs), max_block_len)
            containing block table IDs with -1 padding.
        """
        max_len: int = max(len(seq.block_table) for seq in sequences)
        block_tables: List[List[int]] = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in sequences
        ]
        block_tables_tensor: torch.Tensor = torch.tensor(
            block_tables,
            dtype=torch.int32,
            pin_memory=should_use_pin_memory(self.device))
        return move_tensor_to_device(block_tables_tensor,
                                     self.device,
                                     non_blocking=True)

    def prepare_prefill(
            self,
            sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors for prefill phase.

        Gathers tokens from multiple sequences with cached prefix handling,
        computing positions and slot mappings for KV cache placement.

        Args:
            sequences: List of sequences in prefill phase.

        Returns:
            A tuple containing:
            - input_ids: Tensor of input token IDs on GPU
            - positions: Tensor of token positions on GPU
        """
        input_ids: List[int] = []
        positions: List[int] = []
        cum_seqlens_q: List[int] = [0]
        cum_seqlens_k: List[int] = [0]
        max_seqlen_q: int = 0
        max_seqlen_k: int = 0
        slot_mapping: List[int] = []
        block_tables: Optional[torch.Tensor] = None

        for sequence in sequences:
            seqlen: int = len(sequence)
            # Sequence __getitem__ 只支持 int，不支持切片，这里直接用 token_ids
            input_ids.extend(sequence.token_ids[sequence.num_cached_tokens:])
            positions.extend(list(range(sequence.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - sequence.num_cached_tokens
            seqlen_k = seqlen
            cum_seqlens_q.append(cum_seqlens_q[-1] + seqlen_q)
            cum_seqlens_k.append(cum_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not sequence.block_table:  # warmup
                continue

            # Slot mapping: maps tokens to global KV cache slot indices
            # This allows efficient placement of KV values in the cache
            for i in range(sequence.num_cached_blocks, sequence.num_blocks):
                start = sequence.block_table[i] * self.block_size
                if i != sequence.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + sequence.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        # When transitioning from prefill to decode, we need to start using cached KV
        # Use prepare_block_tables to combine multiple sequence block tables,
        # making block_tables non-empty for prefix cache scenarios
        if cum_seqlens_k[-1] > cum_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(sequences)
        pin_mem = should_use_pin_memory(self.device)
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.int64,
                                 pin_memory=pin_mem)
        input_ids = move_tensor_to_device(input_ids,
                                          self.device,
                                          non_blocking=True)
        positions = torch.tensor(positions,
                                 dtype=torch.int64,
                                 pin_memory=pin_mem)
        positions = move_tensor_to_device(positions,
                                          self.device,
                                          non_blocking=True)
        cum_seqlens_q = torch.tensor(cum_seqlens_q,
                                     dtype=torch.int32,
                                     pin_memory=pin_mem)
        cum_seqlens_q = move_tensor_to_device(cum_seqlens_q,
                                              self.device,
                                              non_blocking=True)
        cum_seqlens_k = torch.tensor(cum_seqlens_k,
                                     dtype=torch.int32,
                                     pin_memory=pin_mem)
        cum_seqlens_k = move_tensor_to_device(cum_seqlens_k,
                                              self.device,
                                              non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.int32,
                                    pin_memory=pin_mem)
        slot_mapping = move_tensor_to_device(slot_mapping,
                                             self.device,
                                             non_blocking=True)
        set_context(
            True,
            max_seqlen_q,
            max_seqlen_k,
            cum_seqlens_q,
            cum_seqlens_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(
            self,
            sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors for decode phase.

        Extracts the last token from each sequence and computes its position
        in the context. This is used during generation when decoding one
        token at a time.

        Args:
            sequences: List of sequences in decode phase.

        Returns:
            A tuple containing:
            - input_ids: Tensor of last token IDs on GPU (shape: [num_seqs])
            - positions: Tensor of token positions on GPU (shape: [num_seqs])
        """
        input_ids = [seq.last_token for seq in sequences]
        positions = [len(seq) - 1 for seq in sequences]
        context_lens = [len(seq) for seq in sequences]
        slot_mapping = [
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens -
            1 for seq in sequences
        ]

        pin_mem = should_use_pin_memory(self.device)
        input_ids_tensor: torch.Tensor = torch.tensor(input_ids,
                                                      dtype=torch.int64,
                                                      pin_memory=pin_mem)
        input_ids_tensor = move_tensor_to_device(input_ids_tensor,
                                                 self.device,
                                                 non_blocking=True)
        positions_tensor: torch.Tensor = torch.tensor(positions,
                                                      dtype=torch.int64,
                                                      pin_memory=pin_mem)
        positions_tensor = move_tensor_to_device(positions_tensor,
                                                 self.device,
                                                 non_blocking=True)
        slot_mapping_tensor: torch.Tensor = torch.tensor(slot_mapping,
                                                         dtype=torch.int32,
                                                         pin_memory=pin_mem)
        slot_mapping_tensor = move_tensor_to_device(slot_mapping_tensor,
                                                    self.device,
                                                    non_blocking=True)
        context_lens_tensor: torch.Tensor = torch.tensor(context_lens,
                                                         dtype=torch.int32,
                                                         pin_memory=pin_mem)
        context_lens_tensor = move_tensor_to_device(context_lens_tensor,
                                                    self.device,
                                                    non_blocking=True)
        block_tables: torch.Tensor = self.prepare_block_tables(sequences)

        set_context(False,
                    slot_mapping=slot_mapping_tensor,
                    context_lens=context_lens_tensor,
                    block_tables=block_tables)

        return input_ids_tensor, positions_tensor

    def prepare_sample(
        self, sequences: List[Sequence]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare sampling parameters for sequences.

        Extracts temperature, top_p, top_k, and min_p values from each sequence
        for use in token sampling.

        Args:
            sequences: List of sequences to sample from.

        Returns:
            A tuple containing tensors on GPU:
            - temperatures (shape: [num_seqs])
            - top_ps (shape: [num_seqs])
            - top_ks (shape: [num_seqs])
            - min_ps (shape: [num_seqs])
        """
        temperatures = [seq.temperature for seq in sequences]
        top_ps = [seq.top_p for seq in sequences]
        top_ks = [seq.top_k for seq in sequences]
        min_ps = [seq.min_p for seq in sequences]

        pin_mem = should_use_pin_memory(self.device)

        temperatures_tensor: torch.Tensor = torch.tensor(temperatures,
                                                         dtype=torch.float32,
                                                         pin_memory=pin_mem)

        top_ps_tensor: torch.Tensor = torch.tensor(top_ps,
                                                   dtype=torch.float32,
                                                   pin_memory=pin_mem)

        top_ks_tensor: torch.Tensor = torch.tensor(top_ks,
                                                   dtype=torch.int32,
                                                   pin_memory=pin_mem)

        min_ps_tensor: torch.Tensor = torch.tensor(min_ps,
                                                   dtype=torch.float32,
                                                   pin_memory=pin_mem)

        return (
            move_tensor_to_device(temperatures_tensor,
                                  self.device,
                                  non_blocking=True),
            move_tensor_to_device(top_ps_tensor,
                                  self.device,
                                  non_blocking=True),
            move_tensor_to_device(top_ks_tensor,
                                  self.device,
                                  non_blocking=True),
            move_tensor_to_device(min_ps_tensor,
                                  self.device,
                                  non_blocking=True),
        )

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor,
                  is_prefill: bool) -> torch.Tensor:
        """Run model inference and compute logits.

        For prefill or small batches, uses eager execution. For decode with
        larger batches on CUDA devices, replays captured CUDA graphs for efficiency.

        Args:
            input_ids: Input token IDs tensor on device.
            positions: Token position tensor on device.
            is_prefill: Whether this is prefill (True) or decode (False) phase.

        Returns:
            Logits tensor of shape (batch_size, vocab_size).

        Raises:
            RuntimeError: If device graph inference fails due to missing graph attributes.
        """
        # Use eager execution for prefill, eager mode, large batches, or non-graph devices
        if (is_prefill or self.enforce_eager or input_ids.size(0) > 512
                or not supports_cuda_graph()):
            # Prefill: Long sequences, better to use eager execution
            # Eager mode: User has requested no graph optimization
            # Large batches: Graphs are cached for specific batch sizes, large batches
            # may not match any cached graph size efficiently
            # Non-graph devices: Graph optimization not supported
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Use device graph replay for decode phase (small batches, supported devices only)
            # Device graphs provide significant speedups for small, repeated workloads
            # like the decode phase where we generate one token at a time
            bs: int = input_ids.size(0)
            context: Context = get_context()

            # Validate device graph attributes are initialized
            if not hasattr(self, 'graphs') or self.graphs is None:
                raise RuntimeError(
                    'Device graphs not initialized. This may happen if enforce_eager=True '
                    'or if graph capture failed during initialization.')

            if not hasattr(self, 'graph_bs') or self.graph_bs is None:
                raise RuntimeError(
                    'Device graph batch sizes not initialized. This may happen if enforce_eager=True '
                    'or if graph capture failed during initialization.')

            if not hasattr(self, 'graph_vars') or self.graph_vars is None:
                raise RuntimeError(
                    'Device graph variables not initialized. This may happen if enforce_eager=True '
                    'or if graph capture failed during initialization.')

            # Find smallest graph batch size >= current batch size
            # This ensures we can reuse the captured device graph for efficient decode
            try:
                graph_bs_idx: int = next(x for x in self.graph_bs if x >= bs)
            except StopIteration:
                raise RuntimeError(
                    f'No cached device graph found for batch size {bs}. '
                    f'Available graph batch sizes: {self.graph_bs}')

            if graph_bs_idx not in self.graphs:
                raise RuntimeError(
                    f'Device graph not found for batch size {graph_bs_idx}. '
                    f'Available graph batch sizes: {list(self.graphs.keys())}')

            graph: Any = self.graphs[graph_bs_idx]
            graph_vars: Dict[str, torch.Tensor] = self.graph_vars

            # Update graph variables with current batch data
            # Reset slot mapping and fill with current batch positions
            graph_vars['slot_mapping'].fill_(-1)
            graph_vars['slot_mapping'][:bs] = context.slot_mapping

            # Update input tensors for this batch
            graph_vars['input_ids'][:bs] = input_ids
            graph_vars['positions'][:bs] = positions

            # Update context information for attention mechanism
            graph_vars['context_lens'].zero_()
            graph_vars['context_lens'][:bs] = context.context_lens

            # Update block table mappings for KV cache
            if context.block_tables is not None:
                max_blocks = context.block_tables.size(1)
                graph_vars[
                    'block_tables'][:bs, :max_blocks] = context.block_tables

            # Replay the captured device graph for efficient inference
            # This avoids the overhead of repeated kernel launches
            graph.replay()
            return self.model.compute_logits(graph_vars['outputs'][:bs])

    def run(self, sequences: List[Sequence],
            is_prefill: bool) -> Optional[List[int]]:
        """Execute inference on a batch of sequences.

        Prepares input tensors, runs the model, samples tokens, and updates
        sequence state. For distributed inference, only rank 0 returns tokens.

        Args:
            sequences: Sequences to process.
            is_prefill: Whether this is prefill (True) or decode (False) phase.

        Returns:
            List of sampled token IDs for each sequence (only on rank 0).
            Returns None for worker processes (rank > 0).
        """
        # Prepare input tensors
        input_ids: torch.Tensor
        positions: torch.Tensor
        input_ids, positions = (self.prepare_prefill(sequences) if is_prefill
                                else self.prepare_decode(sequences))

        # Prepare sampling parameters (rank 0 only)
        sampling_params_tensors: Optional[Tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]] = None
        if self.rank == 0:
            sampling_params_tensors = self.prepare_sample(sequences)

        # Run model inference
        logits: torch.Tensor = self.run_model(input_ids, positions, is_prefill)

        # Sample tokens (rank 0 only)
        token_ids: Optional[List[int]] = None
        if self.rank == 0 and sampling_params_tensors is not None:
            temperatures, top_ps, top_ks, min_ps = sampling_params_tensors
            token_ids = self.sampler(logits, temperatures, top_ps, top_ks,
                                     min_ps).tolist()

        # Clean up context
        reset_context()

        return token_ids

    @torch.inference_mode()
    def capture_device_graph(self) -> None:
        """Capture device graphs for efficient decode phase execution.

        Captures graphs for different batch sizes to accelerate the decode
        phase where token-by-token generation happens. The graphs are captured
        with representative batch sizes: [1, 2, 4, 8, 16, 32, ...].

        Currently, only CUDA devices support graph optimization, but the code
        is structured to support other devices with similar capabilities in the future.

        This method:
        1. Pre-allocates tensors for graph variables on the correct device
        2. Iterates through predefined batch sizes in reverse
        3. Warms up and captures device graphs for each batch size (CUDA devices only)
        4. Stores graphs for later replay during inference

        Side Effects:
            Sets self.graphs, self.graph_bs, self.graph_pool, and
            self.graph_vars which are used during inference in run_model.

        Note:
            This is skipped if enforce_eager=True or if the device does not
            support graph capture.
            On non-CUDA devices, this method will log a warning and return early.
        """
        # Only CUDA devices support graph optimization currently
        if not supports_cuda_graph():
            logger.warning(
                f'CUDA Graph optimization not supported on {self.device.type} device. '
                f'Skipping graph capture.')
            return

        config: Config = self.config
        hf_config: Any = config.hf_config
        max_block_size: int = min(self.config.max_num_seqs, 512)
        max_num_blocks: int = ((config.max_model_len + self.block_size - 1) //
                               self.block_size)

        # Pre-allocate tensors for graphs on the correct device
        input_ids: torch.Tensor = torch.zeros(max_block_size,
                                              dtype=torch.int64,
                                              device=self.device)
        positions: torch.Tensor = torch.zeros(max_block_size,
                                              dtype=torch.int64,
                                              device=self.device)
        slot_mapping: torch.Tensor = torch.zeros(max_block_size,
                                                 dtype=torch.int32,
                                                 device=self.device)
        context_lens: torch.Tensor = torch.zeros(max_block_size,
                                                 dtype=torch.int32,
                                                 device=self.device)
        block_tables: torch.Tensor = torch.zeros(max_block_size,
                                                 max_num_blocks,
                                                 dtype=torch.int32,
                                                 device=self.device)
        outputs: torch.Tensor = torch.zeros(max_block_size,
                                            hf_config.hidden_size,
                                            device=self.device)

        # Batch sizes to capture graphs for
        self.graph_bs: List[int] = [1, 2, 4, 8] + list(
            range(16, max_block_size + 1, 16))
        self.graphs: Dict[int, Any] = {}
        self.graph_pool: Optional[Any] = None

        # Capture graphs in reverse order for efficiency
        for bs in reversed(self.graph_bs):
            # Create device graph for this batch size
            if self.device.type == 'cuda':
                graph: Any = torch.cuda.CUDAGraph()
                set_context(False,
                            slot_mapping=slot_mapping[:bs],
                            context_lens=context_lens[:bs],
                            block_tables=block_tables[:bs])

                # Warmup: run model once to establish memory patterns
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

                # Capture: record the computation graph for replay
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

                # Create graph pool on first iteration for memory efficiency
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()

                self.graphs[bs] = graph
                synchronize(self.device)
                reset_context()
            else:
                # Other devices don't support graph capture yet
                break

        # Store graph variables for use during inference
        self.graph_vars: Dict[str, torch.Tensor] = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
