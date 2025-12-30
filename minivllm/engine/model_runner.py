"""Model Runner module for executing model inference.

This module provides the ModelRunner class which handles model loading,
KV cache management, CUDA graph capture for efficient batching, and
distributed tensor parallelism using PyTorch's distributed backend.
"""

import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from minivllm.config import Config
from minivllm.engine.sequence import Sequence
from minivllm.models.layers import Sampler
from minivllm.models.qwen3 import Qwen3ForCausalLM
from minivllm.utils.context import get_context, reset_context, set_context


class ModelRunner:
    """Executes model inference with distributed tensor parallelism
    and CUDA graph optimization.

    The ModelRunner handles:
    - Loading pre-trained language models
    - Allocating and managing KV cache memory
    - Batching sequences efficiently
    - Capturing and replaying CUDA graphs for decode phase
    - Distributed inference across multiple GPUs using tensor parallelism
    - Token sampling based on model logits

    For single-GPU inference (tensor_parallel_size=1), the ModelRunner runs
    in the main process. For multi-GPU inference, worker processes are spawned
    that communicate via shared memory.

    Attributes:
        config: Engine configuration.
        rank: GPU rank in distributed setup (0 = main process).
        world_size: Total number of GPUs.
        model: The loaded language model.
        tokenizer: Tokenizer for text processing.
        sampler: Token sampler for generation.
        kv_cache: Pre-allocated tensor for KV cache.
        block_size: Number of tokens per cache block.
        enforce_eager: Whether to skip CUDA graph optimization.
        graphs: Dictionary of captured CUDA graphs for different batch sizes.
        graph_vars: Shared tensors for CUDA graphs.
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
            rank: GPU rank (0 = main GPU in distributed setup).
            event: Synchronization event(s) for distributed coordination.
                If single Event, used for worker process synchronization.
                If List[Event], multiple events for synchronizing with
                worker processes.

        Raises:
            RuntimeError: If distributed initialization fails or model loading
                fails.
            ValueError: If configuration is invalid.

        Note:
            For multi-GPU inference, worker processes (rank > 0) will block
            in the __init__ loop waiting for commands from the main process.
        """
        self.config: Config = config
        hf_config: Any = config.hf_config
        self.block_size: int = config.kvcache_block_size
        self.enforce_eager: bool = config.enforce_eager
        self.world_size: int = config.tensor_parallel_size
        self.rank: int = rank
        self.event: Union[Event, List[Event]] = event

        # Optional attributes that may be set later when model/sampler
        # are available. Initialize to None to avoid AttributeError in
        # environments where model loading is skipped (tests, stubs).
        self.model: Optional[Any] = None
        self.sampler: Optional[Any] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.share_memory: Optional[SharedMemory] = None
        self.graphs: Optional[Dict[int, torch.cuda.CUDAGraph]] = None
        self.graph_vars: Optional[Dict[str, torch.Tensor]] = None

        # Initialize distributed communication
        if self.world_size > 1:
            dist.init_process_group('nccl',
                                    'tcp://localhost:2333',
                                    world_size=self.world_size,
                                    rank=rank)

        # Set CUDA device and dtype
        torch.cuda.set_device(rank)
        default_dtype: torch.dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device('cuda')

        # Load model (NOTE: These functions need to be implemented)
        self.model = Qwen3ForCausalLM(hf_config)
        self.sampler = Sampler()

        # Allocate KV cache and warmup
        self.warmup_model()
        self.allocate_kv_cache()

        # Capture CUDA graphs for efficient decode (optional)
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device('cpu')
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
        """
        try:
            if self.world_size > 1:
                self.share_memory.close()
                dist.barrier()
                if self.rank == 0:
                    self.share_memory.unlink()

            # Cleanup CUDA graphs
            if not self.enforce_eager and hasattr(self, 'graphs'):
                del self.graphs
                if hasattr(self, 'graph_pool'):
                    del self.graph_pool

            torch.cuda.synchronize()

            if self.world_size > 1:
                dist.destroy_process_group()
        except Exception as e:
            import warnings
            warnings.warn(f'Error during model runner cleanup: {e}')

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
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_batched_tokens // max_model_len,
                       self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self) -> None:
        """Allocate KV cache memory based on GPU memory availability."""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current = (torch.cuda.memory_stats()['allocated_bytes.all.current'])
        num_kv_heads = (hf_config.num_key_value_heads // self.world_size)
        head_dim = getattr(
            hf_config, 'head_dim',
            hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
                       num_kv_heads * head_dim *
                       hf_config.torch_dtype.itemsize)
        config.num_kvcache_blocks = int(
            (total * config.gpu_memory_utilization - used - peak + current) //
            block_bytes)
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers,
                                    config.num_kvcache_blocks, self.block_size,
                                    num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: List[Sequence]) -> torch.Tensor:
        """Prepare block tables for sequences.

        Creates a padded 2D tensor where each row is a sequence's block table,
        padded to the maximum block table length with -1s.

        Args:
            seqs: List of sequences to prepare block tables for.

        Returns:
            A 2D integer tensor on GPU of shape (len(seqs), max_block_len)
            containing block table IDs with -1 padding.
        """
        max_len: int = max(len(seq.block_table) for seq in seqs)
        block_tables: List[List[int]] = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables_tensor: torch.Tensor = torch.tensor(
            block_tables, dtype=torch.int32,
            pin_memory=True).cuda(non_blocking=True)
        return block_tables_tensor

    def prepare_prefill(
            self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors for prefill phase.

        Gathers tokens from multiple sequences with cached prefix handling,
        computing positions and slot mappings for KV cache placement.

        Args:
            seqs: List of sequences in prefill phase.

        Returns:
            A tuple containing:
            - input_ids: Tensor of input token IDs on GPU
            - positions: Tensor of token positions on GPU
        """
        input_ids: List[int] = []
        positions: List[int] = []
        cu_seqlens_q: List[int] = [0]
        cu_seqlens_k: List[int] = [0]
        max_seqlen_q: int = 0
        max_seqlen_k: int = 0
        slot_mapping: List[int] = []
        block_tables: Optional[torch.Tensor] = None

        for seq in seqs:
            seqlen: int = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64,
                                 pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64,
                                 pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q,
                                    dtype=torch.int32,
                                    pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k,
                                    dtype=torch.int32,
                                    pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.int32,
                                    pin_memory=True).cuda(non_blocking=True)
        set_context(  # noqa: F821
            True,  # noqa: F821
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables)
        return input_ids, positions

    def prepare_decode(
            self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors for decode phase.

        Extracts the last token from each sequence and computes its position
        in the context. This is used during generation when decoding one
        token at a time.

        Args:
            seqs: List of sequences in decode phase.

        Returns:
            A tuple containing:
            - input_ids: Tensor of last token IDs on GPU (shape: [num_seqs])
            - positions: Tensor of token positions on GPU (shape: [num_seqs])
        """
        input_ids: List[int] = []
        positions: List[int] = []
        slot_mapping: List[int] = []
        context_lens: List[int] = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size +
                                seq.last_block_num_tokens - 1)

        input_ids_tensor: torch.Tensor = torch.tensor(
            input_ids, dtype=torch.int64,
            pin_memory=True).cuda(non_blocking=True)
        positions_tensor: torch.Tensor = torch.tensor(
            positions, dtype=torch.int64,
            pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor: torch.Tensor = torch.tensor(
            slot_mapping, dtype=torch.int32,
            pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor: torch.Tensor = torch.tensor(
            context_lens, dtype=torch.int32,
            pin_memory=True).cuda(non_blocking=True)
        block_tables: torch.Tensor = self.prepare_block_tables(seqs)

        set_context(False,
                    slot_mapping=slot_mapping_tensor,
                    context_lens=context_lens_tensor,
                    block_tables=block_tables)

        return input_ids_tensor, positions_tensor

    def prepare_sample(self, seqs: List[Sequence]) -> torch.Tensor:
        """Prepare sampling parameters for sequences.

        Extracts temperature values from each sequence for use in
        token sampling.

        Args:
            seqs: List of sequences to sample from.

        Returns:
            A 1D tensor of temperature values on GPU (shape: [num_seqs]).
        """
        temperatures: List[float] = []
        for seq in seqs:
            temperatures.append(seq.temperature)

        temperatures_tensor: torch.Tensor = torch.tensor(
            temperatures, dtype=torch.float32,
            pin_memory=True).cuda(non_blocking=True)
        return temperatures_tensor

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor,
                  is_prefill: bool) -> torch.Tensor:
        """Run model inference and compute logits.

        For prefill or small batches, uses eager execution. For decode with
        larger batches, replays captured CUDA graphs for efficiency.

        Args:
            input_ids: Input token IDs tensor on GPU.
            positions: Token position tensor on GPU.
            is_prefill: Whether this is prefill (True) or decode (False) phase.

        Returns:
            Logits tensor of shape (batch_size, vocab_size).
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Use eager execution for prefill, eager mode, or large batches
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Use CUDA graph replay for decode phase
            bs: int = input_ids.size(0)
            context: Any = get_context()
            # Find smallest graph batch size >= current batch size
            graph_bs_idx: int = next(x for x in self.graph_bs
                                     if x >= bs)  # type: ignore[attr-defined]
            graph: torch.cuda.CUDAGraph = self.graphs[
                graph_bs_idx]  # type: ignore[attr-defined]
            graph_vars: Dict[
                str,
                torch.Tensor] = self.graph_vars  # type: ignore[attr-defined]

            # Update graph variables with current batch data
            graph_vars['input_ids'][:bs] = input_ids
            graph_vars['positions'][:bs] = positions
            graph_vars['slot_mapping'].fill_(-1)
            graph_vars['slot_mapping'][:bs] = context.slot_mapping
            graph_vars['context_lens'].zero_()
            graph_vars['context_lens'][:bs] = context.context_lens
            graph_vars['block_tables'][:bs, :context.block_tables.size(1)] = (
                context.block_tables)

            # Replay the captured graph
            graph.replay()
            return self.model.compute_logits(graph_vars['outputs'][:bs])

    def run(self, seqs: List[Sequence],
            is_prefill: bool) -> Optional[List[int]]:
        """Execute inference on a batch of sequences.

        Prepares input tensors, runs the model, samples tokens, and updates
        sequence state. For distributed inference, only rank 0 returns tokens.

        Args:
            seqs: Sequences to process.
            is_prefill: Whether this is prefill (True) or decode (False) phase.

        Returns:
            List of sampled token IDs for each sequence (only on rank 0).
            Returns None for worker processes (rank > 0).
        """
        # Prepare input tensors
        input_ids: torch.Tensor
        positions: torch.Tensor
        input_ids, positions = (self.prepare_prefill(seqs)
                                if is_prefill else self.prepare_decode(seqs))

        # Prepare sampling parameters (rank 0 only)
        temperatures: Optional[torch.Tensor] = (self.prepare_sample(seqs)
                                                if self.rank == 0 else None)

        # Run model inference
        logits: torch.Tensor = self.run_model(input_ids, positions, is_prefill)

        # Sample tokens (rank 0 only)
        token_ids: Optional[List[int]] = (
            self.sampler(logits,
                         temperatures).tolist()  # type: ignore[attr-defined]
            if self.rank == 0 else None)

        # Clean up context
        reset_context()

        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        """Capture CUDA graphs for efficient decode phase execution.

        Captures graphs for different batch sizes to accelerate the decode
        phase where token-by-token generation happens. The graphs are captured
        with representative batch sizes: [1, 2, 4, 8, 16, 32, ...].

        This method:
        1. Pre-allocates tensors for graph variables
        2. Iterates through predefined batch sizes in reverse
        3. Warms up and captures CUDA graphs for each batch size
        4. Stores graphs for later replay during inference

        Side Effects:
            Sets self.graphs, self.graph_bs, self.graph_pool, and
            self.graph_vars which are used during inference in run_model.

        Note:
            This is skipped if enforce_eager=True or if the model does not
            support graph capture.
        """
        config: Config = self.config
        hf_config: Any = config.hf_config
        max_bs: int = min(self.config.max_num_seqs, 512)
        max_num_blocks: int = ((config.max_model_len + self.block_size - 1) //
                               self.block_size)

        # Pre-allocate tensors for graphs
        input_ids: torch.Tensor = torch.zeros(max_bs, dtype=torch.int64)
        positions: torch.Tensor = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping: torch.Tensor = torch.zeros(max_bs, dtype=torch.int32)
        context_lens: torch.Tensor = torch.zeros(max_bs, dtype=torch.int32)
        block_tables: torch.Tensor = torch.zeros(max_bs,
                                                 max_num_blocks,
                                                 dtype=torch.int32)
        outputs: torch.Tensor = torch.zeros(max_bs, hf_config.hidden_size)

        # Batch sizes to capture graphs for
        self.graph_bs: List[int] = [1, 2, 4, 8] + list(
            range(16, max_bs + 1, 16))
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_pool: Optional[torch.cuda.graph_pool_handle] = None

        # Capture graphs in reverse order for efficiency
        for bs in reversed(self.graph_bs):
            graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
            set_context(False,
                        slot_mapping=slot_mapping[:bs],
                        context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs])

            # Warmup
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # Capture
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # Create graph pool on first iteration
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # Store graph variables for use during inference
        self.graph_vars: Dict[str, torch.Tensor] = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
