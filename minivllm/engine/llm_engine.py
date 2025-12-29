"""LLM Engine module for managing inference and generation.

This module provides the LLMEngine class which orchestrates the inference
pipeline including model loading, scheduling, and token generation.
"""

import atexit
from dataclasses import fields
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from minivllm.config import Config
from minivllm.engine.model_runner import ModelRunner
from minivllm.engine.scheduler import Scheduler
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams


class LLMEngine:
    """Main inference engine for LLM text generation.

    This class orchestrates the entire inference pipeline, managing:
    - Model loading and distributed tensor parallelism
    - Token scheduling and KV cache management
    - Batch processing for efficient inference
    - Token generation with configurable sampling parameters

    The engine supports both single-GPU and multi-GPU execution through
    distributed tensor parallelism, with automatic sequence scheduling
    to maximize throughput.

    Attributes:
        model_runner: ModelRunner instance handling actual model execution
        tokenizer: HuggingFace tokenizer for encoding/decoding text
        scheduler: Scheduler managing sequence queues and KV cache blocks
        ps: List of worker processes for distributed execution
        events: Synchronization events for distributed communication
    """

    def __init__(self, model: str, **kwargs) -> None:
        """Initialize the LLM engine.

        This method:
        1. Creates engine configuration from model path and kwargs
        2. Initializes worker processes for tensor parallelism
        3. Loads the model and tokenizer
        4. Sets up the scheduler for batch management
        5. Registers cleanup handler for graceful shutdown

        Args:
            model: Path to the model directory (HuggingFace format).
            **kwargs: Additional configuration parameters (see Config class).
                Common parameters include:
                - max_num_seqs: Maximum sequences in a batch
                - max_num_batched_tokens: Maximum tokens per batch
                - gpu_memory_utilization: GPU memory fraction to use
                - tensor_parallel_size: Number of GPUs for parallelism

        Raises:
            ValueError: If model path is invalid or configuration is invalid.
            RuntimeError: If distributed initialization fails.
        """
        # Filter kwargs to only include valid Config parameters
        config_fields: set = {field.name for field in fields(Config)}
        config_kwargs: Dict = {
            k: v
            for k, v in kwargs.items() if k in config_fields
        }
        config: Config = Config(model, **config_kwargs)

        # Initialize distributed processes for tensor parallelism
        self.ps: List[mp.Process] = []
        self.events: List[mp.Event] = []

        if config.tensor_parallel_size > 1:
            ctx: mp.context.SpawnContext = mp.get_context('spawn')
            for i in range(1, config.tensor_parallel_size):
                event: mp.Event = ctx.Event()
                process: mp.Process = ctx.Process(target=ModelRunner,
                                                  args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)

        # Initialize main model runner on rank 0
        self.model_runner: ModelRunner = ModelRunner(config, 0, self.events)

        # Load tokenizer and set EOS token
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # Initialize scheduler for sequence management
        self.scheduler: Scheduler = Scheduler(config)

        # Register cleanup handler
        atexit.register(self.exit)

    def exit(self) -> None:
        """Cleanup engine resources.

        This method:
        1. Signals the model runner to exit
        2. Waits for all worker processes to terminate
        3. Cleans up distributed communication resources

        Called automatically at program exit via atexit handler.
        """
        try:
            self.model_runner.call('exit')
            del self.model_runner
            for p in self.ps:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        except Exception as e:
            import warnings
            warnings.warn(f'Error during engine cleanup: {e}')

    def add_request(self,
                    prompt: Union[str, List[int]],
                    sampling_params: Optional[SamplingParams] = None) -> None:
        """Add a new generation request to the engine.

        Args:
            prompt: Either a text string or list of token IDs.
                If string, it will be tokenized using the engine's tokenizer.
            sampling_params: SamplingParams controlling generation behavior.
                Defaults to SamplingParams() if not provided.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Convert string prompts to token IDs
        if isinstance(prompt, str):
            prompt_tokens: List[int] = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = prompt

        # Create sequence and add to scheduler
        seq: Sequence = Sequence(prompt_tokens, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> Tuple[List[Tuple[int, List[int]]], int]:
        """Execute one inference step.

        This method:
        1. Schedules sequences for prefill or decode
        2. Runs the model on scheduled sequences
        3. Samples tokens from logits
        4. Updates sequence state and collects finished outputs

        Returns:
            A tuple containing:
            - List of (seq_id, completion_token_ids) for finished
              sequences
            - Number of tokens processed (positive for prefill,
              negative for decode)

        The token count is used for throughput measurement and distinguishes
        between prefill (compute-bound) and decode (memory-bound) phases.
        """
        # Get sequences to process this step
        sequences: List[Sequence]
        is_prefill: bool
        sequences, is_prefill = self.scheduler.schedule()

        # Run model inference
        token_ids: List[int] = self.model_runner.call('run', sequences,
                                                      is_prefill)

        # Update sequences with new tokens
        self.scheduler.postprocess(sequences, token_ids)

        # Collect finished sequences
        outputs: List[Tuple[int, List[int]]] = [
            (seq.seq_id, seq.completion_token_ids) for seq in sequences
            if seq.is_finished
        ]

        # Compute token count for throughput tracking
        num_tokens: int = sum(
            len(seq) for seq in sequences) if is_prefill else -len(sequences)
        return outputs, num_tokens

    def is_finished(self) -> bool:
        """Check if all sequences have finished generation.

        Returns:
            True if no sequences are waiting or running.
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: Union[List[str], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams],
                               None] = None,
        use_tqdm: bool = True,
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """Generate text completions for multiple prompts.

        This is the main API for text generation. It handles:
        1. Adding all prompts as requests
        2. Running inference steps until all sequences finish
        3. Decoding token IDs back to text
        4. Optionally showing progress with tqdm

        Args:
            prompts: List of text strings or token ID lists to
                generate from.
            sampling_params: Sampling parameters for generation.
                If a single SamplingParams is provided, it applies to
                all prompts.
                If a list, it should match the length of prompts.
                Defaults to SamplingParams() for all prompts if not provided.
            use_tqdm: Whether to show a progress bar. Default: True.

        Returns:
            List of dictionaries, one per prompt, containing:
            - 'text': The decoded text completion
            - 'token_ids': The list of generated token IDs

        Example:
            >>> engine = LLMEngine("meta-llama/Llama-2-7b")
            >>> results = engine.generate(
            ...     ["Once upon a time", "Hello world"],
            ...     sampling_params=SamplingParams(max_tokens=50)
            ... )
            >>> print(results[0]['text'])
        """
        # Setup progress bar if requested
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=len(prompts),
                        desc='Generating',
                        dynamic_ncols=True)

        # Normalize sampling parameters to list
        if sampling_params is None:
            sampling_params_list: List[SamplingParams] = [
                SamplingParams() for _ in prompts
            ]
        elif not isinstance(sampling_params, list):
            sampling_params_list = [sampling_params] * len(prompts)
        else:
            sampling_params_list = sampling_params

        # Add all requests to the engine
        for prompt, sp in zip(prompts, sampling_params_list):
            self.add_request(prompt, sp)

        # Run inference loop
        outputs: Dict[int, List[int]] = {}
        prefill_throughput: float = 0.0
        decode_throughput: float = 0.0

        while not self.is_finished():
            t: float = perf_counter()
            output, num_tokens = self.step()

            # Update throughput metrics
            if use_tqdm and pbar is not None:
                elapsed: float = perf_counter() - t
                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                else:
                    decode_throughput = -num_tokens / elapsed
                pbar.set_postfix({
                    'Prefill': f'{int(prefill_throughput)}token/s',
                    'Decode': f'{int(decode_throughput)}token/s',
                })

            # Collect outputs and update progress
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm and pbar is not None:
                    pbar.update(1)

        if use_tqdm and pbar is not None:
            pbar.close()

        # Sort outputs by sequence ID to match input order
        sorted_outputs: List[List[int]] = [
            outputs[seq_id] for seq_id in sorted(outputs.keys())
        ]

        # Decode token IDs to text
        result_dicts: List[Dict[str, Union[str, List[int]]]] = [{
            'text':
            self.tokenizer.decode(token_ids),
            'token_ids':
            token_ids
        } for token_ids in sorted_outputs]

        return result_dicts
