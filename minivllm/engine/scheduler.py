"""Scheduler module for managing sequence scheduling and KV cache allocation.

This module provides the Scheduler class which handles scheduling sequences
for prefill and decode phases, managing the KV cache blocks through the
BlockManager, and handling sequence preemption when needed.
"""

from collections import deque
from typing import List, Tuple

from minivllm.config import Config
from minivllm.engine.block_manager import BlockManager
from minivllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """Schedules sequences for execution and manages KV cache allocation.

    The scheduler implements a two-phase scheduling policy:
    1. Prefill phase: Process new sequences, computing initial KV cache
    2. Decode phase: Generate one token per sequence, filling KV cache

    This policy balances compute and memory efficiency by:
    - Maximizing compute utilization during prefill
    - Maintaining decode throughput with memory constraints
    - Preempting sequences when cache memory is exhausted
    - Using fair scheduling when multiple sequences are ready

    Attributes:
        max_num_seqs: Maximum sequences in a single batch.
        max_num_batched_tokens: Maximum total tokens per batch.
        eos: End-of-sequence token ID for generation termination.
        block_manager: Manages KV cache block allocation.
        waiting: Queue of sequences awaiting initialization.
        running: Queue of sequences currently being decoded.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the scheduler.

        Args:
            config: Engine configuration containing scheduling parameters.
        """
        self.max_num_seqs: int = config.max_num_seqs
        self.max_num_batched_tokens: int = config.max_num_batched_tokens
        self.eos: int = config.eos
        self.block_manager: BlockManager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque = deque()
        self.running: deque = deque()

    def is_finished(self) -> bool:
        """Check if all sequences have finished generation.

        Returns:
            True if no sequences are waiting or running.
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        """Add a new sequence to the waiting queue.

        Args:
            seq: Sequence to add for processing.
        """
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], bool]:
        """Schedule sequences for the current inference step.

        This method implements the two-phase scheduling policy:

        Phase 1 - Prefill: Process waiting sequences with their full prompt,
        computing initial KV cache. Continues until:
        - No more waiting sequences
        - Batch size limit reached
        - Token limit reached
        - Cache memory exhausted

        Phase 2 - Decode: Generate one token for running sequences. Handles
        cache memory constraints by preempting sequences if needed.

        Returns:
            A tuple containing:
            - List of sequences to process this step
            - Boolean: True for prefill phase, False for decode

        Raises:
            AssertionError: If scheduling invariants are violated.
        """
        scheduled_seqs: List[Sequence] = []
        num_seqs: int = 0
        num_batched_tokens: int = 0

        # Phase 1: Prefill phase for waiting sequences
        while self.waiting and num_seqs < self.max_num_seqs:
            seq: Sequence = self.waiting[0]

            # Check if sequence fits in current batch
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens
                    or not self.block_manager.can_allocate(seq)):
                break

            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        # Return if prefill sequences were scheduled
        if scheduled_seqs:
            return scheduled_seqs, True

        # Phase 2: Decode phase for running sequences
        while self.running and num_seqs < self.max_num_seqs:
            seq: Sequence = self.running.popleft()

            # Ensure space for appending next token
            while not self.block_manager.can_append(seq):
                if self.running:
                    # Preempt another sequence to make space
                    self.preempt(self.running.pop())
                else:
                    # Preempt current sequence and try next iteration
                    self.preempt(seq)
                    break
            else:
                # Space available: prepare sequence for decode
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs, 'No sequences scheduled (should not happen)'

        # Restore running sequences order (put back those not scheduled)
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    def preempt(self, seq: Sequence) -> None:
        """Preempt a sequence due to cache memory constraints.

        Preempted sequences are returned to the waiting queue and will
        be re-executed from the cached point when cache space is available.

        Args:
            seq: Sequence to preempt.
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: List[Sequence], token_ids: List[int]) -> None:
        """Update sequences with newly generated tokens and handle completion.

        This method:
        1. Appends new tokens to each sequence
        2. Checks for sequence completion (EOS or max tokens reached)
        3. Deallocates cache blocks for finished sequences
        4. Removes finished sequences from running queue

        Args:
            seqs: Sequences that were just processed.
            token_ids: Newly generated token IDs for each sequence.
        """
        for seq, token_id in zip(seqs, token_ids):
            # Append new token to sequence
            seq.append_token(token_id)

            # Check if sequence is finished
            is_eos: bool = (token_id == self.eos) and not seq.ignore_eos
            is_max_len: bool = seq.num_completion_tokens >= seq.max_tokens

            if is_eos or is_max_len:
                # Mark as finished and free resources
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
