"""Scheduler module for managing sequence scheduling and KV cache allocation.

This module provides the Scheduler class which handles scheduling sequences
for prefill and decode phases, managing the KV cache blocks through the
BlockManager, and handling sequence preemption when needed.

Scheduling Strategy:
    The scheduler implements a two-phase policy:

    1. Prefill Phase:
       - Processes new sequences with their full prompts
       - Computes initial KV cache for each sequence
       - Schedules as many sequences as batch/cache constraints allow
       - Priority: FIFO (first-in-first-out)

    2. Decode Phase:
       - Generates one token per running sequence
       - Uses cached KV values from previous steps
       - Handles cache pressure via sequence preemption
       - Priority: Fair scheduling among running sequences

Preemption Policy:
    When KV cache is exhausted:
    - Preempt the most recently scheduled sequence
    - Deallocate its cache blocks
    - Move back to waiting queue
    - Will be rescheduled when cache space is available

    This ensures:
    - Forward progress for older sequences
    - Fair treatment of all sequences
    - Efficient cache utilization

Performance Considerations:
    - Prefill is compute-bound (large batches preferred)
    - Decode is memory-bound (cache size limits batch)
    - Balancing prefill/decode affects throughput
    - Cache pressure may cause sequence starvation
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

from minivllm.config import Config
from minivllm.engine.block_manager import BlockManager
from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ['Scheduler']


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

        Raises:
            ValueError: If configuration is invalid.
        """
        self.max_num_seqs: int = config.max_num_seqs
        self.max_num_batched_tokens: int = config.max_num_batched_tokens
        self.eos: int = config.eos
        self.block_manager: BlockManager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: Deque[Sequence] = deque()
        self.running: Deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """Check if all sequences have finished generation.

        Returns:
            True if no sequences are waiting or running.
        """
        return not self.waiting and not self.running

    def add(self, sequence: Sequence) -> None:
        """Add a new sequence to the waiting queue.

        Args:
            sequence: Sequence to add for processing.
        """
        self.waiting.append(sequence)

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
        # Phase 1: Prefill phase for waiting sequences
        scheduled_sequences, is_prefill = self._schedule_prefill()
        if scheduled_sequences:
            return scheduled_sequences, True

        # Phase 2: Decode phase for running sequences
        scheduled_sequences, is_decode = self._schedule_decode()
        if scheduled_sequences:
            return scheduled_sequences, False

        # Safety check: ensure we scheduled something
        if not self.is_finished():
            # This should never happen if is_finished() is checked properly
            # Provide detailed debugging information
            error_msg = (
                'No sequences scheduled, but scheduler is not finished. '
                f'State: waiting={len(self.waiting)}, running={len(self.running)}, '
                f'max_num_seqs={self.max_num_seqs}, max_num_batched_tokens={self.max_num_batched_tokens}'
            )
            if self.waiting:
                first_waiting = self.waiting[0]
                error_msg += f', first_waiting_seq_len={len(first_waiting)}'
            if self.running:
                error_msg += (
                    f', running_seq_lens={[len(s) for s in list(self.running)[:5]]}'
                )
            raise RuntimeError(error_msg)

        return [], False

    def _schedule_prefill(self) -> Tuple[List[Sequence], bool]:
        """Schedule waiting sequences for prefill."""
        # Phase 1: Prefill phase for waiting sequences
        # Process new sequences with their full prompt to compute initial KV cache

        scheduled_sequences: List[Sequence] = []
        num_seqs: int = 0
        num_batched_tokens: int = 0

        while self.waiting and num_seqs < self.max_num_seqs:
            sequence: Sequence = self.waiting[0]

            # Check if sequence fits in current batch constraints
            # - Total token limit (including cached tokens)
            # - Available KV cache blocks
            if num_batched_tokens + len(
                    sequence
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(
                    sequence):
                logger.debug(
                    f'Cannot schedule sequence {sequence.seq_id} (prefill): '
                    f'len={len(sequence)}, '
                    f'tokens={num_batched_tokens + len(sequence)}/{self.max_num_batched_tokens}, '
                    f'free_blocks={self.block_manager.get_num_free_blocks()}')
                break

            num_seqs += 1
            # Allocate KV cache blocks for this sequence
            self.block_manager.allocate(sequence)
            # Only count uncached tokens for batch size limit
            num_batched_tokens += len(sequence) - sequence.num_cached_tokens
            sequence.status = SequenceStatus.RUNNING

            # Move from waiting to running queue
            self.waiting.popleft()
            self.running.append(sequence)
            scheduled_sequences.append(sequence)

        return scheduled_sequences, bool(scheduled_sequences)

    def _schedule_decode(self) -> Tuple[List[Sequence], bool]:
        """Schedule running sequences for decode."""
        # Phase 2: Decode phase for running sequences
        # Generate one token per sequence while managing cache constraints
        scheduled_sequences: List[Sequence] = []
        num_seqs: int = 0

        while self.running and num_seqs < self.max_num_seqs:
            sequence: Sequence = self.running.popleft()

            # Ensure space for appending next token during decode
            # Handle cache pressure by preempting sequences when necessary
            while not self.block_manager.can_append(sequence):
                # If we cannot allocate KV cache for this sequence (insufficient resources),
                # try to preempt another sequence to free up space
                if self.running:
                    # Preempt another sequence to make space for current one
                    # This maintains fairness by preempting the most recently scheduled
                    # sequence (from the back of the queue), releasing its KV blocks
                    preempted_seq = self.running.pop()
                    self.preempt(preempted_seq)
                else:
                    # No other sequences available: preempt current sequence
                    # It will be rescheduled when cache space becomes available
                    # If there are no other sequences to preempt, we must preempt
                    # the current sequence and move it back to the waiting queue
                    self.preempt(sequence)
                    break
            else:
                # Sufficient cache space: prepare sequence for decode
                num_seqs += 1
                # May allocate new block if we've crossed a block boundary
                self.block_manager.may_append(sequence)
                scheduled_sequences.append(sequence)

        # Restore running sequences order (put back those not scheduled)
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, bool(scheduled_sequences)

    def preempt(self, sequence: Sequence) -> None:
        """Preempt a sequence due to cache memory constraints.

        Preempted sequences are returned to the waiting queue and will
        be re-executed from scratch (recomputation) when cache space is available.
        Any previously cached blocks for this sequence are released, but may be
        recovered via prefix caching if they haven't been overwritten.

        Args:
            sequence: Sequence to preempt.
        """
        logger.info(
            f'Preempting sequence {sequence.seq_id} due to memory constraints.'
        )
        sequence.status = SequenceStatus.WAITING
        self.block_manager.deallocate(sequence)
        self.waiting.appendleft(sequence)

    def postprocess(self, sequences: List[Sequence],
                    token_ids: List[int]) -> None:
        """Update sequences with newly generated tokens and handle completion.

        This method:
        1. Appends new tokens to each sequence
        2. Checks for sequence completion (EOS or max tokens reached)
        3. Deallocates cache blocks for finished sequences
        4. Removes finished sequences from running queue

        Args:
            sequences: Sequences that were just processed.
            token_ids: Newly generated token IDs for each sequence.
        """
        for sequence, token_id in zip(sequences, token_ids):
            # Append new token to sequence
            sequence.append_token(token_id)

            # Check if sequence is finished
            is_eos: bool = (token_id == self.eos) and not sequence.ignore_eos
            is_max_len: bool = sequence.num_completion_tokens >= sequence.max_tokens

            if is_eos or is_max_len:
                # Mark as finished and free resources
                sequence.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(sequence)
                self.running.remove(sequence)
