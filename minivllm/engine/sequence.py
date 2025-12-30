"""Sequence management module for handling LLM input sequences.

This module provides the Sequence class which represents and manages
individual sequences/requests in the LLM engine, including token tracking,
cache management, and state serialization.
"""

from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Any, Iterator, List, Optional, Tuple

from minivllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Enumeration of possible sequence statuses.

    Attributes:
        WAITING: Sequence is waiting to be processed.
        RUNNING: Sequence is currently being processed.
        FINISHED: Sequence generation is complete.
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """Manages a single sequence (request) in the LLM engine.

    This class tracks a sequence's tokens, state, KV cache, and sampling
    parameters. It supports token-level operations, block-based cache
    management, and state serialization for distributed processing.

    Class Attributes:
        block_size: Size of each token block (in tokens). Used for
            organizing tokens into fixed-size blocks for efficient
            cache management. Default: 256.
        counter: Global counter for generating unique sequence IDs.

    Attributes:
        seq_id: Unique identifier for this sequence.
        status: Current status of the sequence (WAITING, RUNNING, FINISHED).
        token_ids: List of all token IDs in the sequence.
        last_token: The most recently added token ID.
        num_tokens: Total number of tokens in the sequence.
        num_prompt_tokens: Number of tokens in the original prompt.
        num_cached_tokens: Number of tokens that have been cached.
        block_table: List of physical block IDs for KV cache mapping.
        temperature: Sampling temperature for text generation.
        max_tokens: Maximum number of tokens to generate.
        ignore_eos: Whether to ignore end-of-sequence tokens.
    """

    block_size: int = 256
    counter: Iterator[int] = count()

    def __init__(
        self,
        token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> None:
        """Initialize a new sequence.

        Args:
            token_ids: List of initial token IDs (the prompt).
            sampling_params: Sampling parameters for generation.
                Defaults to a new SamplingParams() instance if not provided.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        if not token_ids:
            raise ValueError('token_ids must contain at least one token')

        self.seq_id: int = next(Sequence.counter)
        self.status: SequenceStatus = SequenceStatus.WAITING
        self.token_ids: List[int] = copy(token_ids)
        self.last_token: int = self.token_ids[-1]
        self.num_tokens: int = len(self.token_ids)
        self.num_prompt_tokens: int = len(self.token_ids)
        self.num_cached_tokens: int = 0
        self.block_table: List[int] = []
        self.temperature: float = sampling_params.temperature
        self.max_tokens: int = sampling_params.max_tokens
        self.ignore_eos: bool = sampling_params.ignore_eos

    def __len__(self) -> int:
        """Return the total number of tokens in the sequence.

        Returns:
            The total number of tokens (prompt + completion).
        """
        return self.num_tokens

    def __getitem__(self, key: int) -> int:
        """Get a token by index.

        Args:
            key: Index of the token to retrieve.

        Returns:
            The token ID at the specified index.
        """
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        """Check if sequence generation is finished.

        Returns:
            True if the sequence status is FINISHED, False otherwise.
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """Get the number of generated completion tokens.

        Returns:
            The number of tokens generated after the prompt.
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> List[int]:
        """Get the token IDs from the original prompt.

        Returns:
            List of token IDs that make up the prompt.
        """
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> List[int]:
        """Get the generated completion token IDs.

        Returns:
            List of token IDs that have been generated (after prompt).
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        """Get the number of blocks that have been cached.

        Returns:
            Number of complete blocks in the KV cache.
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        """Get the total number of blocks needed for all tokens.

        Calculates the total number of blocks required to store all
        tokens in the sequence, using ceiling division.

        Returns:
            Total number of blocks needed.
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """Get the number of tokens in the last (possibly incomplete) block.

        Returns:
            Number of tokens in the last block, which may be less than
            block_size if not full.
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> List[int]:
        """Get token IDs for a specific block.

        Args:
            i: Index of the block to retrieve (0-indexed).

        Returns:
            List of token IDs in the specified block.

        Raises:
            AssertionError: If block index i is out of valid range.
        """
        if not (0 <= i < self.num_blocks):
            raise IndexError(
                f'Block index {i} out of range [0, {self.num_blocks})')
        return self.token_ids[i * self.block_size:(i + 1) * self.block_size]

    def append_token(self, token_id: int) -> None:
        """Add a new token to the sequence.

        This method is called during generation to add newly sampled
        tokens to the sequence. It updates all relevant sequence state
        including token tracking and completion counter.

        Args:
            token_id: The token ID to append to the sequence.

        Note:
            This method is called for each token generated during the
            decode phase. It updates token_ids, last_token, and num_tokens
            but does NOT check for sequence completion - that should be
            handled by the scheduler based on EOS tokens or max_tokens.

        Raises:
            ValueError: If token_id is negative (invalid token).
        """
        if token_id < 0:
            raise ValueError(f'Token ID must be non-negative, got {token_id}')

        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self) -> Tuple[int, int, int, List[int], Any]:
        """Prepare sequence state for serialization/pickling.

        This method optimizes serialization by only storing complete
        prompt tokens separately; completion tokens are stored as
        just the last generated token to save memory.

        Returns:
            A tuple containing:
            - num_tokens: Total number of tokens
            - num_prompt_tokens: Number of prompt tokens
            - num_cached_tokens: Number of cached tokens
            - block_table: Physical block table for cache mapping
            - token_ids or last_token: Full prompt or just the last
              generated token (for memory efficiency)
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids
            if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state: Tuple[int, int, int, List[int],
                                        Any]) -> None:
        """Restore sequence state from serialization/unpickling.

        This method reconstructs a sequence from its serialized state,
        restoring all required attributes.

        Args:
            state: A tuple containing serialized sequence state in the
                format returned by __getstate__.
        """
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
        ) = state[:-1]

        if self.num_completion_tokens == 0:
            # If no completion tokens, restore full token list
            self.token_ids = list(state[-1])
            self.last_token = self.token_ids[-1]
        else:
            # If has completions, only last token was serialized
            self.token_ids = [0] * self.num_prompt_tokens
            self.last_token = int(state[-1])
