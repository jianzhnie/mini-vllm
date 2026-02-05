"""Block Manager module for KV cache management.

This module provides the BlockManager and Block classes which handle
physical block allocation and management for the KV cache, including
block hashing for prefix caching optimization.

The BlockManager uses a hash-based prefix cache to enable memory sharing
between sequences with common token prefixes, reducing total memory usage.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Set

import numpy as np
import xxhash

from minivllm.engine.sequence import Sequence
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)


class Block:
    """Represents a physical memory block in the KV cache.

    Each block stores a fixed number of tokens' KV values. Blocks support
    prefix caching by storing a hash of their token sequence, allowing
    physical block sharing when token sequences match.

    Attributes:
        block_id: Unique identifier for this block in the block pool.
        ref_count: Number of sequences referencing this block.
            When 0, the block can be deallocated.
        hash: Hash of the token sequence in this block (-1 if not computed).
            Used for prefix caching to identify matching sequences.
        token_ids: List of token IDs stored in this block.
    """

    def __init__(self, block_id: int) -> None:
        """Initialize a new block.

        Args:
            block_id: Unique identifier for this block.
        """
        self.block_id: int = block_id
        self.ref_count: int = 0
        self.hash: int = -1
        self.token_ids: List[int] = []
        self.last_access_time: Optional[float] = None  # 用于缓存策略

    def update(self, hash_val: int, token_ids: List[int]) -> None:
        """Update block with token data and hash.

        Args:
            hash_val: Hash value of the token sequence.
            token_ids: List of token IDs for this block.
        """
        self.hash = hash_val
        self.token_ids = token_ids.copy()
        self.last_access_time = None  # 重置访问时间

    def reset(self) -> None:
        """Reset block to initial allocated state.

        Clears token data and hash, but maintains the block as allocated
        with ref_count=1.
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids.clear()
        self.last_access_time = None  # 重置访问时间


class BlockManager:
    """Manages physical KV cache blocks and prefix caching.

    This class allocates and deallocates memory blocks for the KV cache,
    implementing prefix caching through token sequence hashing. When multiple
    sequences share the same token prefix, they can reuse the same physical
    block, reducing memory usage.

    The BlockManager tracks:
    - Physical block allocation status
    - Reference counting for shared blocks
    - Hash values for prefix cache matching
    - Block table assignments for sequences

    Attributes:
        block_size: Number of tokens per block.
        blocks: List of all available Block objects.
        hash_to_block_id: Mapping from token hash to physical block ID
            for prefix cache lookups.
        free_block_ids: Queue of unallocated block IDs.
        used_block_ids: Set of currently allocated block IDs.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        """Initialize the block manager.

        Allocates the block pool and initializes all tracking structures.

        Args:
            num_blocks: Total number of physical blocks in the KV cache.
            block_size: Number of tokens per block.

        Raises:
            ValueError: If num_blocks or block_size is invalid (<=0).
        """
        if num_blocks <= 0:
            raise ValueError(f'num_blocks must be positive, got {num_blocks}')
        if block_size <= 0:
            raise ValueError(f'block_size must be positive, got {block_size}')

        self.block_size: int = block_size
        self.num_blocks: int = num_blocks
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        # Use a double-ended queue for efficient allocation/deallocation
        self.free_block_ids: Deque[int] = deque(range(num_blocks))
        # Use a set for efficient membership testing
        self.used_block_ids: Set[int] = set()
        self.stats = {
            'total_allocations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix: int = -1) -> int:
        """Compute hash of token sequence for prefix caching.

        Uses xxhash for fast hashing. If a prefix hash is provided, it's
        included in the computation to support chained hashing where each
        block's hash depends on previous blocks' hashes.

        Args:
            token_ids: List of token IDs to hash. Must not be empty.
            prefix: Hash of the previous block (for chained hashing).
                Default -1 means no prefix.

        Returns:
            Integer hash value of the token sequence (including prefix if
            provided).

        Raises:
            ValueError: If token_ids is empty.

        Note:
            The same token_ids with the same prefix will always produce
            the same hash, enabling prefix cache matching.
        """
        if not token_ids:
            raise ValueError('token_ids cannot be empty for hash computation')

        hash_prev: xxhash.xxh64 = xxhash.xxh64()
        if prefix != -1:
            hash_prev.update(prefix.to_bytes(8, 'little'))
        if isinstance(token_ids, list):
            token_array = np.array(token_ids, dtype=np.int32)
        else:
            token_array = token_ids
        hash_prev.update(input=token_array.tobytes())
        return hash_prev.intdigest()

    def _allocate_block(self, block_id: int, reset: bool = True) -> Block:
        """Allocate a block from the free pool.

        Args:
            block_id: ID of the block to allocate.
            reset: Whether to reset the block state (clear data).
                Set to False when resurrecting a cached block.

        Returns:
            The allocated Block object.

        Raises:
            RuntimeError: If block is already allocated or block_id is invalid.
        """
        if block_id < 0 or block_id >= len(self.blocks):
            raise RuntimeError(f'Invalid block_id: {block_id}. '
                               f'Valid range: [0, {len(self.blocks)})')
        if not self.free_block_ids:
            raise RuntimeError('No free blocks available')

        # Check if it is the first element (O(1) operation)
        if self.free_block_ids and self.free_block_ids[0] == block_id:
            actual_id: int = self.free_block_ids.popleft()
        else:
            # Remove specific element from middle (O(n) operation)
            if block_id in self.free_block_ids:
                self.free_block_ids.remove(block_id)
                actual_id = block_id
            else:
                raise RuntimeError(f'Block {block_id} not in free blocks')

        block: Block = self.blocks[actual_id]
        if block.ref_count != 0:
            raise RuntimeError(
                f'Block {actual_id} already allocated with ref_count={block.ref_count}'
            )

        if reset:
            # If resetting, we need to clean up old hash mapping if it exists
            if block.hash != -1 and self.hash_to_block_id.get(
                    block.hash) == actual_id:
                del self.hash_to_block_id[block.hash]
            block.reset()
        else:
            # Resurrecting: just set ref_count to 1
            block.ref_count = 1

        self.used_block_ids.add(actual_id)
        self.stats['total_allocations'] += 1
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """Deallocate a block back to the free pool.

        Args:
            block_id: ID of the block to deallocate.

        Raises:
            RuntimeError: If block is still in use (ref_count > 0) or invalid.
        """
        if block_id < 0 or block_id >= len(self.blocks):
            raise RuntimeError(f'Invalid block_id: {block_id}. '
                               f'Valid range: [0, {len(self.blocks)})')

        block: Block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError(f'Cannot deallocate block {block_id} with '
                               f'ref_count={block.ref_count} (must be 0)')
        # 清理哈希映射
        if block.hash != -1 and self.hash_to_block_id.get(
                block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, sequence: Sequence) -> bool:
        """Check if sequence can be allocated.

        Args:
            sequence: Sequence to check allocation feasibility for.

        Returns:
            True if enough free blocks exist for the sequence.
        """
        required_blocks = sequence.num_blocks
        available_blocks = len(self.free_block_ids)
        return available_blocks >= required_blocks

    def allocate(self, sequence: Sequence) -> None:
        """Allocate blocks for a sequence, using prefix caching when possible.

        This method implements the core prefix caching logic:
        1. Iterates through the sequence's blocks
        2. For full blocks (block_size tokens), computes hash and checks cache
        3. For cache hits (matching hash and tokens), increments ref count
        4. For cache misses, allocates a new block
        5. Updates the sequence's block table with allocated block IDs

        Prefix caching works by:
        - Full blocks have their hash stored in hash_to_block_id mapping
        - When a new sequence's block matches a cached block's hash and
          tokens, it can reuse the physical block
        - Reference counting ensures blocks are kept alive while in use

        Args:
            sequence: Sequence to allocate blocks for. Must have empty block_table.

        Raises:
            AssertionError: If sequence already has an allocated block table.
            ValueError: If not enough free blocks available for allocation.

        Note:
            After allocation, seq.block_table will contain the physical block
            IDs for each logical block in the sequence.
        """
        assert not sequence.block_table, (
            'Sequence already has allocated blocks')

        if len(self.free_block_ids) < sequence.num_blocks:
            raise ValueError(
                f'Not enough free blocks: need {sequence.num_blocks}, '
                f'have {len(self.free_block_ids)}')

        hash_prev: int = -1  # Hash of previous block for chained hashing
        cache_miss: bool = False  # Flag to track if we've encountered a cache miss

        # Iterate through each logical block in the sequence
        for i in range(sequence.num_blocks):
            # Get token IDs for this block
            token_ids: List[int] = sequence.block(i)

            # Compute hash only for full blocks (others can't be cached)
            if len(token_ids) == self.block_size:
                # Compute chained hash: combines current block tokens with previous block's hash
                # This ensures hash uniquely identifies the entire prefix up to this block
                hash_prev = self.compute_hash(token_ids, hash_prev)
            else:
                hash_prev = -1  # Partial blocks can't be cached

            # Look up in prefix cache
            block_id: int = self.hash_to_block_id.get(hash_prev, -1)

            # Validate cache hit: hash match AND token ID match
            # This double-check ensures we don't have hash collisions
            if block_id == -1:
                cache_miss = True  # No hash match found
            elif block_id not in range(len(self.blocks)):
                cache_miss = True  # Invalid block_id
            elif self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # Hash collision detected - token IDs don't match

            if cache_miss:
                # Cache miss: allocate a new block from free pool
                if not self.free_block_ids:
                    raise ValueError(
                        f'No free blocks available for allocation. '
                        f'Need {sequence.num_blocks} blocks, '
                        f'have {len(self.free_block_ids)} free blocks.')
                block_id = self.free_block_ids[0]
                block: Block = self._allocate_block(block_id)
                self.stats['cache_misses'] += 1
            else:
                # Cache hit: reuse existing block
                sequence.num_cached_tokens += self.block_size  # Track cached tokens

                if block_id in self.used_block_ids:
                    # Block is already in use by another sequence
                    block = self.blocks[block_id]
                    block.ref_count += 1  # Increment reference count
                else:
                    # Block exists in cache but not currently allocated
                    # Resurrect it without resetting data
                    block = self._allocate_block(block_id, reset=False)

            # Update block with token data and register in hash table (for full blocks)
            if hash_prev != -1:
                block.update(hash_prev,
                             token_ids)  # Store hash and tokens in block
                self.hash_to_block_id[
                    hash_prev] = block_id  # Register in cache mapping

                sequence.num_cached_tokens += self.block_size

            # Add physical block ID to sequence's block table
            sequence.block_table.append(block_id)

        total = self.stats['cache_hits'] + self.stats['cache_misses']
        if total > 0:
            hit_rate = self.stats['cache_hits'] / total * 100
            logger.info(f'Prefix cache hit rate: {hit_rate:.1f}%')

    def deallocate(self, sequence: Sequence) -> None:
        """Deallocate blocks for a sequence.

        This method:
        1. Decrements reference count for each block
        2. Deallocates blocks with reference count of 0
        3. Clears the sequence's block table

        Args:
            sequence: Sequence to deallocate blocks for.
        """
        for block_id in reversed(sequence.block_table):
            block: Block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        sequence.num_cached_tokens = 0
        sequence.block_table.clear()

    def can_append(self, sequence: Sequence) -> bool:
        """Check if a new block can be allocated for a sequence.

        During decode, sequences grow token by token. When a sequence's
        token count crosses a block boundary (len(seq) % block_size == 0),
        a new block is needed on the next append.

        This method checks if there are enough free blocks to accommodate
        the next block allocation when needed.

        Args:
            sequence: Sequence to check append feasibility for.

        Returns:
            True if a new block can be allocated when the next boundary
            is crossed, False otherwise.

        Note:
            This returns True most of the time (when we're not at a boundary).
            Only when at a block boundary and low on free blocks does it
            return False, triggering sequence preemption.
        """
        # A new block is needed only when crossing a block boundary
        # This happens when: len(sequence) % block_size == 0 -> len(sequence) + 1
        needs_new_block: bool = (len(sequence) % self.block_size == 0)
        required_blocks = 1 if needs_new_block else 0

        return len(self.free_block_ids) >= required_blocks

    def may_append(self, sequence: Sequence) -> None:
        """Append tokens to a sequence's last block, allocating new blocks
        as needed.

        This method handles the decode phase where sequences grow incrementally,
        managing block boundaries:
        1. When starting a new block (just moved past block_size boundary):
           - Allocate a new physical block
        2. When filling a block (just reached block_size tokens):
           - Finalize the block's hash for prefix caching
        3. When in a partial block: Do nothing

        Args:
            sequence: Sequence to append tokens to.

        Raises:
            AssertionError: If invariants about block state are violated
                (e.g., trying to allocate when no blocks are free).
        """
        block_table: List[int] = sequence.block_table
        last_block: Block = self.blocks[block_table[-1]]

        if len(sequence) % self.block_size == 1:
            # Just started a new token in a new block (crossed boundary)
            # The previous block should already have its hash finalized
            assert last_block.hash != -1, (
                f'Previous block {last_block.block_id} hash not finalized')

            # Allocate a new block for this token
            if not self.free_block_ids:
                raise ValueError('No free blocks available for allocation. '
                                 'This may trigger sequence preemption.')

            # Pick any free block
            block_id: int = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(sequence) % self.block_size == 0:
            # Just filled a block (now has block_size tokens)
            # Finalize this block's hash for prefix caching
            assert last_block.hash == -1, (
                f'Block {last_block.block_id} hash already set')

            token_ids: List[int] = sequence.block(sequence.num_blocks - 1)
            if len(block_table) > 1:
                prefix: int = self.blocks[block_table[-2]].hash
            else:
                prefix: int = -1

            hash_val: int = self.compute_hash(token_ids, prefix)
            last_block.update(hash_val, token_ids)
            self.hash_to_block_id[hash_val] = last_block.block_id

        else:
            # In a partial block (between 1 and block_size-1 tokens)
            # Nothing to do until we reach a boundary
            assert last_block.hash == -1, (
                f'Partial block {last_block.block_id} should not have hash')
