"""Block Manager module for KV cache management.

This module provides the BlockManager and Block classes which handle
physical block allocation and management for the KV cache, including
block hashing for prefix caching optimization.
"""

from collections import deque
from typing import Dict, List

import numpy as np
import xxhash

from minivllm.engine.sequence import Sequence


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

    def update(self, hash_val: int, token_ids: List[int]) -> None:
        """Update block with token data and hash.

        Args:
            hash_val: Hash value of the token sequence.
            token_ids: List of token IDs for this block.
        """
        self.hash = hash_val
        self.token_ids = token_ids

    def reset(self) -> None:
        """Reset block to initial allocated state.

        Clears token data and hash, but maintains the block as allocated
        with ref_count=1.
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


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

        Args:
            num_blocks: Total number of physical blocks in the KV cache.
            block_size: Number of tokens per block.
        """
        self.block_size: int = block_size
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        self.free_block_ids: deque = deque(range(num_blocks))
        self.used_block_ids: set = set()

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix: int = -1) -> int:
        """Compute hash of token sequence for prefix caching.

        Args:
            token_ids: List of token IDs to hash.
            prefix: Hash of the previous block (for chained hashing).
                Default -1 for no prefix.

        Returns:
            Hash value of the token sequence (including prefix if provided).
        """
        h: xxhash.xxh64 = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Allocate a block from the free pool.

        Args:
            block_id: ID of the block to allocate.

        Returns:
            The allocated Block object.

        Raises:
            AssertionError: If block is already allocated.
        """
        block: Block = self.blocks[block_id]
        assert block.ref_count == 0, f'Block {block_id} already allocated'
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """Deallocate a block back to the free pool.

        Args:
            block_id: ID of the block to deallocate.

        Raises:
            AssertionError: If block is still in use (ref_count > 0).
        """
        assert self.blocks[block_id].ref_count == 0, (
            f'Cannot deallocate block {block_id} with ref_count > 0')
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if sequence can be allocated.

        Args:
            seq: Sequence to check allocation feasibility for.

        Returns:
            True if enough free blocks exist for the sequence.
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        """Allocate blocks for a sequence, using prefix caching when possible.

        This method:
        1. Iterates through the sequence's blocks
        2. For full blocks, computes hash and checks for matching cached blocks
        3. For matching blocks, increments reference count (cache hit)
        4. For non-matching blocks, allocates a new block (cache miss)
        5. Updates the sequence's block table

        Args:
            seq: Sequence to allocate blocks for. Must have empty block_table.

        Raises:
            AssertionError: If sequence already has a block table.
        """
        assert not seq.block_table, 'Sequence already has allocated blocks'

        h: int = -1
        cache_miss: bool = False

        for i in range(seq.num_blocks):
            token_ids: List[int] = seq.block(i)

            # Compute hash only for full blocks
            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)
            else:
                h = -1

            # Check for cached block
            block_id: int = self.hash_to_block_id.get(h, -1)

            # Validate cache hit (hash match and token ID match)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                # Cache miss: allocate new block
                block_id = self.free_block_ids[0]
                block: Block = self._allocate_block(block_id)
            else:
                # Cache hit: reuse existing block
                seq.num_cached_tokens += self.block_size

                if block_id in self.used_block_ids:
                    # Block already in use: increment reference count
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Block not in use: allocate it
                    block = self._allocate_block(block_id)

            # Update block with token data if needed
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        """Deallocate blocks for a sequence.

        This method:
        1. Decrements reference count for each block
        2. Deallocates blocks with reference count of 0
        3. Clears the sequence's block table

        Args:
            seq: Sequence to deallocate blocks for.
        """
        for block_id in reversed(seq.block_table):
            block: Block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check if a new block can be allocated for a sequence.

        During decode, sequences grow token by token. When a sequence's
        token count crosses a block boundary, a new block is needed.

        Args:
            seq: Sequence to check append feasibility for.

        Returns:
            True if a new block can be allocated when needed.
        """
        # A new block is needed only when moving to a new block boundary
        # This happens when: (len(seq) % block_size == 0) -> (len(seq) + 1)
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 0)

    def may_append(self, seq: Sequence) -> None:
        """Append tokens to a sequence's last block, allocating new
        blocks as needed.

        This method handles the decode phase where sequences grow
        incrementally. It manages:
        1. Allocating new blocks when sequence grows beyond block
           boundary
        2. Finalizing block hashes when blocks become full

        Args:
            seq: Sequence to append tokens to.
        """
        block_table: List[int] = seq.block_table
        last_block: Block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            # Just started a new token in a new block
            # The previous block's hash should be finalized
            assert last_block.hash != -1, 'Previous block hash not finalized'

            # Allocate a new block for this token
            block_id: int = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq) % self.block_size == 0:
            # Just filled a block (now has block_size tokens)
            # Finalize this block's hash
            assert last_block.hash == -1, 'Block hash already set'

            token_ids: List[int] = seq.block(seq.num_blocks - 1)
            prefix: int = (self.blocks[block_table[-2]].hash
                           if len(block_table) > 1 else -1)
            h: int = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        else:
            # Partial block (between 1 and block_size tokens)
            # Nothing to do until we reach a boundary
            assert last_block.hash == -1, 'Partial block should not have hash'
