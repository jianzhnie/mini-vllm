"""Unit tests for BlockManager module.

This module tests the BlockManager class, focusing on:
- Block allocation and deallocation
- Prefix caching logic
- Persistence of cache across deallocations
- Block boundary handling
"""

import pytest

from minivllm.engine.block_manager import BlockManager
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams


class TestBlockManagerInitialization:
    """Test cases for BlockManager initialization."""

    def test_init(self):
        """Test initialization with valid parameters."""
        bm = BlockManager(num_blocks=10, block_size=16)
        assert len(bm.blocks) == 10
        assert bm.block_size == 16
        assert len(bm.free_block_ids) == 10
        assert len(bm.used_block_ids) == 0
        assert len(bm.hash_to_block_id) == 0

    def test_init_invalid(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            BlockManager(num_blocks=0, block_size=16)
        with pytest.raises(ValueError):
            BlockManager(num_blocks=10, block_size=0)


class TestBlockManagerAllocation:
    """Test cases for block allocation."""

    def test_allocate_simple(self):
        """Test simple allocation for a sequence."""
        bm = BlockManager(num_blocks=10, block_size=4)
        seq = Sequence([1, 2, 3, 4], SamplingParams())  # 1 block

        bm.allocate(seq)

        assert len(seq.block_table) == 1
        assert len(bm.used_block_ids) == 1
        assert len(bm.free_block_ids) == 9
        # Check that block has correct hash and tokens
        block_id = seq.block_table[0]
        block = bm.blocks[block_id]
        assert block.token_ids == [1, 2, 3, 4]
        assert block.hash != -1
        assert bm.hash_to_block_id[block.hash] == block_id

    def test_allocate_partial(self):
        """Test allocation with partial blocks."""
        bm = BlockManager(num_blocks=10, block_size=4)
        seq = Sequence([1, 2], SamplingParams())  # Partial block

        bm.allocate(seq)

        assert len(seq.block_table) == 1
        block_id = seq.block_table[0]
        block = bm.blocks[block_id]
        # Partial blocks don't compute hash immediately
        assert block.hash == -1
        assert (block.token_ids == []
                )  # Partial blocks don't store tokens until filled/updated?
        # Wait, allocate() calls block.update() only if hash_prev != -1 (which implies full block)
        # Let's check allocate logic:
        # if len(token_ids) == block_size: compute hash
        # else: hash_prev = -1
        # if hash_prev != -1: block.update()
        # So partial blocks are NOT updated with tokens in allocate()!
        # This seems like a potential bug or design choice?
        # Sequence.block(i) gets tokens.
        # But Block object needs to know its tokens?
        # Block.token_ids is used for collision check.
        # If it's partial, we don't cache it, so we don't check for collision.
        # So it's fine.
        assert block.hash == -1

    def test_allocate_no_free_blocks(self):
        """Test allocation fails when no blocks available."""
        bm = BlockManager(num_blocks=1, block_size=4)
        seq1 = Sequence([1, 2, 3, 4], SamplingParams())
        bm.allocate(seq1)

        seq2 = Sequence([5, 6, 7, 8], SamplingParams())
        with pytest.raises(ValueError, match='No free blocks'):
            bm.allocate(seq2)


class TestPrefixCaching:
    """Test cases for prefix caching logic."""

    def test_prefix_caching_hit(self):
        """Test that identical sequences share blocks."""
        bm = BlockManager(num_blocks=10, block_size=4)
        seq1 = Sequence([1, 2, 3, 4], SamplingParams())
        bm.allocate(seq1)

        seq2 = Sequence([1, 2, 3, 4], SamplingParams())
        bm.allocate(seq2)

        # Should share the same block
        assert seq1.block_table[0] == seq2.block_table[0]
        block_id = seq1.block_table[0]
        assert bm.blocks[block_id].ref_count == 2
        assert len(bm.used_block_ids) == 1
        assert seq2.num_cached_tokens == 4

    def test_prefix_caching_miss(self):
        """Test that different sequences use different blocks."""
        bm = BlockManager(num_blocks=10, block_size=4)
        seq1 = Sequence([1, 2, 3, 4], SamplingParams())
        bm.allocate(seq1)

        seq2 = Sequence([1, 2, 3, 5], SamplingParams())  # Different last token
        bm.allocate(seq2)

        assert seq1.block_table[0] != seq2.block_table[0]
        assert len(bm.used_block_ids) == 2

    def test_caching_across_deallocations(self):
        """Test that cache persists after deallocation."""
        bm = BlockManager(num_blocks=10, block_size=4)
        seq1 = Sequence([1, 2, 3, 4], SamplingParams())
        bm.allocate(seq1)
        block_id_1 = seq1.block_table[0]

        # Deallocate seq1
        bm.deallocate(seq1)
        assert bm.blocks[block_id_1].ref_count == 0
        assert block_id_1 in bm.free_block_ids
        assert block_id_1 not in bm.used_block_ids
        # Hash mapping should still exist!
        block_hash = bm.blocks[block_id_1].hash
        assert block_hash != -1
        assert bm.hash_to_block_id.get(block_hash) == block_id_1

        # Allocate new sequence with same content
        seq2 = Sequence([1, 2, 3, 4], SamplingParams())
        bm.allocate(seq2)

        # Should reuse the same block ID
        assert seq2.block_table[0] == block_id_1
        assert bm.blocks[block_id_1].ref_count == 1
        assert block_id_1 in bm.used_block_ids
        assert block_id_1 not in bm.free_block_ids
        assert seq2.num_cached_tokens == 4

    def test_cache_invalidation_on_reuse(self):
        """Test that cache mapping is removed when block is reused for different content."""
        bm = BlockManager(num_blocks=1, block_size=4)  # Only 1 block
        seq1 = Sequence([1, 2, 3, 4], SamplingParams())
        bm.allocate(seq1)
        old_hash = bm.blocks[0].hash

        bm.deallocate(seq1)

        # Allocate sequence with DIFFERENT content
        # Should reuse the only available block (0)
        seq2 = Sequence([5, 6, 7, 8], SamplingParams())
        bm.allocate(seq2)

        assert seq2.block_table[0] == 0
        # Old hash mapping should be gone
        assert old_hash not in bm.hash_to_block_id
        # New hash should be there
        new_hash = bm.blocks[0].hash
        assert new_hash != old_hash
        assert bm.hash_to_block_id[new_hash] == 0


class TestBlockBoundary:
    """Test cases for block boundary handling."""

    def test_may_append_allocates(self):
        """Test may_append allocates new block at boundary."""
        bm = BlockManager(num_blocks=10, block_size=2)
        seq = Sequence([1, 2], SamplingParams())  # Full block
        bm.allocate(seq)

        assert len(seq.block_table) == 1

        # Append token 3 -> seq len 3 -> needs new block
        seq.append_token(3)
        bm.may_append(seq)

        assert len(seq.block_table) == 2
        assert len(bm.used_block_ids) == 2

    def test_may_append_finalizes_hash(self):
        """Test may_append finalizes hash when block fills up."""
        bm = BlockManager(num_blocks=10, block_size=2)
        seq = Sequence([1], SamplingParams())  # Partial block
        bm.allocate(seq)

        block = bm.blocks[seq.block_table[0]]
        assert block.hash == -1

        # Append token 2 -> seq len 2 -> fills block
        seq.append_token(2)
        bm.may_append(seq)

        assert block.hash != -1
        assert bm.hash_to_block_id[block.hash] == block.block_id
