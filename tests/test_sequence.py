"""Unit tests for Sequence class.

Tests sequence management functionality including token tracking,
block management, and state properties.
"""

import pytest

from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.sampling_params import SamplingParams


class TestSequenceInitialization:
    """Test Sequence initialization and basic properties."""

    def test_sequence_creation(self) -> None:
        """Test basic sequence creation."""
        token_ids = [1, 2, 3, 4, 5]
        seq = Sequence(token_ids)

        assert seq.num_tokens == 5
        assert seq.num_prompt_tokens == 5
        assert seq.status == SequenceStatus.WAITING
        assert len(seq) == 5

    def test_sequence_with_sampling_params(self) -> None:
        """Test sequence creation with sampling parameters."""
        token_ids = [1, 2, 3]
        params = SamplingParams(temperature=0.8,
                                max_tokens=100,
                                ignore_eos=True)
        seq = Sequence(token_ids, params)

        assert seq.temperature == 0.8
        assert seq.max_tokens == 100
        assert seq.ignore_eos is True

    def test_empty_sequence_raises_error(self) -> None:
        """Test that empty token_ids raises ValueError."""
        with pytest.raises(ValueError,
                           match='must contain at least one token'):
            Sequence([])

    def test_sequence_last_token(self) -> None:
        """Test that last_token is correctly set."""
        token_ids = [1, 2, 3, 4, 5]
        seq = Sequence(token_ids)
        assert seq.last_token == 5

    def test_sequence_unique_ids(self) -> None:
        """Test that sequences have unique IDs."""
        seq1 = Sequence([1, 2, 3])
        seq2 = Sequence([1, 2, 3])
        assert seq1.seq_id != seq2.seq_id


class TestSequenceTokenManagement:
    """Test token-level operations on sequences."""

    def test_append_token(self) -> None:
        """Test appending a token to a sequence."""
        seq = Sequence([1, 2, 3])
        seq.status = SequenceStatus.RUNNING

        seq.append_token(4)
        assert seq.num_tokens == 4
        assert seq.last_token == 4
        assert len(seq) == 4

    def test_append_token_to_finished_sequence_raises_error(self) -> None:
        """Test that appending to finished sequence raises error."""
        seq = Sequence([1, 2, 3])
        seq.status = SequenceStatus.FINISHED

        with pytest.raises(RuntimeError,
                           match='Cannot append token to finished'):
            seq.append_token(4)

    def test_append_invalid_token_id(self) -> None:
        """Test that negative token IDs are rejected."""
        seq = Sequence([1, 2, 3])
        seq.status = SequenceStatus.RUNNING

        with pytest.raises(ValueError, match='Token ID must be non-negative'):
            seq.append_token(-1)

    def test_completion_tokens(self) -> None:
        """Test tracking of completion tokens."""
        seq = Sequence([1, 2, 3])
        seq.status = SequenceStatus.RUNNING

        assert seq.num_completion_tokens == 0
        seq.append_token(4)
        assert seq.num_completion_tokens == 1
        seq.append_token(5)
        assert seq.num_completion_tokens == 2


class TestSequenceProperties:
    """Test sequence properties and block management."""

    def test_prompt_token_ids(self) -> None:
        """Test prompt_token_ids property."""
        token_ids = [1, 2, 3]
        seq = Sequence(token_ids)
        assert seq.prompt_token_ids == [1, 2, 3]

    def test_completion_token_ids_empty(self) -> None:
        """Test completion_token_ids when empty."""
        token_ids = [1, 2, 3]
        seq = Sequence(token_ids)
        assert seq.completion_token_ids == []

    def test_completion_token_ids_with_generated(self) -> None:
        """Test completion_token_ids after generating tokens."""
        token_ids = [1, 2, 3]
        seq = Sequence(token_ids)
        seq.status = SequenceStatus.RUNNING

        seq.append_token(4)
        seq.append_token(5)
        assert seq.completion_token_ids == [4, 5]

    def test_is_finished_property(self) -> None:
        """Test is_finished property."""
        seq = Sequence([1, 2, 3])
        assert seq.is_finished is False

        seq.status = SequenceStatus.FINISHED
        assert seq.is_finished is True

    def test_num_blocks_calculation(self) -> None:
        """Test num_blocks calculation."""
        seq = Sequence([1] * 256)  # Exactly one block
        assert seq.num_blocks == 1

        seq.append_token(2)
        seq.status = SequenceStatus.RUNNING
        assert seq.num_blocks == 2  # Now needs 2 blocks

    def test_block_retrieval(self) -> None:
        """Test block() method for retrieving token blocks."""
        token_ids = [i for i in range(512)]  # 2 blocks
        seq = Sequence(token_ids)

        block_0 = seq.block(0)
        assert len(block_0) == 256
        assert block_0 == list(range(256))

        block_1 = seq.block(1)
        assert len(block_1) == 256
        assert block_1 == list(range(256, 512))

    def test_block_retrieval_out_of_range(self) -> None:
        """Test that accessing invalid block raises IndexError."""
        seq = Sequence([1] * 256)

        with pytest.raises(IndexError, match='Block index'):
            seq.block(1)


class TestSequenceGetItem:
    """Test sequence indexing with __getitem__."""

    def test_getitem_access(self) -> None:
        """Test accessing tokens by index."""
        token_ids = [10, 20, 30, 40, 50]
        seq = Sequence(token_ids)

        assert seq[0] == 10
        assert seq[2] == 30
        assert seq[4] == 50

    def test_getitem_negative_index(self) -> None:
        """Test negative indexing."""
        token_ids = [10, 20, 30, 40, 50]
        seq = Sequence(token_ids)

        assert seq[-1] == 50
        assert seq[-2] == 40


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
