"""Integration tests for mini-vLLM components.

Tests the interaction between multiple components to ensure
the system works end-to-end.
"""

from pathlib import Path

import pytest

from minivllm.config import Config
from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.sampling_params import SamplingParams


class TestConfigAndSequenceIntegration:
    """Test integration between Config and Sequence."""

    def test_sampling_params_from_config(self, tmp_path: Path) -> None:
        """Test that SamplingParams works with Config."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text('{"model_type": "llama"}')

        config = Config(str(model_dir), max_num_seqs=128)
        sampling_params = SamplingParams(temperature=0.8, max_tokens=512)

        assert config.max_num_seqs == 128
        assert sampling_params.max_tokens == 512

    def test_sequence_with_config_max_tokens(self, tmp_path: Path) -> None:
        """Test sequence creation with parameters matching config."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text('{"model_type": "llama"}')

        config = Config(str(model_dir), max_model_len=2048)
        sampling_params = SamplingParams(max_tokens=256)

        # Sequence max_tokens should be less than config max_model_len
        assert sampling_params.max_tokens < config.max_model_len

        token_ids = [1, 2, 3]
        seq = Sequence(token_ids, sampling_params)
        assert seq.max_tokens == 256


class TestSequenceLifecycle:
    """Test the lifecycle of a sequence through generation."""

    def test_sequence_status_transitions(self) -> None:
        """Test valid status transitions for a sequence."""
        seq = Sequence([1, 2, 3])

        # Initial status is WAITING
        assert seq.status == SequenceStatus.WAITING

        # Transition to RUNNING
        seq.status = SequenceStatus.RUNNING
        assert seq.status == SequenceStatus.RUNNING

        # Generate some tokens
        seq.append_token(4)
        seq.append_token(5)
        assert seq.num_tokens == 5

        # Transition to FINISHED
        seq.status = SequenceStatus.FINISHED
        assert seq.is_finished is True

    def test_sequence_state_preservation(self) -> None:
        """Test that sequence state is preserved through operations."""
        original_prompt = [1, 2, 3]
        params = SamplingParams(temperature=0.9, max_tokens=100)
        seq = Sequence(original_prompt, params)

        # Verify initial state
        assert seq.prompt_token_ids == original_prompt
        assert seq.num_prompt_tokens == 3

        # Add tokens
        seq.status = SequenceStatus.RUNNING
        seq.append_token(4)
        seq.append_token(5)

        # Verify state after additions
        assert seq.prompt_token_ids == original_prompt  # Unchanged
        assert seq.completion_token_ids == [4, 5]
        assert seq.temperature == 0.9
        assert seq.max_tokens == 100


class TestBlockManagement:
    """Test block-based memory management for sequences."""

    def test_single_block_sequence(self) -> None:
        """Test sequence that fits in single block."""
        token_ids = [i for i in range(200)]
        seq = Sequence(token_ids)

        assert seq.num_blocks == 1
        assert seq.num_cached_blocks == 0

    def test_multi_block_sequence(self) -> None:
        """Test sequence spanning multiple blocks."""
        # Create a sequence with 600 tokens (> 2 blocks of 256)
        token_ids = [i for i in range(600)]
        seq = Sequence(token_ids)

        assert seq.num_blocks == 3
        assert seq.last_block_num_tokens == 88  # 600 - 2*256

    def test_block_allocation_tracking(self) -> None:
        """Test tracking of block allocation."""
        seq = Sequence([1] * 256)

        # Initially no cached tokens
        assert seq.num_cached_tokens == 0
        assert seq.num_cached_blocks == 0

        # Simulate caching tokens (would be done by block_manager)
        seq.num_cached_tokens = 256
        assert seq.num_cached_blocks == 1

    def test_block_table_management(self) -> None:
        """Test block table tracking."""
        seq = Sequence([1] * 512)

        # Block table starts empty
        assert seq.block_table == []

        # Simulate block allocation
        seq.block_table = [0, 1, 2]
        assert len(seq.block_table) == 3


class TestSamplingParameters:
    """Test sampling parameter constraints."""

    def test_sampling_params_constraints(self) -> None:
        """Test that sampling parameters enforce constraints."""
        # Valid params
        params = SamplingParams(temperature=0.7,
                                max_tokens=256,
                                ignore_eos=False)
        assert params.temperature == 0.7
        assert params.max_tokens == 256
        assert params.ignore_eos is False

        # Temperature boundary
        with pytest.raises(ValueError):
            SamplingParams(temperature=0)

        # Max tokens boundary
        with pytest.raises(ValueError):
            SamplingParams(max_tokens=0)

    def test_sampling_params_in_sequence(self) -> None:
        """Test sampling parameters applied to sequences."""
        params = SamplingParams(temperature=0.6,
                                max_tokens=512,
                                ignore_eos=True)
        seq = Sequence([1, 2, 3], params)

        assert seq.temperature == params.temperature
        assert seq.max_tokens == params.max_tokens
        assert seq.ignore_eos == params.ignore_eos


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
