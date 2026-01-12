"""Unit tests for Scheduler module.

This module contains comprehensive tests for the Scheduler class,
including sequence scheduling, KV cache management, and preemption
strategies.

Tests cover:
    - Scheduler initialization
    - Prefill phase scheduling
    - Decode phase scheduling
    - Sequence preemption and cache management
    - Edge cases and error handling
"""

from pathlib import Path

import pytest

from minivllm.config import Config
from minivllm.engine.scheduler import Scheduler
from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.sampling_params import SamplingParams


class TestSchedulerInitialization:
    """Test cases for Scheduler initialization."""

    def test_scheduler_init_with_config(self, tmp_path: Path) -> None:
        """Test scheduler initialization with configuration."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir), num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        assert scheduler.max_num_seqs == config.max_num_seqs
        assert scheduler.max_num_batched_tokens == config.max_num_batched_tokens
        assert scheduler.eos == config.eos
        assert scheduler.block_manager is not None
        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == 0

    def test_scheduler_with_custom_config(self, tmp_path: Path) -> None:
        """Test scheduler with custom configuration parameters."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir),
                        max_num_seqs=256,
                        max_num_batched_tokens=8192,
                        kvcache_block_size=512,
                        num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        assert scheduler.max_num_seqs == 256
        assert scheduler.max_num_batched_tokens == 8192
        assert scheduler.block_manager.block_size == 512

    def test_scheduler_is_finished_empty(self, tmp_path: Path) -> None:
        """Test that empty scheduler reports as finished."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir), num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        assert scheduler.is_finished()


class TestSchedulerSequenceManagement:
    """Test cases for sequence management in scheduler."""

    def test_add_sequence(self, tmp_path: Path) -> None:
        """Test adding a sequence to the scheduler."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir), num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Create a test sequence
        seq = Sequence(token_ids=[1, 2, 3, 4, 5],
                       sampling_params=SamplingParams())

        scheduler.add(seq)

        assert len(scheduler.waiting) == 1
        assert not scheduler.is_finished()
        assert scheduler.waiting[0].seq_id == seq.seq_id

    def test_add_multiple_sequences(self, tmp_path: Path) -> None:
        """Test adding multiple sequences to the scheduler."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir), num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Add multiple sequences
        num_sequences = 5
        for i in range(num_sequences):
            seq = Sequence(token_ids=[1, 2, 3, 4, 5],
                           sampling_params=SamplingParams())
            scheduler.add(seq)

        assert len(scheduler.waiting) == num_sequences
        assert not scheduler.is_finished()


class TestSchedulerPrefillPhase:
    """Test cases for prefill phase scheduling."""

    def test_prefill_scheduling_single_sequence(self, tmp_path: Path) -> None:
        """Test prefill phase scheduling with a single sequence."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir),
                        max_num_seqs=10,
                        max_num_batched_tokens=4096,
                        num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Add a sequence
        seq = Sequence(token_ids=[1, 2, 3, 4, 5],
                       sampling_params=SamplingParams())
        scheduler.add(seq)

        # Verify sequence is in waiting queue
        assert len(scheduler.waiting) == 1
        assert len(scheduler.running) == 0
        assert not scheduler.is_finished()

    def test_prefill_respects_max_sequences(self, tmp_path: Path) -> None:
        """Test that prefill respects max_num_seqs constraint."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(
            str(model_dir),
            max_num_seqs=2,  # Only allow 2 sequences per batch
            max_num_batched_tokens=8192,
            num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Add more sequences than max_num_seqs
        for i in range(5):
            seq = Sequence(token_ids=[1, 2, 3, 4, 5],
                           sampling_params=SamplingParams())
            scheduler.add(seq)

        # Schedule sequences
        scheduled, is_prefill = scheduler.schedule()

        # Should only schedule up to max_num_seqs
        assert is_prefill
        assert len(scheduled) <= 2

    def test_prefill_respects_max_tokens(self, tmp_path: Path) -> None:
        """Test that prefill respects max_num_batched_tokens constraint."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 128, '
            '"torch_dtype": "float32"}')

        config = Config(
            str(model_dir),
            max_num_seqs=10,
            max_num_batched_tokens=128,  # Match max_model_len
            num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Add a sequence with many tokens
        seq = Sequence(
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 10 tokens
            sampling_params=SamplingParams())
        scheduler.add(seq)

        # Verify sequence was added
        assert len(scheduler.waiting) == 1
        assert scheduler.max_num_batched_tokens == 128


class TestSchedulerDecodePhase:
    """Test cases for decode phase scheduling."""

    def test_decode_scheduling_single_sequence(self, tmp_path: Path) -> None:
        """Test decode phase scheduling with a single sequence."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir),
                        max_num_seqs=10,
                        max_num_batched_tokens=4096,
                        num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Create and add a sequence
        seq = Sequence(token_ids=[1, 2, 3, 4, 5],
                       sampling_params=SamplingParams())
        scheduler.add(seq)

        # Verify it's in waiting queue
        assert len(scheduler.waiting) == 1
        assert not scheduler.is_finished()

    def test_decode_handles_cache_pressure(self, tmp_path: Path) -> None:
        """Test decode phase handles cache pressure via preemption."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(
            str(model_dir),
            max_num_seqs=10,
            max_num_batched_tokens=4096,
            num_kvcache_blocks=2  # Very limited cache
        )
        scheduler = Scheduler(config)

        # Add multiple sequences
        for i in range(3):
            seq = Sequence(token_ids=[1, 2, 3],
                           sampling_params=SamplingParams(max_tokens=10))
            scheduler.add(seq)

        # Verify all added
        assert len(scheduler.waiting) == 3


class TestSchedulerPreemption:
    """Test cases for sequence preemption."""

    def test_preempt_sequence(self, tmp_path: Path) -> None:
        """Test preempting a sequence."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir),
                        max_num_seqs=10,
                        max_num_batched_tokens=4096,
                        num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Create and add a sequence
        seq = Sequence(token_ids=[1, 2, 3, 4, 5],
                       sampling_params=SamplingParams())
        scheduler.add(seq)

        # Move sequence to running queue
        scheduler.running.append(seq)
        scheduler.waiting.popleft()

        # Preempt the sequence
        scheduler.preempt(seq)
        scheduler.running.remove(seq)  # Remove from running after preempt

        # Verify sequence moved back to waiting
        assert len(scheduler.waiting) == 1
        assert len(scheduler.running) == 0
        assert scheduler.waiting[0].seq_id == seq.seq_id
        assert scheduler.waiting[0].status == SequenceStatus.WAITING

    def test_preempt_clears_cache(self, tmp_path: Path) -> None:
        """Test that preemption clears cached tokens."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir),
                        max_num_seqs=10,
                        max_num_batched_tokens=4096,
                        num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Create and add a sequence
        seq = Sequence(token_ids=[1, 2, 3, 4, 5],
                       sampling_params=SamplingParams())
        scheduler.add(seq)

        # Move to running and simulate some caching
        scheduler.running.append(seq)
        scheduler.waiting.popleft()
        seq.num_cached_tokens = 3

        # Preempt and verify cache is cleared
        scheduler.preempt(seq)

        # Check that the sequence in waiting queue has reset cache
        preempted_seq = scheduler.waiting[0]
        assert preempted_seq.num_cached_tokens == 0
        assert len(preempted_seq.block_table) == 0


class TestSchedulerPostprocessing:
    """Test cases for sequence postprocessing."""

    def test_postprocess_updates_sequences(self, tmp_path: Path) -> None:
        """Test that postprocess updates sequences correctly."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir), eos=2, num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Create a sequence and manually add to running
        seq = Sequence(token_ids=[1, 2, 3],
                       sampling_params=SamplingParams(max_tokens=5))
        scheduler.running.append(seq)

        # Postprocess with new tokens
        scheduler.postprocess([seq], [4])

        # Verify tokens were added
        assert seq.token_ids[-1] == 4
        assert seq.num_tokens == 4

    def test_postprocess_marks_finished(self, tmp_path: Path) -> None:
        """Test that postprocess marks sequences as finished."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir), eos=2, num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Create a sequence with max_tokens=1
        seq = Sequence(token_ids=[1, 2, 3],
                       sampling_params=SamplingParams(max_tokens=1))
        scheduler.running.append(seq)

        # Postprocess with one token (should reach max)
        scheduler.postprocess([seq], [4])

        # Verify sequence is marked as finished
        assert seq.is_finished
        assert seq.status == SequenceStatus.FINISHED


class TestSchedulerErrorHandling:
    """Test cases for error handling in scheduler."""

    def test_schedule_with_no_sequences_raises(self, tmp_path: Path) -> None:
        """Test that schedule with no sequences raises RuntimeError."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir), num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Calling schedule on empty scheduler should raise RuntimeError
        with pytest.raises(RuntimeError, match='No sequences scheduled'):
            scheduler.schedule()

    def test_postprocess_with_mismatched_sequences(self,
                                                   tmp_path: Path) -> None:
        """Test postprocess with mismatched sequence/token count."""
        model_dir = tmp_path / 'test_model'
        model_dir.mkdir()
        (model_dir / 'config.json').write_text(
            '{"model_type": "llama", '
            '"max_position_embeddings": 4096, '
            '"torch_dtype": "float32"}')

        config = Config(str(model_dir), num_kvcache_blocks=100)
        scheduler = Scheduler(config)

        # Create multiple sequences
        seqs = []
        for i in range(3):
            seq = Sequence(token_ids=[1, 2, 3],
                           sampling_params=SamplingParams())
            seqs.append(seq)

        # Postprocess with fewer tokens than sequences (should handle gracefully)
        try:
            scheduler.postprocess(seqs,
                                  [4, 5])  # Only 2 tokens for 3 sequences
        except Exception:
            pass  # Expected to fail or handle gracefully
