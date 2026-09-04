"""Regression test: a single running sequence at a block boundary with no free
blocks must recover via self-preemption + prefill, not crash the scheduler.

See scheduler.py::schedule() — previously this hit the "should never happen"
RuntimeError because _schedule_decode self-preempts (moving the seq to waiting)
and _schedule_prefill was not re-attempted within the same step.
"""

from __future__ import annotations

from types import SimpleNamespace

from minivllm.engine.scheduler import Scheduler
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams


def _scheduler(num_blocks: int, block_size: int) -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        eos=-1,
        num_kvcache_blocks=num_blocks,
        kvcache_block_size=block_size,
    )
    return Scheduler(config)


def test_self_preemption_recovers_instead_of_raising():
    # 2-block pool of size 4. A 5-token sequence occupies both blocks and sits
    # at len%block_size == 1, so its next decode token needs a 3rd block that
    # does not exist -> self-preemption is the only way to make progress.
    scheduler = _scheduler(num_blocks=2, block_size=4)
    scheduler.waiting.append(
        Sequence(
            token_ids=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(max_tokens=16),
            block_size=4,
        )
    )

    # Step 1: prefill consumes both blocks; the sequence is now running with
    # zero free blocks.
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill
    assert scheduler.block_manager.get_num_free_blocks() == 0

    # Step 2: decode cannot append (no free block at the boundary). The
    # scheduler must preempt the sequence back to waiting and re-prefill it
    # rather than raising.
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill
    assert len(seqs) == 1
    assert seqs[0].status.name == "RUNNING"

    # The pool is still the constraint, so we must not be "finished".
    assert not scheduler.is_finished()
