"""Tests for minivllm.engine.distributed_manager.DistributedManager.

Covers the single-process code paths directly and the distributed init/
validation path with ``torch.distributed`` mocked (no real process group).
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from minivllm.config import Config
from minivllm.engine.distributed_manager import DistributedManager


def _mgr(temp_model_dir, tp=1):
    return DistributedManager(Config(model=str(temp_model_dir), tensor_parallel_size=tp), 0)


def test_single_process_not_distributed(temp_model_dir):
    dm = _mgr(temp_model_dir)
    assert dm.is_distributed is False
    dm.initialize()
    assert dm._initialized is True
    assert dm.backend is None


def test_single_process_broadcast_passthrough(temp_model_dir):
    dm = _mgr(temp_model_dir)
    assert dm.broadcast_data({"a": [1, 2]}) == {"a": [1, 2]}


def test_move_to_device_passthrough_for_gloo(temp_model_dir):
    dm = _mgr(temp_model_dir)
    dm.backend = "gloo"
    t = torch.ones(3)
    assert dm._move_to_device(t) is t


def test_distributed_initialize_validates_allreduce(temp_model_dir):
    from minivllm.engine import distributed_manager as dm_mod

    dm = _mgr(temp_model_dir, tp=2)
    assert dm.is_distributed is True
    with patch.object(dm_mod, "get_distributed_backend", return_value="gloo"), patch.object(
        dm_mod, "dist"
    ) as md:
        md.is_initialized.return_value = False
        md.ReduceOp.SUM = "SUM"

        def fake_all_reduce(t, op=None):
            t.copy_(torch.tensor(1.0))

        md.all_reduce.side_effect = fake_all_reduce
        dm.initialize()
    assert dm._initialized is True
    assert dm.backend == "gloo"


def test_backend_env_override(monkeypatch):
    monkeypatch.setenv("MINIVLLM_TP_BACKEND", "gloo")
    from minivllm.utils.device import get_distributed_backend

    assert get_distributed_backend() == "gloo"
