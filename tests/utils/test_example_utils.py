"""Tests for examples/example_utils (shared example helpers)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# example_utils lives in examples/ (not the minivllm package); put it on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from example_utils import (
    DEFAULT_MODEL,
    MODEL_PATHS,
    apply_darwin_cpu_fallback,
    make_config,
    print_banner,
    resolve_model,
)


def test_default_model_is_a_registry_key():
    assert DEFAULT_MODEL in MODEL_PATHS


def test_resolve_model_short_name():
    assert resolve_model("qwen3") == MODEL_PATHS["qwen3"]


def test_resolve_model_local_dir_passthrough(tmp_path):
    assert resolve_model(str(tmp_path)) == str(tmp_path)


def test_resolve_model_hub_id_passthrough():
    assert resolve_model("acme/foo-model") == "acme/foo-model"


def test_make_config_resolves_and_sets_fields(temp_model_dir):
    cfg = make_config(
        str(temp_model_dir),
        dtype="float16",
        tp=2,
        max_model_len=256,
        device_memory_utilization=0.5,
    )
    assert cfg.model == str(temp_model_dir)
    assert cfg.dtype == "float16"
    assert cfg.tensor_parallel_size == 2
    assert cfg.max_model_len == 256
    assert cfg.device_memory_utilization == 0.5
    assert cfg.enforce_eager is True


def test_make_config_overrides_forwarded(temp_model_dir):
    cfg = make_config(str(temp_model_dir), use_buffered_page_attention=True)
    assert cfg.use_buffered_page_attention is True


def test_print_banner(capsys):
    print_banner("My Section", width=40)
    out = capsys.readouterr().out
    assert "=" * 40 in out
    assert "My Section" in out


def test_apply_darwin_cpu_fallback_respects_existing(monkeypatch):
    monkeypatch.setenv("MINIVLLM_DEVICE", "npu")
    apply_darwin_cpu_fallback()  # no-op on non-Darwin; must not raise
    assert os.environ["MINIVLLM_DEVICE"] == "npu"
