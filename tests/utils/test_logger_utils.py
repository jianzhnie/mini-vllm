"""Tests for minivllm.utils.logger_utils.get_logger."""

from __future__ import annotations

import logging
import uuid

from minivllm.utils.logger_utils import get_logger


def _unique_name(prefix: str) -> str:
    return f"minivllm.test.{prefix}.{uuid.uuid4().hex}"


def test_returns_a_logging_logger():
    logger = get_logger(_unique_name("returns"))
    assert isinstance(logger, logging.Logger)


def test_same_name_returns_same_instance():
    name = _unique_name("idempotent")
    assert get_logger(name) is get_logger(name)


def test_name_is_preserved_for_fresh_logger():
    name = _unique_name("name")
    assert get_logger(name).name == name
