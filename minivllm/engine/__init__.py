"""Engine module for mini-vLLM.

This module contains the core engine components for managing
LLM inference, including:

Core Components:
    Sequence: Represents and manages individual generation requests
    SequenceStatus: Enumeration of possible sequence states
    BlockManager: Manages KV cache block allocation and prefix caching
    Scheduler: Schedules sequences for prefill and decode phases
    InferenceExecutor: Handles model inference execution with optimizations
    DistributedManager: Manages distributed inference coordination
    ModelRunner: Main engine orchestration (optimized version from model_runner_opt)
    LLMEngine: High-level engine interface
"""

from minivllm.engine.block_manager import BlockManager
from minivllm.engine.distributed_manager import DistributedManager
from minivllm.engine.inference_executor import InferenceExecutor
from minivllm.engine.llm_engine import LLMEngine
from minivllm.engine.model_runner import ModelRunner
from minivllm.engine.scheduler import Scheduler
from minivllm.engine.sequence import Sequence, SequenceStatus

__all__ = [
    'Sequence',
    'SequenceStatus',
    'BlockManager',
    'Scheduler',
    'InferenceExecutor',
    'DistributedManager',
    'ModelRunner',
    'LLMEngine',
]
