"""Engine module for mini-vLLM.

This module contains the core engine components for managing
LLM inference, including:

Core Components:
    Sequence: Represents and manages individual generation requests
    SequenceStatus: Enumeration of possible sequence states
    BlockManager: Manages KV cache block allocation and prefix caching
    Scheduler: Schedules sequences for prefill and decode phases
    ModelRunner: Executes model inference with distributed support
    LLMEngine: Main engine orchestration and scheduling
"""

from minivllm.engine.sequence import Sequence, SequenceStatus

__all__ = [
    'Sequence',
    'SequenceStatus',
]
