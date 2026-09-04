"""Model Manager module for handling model loading and lifecycle.

This module provides the ModelManager class which is responsible for:
- Loading pre-trained language models
- Managing model lifecycle
- Model validation and compatibility checks
- Resource allocation and cleanup
"""

from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from minivllm.config import Config
from minivllm.models.registry import create_model
from minivllm.utils.device import (
    empty_cache,
    get_current_device,
    validate_device,
)
from minivllm.utils.loader import load_model
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ["ModelManager"]


class ModelManager:
    """Manages model loading, lifecycle, and resource allocation.

    This class handles all model-related operations including loading,
    validation, and resource management. It isolates model management
    logic from the execution logic.

    Attributes:
        config: Engine configuration.
        device: Device where the model is loaded.
        model: The loaded language model.
        tokenizer: Tokenizer for text processing.
        model_type: Type of model (qwen3).
    """

    def __init__(self, config: Config) -> None:
        """Initialize the model manager.

        Args:
            config: Engine configuration with model path and settings.
        """
        self.config = config
        self.device = None
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self._model_config = None

        logger.debug("ModelManager initialized for model: %s", config.model)

    def initialize(self) -> None:
        """Initialize model manager and load the model.

        Raises:
            RuntimeError: If device validation or model loading fails.
            ValueError: If model configuration is invalid.
        """
        self._setup_device()
        self._validate_model_path()
        self._load_tokenizer()
        self._load_model()
        self._validate_model_compatibility()

        logger.info(
            f"Model manager initialized successfully. "
            f"Model: {self.model_type}, Device: {self.device}"
        )

    def _setup_device(self) -> None:
        """Setup and validate the target device."""
        try:
            self.device = get_current_device()
            validate_device(self.device)
            logger.debug("Device setup successful: %s", self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to setup device: {e}") from e

    def _validate_model_path(self) -> None:
        """Validate the model path and configuration."""
        if not self.config.model:
            raise ValueError("Model path cannot be empty")
        # Additional validation can be added here
        logger.debug("Model path validation passed: %s", self.config.model)

    def _load_tokenizer(self) -> None:
        """Load the tokenizer for the model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=True,
            )

            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.debug(
                f"Tokenizer loaded successfully: {type(self.tokenizer).__name__}"
            )
        except OSError:
            # Fall back to online mode if not cached locally
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=True,
                use_fast=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug(
                f"Tokenizer loaded successfully (online): "
                f"{type(self.tokenizer).__name__}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}") from e

    def _load_model(self) -> None:
        """Load the model based on configuration."""
        try:
            if not self.config.hf_config:
                logger.warning("HF config missing in Config, loading manually")
                self.config.hf_config = AutoConfig.from_pretrained(
                    self.config.model, trust_remote_code=True
                )

            # Attach runtime config to hf_config so layers can read it
            self.config.hf_config.use_buffered_page_attention = (
                self.config.use_buffered_page_attention
            )

            # Build the model directly in the target dtype rather than fp32 and
            # converting afterwards, so weight loading never carries a full fp32
            # copy. "auto" resolves to the checkpoint's native dtype.
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype_str = str(self.config.dtype).lower()
            if dtype_str == "auto":
                target_dtype = getattr(self.config.hf_config, "torch_dtype", torch.float16)
                if not isinstance(target_dtype, torch.dtype):
                    target_dtype = torch.float16
            elif dtype_str not in dtype_map:
                raise ValueError(
                    f"Unsupported dtype {self.config.dtype!r}; expected 'auto' "
                    f"or one of {sorted(dtype_map)}"
                )
            else:
                target_dtype = dtype_map[dtype_str]

            # Set the default dtype for construction only, then restore it
            # (torch.set_default_dtype is not a context manager in all builds).
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(target_dtype)
            try:
                self.model = create_model(self.config.hf_config)
                load_model(self.model, self.config.model)
            finally:
                torch.set_default_dtype(prev_dtype)
            self.model = self.model.to(device=self.device, dtype=target_dtype)

            self.model_type = (
                type(self.model).__name__.replace("ForCausalLM", "").lower()
            )
            self._model_config = self.model.config
            logger.info(
                "Model loaded successfully: %s (dtype=%s)",
                self.model_type,
                target_dtype,
            )

        except Exception as e:
            logger.exception("Failed to load model")
            raise RuntimeError(f"Failed to load model: {e}") from e

    def _validate_model_compatibility(self) -> None:
        """Validate model compatibility with current configuration."""
        if not self.model or not self._model_config:
            raise RuntimeError("Model not loaded properly")

        vocab_size = getattr(self._model_config, "vocab_size", 0)
        hidden_size = getattr(
            self._model_config,
            "hidden_size",
            getattr(self._model_config, "n_embd", 0),
        )
        num_heads = getattr(
            self._model_config,
            "num_attention_heads",
            getattr(self._model_config, "n_head", 0),
        )

        if vocab_size <= 0:
            raise ValueError(f"Invalid vocab_size: {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"Invalid hidden_size: {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"Invalid num_attention_heads: {num_heads}")

        logger.debug(
            f"Model compatibility validated: vocab={vocab_size}, "
            f"hidden={hidden_size}, heads={num_heads}"
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model information.
        """
        if not self.model:
            return {}

        return {
            "model_type": self.model_type,
            "device": str(self.device),
            "vocab_size": getattr(self._model_config, "vocab_size", None),
            "hidden_size": getattr(self._model_config, "hidden_size", None),
            "num_layers": getattr(self._model_config, "num_hidden_layers", None),
            "num_heads": getattr(self._model_config, "num_attention_heads", None),
            "dtype": str(self.config.dtype),
            "tensor_parallel_size": self.config.tensor_parallel_size,
        }

    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear cache for the current device (CUDA/NPU/XPU)
        empty_cache()

        logger.debug("Model manager cleanup completed")

    def __enter__(self) -> "ModelManager":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup()
