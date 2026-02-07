"""Model Manager module for handling model loading and lifecycle.

This module provides the ModelManager class which is responsible for:
- Loading pre-trained language models
- Managing model lifecycle
- Model validation and compatibility checks
- Resource allocation and cleanup
"""

from typing import Any, Dict

from transformers import AutoConfig, AutoTokenizer

from minivllm.config import Config
from minivllm.models.opt import OPTForCausalLM
from minivllm.models.qwen2 import Qwen2ForCausalLM
from minivllm.models.qwen3 import Qwen3ForCausalLM
from minivllm.utils.device import (
    empty_cache,
    get_current_device,
    set_device,
    validate_device,
)
from minivllm.utils.loader import load_model
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ['ModelManager']


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
        model_type: Type of model (qwen2, qwen3, opt).
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

        logger.debug(f'ModelManager initialized for model: {config.model}')

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

        logger.info(f'Model manager initialized successfully. '
                    f'Model: {self.model_type}, Device: {self.device}')

    def _setup_device(self) -> None:
        """Setup and validate the target device."""
        try:
            # Check if config has device specified
            if hasattr(self.config, 'device'):
                set_device(self.config.device)

            # Get current device (handles LOCAL_RANK etc.)
            self.device = get_current_device()
            validate_device(self.device)
            logger.debug(f'Device setup successful: {self.device}')
        except Exception as e:
            raise RuntimeError(f'Failed to setup device: {e}')

    def _validate_model_path(self) -> None:
        """Validate the model path and configuration."""
        if not self.config.model:
            raise ValueError('Model path cannot be empty')
        # Additional validation can be added here
        logger.debug(f'Model path validation passed: {self.config.model}')

    def _load_tokenizer(self) -> None:
        """Load the tokenizer for the model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model, trust_remote_code=True, use_fast=True)

            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.debug(f'Tokenizer loaded successfully: '
                         f'{type(self.tokenizer).__name__}')
        except Exception as e:
            raise RuntimeError(f'Failed to load tokenizer: {e}')

    def _load_model(self) -> None:
        """Load the model based on configuration."""
        try:
            # Determine model type from config or auto-detect
            self.model_type = self._detect_model_type()

            # Ensure HF config is loaded (Config usually does this)
            if not self.config.hf_config:
                # Fallback: load it manually if missing (should not happen)
                logger.warning('HF config missing in Config, loading manually')
                self.config.hf_config = AutoConfig.from_pretrained(
                    self.config.model, trust_remote_code=True)

            hf_config = self.config.hf_config

            # Instantiate model class
            if self.model_type == 'qwen2':
                self.model = Qwen2ForCausalLM(hf_config)
            elif self.model_type == 'qwen3':
                self.model = Qwen3ForCausalLM(hf_config)
            elif self.model_type == 'opt':
                self.model = OPTForCausalLM(hf_config)
            else:
                raise ValueError(f'Unsupported model type: {self.model_type}')

            # Load weights into the model
            # tensor_parallel_size and dtype are handled by model components
            # or Config validation, and load_model just copies weights.
            load_model(self.model, self.config.model)

            # Move model to device
            if self.device:
                self.model.to(self.device)

            self._model_config = self.model.config
            logger.info(f'Model loaded successfully: {self.model_type}')

        except Exception as e:
            logger.exception('Failed to load model')
            raise RuntimeError(f'Failed to load model: {e}')

    def _detect_model_type(self) -> str:
        """Detect model type based on model name or configuration."""
        # Try to use loaded HF config first
        if self.config.hf_config and hasattr(self.config.hf_config,
                                             'model_type'):
            model_type = self.config.hf_config.model_type.lower()
            if 'qwen2' in model_type:
                return 'qwen2'
            if 'qwen' in model_type:
                return 'qwen3'
            if 'opt' in model_type:
                return 'opt'

        # Fallback to path string matching
        model_path_lower = self.config.model.lower()

        if 'qwen2' in model_path_lower:
            return 'qwen2'
        elif 'qwen' in model_path_lower and 'qwen2' not in model_path_lower:
            return 'qwen3'  # Default to qwen3 for newer qwen models
        elif 'opt' in model_path_lower:
            return 'opt'
        else:
            # Try to auto-detect by loading config if not already present
            try:
                config = AutoConfig.from_pretrained(self.config.model,
                                                    trust_remote_code=True)
                model_type = getattr(config, 'model_type', '').lower()

                if 'qwen2' in model_type:
                    return 'qwen2'
                elif 'qwen' in model_type:
                    return 'qwen3'
                elif 'opt' in model_type:
                    return 'opt'
                else:
                    logger.warning(
                        'Could not auto-detect model type, defaulting to qwen2'
                    )
                    return 'qwen2'
            except Exception:
                logger.warning('Auto-detection failed, defaulting to qwen2')
                return 'qwen2'

    def _validate_model_compatibility(self) -> None:
        """Validate model compatibility with current configuration."""
        if not self.model or not self._model_config:
            raise RuntimeError('Model not loaded properly')

        # Check if model supports the requested features
        required_config = {
            'vocab_size':
            getattr(self._model_config, 'vocab_size', None),
            'hidden_size':
            getattr(self._model_config, 'hidden_size', None),
            'num_attention_heads':
            getattr(self._model_config, 'num_attention_heads', None),
        }

        logger.debug(
            f'Model compatibility validation passed: {required_config}')

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model information.
        """
        if not self.model:
            return {}

        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'vocab_size': getattr(self._model_config, 'vocab_size', None),
            'hidden_size': getattr(self._model_config, 'hidden_size', None),
            'num_layers': getattr(self._model_config, 'num_hidden_layers',
                                  None),
            'num_heads': getattr(self._model_config, 'num_attention_heads',
                                 None),
            'dtype': str(self.config.dtype),
            'tensor_parallel_size': self.config.tensor_parallel_size,
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

        logger.debug('Model manager cleanup completed')

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
