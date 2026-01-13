"""
Model weight loading utilities for mini-vllm.

This module provides robust functionality for loading model weights from safetensors files,
supporting both regular parameters and packed modules with custom weight loaders.

Key Features:
- Loads weights from safetensors files with proper error handling
- Supports packed modules through optional `packed_modules_mapping` attribute
- Handles custom weight loaders with flexible argument signatures
- Provides comprehensive logging and validation
- Defensive programming with fallback mechanisms

Example:
    >>> from minivllm.utils.loader import load_model
    >>> model = MyModel()
    >>> load_model(model, "./model_weights")
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from safetensors import safe_open
from torch import nn

from minivllm.utils.logger_utils import get_logger

# Configure module logger
logger = get_logger(__name__)


def default_weight_loader(param: nn.Parameter,
                          loaded_weight: torch.Tensor) -> None:
    """
    Default weight loader that copies tensor data directly to parameter.

    This is the fallback loader when no custom weight_loader is defined on a parameter.
    It performs basic validation and copies the tensor data into the parameter's data.

    Args:
        param: The model parameter to load weights into.
        loaded_weight: The tensor containing the weights to load.

    Raises:
        TypeError: If loaded_weight is not a torch.Tensor.
        RuntimeError: If tensor shapes are incompatible.

    Note:
        Uses `copy_` to avoid replacing the Parameter object itself, preserving
        gradient tracking and other PyTorch parameter properties.
    """
    if not isinstance(loaded_weight, torch.Tensor):
        raise TypeError(
            f'loaded_weight must be a torch.Tensor, got {type(loaded_weight)}')

    if param.data.shape != loaded_weight.shape:
        raise RuntimeError(
            f'Shape mismatch for parameter: expected {param.data.shape}, '
            f'got {loaded_weight.shape}')

    # Use copy_ to avoid replacing the Parameter object itself
    param.data.copy_(loaded_weight)


def _find_parameter(model: nn.Module, name: str) -> Optional[nn.Parameter]:
    """
    Try to retrieve a parameter by name from the model.

    This function supports both models that implement `get_parameter(name)` method
    and standard PyTorch models with `named_parameters()`. It provides fallback
    mechanisms to maximize compatibility.

    Args:
        model: The model to search for the parameter.
        name: The name of the parameter to find.

    Returns:
        The parameter if found, None otherwise.

    Note:
        First tries model.get_parameter() if available, then falls back to
        searching through named_parameters().
    """
    # Try model.get_parameter() first if available
    get_parameter = getattr(model, 'get_parameter', None)
    if callable(get_parameter):
        try:
            return get_parameter(name)
        except Exception:
            # Fall through to search by name
            logger.debug("model.get_parameter failed for '%s'",
                         name,
                         exc_info=True)

    # Fallback: look through named_parameters
    for param_name, param in model.named_parameters():
        if param_name == name:
            return param

    logger.debug("Parameter '%s' not found in model", name)
    return None


def _call_weight_loader(weight_loader: Callable[..., None],
                        param: nn.Parameter,
                        tensor: torch.Tensor,
                        shard_id: Optional[Any] = None) -> None:
    """
    Call a weight_loader that may accept 2 or 3 arguments.

    This function provides flexibility for weight loaders that may require different
    argument signatures. It attempts to call the weight loader with the appropriate
    arguments based on inspection of the function signature.

    Args:
        weight_loader: The weight loading function to call.
        param: The model parameter to load weights into.
        tensor: The tensor containing the weights to load.
        shard_id: Optional shard identifier for packed modules.

    Note:
        Tries the 3-argument form first when shard_id is provided, then falls back
        to 2-argument form. Handles TypeError gracefully by attempting both forms.
    """
    try:
        # Prefer the 3-arg call when shard_id is provided
        if shard_id is not None:
            sig = inspect.signature(weight_loader)
            if len(sig.parameters) >= 3:
                weight_loader(param, tensor, shard_id)
                return

        # Fallback to 2-arg call
        weight_loader(param, tensor)

    except TypeError:
        # If signature inspection was wrong, attempt direct calls gracefully
        logger.debug(
            'Initial weight_loader call failed, trying alternative signatures')
        try:
            weight_loader(param, tensor, shard_id)  # type: ignore[arg-type]
        except TypeError:
            weight_loader(param, tensor)


def load_model(model: nn.Module, path: Union[str, Path]) -> None:
    """
    Load model weights from safetensors files in the specified directory.

    This function loads weights from all .safetensors files in the given directory,
    handling both regular parameters and packed modules with custom weight loaders.
    The model may expose an optional `packed_modules_mapping` attribute that maps
    weight-name patterns to (replacement_prefix, shard_id) tuples for handling
    packed/sharded modules.

    Args:
        model: The PyTorch module to load weights into. Must support parameter lookup.
        path: Directory path containing .safetensors files.

    Raises:
        FileNotFoundError: If the specified path doesn't exist.
        ValueError: If path is not a directory or no safetensors files are found.
        RuntimeError: If weight loading fails for any safetensors file.

    Example:
        >>> model = MyModel()
        >>> load_model(model, "./model_weights")

    Note:
        The function is defensive and logs warnings for missing parameters,
        but will continue loading other weights. File-level errors are raised
        as RuntimeError to ensure the user is aware of loading failures.
    """
    # Validate input path
    base_path = Path(path)
    if not base_path.exists():
        raise FileNotFoundError(
            f'Model weight directory not found: {base_path}')

    if not base_path.is_dir():
        raise ValueError(f'Path must be a directory: {base_path}')

    # Find all safetensors files
    safetensor_files = sorted(base_path.glob('*.safetensors'))
    if not safetensor_files:
        raise ValueError(
            f'No .safetensors files found in directory: {base_path}')

    logger.info('Found %d safetensors files in %s', len(safetensor_files),
                base_path)

    # Get packed modules mapping from model if available
    packed_modules_mapping: Dict[str, Tuple[str, Any]] = getattr(
        model, 'packed_modules_mapping', {}) or {}
    if packed_modules_mapping:
        logger.info('Using packed modules mapping with %d patterns',
                    len(packed_modules_mapping))

    total_weights = 0
    loaded_weights = 0
    missing_params = 0

    # Process each safetensors file
    for file_path in safetensor_files:
        logger.info('Loading weights from %s', file_path.name)
        try:
            with safe_open(str(file_path), framework='pt', device='cpu') as f:
                for weight_name in f.keys():
                    total_weights += 1
                    tensor = f.get_tensor(weight_name)

                    # First, check if this weight name should be mapped using packed mapping
                    matched = False
                    for pattern, mapping in packed_modules_mapping.items():
                        if pattern in weight_name:
                            replacement_prefix, shard_id = mapping
                            param_name = weight_name.replace(
                                pattern, replacement_prefix)
                            param = _find_parameter(model, param_name)

                            if param is None:
                                logger.warning(
                                    "Parameter '%s' not found for packed weight '%s' (pattern: %s)",
                                    param_name, weight_name, pattern)
                                missing_params += 1
                                matched = True
                                break

                            weight_loader = getattr(param, 'weight_loader',
                                                    None)
                            if weight_loader is None:
                                logger.debug(
                                    "Using default loader for packed parameter '%s' (shard_id: %s)",
                                    param_name, shard_id)
                                default_weight_loader(param, tensor)
                            else:
                                logger.debug(
                                    "Using custom weight_loader for packed parameter '%s' (shard_id: %s)",
                                    param_name, shard_id)
                                _call_weight_loader(weight_loader, param,
                                                    tensor, shard_id)

                            loaded_weights += 1
                            matched = True
                            break

                    if matched:
                        continue

                    # No packed mapping matched; load directly by weight name
                    param = _find_parameter(model, weight_name)
                    if param is None:
                        logger.warning(
                            "Parameter '%s' not found in model; skipping weight",
                            weight_name)
                        missing_params += 1
                        continue

                    weight_loader = getattr(param, 'weight_loader',
                                            default_weight_loader)
                    logger.debug(
                        "Loading weight '%s' with %s", weight_name,
                        'custom weight_loader' if hasattr(
                            param, 'weight_loader') else 'default loader')
                    _call_weight_loader(weight_loader, param, tensor)
                    loaded_weights += 1

        except Exception as exc:
            logger.exception('Failed to load weights from %s: %s', file_path,
                             exc)
            raise RuntimeError(
                f'Failed to load weights from {file_path}: {exc}') from exc

    # Summary logging
    if missing_params > 0:
        logger.warning(
            'Weight loading completed with %d missing parameters out of %d total weights',
            missing_params, total_weights)

    logger.info('Successfully loaded %d/%d weights from %d files',
                loaded_weights, total_weights, len(safetensor_files))
