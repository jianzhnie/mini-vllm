"""Model weight loading utilities for mini-vllm.

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
from typing import Any, Callable, Optional, Tuple, Union

import torch
from safetensors import safe_open
from torch import nn

from minivllm.utils.logger_utils import get_logger

# Configure module logger
logger = get_logger(__name__)

# Type alias for packed modules mapping
PackedModulesMapping = dict[str, Tuple[str, Any]]


def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
) -> None:
    """Load weights by copying tensor data directly to a model parameter.

    This is the fallback loader when no custom weight_loader is defined on a parameter.
    It performs basic validation and copies the tensor data into the parameter's data
    while preserving gradient tracking and other PyTorch parameter properties.

    Args:
        param: The model parameter to load weights into.
        loaded_weight: The tensor containing the weights to load.

    Raises:
        TypeError: If loaded_weight is not a torch.Tensor.
        RuntimeError: If tensor shapes are incompatible between param and loaded_weight.

    Note:
        Uses `copy_()` in-place operation to avoid replacing the Parameter object itself,
        which is crucial for preserving gradient tracking and other PyTorch properties.
    """
    if not isinstance(loaded_weight, torch.Tensor):
        raise TypeError(
            f'loaded_weight must be a torch.Tensor, got {type(loaded_weight)}')

    if param.data.shape != loaded_weight.shape:
        raise RuntimeError(
            f'Shape mismatch for parameter: expected {param.data.shape}, '
            f'got {loaded_weight.shape}')

    # Use copy_() to avoid replacing the Parameter object itself
    param.data.copy_(loaded_weight)


def _find_parameter(model: nn.Module, name: str) -> Optional[nn.Parameter]:
    """Retrieve a parameter by name from the model with fallback mechanisms.

    This function supports both models that implement a `get_parameter(name)` method
    and standard PyTorch models with `named_parameters()`. It provides fallback
    mechanisms to maximize compatibility with different model architectures.

    Args:
        model: The PyTorch module to search for the parameter.
        name: The fully qualified name of the parameter to find.

    Returns:
        The nn.Parameter if found, None otherwise.

    Note:
        Attempts to use model.get_parameter() first if available, then falls back to
        searching through named_parameters() for maximum compatibility.
    """
    # Try model.get_parameter() first if available
    get_parameter = getattr(model, 'get_parameter', None)
    if callable(get_parameter):
        try:
            return get_parameter(name)
        except Exception as e:
            # Fall through to search by name
            logger.debug(
                "model.get_parameter failed for '%s': %s",
                name,
                e,
                exc_info=False,
            )

    # Fallback: search through named_parameters
    for param_name, param in model.named_parameters():
        if param_name == name:
            return param

    logger.debug("Parameter '%s' not found in model", name)
    return None


def _call_weight_loader(
    weight_loader: Callable[..., None],
    param: nn.Parameter,
    tensor: torch.Tensor,
    shard_id: Optional[Any] = None,
) -> None:
    """Call a weight loader with flexible argument signatures (2 or 3 arguments).

    This function provides flexibility for weight loaders that may require different
    argument signatures. It intelligently attempts to call the weight loader with the
    appropriate number of arguments based on both signature inspection and fallback
    mechanism to ensure robustness.

    Args:
        weight_loader: The weight loading function/callable to invoke.
        param: The model parameter to load weights into.
        tensor: The tensor containing the weights to load.
        shard_id: Optional shard identifier for packed/sharded modules.
                  Only used if weight_loader accepts 3 arguments.

    Raises:
        TypeError: If weight_loader cannot be called with either 2 or 3 arguments.

    Note:
        Attempts 3-argument form first when shard_id is provided, then falls back
        to 2-argument form. Handles TypeError gracefully by trying alternative
        signatures to ensure compatibility with various weight loader implementations.
    """
    # Try to determine the correct signature
    try:
        sig = inspect.signature(weight_loader)
        num_params = len(sig.parameters)
        accepts_three_args = num_params >= 3
    except (ValueError, TypeError):
        # If signature inspection fails, attempt both forms
        accepts_three_args = None

    # Attempt to call with appropriate argument count
    if accepts_three_args is None or accepts_three_args:
        try:
            if shard_id is not None:
                weight_loader(param, tensor, shard_id)
                return
        except TypeError:
            pass

    # Fall back to 2-argument call
    weight_loader(param, tensor)


def load_model(model: nn.Module, path: Union[str, Path]) -> None:
    """Load model weights from safetensors files in the specified directory.

    This function loads weights from all .safetensors files in the given directory,
    handling both regular parameters and packed modules with custom weight loaders.
    The model may optionally expose a `packed_modules_mapping` attribute that maps
    weight-name patterns to (replacement_prefix, shard_id) tuples for handling
    packed/sharded modules.

    The loading process:
        1. Validates the input path exists and is a directory
        2. Discovers all .safetensors files in the directory
        3. For each weight in each file:
            - Checks if packed module mapping patterns match
            - Finds the corresponding model parameter
            - Applies custom or default weight loader
        4. Logs detailed statistics about the loading process

    Args:
        model: The PyTorch module to load weights into. Must support parameter lookup
               via named_parameters() or optional get_parameter() method.
        path: Directory path containing .safetensors files. Can be a string or Path object.

    Raises:
        FileNotFoundError: If the specified path doesn't exist.
        ValueError: If path is not a directory or no safetensors files are found.
        RuntimeError: If weight loading fails for any safetensors file.

    Example:
        >>> model = MyModel()
        >>> load_model(model, "./model_weights")

    Note:
        - The function is defensive: missing parameters trigger warnings but don't stop loading
        - File-level errors are raised as RuntimeError to ensure user awareness
        - Supports flexible weight loader signatures (2 or 3 arguments)
        - Logs comprehensive statistics about loaded vs. total weights
    """
    # Validate and prepare input path
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

    logger.info(
        'Found %d safetensors file(s) in %s',
        len(safetensor_files),
        base_path,
    )

    # Get packed modules mapping from model if available
    packed_modules_mapping: PackedModulesMapping = (getattr(
        model, 'packed_modules_mapping', None) or {})
    if packed_modules_mapping:
        logger.info(
            'Using packed modules mapping with %d pattern(s)',
            len(packed_modules_mapping),
        )

    # Initialize tracking variables
    total_weights: int = 0
    loaded_weights: int = 0
    missing_params: int = 0

    # Process each safetensors file
    for file_path in safetensor_files:
        logger.info('Loading weights from "%s"', file_path.name)
        try:
            # Open safetensors file and iterate through weights
            with safe_open(str(file_path), framework='pt', device='cpu') as f:
                for weight_name in f.keys():
                    total_weights += 1
                    tensor = f.get_tensor(weight_name)

                    # Attempt to load weight using packed modules mapping
                    loaded = _load_with_packed_mapping(
                        model,
                        weight_name,
                        tensor,
                        packed_modules_mapping,
                    )

                    if loaded:
                        loaded_weights += 1
                        continue

                    # No packed mapping matched; load directly by weight name
                    loaded = _load_weight_direct(model, weight_name, tensor)
                    if loaded:
                        loaded_weights += 1
                    else:
                        missing_params += 1

        except Exception as exc:
            logger.exception(
                'Failed to load weights from "%s"',
                file_path,
            )
            raise RuntimeError(
                f'Failed to load weights from "{file_path}": {exc}') from exc

    # Log summary statistics
    _log_loading_summary(total_weights, loaded_weights, missing_params,
                         len(safetensor_files))


def _load_with_packed_mapping(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
    packed_modules_mapping: PackedModulesMapping,
) -> bool:
    """Attempt to load weight using packed modules mapping patterns.

    Checks if the weight name matches any pattern in the packed modules mapping,
    and if so, applies the corresponding parameter name transformation and loads
    the weight using the appropriate loader.

    Args:
        model: The PyTorch module containing parameters.
        weight_name: The name of the weight being loaded.
        tensor: The tensor containing weight data.
        packed_modules_mapping: Mapping of patterns to (replacement_prefix, shard_id).

    Returns:
        True if a pattern matched and weight was loaded, False otherwise.
    """
    for pattern, (replacement_prefix,
                  shard_id) in packed_modules_mapping.items():
        if pattern not in weight_name:
            continue

        # Pattern matched; transform weight name to parameter name
        param_name = weight_name.replace(pattern, replacement_prefix)
        param = _find_parameter(model, param_name)

        if param is None:
            logger.warning(
                "Parameter '%s' not found for packed weight '%s' (pattern: '%s')",
                param_name,
                weight_name,
                pattern,
            )
            return True  # Skip this weight but mark as processed

        # Load weight using appropriate loader (custom or default)
        weight_loader = getattr(param, 'weight_loader', default_weight_loader)
        logger.debug(
            "Loading packed parameter '%s' (shard_id: %s)",
            param_name,
            shard_id,
        )
        _call_weight_loader(weight_loader, param, tensor, shard_id)

        return True

    return False


def _load_weight_direct(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
) -> bool:
    """Load weight directly by matching parameter name.

    This function attempts to load a weight without any packed module mapping,
    using direct parameter name matching instead.

    Args:
        model: The PyTorch module containing parameters.
        weight_name: The name of the weight/parameter to load.
        tensor: The tensor containing weight data.

    Returns:
        True if weight was successfully loaded, False if parameter not found.
    """
    param = _find_parameter(model, weight_name)
    if param is None:
        logger.warning(
            "Parameter '%s' not found in model; skipping weight",
            weight_name,
        )
        return False

    # Get weight loader with fallback to default
    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
    logger.debug(
        "Loading weight '%s'",
        weight_name,
    )
    _call_weight_loader(weight_loader, param, tensor)
    return True


def _log_loading_summary(
    total_weights: int,
    loaded_weights: int,
    missing_params: int,
    num_files: int,
) -> None:
    """Log a summary of the weight loading operation.

    Provides comprehensive statistics about the loading process, including
    warnings for missing parameters.

    Args:
        total_weights: Total number of weights attempted to load.
        loaded_weights: Number of weights successfully loaded.
        missing_params: Number of parameters not found in model.
        num_files: Number of safetensors files processed.
    """
    if missing_params > 0:
        logger.warning(
            'Weight loading completed with %d missing parameter(s) out of %d total weights',
            missing_params,
            total_weights,
        )
    else:
        logger.info('All weights loaded successfully')

    logger.info(
        'Loaded %d/%d weights from %d file(s)',
        loaded_weights,
        total_weights,
        num_files,
    )
