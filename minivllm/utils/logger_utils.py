import logging
import os
import sys
from logging import Formatter, LogRecord
from pathlib import Path
from typing import Optional, Union

import torch.distributed as dist
from colorama import Fore, Style

logger_initialized: dict = {}


class ColorfulFormatter(Formatter):
    """Formatter that adds ANSI color codes to log messages based on their
    level and includes rank information for distributed training.

    Attributes:
        COLORS: Dictionary mapping log levels to their corresponding color codes

    Example:
        >>> formatter = ColorfulFormatter('%(levelname)s: %(message)s')
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(formatter)
    """

    COLORS = {
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'DEBUG': Fore.LIGHTGREEN_EX,
    }

    def format(self, record: LogRecord) -> str:
        # Add rank information to the record
        record.rank = self._get_rank()
        record.is_main = record.rank == 0

        # Format the log message
        log_message = super().format(record)

        # Add color based on log level
        return self.COLORS.get(record.levelname, '') + log_message + Fore.RESET

    def _get_rank(self) -> int:
        """Get the current process rank in a safe way."""
        try:
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass

        # Fallback to environment variables
        return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))


def get_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    file_mode: str = 'w',
    force_main_process: bool = False,
) -> logging.Logger:
    """Initialize and get a logger by name with optional file output.

    This function creates or retrieves a logger with the specified configuration.
    It handles distributed training scenarios by managing log levels across different
    process ranks and prevents duplicate logging issues with PyTorch DDP.

    Args:
        name: Logger name for identification and hierarchy
        log_file: Path to the log file. If provided, logs will also be written to this file
                 (only for rank 0 process in distributed training)
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
                  Note: Only rank 0 process uses this level; others use ERROR level
        file_mode: File opening mode ('w' for write, 'a' for append)
        force_main_process: If True, only main process (rank 0) will log regardless of log_level

    Returns:
        A configured logging.Logger instance

    Example:
        >>> logger = get_logger("my_model", "training.log", logging.DEBUG)
        >>> logger.info("Training started")
    """
    if file_mode not in ('w', 'a'):
        raise ValueError("file_mode must be either 'w' or 'a'")

    # Get or create logger instance
    logger = logging.getLogger(name)

    # Return existing logger if already initialized
    if name in logger_initialized:
        return logger

    # Check if parent logger is already initialized
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # Get current rank safely
    rank = _get_distributed_rank()
    is_main_process = rank == 0

    # Fix PyTorch DDP duplicate logging issue
    # Clear existing handlers to prevent duplicate logging
    if logger.handlers:
        logger.handlers.clear()

    # Only configure handlers for main process or if explicitly requested
    if is_main_process or not force_main_process:
        # Initialize handlers list
        handlers = []

        # Add StreamHandler for main process only
        if is_main_process:
            stream_handler = logging.StreamHandler(sys.stdout)
            handlers.append(stream_handler)

        # Add FileHandler for rank 0 process if log_file is specified
        if is_main_process and log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(str(log_file), file_mode))

        # Configure formatter with rank information
        if is_main_process:
            fmt = '%(asctime)s - [Rank %(rank)d] - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s'
        else:
            fmt = '%(asctime)s - [Rank %(rank)d] - %(name)s - %(levelname)s - %(message)s'

        formatter = ColorfulFormatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')

        # Apply configuration to all handlers
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level if is_main_process else logging.ERROR)
            logger.addHandler(handler)

    # Set logger level based on rank and configuration
    if force_main_process:
        logger.setLevel(log_level if is_main_process else logging.CRITICAL +
                        1)  # Disable logging for non-main processes
    else:
        logger.setLevel(log_level if is_main_process else logging.ERROR)

    # Mark logger as initialized
    logger_initialized[name] = True

    return logger


def _get_distributed_rank() -> int:
    """Safely get the current distributed rank.

    Returns:
        int: The current process rank (0 for main process)
    """
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass

    # Fallback to environment variables (common distributed training variables)
    rank = os.environ.get('RANK')
    if rank is not None:
        return int(rank)

    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        return int(local_rank)

    return 0  # Default to main process


def get_outdir(path: str, *paths, inc: bool = False) -> str:
    """Get the output directory. If the directory does not exist, it will be
    created. If `inc` is True, the directory will be incremented if the
    directory already exists.

    Args:
        path (str): The root root path.
        *paths: The subdirectories.
        inc (bool, optional): Whether to increment the directory. Defaults to False.

    Returns:
        str: The output directory.
    """
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        return outdir
    elif inc:
        for count in range(1, 100):
            outdir_inc = f'{outdir}-{count}'
            if not os.path.exists(outdir_inc):
                os.makedirs(outdir_inc)
                return outdir_inc
        raise RuntimeError(
            'Failed to create unique output directory after 100 attempts')
    return outdir
