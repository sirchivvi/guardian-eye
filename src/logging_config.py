"""
GuardianEye logging configuration and device detection.

Call ``setup_logging()`` once at application startup (before importing other
project modules) to configure structured logging for the entire process.

Device detection
----------------
``get_device()`` returns a ``torch.device`` for use by all training and
inference code.  CUDA is used when available; otherwise CPU is used.
The selected device and relevant runtime metadata are logged automatically.

Usage
-----
>>> from src.logging_config import setup_logging, get_device
>>> setup_logging()
>>> device = get_device()
"""

from __future__ import annotations

import logging
import platform
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

logger = logging.getLogger(__name__)


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | None = None,
) -> None:
    """
    Configure root logger with a consistent format.

    Parameters
    ----------
    level:
        Logging level for the root logger (e.g. ``logging.DEBUG``).
    log_file:
        Optional path to a file handler.  When provided, log records are
        written to both the console and the file.

    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,  # override any existing root-logger configuration
    )


def get_device() -> torch.device:  # noqa: F821  (torch imported lazily)
    """
    Return the best available compute device and log runtime metadata.

    Checks for CUDA first; falls back to CPU.  All relevant version and
    device information is logged at INFO level so every run leaves a
    reproducible environment record.

    Returns
    -------
    torch.device
        Either ``torch.device("cuda")`` or ``torch.device("cpu")``.

    """
    import torch  # deferred so CPU-only environments can import this module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Python   : %s", sys.version.replace("\n", " "))
    logger.info("Platform : %s", platform.platform())
    logger.info("PyTorch  : %s", torch.__version__)

    if device.type == "cuda":
        logger.info("Device   : CUDA (%s)", torch.cuda.get_device_name(0))
        logger.info("CUDA ver : %s", torch.version.cuda)
        logger.info(
            "VRAM     : %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        logger.info("Device   : CPU (no CUDA available)")

    return device
