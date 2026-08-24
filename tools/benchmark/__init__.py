"""Shared benchmark primitives for the RGB and NIR tracks."""

from .paths import MODELS, NIR_RATIOS, NIR_SEED, REPO_ROOT, RGB_SEEDS
from .protocol import ProtocolError, load_protocol, validate_protocol

__all__ = [
    "MODELS",
    "NIR_RATIOS",
    "NIR_SEED",
    "REPO_ROOT",
    "RGB_SEEDS",
    "ProtocolError",
    "load_protocol",
    "validate_protocol",
]
