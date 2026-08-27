"""YOLOX-Nano NIR 1:2 training configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE = Path(__file__).with_name("base.py")
_SPEC = importlib.util.spec_from_file_location("nir_yolox_base", _BASE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(_BASE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class Exp(_MODULE.Exp):
    def __init__(self) -> None:
        super().__init__("1to2")
