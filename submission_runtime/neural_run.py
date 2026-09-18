"""Organizer entry point for the hybrid E-CUP Task 1 submission."""

# ruff: noqa: E402, I001

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_BUNDLE_ROOT = Path(__file__).resolve().parent
_VENDOR_PATH = _BUNDLE_ROOT / "vendor"
if _VENDOR_PATH.is_dir():
    sys.path.insert(0, str(_VENDOR_PATH))
    libgomp_path = _VENDOR_PATH / "lib/libgomp.so.1"
    if libgomp_path.is_file():
        ctypes.CDLL(str(libgomp_path), mode=ctypes.RTLD_GLOBAL)

from ecup_matching.hybrid_inference import main  # noqa: E402


if __name__ == "__main__":
    main()
