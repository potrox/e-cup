"""Organizer entry point for the frozen neural ensemble submission."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_BUNDLE_ROOT = Path(__file__).resolve().parent
_VENDOR_PATH = _BUNDLE_ROOT / "vendor"
if _VENDOR_PATH.is_dir():
    sys.path.insert(0, str(_VENDOR_PATH))

from ecup_matching.ensemble_inference import main

if __name__ == "__main__":
    main()
