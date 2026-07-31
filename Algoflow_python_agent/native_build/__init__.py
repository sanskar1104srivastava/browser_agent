"""Generated native extension package.

`scripts/build_native_stt.py` builds `native_build/_local_stt.*` here.
"""

from __future__ import annotations

import os
from pathlib import Path


if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(Path(__file__).resolve().parent))
