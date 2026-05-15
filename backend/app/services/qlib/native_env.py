"""Native runtime guards for Qlib/LightGBM on local macOS development."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOMEBREW_LIBOMP_DIR = Path("/opt/homebrew/opt/libomp/lib")


def ensure_libomp_env_before_lightgbm_import() -> None:
    """Relaunch with Homebrew libomp visible before LightGBM is imported.

    On Apple Silicon, the LightGBM wheel links against ``@rpath/libomp.dylib``.
    If the process starts without Homebrew's libomp on ``DYLD_LIBRARY_PATH``, the
    native library can load an incompatible OpenMP runtime and segfault during
    ``Booster.predict``. Mutating the env after importing LightGBM is too late,
    so entrypoints should call this before importing app modules that may import
    LightGBM.
    """

    if sys.platform != "darwin" or not HOMEBREW_LIBOMP_DIR.exists():
        return

    existing = [
        part for part in os.environ.get("DYLD_LIBRARY_PATH", "").split(":") if part
    ]
    if str(HOMEBREW_LIBOMP_DIR) in existing:
        return

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = ":".join([str(HOMEBREW_LIBOMP_DIR), *existing])
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
