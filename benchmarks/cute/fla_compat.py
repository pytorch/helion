# ruff: noqa: ANN401
"""Make flash-linear-attention (FLA) importable as a linear-attention baseline.

FLA is not installed in the benchmark environment. A ``fla-core`` release
unpacked at ``$HELION_FLA_ROOT`` (default: ``<repo>/flash-linear-attention``,
gitignored like ``quack``) is put on ``sys.path``. FLA releases declare
``@triton.autotune`` keys that are not kernel arguments in a few modules
(``fla.ops.cp.chunk_delta_h`` names ``BT``); current Triton rejects those at
import time, which takes every ``fla.ops`` import down with it. The shim
drops such names, which only widens the autotune cache key of kernels the
baselines never launch.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = REPO_ROOT / "flash-linear-attention"


def fla_root() -> Path | None:
    root = Path(os.environ.get("HELION_FLA_ROOT") or _DEFAULT_ROOT)
    return root if (root / "fla").is_dir() else None


def install() -> bool:
    """Put the vendored FLA on sys.path and relax Triton's autotune-key check.

    Returns whether an FLA checkout is available. Safe to call repeatedly.
    """
    root = fla_root()
    if root is None:
        return False
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from triton.runtime import autotuner

    if not getattr(autotuner.Autotuner, "_helion_fla_key_filter", False):
        original = autotuner.Autotuner.__init__

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            # Positional signature: (fn, arg_names, configs, key, ...).
            if len(args) >= 4:
                fn, arg_names, configs, key, *rest = args
                key = [name for name in key if name in arg_names]
                args = (fn, arg_names, configs, key, *rest)
            elif "key" in kwargs:
                arg_names = args[1] if len(args) > 1 else kwargs["arg_names"]
                kwargs["key"] = [n for n in kwargs["key"] if n in arg_names]
            original(self, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]

        autotuner.Autotuner.__init__ = __init__  # pyrefly: ignore[bad-assignment]
        autotuner.Autotuner._helion_fla_key_filter = True  # pyrefly: ignore[missing-attribute]
    return True
