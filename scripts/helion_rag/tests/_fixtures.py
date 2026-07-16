"""Shared test fixtures to avoid duplicating kernel source strings."""

from __future__ import annotations

import examples.add as _add_mod  # type: ignore[import-not-found]
import inspect
import sys
from pathlib import Path

# Ensure repo root (containing examples/) is on path when running pytest from package dir
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_kernel_src = inspect.getsource(_add_mod.add)
SRC = f"import helion\nimport helion.language as hl\n\n\n{_kernel_src}"

SHAPES = "[(16, 16), (16, 16)]"
DTYPES = "['torch.float16', 'torch.float16']"
FAMILY = "test-family"
OTHER_FAMILY = "other-family"

SRC_PLAIN = SRC

SRC_EPILOGUE = """
@helion.kernel
def mm(x, y, epilogue):
    return epilogue(x @ y)
"""

SETTINGS = {
    "allow_warp_specialize": True,
    "backend": "triton",
    "debug_dtype_asserts": False,
    "dot_precision": "tf32",
    "fast_math": False,
    "index_dtype": None,
    "pallas_interpret": False,
    "persistent_reserved_sms": 0,
    "static_shapes": True,
    "triton_do_not_specialize": False,
    "autotune_random_seed": 1234,
}

INGEST_SHAPES = "[(1024, 512), (512, 256)]"
INGEST_DTYPES = "['torch.float16', 'torch.bfloat16']"
RUNTIME_SHAPES = "((1024, 512), (512, 256))"
RUNTIME_DTYPES = "('torch.float16', 'torch.bfloat16')"
