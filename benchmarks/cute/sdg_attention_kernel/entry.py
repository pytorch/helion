"""Exported Python 3.12 entry; isolate bytecode before any bundle imports."""

from __future__ import annotations

import atexit
from pathlib import Path
import sys
import tempfile

# Retain this process-wide prefix for lazy imports too. -B alone does not stop
# Python from reading a preexisting timestamp-based bytecode cache.
_bytecode_cache = tempfile.TemporaryDirectory(prefix="helion-sdg-pycache-")
sys.pycache_prefix = _bytecode_cache.name
atexit.register(_bytecode_cache.cleanup)

from .bootstrap import load_bundle  # noqa: E402

assert __package__ is not None
_implementation = load_bundle(Path(__file__).resolve().parent, __package__)
run = _implementation.run
