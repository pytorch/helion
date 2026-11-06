"""Reference out-of-tree backend used by test/test_backend_plugins.py.

Imports Helion only through ``helion.backend_api`` and reuses Triton codegen
under a new name (the TileIRBackend pattern).
"""

from __future__ import annotations

import os

from helion import backend_api

# Lets tests see which processes (e.g. spawned autotune workers) loaded us.
_log = os.environ.get("HELION_DUMMY_PLUGIN_LOG")
if _log:
    with open(_log, "a") as f:
        f.write(f"{os.getpid()}\n")


class TritonOOTBackend(backend_api.TritonBackend):
    HELION_BACKEND_API_VERSION = 1

    @property
    def name(self) -> str:
        return "triton_oot"

    @property
    def codegen_name(self) -> str:
        return "triton"
