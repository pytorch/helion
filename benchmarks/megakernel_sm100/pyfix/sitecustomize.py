"""Treat FlashInfer AOT libraries that cannot be dlopen'ed here (e.g. built for a
newer glibc) as absent, so FlashInfer's JIT fallback builds and loads its own.

``env.sh`` puts this directory on ``PYTHONPATH``; it changes nothing when every
AOT library loads.
"""

from __future__ import annotations

import ctypes

try:
    from flashinfer.jit import core as _core
except Exception:  # FlashInfer is not installed in this interpreter.
    _core = None

if _core is not None:
    for _cls in vars(_core).values():
        if isinstance(_cls, type) and isinstance(vars(_cls).get("is_aot"), property):
            _orig = vars(_cls)["is_aot"].fget

            def _is_aot(self: object, _orig: object = _orig) -> bool:
                if not _orig(self):  # pyrefly: ignore[not-callable]
                    return False
                try:
                    ctypes.CDLL(str(self.aot_path))  # pyrefly: ignore[missing-attribute]
                except OSError:
                    return False
                return True

            _cls.is_aot = property(_is_aot)  # pyrefly: ignore[missing-attribute]
