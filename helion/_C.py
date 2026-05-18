"""Thin Python shim re-exporting Helion's C extension.

The C sources live under ``helion/_C/`` (see ``CMakeLists.txt``) and are
compiled into ``helion._C_ext`` so the directory and module names don't
collide on the filesystem. This shim lets callers ``import helion._C``
and use the public surface (`tensor_key`, `CompiledLauncher`) directly.

Falls back to a sentinel ``available = False`` when the extension didn't
build (e.g. PyTorch-on-MTIA, debug build), so the pure-Python
``_FastLauncher`` stays usable as the fallback.
"""

from __future__ import annotations

try:
    # pyrefly: ignore [missing-import]  # built by CMake at install time
    from ._C_ext import CompiledLauncher

    # pyrefly: ignore [missing-import]  # built by CMake at install time
    from ._C_ext import inline_cache_match

    # pyrefly: ignore [missing-import]  # built by CMake at install time
    from ._C_ext import tensor_key

    available: bool = True
except ImportError:  # pragma: no cover - exercised only when build is missing
    available = False

    def tensor_key(_tensor: object) -> None:  # type: ignore[misc]
        """Stub matching the C signature; always returns ``None``.

        ``None`` is the documented sentinel telling the caller to fall
        back to the Python implementation (used for SymInts, etc.), so
        having the stub always return it is equivalent to "C ext not
        installed".
        """
        return None

    def inline_cache_match(_cached: object, _new: object) -> bool:  # type: ignore[misc]
        """Stub matching the C signature; always returns False so the caller
        funnels through the full bind path. Safe but slower fallback."""
        return False

    class CompiledLauncher:  # type: ignore[no-redef]
        """Stub raising on construction. Used only as a type marker."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError(
                "helion._C is unavailable in this build; the C extension "
                "didn't compile. Falling back to the pure-Python "
                "_FastLauncher."
            )


__all__ = ["CompiledLauncher", "available", "inline_cache_match", "tensor_key"]
