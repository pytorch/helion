"""Typing compatibility for CUTLASS's dynamic MLIR bindings."""

from __future__ import annotations

from typing import Any

from cutlass._mlir import ir as _ir

# CUTLASS 4.7.0 marks this module as typed, but its extension-provided exports
# do not have declarations for static type checkers.
ir: Any = _ir
