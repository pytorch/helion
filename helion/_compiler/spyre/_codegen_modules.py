"""Spyre-backend codegen module registry.

Importing this module imports every Spyre-backend codegen module, so their
``@_decorators.codegen(op, "spyre")`` handlers register onto the op objects they
extend.
"""

from __future__ import annotations

from . import creation_ops  # noqa: F401
from . import memory_ops  # noqa: F401
from . import tracing_ops  # noqa: F401
