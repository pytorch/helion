"""FlyDSL-backend codegen module registry.

Importing this module imports every FlyDSL-backend codegen module, so their
``@_decorators.codegen(op, "flydsl")`` handlers register onto the op objects
they extend.
"""

from __future__ import annotations

from . import memory_ops  # noqa: F401
from . import tracing_ops  # noqa: F401
from . import view_ops  # noqa: F401
