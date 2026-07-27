"""Metal-backend codegen module registry.

Importing this module imports every Metal-backend codegen module, so their
``@_decorators.codegen(op, "metal")`` and ``register_codegen("metal")``
handlers register onto the op / aten-lowering objects they extend.

It is imported exactly once -- by
:func:`helion._compiler.backend_registry.import_backend_codegen`, after all
language ops are defined -- so adding a new Metal codegen module only requires
listing it here, never editing the core ``helion/language`` files.
"""

from __future__ import annotations

from . import memory_ops  # noqa: F401
from . import tracing_ops  # noqa: F401
