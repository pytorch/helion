"""FlyDSL-backend codegen module registry.

Importing this module imports every FlyDSL-backend codegen module, so their
``@_decorators.codegen(op, "flydsl")`` handlers register onto the op objects
they extend.

It is imported exactly once -- by
:func:`helion._compiler.backend_registry.import_backend_codegen`, after all
language ops are defined -- so adding a new FlyDSL codegen module only requires
listing it here, never editing the core ``helion/language`` files.
"""

from __future__ import annotations

from . import memory_ops  # noqa: F401
from . import tracing_ops  # noqa: F401
from . import view_ops  # noqa: F401
