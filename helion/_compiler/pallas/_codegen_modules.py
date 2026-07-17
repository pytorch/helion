"""Pallas-backend codegen module registry.

Importing this module imports every Pallas-backend codegen module, so their
``@_decorators.codegen(op, "pallas")`` and ``register_codegen("pallas")``
handlers register onto the op / aten-lowering objects they extend.

It is imported exactly once -- by
:func:`helion._compiler.backend_registry.import_backend_codegen`, after all
language ops are defined -- so adding a new Pallas codegen module only requires
listing it here, never editing the core ``helion/language`` files.
"""

from __future__ import annotations

from . import aten_lowering  # noqa: F401
from . import atomic_ops  # noqa: F401
from . import creation_ops  # noqa: F401
from . import gelu_tanh_approx  # noqa: F401
from . import matmul_ops  # noqa: F401
from . import memory_ops  # noqa: F401
from . import pallas_distributed_ops  # noqa: F401
from . import tracing_ops  # noqa: F401
