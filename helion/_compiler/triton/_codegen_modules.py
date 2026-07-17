"""Triton-backend codegen module registry.

Importing this module imports every Triton-backend codegen module, so their
``@_decorators.codegen(op, "triton")`` and ``register_codegen("triton")``
handlers register onto the op / aten-lowering objects they extend.

It is imported exactly once -- by
:func:`helion._compiler.backend_registry.import_backend_codegen`, after all
language ops are defined -- so adding a new Triton codegen module only requires
listing it here, never editing the core ``helion/language`` files.
"""

from __future__ import annotations

from . import aten_lowering  # noqa: F401
from . import atomic_ops  # noqa: F401
from . import barrier  # noqa: F401
from . import debug_ops  # noqa: F401
from . import device_print  # noqa: F401
from . import gelu_tanh_approx  # noqa: F401
from . import inline_asm_ops  # noqa: F401
from . import inline_triton_ops  # noqa: F401
from . import matmul_ops  # noqa: F401
from . import memory_ops  # noqa: F401
from . import quantized_ops  # noqa: F401
from . import reduce_ops  # noqa: F401
from . import scan_ops  # noqa: F401
from . import tracing_ops  # noqa: F401
from . import triton_distributed_ops  # noqa: F401
from . import view_ops  # noqa: F401
