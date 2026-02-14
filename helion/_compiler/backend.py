from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Backend(abc.ABC):
    """Abstract base class for Helion code generation backends.

    Each backend is responsible for defining:
    - How types are represented in generated code
    - What imports are needed in generated code
    - What decorators and annotations are used on generated functions
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Backend name used for codegen dispatch (e.g., 'triton')."""
        ...

    @abc.abstractmethod
    def dtype_str(self, dtype: torch.dtype) -> str:
        """Convert a torch dtype to a backend-specific type string.

        For example, Triton returns 'tl.float32' for torch.float32.
        """
        ...

    @abc.abstractmethod
    def acc_type(self, dtype: torch.dtype) -> str:
        """Get the accumulator type string for reductions.

        Some backends may promote certain types for numerical stability
        during reductions (e.g., fp16 -> fp32).
        """
        ...

    def index_type_str(self, index_dtype: torch.dtype) -> str:
        """Get the index type string for the given dtype.

        Defaults to dtype_str, but backends may override for special handling.
        """
        return self.dtype_str(index_dtype)

    @property
    @abc.abstractmethod
    def function_decorator(self) -> str:
        """Expression string for the kernel function decorator.

        For example, Triton returns 'triton.jit'.
        """
        ...

    @property
    @abc.abstractmethod
    def constexpr_type(self) -> str:
        """Type annotation string for compile-time constant arguments.

        For example, Triton returns 'tl.constexpr'.
        """
        ...

    @property
    @abc.abstractmethod
    def library_imports(self) -> dict[str, str]:
        """Mapping of short names to import statements for generated code.

        Keys are the short names used in generated code (e.g., 'tl'),
        values are the corresponding import statements.
        """
        ...

    @property
    def default_launcher_name(self) -> str:
        """Name of the default launcher variable in generated code.

        This should match a key in library_imports that imports the launcher.
        """
        return "_default_launcher"


class TritonBackend(Backend):
    """Triton code generation backend."""

    @property
    def name(self) -> str:
        return "triton"

    def dtype_str(self, dtype: torch.dtype) -> str:
        from torch._inductor.utils import triton_type

        return triton_type(dtype)

    def acc_type(self, dtype: torch.dtype) -> str:
        from torch._inductor.codegen.triton import triton_acc_type

        return triton_acc_type(dtype)

    @property
    def function_decorator(self) -> str:
        return "triton.jit"

    @property
    def constexpr_type(self) -> str:
        return "tl.constexpr"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "triton": "import triton",
            "tl": "import triton.language as tl",
            "triton_helpers": "from torch._inductor.runtime import triton_helpers",
            "tl_math": "from torch._inductor.runtime.triton_helpers import math as tl_math",
            "libdevice": "from torch._inductor.runtime.triton_compat import libdevice",
            "_default_launcher": "from helion.runtime import default_launcher as _default_launcher",
        }


# Mapping from torch dtype to JAX dtype string (e.g., "jnp.float32")
_TORCH_TO_JAX_DTYPE: dict[str, str] = {
    "torch.float16": "jnp.float16",
    "torch.float32": "jnp.float32",
    "torch.float64": "jnp.float64",
    "torch.bfloat16": "jnp.bfloat16",
    "torch.int8": "jnp.int8",
    "torch.int16": "jnp.int16",
    "torch.int32": "jnp.int32",
    "torch.int64": "jnp.int64",
    "torch.uint8": "jnp.uint8",
    "torch.bool": "jnp.bool_",
    "torch.complex64": "jnp.complex64",
    "torch.complex128": "jnp.complex128",
}


class PallasBackend(Backend):
    """Pallas (JAX) code generation backend."""

    @property
    def name(self) -> str:
        return "pallas"

    @property
    def default_launcher_name(self) -> str:
        return "_default_pallas_launcher"

    def dtype_str(self, dtype: torch.dtype) -> str:
        key = str(dtype)
        if key not in _TORCH_TO_JAX_DTYPE:
            raise ValueError(f"Unsupported dtype for Pallas backend: {dtype}")
        return _TORCH_TO_JAX_DTYPE[key]

    def acc_type(self, dtype: torch.dtype) -> str:
        import torch as _torch

        # Promote half-precision types to float32 for numerical stability
        if dtype in (_torch.float16, _torch.bfloat16):
            return "jnp.float32"
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        return ""

    @property
    def constexpr_type(self) -> str:
        return "int"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "jax": "import jax",
            "jnp": "import jax.numpy as jnp",
            "pl": "from jax.experimental import pallas as pl",
            "lax": "import jax.lax as lax",
            "_default_pallas_launcher": "from helion.runtime import default_pallas_launcher as _default_pallas_launcher",
        }


# Mapping from torch dtype to CUTLASS dtype string
_TORCH_TO_CUTLASS_DTYPE: dict[str, str] = {
    "torch.float16": "cutlass.Float16",
    "torch.float32": "cutlass.Float32",
    "torch.float64": "cutlass.Float64",
    "torch.bfloat16": "cutlass.BFloat16",
    "torch.int8": "cutlass.Int8",
    "torch.int16": "cutlass.Int16",
    "torch.int32": "cutlass.Int32",
    "torch.int64": "cutlass.Int64",
    "torch.uint8": "cutlass.UInt8",
    "torch.bool": "cutlass.Boolean",
    "torch.float8_e4m3fn": "cutlass.Float8E4M3",
    "torch.float8_e5m2": "cutlass.Float8E5M2",
}


class CuteDSLBackend(Backend):
    """CuteDSL (NVIDIA CUTLASS Python DSL) code generation backend."""

    def __init__(self) -> None:
        super().__init__()
        # Import to register CuteDSL codegen functions
        from . import cutedsl_codegen as _  # noqa: F401

    @property
    def name(self) -> str:
        return "cutedsl"

    @property
    def default_launcher_name(self) -> str:
        return "_default_cutedsl_launcher"

    def dtype_str(self, dtype: torch.dtype) -> str:
        key = str(dtype)
        if key not in _TORCH_TO_CUTLASS_DTYPE:
            raise ValueError(f"Unsupported dtype for CuteDSL backend: {dtype}")
        return _TORCH_TO_CUTLASS_DTYPE[key]

    def acc_type(self, dtype: torch.dtype) -> str:
        import torch as _torch

        # Promote half-precision and fp8 types to float32 for numerical stability
        if dtype in (
            _torch.float16,
            _torch.bfloat16,
            _torch.float8_e4m3fn,
            _torch.float8_e5m2,
        ):
            return "cutlass.Float32"
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        return "cute.kernel"

    @property
    def constexpr_type(self) -> str:
        return "cutlass.Constexpr[int]"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "cutlass": "import cutlass",
            "cute": "import cutlass.cute as cute",
            "cutlass_bindings": "from cutlass.cute.runtime import from_dlpack",
            "_default_cutedsl_launcher": "from helion.runtime.cutedsl_launcher import default_cutedsl_launcher as _default_cutedsl_launcher",
        }
