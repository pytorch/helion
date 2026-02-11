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
    - How code generation primitives (load, store, range, etc.) are emitted
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

    # ── Memory ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    def load_expr(
        self,
        ptr: str,
        mask: str,
        other: str | None = None,
        eviction_policy: str | None = None,
    ) -> str:
        """Generate a load expression string.

        For Triton: tl.load(ptr, mask, other=0)
        """
        ...

    @abc.abstractmethod
    def store_expr(self, ptr: str, value: str, mask: str) -> str:
        """Generate a store expression string.

        For Triton: tl.store(ptr, value, mask)
        """
        ...

    # ── Shape manipulation ──────────────────────────────────────────────

    @abc.abstractmethod
    def reshape_expr(self, value: str, shape: str) -> str:
        """Generate a reshape expression string.

        For Triton: tl.reshape(value, shape)
        """
        ...

    @abc.abstractmethod
    def broadcast_to_expr(self, value: str, shape: str) -> str:
        """Generate a broadcast_to expression string.

        For Triton: tl.broadcast_to(value, shape)
        """
        ...

    @abc.abstractmethod
    def permute_expr(self, value: str, dims: str) -> str:
        """Generate a permute expression string.

        For Triton: tl.permute(value, dims)
        """
        ...

    # ── Creation ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def zeros_expr(self, shape: str, dtype: str) -> str:
        """Generate a zeros expression string.

        For Triton: tl.zeros(shape, dtype)
        """
        ...

    @abc.abstractmethod
    def full_expr(self, shape: str, value: str, dtype: str) -> str:
        """Generate a full expression string.

        For Triton: tl.full(shape, value, dtype)
        """
        ...

    # ── Type ────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def cast_expr(self, value: str, dtype: str) -> str:
        """Generate a cast expression string.

        For Triton: tl.cast(value, dtype)
        """
        ...

    # ── Range / iteration ───────────────────────────────────────────────

    @abc.abstractmethod
    def arange_expr(self, begin: str, end: str) -> str:
        """Generate an arange expression string.

        For Triton: tl.arange(begin, end)
        """
        ...

    @abc.abstractmethod
    def range_call(self, args: list[str]) -> str:
        """Generate a range call expression string.

        For Triton: tl.range(args...)
        """
        ...

    @abc.abstractmethod
    def static_range_call(self, args: list[str]) -> str:
        """Generate a static_range call expression string.

        For Triton: tl.static_range(args...)
        """
        ...

    # ── Program IDs ─────────────────────────────────────────────────────

    @abc.abstractmethod
    def program_id_expr(self, dim: int) -> str:
        """Generate a program_id expression string.

        For Triton: tl.program_id(dim)
        """
        ...

    # ── Math ────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def cdiv_expr(self, a: str, b: str) -> str:
        """Generate a device-side ceiling division expression string.

        For Triton: tl.cdiv(a, b)
        """
        ...

    @abc.abstractmethod
    def cdiv_host_expr(self, a: str, b: str) -> str:
        """Generate a host-side ceiling division expression string.

        For Triton: triton.cdiv(a, b)
        """
        ...

    @abc.abstractmethod
    def minimum_expr(self, a: str, b: str) -> str:
        """Generate a minimum expression string.

        For Triton: tl.minimum(a, b)
        """
        ...

    @abc.abstractmethod
    def maximum_expr(self, a: str, b: str) -> str:
        """Generate a maximum expression string.

        For Triton: tl.maximum(a, b)
        """
        ...

    @abc.abstractmethod
    def next_power_of_2_expr(self, x: str) -> str:
        """Generate a next_power_of_2 expression string (host-side).

        For Triton: triton.next_power_of_2(x)
        """
        ...

    # ── Reductions ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def reduce_expr(self, value: str, op: str, dim: int) -> str:
        """Generate a reduction expression string.

        For Triton: tl.sum(value, dim), tl.max(value, dim), etc.
        """
        ...

    @abc.abstractmethod
    def dot_expr(self, lhs: str, rhs: str, kwargs: str) -> str:
        """Generate a dot product expression string.

        For Triton: tl.dot(lhs, rhs, ...)
        """
        ...

    # ── Debug ───────────────────────────────────────────────────────────

    @abc.abstractmethod
    def static_assert_expr(self, cond: str) -> str:
        """Generate a static_assert expression string.

        For Triton: tl.static_assert(cond)
        """
        ...

    # ── Kernel launch ───────────────────────────────────────────────────

    @abc.abstractmethod
    def reserved_names(self) -> list[str]:
        """Return names reserved by the backend's kernel launch protocol."""
        ...


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

    # ── Memory ──────────────────────────────────────────────────────────

    def load_expr(
        self,
        ptr: str,
        mask: str,
        other: str | None = None,
        eviction_policy: str | None = None,
    ) -> str:
        parts = [ptr, mask]
        if other is not None:
            parts.append(f"other={other}")
        if eviction_policy is not None:
            parts.append(f"eviction_policy={eviction_policy}")
        return f"tl.load({', '.join(parts)})"

    def store_expr(self, ptr: str, value: str, mask: str) -> str:
        return f"tl.store({ptr}, {value}, {mask})"

    # ── Shape manipulation ──────────────────────────────────────────────

    def reshape_expr(self, value: str, shape: str) -> str:
        return f"tl.reshape({value}, {shape})"

    def broadcast_to_expr(self, value: str, shape: str) -> str:
        return f"tl.broadcast_to({value}, {shape})"

    def permute_expr(self, value: str, dims: str) -> str:
        return f"tl.permute({value}, {dims})"

    # ── Creation ────────────────────────────────────────────────────────

    def zeros_expr(self, shape: str, dtype: str) -> str:
        return f"tl.zeros({shape}, {dtype})"

    def full_expr(self, shape: str, value: str, dtype: str) -> str:
        return f"tl.full({shape}, {value}, {dtype})"

    # ── Type ────────────────────────────────────────────────────────────

    def cast_expr(self, value: str, dtype: str) -> str:
        return f"tl.cast({value}, {dtype})"

    # ── Range / iteration ───────────────────────────────────────────────

    def arange_expr(self, begin: str, end: str) -> str:
        return f"tl.arange({begin}, {end})"

    def range_call(self, args: list[str]) -> str:
        return f"tl.range({', '.join(args)})"

    def static_range_call(self, args: list[str]) -> str:
        return f"tl.static_range({', '.join(args)})"

    # ── Program IDs ─────────────────────────────────────────────────────

    def program_id_expr(self, dim: int) -> str:
        return f"tl.program_id({dim})"

    # ── Math ────────────────────────────────────────────────────────────

    def cdiv_expr(self, a: str, b: str) -> str:
        return f"tl.cdiv({a}, {b})"

    def cdiv_host_expr(self, a: str, b: str) -> str:
        return f"triton.cdiv({a}, {b})"

    def minimum_expr(self, a: str, b: str) -> str:
        return f"tl.minimum({a}, {b})"

    def maximum_expr(self, a: str, b: str) -> str:
        return f"tl.maximum({a}, {b})"

    def next_power_of_2_expr(self, x: str) -> str:
        return f"triton.next_power_of_2({x})"

    # ── Reductions ──────────────────────────────────────────────────────

    def reduce_expr(self, value: str, op: str, dim: int) -> str:
        return f"tl.{op}({value}, {dim})"

    def dot_expr(self, lhs: str, rhs: str, kwargs: str) -> str:
        if kwargs:
            return f"tl.dot({lhs}, {rhs}, {kwargs})"
        return f"tl.dot({lhs}, {rhs})"

    # ── Debug ───────────────────────────────────────────────────────────

    def static_assert_expr(self, cond: str) -> str:
        return f"tl.static_assert({cond})"

    # ── Kernel launch ───────────────────────────────────────────────────

    def reserved_names(self) -> list[str]:
        return ["grid", "warmup", "num_warps", "num_stages"]
