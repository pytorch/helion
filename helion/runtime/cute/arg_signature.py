"""Per-argument facts of a CuTe launch and the schema / key / guard forms.

The launcher's three launch tiers (the compiled-wrapper schema, the
launch-argument cache key and the last-launch guard) derive the same facts
from the argument tuple: a tensor's dtype, device, rank, sizes, strides,
data pointer and proved pointer alignment; a scalar's kind, value and
Constexpr flag.  :class:`ArgSignature` gathers them once per launch; its
methods spell the three byte-pinned forms:

* schema entries ``("tensor", str(dtype), ndim[, sizes, strides][, alignment])``,
  ``("scalar", kind)`` and ``("scalar_constexpr", kind, cache_value, value)``
  (decoded positionally by the generated wrapper, the standalone exporter and
  the standalone launcher; ``repr`` of the schema is the on-disk compile key);
* cache-key entries ``("tensor", device.type, device.index, str(dtype), ndim,
  data_ptr, sizes, strides)`` / ``("scalar"|"scalar_constexpr", kind, cache_value)``;
* the last-launch guards :class:`_CuteLastTensorArgGuard` /
  :class:`_CuteLastScalarArgGuard`.

``test/test_cute_launcher_arg_signature.py`` pins all three against the
values the launcher produced before the tiers shared this module.  Leaf
module: imports torch and ``helion.exc`` only; ``cutlass`` lazily.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Callable


def normalize_scalar(arg: object) -> tuple[str, object]:
    """``(kind, value)`` of a launcher scalar argument, Sym* values unwrapped."""
    if isinstance(arg, (bool, torch.SymBool)):
        return ("bool", bool(arg))
    if isinstance(arg, (int, torch.SymInt)):
        return ("int", int(arg))
    if isinstance(arg, (float, torch.SymFloat)):
        return ("float", float(arg))
    raise exc.BackendUnsupported("cute", f"launcher scalar argument type: {type(arg)}")


_SCALAR_ANNOTATIONS = {
    "bool": "cutlass.Boolean",
    "int": "cutlass.Int64",
    "float": "cutlass.Float32",
}


def scalar_annotation(kind: str) -> str:
    """The wrapper parameter annotation of a runtime scalar of ``kind``."""
    return _SCALAR_ANNOTATIONS[kind]


def scalar_cache_value(scalar_kind: str, scalar_value: object) -> object:
    """A hashable, NaN- and sign-exact form of a scalar (floats by ``hex``)."""
    return cast("float", scalar_value).hex() if scalar_kind == "float" else scalar_value


def kernel_param_is_constexpr(cute_kernel: object) -> tuple[bool, ...]:
    """Return per-parameter Constexpr flags for a ``@cute.kernel``.

    Cached on the kernel object to avoid repeated signature inspection.
    The newer cutlass DSL (>=4.5) enforces region isolation: a runtime scalar
    passed through the wrapper cannot satisfy a kernel parameter declared as
    ``cutlass.Constexpr``.  When the wrapper sees a Constexpr-typed kernel
    parameter, it must propagate the value as a Constexpr (i.e., baked into
    the compiled wrapper) rather than as a runtime ``cutlass.Int64``.
    """
    cached = getattr(cast("Any", cute_kernel), "_helion_cute_param_constexpr", None)
    if cached is not None:
        return cast("tuple[bool, ...]", cached)
    import cutlass

    try:
        sig = inspect.signature(cute_kernel)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        flags: tuple[bool, ...] = ()
    else:
        from typing import get_origin
        from typing import get_type_hints

        # Helion-emitted kernels use ``from __future__ import annotations`` so
        # ``param.annotation`` is the source string. ``get_type_hints`` resolves
        # those strings against the function's globals (which include
        # ``cutlass``).
        try:
            hints = get_type_hints(cute_kernel)  # type: ignore[arg-type]
        except Exception:
            hints = {}
        flags_list: list[bool] = []
        for name, param in sig.parameters.items():
            ann = hints.get(name, param.annotation)
            is_constexpr = ann is cutlass.Constexpr or get_origin(ann) is (
                cutlass.Constexpr
            )
            flags_list.append(is_constexpr)
        flags = tuple(flags_list)
    with suppress(AttributeError, TypeError):
        cast("Any", cute_kernel)._helion_cute_param_constexpr = flags
    return flags


def validate_launcher_tensor(arg: torch.Tensor) -> None:
    if arg.device.type != "cuda":
        raise exc.BackendUnsupported("cute", "launcher requires CUDA tensors")


def tensor_layout(
    arg: torch.Tensor,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Return ``(rank, sizes, strides)`` for the cute tensor built by the wrapper.

    ``cute.make_layout`` has no rank-0 form, so a 0-d tensor (``scale[()]``)
    is marshalled as a one-element rank-1 view. Device code reads it through
    the base pointer only, so the view never changes which element is read.
    """
    if arg.ndim == 0:
        return 1, (1,), (1,)
    sizes = tuple(int(arg.size(d)) for d in range(arg.ndim))
    strides = tuple(int(arg.stride(d)) for d in range(arg.ndim))
    return arg.ndim, sizes, strides


def pointer_alignment(data_ptr: int) -> int:
    """Prove at most 16-byte alignment from the actual argument pointer."""
    return min(16, data_ptr & -data_ptr) if data_ptr else 16


def schema_with_pointer_alignment(
    entry: tuple[object, ...], alignment: int
) -> tuple[object, ...]:
    # Keep aligned schemas/cache keys unchanged. Lower-alignment pointer types
    # require distinct compiled wrappers, including in the on-disk cache.
    return entry if alignment == 16 else (*entry, alignment)


def schema_pointer_alignment(entry: tuple[object, ...]) -> int:
    """Read a tensor/wrapper tensor entry's optional alignment specialization."""
    has_alignment = len(entry) in (4, 6) if entry[0] == "tensor" else len(entry) == 7
    return cast("int", entry[-1]) if has_alignment else 16


@dataclass(frozen=True)
class _CuteLastTensorArgGuard:
    index: int
    data_ptr: int
    device_type: str
    device_index: int | None
    dtype: torch.dtype
    ndim: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def matches(self, args: tuple[object, ...]) -> bool:
        if self.index >= len(args) or not isinstance(args[self.index], torch.Tensor):
            return False
        tensor = args[self.index]
        assert isinstance(tensor, torch.Tensor)
        return (
            int(tensor.data_ptr()) == self.data_ptr
            and tensor.device.type == self.device_type
            and tensor.device.index == self.device_index
            and tensor.dtype == self.dtype
            and tensor.ndim == self.ndim
            and tensor.size() == self.shape
            and tensor.stride() == self.stride
        )


@dataclass(frozen=True)
class _CuteLastScalarArgGuard:
    index: int
    is_constexpr: bool
    scalar_kind: str
    scalar_value: object

    def matches(
        self,
        args: tuple[object, ...],
        constexpr_flags: tuple[bool, ...],
    ) -> bool:
        if self.index >= len(args):
            return False
        arg = args[self.index]
        if isinstance(arg, torch.Tensor):
            return False
        scalar_kind, scalar_value = normalize_scalar(arg)
        is_constexpr = self.index < len(constexpr_flags) and constexpr_flags[self.index]
        return (
            is_constexpr == self.is_constexpr
            and scalar_kind == self.scalar_kind
            and scalar_cache_value(scalar_kind, scalar_value) == self.scalar_value
        )


# The fast relaunch's per-tensor metadata guard: (index, device type, device
# index, dtype, sizes, strides, compiled pointer alignment).
FastpathTensorGuard = tuple[
    int, str, int | None, torch.dtype, tuple[int, ...], tuple[int, ...], int
]


@dataclass(frozen=True)
class TensorArgFact:
    """What the launch tiers know about one tensor argument."""

    index: int
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    ndim: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    data_ptr: int
    alignment: int

    @staticmethod
    def of(index: int, tensor: torch.Tensor) -> TensorArgFact:
        data_ptr = int(tensor.data_ptr())
        return TensorArgFact(
            index=index,
            dtype=tensor.dtype,
            device_type=tensor.device.type,
            device_index=tensor.device.index,
            ndim=tensor.ndim,
            shape=tuple(int(tensor.size(d)) for d in range(tensor.ndim)),
            stride=tuple(int(tensor.stride(d)) for d in range(tensor.ndim)),
            data_ptr=data_ptr,
            alignment=pointer_alignment(data_ptr),
        )

    def layout(self) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """``(rank, sizes, strides)`` of the cute tensor the wrapper builds; a
        0-d tensor is a one-element rank-1 view (see :func:`tensor_layout`)."""
        if self.ndim == 0:
            return 1, (1,), (1,)
        return self.ndim, self.shape, self.stride

    def bakes(self) -> bool:
        """Whether the layout can be baked into the schema: ``cute.make_layout``
        rejects a 0 in any shape dimension, so zero-sized tensors keep the
        runtime-shape path."""
        return all(size > 0 for size in self.layout()[1])

    def schema_entry(self, bake: bool) -> tuple[object, ...]:
        ndim, sizes, strides = self.layout()
        entry: tuple[object, ...] = (
            ("tensor", str(self.dtype), ndim, sizes, strides)
            if bake
            else ("tensor", str(self.dtype), ndim)
        )
        return schema_with_pointer_alignment(entry, self.alignment)

    def cache_key_entry(self) -> tuple[object, ...]:
        return (
            "tensor",
            self.device_type,
            self.device_index,
            str(self.dtype),
            self.ndim,
            self.data_ptr,
            self.shape,
            self.stride,
        )

    def guard(self) -> _CuteLastTensorArgGuard:
        return _CuteLastTensorArgGuard(
            index=self.index,
            data_ptr=self.data_ptr,
            device_type=self.device_type,
            device_index=self.device_index,
            dtype=self.dtype,
            ndim=self.ndim,
            shape=self.shape,
            stride=self.stride,
        )

    def fastpath_guard(self, compiled_alignment: int) -> FastpathTensorGuard:
        return (
            self.index,
            self.device_type,
            self.device_index,
            self.dtype,
            self.shape,
            self.stride,
            compiled_alignment,
        )


@dataclass(frozen=True)
class ScalarArgFact:
    """What the launch tiers know about one scalar argument."""

    index: int
    kind: str
    value: object
    is_constexpr: bool

    @property
    def cache_value(self) -> object:
        return scalar_cache_value(self.kind, self.value)

    def schema_entry(self) -> tuple[object, ...]:
        # A Constexpr value is baked into the wrapper / cache key: cutlass DSL
        # >=4.5 fails IR verification ("value defined outside the region") if
        # a runtime scalar is fed to a kernel parameter declared as
        # ``cutlass.Constexpr``.
        if self.is_constexpr:
            return ("scalar_constexpr", self.kind, self.cache_value, self.value)
        return ("scalar", self.kind)

    def cache_key_entry(self) -> tuple[object, ...]:
        return (
            "scalar_constexpr" if self.is_constexpr else "scalar",
            self.kind,
            self.cache_value,
        )

    def guard(self) -> _CuteLastScalarArgGuard:
        return _CuteLastScalarArgGuard(
            index=self.index,
            is_constexpr=self.is_constexpr,
            scalar_kind=self.kind,
            scalar_value=self.cache_value,
        )


@dataclass(frozen=True)
class ArgSignature:
    """The per-argument facts of one launch, in argument order."""

    entries: tuple[TensorArgFact | ScalarArgFact, ...]

    @staticmethod
    def build(
        args: tuple[object, ...],
        constexpr_flags: tuple[bool, ...],
        validate_tensor: Callable[[torch.Tensor], None],
    ) -> ArgSignature:
        entries: list[TensorArgFact | ScalarArgFact] = []
        for index, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                validate_tensor(arg)
                entries.append(TensorArgFact.of(index, arg))
                continue
            kind, value = normalize_scalar(arg)
            entries.append(
                ScalarArgFact(
                    index=index,
                    kind=kind,
                    value=value,
                    is_constexpr=index < len(constexpr_flags)
                    and constexpr_flags[index],
                )
            )
        return ArgSignature(tuple(entries))

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def tensors(self) -> tuple[TensorArgFact, ...]:
        return tuple(
            entry for entry in self.entries if isinstance(entry, TensorArgFact)
        )

    @property
    def scalars(self) -> tuple[ScalarArgFact, ...]:
        return tuple(
            entry for entry in self.entries if isinstance(entry, ScalarArgFact)
        )

    def cache_key_entries(self) -> list[object]:
        return [entry.cache_key_entry() for entry in self.entries]

    def guards(self) -> tuple[_CuteLastTensorArgGuard | _CuteLastScalarArgGuard, ...]:
        return tuple(entry.guard() for entry in self.entries)
