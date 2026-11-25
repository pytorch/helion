"""Metal launches as data: a kernel's MSL, grid and arguments, for other runtimes.

:func:`capture_metal_launches` runs a Helion kernel's host code with a
launcher that records each Metal launch instead of compiling and
dispatching it, so another runtime can run the kernels Helion generates.
:meth:`MetalLaunch.to_mlx` arranges one launch for MLX's
``mx.fast.metal_kernel``.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import torch
from torch.utils._cpp_embed_headers import _embed_headers

from ... import exc
from .metal_launcher import DEFAULT_BLOCK_DIMS
from .metal_launcher import check_launcher_kwargs

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ...runtime.kernel import Kernel
    from .metal_jit import MSLKernel
    from .metal_jit import _MetalKernel


@dataclasses.dataclass(frozen=True, eq=False)
class MetalArgument:
    """One argument of a launch, in the order the kernel takes them.

    ``binding`` says how the kernel receives it: ``"buffer"``, a tensor bound
    as a device buffer; ``"scalar"``, an int or float bound by value;
    ``"baked"``, a host value baked into the source, which is not bound.
    Buffers and scalars take the ``[[buffer(i)]]`` slots in order.  For a
    buffer, ``written`` says the kernel stores to it, ``read`` that it uses
    it otherwise (a load, or an operand view), and ``shares_storage`` that
    another argument views the same memory.
    """

    name: str
    value: object
    binding: Literal["buffer", "scalar", "baked"]
    written: bool = False
    read: bool = False
    shares_storage: bool = False


@dataclasses.dataclass(frozen=True, eq=False)
class MLXKernel:
    """A launch arranged for MLX's ``mx.fast.metal_kernel``::

        fn = mx.fast.metal_kernel(
            name=k.name,
            input_names=list(k.input_names),
            output_names=list(k.output_names),
            source=k.source,
            header=k.header,
        )
        outs = fn(
            # mx.array(t.cpu().numpy()); bfloat16 through an int16 view
            inputs=[to_mx(t) for t in k.inputs],
            grid=k.grid,
            threadgroup=k.threadgroup,
            output_shapes=list(k.output_shapes),
            output_dtypes=[...],  # k.output_dtypes as mx dtypes
            init_value=k.init_value,
        )

    The buffers the kernel reads are inputs and the ones it writes are
    outputs, row-contiguous, as MLX passes and allocates them, under their
    own names but for one MLX declares itself (a position attribute such as
    ``threadgroup_position_in_grid``, or another input's ``_shape``,
    ``_strides`` or ``_ndim``), which takes a ``_`` suffix.  MLX's outputs
    are fresh arrays, so what the host code put in them before the launch
    (``torch.zeros``, say, for a kernel that writes only part of its output)
    reaches the kernel only as ``init_value``, one value for every output,
    which the caller of :meth:`MetalLaunch.to_mlx` states.  An input may be
    an array of another dtype with the same bytes (MLX's ``uint32``
    quantized weights for a kernel that reads ``int16``): the source
    reinterprets each buffer as the kernel's own element type.
    ``grid`` counts threads, as MLX does; runtime scalars are baked into the
    header.  MLX compiles the kernel itself, with its own options: a math
    function can round differently in the last bit (c10's erf does for some
    inputs).  MLX writes the kernel's signature, so the launch's
    ``[[required_threads_per_threadgroup]]`` is dropped: without it, Metal
    may compile a register-heavy kernel for fewer threads per threadgroup
    than the launch's ``threadgroup``, a launch MLX may then refuse.
    """

    name: str
    header: str
    source: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    inputs: tuple[torch.Tensor, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[torch.dtype, ...]
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    init_value: float | None


@dataclasses.dataclass(frozen=True, eq=False)
class MetalLaunch:
    """One Metal kernel launch of a Helion kernel: the complete MSL source
    of the ``kernel void`` named ``name``, launched as ``grid`` threadgroups
    of ``threadgroup`` threads on ``arguments``."""

    name: str
    source: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    arguments: tuple[MetalArgument, ...]
    msl: MSLKernel = dataclasses.field(repr=False, compare=False)

    def to_mlx(
        self,
        name: str | None = None,
        *,
        init_value: float | Literal["captured"] | None,
    ) -> MLXKernel:
        """This launch for MLX's ``mx.fast.metal_kernel`` (:class:`MLXKernel`).

        MLX caches kernels by name, so the default name is the kernel's with
        a digest of what MLX compiles.  The kernel body moves into a function
        in the header, which the source calls: MLX reads the source on every
        launch, the header only when it compiles.

        Refused (``BackendUnsupported``): a buffer the kernel both reads and
        writes, since MLX's outputs are fresh arrays; a written buffer that
        shares storage with another argument, since MLX passes each array as
        its own buffer; a buffer that is not row-contiguous, since the
        kernel indexes it by its strides; and an input of fewer than
        ``MLX_MAX_CONSTANT_ARRAY_SIZE`` elements that the kernel views as an
        MPP device tensor, since MLX binds such an input in ``constant``
        memory, which MPP's views cannot read.

        ``init_value`` is MLX's fill for every output, all that reaches the
        kernel of what its outputs held before the launch: None when the
        kernel writes every element of its outputs, else the value of the
        elements it leaves.  ``"captured"`` takes it from the outputs as they
        are when this is called (a capture dispatches nothing, so as the host
        code left them, unless it or the caller changed them after the
        launch), and refuses outputs one float32 cannot reproduce: an output
        that holds several values, since a capture cannot tell data the
        kernel keeps from what ``torch.empty`` happened to hold; two outputs
        that hold different values; or a value float32 does not hold.
        """
        msl = self.msl
        fn = f"{self.name}_body"
        arguments = {a.name: a for a in self.arguments}
        bound = [(p, arguments[p.name]) for p in msl.params]
        buffers = {p.name: a for p, a in bound if not p.by_value}
        _check_mlx_buffers(buffers, msl.device_views)
        inputs = [n for n, a in buffers.items() if not a.written]
        outputs = [n for n, a in buffers.items() if a.written]
        mlx_names = _mlx_names(list(buffers))
        scalars = [f"    {p.baked(a.value)}" for p, a in bound if p.by_value]
        pointers = [p for p, _ in bound if not p.by_value]
        # MLX passes an input in the ``device`` or the ``constant`` address
        # space, so the body takes each buffer as a template parameter, of
        # the kernel's element type in the space MLX chose (_BUFFER_CAST).
        # The body keeps the buffers' own names; MLX's signature has
        # ``mlx_names``.
        # Template parameters named as no buffer's name begins.
        tparam = "_helion_T"
        while any(p.name.startswith(tparam) for p in pointers):
            tparam += "_"
        fn_params = [
            *(f"{tparam}{i} {p.name}" for i, p in enumerate(pointers)),
            *(f"{t} {n}" for t, n, _ in msl.attributes),
            *(f"threadgroup {t} (&{n})[{slots}]" for t, n, slots in msl.scratch),
        ]
        typenames = ", ".join(f"typename {tparam}{i}" for i in range(len(pointers)))
        function = "\n".join(
            [
                *([f"template <{typenames}>"] if pointers else []),
                f"inline void {fn}({', '.join(fn_params)}) {{",
                *scalars,
                *msl.prologue,
                *msl.body,
                "}",
            ]
        )
        call_args = [
            *(f"_helion_buffer<{p.type}>({mlx_names[p.name]})" for p in pointers),
            *(a for _, _, a in msl.attributes),
            *(n for _, n, _ in msl.scratch),
        ]
        source = "\n".join(
            [
                *(f"threadgroup {t} {n}[{slots}];" for t, n, slots in msl.scratch),
                f"{fn}({', '.join(call_args)});",
                "",
            ]
        )
        header = "\n".join(
            [
                _inline_c10("\n".join(msl.preamble)),
                _BUFFER_CAST,
                *msl.tables,
                function,
            ]
        )
        if name is None:
            digest = hashlib.sha256(f"{header}\0{source}".encode()).hexdigest()
            name = f"{self.name}_{digest[:16]}"
        gx, gy, gz = self.grid
        bx, by, bz = self.threadgroup
        written = {n: cast("torch.Tensor", buffers[n].value) for n in outputs}
        return MLXKernel(
            name=name,
            header=header,
            source=source,
            input_names=tuple(mlx_names[n] for n in inputs),
            output_names=tuple(mlx_names[n] for n in outputs),
            inputs=tuple(cast("torch.Tensor", buffers[n].value) for n in inputs),
            output_shapes=tuple(tuple(t.shape) for t in written.values()),
            output_dtypes=tuple(t.dtype for t in written.values()),
            grid=(gx * bx, gy * by, gz * bz),
            threadgroup=self.threadgroup,
            init_value=(
                _captured_init_value(written)
                if init_value == "captured"
                else init_value
            ),
        )


def capture_metal_launches(
    kernel: Kernel[object],
    args: Sequence[object],
    config: Config | None = None,
    *,
    num_sm: int | None = None,
) -> list[MetalLaunch]:
    """The Metal launches ``kernel`` makes on ``args`` at ``config`` (None:
    the config ``BoundKernel.to_code`` picks), without compiling or
    dispatching any of them.

    The host code runs as it would, so it allocates its outputs, on the
    device of ``args``.  The MSL depends on the tensors' dtypes, shapes and
    strides (as the kernel specializes on them) and which share storage, not
    their contents.  It also depends on the GPU in two places: a register
    matmul's operand layout is probed on an Apple GPU when the tensors are
    on MPS, and assumed otherwise; and a persistent kernel
    (``pid_type="persistent_*"``) bakes in the GPU's core count, ``num_sm``
    (less the cores ``persistent_reserved_sms`` reserves, at least 1, as on
    a GPU).  Without it, the host code reads the count from the tensors' device
    (``helion.runtime.get_num_sm``), which only MPS tensors have, so a
    persistent kernel captured on CPU tensors takes ``num_sm``.  For MSL
    that will run, capture on MPS tensors, with the ``num_sm`` of the GPU
    that will run it if that is another; CPU tensors suffice to read the
    code.
    """
    args = kernel.normalize_args(*args)
    bound = kernel.bind(args)
    if bound.env.backend.name != "metal":
        raise exc.BackendUnsupported(
            bound.env.backend.name, "capture_metal_launches takes a Metal kernel"
        )
    return record_launches(bound.compile_config(config), args, num_sm=num_sm)


def record_launches(
    entry: Callable[..., object],
    args: Sequence[object],
    *,
    num_sm: int | None = None,
) -> list[MetalLaunch]:
    """The launches ``entry``, a Metal kernel's generated host function,
    makes on ``args``: recorded by its launcher, not compiled or
    dispatched.  ``num_sm`` is the core count a persistent kernel bakes in
    (:func:`capture_metal_launches`)."""
    launches: list[MetalLaunch] = []

    def record(
        metal_kernel: _MetalKernel,
        grid: tuple[int, ...],
        *launch_args: object,
        _block_dims: tuple[int, int, int] = DEFAULT_BLOCK_DIMS,
        _by_value: tuple[int, ...] = (),
        **kwargs: object,
    ) -> None:
        check_launcher_kwargs(kwargs)
        launches.append(metal_kernel.capture(grid, _block_dims, launch_args, _by_value))

    if num_sm is not None and num_sm < 1:
        raise ValueError(f"num_sm is a core count, at least 1: got {num_sm}")

    def run() -> None:
        # Imported here: helion.runtime imports this module.
        from ... import runtime

        runtime._num_sm_override.set(functools.partial(_core_count, num_sm))
        entry(*args, _launcher=record)

    # The override is set in a copy of the context, so it ends with the
    # capture and no other thread's host code reads it.
    contextvars.copy_context().run(run)
    return launches


def _core_count(
    num_sm: int | None, device: torch.device, reserved_sms: int, cluster_size: int
) -> int | None:
    """The core count a capture's host code reads (``get_num_sm``):
    ``num_sm`` less the reserved cores, at least 1 as on a GPU; without
    ``num_sm``, the device's own (None) on MPS, and refused on another
    device, which has no Apple GPU's count to read."""
    if num_sm is not None:
        return max(num_sm - reserved_sms, 1)
    if device.type != "mps":
        raise exc.BackendUnsupported(
            "metal",
            "a persistent kernel bakes in the GPU's core count, which "
            f"{device.type} tensors do not have: capture it on MPS tensors, "
            "or pass num_sm",
        )
    return None


def _check_mlx_buffers(
    buffers: dict[str, MetalArgument], device_views: frozenset[str]
) -> None:
    for name, a in buffers.items():
        t = cast("torch.Tensor", a.value)
        if a.written and a.read:
            reason = "is read and written, but MLX hands a kernel fresh outputs"
        elif a.written and a.shares_storage:
            reason = "is written and shares storage, unlike MLX's arrays"
        elif not t.is_contiguous():
            reason = (
                f"has strides {t.stride()}, but MLX passes and allocates "
                "row-contiguous arrays"
            )
        elif (
            not a.written
            and name in device_views
            and t.numel() < MLX_MAX_CONSTANT_ARRAY_SIZE
        ):
            reason = (
                f"has {t.numel()} elements, which MLX binds in constant "
                "memory, but the kernel views it as an MPP device tensor"
            )
        else:
            continue
        raise exc.BackendUnsupported("metal", f"buffer {name} {reason}")


#: MLX binds an input of fewer elements than this in ``constant`` memory,
#: a larger one in ``device`` memory (``max_constant_array_size`` in the
#: signature ``mx.fast.metal_kernel`` writes).
MLX_MAX_CONSTANT_ARRAY_SIZE = 8

#: The position attributes ``mx.fast.metal_kernel`` declares for a source
#: that names them, each as a parameter of that name (its attribute table,
#: in MLX 0.32).
_MLX_ATTRIBUTES = frozenset(
    {
        "dispatch_quadgroups_per_threadgroup",
        "dispatch_simdgroups_per_threadgroup",
        "dispatch_threads_per_threadgroup",
        "grid_origin",
        "grid_size",
        "quadgroup_index_in_threadgroup",
        "quadgroups_per_threadgroup",
        "simdgroup_index_in_threadgroup",
        "simdgroups_per_threadgroup",
        "thread_execution_width",
        "thread_index_in_quadgroup",
        "thread_index_in_simdgroup",
        "thread_index_in_threadgroup",
        "thread_position_in_grid",
        "thread_position_in_threadgroup",
        "threadgroup_position_in_grid",
        "threadgroups_per_grid",
        "threads_per_grid",
        "threads_per_simdgroup",
        "threads_per_threadgroup",
    }
)

#: The parameters ``mx.fast.metal_kernel`` adds for an input whose name,
#: with one of these suffixes, the source holds: its shape, strides, rank.
_MLX_INPUT_SUFFIXES = ("_shape", "_strides", "_ndim")


def _mlx_names(names: list[str]) -> dict[str, str]:
    """Each buffer's parameter name in MLX's signature: its own, unless MLX
    declares that name itself (a position attribute, or another buffer's
    shape, strides or rank under that buffer's name in the signature) or
    another buffer has it; then its own with ``_`` appended until neither.
    Renaming one buffer changes the names MLX derives from it, so the
    check repeats until no name changes."""
    final = {name: name for name in names}
    changed = True
    while changed:
        changed = False
        for name in names:
            others = [final[other] for other in names if other != name]
            declared = _MLX_ATTRIBUTES | {
                other + suffix for other in others for suffix in _MLX_INPUT_SUFFIXES
            }
            while final[name] in declared or final[name] in others:
                final[name] += "_"
                changed = True
    return final


#: A buffer MLX binds, as a pointer to the kernel's element type ``T`` in the
#: same address space and without ``const``: MLX's array may hold another
#: element type of the same bytes, and MPP's operand views take a ``device
#: T*`` where MLX binds an input as ``const device``.  MLX binds a 0-dim
#: input by reference.
_BUFFER_CAST = """\
template <typename T, typename P> inline device T* _helion_buffer(const device P* p) {
  return (device T*)(p);
}
template <typename T, typename P> inline device T* _helion_buffer(device P* p) {
  return (device T*)(p);
}
template <typename T, typename P> inline constant T* _helion_buffer(const constant P* p) {
  return (constant T*)(p);
}
template <typename T, typename P> inline constant T* _helion_buffer(const constant P& p) {
  return (constant T*)(&p);
}"""


def _captured_init_value(outputs: dict[str, torch.Tensor]) -> float | None:
    """MLX's ``init_value`` that reproduces what ``outputs`` hold
    (``to_mlx(init_value="captured")``): the one value they all hold, None
    when they are empty; refused when one float32 cannot reproduce them."""
    fill: torch.Tensor | None = None
    filled = ""
    for name, t in outputs.items():
        flat = t.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        first = flat[0]
        nan = bool(torch.isnan(first))
        if not bool(_same_value(flat, first).all()):
            raise exc.BackendUnsupported(
                "metal",
                f"output {name} holds several values, which MLX's one "
                "init_value cannot reproduce: pass to_mlx(init_value=None) "
                "when the kernel writes every element of its outputs, else "
                "the value of the elements it leaves",
            )
        value = first.to(torch.float32)
        if not nan and not bool(value.to(t.dtype) == first):
            raise exc.BackendUnsupported(
                "metal",
                f"output {name} holds {first.item()}, which MLX's float32 "
                "init_value does not",
            )
        if fill is None:
            fill, filled = value, name
        elif not bool(_same_value(fill, value)):
            raise exc.BackendUnsupported(
                "metal",
                f"outputs {filled} and {name} hold {fill.item()} and "
                f"{value.item()}, but MLX's init_value is one for all outputs",
            )
    return None if fill is None else float(fill)


def _same_value(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise, whether ``a`` and ``b`` hold the same value: NaN is NaN,
    and ``-0.0`` is not ``0.0``, which an ``init_value`` reproduces too."""
    same = (torch.isnan(a) & torch.isnan(b)) | (a == b)
    return same & (torch.signbit(a) == torch.signbit(b))


def _inline_c10(text: str) -> str:
    """``text`` with torch's headers inlined, as ``torch.mps.compile_shader``
    inlines them, from the same directory: MLX compiles a kernel's header
    without torch's include path."""
    return _embed_headers(
        [line + "\n" for line in text.split("\n")],
        [Path(torch.__file__).parent / "include"],
        set(),
    )
