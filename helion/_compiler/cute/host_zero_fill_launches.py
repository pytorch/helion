"""Launch the host zero fill of an output as a programmatic dependent kernel.

``acc = torch.zeros(...)`` ahead of a device loop that accumulates into
``acc`` (atomics, bulk reduce-adds) is a memset node in the stream
(``aten::zero_`` issues ``cudaMemsetAsync``).  In a CUDA graph such a node
costs 1.1-1.4 us of device time for a few hundred KB and, standing between
the previous kernel and this kernel, it also defeats programmatic dependent
launch: a ``cute_pdl`` kernel behind a memset node launches as a plain edge.
The example attention backward on (1, 4, 256, 64) reads 11.98 us per call
with the memset, 11.98 with ``cute_pdl`` on top of it and 9.99 with the fill
out of the chain (B200, 64-call graph).

When the kernel is launched as a programmatic dependent (``cute_pdl``), every
remaining host zero fill (``lower_host_zero_fills`` has already turned the
provably dead ones into ``empty``) becomes ``empty`` plus the launch of a
small fill kernel emitted into the same module: itself a dependent of the
previous kernel (its wait precedes its stores), releasing its own dependents
at entry so this kernel's launch overlaps the fill, and ordered ahead of this
kernel by this kernel's own wait.  The launch sits where the fill statement
stood, so stream order is the memset's and a host read of the tensor after
the statement sees zeros exactly as before (11.31 us per call with a
first-cut fill kernel against 11.98 with the memset).  A plain fill kernel
is slower than the memset (12.29 us), so without ``cute_pdl`` the fills stay
memsets.

Admitted fills: ``torch.zeros`` / ``torch.zeros_like`` / ``x.new_zeros``
top-level host assignments whose value is a contiguous tensor on the
kernel's device of a dtype the fill kernel spells (the 16-, 32- and 64-bit
floats and integers) and of fewer than 2**31 elements: the kernel counts in
Int32.  A static element count at or past that stays a memset (an 8 GB
fill is bandwidth-bound either way); a count the host sizes at runtime
(dynamic shapes) launches behind a host check that falls back to the
memset (``zero_()``) past the bound.  The kernel stores 16-byte packets over
the flat view from 256-thread CTAs in a grid-stride loop (at most four CTAs
per SM, one wave, so the entry release never holds slots the fill needs)
and a scalar tail.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from .host_zero_fills import _EMPTY_FACTORY
from .host_zero_fills import _factory_call_source
from .host_zero_fills import _zero_factory
from .pdl_release import ZERO_FILL_KERNEL_PREFIX

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction
    from ..host_function import HostFunction
    from ..source_location import SourceLocation

__all__ = [
    "MAX_FILL_NUMEL",
    "lower_host_zero_fill_launches",
    "static_numel",
    "zero_fill_launch_tensors",
    "zero_fill_launches_apply",
]

_FILL_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "cutlass.Float32",
    torch.float16: "cutlass.Float16",
    torch.bfloat16: "cutlass.BFloat16",
    torch.int64: "cutlass.Int64",
    torch.int32: "cutlass.Int32",
    torch.int16: "cutlass.Int16",
}
_THREADS = 256
_CTAS_PER_SM = 4
_PACKET_BYTES = 16
# The fill kernel indexes in Int32: fills of this many elements or more
# keep their memset (statically) or fall back to it (a runtime check).
MAX_FILL_NUMEL = 2**31


def zero_fill_launches_apply(
    device_function: DeviceFunction, env: CompileEnvironment, config: Config
) -> bool:
    """Whether this kernel's host zero fills launch as dependent kernels:
    the kernel itself launches as a programmatic dependent (``cute_pdl``,
    through the generic path, a wrapper plan's ``use_pdl`` or the tcgen05
    entry wait) on a target with ``griddepcontrol``."""
    if env.backend_name != "cute" or config.config.get("cute_pdl") is not True:
        return False
    capability = env.config_spec.target_device_capability
    if capability is None or capability < (9, 0):
        return False
    state = device_function.cute_state
    return (
        state.cute_pdl_entry_wait
        or device_function._cute_pdl_applies()
        or any(
            plan.get("use_pdl")
            for plan in device_function.codegen.cute_wrapper_plans  # type: ignore[attr-defined]
        )
    )


def _admitted_fill(
    statement: ast.AST, env: CompileEnvironment
) -> tuple[str, str, ast.Call, torch.Tensor] | None:
    """``(target, factory, call, fake)`` when ``statement`` is a host zero
    fill the fill kernel can take: a contiguous strided tensor on the
    kernel's device of a dtype the kernel spells, of fewer than
    ``MAX_FILL_NUMEL`` elements when its element count is static (a dynamic
    count is checked on the host at launch)."""
    found = _zero_factory(statement)
    if found is None:
        return None
    target, factory, call = found
    fake = call._type_info.fake_value  # type: ignore[attr-defined]
    if (
        fake.device != env.device
        or fake.layout != torch.strided
        or fake.dtype not in _FILL_DTYPES
        or not fake.is_contiguous()
    ):
        return None
    numel = static_numel(fake)
    if numel is not None and numel >= MAX_FILL_NUMEL:
        return None
    return target, factory, call, fake


def static_numel(fake: torch.Tensor) -> int | None:
    """The element count of ``fake`` when every size is a plain int (static
    shapes); None when the host sizes it at runtime."""
    numel = 1
    for size in fake.size():
        if not isinstance(size, int):
            return None
        numel *= size
    return numel


def zero_fill_launch_tensors(
    host: HostFunction, env: CompileEnvironment, config: Config
) -> list[torch.Tensor]:
    """The fake tensors of the host zero fills that launch as fill kernels
    under ``config``: the fills the fill kernel can take, less the ones the
    dead-fill elision (``prove_host_zero_fills``) turns into ``empty``."""
    from .host_zero_fills import prove_host_zero_fills

    dead = {fill.target for fill in prove_host_zero_fills(host, env, config)}
    return [
        fill[3]
        for statement in host.body
        if (fill := _admitted_fill(statement, env)) is not None and fill[0] not in dead
    ]


def _fill_kernel_source(name: str, dtype: torch.dtype) -> str:
    element = _FILL_DTYPES[dtype]
    width = _PACKET_BYTES // dtype.itemsize
    return (
        "@cute.kernel\n"
        f"def {name}(zf_t):\n"
        "    cute.arch.griddepcontrol_wait()\n"
        "    cute.arch.griddepcontrol_launch_dependents()\n"
        "    zf_n = cutlass.Int32(cute.size(zf_t.layout))\n"
        f"    zf_tid = cutlass.Int32(cute.arch.block_idx()[0]) * {_THREADS}"
        " + cutlass.Int32(cute.arch.thread_idx()[0])\n"
        f"    zf_stride = cutlass.Int32(cute.arch.grid_dim()[0]) * {_THREADS}\n"
        f"    zf_zero = cute.make_rmem_tensor(({width},), {element})\n"
        f"    zf_zero.fill({element}(0))\n"
        "    zf_atom = cute.make_copy_atom("
        f"cute.nvgpu.CopyUniversalOp(), {element}, num_bits_per_copy={_PACKET_BYTES * 8})\n"
        f"    for zf_i in range(zf_tid, zf_n // {width}, zf_stride):\n"
        "        zf_dst = cute.make_tensor("
        f"zf_t.iterator + cute.assume(zf_i * {width}, divby={width}), "
        f"cute.make_layout(({width},), stride=(1,)))\n"
        "        cute.copy(zf_atom, zf_zero, zf_dst)\n"
        f"    zf_tail = zf_n % {width}\n"
        "    if zf_tid < zf_tail:\n"
        f"        (zf_t.iterator + (zf_n - zf_tail + zf_tid)).store({element}(0))\n"
    )


def lower_host_zero_fill_launches(
    host: HostFunction,
    body: list[ast.AST],
    *,
    env: CompileEnvironment,
    kernel_name: str,
    launcher: str = "_launcher",
) -> tuple[list[ast.AST], list[ast.AST]]:
    """Rewrite the admitted zero fills of ``body`` (the host statements after
    the dead-fill elision) into ``empty`` plus a fill-kernel launch.

    The fills are read off ``host.body`` (the typed source statements; the
    generated statements keep the call's type but not its ``torch`` owner's)
    and rewritten where their statement survived generation unchanged, as
    ``lower_host_zero_fills`` does.  Returns the new host statements and the
    module-level statements (the fill kernel definitions and their launch
    attributes) to place ahead of the host function.  ``kernel_name`` is the
    device function's name; the fill kernels are named after it so a module
    never holds two alike.
    """
    admitted: dict[SourceLocation, tuple[str, str, str, torch.dtype, int | None]] = {}
    for statement in host.body:
        fill = _admitted_fill(statement, env)
        if fill is None:
            continue
        target, factory, _call, fake = fill
        assert isinstance(statement, ExtendedAST)
        admitted[statement._location] = (
            target,
            factory,
            ast.unparse(statement),
            fake.dtype,
            static_numel(fake),
        )
    if not admitted:
        return body, []
    num_sm = env.config_spec.num_sm
    max_ctas = _CTAS_PER_SM * (max(1, num_sm))
    result: list[ast.AST] = []
    module: list[ast.AST] = []
    index = 0
    for statement in body:
        fill = (
            admitted.get(statement._location)
            if isinstance(statement, ExtendedAST)
            else None
        )
        if fill is None or ast.unparse(statement) != fill[2]:
            result.append(statement)
            continue
        target, factory, _text, dtype, numel = fill
        assert isinstance(statement, ast.Assign)
        call = statement.value
        assert isinstance(call, ast.Call)
        fill_name = f"{ZERO_FILL_KERNEL_PREFIX}{index}_{kernel_name.lstrip('_')}"
        index += 1
        module.extend(ast.parse(_fill_kernel_source(fill_name, dtype)).body)
        module.append(statement_from_string(f"{fill_name}._helion_cute_use_pdl = True"))
        per_cta = _THREADS * (_PACKET_BYTES // dtype.itemsize)
        launch = (
            f"{launcher}({fill_name}, (max(1, min(({target}.numel() + "
            f"{per_cta - 1}) // {per_cta}, {max_ctas})),), {target}.view(-1), "
            f"block=({_THREADS}, 1, 1))"
        )
        if numel is None or numel >= MAX_FILL_NUMEL:
            # The kernel counts in Int32: a count the codegen cannot bound
            # below MAX_FILL_NUMEL launches behind a host check that falls
            # back to the memset past the bound.
            launch = (
                f"if {target}.numel() < {MAX_FILL_NUMEL}:\n"
                f"    {launch}\n"
                "else:\n"
                f"    {target}.zero_()\n"
            )
        result.extend(
            (
                statement_from_string(
                    f"{target} = {_factory_call_source(call, _EMPTY_FACTORY[factory])}"
                ),
                ast.parse(launch).body[0],
            )
        )
    return result, module
