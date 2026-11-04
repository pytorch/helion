"""@metal_jit — decorator that JIT-compiles a Python function to an MSL kernel.

Helion's codegen produces a Python function decorated with ``@metal_jit``.
The function body is Python-syntax Metal — valid Python AST that maps to
MSL C++.  Expressions come from Inductor's MetalOverrides, parsed by
``msl_expr`` (``metal.precise.sin``, ``c10.metal.max``,
``static_cast(float, x)``),
pointer ops from PointerIndexingStrategy (``tl.load``, ``tl.store``),
and thread indices from MetalBackend (``tgid[0]``, ``tid[0]``).

The decorator infers all metadata at runtime from the actual call arguments:
- **Tensor dtypes** → from ``arg.dtype`` (mapped via ``DTYPE_TO_METAL``)
- **Runtime scalars** → bound by value, an int as ``constant long&`` and a
  float as ``constant float&``, at the positions the launcher names; 0-dim
  tensors are read from their buffer
- **Block sizes** → read from module-level globals (``_BLOCK_SIZE_0 = 256``)

When the launcher calls ``metal_kernel.launch(...)`` with a signature it has
not seen, the kernel:
1. Parses the function source to recover the Python AST
2. Infers the MSL kernel signature from the actual args
3. Reads block sizes from the function's module globals
4. Translates the AST body to MSL C++ via the walker
5. Compiles via ``torch.mps.compile_shader``
6. Keeps the kernel function and the positions of its dispatched arguments,
   so a repeat launch is a lookup and a dispatch
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
from typing import TYPE_CHECKING

import torch

from ... import exc
from ._constants import MAX_THREADGROUP_MEMORY_BYTES
from .lane_fold import unroll_lane_folds
from .msl_ast_walker import EmitState
from .msl_ast_walker import _emit_stmts
from .msl_reduction import GROUP_PATHS
from .msl_reduction import REDUCTION_INCLUDES
from .msl_reduction import REDUCTION_NAMESPACE
from .msl_reduction import REDUCTION_PREAMBLE
from .reduction import LAUNCH_DIM_VARS
from .reduction import RED_SITE_GLOBAL_PREFIX
from .reduction import TG_BUFFER_GLOBAL_PREFIX
from .reduction import group_path
from .reduction import group_scratch_slots
from .tile_sharing import share_tiles

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

#: The MSL type each runtime scalar's Python type binds as.  torch binds a
#: Python int as 8 bytes and a float as 4.
_RUNTIME_SCALAR_TYPES: dict[type, str] = {int: "long", float: "float"}
_NO_SCALARS: Mapping[str, str] = types.MappingProxyType({})


def metal_jit(fn: Callable[..., object]) -> _MetalKernel:
    """Decorator that JIT-compiles a Python function to an MSL Metal kernel."""
    return _MetalKernel(fn)


class _MetalKernel:
    """Wrapped Metal kernel that translates Python AST to MSL on first call."""

    def __init__(self, fn: Callable[..., object]) -> None:
        self._fn = fn
        self._name = fn.__name__
        self.msl_source: str | None = None
        self._plans: dict[
            tuple[object, ...], tuple[Callable[..., object], tuple[int, ...]]
        ] = {}

    def launch(
        self,
        grid: tuple[int, ...],
        block_dims: tuple[int, int, int],
        args: tuple[object, ...],
        by_value: tuple[int, ...],
    ) -> None:
        """Dispatch ``grid`` threadgroups of ``block_dims`` threads on ``args``.

        ``by_value`` holds the positions of the runtime scalars.
        """
        key = self._plan_key(block_dims, args, by_value)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plans[key] = self._plan(block_dims, args, by_value)
        fn, dispatched = plan
        gx, gy, gz = (*grid, 1, 1)[:3]
        bx, by, bz = block_dims
        fn(
            *[args[i] for i in dispatched],
            threads=(gx * bx, gy * by, gz * bz),
            group_size=block_dims,
        )

    @staticmethod
    def _plan_key(
        block_dims: tuple[int, int, int],
        args: tuple[object, ...],
        by_value: tuple[int, ...],
    ) -> tuple[object, ...]:
        """Everything the compiled shader depends on besides the body: block
        dims, tensor dtypes and ranks, and baked non-tensor values.  Runtime
        scalars contribute only their type: they bind by value
        (:data:`_RUNTIME_SCALAR_TYPES`) instead of being baked into the
        shader, so a new value reuses the compiled kernel."""
        return (
            block_dims,
            *[
                (arg.dtype, arg.ndim)
                if isinstance(arg, torch.Tensor)
                else type(arg)
                if i in by_value
                else (type(arg), arg)
                for i, arg in enumerate(args)
            ],
        )

    def _plan(
        self,
        block_dims: tuple[int, int, int],
        args: tuple[object, ...],
        by_value: tuple[int, ...],
    ) -> tuple[Callable[..., object], tuple[int, ...]]:
        """Compile the kernel for a new launch signature.

        Returns the kernel function and the positions of the arguments it
        binds: tensors as buffers and runtime scalars by value, in launch
        order, which is the order ``_generate_msl`` numbers their slots in.
        """
        for i in by_value:
            if type(args[i]) not in _RUNTIME_SCALAR_TYPES:
                raise exc.BackendUnsupported(
                    "metal",
                    f"runtime scalar argument {i} is a {type(args[i]).__name__}; "
                    "Metal binds runtime scalars as int or float",
                )
        lib = self._compile(args, by_value, block_dims)
        dispatched = tuple(
            i
            for i, arg in enumerate(args)
            if i in by_value or isinstance(arg, torch.Tensor)
        )
        return getattr(lib, self._name), dispatched

    def _compile(
        self,
        args: tuple[object, ...],
        by_value: tuple[int, ...],
        block_dims: tuple[int, int, int],
    ) -> object:
        """Translate the function body to MSL for ``args`` and compile it."""
        source = textwrap.dedent(inspect.getsource(self._fn))
        fn_def = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
        )
        param_names = [a.arg for a in fn_def.args.args]
        self.msl_source = _generate_msl(
            kernel_name=self._name,
            body_stmts=fn_def.body,
            param_names=param_names,
            args=args,
            fn_globals=self._fn.__globals__,
            required_threads_per_threadgroup=block_dims,
            by_value={
                param_names[i]: _RUNTIME_SCALAR_TYPES[type(args[i])] for i in by_value
            },
        )
        return torch.mps.compile_shader(self.msl_source)  # type: ignore[attr-defined]


def _generate_msl(
    kernel_name: str,
    body_stmts: list[ast.stmt],
    param_names: list[str],
    args: tuple[object, ...],
    fn_globals: dict[str, object],
    required_threads_per_threadgroup: tuple[int, int, int] | None = None,
    by_value: Mapping[str, str] = _NO_SCALARS,
) -> str:
    """Generate complete MSL source from Python AST body and actual args.

    Infers Metal dtypes and scalar vs tensor distinction directly from
    the args, apart from the ``by_value`` parameters: runtime scalars, each
    mapped to the MSL type it binds as.  Reads block sizes from fn_globals.
    """
    from torch._inductor.codegen.mps import DTYPE_TO_METAL

    from .lane_fold import BF16_CAST
    from .mpp_matmul import BF16_CAST_PREAMBLE
    from .mpp_matmul import ROUND
    from .mpp_matmul import ROUND_PREAMBLE
    from .mpp_matmul import elementize
    from .mpp_matmul import names_read

    tensor_dtypes = {
        name: arg.dtype
        for name, arg in zip(param_names, args, strict=True)
        if isinstance(arg, torch.Tensor)
    }
    for dtype in tensor_dtypes.values():
        if dtype not in DTYPE_TO_METAL:
            raise ValueError(f"Unsupported Metal dtype: {dtype}")
    body_stmts = elementize(
        body_stmts,
        tensor_dtypes,
        frozenset(
            k[len(TG_BUFFER_GLOBAL_PREFIX) :]
            for k in fn_globals
            if k.startswith(TG_BUFFER_GLOBAL_PREFIX)
        ),
        required_threads_per_threadgroup,
    )
    body_stmts = unroll_lane_folds(body_stmts, names_read)
    msl_parts: list[str] = [
        "#include <metal_stdlib>",
        "#include <c10/metal/utils.h>",
        "#include <c10/metal/special_math.h>",
        "using namespace metal;",
    ]
    referenced = {
        node.id
        for stmt in body_stmts
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    }
    reduction_sites = _reduction_sites(
        fn_globals, referenced, required_threads_per_threadgroup
    )
    body_stmts, threadgroup_buffers = _threadgroup_buffers(
        body_stmts,
        fn_globals,
        referenced,
        reduction_sites,
        required_threads_per_threadgroup,
    )
    uses_reductions = _uses_reductions(body_stmts)
    if uses_reductions:
        msl_parts.extend(REDUCTION_INCLUDES)
        msl_parts.extend(("", REDUCTION_PREAMBLE))
    if _uses_mpp_markers(body_stmts):
        _raise_if_mpp_unsupported_device()
        msl_parts.extend(
            [
                "#if !__has_include(<metal_tensor>)",
                '#error "Helion Metal MPP lowering requires <metal_tensor>"',
                "#endif",
                (
                    "#if !__has_include("
                    "<MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>)"
                ),
                (
                    '#error "Helion Metal MPP lowering requires '
                    '<MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>"'
                ),
                "#endif",
                "#include <metal_tensor>",
                "#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>",
                "using namespace mpp::tensor_ops;",
            ]
        )
    if ROUND in referenced or BF16_CAST in referenced:
        msl_parts.append(ROUND_PREAMBLE)
    if BF16_CAST in referenced:
        msl_parts.append(BF16_CAST_PREAMBLE)
    msl_parts.append("")

    params: list[str] = []
    scalar_preamble: list[str] = []
    buf_idx = 0
    for name, arg in zip(param_names, args, strict=True):
        if name in by_value:
            params.append(f"constant {by_value[name]}& {name} [[buffer({buf_idx})]]")
            buf_idx += 1
            continue
        if not isinstance(arg, torch.Tensor):
            # A host-side constexpr the launcher passes positionally (e.g. a
            # rolled reduction's ``_REDUCTION_BLOCK_*``).  The launcher binds
            # only tensors and runtime scalars, so bake the value into the
            # shader.  The launch plan keys on these values, so a different
            # one recompiles rather than reusing a stale shader.
            scalar_preamble.append(f"    {_constexpr_decl(name, arg)}")
            continue
        metal_dtype = DTYPE_TO_METAL[arg.dtype]
        is_scalar = arg.ndim == 0
        if is_scalar:
            buf_param = f"_buf_{name}"
            params.append(
                f"device const {metal_dtype}* {buf_param} [[buffer({buf_idx})]]"
            )
            scalar_preamble.append(f"    {metal_dtype} {name} = {buf_param}[0];")
        else:
            params.append(f"device {metal_dtype}* {name} [[buffer({buf_idx})]]")
        buf_idx += 1

    params.extend(
        (
            "uint3 tgid [[threadgroup_position_in_grid]]",
            "uint3 tid [[thread_position_in_threadgroup]]",
        )
    )
    sig = ",\n    ".join(params)
    required_threads_attr = ""
    if required_threads_per_threadgroup is not None:
        tx, ty, tz = required_threads_per_threadgroup
        required_threads_attr = (
            f"[[required_threads_per_threadgroup({tx}, {ty}, {tz})]] "
        )
    msl_parts.append(
        f"{required_threads_attr}kernel void {kernel_name}(\n    {sig}\n) {{"
    )
    msl_parts.extend(scalar_preamble)

    # Declare _BLOCK_SIZE_* constants from module globals
    block_sizes = {
        name: int(val)
        for name, val in fn_globals.items()
        if name.startswith("_BLOCK_SIZE_") and isinstance(val, int)
    }
    for name in sorted(block_sizes):
        msl_parts.append(f"    constexpr int {name} = {block_sizes[name]};")
    if required_threads_per_threadgroup is not None:
        for name, size in zip(
            LAUNCH_DIM_VARS, required_threads_per_threadgroup, strict=True
        ):
            msl_parts.append(f"    constexpr uint {name} = {size};")
    for site, (path, _) in sorted(reduction_sites.items()):
        msl_parts.append(f"    constexpr uint {site} = {GROUP_PATHS.index(path)};")

    # ``threadgroup`` variables are only legal at kernel scope, but a reduction
    # can be emitted from inside a loop body, so their declarations are hoisted
    # here from the generated ``_METAL_TG_BUF_*`` module globals.
    for buf_name, buf_dtype, buf_slots in threadgroup_buffers:
        msl_parts.append(f"    threadgroup {buf_dtype} {buf_name}[{buf_slots}];")

    state = EmitState(
        declared={
            *block_sizes,
            *LAUNCH_DIM_VARS,
            *reduction_sites,
            *(name for name, _, _ in threadgroup_buffers),
        }
    )

    _emit_stmts(body_stmts, msl_parts, indent=4, state=state)

    msl_parts.append("}")
    return "\n".join(msl_parts)


def _constexpr_decl(name: str, value: object) -> str:
    """MSL declaration for a non-tensor kernel argument."""
    if isinstance(value, bool):
        return f"constexpr bool {name} = {'true' if value else 'false'};"
    if isinstance(value, int):
        if not -(2**31) <= value < 2**31:
            raise exc.BackendUnsupported(
                "metal", f"kernel argument {name} = {value} does not fit an int"
            )
        return f"constexpr int {name} = {value};"
    if isinstance(value, float):
        return f"constexpr float {name} = {value!r};"
    raise exc.BackendUnsupported("metal", f"kernel argument type: {type(value)}")


def _uses_mpp_markers(body_stmts: list[ast.stmt]) -> bool:
    for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id.startswith("_helion_mpp_"):
            return True
    return False


def _uses_reductions(body_stmts: list[ast.stmt]) -> bool:
    """Return True if the body calls into the ``helion_red`` MSL namespace.

    Every reduction the Metal backend emits is routed through that namespace
    (see ``metal/msl_reduction.py``), so this one check decides whether the
    kernel needs the reduction preamble and the SIMD-group index parameter.
    """
    for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[])):
        if isinstance(node, ast.Name) and node.id == REDUCTION_NAMESPACE:
            return True
    return False


def _reduction_sites(
    fn_globals: dict[str, object],
    referenced: set[str],
    threads_per_threadgroup: tuple[int, int, int] | None,
) -> dict[str, tuple[str, int]]:
    """Pick each group reduction site's path: ``{site: (path, axis)}``.

    Sites are written by ``metal/reduction.py::_group_site`` as generated
    module-level globals.  The path depends on the launch's threadgroup size,
    which is only final once the whole body is emitted, so it is chosen here
    rather than at codegen.
    """
    sites: dict[str, tuple[str, int]] = {}
    for name in sorted(fn_globals):
        if not name.startswith(RED_SITE_GLOBAL_PREFIX):
            continue
        site, axis = fn_globals[name]  # type: ignore[misc]
        if site not in referenced:
            continue
        assert threads_per_threadgroup is not None
        path = group_path(axis, threads_per_threadgroup)
        sites[str(site)] = (path, int(axis))
    return sites


#: Size of each MSL scalar type a scratch buffer can hold.
_MSL_SCALAR_BYTES = {
    "bool": 1,
    "char": 1,
    "uchar": 1,
    "short": 2,
    "ushort": 2,
    "half": 2,
    "bfloat": 2,
    "int": 4,
    "uint": 4,
    "float": 4,
    "long": 8,
    "ulong": 8,
}


def _threadgroup_buffers(
    body_stmts: list[ast.stmt],
    fn_globals: dict[str, object],
    referenced: set[str],
    reduction_sites: dict[str, tuple[str, int]],
    threads_per_threadgroup: tuple[int, int, int] | None,
) -> tuple[list[ast.stmt], list[tuple[str, str, int]]]:
    """Collect ``(name, metal_dtype, slots)`` scratch declarations to hoist,
    with MPP tiles that are never live together sharing one
    (:func:`tile_sharing.share_tiles`, which edits the body in place and
    returns it).

    Written by ``metal/reduction.py::alloc_threadgroup_buffer`` as generated
    module-level globals, the same transport ``_BLOCK_SIZE_*`` uses.  Only
    buffers this kernel body actually references are declared, so a global left
    behind by discarded codegen cannot consume threadgroup memory here.  A
    group reduction's buffer is sized for its site's path; one the path never
    touches still needs a declaration, of one slot.  A ``strided`` site spends
    up to a slot per thread to save a collective per row, so while the scratch
    exceeds the threadgroup memory the largest such site falls back to the
    ``masked`` path (updating ``reduction_sites``), whose slots are one per
    SIMD group.  Kernels whose scratch still exceeds it are unsupported.
    """
    declared: list[tuple[str, str, int | str]] = []
    for name in sorted(fn_globals):
        if not name.startswith(TG_BUFFER_GLOBAL_PREFIX):
            continue
        buf_name, buf_dtype, buf_slots = fn_globals[name]  # type: ignore[misc]
        if buf_name in referenced:
            declared.append((str(buf_name), str(buf_dtype), buf_slots))  # type: ignore[arg-type]
    body_stmts, declared = share_tiles(body_stmts, declared)

    def slots(buf_slots: int | str) -> int:
        if isinstance(buf_slots, int):
            return buf_slots
        path, axis = reduction_sites[buf_slots]
        assert threads_per_threadgroup is not None
        return max(1, group_scratch_slots(path, threads_per_threadgroup, axis))

    def site_bytes(site: str) -> int:
        return sum(
            _MSL_SCALAR_BYTES[dtype] * slots(buf_slots)
            for _, dtype, buf_slots in declared
            if buf_slots == site
        )

    def buffers() -> list[tuple[str, str, int]]:
        return [(name, dtype, slots(buf_slots)) for name, dtype, buf_slots in declared]

    def used() -> int:
        return sum(_MSL_SCALAR_BYTES[dtype] * n for _, dtype, n in buffers())

    strided = sorted(s for s, (path, _) in reduction_sites.items() if path == "strided")
    while used() > MAX_THREADGROUP_MEMORY_BYTES and strided:
        site = max(strided, key=site_bytes)
        strided.remove(site)
        reduction_sites[site] = ("masked", reduction_sites[site][1])
    if used() > MAX_THREADGROUP_MEMORY_BYTES:
        raise exc.BackendUnsupported(
            "metal",
            f"reduction scratch and MPP staging need {used()} bytes of "
            f"threadgroup memory (max {MAX_THREADGROUP_MEMORY_BYTES})",
        )
    return body_stmts, buffers()


def _raise_if_mpp_unsupported_device() -> None:
    device_name = torch._C._mps_get_name()  # type: ignore[attr-defined]
    if "paravirtual" in device_name.lower():
        raise exc.BackendUnsupported(
            "metal",
            "MPP matmul lowering is not supported on Apple paravirtual MPS "
            "devices; use a physical Apple GPU such as a self-hosted M-series "
            "runner",
        )
