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

``metal_kernel.capture(...)`` takes the same launch and runs steps 1 to 4
only: it returns the launch's MSL, grid and arguments
(:class:`~helion._compiler.metal.export.MetalLaunch`) without compiling or
dispatching anything.
"""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses
import functools
import inspect
import textwrap
import types
from typing import TYPE_CHECKING

import torch

from ... import exc
from ._constants import DTYPE_TO_METAL
from ._constants import MAX_THREADGROUP_MEMORY_BYTES
from .c10_helpers import c10_source
from .deferred_reduction import defer_group_sums
from .export import MetalArgument
from .export import MetalLaunch
from .fma_fusion import fuse_multiply_adds
from .fragment_mma import FRAGMENT_LOWERING
from .fragment_mma import LANE
from .fragment_mma import LANE_PREAMBLE
from .fragment_mma import is_fragment_acc
from .lane_fold import MASKED_LOAD
from .lane_fold import MSL_SCALAR_BYTES
from .lane_fold import jam_lanes_into_loops
from .lane_fold import tl_store
from .lane_fold import unroll_lane_folds
from .masked_branch import branch_masked_steps
from .mpp_matmul import ACC
from .mpp_matmul import REG_RUN
from .mpp_matmul import RUN
from .mpp_registers import GRID_TILE
from .mpp_registers import LANE as REGISTER_LANE
from .mpp_registers import REGISTER_LOWERING
from .mpp_registers import grid_tiles
from .msl_ast_walker import MASKED_LOAD_PREAMBLE
from .msl_ast_walker import EmitState
from .msl_ast_walker import _emit_stmts
from .msl_ast_walker import select_masked_loads
from .msl_reduction import GROUP_PATHS
from .msl_reduction import REDUCTION_NAMESPACE
from .msl_reduction import REDUCTION_PREAMBLE
from .operand_elements import SIMDGROUP
from .operand_elements import SPLIT_BUFFER
from .operand_elements import kernel_constants
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

    def capture(
        self,
        grid: tuple[int, ...],
        block_dims: tuple[int, int, int],
        args: tuple[object, ...],
        by_value: tuple[int, ...],
    ) -> MetalLaunch:
        """The launch :meth:`launch` would dispatch, with the MSL it would
        compile, as data: nothing is compiled or dispatched."""
        _check_runtime_scalars(args, by_value)
        msl = self._msl(args, by_value, block_dims)
        names = self._param_names
        written, read = _buffer_access(self._fn_def().body, names)
        shared = _shared_storage(args)
        params = {p.name: p for p in msl.params}
        gx, gy, gz = (*grid, 1, 1)[:3]
        return MetalLaunch(
            name=self._name,
            source=msl.render(),
            grid=(gx, gy, gz),
            threadgroup=block_dims,
            arguments=tuple(
                MetalArgument(
                    name=name,
                    value=arg,
                    binding=(
                        "baked"
                        if name not in params
                        else "scalar"
                        if params[name].by_value
                        else "buffer"
                    ),
                    written=name in written,
                    read=name in read,
                    shares_storage=i in shared,
                )
                for i, (name, arg) in enumerate(zip(names, args, strict=True))
            ),
            msl=msl,
        )

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
        dims, tensor dtypes, ranks, emptiness and which tensors share storage,
        and baked non-tensor values.  Runtime scalars contribute only their
        type: they bind by value (:data:`_RUNTIME_SCALAR_TYPES`) instead of
        being baked into the shader, so a new value reuses the compiled
        kernel."""
        return (
            block_dims,
            *[
                (arg.dtype, arg.ndim, arg.numel() > 0)
                if isinstance(arg, torch.Tensor)
                else type(arg)
                if i in by_value
                else (type(arg), arg)
                for i, arg in enumerate(args)
            ],
            _shared_storage(args),
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
        order, which is the order ``_generate_msl_kernel`` numbers their slots in.
        """
        _check_runtime_scalars(args, by_value)
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
        self.msl_source = self._msl(args, by_value, block_dims).render()
        return torch.mps.compile_shader(self.msl_source)  # type: ignore[attr-defined]

    def _msl(
        self,
        args: tuple[object, ...],
        by_value: tuple[int, ...],
        block_dims: tuple[int, int, int],
    ) -> MSLKernel:
        """Translate the function body to MSL for ``args``."""
        param_names = self._param_names
        return _generate_msl_kernel(
            kernel_name=self._name,
            body_stmts=self._fn_def().body,
            param_names=param_names,
            args=args,
            fn_globals=self._fn.__globals__,
            required_threads_per_threadgroup=block_dims,
            by_value={
                param_names[i]: _RUNTIME_SCALAR_TYPES[type(args[i])] for i in by_value
            },
        )

    def _fn_def(self) -> ast.FunctionDef:
        """The function's AST, parsed again on each call: the translation
        rewrites the statements it is given."""
        return next(
            node
            for node in ast.walk(ast.parse(self._source))
            if isinstance(node, ast.FunctionDef)
        )

    @functools.cached_property
    def _source(self) -> str:
        return textwrap.dedent(inspect.getsource(self._fn))

    @functools.cached_property
    def _param_names(self) -> list[str]:
        return [a.arg for a in self._fn_def().args.args]


def _check_runtime_scalars(args: tuple[object, ...], by_value: tuple[int, ...]) -> None:
    for i in by_value:
        if type(args[i]) not in _RUNTIME_SCALAR_TYPES:
            raise exc.BackendUnsupported(
                "metal",
                f"runtime scalar argument {i} is a {type(args[i]).__name__}; "
                "Metal binds runtime scalars as int or float",
            )


def _buffer_access(
    body_stmts: list[ast.stmt], param_names: list[str]
) -> tuple[set[str], set[str]]:
    """The parameters the body writes and the ones it reads.  A buffer is
    written through the base pointer of a ``tl.store(buf + offset, ...)``,
    the one way the generated code writes one (``msl_ast_walker``); any
    other use of its name reads it."""
    params = set(param_names)
    stores: Counter[str] = Counter()
    named: Counter[str] = Counter()
    for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[])):
        if isinstance(node, ast.Name) and node.id in params:
            named[node.id] += 1
        elif (store := tl_store(node)) is not None:
            ptr = store.args[0]
            while isinstance(ptr, ast.BinOp):
                ptr = ptr.left
            if isinstance(ptr, ast.Name) and ptr.id in params:
                stores[ptr.id] += 1
    return set(stores), {name for name in named if named[name] > stores[name]}


@dataclasses.dataclass(frozen=True)
class MSLParam:
    """A bound kernel parameter: a buffer (``device half* x``) or a runtime
    scalar bound by value (``constant long& n``)."""

    space: str
    type: str
    name: str
    by_value: bool

    def decl(self) -> str:
        return f"{self.space} {self.type}{'&' if self.by_value else '*'} {self.name}"

    def baked(self, value: object) -> str:
        """A by-value scalar's declaration with ``value`` baked in, for a
        runtime that binds buffers only."""
        assert self.by_value
        literal = (
            _float_literal(float(value))  # type: ignore[arg-type]
            if self.type == "float"
            else str(int(value))  # type: ignore[call-overload]
        )
        return f"const {self.type} {self.name} = {literal};"


@dataclasses.dataclass(frozen=True)
class MSLKernel:
    """One kernel's MSL, in the parts that its renderings arrange: the
    ``kernel void`` torch compiles (:meth:`render`), and the forms other
    runtimes take (``export.py``)."""

    name: str
    #: File scope: includes and helpers, then the ``constant`` tables.
    preamble: tuple[str, ...]
    tables: tuple[str, ...]
    #: The bound parameters, in buffer-slot order, then the position
    #: attributes: ``(type, name, attribute)``.
    params: tuple[MSLParam, ...]
    attributes: tuple[tuple[str, str, str], ...]
    #: ``[[required_threads_per_threadgroup]]``, when the launch fixes it.
    threads: tuple[int, int, int] | None
    #: Kernel scope, each line indented by 4: the declarations before the
    #: body (baked host constants, constexprs), then the threadgroup arrays
    #: ``(type, name, slots)``, then the body.
    prologue: tuple[str, ...]
    scratch: tuple[tuple[str, str, int], ...]
    body: tuple[str, ...]
    #: The buffers the body reads through a ``device`` pointer
    #: (``EmitState.device_views``), which a runtime that binds small inputs
    #: in ``constant`` memory, as MLX does, cannot run.
    device_views: frozenset[str]

    def render(self) -> str:
        """The complete MSL source of the ``kernel void``."""
        sig = ",\n    ".join(
            [
                *(f"{p.decl()} [[buffer({i})]]" for i, p in enumerate(self.params)),
                *(f"{t} {n} [[{a}]]" for t, n, a in self.attributes),
            ]
        )
        threads = ""
        if self.threads is not None:
            tx, ty, tz = self.threads
            threads = f"[[required_threads_per_threadgroup({tx}, {ty}, {tz})]] "
        return "\n".join(
            [
                *self.preamble,
                *self.tables,
                f"{threads}kernel void {self.name}(\n    {sig}\n) {{",
                *self.prologue,
                *(f"    threadgroup {t} {n}[{slots}];" for t, n, slots in self.scratch),
                *self.body,
                "}",
            ]
        )


def _shared_storage(args: tuple[object, ...]) -> frozenset[int]:
    """The positions of the tensor arguments that share storage with another
    (an in-place call passing one tensor twice, or views of one tensor)."""
    seen: dict[int, list[int]] = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and arg.numel() > 0:
            seen.setdefault(arg.untyped_storage().data_ptr(), []).append(i)
    return frozenset(i for group in seen.values() if len(group) > 1 for i in group)


def _generate_msl_kernel(
    kernel_name: str,
    body_stmts: list[ast.stmt],
    param_names: list[str],
    args: tuple[object, ...],
    fn_globals: dict[str, object],
    required_threads_per_threadgroup: tuple[int, int, int] | None = None,
    by_value: Mapping[str, str] = _NO_SCALARS,
) -> MSLKernel:
    """Generate a kernel's MSL from Python AST body and actual args.

    Infers Metal dtypes and scalar vs tensor distinction directly from
    the args, apart from the ``by_value`` parameters: runtime scalars, each
    mapped to the MSL type it binds as.  Reads block sizes from fn_globals.
    """
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
    body_stmts = defer_group_sums(body_stmts, param_names)
    # The threads each SIMD group lays the tiles out over when SIMD groups
    # split the fragment or register runs' K loops (``fragment_mma``,
    # ``mpp_registers``): the launch stacks one such layout per SIMD group
    # along z.
    split = _k_split(body_stmts)
    grid = _grid_runs(body_stmts)
    layout = required_threads_per_threadgroup
    if layout is not None and split > 1:
        tx, ty, tz = layout
        assert tz % split == 0, (layout, split)
        layout = (tx, ty, tz // split)
    body_stmts = elementize(
        body_stmts,
        tensor_dtypes,
        frozenset(
            k[len(TG_BUFFER_GLOBAL_PREFIX) :]
            for k in fn_globals
            if k.startswith(TG_BUFFER_GLOBAL_PREFIX)
        ),
        layout,
        FRAGMENT_LOWERING,
        REGISTER_LOWERING,
    )
    body_stmts = jam_lanes_into_loops(unroll_lane_folds(body_stmts, names_read))
    msl_parts: list[str] = [
        "#include <metal_stdlib>",
        # The c10 helpers the kernel uses, once the rest is known
        # (``c10_helpers``).
        "",
        "using namespace metal;",
    ]
    c10_at = 1
    reduction_at = preamble_at = None
    referenced = {
        node.id
        for stmt in body_stmts
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    }
    reduction_sites = _reduction_sites(fn_globals, referenced, layout)
    body_stmts, threadgroup_buffers = _threadgroup_buffers(
        body_stmts, fn_globals, referenced, reduction_sites, layout
    )
    tensor_types = {n: DTYPE_TO_METAL[d] for n, d in tensor_dtypes.items()}
    body_stmts = fuse_multiply_adds(
        body_stmts,
        {
            **tensor_types,
            **{n: dtype for n, dtype, _ in threadgroup_buffers},
            **by_value,
            **{
                n: "float" if type(a) is float else "int"
                for n, a in zip(param_names, args, strict=True)
                if n not in by_value and type(a) in (int, float)
            },
        },
    )
    shared = _shared_storage(args)
    body_stmts = branch_masked_steps(body_stmts)
    body_stmts = select_masked_loads(
        body_stmts,
        frozenset(
            name
            for i, (name, arg) in enumerate(zip(param_names, args, strict=True))
            if isinstance(arg, torch.Tensor) and arg.numel() > 0 and i not in shared
        ),
    )
    uses_reductions = _uses_reductions(body_stmts)
    # Each SIMD group that splits the K loops runs the kernel's other code
    # too, as the only one would, so codegen refuses kernels with reductions
    # or staged tiles (``operand_elements.split_refusal``): their threadgroup
    # memory would take writes from all of them.
    assert split == 1 or not (uses_reductions or threadgroup_buffers)
    if uses_reductions:
        reduction_at, preamble_at = len(msl_parts), len(msl_parts) + 2
        msl_parts.extend(("", "", REDUCTION_PREAMBLE))
    uses_mpp, uses_fragments, uses_registers = _matmul_kinds(body_stmts)
    if uses_fragments:
        msl_parts.append("#include <metal_simdgroup_matrix>")
    if uses_mpp:
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
    # A register run writes its bfloat operands' elements through BF16_CAST
    # (``msl_ast_walker``'s put).
    if ROUND in referenced or BF16_CAST in referenced or uses_registers:
        msl_parts.append(ROUND_PREAMBLE)
    if BF16_CAST in referenced or uses_registers:
        msl_parts.append(BF16_CAST_PREAMBLE)
    if any(
        isinstance(node, ast.Name) and node.id == MASKED_LOAD
        for stmt in body_stmts
        for node in ast.walk(stmt)
    ):
        msl_parts.append(MASKED_LOAD_PREAMBLE)
    msl_parts.append("")

    params: list[MSLParam] = []
    scalar_preamble: list[str] = []
    for name, arg in zip(param_names, args, strict=True):
        if name in by_value:
            params.append(MSLParam("constant", by_value[name], name, by_value=True))
            continue
        if not isinstance(arg, torch.Tensor):
            # A host-side constexpr the launcher passes positionally (e.g. a
            # rolled reduction's ``_REDUCTION_BLOCK_*``).  The launcher binds
            # only tensors and runtime scalars, so bake the value into the
            # shader.  The launch plan keys on these values, so a different
            # one recompiles rather than reusing a stale shader.
            scalar_preamble.append(f"    {_constexpr_decl(name, arg)}")
            continue
        # A 0-dim tensor too: its loads index element 0.
        params.append(
            MSLParam("device", DTYPE_TO_METAL[arg.dtype], name, by_value=False)
        )

    attributes = [("uint3", "tgid", "threadgroup_position_in_grid")]
    if split == 1:
        attributes.append(("uint3", "tid", "thread_position_in_threadgroup"))
    if uses_fragments:
        attributes.append(("uint", LANE, "thread_index_in_simdgroup"))
    if uses_registers:
        attributes.append(("uint", REGISTER_LANE, "thread_index_in_simdgroup"))
    if split > 1 or grid:
        # The same integer type as the other position attributes.
        attributes.append(("uint", SIMDGROUP, "simdgroup_index_in_threadgroup"))
    preamble = msl_parts
    prologue = list(scalar_preamble)

    # Declare _BLOCK_SIZE_* constants from module globals
    block_sizes = {
        name: int(val)
        for name, val in fn_globals.items()
        if name.startswith("_BLOCK_SIZE_") and isinstance(val, int)
    }
    for name in sorted(block_sizes):
        prologue.append(f"    constexpr int {name} = {block_sizes[name]};")
    if layout is not None:
        for name, size in zip(LAUNCH_DIM_VARS, layout, strict=True):
            prologue.append(f"    constexpr uint {name} = {size};")
    for site, (path, _) in sorted(reduction_sites.items()):
        prologue.append(f"    constexpr uint {site} = {GROUP_PATHS.index(path)};")
    if uses_fragments:
        prologue.extend(f"    {line}" for line in LANE_PREAMBLE)
    if split > 1:
        # Each SIMD group's lanes take the layout's threads, in launch order.
        assert layout is not None
        tx, ty, _ = layout
        lane = LANE if uses_fragments else REGISTER_LANE
        prologue.append(
            f"    const uint3 tid = uint3({lane} % {tx}, ({lane} / {tx}) % {ty}, "
            f"{lane} / {tx * ty});"
        )

    # ``threadgroup`` variables are only legal at kernel scope, but a reduction
    # can be emitted from inside a loop body, so their declarations are hoisted
    # to the kernel's scratch from the generated ``_METAL_TG_BUF_*`` module
    # globals.
    scratch = [(dtype, name, slots) for name, dtype, slots in threadgroup_buffers]

    state = EmitState(
        declared={
            *block_sizes,
            *LAUNCH_DIM_VARS,
            *reduction_sites,
            *(name for name, _, _ in threadgroup_buffers),
        },
        constants=(
            kernel_constants(body_stmts) if uses_fragments or uses_registers else {}
        ),
        split=split,
        # Register operands' layouts are probed where the kernel runs
        # (``coop_layout``).
        on_gpu=any(
            isinstance(arg, torch.Tensor) and arg.device.type == "mps" for arg in args
        ),
        tensor_types=tensor_types,
    )

    body: list[str] = []
    _emit_stmts(body_stmts, body, indent=4, state=state)
    # The grid runs' tiles share the threadgroup memory with the planned
    # buffers; the tiles' leading dimensions and the lane bases of the
    # register operands' layouts open the body.
    planned = sum(MSL_SCALAR_BYTES[d] * n for _, d, n in threadgroup_buffers)
    tiles, lds, used = grid_tiles(state.grid_runs, planned)
    if tiles and used > MAX_THREADGROUP_MEMORY_BYTES:
        raise exc.BackendUnsupported(
            "metal",
            f"MPP staging and grid runs' tiles need {used} bytes of threadgroup "
            f"memory (max {MAX_THREADGROUP_MEMORY_BYTES})",
        )
    scratch += [(t, f"{GRID_TILE}_{t}", n) for t, n in tiles.items()]
    decls = [
        *(f"constexpr int _mst{gid}_ld = {ld};" for gid, ld in lds.items()),
        *state.lane_decls,
    ]
    body[:0] = [f"    {line}" for line in decls]
    if state.split_slots:
        # ``fragment_mma._split`` and ``mpp_registers.register_split`` kept
        # it within the threadgroup memory.
        assert (
            state.split_slots * MSL_SCALAR_BYTES["float"]
            <= MAX_THREADGROUP_MEMORY_BYTES
        )
        scratch.append(("float", SPLIT_BUFFER, state.split_slots))
    slots = (c10_at, reduction_at, preamble_at)
    kernel = MSLKernel(
        name=kernel_name,
        preamble=tuple(p for i, p in enumerate(preamble) if i not in slots),
        tables=tuple(
            f"constant {ctype} {name}[{len(entries)}] = {{{', '.join(entries)}}};"
            for (ctype, entries), name in state.tables.items()
        ),
        params=tuple(params),
        attributes=tuple(attributes),
        threads=required_threads_per_threadgroup,
        prologue=tuple(prologue),
        scratch=tuple(scratch),
        body=tuple(body),
        device_views=frozenset(state.device_views),
    )
    c10_lines, reduction_lines = c10_source(kernel.render(), uses_reductions)
    filled: list[str | None] = [*preamble]
    filled[c10_at] = "\n".join(c10_lines) if c10_lines else None
    if reduction_at is not None:
        filled[reduction_at] = "\n".join(reduction_lines) if reduction_lines else None
    return dataclasses.replace(
        kernel, preamble=tuple(p for p in filled if p is not None)
    )


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
        return f"constexpr float {name} = {_float_literal(value)};"
    raise exc.BackendUnsupported("metal", f"kernel argument type: {type(value)}")


def _float_literal(value: float) -> str:
    """``value`` as an MSL float: ``repr`` but for infinities and NaN, which
    MSL spells with macros."""
    if value != value:
        return "NAN"
    if value in (float("inf"), float("-inf")):
        return "INFINITY" if value > 0 else "-INFINITY"
    return repr(value)


def _matmul_kinds(body_stmts: list[ast.stmt]) -> tuple[bool, bool, bool]:
    """Whether the body needs MPP's ``matmul2d``, whether it has fragment
    groups (``fragment_mma``), each of which declares its accumulator with a
    ``fragment`` keyword, and whether it has register runs
    (``mpp_registers``).  The planner lowers all of a kernel's groups one
    way, so MPP markers with no fragment group are ``matmul2d``'s."""
    markers = fragments = registers = False
    for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id.startswith("_helion_mpp_"):
            markers = True
            fragments |= func.id == ACC and is_fragment_acc(node)
            registers |= func.id == REG_RUN
    return markers and not fragments, fragments, registers


def _grid_runs(body_stmts: list[ast.stmt]) -> bool:
    """Whether the body has register runs whose SIMD groups split the tile
    (``mpp_registers._grid_run``), the ``grid`` keyword of their ``RUN``
    calls, as the body reads before elementize."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RUN
        and any(k.arg == "grid" for k in node.keywords)
        for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[]))
    )


def _k_split(body_stmts: list[ast.stmt]) -> int:
    """The SIMD groups that split the body's K loops, 1 when one SIMD group
    runs them: the ``split`` of each fragment run (``fragment_mma``) and the
    ``k_split`` of each register run (``mpp_registers``), both keywords of
    the run's ``RUN`` call, as the body reads before elementize."""
    splits = set()
    for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[])):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id != RUN:
                continue
            for keyword in node.keywords:
                if keyword.arg in ("split", "k_split"):
                    split = ast.literal_eval(keyword.value)
                    splits.add(1 if split is None else split[1])
    assert len(splits) <= 1, splits
    return splits.pop() if splits else 1


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
            MSL_SCALAR_BYTES[dtype] * slots(buf_slots)
            for _, dtype, buf_slots in declared
            if buf_slots == site
        )

    def buffers() -> list[tuple[str, str, int]]:
        return [(name, dtype, slots(buf_slots)) for name, dtype, buf_slots in declared]

    def used() -> int:
        return sum(MSL_SCALAR_BYTES[dtype] * n for _, dtype, n in buffers())

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
    if not torch.backends.mps.is_available():
        return
    device_name = torch._C._mps_get_name()  # type: ignore[attr-defined]
    if "paravirtual" in device_name.lower():
        raise exc.BackendUnsupported(
            "metal",
            "MPP matmul lowering is not supported on Apple paravirtual MPS "
            "devices; use a physical Apple GPU such as a self-hosted M-series "
            "runner",
        )
