"""CUDA's device assertion for a scalar predicate computed by a CTA."""

from __future__ import annotations

import ast
import math
import os.path
from typing import TYPE_CHECKING
from typing import cast

import sympy
from sympy.core.relational import Relational
import torch
from torch.fx import Node
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.utils._sympy.value_ranges import SymPyValueRangeAnalysis
from torch.utils._sympy.value_ranges import ValueRanges

from ... import exc
from ...language import _tracing_ops
from ...language import tile_ops
from ..ast_extension import statement_from_string
from ..aten_lowering import Lowering
from ..aten_lowering import LoweringContext
from ..compile_environment import CompileEnvironment
from ..device_ir import DeviceIR
from ..device_ir import codegen_node_order
from ..host_function import HostFunction
from ..program_id import L2GroupingProgramIDs
from ..program_id import XYZProgramIDs
from ..variable_origin import GridOrigin
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileIdOrigin
from ._ast_pass_utils import call_path_str
from .bounded_gather import integer_bounds
from .direct_affine_plan import DIRECT_AFFINE_ORDINARY_SCHEDULE
from .fragment_slab_loads import capture_map
from .fragment_slab_loads import resolve_capture
from .state import CuteSearchFacts
from .target_facts import validate_grid_yz

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import Config
    from .fragment_context import Fragment
    from .fragment_context import FragmentContext


class ScalarFragmentAssertionLowering(Lowering):
    requirement = "torch._assert_async requires a computed scalar Boolean predicate in a CTA fragment"

    def __init__(self, node: Node) -> None:
        # Validate before selecting a schedule: a declined fragment could
        # otherwise report an unrelated unsupported operation first.
        predicate, message = node.args
        if isinstance(predicate, Node) and (
            predicate.op == "placeholder" or predicate.target is _tracing_ops._new_var
        ):
            # A branch captures the value from its parent. Follow only copies
            # and branch captures, not loop carries that may have been updated.
            predicate = resolve_capture(
                predicate, capture_map(DeviceIR.current().graphs)
            )
        if (
            not isinstance(predicate, Node)
            or predicate.target is _tracing_ops._host_tensor
        ):
            raise exc.BackendUnsupported("cute", self.requirement)
        value = predicate.meta.get("val")
        if not (
            isinstance(value, torch.Tensor)
            and value.ndim == 0
            and value.dtype is torch.bool
            and isinstance(message, str)
        ):
            raise exc.BackendUnsupported("cute", self.requirement)

        if len(message.encode()) >= 255:
            raise exc.InvalidAPIUsage(
                "CUDA assertion message must be shorter than 255 bytes"
            )
        loc = node.meta.get("location")
        self.program = _assertion_asm(
            message,
            os.path.basename(loc.filename) if loc else "helion",
            (loc.lineno or 1) if loc else 1,
        )

    def emit(self, ctx: FragmentContext, predicate: Fragment) -> None:
        # The owner may read another lane's shared predicate. Complete prior
        # CTA stores (including peer writes) before a trap. An immediately
        # preceding unconditional barrier already orders both.
        body = ctx.cg.statements_stack[-1]
        if not (
            body
            and isinstance(last := body[-1], ast.Expr)
            and isinstance(last.value, ast.Call)
            and call_path_str(last.value.func) == "cute.arch.sync_threads"
        ):
            ctx.synchronize()
        ctx.cg.add_statement(
            statement_from_string(
                f"if {ctx.thread} == 0:\n    _cute_inline_asm_elementwise("
                f"(cutlass.Int32({ctx.scalar(predicate)}),), asm={{program}}, "
                "constraints={constraint}, dtype=cutlass.Int32, is_pure=False)",
                program=ast.Constant(value=self.program),
                constraint=ast.Constant(value="=r,r,~{memory}"),
            )
        )
        # Other lanes must not overwrite a shared predicate during the read.
        ctx.synchronize()

    def codegen(self, ctx: LoweringContext, node: Node) -> object:
        # A lane-only schedule cannot safely issue this effect once per CTA.
        raise exc.BackendUnsupported("cute", self.requirement)


def requires_assertion_owner(spec: ConfigSpec | None = None) -> bool:
    if spec is None:
        spec = CompileEnvironment.current().config_spec
    facts = spec.backend_facts
    return isinstance(facts, CuteSearchFacts) and facts.has_scalar_fragment_assertion


def check_root_widths(block_ids: list[int], config: Config) -> None:
    # The thread planner searches divisors of the physical tile width.
    # The computed-fragment owner checks the logical grid stride separately.
    env = CompileEnvironment.current()
    for bid in block_ids:
        size = env.block_sizes[bid].from_config(config)
        if isinstance(size, torch.SymInt):
            expr = size._sympy_()
            if isinstance(expr, sympy.Expr):
                expr = env.specialize_expr(expr)
            size = int(expr) if isinstance(expr, sympy.Integer) else None
        if type(size) is not int or not 0 < size < 2**31:
            raise exc.BackendUnsupported(
                "cute",
                "a device assertion requires a proved positive Int32 root tile width",
            )


def checked_range_domain(
    ctx: FragmentContext,
    begin: object,
    end: object,
    step: object,
    *,
    root: bool = False,
) -> range:
    """Bound the induction values of a representable checked Int32 axis."""

    def constant(value: object) -> int | None:
        if isinstance(value, torch.SymInt):
            value = value._sympy_()
        if isinstance(value, sympy.Expr):
            expr = ctx.configured_expr(value)
            value = (
                ctx.env.specialize_expr(expr) if isinstance(expr, sympy.Expr) else expr
            )
        if type(value) is int or isinstance(value, sympy.Integer):
            return int(value)
        return None

    start, stop, stride = map(constant, (begin, end, step))
    limit = (1 << 31) - 1
    if (
        not root
        and stride is not None
        and 0 < stride <= limit
        and isinstance(begin, Node)
        and begin.target is tile_ops.tile_begin
        and isinstance(end, Node)
        and end.target is tile_ops.tile_end
        and begin.args == end.args
    ):
        # Both ends of this tile are bounded by its enclosing grid interval.
        outer = ctx.env.resolve_block_id(begin.meta.get("val"))
        if (
            outer is not None
            and (enclosing := ctx.checked_domains.get(ctx.offsets.get(outer, "")))
            is not None
        ):
            low, high = enclosing.start, enclosing.stop
            if 0 <= low <= high and high + stride - 1 <= limit:
                # Each outer tile can start at a different stride phase. Its
                # envelope bounds every inner coordinate, including tail tiles.
                return range(low, high)
    if (
        start is not None
        and stop is not None
        and stride is not None
        and -(1 << 31) <= start <= limit
        and -(1 << 31) <= stop <= limit
        and 0 < stride <= limit
    ):
        distance = max(0, stop - start)
        trips = (distance + stride - 1) // stride
        # The grid computes ceildiv and tile origins in Int32; an inner loop
        # needs only the absolute induction, including the first excluded value.
        if (
            start + trips * stride <= limit
            and distance + stride - 1 <= limit
            and (not root or 0 <= start <= stop)
        ):
            return range(start, stop, stride)
    raise exc.BackendUnsupported(
        "cute", "a device assertion requires a proved positive-step Int32 loop range"
    )


def check_grid_count(ctx: FragmentContext, count: int) -> None:
    """Flat PID products, grouped decoding and persistent end updates use Int32."""
    config = ctx.df.config
    pid = ctx.df.pid
    limit = (1 << 31) - 1
    if isinstance(pid, L2GroupingProgramIDs):
        group_size = pid.group_size
        pid = pid.parent_strategy  # Grouping changes the decoder, not the launch.
        assert pid is not None
        # The decoder computes num_pid_in_group = group_size * num_pid_n.
        n = pid.pid_info[1].block_id
        if group_size * len(ctx.checked_domains[ctx.offsets[n]]) > limit:
            raise exc.InvalidConfig("device assertion grouped grid count exceeds Int32")
    if isinstance(pid, XYZProgramIDs):
        # Checked domains also cover dimensions specialized during codegen.
        validate_grid_yz(
            len(ctx.checked_domains[ctx.offsets[axis.block_id]])
            for axis in pid.pid_info[1:]
        )
    workers = 1
    if config.pid_type.startswith("persistent"):
        num_sm = ctx.env.config_spec.num_sm
        if num_sm <= 1:
            raise exc.InvalidConfig(
                "a persistent device assertion requires a known multi-SM target"
            )
        workers = num_sm * config.num_sm_multiplier
    if workers <= 0 or count + workers - 1 > limit:
        raise exc.InvalidConfig(
            "device assertion grid count is not proved to fit Int32"
        )


def _grid_value_bounds(
    ctx: FragmentContext, expr: sympy.Basic
) -> tuple[ValueRanges[sympy.Expr], ValueRanges[sympy.Expr]] | None:
    """Return the result range and the range needed by every intermediate."""
    if isinstance(expr, sympy.Integer):
        value = ValueRanges.wrap(expr)
        return value, value
    if isinstance(expr, sympy.Symbol) and expr not in ctx.sym_indices:
        info = HostFunction.current().expr_to_origin.get(expr)
        if info is not None and type(info.origin) is GridOrigin:
            domain = ctx.checked_domains.get(ctx.offsets.get(info.origin.block_id, ""))
            if domain:
                low, high = sorted((domain[0], domain[-1]))
                value = ValueRanges(low, high)
                return value, value
    if not isinstance(expr, (sympy.Add, sympy.Mul)):
        return None
    add = isinstance(expr, sympy.Add)
    # Both interval methods also accept Boolean domains; our operands are integers.
    op = cast(
        "Callable[[ValueRanges[sympy.Expr], ValueRanges[sympy.Expr]], ValueRanges[sympy.Expr]]",
        SymPyValueRangeAnalysis.add if add else SymPyValueRangeAnalysis.mul,
    )
    identity = ValueRanges.wrap(sympy.Integer(0 if add else 1))
    result: ValueRanges[sympy.Expr] = identity
    partial: ValueRanges[sympy.Expr] = identity
    intermediates = identity
    for arg in expr.args:
        ranges = _grid_value_bounds(ctx, arg)
        if ranges is None:
            return None
        term, child_intermediates = ranges
        result = op(result, term)
        # The printer can reorder an n-ary sum/product. Including the identity
        # bounds every subset of its operands, hence every partial evaluation.
        partial = op(partial, term | identity)
        intermediates = intermediates | child_intermediates
    return result, intermediates | partial


def checked_i32_grid_bounds(ctx: FragmentContext, node: Node) -> tuple[int, int] | None:
    value = node.meta.get("val")
    if isinstance(value, torch.SymInt):
        ranges = _grid_value_bounds(ctx, value._sympy_())
        if ranges is not None and ranges[1].issubset(
            ValueRanges(-(1 << 31), (1 << 31) - 1)
        ):
            return int(ranges[0].lower), int(ranges[0].upper)
    return None


def _integer_assignment_bounds(dtype: torch.dtype) -> tuple[int, int] | None:
    if dtype == torch.bool or dtype.is_complex:
        return None
    if dtype.is_floating_point:
        limits = torch.finfo(dtype)
        return math.ceil(limits.min), math.floor(limits.max)
    limits = torch.iinfo(dtype)
    # C10 permits negative Python integers down to -max for unsigned scalars.
    return limits.min or -limits.max, limits.max


def check_literal_assignment(value: float, dtype: torch.dtype) -> None:
    """CUDA scalar assignment checks the value before rounding to the destination."""
    if dtype == torch.bool or dtype.is_complex:
        return
    if isinstance(value, int):
        bounds = _integer_assignment_bounds(dtype)
        assert bounds is not None
        fits = bounds[0] <= value <= bounds[1]
    elif dtype.is_floating_point:
        # CUDA permits infinity only for formats that encode infinity.
        if math.isnan(value) or (
            math.isinf(value)
            and dtype
            in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
                torch.float8_e5m2,
            )
        ):
            return
        limits = torch.finfo(dtype)
        fits = limits.min <= value <= limits.max
    else:
        limits = torch.iinfo(dtype)
        # Floating scalars truncate towards zero after the range check.
        fits = limits.min <= value < limits.max + 1
    if not fits:
        raise exc.BackendUnsupported(
            "cute", f"a checked scalar assignment overflows {dtype}"
        )


def checked_scalar_wide(
    ctx: FragmentContext,
    expr: sympy.Basic,
    tensor_dtypes: tuple[torch.dtype, ...] = (),
    *,
    assignment_dtype: torch.dtype | None = None,
) -> bool:
    """Python value arithmetic must fit before any store or tensor promotion."""
    narrow_dtypes = [
        dtype
        for dtype in tensor_dtypes
        if assignment_dtype is None
        and dtype
        in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    ]
    int64_bounds = ValueRanges(-(1 << 63), (1 << 63) - 1)
    bounds = (
        _integer_assignment_bounds(assignment_dtype)
        if assignment_dtype is not None
        else None
    )
    assignment_range = ValueRanges(*bounds) if bounds is not None else int64_bounds
    if (
        isinstance(expr, sympy.Symbol)
        and not narrow_dtypes
        and int64_bounds.issubset(assignment_range)
    ):
        return False  # No intermediate computation or narrowing conversion.
    ranges = _grid_value_bounds(ctx, expr)
    if ranges is None or not ranges[1].issubset(int64_bounds):
        raise exc.BackendUnsupported(
            "cute", "a checked symbolic integer expression not proved to fit Int64"
        )
    result, intermediates = ranges
    if not result.issubset(assignment_range):
        raise exc.BackendUnsupported(
            "cute", f"a checked scalar assignment not proved to fit {assignment_dtype}"
        )
    for dtype in narrow_dtypes:
        # A wider comparison can disagree with conversion to the tensor dtype.
        limits = torch.iinfo(dtype)
        if not result.issubset(ValueRanges(limits.min, limits.max)):
            raise exc.BackendUnsupported(
                "cute",
                f"a checked symbolic integer not proved representable in the narrow integer tensor dtype {dtype}",
            )
    return not intermediates.issubset(ValueRanges(-(1 << 31), (1 << 31) - 1))


def checked_predicate_wide(ctx: FragmentContext, expr: sympy.Basic) -> bool:
    """A Bool comparison must preserve the arithmetic on each side."""
    if isinstance(expr, Relational) and all(
        getattr(arg, "is_integer", None) is True for arg in expr.args
    ):
        # Validate both operands even when the first already needs Int64.
        return checked_scalar_wide(ctx, expr.lhs) | checked_scalar_wide(ctx, expr.rhs)
    if isinstance(expr, (sympy.And, sympy.Or, sympy.Not)):
        wide = False
        for arg in expr.args:
            wide |= checked_predicate_wide(ctx, arg)
        return wide
    if expr in (sympy.true, sympy.false) or isinstance(expr, sympy.Symbol):
        return False
    raise exc.BackendUnsupported(
        "cute", "a checked symbolic Boolean with unproved operand arithmetic"
    )


def check_host_subscripts(
    ctx: FragmentContext,
    indices: list[object],
    shape: torch.Size,
    *,
    unmasked_read: bool,
) -> None:
    """A checked unmasked read cannot replace clipping or an invalid index with zero."""
    for axis, index in enumerate(i for i in indices if i is not None):
        size = shape[axis]
        value = index.meta.get("val") if isinstance(index, Node) else index
        scalar: tuple[int, int] | None = None
        if type(index) is int:
            # Only literal indices take the ordinary Python normalization path.
            if index < 0 and statically_known_true(-size <= index):
                continue
            scalar = index, index
        elif isinstance(value, torch.SymInt):
            expr = value._sympy_()
            block = ctx.env.resolve_block_id(value)
            if block is not None and expr == ctx.env.block_sizes[block].var._sympy_():
                domain = ctx.checked_domains.get(ctx.offsets[block])
                if domain is None or domain.start < 0:
                    raise exc.BackendUnsupported(
                        "cute",
                        "a device assertion with a tile origin not proved nonnegative",
                    )
                if (
                    unmasked_read
                    and domain
                    and not statically_known_true(domain.stop <= size)
                ):
                    # A slice can be shorter than its logical tile. Padded
                    # zero lanes would then participate in a checked reduction.
                    raise exc.BackendUnsupported(
                        "cute",
                        "a device assertion with an unmasked tile not proved inside its tensor dimension",
                    )
                continue
            if isinstance(expr, sympy.Integer):
                scalar = int(expr), int(expr)
            elif isinstance(expr, sympy.Symbol) and expr not in ctx.sym_indices:
                info = HostFunction.current().expr_to_origin.get(expr)
                origin = info.origin if info is not None else None
                if type(origin) is GridOrigin or isinstance(
                    origin, (TileBeginOrigin, TileIdOrigin)
                ):
                    domain = ctx.checked_domains.get(
                        ctx.offsets.get(origin.block_id, "")
                    )
                    if domain is not None:
                        if not domain:
                            continue  # No indexed access can execute.
                        low, high = domain[0], domain[-1]
                        if isinstance(origin, TileIdOrigin):
                            width = ctx.df.resolved_block_size(origin.block_id)
                            if type(width) is not int or width <= 0:
                                raise exc.BackendUnsupported(
                                    "cute",
                                    "a device assertion requires a known tile width",
                                )
                            low, high = low // width, high // width
                        scalar = low, high
        elif isinstance(value, int) and not isinstance(value, bool):
            scalar = value, value
        elif isinstance(value, torch.Tensor) and value.ndim != 0:
            if unmasked_read:
                bounds = integer_bounds(
                    index,
                    captured_bounds={},
                    symint_bounds=lambda n: checked_i32_grid_bounds(ctx, n),
                )
                if (
                    bounds is None
                    or bounds[0] < 0
                    or not statically_known_true(bounds[1] < size)
                ):
                    raise exc.BackendUnsupported(
                        "cute",
                        "a device assertion with an unmasked tensor index not proved in bounds",
                    )
            continue
        elif not isinstance(value, (torch.Tensor, int)):
            continue  # Slice.
        if (
            scalar is None
            or scalar[0] < 0
            or (unmasked_read and not statically_known_true(scalar[1] < size))
        ):
            raise exc.BackendUnsupported(
                "cute",
                "a device assertion with a scalar memory index not proved in bounds",
            )


def conflicting_fragment_schedule(spec: ConfigSpec, config: Config) -> str | None:
    """Options for which codegen_computed_fragment_root cannot own a root."""
    from .tile_packing import packed_grid_block

    # Packed subgroups have different logical tile IDs, but this owner
    # evaluates one assertion on lane zero for the entire CTA.
    if (packed := packed_grid_block(spec, config)) is not None:
        return f"cute_tiles_per_block={packed[1]}"
    for key, ordinary in (
        ("cute_affine_scan_schedule", DIRECT_AFFINE_ORDINARY_SCHEDULE),
        ("cute_reduction_schedule", "scalar"),
        ("cute_collective_mma", False),
        ("cute_register_chain", False),
    ):
        value = config.get(key, ordinary)
        if value != ordinary:
            return f"{key}={value}"
    return None


def validate_assertion_graphs(ir: DeviceIR) -> None:
    """The fragment emitter owns a single root and its control-flow children."""

    reachable = {
        node
        for _, node in codegen_node_order(
            ir.graphs, ir.root_ids, repeat_conditions=False, unreached=False
        )
        if node.target is torch.ops.aten._assert_async.msg
    }
    if (
        len(ir.root_ids) != 1
        or not reachable
        or any(
            node.target is torch.ops.aten._assert_async.msg and node not in reachable
            for info in ir.graphs
            for node in info.graph.nodes
        )
    ):
        raise exc.BackendUnsupported(
            "cute", "torch._assert_async requires a single complete CTA fragment root"
        )


def _assertion_asm(message: str, filename: str, lineno: int) -> str:
    # Encode user text as data rather than interpolating it into PTX.
    def c_string(name: str, value: str) -> str:
        data = value.encode() + b"\0"
        octets = ",".join(map(str, data))
        return f".global .align 1 .b8 {name}[{len(data)}] = {{{octets}}};"

    # LLVM substitutes a unique ID at each inline site, including unrolled
    # copies. __assertfail reports the message and CUDA_ERROR_ASSERT (710).
    msg, file, op = (f"hl_{name}_${{:uid}}" for name in ("message", "file", "op"))
    return f"""{{
    .extern .func __assertfail(.param .b64 x0, .param .b64 x1, .param .b32 x2, .param .b64 x3, .param .b64 x4);
    {c_string(msg, message)}
    {c_string(file, filename)}
    {c_string(op, "torch._assert_async")}
    .reg .pred good;
    .reg .b64 m, f, o;
    mov.u32 $0, $1;
    setp.ne.u32 good, $1, 0;
    @good bra DONE;
    mov.u64 m, {msg}; cvta.global.u64 m, m;
    mov.u64 f, {file}; cvta.global.u64 f, f;
    mov.u64 o, {op}; cvta.global.u64 o, o;
    {{ .param .b64 p0, p1, p3, p4; .param .b32 p2;
       st.param.b64 [p0], m; st.param.b64 [p1], f; st.param.b32 [p2], {lineno};
       st.param.b64 [p3], o; st.param.b64 [p4], 1;
       call __assertfail, (p0, p1, p2, p3, p4);
    }}
    DONE:
}}"""
