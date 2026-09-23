"""Independent vector-lane sums and single-use replicated CTA reductions.

These are selectable source transformations, after vector reduction hoisting.
They preserve the per-element expression and its casts, but reassociate sums.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from ._ast_pass_utils import _assignment_lhs_name
from ._ast_pass_utils import _names_read

if TYPE_CHECKING:
    from collections.abc import Mapping


_AST = TypeVar("_AST", bound=ast.AST)


def _clone(node: _AST) -> _AST:
    fields: dict[str, object] = {}
    for name, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            fields[name] = _clone(value)
        elif isinstance(value, list):
            fields[name] = [
                _clone(item) if isinstance(item, ast.AST) else item for item in value
            ]
        else:
            fields[name] = value
    result = node.new(fields) if isinstance(node, ExtendedAST) else type(node)(**fields)
    assert isinstance(result, ast.AST)
    # Compiler annotations are outside Python's AST fields. Keep their identity,
    # including lane-loop markers consumed by later reduction passes.
    result.__dict__.update(
        {name: value for name, value in node.__dict__.items() if name not in fields}
    )
    return cast("_AST", ast.copy_location(result, node))


def _integer(node: ast.expr, constants: Mapping[str, object]) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        value = constants.get(node.id)
        return value if type(value) is int else None
    if (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("cutlass.Int32", "cutlass.Int64")
        and len(node.args) == 1
        and not node.keywords
    ):
        return _integer(node.args[0], constants)
    return None


def _range_size(loop: ast.For, constants: Mapping[str, object]) -> int | None:
    call = loop.iter
    if not (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) in ("range", "cutlass.range_constexpr")
        and 1 <= len(call.args) <= 3
        and not call.keywords
        and not loop.orelse
    ):
        return None
    args = [_integer(value, constants) for value in call.args]
    if any(value is None for value in args):
        return None
    start, stop, step = 0, args[0], 1
    if len(args) > 1:
        start, stop = args[:2]
    if len(args) > 2:
        step = args[2]
    assert start is not None and stop is not None and step is not None
    if step == 0:
        return None
    return len(range(start, stop, step))


def _float_zero(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cutlass.Float32"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and type(node.args[0].value) in (int, float)
        and node.args[0].value == 0
        and not node.keywords
        and not ast.unparse(node.args[0]).startswith("-")
    )


_SCALAR_TYPES = frozenset(
    f"cutlass.{name}"
    for name in (
        "Float16",
        "BFloat16",
        "Float32",
        "Int32",
        "Uint16",
        "Uint32",
    )
)
_VECTOR_LOADS = frozenset(
    {
        "cute.arch.load",
        "_cute_load_l2_evict_last",
        "_cute_load_l1_l2_evict_first",
        "_cute_load_l1_l2_evict_last",
        "_cute_load_l1_l2_evict_first_8b",
        "_cute_load_l1_l2_evict_last_8b",
    }
)


def _pure_scalar(node: ast.expr, scalars: set[str], vectors: set[str]) -> bool:
    """Closed scalar grammar; vector roots are actual register-load results."""
    if isinstance(node, ast.Name):
        return node.id in scalars
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, float)
    if isinstance(node, ast.BinOp):
        return (
            isinstance(
                node.op,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.FloorDiv,
                    ast.Mod,
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                    ast.LShift,
                    ast.RShift,
                ),
            )
            and _pure_scalar(node.left, scalars, vectors)
            and _pure_scalar(node.right, scalars, vectors)
        )
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)) and _pure_scalar(
            node.operand, scalars, vectors
        )
    if isinstance(node, ast.Subscript):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id in vectors
            and _pure_scalar(node.slice, scalars, set())
        )
    if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
        return False
    if ast.unparse(node.func) in _SCALAR_TYPES:
        return _pure_scalar(node.args[0], scalars, vectors)
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "bitcast"
        and ast.unparse(node.args[0]) in _SCALAR_TYPES
        and _pure_scalar(node.func.value, scalars, vectors)
    )


class _Substitute(ast.NodeTransformer):
    def __init__(self, replacements: Mapping[str, ast.expr]) -> None:
        self.replacements = replacements

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id not in self.replacements:
            return node
        replacement = _clone(self.replacements[node.id])
        if isinstance(replacement, ast.Name):
            replacement.ctx = _clone(node.ctx)
        return ast.copy_location(replacement, node)


class _PacketReductions:
    def __init__(
        self,
        body: list[ast.stmt],
        constants: Mapping[str, object],
        thread_block_dims: tuple[int, int, int] | None,
    ) -> None:
        self.constants = constants
        self.thread_block_dims = thread_block_dims
        self.names = {
            node.id
            for stmt in body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name)
        }
        self.counter = 0
        self.read_counts = Counter(
            node.id
            for stmt in body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )

    def fresh(self, suffix: str) -> str:
        while True:
            value = f"_helion_packet_{suffix}_{self.counter}"
            self.counter += 1
            if value not in self.names:
                self.names.add(value)
                return value

    def split(self, init: ast.stmt, outer: ast.For) -> list[ast.stmt] | None:
        accumulator = _assignment_lhs_name(init)
        if not (
            accumulator is not None
            and isinstance(init, ast.Assign)
            and _float_zero(init.value)
            and outer.body
            and isinstance(outer.body[-1], ast.For)
        ):
            return None
        inner = outer.body[-1]
        if not isinstance(inner.target, ast.Name) or not isinstance(
            outer.target, ast.Name
        ):
            return None
        if accumulator in (inner.target.id, outer.target.id):
            return None
        inner_reads = sum(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == inner.target.id
            for node in ast.walk(inner)
        )
        if self.read_counts[inner.target.id] != inner_reads:
            return None
        width = _range_size(inner, self.constants)
        outer_size = _range_size(outer, self.constants)
        if width not in (2, 4, 8) or outer_size is None or not 1 <= outer_size <= 16:
            return None
        if not (
            isinstance(inner.iter, ast.Call)
            and ast.unparse(inner.iter.func) == "cutlass.range_constexpr"
            and len(inner.iter.args) == 1
            and isinstance(outer.iter, ast.Call)
            and len(outer.iter.args) == 1
            and isinstance(outer.iter.args[0], ast.Constant)
            and all(_assignment_lhs_name(stmt) is not None for stmt in outer.body[:-1])
        ):
            return None
        prefix = ast.Module(body=outer.body[:-1], type_ignores=[])
        vectors: set[str] = set()
        for stmt in outer.body[:-1]:
            name = _assignment_lhs_name(stmt)
            assert name is not None and isinstance(stmt, ast.Assign)
            vectors.discard(name)
            if (
                isinstance(stmt.value, ast.Call)
                and ast.unparse(stmt.value.func) in _VECTOR_LOADS
            ):
                vectors.add(name)
        scalars = {accumulator, inner.target.id, outer.target.id}
        for stmt in inner.body:
            name = _assignment_lhs_name(stmt)
            if (
                name is None
                or not isinstance(stmt, ast.Assign)
                or not _pure_scalar(stmt.value, scalars, vectors)
            ):
                return None
            scalars.add(name)
            vectors.discard(name)
        if any(
            isinstance(node, ast.Name) and node.id == accumulator
            for node in ast.walk(prefix)
        ):
            return None
        writes = [
            stmt for stmt in inner.body if _assignment_lhs_name(stmt) == accumulator
        ]
        if len(writes) != 1 or writes[0] is not inner.body[-1]:
            return None
        update = writes[0]
        assert isinstance(update, ast.Assign)
        rhs = update.value
        if not (
            isinstance(rhs, ast.BinOp)
            and isinstance(rhs.op, ast.Add)
            and isinstance(rhs.left, ast.Name)
            and rhs.left.id == accumulator
            and accumulator not in _names_read(rhs.right)
            and all(accumulator not in _names_read(stmt) for stmt in inner.body[:-1])
        ):
            return None
        # No other loop-carried value may depend on the inner traversal order.
        local_names = {_assignment_lhs_name(stmt) for stmt in inner.body[:-1]}
        if _names_read(prefix) & (local_names | {inner.target.id}):
            return None
        if any(
            _assignment_lhs_name(stmt) == inner.target.id for stmt in outer.body[:-1]
        ):
            return None
        if inner.target.id in local_names or (
            isinstance(outer.target, ast.Name) and outer.target.id in local_names
        ):
            return None
        assigned: set[str] = set()
        for stmt in inner.body[:-1]:
            name = _assignment_lhs_name(stmt)
            assert name is not None
            if (_names_read(stmt) & local_names) - assigned:
                return None
            assigned.add(name)
        if (_names_read(rhs.right) & local_names) - assigned:
            return None
        partials = [self.fresh("sum") for _ in range(width)]
        result = [
            statement_from_string(f"{name} = cutlass.Float32(0)") for name in partials
        ]
        transformed = _clone(outer)
        transformed.iter = _clone(outer.iter)
        assert isinstance(transformed.iter, ast.Call)
        transformed.iter.func = ast.parse("cutlass.range_constexpr", mode="eval").body
        transformed.body = [_clone(stmt) for stmt in outer.body[:-1]]
        for lane, name in enumerate(partials):
            substitute = _Substitute(
                {
                    inner.target.id: ast.Constant(lane),
                    accumulator: ast.Name(id=name, ctx=ast.Load()),
                }
            )
            transformed.body.extend(
                substitute.visit(_clone(stmt)) for stmt in inner.body
            )
        result.append(transformed)
        # Butterfly halves give (0+2)+(1+3) for a four-lane vector.
        while len(partials) > 1:
            half = len(partials) // 2
            combined = []
            for index in range(half):
                name = self.fresh("join")
                result.append(
                    statement_from_string(
                        f"{name} = {partials[index]} + {partials[index + half]}"
                    )
                )
                combined.append(name)
            partials = combined
        result.append(statement_from_string(f"{accumulator} = {partials[0]}"))
        return result

    def replicate(
        self, stmt: ast.stmt, prefix: list[ast.stmt]
    ) -> list[ast.stmt] | None:
        target = _assignment_lhs_name(stmt)
        if target is None or not isinstance(stmt, ast.Assign):
            return None
        call = stmt.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_cute_grouped_reduce_shared_two_stage"
            and len(call.args) == 6
            and isinstance(call.args[1], ast.Constant)
            and call.args[1].value == "sum"
            and _float_zero(call.args[2])
            and self.thread_block_dims is not None
            and len(call.keywords) == 3
        ):
            return None
        kwargs = {kw.arg: _integer(kw.value, self.constants) for kw in call.keywords}
        span = kwargs.get("group_span")
        if not (
            set(kwargs) == {"pre", "group_span", "group_count"}
            and kwargs["pre"] == kwargs["group_count"] == 1
            and span in (64, 128, 256, 512, 1024)
            and self.thread_block_dims == (span, 1, 1)
        ):
            return None
        # The subgroup shuffle is over physical lanes. The old helper also
        # accepts logical lane coordinates, which need not preserve its new
        # subgroup partitions. Require the canonical, dominating full-CTA
        # coordinate assignments emitted by the reduction lowering.
        coordinates = call.args[3:]
        if len(prefix) < 3 or not all(isinstance(arg, ast.Name) for arg in coordinates):
            return None
        names = [ast.unparse(arg) for arg in coordinates]
        if len(set(names)) != 3:
            return None
        expected = ast.parse(
            f"{names[0]} = cutlass.Int32(cute.arch.thread_idx()[0])\n"
            f"{names[1]} = {names[0]} % {span}\n"
            f"{names[2]} = {names[1]} % 1\n"
        ).body
        if any(
            ast.dump(old) != ast.dump(new)
            for old, new in zip(prefix[-3:], expected, strict=True)
        ):
            return None
        warps = span // 32
        lane = ast.unparse(call.args[3])
        value = ast.unparse(call.args[0])
        partial, pointer, shared, reloaded = [
            self.fresh(s) for s in ("warp", "ptr", "smem", "reload")
        ]
        return ast.parse(
            f"{partial} = cute.arch.warp_reduction_sum({value}, threads_in_group=32)\n"
            f"{pointer} = cute.arch.alloc_smem(cutlass.Float32, {warps})\n"
            f"{shared} = cute.make_tensor({pointer}, ({warps},))\n"
            f"if ({lane}) % 32 == 0:\n"
            f"    {shared}[({lane}) // 32] = {partial}\n"
            "cute.arch.sync_threads()\n"
            f"{reloaded} = {shared}[({lane}) % {warps}]\n"
            f"{target} = cute.arch.warp_reduction_sum({reloaded}, threads_in_group={warps})\n"
        ).body

    def unroll(self, loop: ast.For) -> None:
        size = _range_size(loop, self.constants)
        if (
            size is None
            or not 1 <= size <= 16
            or not isinstance(loop.iter, ast.Call)
            or ast.unparse(loop.iter.func) != "range"
        ):
            return
        if not any(
            isinstance(stmt, ast.For)
            and isinstance(stmt.iter, ast.Call)
            and ast.unparse(stmt.iter.func) == "cutlass.range_constexpr"
            for stmt in loop.body
        ):
            return
        # Only straight-line vector load/compute/store packet loops. In
        # particular, do not unroll scalar MMA or synchronization protocols.
        calls = (
            _VECTOR_LOADS
            | _SCALAR_TYPES
            | {
                "ir.VectorType.get",
                "cute.arch.thread_idx",
                "cutlass.range_constexpr",
                "_cute_store_u16_vec",
                "_cute_store_u32_vec",
            }
        )
        for stmt in loop.body:
            for node in ast.walk(stmt):
                if isinstance(
                    node, (ast.If, ast.While, ast.Break, ast.Continue, ast.Return)
                ):
                    return
                if isinstance(node, ast.Call) and ast.unparse(node.func) not in calls:
                    if not isinstance(
                        node.func, ast.Attribute
                    ) or node.func.attr not in ("bitcast", "append"):
                        return
        loop.iter.func = ast.parse("cutlass.range_constexpr", mode="eval").body

    def block(
        self,
        body: list[ast.stmt],
        *,
        once: bool,
        split: bool,
        replicate: bool,
        unroll: bool,
    ) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for original in body:
            stmt = _clone(original)
            if isinstance(stmt, ast.For):
                if split and result:
                    changed = self.split(result[-1], stmt)
                    if changed is not None:
                        result.pop()
                        result.extend(changed)
                        continue
                stmt.body = self.block(
                    stmt.body,
                    once=once and _range_size(stmt, self.constants) == 1,
                    split=split,
                    replicate=replicate,
                    unroll=unroll,
                )
                if unroll:
                    self.unroll(stmt)
            # Branches/while loops are deliberately not entered. A new CTA
            # barrier needs convergence and a proof of no shared-slot reuse.
            if once and replicate:
                changed = self.replicate(stmt, result)
                if changed is not None:
                    result.extend(changed)
                    continue
            result.append(stmt)
        return result


def optimize_vector_reductions(
    body: list[ast.stmt],
    constants: Mapping[str, object],
    *,
    thread_block_dims: tuple[int, int, int] | None,
    independent_accumulators: bool,
    replicated_single_use: bool,
    unroll_packets: bool = False,
) -> list[ast.stmt]:
    if (
        not independent_accumulators
        and not replicated_single_use
        and not unroll_packets
    ):
        return body
    result = _PacketReductions(body, constants, thread_block_dims).block(
        body,
        once=True,
        split=independent_accumulators,
        replicate=replicated_single_use,
        unroll=unroll_packets,
    )
    for stmt in result:
        ast.fix_missing_locations(stmt)
    return result
