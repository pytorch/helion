"""Materialize pure reused matrix operands into an ordered CuTe launch.

The first envelope owns a two-axis output grid, one exclusive FP32 contraction
loop, and one complete fresh output. Pointwise RHS recipes may interleave a
fixed number of physical rows. Their operations and casts are copied unchanged;
codegen preserves separate FP32 products unless fast math is enabled. The
consumer sees a fresh dense tensor and an ordinary logical K tile loop.

The accumulator may be seeded before the contraction by additive pointwise
terms over the output tile (``acc += 2 * grad[m, n] * s[m, n]``); the consumer
adds those terms after the contraction instead, because the native K-loop
pipelines start from a zero accumulator.  The direct operand may be a
transposed input tile (``x[k, m].T``); the consumer keeps the transposed read.

A temporary compiler proves the original program. Shape specialization inserted
in that proof is also inserted into the owning compiler, so sample sizes never
stand in for runtime guards. Rejected programs retain their original source.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import logging
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ... import language as hl
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language.matmul_ops import dot
from ..ast_extension import ExtendedAST
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..kernel_compiler import KernelCompiler
from ..type_info import TensorType
from ..type_info import TileIndexType
from ..variable_origin import ArgumentOrigin
from .collective_matmul import _loop_entry
from .cute_mma import _mma_loop_is_exclusive
from .cute_mma_dag import match_mma_dag
from .full_slice_matmul import _bound_names
from .full_slice_matmul import _global_reference
from .full_slice_matmul import _global_value
from .full_slice_matmul import _pure_host_prelude
from .materialized_fission import _ANALYSIS_CALLS
from .materialized_fission import _ANALYSIS_METHODS
from .materialized_fission import MaterializedFissionPlan
from .materialized_fission import _clone_untyped
from .materialized_fission import _exact_tile
from .materialized_fission import _Ineligible
from .materialized_fission import _is_contiguous_2d
from .materialized_fission import _require
from .matmul_utils import strip_new_vars
from .packed_matmul import interleaved_matrix_terms
from .packed_matmul import packed_matmul_axis
from .promote_output_axis import _access_tensor
from .promote_output_axis import _fresh_tensors
from .promote_output_axis import _indices
from .promote_output_axis import _ordinary_operation
from .signed_bitfield import _CASTS
from .signed_bitfield import _cast_input
from .signed_bitfield import _integer_field

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from torch.fx import Node

    from ...runtime.kernel import Kernel
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction


log = logging.getLogger(__name__)
_PURE_CALLS = (*_ANALYSIS_CALLS, hl.register_block_size, torch.stack)
# Per-CTA tcgen05 MMA tile limits (one ``tcgen05.mma`` instruction covers at
# most 128 accumulator rows per CTA and 256 columns).  A fused kernel derives
# the recipe once per output tile along the axis the recipe does not depend
# on, so these bound how often it recomputes the operand: see
# ``_fused_recompute_pays``.
_TCGEN05_CTA_TILE_M = 128
_TCGEN05_CTA_TILE_N = 256
_TRANSPOSES = (torch.ops.aten.permute.default, torch.ops.aten.t.default)


@dataclass
class _Source:
    index: int
    root: ast.For
    loop: ast.For
    accumulator: str
    m: str
    n: str
    k: str
    output: str
    lhs: ast.expr
    rhs: ast.expr
    definitions: dict[str, ast.Assign]
    call: ast.Call
    # Additive rebindings of the accumulator between its zero initialization
    # and the contraction loop, in program order (``_additive_seed``).
    seeds: tuple[ast.stmt, ...]


@dataclass(frozen=True)
class _Proof:
    factor: int
    dtype: torch.dtype
    direct_name: str
    expressions: tuple[ast.expr, ...]
    captures: tuple[str, ...]
    side: str
    # The direct operand is read as ``name[...].T``.
    transposed: bool


def _callee(node: ast.AST, host: HostFunction) -> object:
    return _global_value(node, host, _bound_names(host))


def _reference(value: object, host: HostFunction) -> ast.expr:
    reference = _global_reference(value, host, _bound_names(host))
    _require(reference is not None, "required stable language global unavailable")
    assert reference is not None
    # Fill in ExtendedAST context nodes as well as the Name/Attribute nodes.
    return cast("ast.expr", expr_from_string(ast.unparse(reference)))


def _safe_source(host: HostFunction) -> bool:
    """Do not retrace opaque Python helpers or host-side tensor effects."""
    locals_ = _bound_names(host)
    if not _pure_host_prelude(host, locals_, allow_block_size_registration=True):
        return False

    def global_value(node: ast.AST) -> object:
        return _global_value(node, host, locals_)

    def known(value: object) -> bool:
        return (
            value is torch
            or value is hl
            or type(value) in (int, float, bool, str, type(None), torch.dtype)
            or any(value is candidate for candidate in _PURE_CALLS)
        )

    def local(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in locals_
        if isinstance(node, ast.Subscript):
            return local(node.value)
        if isinstance(node, ast.BinOp):
            return local(node.left) or local(node.right)
        if isinstance(node, ast.UnaryOp):
            return local(node.operand)
        if isinstance(node, ast.Call):
            return any(global_value(node.func) is call for call in _PURE_CALLS) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _ANALYSIS_METHODS
                and local(node.func.value)
            )
        return False

    for statement in host.body:
        for node in ast.walk(statement):
            if isinstance(
                node,
                (
                    ast.AnnAssign,
                    ast.Delete,
                    ast.If,
                    ast.While,
                    ast.With,
                    ast.Try,
                    ast.FunctionDef,
                    ast.Lambda,
                    ast.ListComp,
                    ast.GeneratorExp,
                    ast.NamedExpr,
                ),
            ):
                return False
            # ``name op= value`` rebinds a local; a subscript or attribute
            # target would be an in-place tensor write.  ``_source`` admits
            # the rebinding only as a pre-contraction accumulator seed.
            if isinstance(node, ast.AugAssign) and not isinstance(
                node.target, ast.Name
            ):
                return False
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) for target in node.targets
            ):
                return False
            if isinstance(node, ast.Call):
                if any(global_value(node.func) is call for call in _PURE_CALLS):
                    continue
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in _ANALYSIS_METHODS
                    and local(node.func.value)
                ):
                    continue
                return False
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in locals_
                and not known(global_value(node))
            ):
                return False
            if (
                isinstance(node, ast.Attribute)
                and not local(node.value)
                and not known(global_value(node))
            ):
                return False
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id in host.params.arguments
            ):
                return False
    return True


def _additive_seed(statement: ast.stmt, accumulator: str, k: str) -> bool:
    """``acc += e``, ``acc -= e``, ``acc = acc + e``, ``acc = e + acc`` or
    ``acc = acc - e`` with ``e`` free of the accumulator and the K loop variable.

    The accumulator starts at zero, so such a statement makes the value
    entering the contraction ``0 + e`` (``0 - e``); the same statement after
    the contraction adds ``e`` to the finished sum instead.  Any other
    rebinding (``acc = acc * 2 + e``, ``acc = e - acc``, ...) would need the
    accumulator's value at that point and is refused.
    """
    if isinstance(statement, ast.AugAssign):
        target: ast.expr = statement.target
        additive = isinstance(statement.op, (ast.Add, ast.Sub))
        operand = statement.value
    elif (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.value, ast.BinOp)
    ):
        target = statement.targets[0]
        value = statement.value
        if isinstance(value.left, ast.Name) and value.left.id == accumulator:
            additive = isinstance(value.op, (ast.Add, ast.Sub))
            operand = value.right
        elif isinstance(value.right, ast.Name) and value.right.id == accumulator:
            additive = isinstance(value.op, ast.Add)
            operand = value.left
        else:
            return False
    else:
        return False
    return (
        additive
        and isinstance(target, ast.Name)
        and target.id == accumulator
        and not any(
            isinstance(node, ast.Name) and node.id in (accumulator, k)
            for node in ast.walk(operand)
        )
    )


def _source(host: HostFunction) -> _Source:
    roots = [
        (index, node)
        for index, node in enumerate(host.body)
        if isinstance(node, ast.For)
    ]
    _require(len(roots) == 1, "requires one output grid")
    index, root = roots[0]
    _require(
        isinstance(root.iter, ast.Call)
        and _callee(root.iter.func, host) is hl.tile
        and len(root.iter.args) == 1
        and isinstance(root.iter.args[0], (ast.Tuple, ast.List))
        and len(root.iter.args[0].elts) == 2
        and not root.iter.keywords
        and isinstance(root.target, (ast.Tuple, ast.List))
        and len(root.target.elts) == 2
        and all(isinstance(item, ast.Name) for item in root.target.elts)
        and not root.orelse
        and len(root.body) >= 3,
        "requires a canonical M,N tile and one contraction",
    )
    assert isinstance(root.target, (ast.Tuple, ast.List))
    m, n = (cast("ast.Name", item).id for item in root.target.elts)
    # ``init, seed*, K loop, store``: the seeds are checked once the
    # accumulator and K names are known.
    loops = [
        position
        for position, statement in enumerate(root.body)
        if isinstance(statement, ast.For)
    ]
    _require(
        len(loops) == 1 and loops[0] == len(root.body) - 2,
        "requires one contraction loop followed by the output store",
    )
    initial, loop, store = root.body[0], root.body[loops[0]], root.body[-1]
    seeds = tuple(root.body[1 : loops[0]])
    _require(
        isinstance(initial, ast.Assign)
        and len(initial.targets) == 1
        and isinstance(initial.targets[0], ast.Name)
        and isinstance(loop, ast.For)
        and isinstance(loop.target, ast.Name)
        and isinstance(loop.iter, ast.Call)
        and _callee(loop.iter.func, host) is hl.tile
        and len(loop.iter.args) == 1
        and all(keyword.arg == "block_size" for keyword in loop.iter.keywords)
        and not loop.orelse
        and bool(loop.body)
        and isinstance(store, ast.Assign)
        and len(store.targets) == 1
        and isinstance(store.targets[0], ast.Subscript)
        and isinstance(store.targets[0].value, ast.Name),
        "requires a local accumulator, one K loop and one output store",
    )
    assert isinstance(initial, ast.Assign) and isinstance(initial.targets[0], ast.Name)
    assert isinstance(loop, ast.For) and isinstance(loop.target, ast.Name)
    assert isinstance(store, ast.Assign) and isinstance(store.targets[0], ast.Subscript)
    assert isinstance(store.targets[0].value, ast.Name)
    accumulator = initial.targets[0].id
    output = store.targets[0].value.id
    for statement in seeds:
        _require(
            _additive_seed(statement, accumulator, loop.target.id),
            "pre-contraction statement is not an additive accumulator seed",
        )
    tail = host.body[index + 1 :]
    _require(
        len(tail) == 1
        and isinstance(tail[0], ast.Return)
        and isinstance(tail[0].value, ast.Name)
        and tail[0].value.id == output,
        "requires a single materialized result",
    )
    definitions: dict[str, ast.Assign] = {}
    host_names = (
        {
            node.id
            for statement in host.body[:index]
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        | set(host.params.arguments)
        | {m, n, loop.target.id, accumulator}
    )
    for statement in loop.body[:-1]:
        _require(
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name),
            "K loop has a nonlocal effect",
        )
        assert isinstance(statement, ast.Assign)
        target = cast("ast.Name", statement.targets[0]).id
        _require(
            target not in host_names | definitions.keys(),
            "K recipe rebinds an existing value",
        )
        definitions[target] = statement
    update = loop.body[-1]
    _require(
        isinstance(update, ast.Assign)
        and len(update.targets) == 1
        and isinstance(update.targets[0], ast.Name)
        and update.targets[0].id == accumulator
        and isinstance(update.value, ast.Call),
        "requires one final accumulator update",
    )
    assert isinstance(update, ast.Assign) and isinstance(update.value, ast.Call)
    call = update.value
    callee = _callee(call.func, host)
    if callee is hl.dot:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        _require(
            len(call.args) == 2
            and isinstance(keywords.get("acc"), ast.Name)
            and cast("ast.Name", keywords["acc"]).id == accumulator,
            "dot must carry the local accumulator",
        )
        lhs, rhs = call.args
    else:
        _require(
            callee is torch.addmm
            and len(call.args) == 3
            and not call.keywords
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == accumulator,
            "requires a unit-scale addmm or explicit-accumulator dot",
        )
        lhs, rhs = call.args[1:]
    return _Source(
        index,
        root,
        loop,
        accumulator,
        m,
        n,
        loop.target.id,
        output,
        lhs,
        rhs,
        definitions,
        call,
        seeds,
    )


def _resolve(value: ast.expr, source: _Source) -> ast.expr:
    seen: set[str] = set()
    while isinstance(value, ast.Name) and value.id in source.definitions:
        _require(value.id not in seen, "cyclic recipe")
        seen.add(value.id)
        value = source.definitions[value.id].value
    return value


def _needed_statements(
    values: Sequence[ast.expr],
    source: _Source,
    *,
    forbidden: tuple[str, ...] | None = None,
) -> list[ast.Assign]:
    """Collect the K-loop definitions feeding ``values``.

    ``forbidden`` names the loop-carried values a materialized operand may not
    depend on: the accumulator and the output axis it is reused across (the
    output rows for an RHS ``[K, N]`` recipe, the output columns for an LHS
    ``[M, K]`` recipe).
    """
    if forbidden is None:
        forbidden = (source.m, source.accumulator)
    pending = [
        node.id
        for value in values
        for node in ast.walk(value)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    ]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        statement = source.definitions.get(name)
        if statement is not None:
            pending.extend(
                node.id
                for node in ast.walk(statement.value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
    _require(
        not seen.intersection(forbidden),
        "operand depends on the reused output axis or the accumulator",
    )
    return [statement for name, statement in source.definitions.items() if name in seen]


def _direct_input_load(value: ast.expr, host: HostFunction) -> tuple[str, bool] | None:
    """``(name, transposed)`` when ``value`` is a plain ``name[...]`` load of an
    input binding or its transpose ``name[...].T``."""
    transposed = False
    if isinstance(value, ast.Attribute) and value.attr == "T":
        value = value.value
        transposed = True
    if (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and value.value.id in host.params.arguments
    ):
        return value.value.id, transposed
    return None


def _direct_operand_load(node: Node) -> tuple[Node, bool] | None:
    """The load behind a direct operand and whether it is read transposed.

    ``name[a, b].T`` traces to a two-dimensional permute (or ``t``) of the
    load; the consumer keeps that view and the proof flips the K position.
    """
    if node.target is memory_ops.load:
        return node, False
    if node.target in _TRANSPOSES and isinstance(node.args[0], torch.fx.Node):
        dims = node.args[1] if len(node.args) > 1 else [1, 0]
        source = node.args[0]
        if (
            isinstance(dims, (list, tuple))
            and list(dims) == [1, 0]
            and source.target is memory_ops.load
        ):
            return source, True
    return None


def _recipe_side(source: _Source, host: HostFunction) -> str:
    """``"rhs"`` when the RHS carries the recipe (direct A), else ``"lhs"``.

    A program whose LHS is the recipe (``x[tile_m, tile_k].to(bf16) @ w``)
    materializes an ``[M, K]`` operand reused across the output columns; the
    RHS must then be the direct input load.
    """
    if _direct_input_load(_resolve(source.lhs, source), host) is not None:
        return "rhs"
    if _direct_input_load(_resolve(source.rhs, source), host) is not None:
        return "lhs"
    return "rhs"


def _recipe_forbidden(source: _Source, side: str) -> tuple[str, str]:
    return ((source.m if side == "rhs" else source.n), source.accumulator)


def _static(value: int | torch.SymInt, env: CompileEnvironment) -> int | None:
    expression = env.specialize_expr(env.shape_env.replace(sympy.sympify(value)))
    return int(expression) if expression.is_Integer else None


def _exact_integer_cast(node: Node) -> bool:
    """Keep wide integral recipes on their original FP32 accumulation path.

    Materialization makes the result eligible for native half/BF16 MMA. Admit
    integral conversions only when every value in a proved range is exact in
    that format; a sample tensor or copied node annotation is not range proof.
    """
    if node.target not in _CASTS:
        return True
    source = _cast_input(node)
    value = node.meta.get("val")
    original = source.meta.get("val") if source is not None else None
    if (
        source is None
        or source.graph is not node.graph
        or not isinstance(value, torch.Tensor)
        or not isinstance(original, torch.Tensor)
    ):
        return False
    if (
        value.dtype not in (torch.float16, torch.bfloat16)
        or original.is_floating_point()
        or original.is_complex()
        or original.dtype is torch.bool
    ):
        return True
    if source.target is memory_ops.load:
        # A bare cast of the loaded integer tile: the program itself rounds
        # every loaded value to the half format before the contraction, so the
        # materialized tensor holds exactly the values the original recipe
        # feeds the FP32 products.  Only derived integer arithmetic needs the
        # proved-range check below.
        return True
    if original.dtype not in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return False
    bounds = torch.iinfo(original.dtype)
    low, high = bounds.min, bounds.max
    field = _integer_field(source)
    if field is not None:
        if any(item.graph is not node.graph for item in field.nodes):
            return False
        low, high = field.bounds
    # A p-bit significand represents every integer in [-2**p, 2**p].
    limit = 256 if value.dtype is torch.bfloat16 else 2048
    return -limit <= low <= high <= limit


def _prove_recipe(
    nodes: Sequence[Node],
    *,
    k: int,
    n: int,
    inputs: set[torch.Tensor],
    env: CompileEnvironment,
) -> set[Node]:
    shape = (env.block_sizes[k].var, env.block_sizes[n].var)
    visited: set[Node] = set()
    pending = list(nodes)
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            _require(
                tuple(map(sympy.sympify, value.shape)) == shape,
                "recipe changes coordinates or tile shape",
            )
        if node.target is memory_ops.load:
            tensor = _access_tensor(node)
            indices = node.args[1]
            _require(
                tensor in inputs
                and isinstance(indices, (tuple, list))
                and len(indices) == 2
                and _exact_tile(indices[0], k, env)
                and _exact_tile(indices[1], n, env)
                and len(node.args) == 4
                and node.args[2] is None
                and node.args[3] is None,
                "recipe load is not a canonical read-only K,N tile",
            )
            continue
        target = node.target
        _require(
            node.op == "call_function"
            and isinstance(target, torch._ops.OpOverload)
            and torch.Tag.pointwise in target.tags
            and _ordinary_operation(node),
            "recipe contains a reduction, gather or opaque operation",
        )
        _require(
            _exact_integer_cast(node),
            "integer cast lacks an exactly representable range",
        )
        pending.extend(node.all_input_nodes)
    return visited


def _prove_seed(
    graphs: Sequence[GraphInfo],
    node: Node,
    acc: Node,
    *,
    m: int,
    n: int,
    inputs: set[torch.Tensor],
    env: CompileEnvironment,
) -> set[Node]:
    """The accumulator entering the contraction is the K loop's carry of a root
    value built from one zero tile, pointwise operations and unmasked loads of
    complete ``[M, N]`` input tiles.

    ``_source`` fixed the statement shape (additive rebindings of the zero
    accumulator); this proves the dataflow those statements traced, so the
    consumer may add the same terms after the contraction.  The K loop is this
    contraction's own loop (checked by the caller), so the carry restarts from
    the captured value for every output tile.  Returns the seed's root nodes.
    """
    current = strip_new_vars(acc)
    _require(
        current.op == "placeholder" and current.graph is node.graph,
        "requires one exclusive FP32 accumulator carried by the K loop",
    )
    entry = _loop_entry(graphs, current)
    _require(entry is not None, "the accumulator is not captured by the K loop")
    assert entry is not None
    shape = (env.block_sizes[m].var, env.block_sizes[n].var)
    visited: set[Node] = set()
    zeros = 0
    pending = [entry[1]]
    while pending:
        item = pending.pop()
        if item in visited:
            continue
        visited.add(item)
        value = item.meta.get("val")
        _require(
            isinstance(value, torch.Tensor)
            and tuple(map(sympy.sympify, value.shape)) == shape,
            "seed changes coordinates or tile shape",
        )
        assert isinstance(value, torch.Tensor)
        if item.target is creation_ops.full:
            fill = item.args[1]
            _require(
                isinstance(fill, (int, float))
                and not isinstance(fill, bool)
                and fill == 0
                and value.dtype == torch.float32,
                "requires a local zero accumulator seed",
            )
            zeros += 1
            continue
        if item.target is memory_ops.load:
            tensor = _access_tensor(item)
            indices = item.args[1]
            _require(
                tensor in inputs
                and isinstance(indices, (tuple, list))
                and len(indices) == 2
                and _exact_tile(indices[0], m, env)
                and _exact_tile(indices[1], n, env)
                and len(item.args) == 4
                and item.args[2] is None
                and item.args[3] is None,
                "seed load is not a canonical read-only M,N tile",
            )
            continue
        target = item.target
        _require(
            item.op == "call_function"
            and isinstance(target, torch._ops.OpOverload)
            and torch.Tag.pointwise in target.tags
            and _ordinary_operation(item),
            "seed contains a reduction, gather or opaque operation",
        )
        pending.extend(item.all_input_nodes)
    _require(zeros == 1, "requires a local zero accumulator seed")
    return visited


def _fused_recompute_pays(
    host: HostFunction,
    recipe_nodes: set[Node],
    *,
    reused_extent: int,
    reused_tile: int,
    operand_dtype: torch.dtype,
) -> bool:
    """Whether the fused GEMM derives its operand expensively enough for a
    producer launch plus an operand round trip to pay off.

    A fused tcgen05 kernel stages every source tile of the recipe beside the
    plain operand and derives the operand tile on chip, in every CTA of the
    output axis the recipe does not depend on.  Relative to the plain GEMM the
    consumer runs, that costs

    * traffic = sum of the distinct source tensors' element bytes / the
      operand's element bytes (the extra operand-path bytes per element), and
    * R = extent of the reused output axis / the per-CTA MMA tile along it
      (how many times each operand element is derived).

    Materialize when both are at least 2: a one-tensor same-width recipe or a
    reused axis the MMA tile covers twice keeps its single launch.  The gate
    compares against the fused derived-operand lowering (``cute_mma_dag``), so
    it applies when that lowering's config-free matcher admits the program
    (a floating recipe over the operand path); integer casts and bitfield
    unpacks have no fused native operand path and keep the admission above
    (``_fused_register_mma_admitted`` covers their register-MMA regime).
    ``host`` is the planner's temporary compile of the unfissioned program,
    whose shapes the planner specialized; the fused program only has them
    when the kernel is static, so the gate is honored for static kernels
    alone (the matcher refuses the fused dynamic program's symbolic tiles).
    """
    env = CompileEnvironment.current()
    if not env.settings.static_shapes or match_mma_dag(host.device_ir.graphs) is None:
        return True
    sources: set[torch.Tensor] = set()
    for item in recipe_nodes:
        if item.target is memory_ops.load:
            tensor = _access_tensor(item)
            assert tensor is not None
            sources.add(tensor)
    traffic = sum(tensor.element_size() for tensor in sources) / operand_dtype.itemsize
    return traffic >= 2 and reused_extent >= 2 * reused_tile


def _strip_zero_padding(node: Node) -> tuple[Node, set[Node]]:
    """A dense materialization reload supplies the same out-of-domain zeros."""
    masks: set[Node] = set()
    while node.target is _tracing_ops._mask_to:
        _require(
            len(node.args) == 2
            and isinstance(node.args[1], (int, float))
            and node.args[1] == 0
            and not node.kwargs
            and isinstance(node.args[0], torch.fx.Node),
            "unrecognized contraction padding",
        )
        masks.add(node)
        assert isinstance(node.args[0], torch.fx.Node)
        node = node.args[0]
    return node, masks


def _prove(host: HostFunction, source: _Source, env: CompileEnvironment) -> _Proof:
    ir = host.device_ir
    _require(len(ir.root_ids) == 1 and len(ir.graphs) == 2, "unexpected nested graphs")
    root_axes = _indices(source.root)
    k_axes = _indices(source.loop)
    _require(
        root_axes is not None
        and len(root_axes) == 2
        and all(isinstance(axis, TileIndexType) for axis in root_axes)
        and k_axes is not None
        and len(k_axes) == 1
        and isinstance(k_axes[0], TileIndexType)
        and not ir.noncanonical_task_origin_block_ids,
        "requires canonical tile axes",
    )
    assert root_axes is not None and k_axes is not None
    m_id, n_id = (axis.block_id for axis in root_axes)
    k_id = k_axes[0].block_id
    contractions = [
        node
        for graph in ir.graphs
        for node in graph.graph.nodes
        if node.target in (dot, torch.ops.aten.addmm.default)
    ]
    _require(len(contractions) == 1, "requires one matrix contraction")
    node = contractions[0]
    if node.target is dot:
        lhs, rhs, acc = node.args[:3]
    else:
        acc, lhs, rhs = node.args[:3]
    _require(
        all(isinstance(value, torch.fx.Node) for value in (lhs, rhs, acc)),
        "unrecognized matrix operands",
    )
    assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
    assert isinstance(acc, torch.fx.Node)
    _require(
        _mma_loop_is_exclusive(node) and acc.meta["val"].dtype == torch.float32,
        "requires one exclusive FP32 accumulator",
    )
    loop_graph = next(graph for graph in ir.graphs if graph.graph is node.graph)
    _require(
        isinstance(loop_graph, ForLoopGraphInfo) and loop_graph.block_ids == [k_id],
        "contraction is not owned by the canonical K loop",
    )
    inputs = {
        value
        for value in host.params.arguments.values()
        if isinstance(value, torch.Tensor)
    }
    seed_nodes = _prove_seed(
        ir.graphs, node, acc, m=m_id, n=n_id, inputs=inputs, env=env
    )
    a, b = lhs.meta["val"], rhs.meta["val"]
    side = _recipe_side(source, host)
    # ``direct`` is the operand that stays a plain input load; ``recipe`` is
    # the operand that is materialized.  An RHS recipe is reused across the
    # output rows (M), an LHS recipe across the output columns (N).
    direct_root, recipe_root = (lhs, rhs) if side == "rhs" else (rhs, lhs)
    direct_expr = source.lhs if side == "rhs" else source.rhs
    direct_load = _direct_operand_load(direct_root)
    _require(
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.ndim == b.ndim == 2
        and a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and direct_load is not None,
        "requires one direct half/BF16 operand and a matching recipe",
    )
    assert direct_load is not None
    direct, transposed = direct_load
    named = _direct_input_load(_resolve(direct_expr, source), host)
    _require(
        named is not None and named[1] == transposed,
        "the direct operand must load from an unchanged input binding",
    )
    assert named is not None
    a_name = named[0]
    a_tensor = _access_tensor(direct)
    _require(
        a_tensor is host.params.arguments[a_name] and _is_contiguous_2d(a_tensor, env),
        "the direct operand has an unsupported alias or layout",
    )
    assert a_tensor is not None
    recipe, padding = _strip_zero_padding(recipe_root)
    packed = packed_matmul_axis(env, lhs, recipe) if side == "rhs" else None
    factor = packed.factor if packed is not None else 1
    # A direct A operand reads ``[M, K]`` (K in position 1), a direct B operand
    # ``[K, N]``; a transposed read swaps the two positions.
    direct_k_position = 1 if side == "rhs" else 0
    if transposed:
        direct_k_position = 1 - direct_k_position
    direct_outer_position = 1 - direct_k_position
    if packed is not None:
        _require(packed.block_id == k_id, "packed K does not own the contraction")
        terms = interleaved_matrix_terms(recipe, lhs=False)
        _require(terms is not None, "unrecognized interleaved RHS")
        value = _resolve(source.rhs, source)
        _require(
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in ("reshape", "view"),
            "interleaved source is not a reshape",
        )
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
        stack = _resolve(value.func.value, source)
        _require(
            isinstance(stack, ast.Call)
            and _callee(stack.func, host) is torch.stack
            and len(stack.args) >= 1
            and isinstance(stack.args[0], (ast.Tuple, ast.List))
            and len(stack.args[0].elts) == factor,
            "source and FX interleave order disagree",
        )
        assert isinstance(stack, ast.Call) and isinstance(
            stack.args[0], (ast.Tuple, ast.List)
        )
        expressions = tuple(stack.args[0].elts)
    else:
        indices = direct.args[1]
        _require(
            isinstance(indices, (list, tuple))
            and len(indices) == 2
            and _exact_tile(indices[direct_k_position], k_id, env),
            "the direct operand has a noncanonical K mapping",
        )
        terms = (recipe,)
        expressions = (source.rhs if side == "rhs" else source.lhs,)
    indices = direct.args[1]
    direct_outer_id = m_id if side == "rhs" else n_id
    _require(
        isinstance(indices, (list, tuple))
        and len(indices) == 2
        and _exact_tile(indices[direct_outer_position], direct_outer_id, env),
        "the direct operand changes output ownership",
    )
    extents = [
        _static(cast("int | torch.SymInt", env.block_sizes[axis].size), env)
        for axis in (m_id, n_id, k_id)
    ]
    _require(
        all(extent is not None and extent > 0 for extent in extents),
        "guarded matrix extents do not cover the complete logical K domain",
    )
    m_extent, n_extent, k_extent = (cast("int", extent) for extent in extents)
    direct_outer_extent = m_extent if side == "rhs" else n_extent
    direct_k_extent = k_extent * factor if side == "rhs" else k_extent
    _require(
        _static(a_tensor.shape[direct_outer_position], env) == direct_outer_extent
        and _static(a_tensor.shape[direct_k_position], env) == direct_k_extent,
        "guarded matrix extents do not cover the complete logical K domain",
    )
    assert terms is not None
    if side == "rhs":
        recipe_nodes = _prove_recipe(terms, k=k_id, n=n_id, inputs=inputs, env=env)
    else:
        recipe_nodes = _prove_recipe(terms, k=m_id, n=k_id, inputs=inputs, env=env)
    _require(
        factor > 1 or any(item.target is not memory_ops.load for item in recipe_nodes),
        "direct operands are already materialized",
    )
    _needed_statements(expressions, source, forbidden=_recipe_forbidden(source, side))
    _require(
        _fused_recompute_pays(
            host,
            recipe_nodes,
            reused_extent=m_extent if side == "rhs" else n_extent,
            reused_tile=_TCGEN05_CTA_TILE_M if side == "rhs" else _TCGEN05_CTA_TILE_N,
            operand_dtype=a.dtype,
        ),
        "the fused GEMM derives its operand from one same-width source or "
        "recomputes it fewer than twice per element",
    )
    recipe_nodes.update(padding)
    stores = [
        item
        for graph in ir.graphs
        for item in graph.graph.nodes
        if item.target is memory_ops.store
    ]
    _require(len(stores) == 1, "requires one complete output store")
    store = stores[0]
    output = _access_tensor(store)
    fresh = _fresh_tensors(host)
    _require(
        output is not None
        and output in fresh
        and _is_contiguous_2d(output, env)
        and store.graph is not node.graph
        and len(store.args) == 4
        and store.args[3] is None,
        "output is not one unconditional fresh row-major allocation",
    )
    assert output is not None
    store_indices = store.args[1]
    _require(
        isinstance(store_indices, (tuple, list))
        and len(store_indices) == 2
        and _exact_tile(store_indices[0], m_id, env)
        and _exact_tile(store_indices[1], n_id, env)
        and _static(output.shape[0], env) == m_extent
        and _static(output.shape[1], env) == n_extent,
        "output grid does not overwrite every element exactly once",
    )
    for graph in ir.graphs:
        for item in graph.graph.nodes:
            if item.target is memory_ops.load:
                _require(
                    item is direct or item in recipe_nodes or item in seed_nodes,
                    "unexpected memory read",
                )
            _require(
                item is node or item in recipe_nodes or _ordinary_operation(item),
                "unexpected side effect or control flow",
            )
    captures: list[str] = []
    for statement in host.body:
        if isinstance(statement, ast.For):
            break
        if isinstance(statement, ast.Assign) and isinstance(
            value_type := cast("ExtendedAST", statement.value)._type_info, TensorType
        ):
            _require(
                value_type.proxy() in fresh
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name),
                "unrecognized shared allocation",
            )
            captures.append(cast("ast.Name", statement.targets[0]).id)
    _require(source.output in captures, "output is not captured by its original name")
    return _Proof(
        factor, b.dtype, a_name, expressions, tuple(captures), side, transposed
    )


def _epilogue_seed(statement: ast.stmt, accumulator: str) -> ast.stmt:
    """The consumer's copy of a pre-contraction seed, placed after the loop.

    ``acc += e`` becomes ``acc = acc + e`` (the form the tracer gives both);
    an ``Assign`` seed is copied as is.  The accumulator now holds the finished
    sum, so the statement adds the seed to it: the fp32 sum is reassociated
    (``sum + e`` instead of ``(e + p_0) + p_1 ...``), nothing is approximated.
    """
    if isinstance(statement, ast.AugAssign):
        operator = "+" if isinstance(statement.op, ast.Add) else "-"
        return statement_from_string(
            f"{accumulator} = {accumulator} {operator} {{value}}",
            value=_clone_untyped(statement.value),
        )
    return cast("ast.stmt", _clone_untyped(statement))


def _consumer_body(
    consumer: ast.For, contraction: ast.For, source: _Source
) -> list[ast.stmt]:
    """``init, contraction over the materialized operand, seed*, store``."""
    return [
        consumer.body[0],
        contraction,
        *(_epilogue_seed(seed, source.accumulator) for seed in source.seeds),
        consumer.body[-1],
    ]


def _replacement(
    host: HostFunction,
    source: _Source,
    original_body: list[ast.stmt],
    specializations: list[ast.stmt],
    proof: _Proof,
) -> MaterializedFissionPlan:
    if proof.side == "lhs":
        return _replacement_lhs(host, source, original_body, specializations, proof)
    factor, dtype, a_name = proof.factor, proof.dtype, proof.direct_name
    expressions, captures = proof.expressions, proof.captures
    # The direct A operand is ``a[m, k]`` or ``a[k, m].T``.
    k_dim = 0 if proof.transposed else 1
    used = _bound_names(host) | set(host.fn.__globals__)

    def fresh(name: str) -> str:
        while name in used:
            name += "_"
        used.add(name)
        return name

    matrix = fresh("_helion_materialized_operand")
    row = fresh("_helion_materialize_k")
    column = fresh("_helion_materialize_n")
    index = fresh("_helion_materialize_index")
    logical_k = fresh("_helion_materialize_logical_k")
    assert isinstance(source.root, ExtendedAST)
    with source.root:
        tile = _reference(hl.tile, host)
        empty = _reference(torch.empty, host)
        arange = _reference(hl.arange, host)
        element_type = _reference(dtype, host)
        assert isinstance(source.loop.iter, ast.Call)
        assert isinstance(source.root.iter, ast.Call)
        assert isinstance(source.root.iter.args[0], (ast.Tuple, ast.List))
        n_bound = source.root.iter.args[0].elts[1]
        allocation = statement_from_string(
            f"{matrix} = {{empty}}(({a_name}.size({k_dim}), {{n}}), "
            f"dtype={{dtype}}, device={a_name}.device)",
            empty=empty,
            n=n_bound,
            dtype=element_type,
        )
        producer = statement_from_string(
            f"for {row}, {column} in {{tile}}([{{k}}, {{n}}]):\n    pass",
            tile=tile,
            k=source.loop.iter.args[0],
            n=n_bound,
        )
        assert isinstance(producer, ast.For) and isinstance(producer.iter, ast.Call)
        if source.loop.iter.keywords:
            block = source.loop.iter.keywords[0].value
            block_call = expr_from_string("f(block_size=[{block}, None])", block=block)
            assert isinstance(block_call, ast.Call)
            producer.iter.keywords = block_call.keywords
        body: list[ast.stmt] = list(_needed_statements(expressions, source))
        if factor > 1:
            body.append(
                statement_from_string(
                    f"{index} = {{arange}}({source.k}.begin, "
                    f"{source.k}.begin + {source.k}.block_size)",
                    arange=arange,
                )
            )
        for position, expression in enumerate(expressions):
            row_expression = (
                f"{index} * {factor} + {position}" if factor > 1 else source.k
            )
            body.append(
                statement_from_string(
                    f"{matrix}[{row_expression}, {source.n}] = {{value}}",
                    value=expression,
                )
            )
        producer.body = [cast("ast.stmt", _clone_untyped(stmt)) for stmt in body]
        for node in ast.walk(producer):
            if isinstance(node, ast.Name):
                if node.id == source.k:
                    node.id = row
                elif node.id == source.n:
                    node.id = column
        consumer = cast("ast.For", _clone_untyped(source.root))
        original_update = source.loop.body[-1]
        assert isinstance(original_update, ast.Assign)
        update = cast("ast.Assign", _clone_untyped(original_update))
        assert isinstance(update.value, ast.Call)
        load_a = cast(
            "ast.expr",
            expr_from_string(
                f"{a_name}[{logical_k}, {source.m}].T"
                if proof.transposed
                else f"{a_name}[{source.m}, {logical_k}]"
            ),
        )
        load_b = cast(
            "ast.expr", expr_from_string(f"{matrix}[{logical_k}, {source.n}]")
        )
        if _callee(source.call.func, host) is hl.dot:
            update.value.args = [load_a, load_b]
        else:
            update.value.args[1:] = [load_a, load_b]
        contraction = statement_from_string(
            f"for {logical_k} in {{tile}}({a_name}.size({k_dim})):\n    pass",
            tile=tile,
        )
        assert isinstance(contraction, ast.For)
        contraction.body = [update]
        consumer.body = _consumer_body(consumer, contraction, source)
        prefix = _rewritten_prefix(host, source, original_body)
        replacement_body = (
            *specializations,
            *prefix,
            allocation,
            producer,
            consumer,
            *original_body[source.index + 1 :],
        )
    return MaterializedFissionPlan(
        root_index=len(specializations) + source.index + 1,
        source_root_key=ast.dump(original_body[source.index]),
        materialized_names=(*captures, matrix),
        region_count=2,
        source_root_index=source.index,
        replacement_body=tuple(
            cast("ast.stmt", _clone_untyped(statement))
            for statement in replacement_body
        ),
        pointwise_region_indices=(0,),
        operand_interleave_factor=factor,
    )


def _rewritten_prefix(
    host: HostFunction, source: _Source, original_body: list[ast.stmt]
) -> list[ast.stmt]:
    """Clone the host statements before the grid; the consumer overwrites every
    output element, so a zero/one-filled output factory becomes ``empty``."""
    prefix = [
        cast("ast.stmt", _clone_untyped(stmt)) for stmt in original_body[: source.index]
    ]
    for statement in prefix:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == source.output
            and isinstance(statement.value, ast.Call)
        ):
            factory = _callee(statement.value.func, host)
            replacements: dict[object, object] = {
                torch.zeros: torch.empty,
                torch.ones: torch.empty,
                torch.zeros_like: torch.empty_like,
                torch.ones_like: torch.empty_like,
            }
            replacement = replacements.get(factory)
            if replacement is not None:
                reference = _reference(replacement, host)
                statement.value.func = cast("ast.expr", reference)
    return prefix


def _replacement_lhs(
    host: HostFunction,
    source: _Source,
    original_body: list[ast.stmt],
    specializations: list[ast.stmt],
    proof: _Proof,
) -> MaterializedFissionPlan:
    """Materialize an ``[M, K]`` LHS recipe reused across the output columns.

    The producer tiles ``[M, K]`` and stores the recipe; the consumer keeps the
    original ``[M, N]`` grid and contracts ``matrix[m, k] @ b[k, n]`` (or
    ``matrix[m, k] @ b[n, k].T``) over the direct RHS input's K extent.
    """
    dtype, b_name = proof.dtype, proof.direct_name
    expressions, captures = proof.expressions, proof.captures
    # The direct B operand is ``b[k, n]`` or ``b[n, k].T``.
    k_dim = 1 if proof.transposed else 0
    used = _bound_names(host) | set(host.fn.__globals__)

    def fresh(name: str) -> str:
        while name in used:
            name += "_"
        used.add(name)
        return name

    matrix = fresh("_helion_materialized_operand")
    row = fresh("_helion_materialize_m")
    column = fresh("_helion_materialize_k")
    logical_k = fresh("_helion_materialize_logical_k")
    assert isinstance(source.root, ExtendedAST)
    with source.root:
        tile = _reference(hl.tile, host)
        empty = _reference(torch.empty, host)
        element_type = _reference(dtype, host)
        assert isinstance(source.loop.iter, ast.Call)
        assert isinstance(source.root.iter, ast.Call)
        assert isinstance(source.root.iter.args[0], (ast.Tuple, ast.List))
        m_bound = source.root.iter.args[0].elts[0]
        allocation = statement_from_string(
            f"{matrix} = {{empty}}(({{m}}, {b_name}.size({k_dim})), "
            f"dtype={{dtype}}, device={b_name}.device)",
            empty=empty,
            m=m_bound,
            dtype=element_type,
        )
        producer = statement_from_string(
            f"for {row}, {column} in {{tile}}([{{m}}, {{k}}]):\n    pass",
            tile=tile,
            m=m_bound,
            k=source.loop.iter.args[0],
        )
        assert isinstance(producer, ast.For) and isinstance(producer.iter, ast.Call)
        if source.loop.iter.keywords:
            block = source.loop.iter.keywords[0].value
            block_call = expr_from_string("f(block_size=[None, {block}])", block=block)
            assert isinstance(block_call, ast.Call)
            producer.iter.keywords = block_call.keywords
        body: list[ast.stmt] = list(
            _needed_statements(
                expressions, source, forbidden=_recipe_forbidden(source, "lhs")
            )
        )
        (expression,) = expressions
        body.append(
            statement_from_string(
                f"{matrix}[{source.m}, {source.k}] = {{value}}", value=expression
            )
        )
        producer.body = [cast("ast.stmt", _clone_untyped(stmt)) for stmt in body]
        for node in ast.walk(producer):
            if isinstance(node, ast.Name):
                if node.id == source.m:
                    node.id = row
                elif node.id == source.k:
                    node.id = column
        consumer = cast("ast.For", _clone_untyped(source.root))
        original_update = source.loop.body[-1]
        assert isinstance(original_update, ast.Assign)
        update = cast("ast.Assign", _clone_untyped(original_update))
        assert isinstance(update.value, ast.Call)
        load_a = cast(
            "ast.expr", expr_from_string(f"{matrix}[{source.m}, {logical_k}]")
        )
        load_b = cast(
            "ast.expr",
            expr_from_string(
                f"{b_name}[{source.n}, {logical_k}].T"
                if proof.transposed
                else f"{b_name}[{logical_k}, {source.n}]"
            ),
        )
        if _callee(source.call.func, host) is hl.dot:
            update.value.args = [load_a, load_b]
        else:
            update.value.args[1:] = [load_a, load_b]
        contraction = statement_from_string(
            f"for {logical_k} in {{tile}}({b_name}.size({k_dim})):\n    pass",
            tile=tile,
        )
        assert isinstance(contraction, ast.For)
        contraction.body = [update]
        consumer.body = _consumer_body(consumer, contraction, source)
        prefix = _rewritten_prefix(host, source, original_body)
        replacement_body = (
            *specializations,
            *prefix,
            allocation,
            producer,
            consumer,
            *original_body[source.index + 1 :],
        )
    return MaterializedFissionPlan(
        root_index=len(specializations) + source.index + 1,
        source_root_key=ast.dump(original_body[source.index]),
        materialized_names=(*captures, matrix),
        region_count=2,
        source_root_index=source.index,
        replacement_body=tuple(
            cast("ast.stmt", _clone_untyped(statement))
            for statement in replacement_body
        ),
        pointwise_region_indices=(0,),
        operand_interleave_factor=1,
    )


def _fused_register_mma_admitted(env: CompileEnvironment) -> bool:
    """Whether the unfissioned program's search admits the register-MMA family
    for its cast operand (``CuteTcgen05Config.tcgen05_family_admitted`` is
    False exactly when the matmul reads an in-register integer cast).

    That family runs the latency-bound problems a materialization launch
    would only slow down (the extra launch and its DRAM round trip cost more
    than the whole fused GEMM: bf16 x int16 at (128, 256, 128) on B200 runs
    4.3 us as two launches against ~2.2 us fused), so such a program keeps
    its single launch; outside the family's admitted regime the tcgen05
    pipeline needs the materialized operand and the fission proceeds.
    ``env`` is the planner's temporary compile of the unfissioned program.
    """
    state = env.config_spec._cute_tcgen05_config
    return state.warp_mma_admitted and not state.tcgen05_family_admitted


def plan_operand_materialization(
    kernel: Kernel[Any],
    args: tuple[object, ...],
    owning_env: CompileEnvironment,
) -> MaterializedFissionPlan | None:
    if (
        not owning_env.settings.cute_materialize_transformed_operands
        or owning_env.backend_name != "cute"
        or owning_env._is_distributed
        or any(type(value) not in (torch.Tensor, torch.nn.Parameter) for value in args)
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in kernel.signature.parameters.values()
        )
    ):
        return None
    env = CompileEnvironment(
        owning_env.device,
        owning_env.settings,
        index_dtype=owning_env.index_dtype,
        is_distributed=False,
    )
    with env:
        fake_args = [
            env.to_fake(value, ArgumentOrigin(name))
            for name, value in zip(kernel.signature.parameters, args, strict=True)
        ]
        compiler = KernelCompiler(env)
        host = compiler.parse(kernel.fn, fake_args, {})
        with host, compiler._compilation_context():
            if not _safe_source(host):
                return None
            try:
                _source(host)
                compiler.unroll(host)
                compiler.customize_ast(host)
                source = _source(host)
                side = _recipe_side(source, host)
                _needed_statements(
                    (source.rhs if side == "rhs" else source.lhs,),
                    source,
                    forbidden=_recipe_forbidden(source, side),
                )
                _require(
                    not host.args.posonlyargs
                    and not host.args.kwonlyargs
                    and host.args.vararg is None
                    and host.args.kwarg is None
                    and not host.constexpr_args,
                    "requires ordinary positional tensor inputs",
                )
                original_body = [
                    cast("ast.stmt", _clone_untyped(statement))
                    for statement in host.body
                ]
                specializations: list[ast.stmt] = []
                assert isinstance(source.root, ExtendedAST)
                with source.root:
                    specialize = _reference(hl.specialize, host)
                    used = _bound_names(host)
                    for name in host.params.arguments:
                        temporary = f"_helion_materialize_shape_{name}"
                        while temporary in used:
                            temporary += "_"
                        used.add(temporary)
                        specializations.append(
                            statement_from_string(
                                f"{temporary} = {{specialize}}({name}.shape)",
                                specialize=specialize,
                            )
                        )
                host.body[:0] = specializations
                compiler.propagate_types(host)
                compiler.finalize_config()
                compiler.lower(host)
                # The proof below specializes the shapes it needs; the fused
                # program only has them when the kernel is static, so the
                # register-MMA regime is honored for static kernels alone.
                _require(
                    not (
                        owning_env.settings.static_shapes
                        and _fused_register_mma_admitted(env)
                    ),
                    "the fused GEMM is admitted to the register-MMA family, "
                    "which converts the operand in registers in one launch",
                )
                proof = _prove(host, source, env)
                return _replacement(host, source, original_body, specializations, proof)
            except _Ineligible as error:
                log.debug("CuTe operand materialization rejected: %s", error)
                return None
