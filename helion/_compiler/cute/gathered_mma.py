"""Admission for a direct gathered-row native-MMA pipeline.

The proof is structural and independent of argument names and index contents.
Two logical rows may execute in different CTAs only when equal gathered rows
read the same A/B elements and store the same rounded accumulator. All original
row-index arithmetic and store predicates are retained for emission.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import itertools
from typing import TYPE_CHECKING

from ..ast_read_writes import ReadWrites
from .scalar_recipe import _clone
from .scalar_recipe import _read_names
from .scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .collective_matmul import CollectiveMmaSite

_EXPANSION_NODE_BUDGET = 4096


@dataclass(frozen=True)
class GatherTensorFacts:
    """Exact metadata guarded by dispatch, never sampled array contents."""

    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    alignment: int


@dataclass(frozen=True)
class GatheredMmaRegion:
    lhs: str
    rhs: str
    output: str
    m_extent: ast.expr
    row: ast.expr
    row_predicate: ast.expr
    group: ast.expr
    m_loop: ast.For
    m_parent: list[ast.stmt]
    m_position: int
    group_count: int
    n_size: int
    k_size: int
    tensor_reads: frozenset[str]


def _same(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left) == ast.dump(right)


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _replace(value: ast.expr, replacements: Mapping[str, ast.expr]) -> ast.expr:
    class Replace(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return _clone(replacements.get(node.id, node))

    result = Replace().visit(_clone(value))
    assert isinstance(result, ast.expr)
    return result


def _replace_tree(value: ast.expr, before: ast.expr, after: ast.expr) -> ast.expr:
    class Replace(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            if _same(node, before):
                return _clone(after)
            return super().visit(node)

    result = Replace().visit(_clone(value))
    assert isinstance(result, ast.expr)
    return result


def _expanded(
    value: ast.expr,
    statements: Sequence[ast.stmt],
    boundaries: set[str],
) -> ast.expr | None:
    recipe = build_recipe(value, statements, boundaries)
    if recipe is None:
        return None
    counter = itertools.count()
    used_names = boundaries | {
        node.id
        for root in (value, *statements)
        for node in ast.walk(root)
        if isinstance(node, ast.Name)
    }

    def unique_name(hint: str) -> str:
        while True:
            name = f"gather_{next(counter)}_{hint}"
            if name not in used_names:
                used_names.add(name)
                return name

    emitted, result = recipe.emit({}, unique_name)
    replacements: dict[str, ast.expr] = {}
    sizes: dict[str, int] = {}
    remaining = _EXPANSION_NODE_BUDGET

    def expanded_size(value: ast.expr) -> int | None:
        """Count substituted trees before cloning, without visiting each copy."""
        pending: list[ast.AST] = [value]
        total = 0
        while pending:
            node = pending.pop()
            if isinstance(node, ast.Name) and node.id in sizes:
                total += sizes[node.id]
            else:
                total += 1
                pending.extend(ast.iter_child_nodes(node))
            if total > remaining:
                return None
        return total

    for statement in emitted:
        target = statement.targets[0]
        assert isinstance(target, ast.Name)
        size = expanded_size(statement.value)
        if size is None:
            return None
        # Bound all materialized intermediate expressions, not merely the
        # final result. An exponentially reused integer DAG must decline as a
        # whole, never become an opaque name that weakens dependence proofs.
        remaining -= size
        sizes[target.id] = size
        replacements[target.id] = _replace(statement.value, replacements)
    if expanded_size(result) is None:
        return None
    return _replace(result, replacements)


def _pointer_indices(
    pointer: ast.expr,
    tensors: Mapping[str, GatherTensorFacts],
) -> tuple[str, tuple[ast.expr, ...]] | None:
    """Match the compiler's explicit per-dimension pointer arithmetic."""
    terms: list[ast.expr] = []

    def flatten(value: ast.expr) -> None:
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            flatten(value.left)
            flatten(value.right)
        else:
            terms.append(value)

    def uncast(value: ast.expr) -> ast.expr:
        while (
            isinstance(value, ast.Call)
            and ast.unparse(value.func) in {"cutlass.Int32", "cutlass.Int64"}
            and len(value.args) == 1
            and not value.keywords
        ):
            value = value.args[0]
        return value

    flatten(pointer)
    bases = [
        value
        for value in terms
        if isinstance(value, ast.Attribute)
        and value.attr == "iterator"
        and isinstance(value.value, ast.Name)
        and value.value.id in tensors
    ]
    if len(bases) != 1:
        return None
    base = bases[0]
    assert isinstance(base.value, ast.Name)
    name = base.value.id
    rank = len(tensors[name].shape)
    indices: dict[int, ast.expr] = {}
    for term in terms:
        if term is base:
            continue
        if not isinstance(term, ast.BinOp) or not isinstance(term.op, ast.Mult):
            return None
        matches = []
        for stride, index in ((term.left, term.right), (term.right, term.left)):
            stride = uncast(stride)
            if (
                isinstance(stride, ast.Subscript)
                and isinstance(stride.slice, ast.Constant)
                and type(stride.slice.value) is int
                and ast.unparse(stride.value)
                in {f"{name}.layout.stride", f"{name}.stride"}
            ):
                matches.append((stride.slice.value, index))
        if len(matches) != 1:
            return None
        dim, index = matches[0]
        if dim in indices or not 0 <= dim < rank:
            return None
        indices[dim] = index
    if set(indices) != set(range(rank)):
        return None
    return name, tuple(indices[dim] for dim in range(rank))


class _Normalize(ast.NodeTransformer):
    """Proof-only normalization; emitted metadata arithmetic stays untouched."""

    def __init__(
        self,
        constants: Mapping[str, int],
        thread_dims: tuple[int, int, int],
    ) -> None:
        self.constants = constants
        self.thread_dims = thread_dims

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in self.constants:
            return ast.Constant(value=self.constants[node.id])
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        visited = self.generic_visit(node)
        assert isinstance(visited, ast.Call)
        node = visited
        name = ast.unparse(node.func)
        if (
            name in {"cutlass.Int32", "cutlass.Int64"}
            and len(node.args) == 1
            and not node.keywords
        ):
            return node.args[0]
        comparisons = {
            "operator.lt": ast.Lt,
            "operator.le": ast.LtE,
            "operator.gt": ast.Gt,
            "operator.ge": ast.GtE,
            "operator.eq": ast.Eq,
            "operator.ne": ast.NotEq,
        }
        if name in comparisons and len(node.args) == 2 and not node.keywords:
            return self.visit_Compare(
                ast.Compare(
                    left=node.args[0],
                    ops=[comparisons[name]()],
                    comparators=[node.args[1]],
                )
            )
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if (
            isinstance(node.value, ast.Call)
            and not node.value.args
            and not node.value.keywords
            and isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is int
        ):
            function = ast.unparse(node.value.func)
            if function == "cute.arch.block_idx" and node.slice.value == 0:
                return ast.Name(id="_gather_block", ctx=ast.Load())
            if function == "cute.arch.thread_idx" and 0 <= node.slice.value < 3:
                return ast.Name(id=f"_gather_thread_{node.slice.value}", ctx=ast.Load())
        result = self.generic_visit(node)
        assert isinstance(result, ast.expr)
        return result

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        result = self.generic_visit(node)
        assert isinstance(result, ast.Compare)
        if (
            len(result.ops) == 1
            and isinstance(result.ops[0], ast.Lt)
            and isinstance(result.left, ast.Name)
            and isinstance(result.comparators[0], ast.Constant)
            and type(result.comparators[0].value) is int
        ):
            for dim, extent in enumerate(self.thread_dims):
                if (
                    result.left.id == f"_gather_thread_{dim}"
                    and extent <= result.comparators[0].value
                ):
                    return ast.Constant(value=True)
        return result


def _normalize(
    value: ast.expr,
    constants: Mapping[str, int],
    thread_dims: tuple[int, int, int],
) -> ast.expr:
    result = _Normalize(constants, thread_dims).visit(_clone(value))
    assert isinstance(result, ast.expr)
    return result


def _conjuncts(value: ast.expr) -> list[ast.expr]:
    if isinstance(value, ast.Constant) and value.value is True:
        return []
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And):
        return [part for child in value.values for part in _conjuncts(child)]
    return [value]


def _matching_conjunction(
    value: ast.expr,
    expected: Sequence[ast.expr],
    optional: Sequence[ast.expr] = (),
) -> bool:
    actual = {ast.dump(part) for part in _conjuncts(value)}
    required = {ast.dump(part) for part in expected}
    allowed = required | {ast.dump(part) for part in optional}
    return required <= actual <= allowed


def _integer_constant(value: ast.expr, constants: Mapping[str, int]) -> int | None:
    if isinstance(value, ast.Constant) and type(value.value) is int:
        return value.value
    if isinstance(value, ast.Name):
        return constants.get(value.id)
    if isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords:
        bits = {"cutlass.Int32": 32, "cutlass.Int64": 64}.get(ast.unparse(value.func))
        constant = _integer_constant(value.args[0], constants)
        if (
            bits is not None
            and constant is not None
            and -(1 << (bits - 1)) <= constant < 1 << (bits - 1)
        ):
            return constant
    return None


def _full_thread_mask(
    value: ast.expr,
    constants: Mapping[str, int],
    thread_dims: tuple[int, int, int],
) -> bool:
    if not (
        isinstance(value, ast.Compare)
        and len(value.ops) == 1
        and isinstance(value.ops[0], ast.Lt)
    ):
        return False
    bound = _integer_constant(value.comparators[0], constants)
    # Unlike proof-only normalization, an emitted rewrite must preserve every
    # narrowing cast, including nested casts and the comparison's Int32 domain.
    if bound is None or not 0 <= bound < 1 << 31:
        return False
    index = value.left
    while (
        isinstance(index, ast.Call)
        and ast.unparse(index.func) in {"cutlass.Int32", "cutlass.Int64"}
        and len(index.args) == 1
        and not index.keywords
    ):
        index = index.args[0]
    if not (
        isinstance(index, ast.Subscript)
        and isinstance(index.value, ast.Call)
        and ast.unparse(index.value.func) == "cute.arch.thread_idx"
        and not index.value.args
        and not index.value.keywords
        and isinstance(index.slice, ast.Constant)
        and type(index.slice.value) is int
        and 0 <= index.slice.value < 3
    ):
        return False
    # All values of this original launch axis fit each supported index cast.
    return thread_dims[index.slice.value] <= bound


def _direct_load(
    value: ast.expr,
    dtype: str,
    tensors: Mapping[str, GatherTensorFacts],
) -> tuple[str, tuple[ast.expr, ...], ast.expr] | None:
    predicate: ast.expr = ast.Constant(True)
    load = value
    if isinstance(value, ast.IfExp):
        zero = value.orelse
        if not (
            isinstance(zero, ast.Call)
            and ast.unparse(zero.func) == dtype
            and len(zero.args) == 1
            and not zero.keywords
            and isinstance(zero.args[0], ast.Constant)
            and type(zero.args[0].value) in (int, float)
            and zero.args[0].value == 0
        ):
            return None
        load, predicate = value.body, value.test
    if not (
        isinstance(load, ast.Call)
        and isinstance(load.func, ast.Attribute)
        and load.func.attr == "load"
        and not load.args
        and not load.keywords
        and (pointer := _pointer_indices(load.func.value, tensors)) is not None
    ):
        return None
    return *pointer, predicate


def _integer_recipe(value: ast.expr, tensors: Mapping[str, GatherTensorFacts]) -> bool:
    """Admit exact integer/boolean replay without floating-point contraction."""
    integer_calls = {
        "cutlass.Int32",
        "operator.add",
        "operator.sub",
        "operator.mul",
        "operator.floordiv",
        "operator.mod",
        "operator.and_",
        "operator.or_",
        "operator.xor",
        "operator.lshift",
        "operator.rshift",
        "operator.lt",
        "operator.le",
        "operator.gt",
        "operator.ge",
        "operator.eq",
        "operator.ne",
        "operator.neg",
        "operator.pos",
        "operator.invert",
        "cute.arch.block_idx",
    }
    for node in ast.walk(value):
        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "cute.arch.block_idx"
                and not node.value.args
                and not node.value.keywords
                and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int
                and node.slice.value == 0
            ):
                continue
            metadata_tensor = next(
                (
                    name
                    for name in tensors
                    if ast.unparse(node.value)
                    in {
                        f"{name}.shape",
                        f"{name}.stride",
                        f"{name}.layout.shape",
                        f"{name}.layout.stride",
                    }
                ),
                None,
            )
            if (
                metadata_tensor is None
                or not isinstance(node.slice, ast.Constant)
                or type(node.slice.value) is not int
                or not 0 <= node.slice.value < len(tensors[metadata_tensor].shape)
            ):
                # Direct tensor indexing is a memory read too. This envelope
                # models its loads only through the typed .load() path below;
                # allowing an arbitrary subscript would hide floating inputs.
                return False
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "load":
                pointer = _pointer_indices(node.func.value, tensors)
                if pointer is None or tensors[pointer[0]].dtype != "cutlass.Int32":
                    return False
            elif ast.unparse(node.func) not in integer_calls:
                return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Pow)):
            return False
        if isinstance(node, ast.Constant) and type(node.value) not in (int, bool):
            return False
    return True


def _find_store_path(
    body: list[ast.stmt],
) -> list[tuple[list[ast.stmt], int]] | None:
    paths = []
    for index, statement in enumerate(body):
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "store"
        ):
            paths.append([(body, index)])
        if isinstance(statement, (ast.For, ast.If)):
            for branch in (statement.body, statement.orelse):
                if path := _find_store_path(branch):
                    paths.append([(body, index), *path])
    return paths[0] if len(paths) == 1 else None


def _scalar_range(loop: ast.For) -> tuple[ast.expr, ast.expr, ast.expr] | None:
    iterator = loop.iter
    if (
        not isinstance(iterator, ast.Call)
        or ast.unparse(iterator.func) not in {"range", "cutlass.range"}
        or len(iterator.args) != 3
        or loop.orelse
    ):
        return None
    return iterator.args[0], iterator.args[1], iterator.args[2]


def _linear(value: ast.expr) -> dict[str, int] | None:
    if isinstance(value, ast.Constant) and type(value.value) is int:
        return {"": value.value} if value.value else {}
    if isinstance(value, ast.Name):
        return {value.id: 1}
    if not isinstance(value, ast.BinOp):
        return None
    left, right = _linear(value.left), _linear(value.right)
    if left is None or right is None:
        return None
    if isinstance(value.op, (ast.Add, ast.Sub)):
        sign = 1 if isinstance(value.op, ast.Add) else -1
        result = dict(left)
        for key, number in right.items():
            result[key] = result.get(key, 0) + sign * number
        return {key: number for key, number in result.items() if number}
    if isinstance(value.op, ast.Mult):
        if set(left) <= {""}:
            return {key: number * left.get("", 0) for key, number in right.items()}
        if set(right) <= {""}:
            return {key: number * right.get("", 0) for key, number in left.items()}
    return None


def analyze_gathered_mma_region(
    body: list[ast.stmt],
    site: CollectiveMmaSite,
    *,
    boundary_names: set[str],
    tensors: Mapping[str, GatherTensorFacts],
    constants: Mapping[str, int],
    thread_dims: tuple[int, int, int],
    disjoint_pairs: set[frozenset[str]],
) -> GatheredMmaRegion | None:
    """Prove the first gathered pipeline envelope, retaining scalar recipes.

    Constants must be constexpr values or exact input-size facts guarded by
    dispatch. All tensor spans must fit signed 32-bit element addressing, which
    prevents TMA's wider address arithmetic from changing source overflow.
    """
    from .collective_matmul import _marker_path
    from .collective_matmul import _region_write_roots
    from .collective_matmul import _tensor_roots

    proof_names = {
        "_gather_block",
        "_gather_group",
        *(f"_gather_thread_{dimension}" for dimension in range(3)),
    }
    if boundary_names & proof_names or any(
        isinstance(node, ast.Name) and node.id in proof_names
        for statement in body
        for node in ast.walk(statement)
    ):
        return None
    if (
        not site.zero_seed
        or site.grid_row_lane is not None
        or site.synthetic_k_lane is not None
        or site.k_factor != 1
        or (site.bm, site.bk) != (128, 64)
        or site.bn not in (32, 64)
        or thread_dims
        != tuple(
            {site.m_axis: 128 // site.bn, site.n_axis: site.bn}.get(axis, 1)
            for axis in range(3)
        )
    ):
        return None
    marker_path = _marker_path(body, site.identity)
    store_path = _find_store_path(body)
    if marker_path is None or store_path is None:
        return None
    if (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
            for statement in body
            for node in ast.walk(statement)
        )
        != 1
    ):
        return None
    writes = _region_write_roots(body, site.identity, allow_relaxed_atomics=False)
    reads = _tensor_roots(body, "load", tensor_names=frozenset(tensors))
    if (
        writes is None
        or len(writes) != 1
        or reads is None
        or any(
            frozenset((read, write)) not in disjoint_pairs
            for read in reads
            for write in writes
        )
    ):
        return None
    statements = [part[index] for part, index in marker_path]
    m_positions = [
        i
        for i, statement in enumerate(statements)
        if isinstance(statement, ast.For)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == site.m_offset
    ]
    k_positions = [
        i
        for i, statement in enumerate(statements)
        if isinstance(statement, ast.For)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == site.k_offset
    ]
    if len(m_positions) != 1 or len(k_positions) != 1:
        return None
    m_depth, k_depth = m_positions[0], k_positions[0]
    if not m_depth < k_depth or any(
        not isinstance(value, ast.If) for value in statements[:m_depth]
    ):
        return None
    m_loop, k_loop = statements[m_depth], statements[k_depth]
    assert isinstance(m_loop, ast.For) and isinstance(k_loop, ast.For)
    prefix_before_m = [
        statement
        for part, index in marker_path[: m_depth + 1]
        for statement in part[:index]
    ]
    if set(ReadWrites.from_list(prefix_before_m).writes) & set(
        ReadWrites.from_list(m_loop.body).writes
    ):
        return None
    if not any(part[index] is m_loop for part, index in store_path):
        return None
    if any(
        not isinstance(value, ast.For) for value in statements[m_depth : k_depth + 1]
    ):
        return None
    if k_depth != m_depth + 2 or len(statements) != k_depth + 3:
        return None
    m_lane, k_lane = statements[m_depth + 1], statements[k_depth + 1]
    if not (
        isinstance(m_lane, ast.For)
        and isinstance(k_lane, ast.For)
        and isinstance(m_lane.target, ast.Name)
        and isinstance(k_lane.target, ast.Name)
    ):
        return None
    # The replacement scatters every admitted M/N coordinate exactly once.
    # A store nested in another loop (including a zero-trip loop), or placed
    # after the scalar M-lane traversal, has a different execution domain.
    store_loops = [
        part[index] for part, index in store_path if isinstance(part[index], ast.For)
    ]
    if len(store_loops) != 2 or not (
        store_loops[0] is m_loop and store_loops[1] is m_lane
    ):
        return None
    m_range, k_range = _scalar_range(m_loop), _scalar_range(k_loop)
    if m_range is None or k_range is None:
        return None
    if any(
        not isinstance(value, ast.Call)
        or ast.unparse(value.func) != "cutlass.Int32"
        or len(value.args) != 1
        or value.keywords
        for value in (*m_range, *k_range)
    ):
        return None
    dominating = [
        statement for part, index in marker_path for statement in part[:index]
    ]
    scalar_boundaries = boundary_names | {site.m_index, site.n_index, site.k_index}
    marker_statement = statements[-1]
    assert isinstance(marker_statement, ast.Assign)
    if len(marker_statement.targets) != 1 or not isinstance(
        marker_statement.targets[0], ast.Name
    ):
        return None
    accumulator = marker_statement.targets[0].id
    marker = marker_statement.value
    assert isinstance(marker, ast.Call)
    if (
        len(marker.args) != 4
        or marker.keywords
        or not isinstance(marker.args[3], ast.Name)
        or sum(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == accumulator
            for node in ast.walk(m_loop)
        )
        != 2
    ):
        return None
    # Generated phi aliases may copy the current accumulator at the start of
    # this exact scalar K iteration. Copies made outside that iteration could
    # capture a stale value and are not a carried-accumulator proof.
    current_iteration = marker_path[-1][0][: marker_path[-1][1]]
    carried = _expanded(
        marker.args[3], current_iteration, scalar_boundaries | {accumulator}
    )
    if not isinstance(carried, ast.Name) or carried.id != accumulator:
        return None
    before_k = [
        statement
        for part, index in marker_path[: k_depth + 1]
        for statement in part[:index]
    ]
    seed = _expanded(_expr(accumulator), before_k, scalar_boundaries)
    if not (
        isinstance(seed, ast.Call)
        and ast.unparse(seed.func) == "cutlass.Float32"
        and len(seed.args) == 1
        and not seed.keywords
        and isinstance(seed.args[0], ast.Constant)
        and type(seed.args[0].value) in (int, float)
        and seed.args[0].value == 0
    ):
        return None
    expanded = [
        _expanded(value, dominating, scalar_boundaries) for value in marker.args[1:3]
    ]
    if any(value is None for value in expanded):
        return None
    a_value, b_value = expanded
    assert a_value is not None and b_value is not None
    a_load, b_load = (
        _direct_load(a_value, site.dtype, tensors),
        _direct_load(b_value, site.dtype, tensors),
    )
    if a_load is None or b_load is None:
        return None
    a_name, a_indices, a_mask = a_load
    b_name, b_indices, b_mask = b_load
    if len(a_indices) != 2 or len(b_indices) != 3:
        return None
    a_facts, b_facts = tensors[a_name], tensors[b_name]
    output_name = next(iter(writes))
    if output_name not in tensors:
        return None
    out_facts = tensors[output_name]
    e_size, k_size, n_size = b_facts.shape
    if (
        a_facts.dtype != site.dtype
        or b_facts.dtype != site.dtype
        or out_facts.dtype != site.dtype
        or min(*a_facts.shape, *b_facts.shape) <= 0
        or a_facts.shape[1] != k_size
        or out_facts.shape != (a_facts.shape[0], n_size)
        or a_facts.strides[1] != 1
        or b_facts.strides[2] != 1
        or a_facts.strides[0] < k_size
        or b_facts.strides[1] < n_size
        or b_facts.strides[0] < k_size * b_facts.strides[1]
        or out_facts.strides[1] != 1
        or out_facts.strides[0] < n_size
        or min(a_facts.alignment, b_facts.alignment, out_facts.alignment) < 16
        or a_facts.strides[0] % 8
        or b_facts.strides[0] % 8
        or b_facts.strides[1] % 8
    ):
        return None
    for facts in (a_facts, b_facts, out_facts):
        if (
            any(stride <= 0 for stride in facts.strides)
            or sum(
                (size - 1) * stride
                for size, stride in zip(facts.shape, facts.strides, strict=True)
            )
            >= 1 << 31
        ):
            return None

    def normal(value: ast.expr) -> ast.expr:
        return _normalize(value, constants, thread_dims)

    m_threads = 128 // site.bn
    for loop, extent in ((m_lane, 128 // m_threads), (k_lane, 64)):
        if (
            not isinstance(loop.iter, ast.Call)
            or ast.unparse(loop.iter.func) not in {"range", "cutlass.range"}
            or loop.iter.keywords
            or len(loop.iter.args) != 1
            or not _same(normal(loop.iter.args[0]), ast.Constant(extent))
            or loop.orelse
        ):
            return None
    coordinate_boundaries = boundary_names | {
        site.m_offset,
        site.n_offset,
        site.k_offset,
        m_lane.target.id,
        k_lane.target.id,
    }
    coordinate_forms = []
    for name in (site.m_index, site.n_index, site.k_index):
        expanded_coordinate = _expanded(_expr(name), dominating, coordinate_boundaries)
        if expanded_coordinate is None:
            return None
        coordinate_forms.append(_linear(normal(expanded_coordinate)))
    if (
        coordinate_forms[0]
        not in (
            {
                site.m_offset: 1,
                f"_gather_thread_{site.m_axis}": 128 // m_threads,
                m_lane.target.id: 1,
            },
            {
                site.m_offset: 1,
                f"_gather_thread_{site.m_axis}": 1,
                m_lane.target.id: m_threads,
            },
        )
        or coordinate_forms[1] != {site.n_offset: 1, f"_gather_thread_{site.n_axis}": 1}
        or coordinate_forms[2] != {site.k_offset: 1, k_lane.target.id: 1}
    ):
        return None
    n_origin = _expanded(_expr(site.n_offset), prefix_before_m, boundary_names)
    if n_origin is None or not _same(
        normal(n_origin), _expr(f"_gather_block // {e_size} * {site.bn}")
    ):
        return None

    normalized_m = [normal(value) for value in m_range]
    normalized_k = [normal(value) for value in k_range]
    if (
        not _same(normalized_m[0], ast.Constant(0))
        or not _same(normalized_m[2], ast.Constant(site.bm))
        or not _same(normalized_k[0], ast.Constant(0))
        or not _same(normalized_k[1], ast.Constant(k_size))
        or not _same(normalized_k[2], ast.Constant(site.bk))
        or not isinstance(normalized_m[1], (ast.Name, ast.Constant))
        or isinstance(normalized_m[1], ast.Name)
        and normalized_m[1].id not in boundary_names
    ):
        return None

    m_coordinate, n_coordinate, k_coordinate = (
        _expr(value) for value in (site.m_index, site.n_index, site.k_index)
    )
    row = a_indices[0]
    # Tensor-index domain masks can guard the metadata load itself. Remove only
    # thread predicates proved true for the original launch before replaying
    # this recipe under the replacement layout; retain its casts and bounds.
    for node in ast.walk(row):
        if isinstance(node, ast.Compare) and _full_thread_mask(
            node, constants, thread_dims
        ):
            row = _replace_tree(row, node, ast.Constant(True))
    normalized_row = normal(row)
    group = b_indices[0]
    normalized_group = normal(group)
    expected_group = _expr(f"_gather_block % {e_size}")
    if (
        not _same(normal(a_indices[1]), k_coordinate)
        or not _same(normal(b_indices[1]), k_coordinate)
        or not _same(normal(b_indices[2]), n_coordinate)
        or not _same(normalized_group, expected_group)
        or not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            for node in ast.walk(row)
        )
    ):
        return None
    row_key = _replace_tree(normalized_row, expected_group, _expr("_gather_group"))
    if {
        site.n_index,
        site.k_index,
        "_gather_block",
        *(f"_gather_thread_{d}" for d in range(3)),
    } & _read_names(row_key) or any(
        isinstance(node, ast.Call) and ast.unparse(node.func).startswith("cute.arch.")
        for node in ast.walk(row_key)
    ):
        return None
    if not _integer_recipe(row, tensors):
        return None
    if _read_names(row) - (
        set(tensors)
        | set(constants)
        | _read_names(m_range[1])
        | {site.m_index, "cutlass", "cute", "operator"}
    ):
        return None
    m_mask = ast.Compare(
        left=m_coordinate, ops=[ast.Lt()], comparators=[normalized_m[1]]
    )
    k_mask = _expr(f"{site.k_index} < {k_size}")
    n_mask = _expr(f"{site.n_index} < {n_size}")
    if not _matching_conjunction(
        normal(a_mask),
        [
            ast.Compare(
                left=normalized_row,
                ops=[ast.Lt()],
                comparators=[ast.Constant(a_facts.shape[0])],
            ),
            *([k_mask] if k_size % site.bk else []),
        ],
        # The exact logical-M bound is also required on the scatter below.
        # No observable row can depend on loads excluded by this domain mask.
        optional=[m_mask, *([k_mask] if k_size % site.bk == 0 else [])],
    ):
        return None
    if not _matching_conjunction(
        normal(b_mask),
        [
            *([k_mask] if k_size % site.bk else []),
            *([n_mask] if n_size % site.bn else []),
        ],
        optional=[
            *([k_mask] if k_size % site.bk == 0 else []),
            *([n_mask] if n_size % site.bn == 0 else []),
        ],
    ):
        return None
    store_statement = store_path[-1][0][store_path[-1][1]]
    assert isinstance(store_statement, ast.Expr) and isinstance(
        store_statement.value, ast.Call
    )
    store = store_statement.value
    assert isinstance(store.func, ast.Attribute)
    if len(store.args) != 1 or store.keywords:
        return None
    store_prefix = [
        value
        for part, index in store_path
        for value in part[:index]
        if value is not k_loop
    ]
    store_boundaries = scalar_boundaries | {accumulator}
    address = _expanded(store.func.value, store_prefix, store_boundaries)
    value = _expanded(store.args[0], store_prefix, store_boundaries)
    if address is None or value is None:
        return None
    output = _pointer_indices(address, tensors)
    if (
        output is None
        or output[0] != output_name
        or len(output[1]) != 2
        or not _same(normal(output[1][0]), normalized_row)
        or not _same(normal(output[1][1]), n_coordinate)
    ):
        return None
    while (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) == site.dtype
        and len(value.args) == 1
        and not value.keywords
    ):
        value = value.args[0]
    if not isinstance(value, ast.Name) or value.id != accumulator:
        return None
    predicates: list[ast.expr] = []
    inside_m = False
    for depth, (part, index) in enumerate(store_path[:-1]):
        statement = part[index]
        inside_m |= statement is m_loop
        if not isinstance(statement, ast.If):
            continue
        prefix = [
            item
            for previous, position in store_path[: depth + 1]
            for item in previous[:position]
            if item is not k_loop
        ]
        predicate = _expanded(statement.test, prefix, scalar_boundaries)
        if predicate is None:
            return None
        if store_path[depth + 1][0] is statement.orelse:
            predicate = ast.UnaryOp(op=ast.Not(), operand=predicate)
        if inside_m:
            predicates.extend(_conjuncts(predicate))
        else:
            # This enclosing condition remains around a 288x1x1 CTA. Even a
            # thread predicate tautological for the original launch may become
            # divergent after that change, so its raw AST must be independent
            # of thread/warp geometry before proof-only simplification.
            if any(
                isinstance(node, ast.Call)
                and ast.unparse(node.func).startswith("cute.arch.")
                and ast.unparse(node.func) != "cute.arch.block_idx"
                for node in ast.walk(predicate)
            ):
                return None
            uniform = _replace_tree(
                normal(predicate), expected_group, _expr("_gather_group")
            )
            if {
                site.m_index,
                site.n_index,
                site.k_index,
                "_gather_block",
                *(f"_gather_thread_{d}" for d in range(3)),
            } & _read_names(uniform) or any(
                isinstance(node, ast.Call)
                and ast.unparse(node.func).startswith("cute.arch.")
                for node in ast.walk(uniform)
            ):
                return None
    if (
        n_size % site.bn != 0
        and not any(_same(normal(part), n_mask) for part in predicates)
        or not any(_same(normal(part), m_mask) for part in predicates)
    ):
        return None
    row_predicates = [
        value
        for value in predicates
        if not _same(normal(value), n_mask)
        and not _same(normal(value), ast.Constant(True))
    ]
    row_predicate = (
        ast.BoolOp(op=ast.And(), values=row_predicates)
        if len(row_predicates) > 1
        else row_predicates[0]
    )
    if not _integer_recipe(row_predicate, tensors) or _read_names(row_predicate) - (
        set(tensors)
        | set(constants)
        | _read_names(m_range[1])
        | {site.m_index, "cutlass", "cute", "operator"}
    ):
        return None
    # Only complete conjuncts proved true for every original thread were
    # removed above. Do not retain nested geometry-dependent expressions and
    # evaluate them under the replacement epilogue's different thread layout.
    if any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func).startswith("cute.arch.")
        and ast.unparse(node.func) != "cute.arch.block_idx"
        for node in ast.walk(row_predicate)
    ):
        return None
    row_only = _replace_tree(
        normal(row_predicate), expected_group, _expr("_gather_group")
    )
    if {
        site.n_index,
        site.k_index,
        "_gather_block",
        *(f"_gather_thread_{d}" for d in range(3)),
    } & _read_names(row_only) or any(
        isinstance(node, ast.Call) and ast.unparse(node.func).startswith("cute.arch.")
        for node in ast.walk(row_only)
    ):
        return None
    m_parent, m_position = marker_path[m_depth]
    if not isinstance(m_parent, list):
        return None
    if set(ReadWrites.from_list(m_parent[m_position + 1 :]).reads) & set(
        ReadWrites.from_ast(m_loop).writes
    ):
        return None
    return GatheredMmaRegion(
        a_name,
        b_name,
        output_name,
        m_range[1],
        row,
        row_predicate,
        group,
        m_loop,
        m_parent,
        m_position,
        e_size,
        n_size,
        k_size,
        frozenset(reads),
    )
