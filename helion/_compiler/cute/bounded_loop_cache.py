"""Proofs and guarded dispatch for a one-logical-tile reload variant.

The bounded-cache code generator specializes typed zero-based sweeps, applies
the normal reduction scheduling, then caches identical raw loads before
vector I/O. ``attach_variant`` supplies
the required host guard and preserves the original generated fallback.

The cache is private to a CUDA thread. Its index is the mixed-radix index of
the original lexical lane loops, never a guessed physical thread coordinate.
"""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from .collective_matmul import _region_write_roots
from .fuse_two_pass_loads import _range_bounds
from .fuse_two_pass_loads import _tensor_arg_roots
from .persistent_branch_vec import _binding_write_roots
from .persistent_branch_vec import _freeze_definition
from .resident_sequence import _clone
from .scalar_recipe import _GLOBALS
from .scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping


_INTEGER_CASTS = frozenset({"cutlass.Int32", "cutlass.Int64"})
_DTYPE_BYTES = {
    "cutlass.Float16": 2,
    "cutlass.BFloat16": 2,
    "cutlass.Float32": 4,
    "cutlass.Int32": 4,
    "cutlass.Uint32": 4,
}
# These compiler helpers do not write global memory. Their original calls and
# arguments are never rewritten; this list is used only by the effects proof.
_REDUCTIONS = frozenset(
    {
        "_cute_grouped_reduce_warp",
        "_cute_grouped_reduce_shared_two_stage",
        "_cute_grouped_reduce_shared_serial",
        "_cute_grouped_reduce_shared_tree",
        "cute.arch.warp_reduction_sum",
        "cute.arch.warp_reduction_max",
        "cute.arch.warp_reduction_min",
        "cute.arch.fmax",
        "cute.arch.fmin",
    }
)
_MAX_INPUT_NODES = 16384
_MAX_CACHE_ELEMENTS = 128
_MAX_TOTAL_CACHE_BYTES = 1024


class _Decline(Exception):
    pass


def _text(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _expr(text: str) -> ast.expr:
    return cast("ast.expr", expr_from_string(text))


def _copies(body: list[ast.stmt]) -> list[ast.stmt]:
    return [_clone(statement) for statement in body]


def _strip_integer_casts(node: ast.expr) -> ast.expr:
    while (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) in _INTEGER_CASTS
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    return node


def _integer(node: ast.expr, constants: Mapping[str, int]) -> int | None:
    value = _strip_integer_casts(node)
    if isinstance(value, ast.Constant) and type(value.value) is int:
        return value.value
    if isinstance(value, ast.Name):
        return constants.get(value.id)
    return None


@dataclasses.dataclass(frozen=True)
class BoundedExtent:
    """A runtime integer argument, or a constant already inside the bound."""

    argument: str | None
    constant: int | None
    upper: int

    def predicate(self, argument: ast.expr | None = None) -> ast.expr:
        if self.argument is None:
            return _expr("True")
        value = argument if argument is not None else _expr(self.argument)
        return create(
            ast.Compare,
            left=create(ast.Constant, value=0),
            ops=[create(ast.Lt), create(ast.LtE)],
            comparators=[value, create(ast.Constant, value=self.upper)],
        )


@dataclasses.dataclass(frozen=True)
class Sweep:
    target: str
    iterator: str
    integer_type: str


@dataclasses.dataclass(frozen=True)
class BoundedLoopPlan:
    extent: BoundedExtent
    sweeps: tuple[Sweep, ...]
    # These are the actual launcher dimensions, recorded for provenance only.
    # Register slot ownership does not flatten x/y/z.
    launch_block: tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class SpecializedLoops:
    body: list[ast.stmt]
    plan: BoundedLoopPlan


def specialize_single_tile_sweeps(
    body: list[ast.stmt],
    *,
    integer_arguments: Collection[str],
    constexpr_values: Mapping[str, int],
    launch_block: tuple[int, int, int],
) -> SpecializedLoops | None:
    """Prepare a fast variant; it is valid only behind ``plan.extent``.

    Admission is structural: at least two root sibling loops with identical
    typed zero/start/step semantics and one immutable integer extent. No size
    hint is treated as a runtime bound. Unknown/narrowing casts, dynamic starts,
    negative steps, branches, and oversized proof DAGs are not specialized.
    """
    if (
        any(type(size) is not int or size < 1 for size in launch_block)
        or math.prod(launch_block) > 1024
        or sum(1 for statement in body for _ in ast.walk(statement)) > _MAX_INPUT_NODES
    ):
        return None
    writes = set().union(*(_binding_write_roots(statement) for statement in body))
    if writes & (_GLOBALS | _REDUCTIONS):
        return None
    groups: dict[tuple[BoundedExtent, str], list[int]] = {}
    for ordinal, statement in enumerate(body):
        if (
            not isinstance(statement, ast.For)
            or not isinstance(statement.target, ast.Name)
            or statement.orelse
            or not isinstance(statement.iter, ast.Call)
            or ast.unparse(statement.iter.func) != "range"
            or statement.iter.keywords
        ):
            continue
        bounds = _range_bounds(statement.iter)
        if bounds is None:
            continue
        start, stop, step = bounds
        if not all(
            isinstance(value, ast.Call)
            and ast.unparse(value.func) in _INTEGER_CASTS
            and len(value.args) == 1
            and not value.keywords
            for value in bounds
        ):
            continue
        types = {ast.unparse(cast("ast.Call", value).func) for value in bounds}
        block = _integer(step, constexpr_values)
        if (
            len(types) != 1
            or _integer(start, constexpr_values) != 0
            or block is None
            or not 1 <= block < 1 << 31
            or statement.target.id
            in _binding_write_roots(ast.Module(body=statement.body, type_ignores=[]))
        ):
            continue
        leaf = _strip_integer_casts(stop)
        constant = _integer(stop, constexpr_values)
        if constant is not None:
            if not 0 < constant <= block:
                continue
            extent = BoundedExtent(None, constant, block)
        elif (
            isinstance(leaf, ast.Name)
            and leaf.id in integer_arguments
            and leaf.id not in writes
        ):
            extent = BoundedExtent(leaf.id, None, block)
        else:
            continue
        groups.setdefault((extent, types.pop()), []).append(ordinal)
    candidates = [
        (key, indices) for key, indices in groups.items() if len(indices) >= 2
    ]
    if len(candidates) != 1:
        return None
    (extent, integer_type), indices = candidates[0]
    result = _copies(body)
    sweeps = []
    for index in indices:
        loop = result[index]
        assert isinstance(loop, ast.For)
        assert isinstance(loop.target, ast.Name)
        assert isinstance(loop.iter, ast.Call)
        # Keep the original typed range until normal reduction scheduling has
        # run. The positive bound proves the old and new iterator each yield
        # precisely one value, with the same integer type.
        loop.iter.args[1] = cast("ast.expr", _clone(loop.iter.args[2]))
        sweeps.append(Sweep(loop.target.id, _text(loop.iter), integer_type))
    return SpecializedLoops(
        result, BoundedLoopPlan(extent, tuple(sweeps), launch_block)
    )


@dataclasses.dataclass(frozen=True)
class OwnedFragment:
    """Non-escaping private storage emitted by this pass, for vector I/O."""

    name: str
    dtype: str
    elements: int
    accesses: frozenset[str]


@dataclasses.dataclass(frozen=True)
class CachedLoadOrigin:
    """Exact first producer justified by the complete immutable-load proof."""

    statement: ast.Assign
    expression: str
    fragment: str
    index: ast.expr
    dtype: str


@dataclasses.dataclass(frozen=True)
class CachedLoops:
    body: list[ast.stmt]
    fragments: tuple[OwnedFragment, ...]
    replaced_loads: int
    original_loads: int
    origins: tuple[CachedLoadOrigin, ...] = ()


@dataclasses.dataclass
class _Frame:
    loop: ast.For
    extent: int
    constexpr: bool
    integer_type: str
    coordinate: str = ""


@dataclasses.dataclass
class _Site:
    statement: ast.Assign
    container: list[ast.stmt]
    frames: tuple[_Frame, ...]
    key: str
    root: str
    dtype: str


def _frame(loop: ast.For) -> _Frame | None:
    if not isinstance(loop.target, ast.Name) or loop.orelse:
        return None
    bounds = _range_bounds(loop.iter)
    if bounds is None:
        return None
    start, stop, step = bounds
    count = _integer(stop, {})
    if (
        _integer(start, {}) != 0
        or _integer(step, {}) != 1
        or count is None
        or not 1 <= count <= _MAX_CACHE_ELEMENTS
        or loop.target.id
        in _binding_write_roots(ast.Module(body=loop.body, type_ignores=[]))
        or not isinstance(loop.iter, ast.Call)
        or loop.iter.keywords
    ):
        return None
    constexpr = ast.unparse(loop.iter.func) == "cutlass.range_constexpr"
    # A normal CuTe range defaults to Int32; explicitly typed ranges may use
    # Int64. Keep that width when a constexpr counter replaces the runtime
    # iterator, since later arithmetic can overflow before pointer addition.
    types = {
        ast.unparse(value.func)
        for value in bounds
        if isinstance(value, ast.Call) and ast.unparse(value.func) in _INTEGER_CASTS
    }
    if len(types) > 1:
        return None
    integer_type = types.pop() if types else "cutlass.Int32"
    return _Frame(loop, count, constexpr, integer_type)


def _write_roots(body: list[ast.stmt]) -> set[str] | None:
    """Use the existing fail-closed effect proof, with known reduction calls."""

    class ReduceCalls(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.expr:
            rewritten = self.generic_visit(node)
            assert isinstance(rewritten, ast.Call)
            node = rewritten
            if ast.unparse(node.func) not in _REDUCTIONS:
                return node
            for value in [*node.args, *(keyword.value for keyword in node.keywords)]:
                if (
                    build_recipe(value, [], set(ReadWrites.from_ast(value).reads))
                    is None
                ):
                    raise _Decline
            return _expr("cutlass.Float32(0)")

    try:
        probe = [cast("ast.stmt", ReduceCalls().visit(_clone(item))) for item in body]
    except _Decline:
        return None
    return _region_write_roots(probe, set())


def _load_dtype(value: ast.expr, dtypes: Mapping[str, str]) -> tuple[str, str] | None:
    load = value.body if isinstance(value, ast.IfExp) else value
    if not (
        isinstance(load, ast.Call)
        and isinstance(load.func, ast.Attribute)
        and load.func.attr == "load"
        and not load.args
        and not load.keywords
    ):
        return None
    roots = _tensor_arg_roots(load.func.value, set(dtypes))
    if roots is None or len(roots) != 1:
        return None
    root = next(iter(roots))
    dtype = dtypes[root]
    if dtype not in _DTYPE_BYTES:
        return None
    if isinstance(value, ast.IfExp) and not (
        isinstance(value.orelse, ast.Call)
        and ast.unparse(value.orelse.func) == dtype
        and len(value.orelse.args) == 1
        and not value.orelse.keywords
    ):
        return None
    return root, dtype


def _key(
    value: ast.expr,
    prefix: list[ast.stmt],
    coordinates: Mapping[str, ast.expr],
    arguments: Collection[str],
    constants: Mapping[str, int],
    mutable: Collection[str],
    reserved_names: Collection[str],
) -> str | None:
    recipe = build_recipe(
        value,
        prefix,
        set(arguments) | constants.keys() | coordinates.keys(),
        mutable_names=mutable,
    )
    if recipe is None:
        return None
    counter = 0
    used = set(reserved_names)

    def fresh(hint: str) -> str:
        nonlocal counter
        while True:
            counter += 1
            name = f"__bounded_recipe_{counter}_{hint}"
            if name not in used:
                used.add(name)
                return name

    assignments, result = recipe.emit(coordinates, fresh)
    definitions: dict[str, ast.expr] = {}
    for assignment in assignments:
        expanded = _freeze_definition(assignment.value, definitions)
        if expanded is None:
            return None
        target = assignment.targets[0]
        assert isinstance(target, ast.Name)
        definitions[target.id] = expanded
    expanded = _freeze_definition(result, definitions)
    if expanded is None:
        return None
    # Indirect loads, tensor subscripts, or opaque pointer aliases require a
    # separate invariant-memory proof. This bounded first implementation only
    # retains the one original scalar data read.
    loads = [
        node
        for node in ast.walk(expanded)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
    ]
    if len(loads) != 1 or any(
        isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        for node in ast.walk(expanded)
    ):
        return None
    return _text(expanded)


def _identifier_names(tree: ast.AST) -> set[str]:
    """Include lexical bindings that do not appear as ast.Name nodes."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name is not None:
                names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            names.add(node.rest)
    return names


def cache_single_tile_loads(
    body: list[ast.stmt],
    plan: BoundedLoopPlan,
    *,
    argument_names: Collection[str],
    constexpr_values: Mapping[str, int],
    tensor_dtypes: Mapping[str, str],
    proven_disjoint_tensor_pairs: Collection[frozenset[str]],
    rename_groups: Mapping[str, str],
) -> CachedLoops | None:
    """Cache raw loads after reduction scheduling and before vector I/O.

    Keys retain every cast, mask and pointer-addition boundary. Every cache
    writer is an unconditional statement in positive static lexical loops;
    every reader has exactly the same loop-coordinate domains. All loaded
    roots must be disjoint from every global write in the complete function.
    """
    if sum(1 for statement in body for _ in ast.walk(statement)) > _MAX_INPUT_NODES:
        return None
    writes = set().union(*(_binding_write_roots(statement) for statement in body))
    if writes & (
        set(argument_names) | constexpr_values.keys() | _GLOBALS | _REDUCTIONS
    ):
        return None
    root_proof = _write_roots(body)
    if root_proof is None:
        return None
    written_roots = frozenset(root_proof)
    result = _copies(body)
    signatures = Counter((sweep.target, sweep.iterator) for sweep in plan.sweeps)
    selected = [
        (index, loop)
        for index, loop in enumerate(result)
        if isinstance(loop, ast.For)
        and isinstance(loop.target, ast.Name)
        and (loop.target.id, _text(loop.iter)) in signatures
    ]
    if (
        Counter(
            (cast("ast.Name", loop.target).id, _text(loop.iter)) for _, loop in selected
        )
        != signatures
    ):
        return None
    sites: list[_Site] = []
    mutable = set(rename_groups) | set(rename_groups.values())
    reserved = (
        _identifier_names(ast.Module(body=result, type_ignores=[]))
        | set(argument_names)
        | constexpr_values.keys()
        | mutable
        | _GLOBALS
        | _REDUCTIONS
    )
    proof_names = set(reserved)
    coordinate_names: dict[int, str] = {}

    def coordinate_name(depth: int) -> str:
        if depth not in coordinate_names:
            name = f"__bounded_coordinate_{depth}"
            while name in proof_names:
                name += "_"
            proof_names.add(name)
            coordinate_names[depth] = name
        return coordinate_names[depth]

    def collect(
        container: list[ast.stmt],
        prefix: list[ast.stmt],
        frames: tuple[_Frame, ...],
        coordinates: dict[str, ast.expr],
    ) -> None:
        for index, statement in enumerate(container):
            dominating = [*prefix, *container[:index]]
            if isinstance(statement, ast.For):
                frame = _frame(statement)
                if (
                    frame is None
                    or math.prod(item.extent for item in (*frames, frame))
                    > _MAX_CACHE_ELEMENTS
                ):
                    continue
                assert isinstance(statement.target, ast.Name)
                child_coordinates = {
                    **coordinates,
                    statement.target.id: _expr(coordinate_name(len(frames))),
                }
                collect(statement.body, dominating, (*frames, frame), child_coordinates)
            elif (
                frames
                and isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and (typed := _load_dtype(statement.value, tensor_dtypes)) is not None
            ):
                root, dtype = typed
                if any(
                    root == written
                    or frozenset((root, written)) not in proven_disjoint_tensor_pairs
                    for written in written_roots
                ):
                    continue
                key = _key(
                    statement.value,
                    dominating,
                    coordinates,
                    argument_names,
                    constexpr_values,
                    mutable,
                    proof_names,
                )
                if key is not None:
                    domains = tuple(
                        (frame.extent, frame.constexpr, frame.integer_type)
                        for frame in frames
                    )
                    sites.append(
                        _Site(
                            statement,
                            container,
                            frames,
                            repr((domains, dtype, key)),
                            root,
                            dtype,
                        )
                    )

    for index, loop in selected:
        target = cast("ast.Name", loop.target).id
        sweep = next(
            item
            for item in plan.sweeps
            if item.target == target and item.iterator == _text(loop.iter)
        )
        collect(
            loop.body, result[:index], (), {target: _expr(f"{sweep.integer_type}(0)")}
        )
    groups: dict[str, list[_Site]] = {}
    for site in sites:
        groups.setdefault(site.key, []).append(site)
    groups = {key: group for key, group in groups.items() if len(group) > 1}
    if not groups:
        return None
    footprint = sum(
        math.prod(frame.extent for frame in group[0].frames)
        * _DTYPE_BYTES[group[0].dtype]
        for group in groups.values()
    )
    if footprint > _MAX_TOTAL_CACHE_BYTES:
        return None
    used = set(proof_names)

    def fresh(prefix: str) -> str:
        while prefix in used:
            prefix += "_"
        used.add(prefix)
        return prefix

    active_frames = {
        id(frame.loop): frame
        for group in groups.values()
        for site in group
        for frame in site.frames
    }
    for frame in active_frames.values():
        assert isinstance(frame.loop.target, ast.Name)
        frame.coordinate = (
            frame.loop.target.id if frame.constexpr else fresh("_helion_cache_lane")
        )
    declarations = []
    fragments = []
    origins = []
    replaced = 0
    for group in groups.values():
        first = group[0]
        name = fresh("_helion_bounded_cache")
        size = math.prod(frame.extent for frame in first.frames)
        declarations.append(
            statement_from_string(
                f"{name} = cute.make_rmem_tensor({size}, {first.dtype})"
            )
        )
        access_texts: set[str] = set()
        for ordinal, site in enumerate(group):
            index = "0"
            for frame in site.frames:
                index = f"({index}) * {frame.extent} + {frame.coordinate}"
            reference = _expr(f"{name}[{index}]")
            assert isinstance(reference, ast.Subscript)
            if ordinal == 0:
                origins.append(
                    CachedLoadOrigin(
                        site.statement,
                        _text(site.statement.value),
                        name,
                        reference.slice,
                        first.dtype,
                    )
                )
                reference.ctx = ast.Store()
                target = cast("ast.Name", site.statement.targets[0]).id
                store = create(ast.Assign, targets=[reference], value=_expr(target))
                site.container.insert(site.container.index(site.statement) + 1, store)
            else:
                site.statement.value = reference
                replaced += 1
            access_texts.add(_text(reference))
        fragments.append(
            OwnedFragment(name, first.dtype, size, frozenset(access_texts))
        )
    for frame in active_frames.values():
        if frame.constexpr:
            continue
        original_name = cast("ast.Name", frame.loop.target).id
        frame.loop.target = create(ast.Name, id=frame.coordinate, ctx=ast.Store())
        frame.loop.iter = _expr(f"cutlass.range_constexpr({frame.extent})")
        frame.loop.body.insert(
            0,
            statement_from_string(
                f"{original_name} = {frame.integer_type}({frame.coordinate})"
            ),
        )
        if getattr(frame.loop, HELION_LANE_LOOP_VAR_ATTR, None) is not None:
            setattr(frame.loop, HELION_LANE_LOOP_VAR_ATTR, frame.coordinate)
    for _, loop in selected:
        target = cast("ast.Name", loop.target).id
        sweep = next(
            item
            for item in plan.sweeps
            if item.target == target and item.iterator == _text(loop.iter)
        )
        loop.target = create(ast.Name, id=fresh("_helion_single_tile"), ctx=ast.Store())
        loop.iter = _expr("cutlass.range_constexpr(1)")
        loop.body.insert(
            0, statement_from_string(f"{target} = {sweep.integer_type}(0)")
        )
    return CachedLoops(
        [*declarations, *result], tuple(fragments), replaced, len(sites), tuple(origins)
    )


def private_fragment_accesses(
    body: list[ast.stmt], fragments: Collection[OwnedFragment]
) -> frozenset[str]:
    """Validate nonescape/unique allocation before extending vector analysis.

    Only exact accesses created by this pass receive the private-memory fact.
    A caller cannot make an arbitrary tensor subscript pure by naming a buffer.
    """
    by_name = {fragment.name: fragment for fragment in fragments}
    if len(by_name) != len(fragments):
        return frozenset()
    initializers: Counter[str] = Counter()
    observed: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in by_name
                for target in node.targets
            ):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    return frozenset()
                fragment = by_name[node.targets[0].id]
                if (
                    _text(node.value)
                    != _text(
                        _expr(
                            f"cute.make_rmem_tensor({fragment.elements}, {fragment.dtype})"
                        )
                    )
                    or node is not statement
                ):
                    return frozenset()
                initializers[fragment.name] += 1
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, ast.Name) or child.id not in by_name:
                    continue
                if isinstance(child.ctx, ast.Store):
                    if not isinstance(node, ast.Assign):
                        return frozenset()
                elif (
                    not isinstance(node, ast.Subscript)
                    or node.value is not child
                    or _text(node) not in by_name[child.id].accesses
                ):
                    return frozenset()
                else:
                    if initializers[child.id] != 1:
                        return frozenset()
                    observed.add(_text(node))
    if any(initializers[name] != 1 for name in by_name):
        return frozenset()
    expected = frozenset().union(*(fragment.accesses for fragment in fragments))
    return expected if observed == expected else frozenset()


def attach_variant(
    original_source: str,
    fast_source: str,
    plan: BoundedLoopPlan,
    *,
    kernel_name: str,
) -> str | None:
    """Attach guarded dispatch while preserving the fallback kernel and caller.

    Decline tensor-map wrappers, multiple launches, changed signatures,
    constants, import bindings, and kernel metadata. Callers preserve both
    variant identities through precompile and serialization.
    """
    original = ast.parse(original_source)
    fast = ast.parse(fast_source)
    old_functions = {
        item.name: item for item in original.body if isinstance(item, ast.FunctionDef)
    }
    new_functions = {
        item.name: item for item in fast.body if isinstance(item, ast.FunctionDef)
    }
    if kernel_name not in old_functions or kernel_name not in new_functions:
        return None
    old_kernel = old_functions[kernel_name]
    new_kernel = new_functions[kernel_name]
    if _text(old_kernel.args) != _text(new_kernel.args) or [
        _text(item) for item in old_kernel.decorator_list
    ] != [_text(item) for item in new_kernel.decorator_list]:
        return None
    old_imports = {
        _text(item): item
        for item in original.body
        if isinstance(item, (ast.Import, ast.ImportFrom))
    }
    old_other = [
        item
        for item in original.body
        if item is not old_kernel and not isinstance(item, (ast.Import, ast.ImportFrom))
    ]
    new_other = [
        item
        for item in fast.body
        if item is not new_kernel and not isinstance(item, (ast.Import, ast.ImportFrom))
    ]
    if _text(ast.Module(body=old_other, type_ignores=[])) != _text(
        ast.Module(body=new_other, type_ignores=[])
    ):
        return None
    old_names = _identifier_names(original)
    if "*" in old_names:
        return None
    extra_imports = []
    for item in fast.body:
        if (
            not isinstance(item, (ast.Import, ast.ImportFrom))
            or _text(item) in old_imports
        ):
            continue
        bindings = {alias.asname or alias.name.split(".")[0] for alias in item.names}
        if "*" in bindings or bindings & old_names:
            return None
        extra_imports.append(item)
        old_names.update(bindings)
    if any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == kernel_name
        for node in ast.walk(original)
    ):
        return None
    launches = [
        node
        for node in ast.walk(original)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == kernel_name
    ]
    if len(launches) != 1:
        return None
    launch = launches[0]
    block = next(
        (keyword.value for keyword in launch.keywords if keyword.arg == "block"), None
    )
    if block is None or _text(block) != _text(
        ast.parse(repr(plan.launch_block), mode="eval").body
    ):
        return None
    parameters = [argument.arg for argument in old_kernel.args.args]
    argument = None
    if plan.extent.argument is not None:
        if plan.extent.argument not in parameters:
            return None
        position = parameters.index(plan.extent.argument) + 2
        if position >= len(launch.args):
            return None
        argument = launch.args[position]
        # Do not duplicate a host-side call or other effect when building the
        # guard. Generated launchers expose integer sizes as local names.
        if not isinstance(argument, ast.Name):
            return None
    names = old_names | _identifier_names(fast)
    variant = f"{kernel_name}_bounded"
    while variant in names:
        variant += "_"
    new_kernel.name = variant
    launch.args[0] = ast.IfExp(
        test=plan.extent.predicate(argument),
        body=ast.Name(id=variant, ctx=ast.Load()),
        orelse=launch.args[0],
    )
    original.body[original.body.index(old_kernel) : original.body.index(old_kernel)] = (
        extra_imports
    )
    original.body.insert(original.body.index(old_kernel) + 1, new_kernel)
    return ast.unparse(ast.fix_missing_locations(original)) + "\n"
