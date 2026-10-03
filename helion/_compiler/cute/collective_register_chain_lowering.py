"""Admit a whole scalar contraction chain before changing its ownership."""

from __future__ import annotations

import ast
from itertools import count
from itertools import starmap
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

from ... import exc
from ..ast_read_writes import ReadWrites
from ..ast_read_writes import ast_rename
from ..compile_environment import CompileEnvironment
from .collective_epilogue import _load_pointer_roots
from .collective_epilogue import _RemoveFullThreadBounds
from .collective_epilogue import _scalar_epilogue
from .collective_epilogue import prune_pure_scalar_statements
from .collective_matmul import MARKER
from .collective_matmul import CollectiveSiteAnalysis
from .collective_matmul import _copy_tensor_facts
from .collective_matmul import _lower_site
from .collective_matmul import _remove_dead_staged_operands
from .collective_matmul import _tensor_roots
from .collective_matmul import _uses_thread_coordinates
from .collective_register_chain import RegisterChainEpilogue
from .collective_register_chain import RegisterChainSite
from .collective_register_chain import emit_register_chain
from .collective_register_chain import plan_register_chain
from .contiguous_copy import CopyTensorFacts
from .proven_loop_bounds import _Context
from .proven_loop_bounds import _Proof
from .scalar_recipe import _GLOBALS
from .scalar_recipe import _clone
from .scalar_recipe import _read_names
from .scalar_recipe import build_recipe
from .simplify_proven_bounds import _assigned_names
from .thread_block_projection import flatten_thread_coordinates

if TYPE_CHECKING:
    from ..device_function import DeviceFunction


def _reject(reason: str) -> NoReturn:
    raise exc.BackendUnsupported("cute", f"warp register chain: {reason}")


def _clone_statement(statement: ast.stmt) -> ast.stmt:
    """Keep compiler annotations, including synthetic-lane ownership marks.

    Scalar recipe clones need only structural fields. A whole compiler region
    must also retain its opaque annotations for subsequent lowering passes.
    Analysis reads those metadata values without modifying them.
    """
    result = _clone(statement)
    for original, copied in zip(ast.walk(statement), ast.walk(result), strict=True):
        copied.__dict__.update(
            (name, value)
            for name, value in vars(original).items()
            if name not in original._fields
        )
    return result


def _terminal_arena_dead(body: list[ast.stmt], region: ast.For) -> bool:
    """Prove there is no device code after or enclosing the replaced region.

    The caller supplies the whole device body, not a nested statement list.
    The complete root row loop is replaced once, so its backedge disappears.
    The new arena is private to that replacement and cannot escape through the
    already-proved original scalar/global-output recipes. Any suffix, including
    an unknown call or another shared allocation, keeps the retirement barrier.
    """
    return (
        bool(body)
        and body[-1] is region
        and not region.orelse
        and sum(node is region for statement in body for node in ast.walk(statement))
        == 1
    )


def _constexpr_values(df: DeviceFunction) -> dict[str, int]:
    from ..device_function import ConstExprArg

    values = {}
    for argument in df.arguments:
        if isinstance(argument, ConstExprArg):
            value = ast.parse(argument.host_str(), mode="eval").body
            if isinstance(value, ast.Constant) and type(value.value) is int:
                values[argument.name] = value.value
    return values


def _packet_extent(
    analysis: CollectiveSiteAnalysis,
    constexprs: dict[str, int],
    threads: tuple[int, int, int],
) -> int:
    """Prove an invariant positive packet count, retaining each scalar mask.

    A tile-dependent stop may still execute exactly one full BK packet. The
    original scalar recipes, including their stop-derived predicates, remain
    unchanged. Intervals come from actual constexprs and architectural index
    bounds; tensor size hints and memory contents are never proof inputs.
    """
    proof = _Proof(threads)
    context = _Context()
    for name, value in constexprs.items():
        known = proof.value(ast.Constant(value), context)
        if known is not None:
            context.values[name] = known
    for statement in analysis.seed_dominating:
        value = (
            proof.value(statement.value, context)
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            else None
        )
        for name in _assigned_names(statement):
            context.values.pop(name, None)
        if value is not None:
            assert isinstance(statement, ast.Assign)
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            context.values[target.id] = value
    iterator = analysis.reduction_iterator
    if (
        ast.unparse(iterator.func)
        not in {"range", "cutlass.range", "cutlass.range_constexpr"}
        or len(iterator.args) != 3
        or any(
            keyword.arg != "unroll"
            or not isinstance(keyword.value, ast.Constant)
            or type(keyword.value.value) is not int
            or keyword.value.value < 0
            for keyword in iterator.keywords
        )
    ):
        _reject("noncanonical packet iterator")
    start, stop, step = (proof.value(arg, context) for arg in iterator.args)
    if (
        start is None
        or stop is None
        or step is None
        or (start.low, start.high) != (0, 0)
        or (step.low, step.high) != (analysis.site.bk, analysis.site.bk)
        or stop.low <= 0
    ):
        _reject("packet extent is not a proven positive zero-based range")
    if stop.high + step.low - 1 > (1 << 31) - 1:
        _reject("rounding the packet domain could overflow signed int32")
    packets = (stop.low + step.low - 1) // step.low
    if packets != (stop.high + step.low - 1) // step.low:
        _reject("the number of K packets varies across the launch")
    return packets * step.low


def _constant_bound(original: ast.expr, value: int) -> ast.expr:
    """Retain explicit induction-width casts while fixing a proven bound."""
    if (
        isinstance(original, ast.Call)
        and ast.unparse(original.func)
        in {"cutlass.Int32", "cutlass.Int64", "cutlass.Uint32", "cutlass.Uint64"}
        and len(original.args) == 1
        and not original.keywords
    ):
        result = _clone(original)
        result.args = [_constant_bound(original.args[0], value)]
        return result
    return ast.Constant(value)


def lower_register_chain(
    body: list[ast.stmt],
    df: DeviceFunction,
    *,
    boundary_names: set[str],
    disjoint_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
) -> list[ast.stmt]:
    from ..device_function import TensorArg

    sites = df.cute_state.collective_mma_sites
    if len(sites) < 2:
        _reject("requires an admitted chain of contractions")
    first = sites[0]
    coordinates = (
        first.m_index,
        first.n_index,
        first.m_offset,
        first.n_offset,
        first.bm,
        first.bn,
        first.m_axis,
        first.n_axis,
        first.grid_row_lane,
    )
    if first.grid_row_lane is None or any(
        (
            site.m_index,
            site.n_index,
            site.m_offset,
            site.n_offset,
            site.bm,
            site.bn,
            site.m_axis,
            site.n_axis,
            site.grid_row_lane,
        )
        != coordinates
        or site.k_factor != 1
        for site in sites
    ):
        _reject("contractions do not share one unpacked output tile")

    # The reaching-definition bridge exists only in this private analysis
    # tree. Original AST nodes remain untouched if any later proof refuses.
    working = [_clone_statement(statement) for statement in body]
    ast_rename(ast.Module(body=working, type_ignores=[]), rename_groups)
    occupied = boundary_names | {
        node.id
        for statement in working
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    ordinals = count()

    def fresh(hint: str) -> str:
        while True:
            name = f"_helion_chain_proof_{next(ordinals)}_{hint}"
            if name not in occupied:
                occupied.add(name)
                return name

    axes = {first.m_axis: 64 // first.bn, first.n_axis: first.bn}
    threads = cast("tuple[int, int, int]", tuple(axes.get(i, 1) for i in range(3)))
    constexprs = _constexpr_values(df)
    analyses: list[CollectiveSiteAnalysis] = []
    plans: list[RegisterChainSite] = []
    frontiers: list[str] = []
    row_loop: ast.For | None = None
    for site in sites:
        _lower_site(
            working,
            site,
            df,
            boundary_names | set(frontiers),
            disjoint_pairs,
            register_chain_analysis=analyses,
            analysis_fresh_name=fresh,
        )
        analysis = analyses[-1]
        if analysis.row_depth != 0 or analysis.row_loop.orelse:
            _reject("requires a root output-row region without loop live-outs")
        if row_loop is None:
            row_loop = analysis.row_loop
        elif analysis.row_loop is not row_loop:
            _reject("contractions occupy different row regions")
        for recipe in (analysis.a_recipe, analysis.b_recipe, analysis.seed_recipe):
            if recipe is None:
                continue
            statements, value = recipe.emit({}, fresh)
            if _uses_thread_coordinates([*statements, value]):
                _reject(
                    "a replayed operand still depends on physical thread coordinates"
                )
            used_frontiers = set(frontiers) & _read_names(
                ast.Module(body=[*statements, ast.Expr(value)], type_ignores=[])
            )
            if recipe is analysis.seed_recipe:
                if frontiers and used_frontiers != {frontiers[-1]}:
                    _reject("seed is not the adjacent same-cell accumulator")
            elif used_frontiers:
                _reject("an operand reads a differently shaped accumulator frontier")
        if frontiers and analysis.seed_recipe is None:
            _reject("a later contraction does not consume its preceding accumulator")
        extent = _packet_extent(analysis, constexprs, threads)
        iterator = _clone(analysis.reduction_iterator)
        iterator.args = list(
            starmap(
                _constant_bound, zip(iterator.args, (0, extent, site.bk), strict=True)
            )
        )
        plans.append(
            RegisterChainSite(
                identity=site.identity,
                bm=site.bm,
                bn=site.bn,
                bk=site.bk,
                dtype=site.dtype,
                m_index=site.m_index,
                n_index=site.n_index,
                k_index=site.k_index,
                m_offset=site.m_offset,
                n_offset=site.n_offset,
                k_offset=site.k_offset,
                reduction_iterator=iterator,
                static_k_extent=extent,
                a_recipe=analysis.a_recipe,
                b_recipe=analysis.b_recipe,
                seed_recipe=analysis.seed_recipe,
                seed_from=sites[len(plans) - 1].identity if frontiers else None,
                seed_name=frontiers[-1] if frontiers else None,
            )
        )
        frontier = fresh("frontier")
        replacement = ast.Assign(
            targets=[ast.Name(analysis.result_name, ast.Store())],
            value=ast.Name(frontier, ast.Load()),
        )
        parent, index = analysis.path[
            -1 if site.synthetic_k_lane is not None else analysis.reduction_depth
        ]
        assert isinstance(parent, list)
        parent[index] = replacement
        frontiers.append(frontier)

    _remove_dead_staged_operands(
        working,
        df,
        pure_statement_ids={
            identity
            for analysis in analyses
            for identity in analysis.pure_statement_ids
        },
    )
    assert row_loop is not None
    row_index = working.index(row_loop)
    terminal_arena_dead = _terminal_arena_dead(working, row_loop)
    writes = _assigned_names(row_loop)
    tensor_names = frozenset(
        argument.name for argument in df.arguments if isinstance(argument, TensorArg)
    )
    if writes & (
        tensor_names
        | _GLOBALS
        | set(constexprs)
        | set(ReadWrites.from_list(working[row_index + 1 :]).reads)
    ):
        _reject("row-region definitions escape or overwrite a proof input")
    simplifier = _RemoveFullThreadBounds(threads)
    row_loop = cast("ast.For", simplifier.visit(_clone(row_loop)))
    dominating = [
        cast("ast.stmt", simplifier.visit(_clone(statement)))
        for statement in working[:row_index]
    ]
    terminal = row_loop.body[-1]
    predicate: ast.expr = ast.Constant(True)
    if isinstance(terminal, ast.If):
        if terminal.orelse or len(terminal.body) != 1:
            _reject("output store has unsupported control flow")
        predicate, terminal = terminal.test, terminal.body[0]
    if not (
        isinstance(terminal, ast.Expr)
        and isinstance(terminal.value, ast.Call)
        and isinstance(terminal.value.func, ast.Attribute)
        and terminal.value.func.attr == "store"
        and len(terminal.value.args) == 1
        and not terminal.value.keywords
    ):
        _reject("requires one terminal ordinary output store")
    store = terminal.value
    outputs = _tensor_roots([store], "store")
    if outputs is None or len(outputs) != 1:
        _reject("output does not have one proven tensor root")
    output = next(iter(outputs))
    boundaries = (
        boundary_names
        | set(frontiers)
        | {
            first.m_index,
            first.n_index,
            first.m_offset,
            first.n_offset,
        }
    )
    live = _read_names(row_loop.body[-1])
    retained, _ = prune_pure_scalar_statements(row_loop.body[1:-1], live)
    statements = [row_loop.body[0], *retained]
    value = _scalar_epilogue(
        statements,
        store.args[0],
        m_index=first.m_index,
        n_index=first.n_index,
        dominating=dominating,
        boundaries=boundaries,
        fresh_name=fresh,
    )
    prefix = [*dominating, *statements]
    assert isinstance(store.func, ast.Attribute)
    pointer = build_recipe(store.func.value, prefix, boundaries)
    mask = build_recipe(predicate, prefix, boundaries)
    if value is None or pointer is None or mask is None:
        _reject("output is not a pure coordinate-replayable scalar epilogue")
    proof: list[ast.AST] = []
    for recipe in (value, pointer, mask):
        setup, expression = recipe.emit({}, fresh)
        terms = [*setup, expression]
        used_frontiers = set(frontiers) & set().union(
            *(_read_names(term) for term in terms)
        )
        allowed_frontiers = {frontiers[-1]} if recipe is value else set()
        if used_frontiers - allowed_frontiers:
            _reject("output uses an earlier accumulator or a data-dependent address")
        proof.extend(terms)
    reads = _load_pointer_roots(
        proof,
        tensor_names=tensor_names,
        integer_names={first.m_index, first.n_index, first.m_offset, first.n_offset},
    )
    if (
        _uses_thread_coordinates(proof)
        or reads is None
        or any(frozenset((root, output)) not in disjoint_pairs for root in reads)
    ):
        _reject("output replay has unproved coordinates or read/write aliases")
    env = CompileEnvironment.current()
    argument = next(
        (
            arg
            for arg in df.arguments
            if isinstance(arg, TensorArg) and arg.name == output
        ),
        None,
    )
    if argument is None:
        _reject("output is not an original tensor argument")
    dtype = env.backend.dtype_str(argument.fake_value.dtype)
    # A fresh host output is not an input specialization. These candidate
    # strides are guarded against the actual output before vector stores.
    facts = CopyTensorFacts(
        dtype,
        tuple(env.size_hint(stride) for stride in argument.fake_value.stride()),
        16,
    )
    if any(
        isinstance(node, ast.Name) and node.id == MARKER
        for statement in working
        for node in ast.walk(statement)
    ):
        _reject("unhandled contraction remains outside the planned region")
    plan = plan_register_chain(
        plans,
        RegisterChainEpilogue(
            m_index=first.m_index,
            n_index=first.n_index,
            accumulator_name=frontiers[-1],
            value_recipe=value,
            pointer_recipe=pointer,
            predicate_recipe=mask,
            dtype=dtype,
            output_tensor=output,
            output_facts=facts,
        ),
        thread_count=64,
        tensor_facts=_copy_tensor_facts(df),
        fast_math=env.settings.fast_math,
        terminal_arena_dead=terminal_arena_dead,
        constexpr_values=constexprs,
        target_device_capability=env.config_spec.target_device_capability,
    )
    fast_axis, slow_axis = sorted(axes)
    tid = (
        f"cutlass.Int32(cute.arch.thread_idx()[{fast_axis}]) + "
        f"cutlass.Int32(cute.arch.thread_idx()[{slow_axis}]) * {axes[fast_axis]}"
    )
    flat_threads = flatten_thread_coordinates(working, threads)
    if flat_threads:
        tid = "cutlass.Int32(cute.arch.thread_idx()[0])"
    emitted = emit_register_chain(
        plan, prefix=df.new_var("register_chain"), tid=tid, fresh_name=df.new_var
    )
    working[row_index : row_index + 1] = emitted
    df.cute_state.collective_register_chain_lowered = True
    if flat_threads:
        df.cute_state.collective_register_chain_block_dims = (64, 1, 1)
    return working
