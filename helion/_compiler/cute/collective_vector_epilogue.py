"""Vector stores for an already owned, pure collective scalar epilogue."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from ..ast_read_writes import ReadWrites
from .collective_epilogue import _RemoveFullThreadBounds
from .collective_epilogue import _scalar_epilogue
from .collective_tmem_seed import prune_pure_scalar_statements
from .collective_vector_recipe import emit_vector_recipe
from .contiguous_copy import _expr
from .contiguous_copy import _path
from .contiguous_copy import plan_contiguous_copy
from .scalar_recipe import _GLOBALS
from .scalar_recipe import _clone
from .scalar_recipe import _read_names
from .scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..device_function import DeviceFunction
    from .collective_matmul import CollectiveMmaSite
    from .contiguous_copy import CopyTensorFacts


def _load_pointer_roots(
    statements: Sequence[ast.AST],
    *,
    tensor_names: frozenset[str],
    integer_names: set[str],
) -> set[str] | None:
    """Require an explicit tensor base and integral offsets for every load.

    Counting iterator syntax can miss another base behind a pointer alias or
    conditional. Decline those bases rather than infer their identity. Track
    only integer type facts in statement order; aliases never expand into an
    expression tree, and a rebind cannot retain an earlier integer fact.
    """
    roots: set[str] = set()

    def integer_type(value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "cutlass"
            and value.attr
            in {
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "Uint8",
                "Uint16",
                "Uint32",
                "Uint64",
            }
        )

    def integer(value: ast.expr, known: set[str]) -> bool:
        pending: list[ast.AST] = [value]
        while pending:
            node = pending.pop()
            if isinstance(node, ast.Constant) and type(node.value) is int:
                continue
            if isinstance(node, ast.Name) and node.id in known:
                continue
            if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
                if integer_type(node.func) or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"to", "bitcast"}
                    and integer_type(node.args[0])
                ):
                    continue
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub, ast.Invert)
            ):
                pending.append(node.operand)
                continue
            if isinstance(node, ast.BinOp) and isinstance(
                node.op,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.FloorDiv,
                    ast.Mod,
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                    ast.LShift,
                    ast.RShift,
                ),
            ):
                pending.extend((node.left, node.right))
                continue
            if isinstance(node, ast.IfExp):
                pending.extend((node.body, node.orelse))
                continue
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int
                and node.slice.value >= 0
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in {"shape", "stride"}
            ):
                owner = node.value.value
                if isinstance(owner, ast.Attribute) and owner.attr == "layout":
                    owner = owner.value
                if isinstance(owner, ast.Name) and owner.id in tensor_names:
                    continue
            if (
                isinstance(node, ast.Call)
                and _path(node.func) == "cute.crd2idx"
                and len(node.args) == 2
                and not node.keywords
                and isinstance(node.args[0], ast.Tuple)
                and isinstance(node.args[1], ast.Attribute)
                and node.args[1].attr == "layout"
                and isinstance(node.args[1].value, ast.Name)
                and node.args[1].value.id in tensor_names
            ):
                pending.extend(node.args[0].elts)
                continue
            return False
        return True

    def expression(value: ast.AST, known: set[str]) -> bool:
        for node in ast.walk(value):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in tensor_names
            ):
                roots.add(node.value.id)
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
            ):
                continue
            pointer = node.func.value
            if _path(pointer) == "cute.arch":
                if len(node.args) != 2 or node.keywords:
                    return False
                pointer = node.args[0]
            elif node.args or node.keywords:
                return False
            while isinstance(pointer, ast.BinOp) and isinstance(
                pointer.op, (ast.Add, ast.Sub)
            ):
                if not integer(pointer.right, known):
                    return False
                pointer = pointer.left
            if not (
                isinstance(pointer, ast.Attribute)
                and pointer.attr == "iterator"
                and isinstance(pointer.value, ast.Name)
                and pointer.value.id in tensor_names
            ):
                return False
            roots.add(pointer.value.id)
        return True

    def visit(body: Sequence[ast.AST], known: set[str]) -> bool:
        for statement in body:
            if isinstance(statement, ast.Assign):
                if (
                    len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)
                    or not expression(statement.value, known)
                ):
                    return False
                is_integer = integer(statement.value, known)
                known.discard(statement.targets[0].id)
                if is_integer:
                    known.add(statement.targets[0].id)
            elif isinstance(statement, ast.If):
                if not expression(statement.test, known):
                    return False
                left, right = set(known), set(known)
                if not visit(statement.body, left) or not visit(
                    statement.orelse, right
                ):
                    return False
                known.clear()
                known.update(left & right)
            elif isinstance(statement, ast.Pass):
                continue
            elif not isinstance(statement, (ast.expr, ast.Expr)) or not expression(
                statement, known
            ):
                return False
        return True

    return roots if visit(statements, set(integer_names)) else None


def vectorize_collective_store_epilogue(
    loop: ast.For,
    *,
    m_index: str,
    n_index: str,
    m_offset: str,
    n_offset: str,
    block_m: int,
    block_n: int,
    thread_index: str,
    thread_dimensions: tuple[int, int, int],
    shared_results: frozenset[str],
    tensors: Mapping[str, CopyTensorFacts],
    output_layouts: Mapping[str, CopyTensorFacts],
    tensor_names: frozenset[str],
    disjoint_pairs: set[frozenset[str]],
    boundary_names: set[str],
    dominating: Sequence[ast.stmt],
    live_after: Sequence[ast.stmt],
    unroll: bool,
    fresh_name: Callable[[str], str],
) -> ast.stmt | None:
    """Repartition one complete M/N tile, retaining all scalar expressions.

    The collective caller has proved exact 128-thread ownership of this tile.
    We additionally require a single ordinary output store, pure scalar work,
    no loop live-outs, and no possible alias between its reads and its output.
    The existing contiguous-copy analysis proves output alignment, consecutive
    addresses and masks. Masked output tails still store one scalar at a time.
    """
    from .collective_matmul import _tensor_roots
    from .collective_matmul import _uses_thread_coordinates

    if not loop.body or loop.orelse:
        return None
    # Storage facts and helper calls describe the original tensors and globals.
    # Include loop targets, which ReadWrites does not count as writes.
    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and node.id in tensor_names | _GLOBALS
        for statement in dominating
        for node in ast.walk(statement)
    ):
        return None
    if set(ReadWrites.from_list(list(dominating)).writes) & _GLOBALS:
        return None
    loop_writes = {
        node.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    if loop_writes & (_GLOBALS | set(ReadWrites.from_list(list(live_after)).reads)):
        return None
    simplifier = _RemoveFullThreadBounds(thread_dimensions)
    loop = cast("ast.For", simplifier.visit(_clone(loop)))
    dominating = [
        cast("ast.stmt", simplifier.visit(_clone(statement)))
        for statement in dominating
    ]
    terminal = loop.body[-1]
    predicate: ast.expr = ast.Constant(True)
    if isinstance(terminal, ast.If):
        if terminal.orelse or len(terminal.body) != 1:
            return None
        predicate = terminal.test
        terminal = terminal.body[0]
    if not (
        isinstance(terminal, ast.Expr)
        and isinstance(terminal.value, ast.Call)
        and isinstance(terminal.value.func, ast.Attribute)
        and terminal.value.func.attr == "store"
        and len(terminal.value.args) == 1
        and not terminal.value.keywords
    ):
        return None
    store = terminal.value
    written = _tensor_roots([store], "store")
    if written is None or len(written) != 1:
        return None
    output = next(iter(written))
    facts = output_layouts.get(output)
    if facts is None or facts.dtype not in {
        "cutlass.Float16",
        "cutlass.BFloat16",
        "cutlass.Float32",
    }:
        return None
    width = 4 if facts.dtype == "cutlass.Float32" else 8
    if block_n % width or block_m * block_n % (128 * width):
        return None
    boundaries = (
        boundary_names
        | shared_results
        | {
            m_index,
            n_index,
            m_offset,
            n_offset,
        }
    )
    pure = _scalar_epilogue(
        loop.body[:-1],
        store.args[0],
        m_index=m_index,
        n_index=n_index,
        dominating=dominating,
        boundaries=boundaries,
        fresh_name=fresh_name,
    )
    if pure is None:
        return None
    prefix = [*dominating, *loop.body[:-1]]
    # Inline store arguments originally execute only under the store mask.
    # Already-computed scalar definitions keep their original predicates,
    # while these inline loads/calls must remain protected after repartition.
    guarded_value = ast.IfExp(
        test=_clone(predicate),
        body=_clone(store.args[0]),
        orelse=_expr(f"{facts.dtype}(0)"),
    )
    value_recipe = build_recipe(guarded_value, prefix, boundaries)
    assert isinstance(store.func, ast.Attribute)
    synthetic_load = ast.IfExp(
        test=_clone(predicate),
        body=ast.Call(
            func=ast.Attribute(_clone(store.func.value), "load", ast.Load()),
            args=[],
            keywords=[],
        ),
        orelse=_expr(f"{facts.dtype}(0)"),
    )
    address_recipe = build_recipe(synthetic_load, prefix, boundaries)
    if value_recipe is None or address_recipe is None:
        return None
    proof, value = pure.emit({}, fresh_name)
    address_proof, address_value = address_recipe.emit({}, fresh_name)
    proof.extend([ast.Expr(value), *address_proof, ast.Expr(address_value)])
    if _uses_thread_coordinates(proof):
        return None
    known_tensors = tensor_names | shared_results
    for statement in proof:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Subscript):
                continue
            container = node.value
            if isinstance(container, ast.Name):
                if container.id not in known_tensors:
                    return None
            elif not (
                isinstance(container, ast.Tuple)
                or isinstance(container, ast.Attribute)
                and container.attr in {"shape", "stride"}
                or isinstance(container, ast.Call)
                and ast.unparse(container.func)
                in {"cute.arch.block_idx", "cute.arch.grid_dim", "cute.arch.block_dim"}
            ):
                return None
    # The synthetic load is only an address proof. Exclude it from read/write
    # alias analysis; include the actual epilogue and predicate dependencies.
    value_proof, proof_value = pure.emit({}, fresh_name)
    mask_recipe = build_recipe(predicate, prefix, boundaries)
    if mask_recipe is None:
        return None
    mask_proof, mask_value = mask_recipe.emit({}, fresh_name)
    reads = _load_pointer_roots(
        [*value_proof, proof_value, *mask_proof, mask_value, *address_proof],
        tensor_names=known_tensors,
        integer_names={m_index, n_index, m_offset, n_offset},
    )
    if reads is None or any(
        root not in shared_results and frozenset((root, output)) not in disjoint_pairs
        for root in reads
    ):
        return None
    slot = fresh_name("epilogue_vector_slot")
    flat = fresh_name("epilogue_vector_flat")
    row = fresh_name("epilogue_vector_m")
    column = fresh_name("epilogue_vector_n")
    replacements = {
        m_index: _expr(f"{m_offset} + {row}"),
        n_index: _expr(f"{n_offset} + {column}"),
    }
    address_statements, address = address_recipe.emit(replacements, fresh_name)
    aligned = {column: width, n_offset: block_n}
    store_plan = plan_contiguous_copy(
        address_statements,
        address,
        coordinate=column,
        tensors={**tensors, output: facts},
        aligned_names=aligned,
    )
    if store_plan is None or store_plan.width != width:
        return None
    statements, value = value_recipe.emit(replacements, fresh_name)
    registers = fresh_name("epilogue_vector_values")
    body = ast.parse(
        f"{registers} = cute.make_rmem_tensor(cute.make_layout(({width},)), {facts.dtype})"
    ).body
    body.extend(
        emit_vector_recipe(
            statements,
            value,
            coordinate=column,
            width=width,
            tensors=tensors,
            aligned_names=aligned,
            destination=registers,
            destination_indices=None,
            fresh_name=fresh_name,
        )
    )
    body.extend(store_plan.emit_from_registers(registers, fresh_name))
    count = block_m * block_n // (128 * width)
    iterator = (
        f"cutlass.range_constexpr({count})"
        if unroll
        else f"cutlass.range({count}, unroll=1)"
    )
    result = cast(
        "ast.For",
        ast.parse(
            f"for {slot} in {iterator}:\n"
            f"    {flat} = ({thread_index} + {slot} * 128) * {width}\n"
            f"    {row} = {flat} // {block_n}\n"
            f"    {column} = {flat} % {block_n}\n"
        ).body[0],
    )
    result.body.extend(body)
    if tensors.get(output) == facts:
        return ast.fix_missing_locations(result)
    # The output may be a host allocation with no input specialization. Its
    # traced layout is only a candidate: guard its actual pointer and every
    # stride before relying on that candidate's vector address proof.
    guard = [f"{output}.iterator.toint() % 16 == 0"]
    guard.extend(
        f"{output}.layout.stride[{axis}] == {stride}"
        for axis, stride in enumerate(facts.strides)
    )
    # All scalar assignments have been proved pure and have no loop live-outs.
    # Remove overwritten/dead definitions from the fallback too, so a dead
    # earlier collective result does not retain its shared-memory lifetime.
    fallback, _ = prune_pure_scalar_statements(
        loop.body[:-1], _read_names(loop.body[-1])
    )
    loop.body = [*fallback, loop.body[-1]]
    return ast.fix_missing_locations(
        ast.If(test=_expr(" and ".join(guard)), body=[result], orelse=[loop])
    )


def vectorize_collective_store_epilogues(
    body: list[ast.stmt],
    df: DeviceFunction,
    lowered_sites: Sequence[tuple[CollectiveMmaSite, ast.For, str, str]],
    *,
    boundary_names: set[str],
    disjoint_pairs: set[frozenset[str]],
) -> None:
    """Rewrite after every collective and its dead synthetic K loop are removed."""
    from ..compile_environment import CompileEnvironment
    from ..device_function import TensorArg
    from .collective_matmul import _copy_tensor_facts
    from .contiguous_copy import CopyTensorFacts

    # Several contractions can feed the same epilogue. Each has independently
    # proved its row-lane ownership, and the final one supplies the live tile.
    candidates = {
        loop: (site, tid)
        for site, loop, tid, shared in lowered_sites
        if site.grid_row_lane is not None
    }
    tensors = _copy_tensor_facts(df)
    env = CompileEnvironment.current()
    output_layouts = {
        argument.name: CopyTensorFacts(
            env.backend.dtype_str(argument.fake_value.dtype),
            tuple(env.size_hint(stride) for stride in argument.fake_value.stride()),
            16,
        )
        for argument in df.arguments
        if isinstance(argument, TensorArg)
        and env.backend.dtype_str(argument.fake_value.dtype)
        in {"cutlass.Float16", "cutlass.BFloat16", "cutlass.Float32"}
    }
    tensor_names = frozenset(
        argument.name for argument in df.arguments if isinstance(argument, TensorArg)
    )
    shared_results = frozenset(shared for site, loop, tid, shared in lowered_sites)

    def visit(
        statements: list[ast.stmt],
        prefix: list[ast.stmt],
        suffix: list[ast.stmt],
        loop_prefix: list[ast.stmt] | None,
    ) -> None:
        for index, statement in enumerate(statements):
            dominating = [*prefix, *statements[:index]]
            live_after = [*statements[index + 1 :], *suffix]
            backedge = (
                [*loop_prefix, *statements[:index]] if loop_prefix is not None else []
            )
            if statement in candidates:
                site, tid = candidates[statement]
                assert isinstance(statement, ast.For)
                axes = {site.m_axis: 128 // site.bn, site.n_axis: site.bn}
                replacement = vectorize_collective_store_epilogue(
                    statement,
                    m_index=site.m_index,
                    n_index=site.n_index,
                    m_offset=site.m_offset,
                    n_offset=site.n_offset,
                    block_m=site.bm,
                    block_n=site.bn,
                    thread_index=tid,
                    thread_dimensions=cast(
                        "tuple[int, int, int]",
                        tuple(axes.get(axis, 1) for axis in range(3)),
                    ),
                    shared_results=shared_results,
                    tensors=tensors,
                    output_layouts=output_layouts,
                    tensor_names=tensor_names,
                    disjoint_pairs=disjoint_pairs,
                    boundary_names=boundary_names,
                    dominating=dominating,
                    live_after=[*live_after, *backedge],
                    unroll=df.config.get("cute_collective_epilogue")
                    == "vector_unrolled",
                    fresh_name=df.new_var,
                )
                if replacement is not None:
                    statements[index] = replacement
                    continue
            if isinstance(statement, ast.For):
                if any(
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    and node.id in tensor_names | _GLOBALS
                    for node in ast.walk(statement.target)
                ):
                    continue
                # A value read earlier in an enclosing iteration is live on
                # its backedge even when the remaining suffix never reads it.
                # Keep those lexical reads conservatively, including control
                # expressions, and preserve normal For-else live-outs.
                visit(
                    statement.body,
                    dominating,
                    [*statement.orelse, *live_after],
                    [*backedge, ast.Expr(statement.iter)],
                )
                else_backedge = (
                    [*backedge, ast.Expr(statement.iter), *statement.body]
                    if loop_prefix is not None
                    else None
                )
                visit(statement.orelse, dominating, live_after, else_backedge)
            elif isinstance(statement, ast.If):
                branch_prefix = (
                    [*backedge, ast.Expr(statement.test)]
                    if loop_prefix is not None
                    else None
                )
                visit(statement.body, dominating, live_after, branch_prefix)
                visit(statement.orelse, dominating, live_after, branch_prefix)

    visit(body, [], [], None)
