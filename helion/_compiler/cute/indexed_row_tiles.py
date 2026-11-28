"""Stage two selected BF16 row fragments in each warp of a complete row tile."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
from math import prod
from typing import TYPE_CHECKING
from typing import cast

from ...exc import InvalidConfig
from ..ast_extension import clone_ast
from ..device_function import TensorArg
from . import indexed_row_epilogue as row_epilogue
from ._ast_pass_utils import _bound_names
from ._ast_pass_utils import names_written
from ._ast_pass_utils import scalar_index_offsets
from ._ast_pass_utils import scalar_index_ptr
from ._ast_pass_utils import scalar_tensor_access
from .analysis.indexed_row_epilogue import indexed_row_epilogue
from .autotune.config_keys import INDEXED_ROW_EPILOGUE_KEY
from .autotune.config_keys import REUSE_INDEXED_INPUT_KEY
from .autotune.indexed_row_epilogue import reusable_row_axes
from .warp_row_partition import pure_row_expression

if TYPE_CHECKING:
    from ..device_ir import DeviceIR
    from .analysis.indexed_row_phases import IndexedRowPhase
    from .passes.pipeline import PassContext


_COPY_PANEL = 32
_WARP = 32
_ROW_GROUP = 8
_STAGE_NAME = "_cute_staged_indexed_row_dot"
_LOCAL = "_stage_indexed_"


def _simple_name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _target(st: ast.AST) -> str | None:
    if isinstance(st, ast.Assign) and len(st.targets) == 1:
        return _simple_name(st.targets[0])
    return None


def _names(expr: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(expr)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _effects(walk: ast.AST, s: IndexedRowPhase) -> bool:
    expected = Counter(
        (
            (s.selection, "load"),
            (s.weights, "load"),
            (s.vector, "load"),
            (s.matrix, "load"),
            (s.output, "store"),
        )
    )
    reductions = Counter({"_cute_grouped_reduce_shared_two_stage": 1})
    if s.paired:
        assert s.input_scale and s.selected_ids and s.selected_weights
        expected.update(
            (
                (s.vector, "load"),
                (s.matrix, "load"),
                (s.input_scale, "load"),
                (s.output, "store"),
                (s.selected_ids, "store"),
                (s.selected_weights, "store"),
            )
        )
        reductions.update(
            (
                "cute.arch.warp_reduction_sum",
                "cute.math.rsqrt",
                "_cute_grouped_reduce_shared_two_stage",
            )
        )
    allowed = {
        "range",
        "cutlass.Int32",
        "cutlass.Int64",
        "cutlass.Float32",
        "cutlass.BFloat16",
        "cute.arch.thread_idx",
    }
    for n in ast.walk(walk):
        if isinstance(n, ast.stmt) and not isinstance(
            n, (ast.For, ast.Assign, ast.If, ast.Expr, ast.Pass)
        ):
            return False
        if isinstance(n, ast.Assign) and _target(n) is None:
            return False
        if not isinstance(n, ast.Call):
            continue
        if m := scalar_tensor_access(n):
            expected[m] -= 1
            continue
        callee = ast.unparse(n.func)
        if callee == "cute.math.div" and s.paired:
            if not pure_row_expression(n, set()):
                return False
            continue
        if callee in reductions:
            if callee == "_cute_grouped_reduce_shared_two_stage" and (
                len(n.args) < 3
                or ast.unparse(n.args[1]) != "'sum'"
                or ast.unparse(n.args[2]) != "cutlass.Float32(0)"
            ):
                return False
            reductions[callee] -= 1
        elif callee not in allowed:
            return False
    return not any(expected.values()) and not any(reductions.values())


@dataclass(frozen=True)
class _Recipes:
    branch: ast.If
    original: list[ast.stmt]
    # Coordinates irrespective of whether a dimension was folded to zero.
    batch: str
    slot: str
    row_base: str
    selected: ast.Assign
    epsilon: str | None
    approximate: bool


def _recipes(
    fn: ast.FunctionDef,
    phase: ast.For,
    site: IndexedRowPhase,
    width: int,
    block: tuple[int, int, int],
    constants: set[str],
) -> _Recipes | None:
    if (
        not phase.body
        or not isinstance(branch := phase.body[-1], ast.If)
        or any(not isinstance(n, ast.Pass) for n in branch.orelse)
    ):
        return None
    if not branch.body:
        return None
    original: list[ast.stmt]
    walk = branch.body[-1]
    if isinstance(walk, ast.For):
        original = [walk]
        body = walk.body
        outer = branch.body[:-1]
    else:
        # A fixed or fully occupied row may use one output per x lane.
        # Its scalar row body has no outer lane loop. Split only at the
        # unique exact row-coordinate assignment; keep the fallback intact.
        if width != block[0]:
            return None
        starts = [
            i
            for i, stmt in enumerate(branch.body)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.BinOp)
            and isinstance(stmt.value.op, ast.Add)
            and isinstance(stmt.value.left, ast.Name)
            and ast.unparse(stmt.value.right)
            == "cutlass.Int32(cute.arch.thread_idx()[0])"
        ]
        if len(starts) != 1:
            return None
        start = starts[0]
        original = body = branch.body[start:]
        outer = branch.body[:start]
    if not _effects(ast.Module(body=original, type_ignores=[]), site):
        return None
    if any(
        not isinstance(st, ast.Assign)
        or _target(st) is None
        or any(
            ast.unparse(n.func)
            not in ("cutlass.Int32", "cutlass.Int64", "cute.arch.block_idx")
            for n in ast.walk(st.value)
            if isinstance(n, ast.Call)
        )
        for st in outer
    ):
        return None
    # Admit only uniformly computed launch coordinates. Values defined by a
    # previous phase, opaque calls, memory and individual lane coordinates
    # can never determine whether a CTA reaches one of the new barriers.
    allowed_calls = {"cutlass.Int32", "cutlass.Int64", "cute.arch.block_idx", "range"}

    def uniform(node: ast.AST, names: set[str]) -> bool:
        return _names(node) <= names and all(
            ast.unparse(v.func) in allowed_calls
            for v in ast.walk(node)
            if isinstance(v, ast.Call)
        )

    known = {a.arg for a in fn.args.args} | constants | {"cutlass", "cute", "range"}
    # In the globally blocked persistent schedule the shared PID bounds are
    # defined outside all phases. Reaching scalar definitions, including
    # kills by earlier loops, are necessary to certify those bounds.

    def reach(stmt: ast.stmt, before: set[str]) -> set[str]:
        if isinstance(stmt, ast.If) and uniform(stmt.test, before):
            alternatives = []
            for arm in (stmt.body, stmt.orelse):
                arm_known = before.copy()
                for child in arm:
                    arm_known = reach(child, arm_known)
                alternatives.append(arm_known)
            return alternatives[0] & alternatives[1]
        written = {
            n.id
            for n in ast.walk(stmt)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        after = before.difference(written)
        if (
            isinstance(stmt, ast.Assign)
            and (direct := _target(stmt)) is not None
            and uniform(stmt.value, before)
        ):
            after.add(direct)
        return after

    for stmt in fn.body:
        if stmt is phase:
            break
        known = reach(stmt, known)
    if not uniform(phase.iter, known) or not isinstance(phase.target, ast.Name):
        return None
    known.add(phase.target.id)
    for line in phase.body[:-1]:
        if (
            not isinstance(line, ast.Assign)
            or (nm := _target(line)) is None
            or not uniform(line.value, known)
        ):
            return None
        known.add(nm)
    if not uniform(branch.test, known):
        return None
    for line in outer:
        if (
            not isinstance(line, ast.Assign)
            or (nm := _target(line)) is None
            or not uniform(line.value, known)
        ):
            return None
        known.add(nm)
    if (
        not body
        or not isinstance(st := body[0], ast.Assign)
        or not isinstance(add := st.value, ast.BinOp)
    ):
        return None
    if isinstance(walk, ast.For):
        if (
            not isinstance(add.left, ast.BinOp)
            or (base := _simple_name(add.left.left)) is None
            or (oldlane := _simple_name(walk.target)) is None
            or not isinstance(walk.iter, ast.Call)
            or ast.unparse(walk.iter.func) != "range"
            or len(walk.iter.args) != 1
            or not isinstance(v := walk.iter.args[0], ast.Constant)
            or type(lanes := v.value) is not int
            or lanes * block[0] != width
            or ast.unparse(add)
            != f"{base} + cutlass.Int32(cute.arch.thread_idx()[0]) * {lanes} + cutlass.Int32({oldlane})"
        ):
            return None
    else:
        base = _simple_name(add.left)
    if base is None or base not in known:
        return None
    loads = [
        z
        for z in body
        if isinstance(z, ast.Assign)
        and isinstance(z.value, ast.Call)
        and scalar_tensor_access(z.value) == (site.selection, "load")
    ]
    if len(loads) != 1 or _target(loads[0]) is None:
        return None
    load = loads[0].value
    assert isinstance(load, ast.Call) and isinstance(load.func, ast.Attribute)
    ix = scalar_index_offsets(load.func.value, site.selection, 2)
    if ix is None or not all(_names(i) <= known for i in ix):
        return None
    eps = None
    approximate = False
    if site.paired:
        # The IR proves rsqrt(mean(x*x) + scalar). Resolve the precise scalar
        # argument after codegen renaming, rejecting any new arithmetic.
        args = {a.arg for a in fn.args.args}
        defs = {
            nm: z.value
            for z in body
            if isinstance(z, ast.Assign) and (nm := _target(z))
        }
        rs = [
            n
            for n in defs.values()
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "cute.math.rsqrt"
        ]
        if (
            len(rs) != 1
            or len(rs[0].args) != 1
            or (_var := _simple_name(rs[0].args[0])) is None
            or not isinstance(add := defs.get(_var), ast.BinOp)
            or not isinstance(add.op, ast.Add)
            or (eps := _simple_name(add.right)) not in args
        ):
            return None
        approximate = bool(rs[0].keywords)
        if approximate and not (
            len(rs[0].keywords) == 1
            and rs[0].keywords[0].arg == "fastmath"
            and isinstance(rs[0].keywords[0].value, ast.Constant)
            and rs[0].keywords[0].value.value is True
        ):
            return None
    return _Recipes(
        branch,
        original,
        ast.unparse(ix[0]),
        ast.unparse(ix[1]),
        base,
        loads[0],
        eps,
        approximate,
    )


def _guard_scalar_bank_reads(site: IndexedRowPhase, phase: ast.For) -> None:
    """Wrap only the scalar fallback loads of the proved, flattened bank."""
    rows = site.rows * (2 if site.paired else 1) * site.groups
    loads = [
        node
        for node in ast.walk(phase)
        if isinstance(node, ast.Call)
        and scalar_tensor_access(node) == (site.matrix, "load")
    ]
    if rows >= 2**31 or len(loads) != 1 + int(site.paired):
        raise InvalidConfig(
            "cannot certify the scalar loads of the staged row projection"
        )
    replacements: dict[ast.Call, ast.expr] = {}
    for load in loads:
        assert isinstance(load.func, ast.Attribute)
        coordinates = scalar_index_offsets(load.func.value, site.matrix, 2)
        if coordinates is None:
            raise InvalidConfig("unknown scalar bank address in staged row projection")
        raw = coordinates[0]
        # The FX proof is for ordinary bracket indexing of the flattened
        # matrix, with no explicit input mask. Wrap *that row*, after the
        # original typed arithmetic, without changing the public expert ID.
        expr = ast.unparse(raw)
        (cast_parent,) = (
            term
            for term in ast.walk(load.func.value)
            if isinstance(term, ast.Call)
            and len(term.args) == 1
            and term.args[0] is raw
        )
        cast_parent.args[0] = ast.parse(
            f"(({expr}) + {rows} if ({expr}) < 0 else ({expr}))", mode="eval"
        ).body
        replacements[load] = ast.IfExp(
            ast.parse(f"({expr}) >= {-rows} and ({expr}) < {rows}", mode="eval").body,
            load,
            ast.parse("cutlass.BFloat16(0)", mode="eval").body,
        )

    class GuardLoads(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            if node in replacements:
                return replacements[node]
            return self.generic_visit(node)

    GuardLoads().visit(phase)


def _fast(
    s: IndexedRowPhase,
    r: _Recipes,
    width: int,
    threads: int,
    *,
    reuse_input: bool,
    stage_name: str,
) -> tuple[ast.If, list[ast.stmt], list[ast.stmt]] | None:
    k = s.k
    w = s.matrix
    x = s.vector
    total = "cute.arch.warp_reduction_sum(_stage_indexed_sum,threads_in_group=32)"
    inv = (
        f"cute.math.rsqrt(cute.math.div(cutlass.Float32({total}), cutlass.Float32({k}), approx=True, ftz=True) + {r.epsilon}, fastmath=True)"
        if r.approximate
        else f"cute.math.rsqrt({total} / cutlass.Float32({k}) + {r.epsilon})"
    )
    y = s.output
    idx = _target(r.selected)
    batch = r.batch
    slot = r.slot
    assert idx is not None
    prelude: list[ast.stmt] = []
    if s.rows % width or threads % 32 or width <= 0 or threads <= 0:
        return None
    if s.paired:
        if (
            s.input_scale is None
            or s.selected_ids is None
            or s.selected_weights is None
            or r.epsilon is None
        ):
            return None
        # Two warps collaborate across K for each eight-row group.
        groups = 2
        panel = groups * _COPY_PANEL
        if (
            width % _ROW_GROUP
            or threads // _WARP != (width // _ROW_GROUP) * groups
            or k % threads
            or k % panel
            or k // panel < 2
        ):
            return None
        j = width * groups
        score = s.weights
        guard = f"cutlass.const_expr({w}.iterator.alignment >= 16) and {w}.layout.stride[1] == 1 and {w}.layout.stride[0] % 8 == 0"
        setup = f"""_stage_indexed_xptr = cute.arch.alloc_smem(cutlass.BFloat16,{k},alignment=16)
_stage_indexed_x = cute.make_tensor(_stage_indexed_xptr,({k},))
_stage_indexed_pptr = cute.arch.alloc_smem(cutlass.Float32,{2 * j})
_stage_indexed_part = cute.make_tensor(_stage_indexed_pptr,({2 * j},))
"""
        lanes = """_stage_indexed_lane = cutlass.Int32(cute.arch.lane_idx())
_stage_indexed_warp = cutlass.Int32(cute.arch.warp_idx())
_stage_indexed_tid = _stage_indexed_lane + 32 * _stage_indexed_warp
"""
        normalize = f"""_stage_indexed_sum = cutlass.Float32(0)
for _stage_indexed_i in range({k // 32}):
    _stage_indexed_k = _stage_indexed_i * 32 + _stage_indexed_lane
    _stage_indexed_v = cutlass.Float32(({scalar_index_ptr(x, batch, "_stage_indexed_k")}).load())
    _stage_indexed_sum = _stage_indexed_sum + _stage_indexed_v * _stage_indexed_v
_stage_indexed_inv = {inv}
for _stage_indexed_band in range({k // threads}):
    _stage_indexed_k = _stage_indexed_band * {threads} + _stage_indexed_tid
    _stage_indexed_v = ({scalar_index_ptr(x, batch, "_stage_indexed_k")}).load()
    _stage_indexed_a = ({scalar_index_ptr(s.input_scale, "_stage_indexed_k")}).load()
    _stage_indexed_x[_stage_indexed_k] = cutlass.BFloat16(cutlass.Float32(_stage_indexed_v) * _stage_indexed_inv) * _stage_indexed_a
cute.arch.sync_threads()
"""
        # Both the typed access proof and the emitted address must describe
        # the sole batch row. Its scale and value cannot alias any phase
        # destination. Keep arithmetic and the CTA publication barrier intact.
        if reuse_input and s.invariant_rows and batch == "0":
            prelude = ast.parse(lanes + normalize).body
        fast = (
            lanes
            + ("" if prelude else normalize)
            + f"""_stage_indexed_group = _stage_indexed_warp // {width // 8}
_stage_indexed_local = _stage_indexed_warp % {width // 8} * 8 + _stage_indexed_lane // 4
_stage_indexed_row = {r.row_base} + _stage_indexed_local
_stage_indexed_low = {idx} * {2 * s.rows} + _stage_indexed_row
_stage_indexed_l,_stage_indexed_h = {stage_name}(_stage_indexed_scratch,{scalar_index_ptr(w, "_stage_indexed_low", "0")},{scalar_index_ptr(w, f"_stage_indexed_low + {s.rows}", "0")},_stage_indexed_xptr,_stage_indexed_lane%4*8+_stage_indexed_group*32,{panel},{k // panel},{threads})
if _stage_indexed_lane % 4 == 0:
    _stage_indexed_part[_stage_indexed_group*{width}+_stage_indexed_local] = _stage_indexed_l
    _stage_indexed_part[{j}+_stage_indexed_group*{width}+_stage_indexed_local] = _stage_indexed_h
cute.arch.sync_threads()
if _stage_indexed_group==0 and _stage_indexed_lane%4==0:
    _stage_indexed_dst = cutlass.Int32({batch}) * cutlass.Int32({s.selection}.shape[1]) + cutlass.Int32({slot})
    ({scalar_index_ptr(y, "_stage_indexed_dst", "_stage_indexed_row")}).store(cutlass.BFloat16(_stage_indexed_part[_stage_indexed_local]+_stage_indexed_part[{width}+_stage_indexed_local]))
    ({scalar_index_ptr(y, "_stage_indexed_dst", f"_stage_indexed_row + {s.rows}")}).store(cutlass.BFloat16(_stage_indexed_part[{j}+_stage_indexed_local]+_stage_indexed_part[{j + width}+_stage_indexed_local]))
if {r.row_base}==0 and _stage_indexed_tid==0:
    ({scalar_index_ptr(s.selected_ids, batch, slot)}).store({idx})
    ({scalar_index_ptr(s.selected_weights, batch, slot)}).store(({scalar_index_ptr(score, batch, slot)}).load())
cute.arch.sync_threads()
"""
        )
    else:
        if width != 16 * (threads // _WARP) or k % _COPY_PANEL or k // _COPY_PANEL < 2:
            return None
        guard = f"cutlass.const_expr({w}.iterator.alignment >= 16 and {x}.iterator.alignment >= 16) and {w}.layout.stride[1] == 1 and {w}.layout.stride[0] % 8 == 0 and {x}.layout.stride[2] == 1 and {x}.layout.stride[1] % 8 == 0 and {x}.layout.stride[0] % 8 == 0"
        setup = ""
        fast = f"""_stage_indexed_lane = cutlass.Int32(cute.arch.lane_idx())
_stage_indexed_row = {r.row_base} + 16*cutlass.Int32(cute.arch.warp_idx()) + _stage_indexed_lane//4
_stage_indexed_low = {idx} * {s.rows} + _stage_indexed_row
_stage_indexed_scale = ({scalar_index_ptr(s.weights, batch, slot)}).load()
_stage_indexed_l,_stage_indexed_h = {stage_name}(_stage_indexed_scratch,{scalar_index_ptr(w, "_stage_indexed_low", "0")},{scalar_index_ptr(w, "_stage_indexed_low + 8", "0")},{scalar_index_ptr(x, batch, slot, "0")},_stage_indexed_lane%4*8,32,{k // 32},{threads})
if _stage_indexed_lane%4==0:
    ({scalar_index_ptr(y, batch, slot, "_stage_indexed_row")}).store(cutlass.BFloat16(_stage_indexed_l*_stage_indexed_scale))
    ({scalar_index_ptr(y, batch, slot, "_stage_indexed_row + 8")}).store(cutlass.BFloat16(_stage_indexed_h*_stage_indexed_scale))
cute.arch.sync_threads()
"""
    test = ast.parse(f"{guard} and {idx} >= 0 and {idx} < {s.groups}", mode="eval").body
    return (
        ast.If(test, ast.parse(fast).body, r.original),
        ast.parse(setup).body,
        prelude,
    )


def lower_indexed_row_tiles(
    definition: ast.FunctionDef,
    prefix: list[ast.stmt],
    ctx: PassContext,
    sites: tuple[IndexedRowPhase, ...],
    ir: DeviceIR,
) -> None:
    request_epilogue = ctx.config.get(INDEXED_ROW_EPILOGUE_KEY) is True

    request_reuse = ctx.config.get(REUSE_INDEXED_INPUT_KEY) is True

    def decline() -> None:
        if request_reuse:
            raise InvalidConfig(
                "cannot reuse the staged row input on this physical launch"
            )
        if request_epilogue:
            raise InvalidConfig("cannot fuse row epilogue with this physical launch")

    root_ids = ir.root_ids
    phases = [n for n in definition.body if isinstance(n, ast.For)]
    if not sites:
        decline()
        return
    if len(phases) != len(root_ids):
        raise InvalidConfig(
            "cannot identify a scalar phase for each indexed row projection"
        )
    by_root = dict(zip(root_ids, phases, strict=True))
    # FX identifies host views; emitted loads/stores use the allocated device
    # argument, whose name can differ when the host variable shadows an import.
    arguments: dict[str, str | None] = {}
    for arg in ctx.fn.arguments:
        if type(arg) is TensorArg and arg._host_str is not None:
            host = arg.host_str()
            arguments[host] = arg.name if host not in arguments else None
    names = {host: device for host, device in arguments.items() if device is not None}
    scalar_sites = []
    for original in sites:
        if original.root not in by_root or original.matrix not in names:
            raise InvalidConfig(
                "cannot identify the scalar bank of an indexed row projection"
            )
        scalar_sites.append(replace(original, matrix=names[original.matrix]))

    def guard_scalar_fallbacks() -> None:
        for site in scalar_sites:
            _guard_scalar_bank_reads(site, by_root[site.root])

    block = ctx.exact_thread_block_dims
    if (
        block is None
        or block[2] != 1
        or ctx.config.get("cross_loop_pipeline") != "barrier"
        or ctx.config.get("cute_defer_tile_reduce") is not True
        or any(
            n.id.startswith(_LOCAL)
            for n in ast.walk(definition)
            if isinstance(n, ast.Name)
        )
    ):
        guard_scalar_fallbacks()
        decline()
        return
    # The generated host definition follows the imports. Neither it nor a
    # device parameter may hide the helper from a traced call.
    occupied = _bound_names(ast.Module(body=[*prefix, definition], type_ignores=[]))
    occupied.add(ctx.fn.codegen.host_function.name)
    stage_name = _STAGE_NAME
    while stage_name in occupied:
        stage_name += "_"
    specs = ctx.env.config_spec
    reusable = reusable_row_axes(specs, ctx.config) if request_reuse else set()
    accepted = []
    for original in sites:
        # Keep the typed site for the IR analysis of its optional epilogue.
        # The physical pass must only read arguments of this actual function.
        try:
            s = replace(
                original,
                matrix=names[original.matrix],
                vector=names[original.vector],
                selection=names[original.selection],
                weights=names[original.weights],
                output=names[original.output],
                input_scale=names[original.input_scale]
                if original.input_scale is not None
                else None,
                selected_ids=names[original.selected_ids]
                if original.selected_ids is not None
                else None,
                selected_weights=names[original.selected_weights]
                if original.selected_weights is not None
                else None,
            )
        except KeyError:
            continue
        width = ctx.fn.resolved_block_size(s.row_axis)
        if not isinstance(width, int) or width <= 0 or s.root not in by_root:
            continue
        r = _recipes(
            definition, by_root[s.root], s, width, block, set(ctx.constexpr_values)
        )
        if r is None:
            continue
        keep_input = (
            s.row_axis in reusable
            and s.invariant_rows > 0
            and not {s.vector, s.input_scale, r.epsilon}.intersection(
                names_written(by_root[s.root])
            )
        )
        if (
            result := _fast(
                s, r, width, prod(block), reuse_input=keep_input, stage_name=stage_name
            )
        ) is not None:
            accepted.append((original, r, *result))
    guard_scalar_fallbacks()
    if not accepted:
        decline()
        return
    allocated = set()
    # A phase is a sequential outer loop. Its staged branch ends in a CTA
    # fence and drains all copies in the helper, and the fallback never
    # touches these buffers. No two phases execute concurrently on one CTA.
    setup = ast.parse(
        f"_stage_indexed_scratch = cute.arch.alloc_smem(cutlass.BFloat16,{48 * prod(block)},alignment=128)"
    ).body
    if request_epilogue:
        # Shared scratch for a pair is named once per kernel. Give it to an
        # original producer whose *complete* consumer recipe actually lowers.
        # A failed trial leaves both the detached branch and the kernel alone.
        for i, (site, _, staged, _, _) in enumerate(accepted):
            p = indexed_row_epilogue(ctx.env, ir, site) if site.paired else None
            if p is not None:
                src, dest = arguments.get(p.input), arguments.get(p.output)
                if src is None or dest is None:
                    continue
                p = replace(p, input=src, output=dest)
            if p is not None and row_epilogue.fuse_row_epilogue(
                definition,
                by_root[site.root],
                by_root[p.consumer],
                staged,
                p,
            ):
                accepted.insert(0, accepted.pop(i))
                break
        else:
            decline()
            return
    reused = False
    for site, r, new, extra, prelude in accepted:
        new_allocations = {_target(n) for n in extra}
        if new_allocations & allocated:
            continue
        ctx.fn.cute_state.installed_staged_roots.add(site.root)
        ctx.fn.cute_state.staged_indexed_block = block
        allocated.update(new_allocations)
        r.branch.body[-len(r.original) :] = [
            cast("ast.Assign", clone_ast(r.selected)),
            new,
        ]
        setup.extend(extra)
        if prelude:
            # Emit only for an installed branch, after its incoming grid join.
            # A skipped producer must never eagerly read its input.
            phase = by_root[site.root]
            i = definition.body.index(phase)
            definition.body[i:i] = prelude
            reused = True
    if request_reuse and not reused:
        raise InvalidConfig("cannot reuse the staged row input on this physical launch")
    definition.body[:0] = setup
    prefix.extend(
        ast.parse(
            f"from helion._compiler.cute.device.staged_bf16_rows import staged_row_dot as {stage_name}"
        ).body
    )
