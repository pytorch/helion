"""Device-code emitter of the ``cute_mma_dag`` warp mode (see cute_mma_dag.py).

The body is rendered as source text from the :class:`MmaDagPlan` (as
``cute_warp_mma_gemm.py`` does) and parsed into the root tile loop's
statements; the launch keeps the generic flat grid (one CTA per root tile)
and owns the block shape through ``cute_state.owned_root_block_dims``.

Program of one CTA (``32 * warps`` threads; every fragment slot has its own
warp grid of 16x8 ``mma.sync`` atoms; ``S`` operand ring slots)::

    fragment registers <- zeros | per-lane loads of the seed tiles
    cp.async the epilogue's operand tiles (one group; they land during the loop)
    cp.async the source tiles of the first S - 1 loop iterations (one group each)
    for i in loop:
        [S == 1] cp.async this iteration's tiles, commit
        synchronous operand passes (unaligned sources: pair loads, pointwise
            DAG, st.shared into the single-slot swizzled tile)
        snapshots of the carries (staging tile, or pair stores to global)
        accumulators that depend on nothing computed in the iteration:
            zeros, the carry, a rescaled carry no earlier step reads
        exchange tiles of the carry at loop entry (bf16 pairs at the fragment
            coordinates, double-buffered by iteration parity)
        cp.async wait (S - 2 younger groups may stay in flight)
        in-place operand passes: each thread rewrites its own landed packets
            through the pointwise DAG (ld.shared.v4, Inductor lowering, pack,
            st.shared.v4)
        CTA barrier
        [S > 1] cp.async the tiles of iteration i + S - 1 into the slot read
            one iteration ago, commit (empty past the end)
        staging tile -> 16-byte global stores; then the body in program order:
            dot: exchange tiles of fragments computed in this iteration + CTA
                barrier, the remaining accumulator rescale, K / 16 steps of
                ldmatrix(.trans) / register repack / movmatrix + mma.sync
            derived fragment: pointwise DAG per register over the input
                fragments, 2-D tiles loaded as pairs at the coordinates
            store of a dot / derived value: pair stores at its program point
        [S == 1] CTA barrier before the refill
    epilogue: wait for the epilogue tiles, their passes, fragment -> shared
        exchange tiles (bf16 pairs at the fragment coordinates), CTA barrier,
        the epilogue dots (operands from shared tiles, exchange tiles or the
        identity repack of a same-layout fragment), stores of the results

Pointwise values (operand prologues, masks, rescales, store values) go
through Inductor's ``PointwiseLowering.codegen_from_input_asts`` (and the
``where`` / ``scalar_tensor`` aten lowerings) per element, so ``exp`` /
``exp2`` / casts / ``where`` are the expressions the SIMT path emits; 16-bit
values are re-typed to their DSL dtype before the ops run and packed with
``cvt.rn`` pairs, so every rounding step matches.  Fragment elements are
placed by coordinates, so a ``[C, C]`` tile over one reduction dim and an
``hl.arange`` mask need no lane bookkeeping.
"""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ..compile_environment import CompileEnvironment
from .cute_mma_dag import AXIS_LEAD
from .cute_mma_dag import AXIS_LOOP
from .cute_mma_dag import AXIS_RDIM
from .cute_mma_dag import MMA_DAG_TCGEN05
from .cute_mma_dag import REGISTER_ROUTES
from .cute_mma_dag import ROUTE_CONVERT
from .cute_mma_dag import ROUTE_DIRECT
from .cute_mma_dag import ROUTE_EXCHANGE
from .cute_mma_dag import ROUTE_INPLACE
from .cute_mma_dag import ROUTE_MOVMATRIX
from .cute_mma_dag import ROUTE_SYNC
from .cute_mma_dag import Access
from .cute_mma_dag import Axis
from .cute_mma_dag import Derived
from .cute_mma_dag import Dot
from .cute_mma_dag import MmaDagPlan
from .cute_mma_dag import Operand
from .cute_mma_dag import Store
from .cute_mma_dag import _decline
from .cute_mma_dag import frag_word_blocks
from .cute_warp_mma_gemm import _swizzled_chunk
from .cute_warp_mma_gemm import warp_mma_warp_layout
from .matmul_utils import strip_new_vars

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx.node import Node

    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST

_RUNTIME = "_helion_mma_dag"
_PREAMBLE = f"import helion._compiler.cute._mma_dag_runtime as {_RUNTIME}\n"
Coords = tuple[str, ...]


def emit_mma_dag_module_statements(cg: GenerateAST) -> None:
    """Emit the once-per-module import of the runtime helpers."""
    if getattr(cg, "_helion_mma_dag_module_emitted", False):
        return
    cg._helion_mma_dag_module_emitted = True  # type: ignore[attr-defined]
    for stmt in ast.parse(_PREAMBLE).body:
        cg.module_statements.append(stmt)


def codegen_mma_dag(cg: GenerateAST) -> bool:
    """Emit the warp-MMA program for the planned root (False: not this root)."""
    df = cg.device_function
    plan = df.cute_state.mma_dag_plan
    root = cg.current_root_graph_info
    if plan is None or root is None or root.graph_id != plan.program.root_graph_id:
        return False
    if plan.compute == MMA_DAG_TCGEN05:
        from .cute_mma_dag_tcgen05 import codegen_mma_dag_tcgen05

        codegen_mma_dag_tcgen05(cg, plan)
        return True
    emit_mma_dag_module_statements(cg)
    source = _Emitter(cg, plan).render()
    for statement in ast.parse(textwrap.dedent(source)).body:
        cg.add_statement(statement)
    df.cute_state.owned_root_block_dims = (plan.threads, 1, 1)
    return True


class _Layout:
    """The warp grid of one fragment shape and the lane's coordinates in it."""

    def __init__(
        self,
        index: int,
        rows: int,
        cols: int,
        warps: int,
        layout: tuple[int, int] | None = None,
    ) -> None:
        if layout is None:
            layout = warp_mma_warp_layout(rows, cols, warps)
        if layout is None:
            raise _decline(f"{warps} warps cannot tile a {rows}x{cols} fragment")
        self.index = index
        self.rows, self.cols = rows, cols
        self.wm, self.wn = layout
        self.wtm, self.wtn = rows // self.wm, cols // self.wn
        self.atoms_m, self.atoms_n = self.wtm // 16, self.wtn // 8

    @property
    def rowl(self) -> str:
        return f"md_L{self.index}_rowl"

    @property
    def coll(self) -> str:
        return f"md_L{self.index}_coll"

    def row(self, am: int, half: int) -> str:
        offset = 16 * am + 8 * half
        return f"({self.rowl} + {offset})" if offset else self.rowl

    def col(self, an: int, bit: int = 0) -> str:
        offset = 8 * an + bit
        return f"({self.coll} + {offset})" if offset else self.coll

    def registers(self) -> list[tuple[int, int, int]]:
        return [
            (am, an, j)
            for am in range(self.atoms_m)
            for an in range(self.atoms_n)
            for j in range(4)
        ]


class _Evaluator:
    """Render FX values element by element at positional coordinates.

    ``coords`` holds one tile-local coordinate expression per dim of the
    value (``"0"`` for size-1 and leading dims); fragment leaves resolve to
    the register of the current ``reg`` (set by the per-register loops);
    2-D operand loads are supplied by the caller as already-loaded leaves
    keyed by the element's position in the loaded pair / packet.
    """

    def __init__(self, emitter: _Emitter) -> None:
        self.emitter = emitter
        self.program = emitter.program
        self.scopes: list[dict[object, str]] = [{}]
        self.tile_leaves: dict[tuple[Node, int], str] = {}
        # The exact fp32 values of 16-bit tile leaves (before re-typing).
        self.raw_leaves: dict[tuple[Node, int], str] = {}
        self.reg: tuple[int, int, int] | None = None
        self.element: int | None = None

    def push(self) -> None:
        self.scopes.append({})

    def pop(self) -> None:
        self.scopes.pop()

    def _lookup(self, key: object) -> str | None:
        for scope in reversed(self.scopes):
            if key in scope:
                return scope[key]
        return None

    def evaluate(self, node: Node, coords: Coords) -> str:
        from ...language import _tracing_ops
        from ...language import memory_ops
        from ...language import view_ops

        em = self.emitter
        if node.op == "call_function" and node.target is _tracing_ops._new_var:
            return self.evaluate(cast("Node", node.args[0]), coords)
        if node.op == "call_function" and node.target is _tracing_ops._mask_to:
            # An identity on whole tiles (every tile here divides its extent).
            return self.evaluate(cast("Node", node.args[0]), coords)
        if node in self.program.frag_slots:
            assert self.reg is not None
            return em.reg(self.program.frag_slots[node], *self.reg)
        # Memoized per element position: values that do not depend on the
        # packet element (a K-vector decay of the packet's row) are shared
        # across the packet; the loaded tile leaves are keyed by element.
        key = (node, coords)
        cached = self._lookup(key)
        if cached is not None:
            return cached
        widened = self.widened_leaf(node, coords)
        if widened is not None:
            result = widened
        elif node.op == "call_function" and node.target is memory_ops.load:
            result = self._load(node, coords)
        elif node.op == "call_function" and node.target is _tracing_ops._get_symnode:
            val = node.meta["val"]
            result = em.df.sympy_expr(val._sympy_())
        elif node.op == "call_function" and node.target is view_ops.subscript:
            index = cast("list[object]", node.args[1])
            base_coords = tuple(
                coord
                for coord, item in zip(coords, index, strict=True)
                if item is not None
            )
            result = self.evaluate(cast("Node", node.args[0]), base_coords)
        elif node.op == "call_function" and node.target in (
            torch.ops.aten.permute.default,
            torch.ops.aten.transpose.int,
        ):
            result = self.evaluate(
                cast("Node", node.args[0]), _permute_in(node, coords)
            )
        elif node.op == "call_function" and node.target is torch.ops.prims.iota.default:
            result = em.bind("md_v", f"cutlass.Int32({coords[0]})")
        else:
            result = self._lowered(node, coords)
        self.scopes[-1][key] = result
        return result

    def define(self, node: Node, coords: Coords) -> str:
        """Render the *definition* of a derived fragment at ``coords`` (the
        node is a fragment leaf, so ``evaluate`` would name its register)."""
        return self._lowered(node, coords)

    def unrounded(self, node: Node, coords: Coords) -> str:
        """The fp32 value whose round-to-nearest 16-bit conversion is
        ``node``'s value, for a consumer that rounds itself (``cvt.rn.*x2``
        of a pack): a ``convert_element_type(x, 16-bit)`` root yields ``x``
        (one rounding instead of a scalar ``cvt`` followed by the pack's),
        anything else its own value."""
        source = strip_new_vars(node)
        if (
            source.op == "call_function"
            and source.target is torch.ops.prims.convert_element_type.default
            and source.args[1] in (torch.bfloat16, torch.float16)
        ):
            return self.evaluate(cast("Node", source.args[0]), coords)
        return self.evaluate(node, coords)

    def widened_leaf(self, node: Node, coords: Coords) -> str | None:
        """``node`` is ``convert_element_type(load, float32)`` of a 16-bit
        load whose fp32 value is already at hand (the unpacked packet half or
        the widening global load): that value, exact, instead of the DSL's
        16-bit re-typing and widening casts."""
        if (
            node.op != "call_function"
            or node.target is not torch.ops.prims.convert_element_type.default
            or node.args[1] is not torch.float32
        ):
            return None
        source = strip_new_vars(cast("Node", node.args[0]))
        access = self.emitter.accesses.get(source)
        if access is None or access.dtype == torch.float32:
            return None
        if len(access.tile_dims) == 2:
            assert self.element is not None
            return self.raw_leaves.get((source, self.element))
        key = (source, coords)
        raw = self._lookup(("raw", key))
        if raw is None:
            # Evaluating the load binds both the raw and the typed value.
            self.evaluate(source, coords)
            raw = self._lookup(("raw", key))
        return raw

    def _load(self, node: Node, coords: Coords) -> str:
        em = self.emitter
        access = em.accesses[node]
        if len(access.tile_dims) == 2:
            assert self.element is not None
            return self.tile_leaves[(node, self.element)]
        address = em.address(access, coords)
        value = em.bind("md_v", em.scalar_load(address, access.dtype))
        if access.dtype == torch.float32:
            return value
        self.scopes[-1][("raw", (node, coords))] = value
        return em.bind("md_v", em.typed(value, access.dtype))

    def _lowered(self, node: Node, coords: Coords) -> str:
        from torch._inductor.virtualized import V
        from torch.fx.node import map_arg

        from ..ast_extension import expr_from_string
        from ..aten_lowering import AtenLowering
        from ..aten_lowering import LoweringContext
        from ..inductor_lowering import PointwiseLowering

        lowering = node.meta.get("lowering")
        em = self.emitter
        shape = _shape(node)
        env: dict[Node, object] = {}

        def visit(value: Node) -> Node:
            if value not in env:
                env[value] = expr_from_string(
                    self.evaluate(value, _broadcast(coords, shape, _shape(value)))
                )
            return value

        map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
        ctx = LoweringContext.__new__(LoweringContext)
        ctx.cg = em.cg
        ctx.env = env  # pyrefly: ignore[bad-assignment]
        statements: list[ast.AST] = []
        with em.cg.set_statements(statements), V.set_current_node(node):
            if isinstance(lowering, PointwiseLowering):
                result = lowering.codegen_from_input_asts(
                    ctx, node, lowering.input_asts(ctx, node)
                )
            elif isinstance(lowering, AtenLowering):
                result = lowering.codegen(ctx, node)
            else:
                raise _decline(f"{node.name} is not a pointwise value")
        if not isinstance(result, ast.AST):
            raise _decline(f"invalid pointwise result for {node.name}")
        for statement in statements:
            for line in ast.unparse(ast.fix_missing_locations(statement)).splitlines():
                em.emit(line)
        return em.bind("md_v", ast.unparse(result))


def _shape(node: Node) -> tuple[object, ...]:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    return ()


def _broadcast(
    coords: Coords, out_shape: tuple[object, ...], in_shape: tuple[object, ...]
) -> Coords:
    """The input's coordinates under right-aligned broadcasting."""
    offset = len(out_shape) - len(in_shape)
    return tuple(
        "0" if (isinstance(size, int) and size == 1) else coords[offset + i]
        for i, size in enumerate(in_shape)
    )


def _permute_in(node: Node, coords: Coords) -> Coords:
    """Coordinates of the input of a permute given the output's."""
    ndim = len(coords)
    if node.target is torch.ops.aten.permute.default:
        perm = [int(p) % ndim for p in cast("list[int]", node.args[1])]
    else:
        a, b = (
            int(cast("int", node.args[1])) % ndim,
            int(cast("int", node.args[2])) % ndim,
        )
        perm = list(range(ndim))
        perm[a], perm[b] = perm[b], perm[a]
    result = [""] * ndim
    for out_pos, in_pos in enumerate(perm):
        result[in_pos] = coords[out_pos]
    return tuple(result)


class _Tile:
    """A shared operand tile: where it lives and how it is shaped."""

    def __init__(
        self,
        name: str,
        offset: int,
        rows: int,
        cols: int,
        elem_bytes: int,
        slots: int,
    ) -> None:
        self.name = name
        self.offset = offset
        self.rows, self.cols = rows, cols
        self.row_bytes = cols * elem_bytes
        self.cpr = self.row_bytes // 16
        self.bytes = rows * self.row_bytes
        self.slots = slots
        # The raw source tile of a converted operand (same ring depth).
        self.raw: _Tile | None = None

    @property
    def packets(self) -> int:
        return self.rows * self.cpr

    def base(self, stage: str) -> str:
        if self.slots == 1 or stage == "0":
            return self.name
        return f"({self.name} + ({stage}) * {self.bytes})"


class _Emitter:
    def __init__(self, cg: GenerateAST, plan: MmaDagPlan) -> None:
        self.cg = cg
        self.df: DeviceFunction = cg.device_function
        self.env = CompileEnvironment.current()
        self.plan = plan
        self.program = plan.program
        self.lines: list[str] = []
        self.indent = ""
        self.accesses: dict[Node, Access] = {
            access.node: access for access in self.program.accesses
        }
        self.base_vars: dict[Node, str] = {}
        self.iter_vars: dict[Node, str] = {}
        self.deferred: list[str] = []
        self._defined_tiles: set[str] = set()
        self.evaluator = _Evaluator(self)
        program = self.program
        self.es = program.elem_bytes
        self.kind = "bf16" if program.dtype == torch.bfloat16 else "f16"
        self.sync = "cute.arch.barrier()" if plan.warps > 1 else "cute.arch.sync_warp()"
        self.loop_var = "md_i"
        # One warp grid per distinct fragment shape (the plan's solver picks
        # M-first or N-first per shape).
        self.layouts: dict[tuple[int, int], _Layout] = {}
        self.slot_layouts: list[_Layout] = []
        for slot in range(len(program.slots)):
            shape = plan.slot_shape(slot)
            if shape not in self.layouts:
                self.layouts[shape] = _Layout(
                    len(self.layouts), *shape, plan.warps, plan.slot_layout(slot)
                )
            self.slot_layouts.append(self.layouts[shape])
        # Shared memory: the plan's layout (ring tiles, synchronous /
        # epilogue tiles, exchange tiles shared by key, the derived values'
        # ring tiles, then the carry staging buffers); ``q`` feeds two dots
        # of fwd_o through one tile, ``dA`` is read straight and transposed
        # in dqk; an exchange tile of the carry at loop entry is written
        # before the iteration barrier and so double-buffered by parity.
        operand_specs, derived_specs, staging_specs, self.smem_bytes = (
            plan.smem_layout()
        )
        self.shared_tiles: dict[tuple[object, ...], _Tile] = {}
        for key, spec in operand_specs.items():
            tile = _Tile(
                spec.name,
                spec.offset,
                spec.rows,
                spec.cols,
                spec.elem_bytes,
                spec.slots,
            )
            if spec.raw_elem_bytes:
                tile.raw = _Tile(
                    f"{spec.name}r",
                    spec.raw_offset,
                    spec.rows,
                    spec.cols,
                    spec.raw_elem_bytes,
                    spec.slots,
                )
            self.shared_tiles[key] = tile
        self.tiles: dict[tuple[Node, str], _Tile] = {}
        self.tile_buffers: dict[tuple[Node, str], str] = {}
        for dot in program.dots:
            for operand in (dot.a, dot.b):
                route = plan.route(dot, operand)
                if route in REGISTER_ROUTES:
                    continue
                if dot.region != "loop":
                    buffer = "0"
                elif route == ROUTE_EXCHANGE:
                    buffer = (
                        "md_parity" if program.early_exchange(dot, operand) else "0"
                    )
                else:
                    buffer = "md_stage"
                self.tile_buffers[(dot.node, operand.role)] = buffer
                self.tiles[(dot.node, operand.role)] = self.shared_tiles[
                    operand.chain_key
                ]
        self.derived_tiles: dict[Node, _Tile] = {
            node: _Tile(
                spec.name,
                spec.offset,
                spec.rows,
                spec.cols,
                spec.elem_bytes,
                spec.slots,
            )
            for node, spec in derived_specs.items()
        }
        self.staging: dict[Node, _Tile] = {
            node: _Tile(
                spec.name, spec.offset, spec.rows, spec.cols, spec.elem_bytes, 2
            )
            for node, spec in staging_specs.items()
        }

    # -- rendering primitives ---------------------------------------------

    def emit(self, line: str) -> None:
        self.lines.append(f"{self.indent}{line}")

    def bind(self, prefix: str, expr: str) -> str:
        name = self.df.new_var(prefix)
        self.emit(f"{name} = {expr}")
        return name

    def nest(self, header: str) -> _Nest:
        return _Nest(self, header)

    def reg(self, slot: int, am: int, an: int, j: int) -> str:
        return f"md_c{slot}_{am}_{an}_{j}"

    def typed(self, value: str, dtype: torch.dtype) -> str:
        """``value`` (an fp32 scalar) re-typed to the DSL type of ``dtype``."""
        if dtype == torch.float32:
            return value
        return f"{self.env.backend.dtype_str(dtype)}({value})"

    def scalar_load(self, address: str, dtype: torch.dtype) -> str:
        if dtype == torch.float32:
            return f"{_RUNTIME}.ldg_f32({address})"
        if dtype == torch.float16:
            return f"{_RUNTIME}.ldg_f16_as_f32({address})"
        assert dtype == torch.bfloat16, dtype
        return f"{_RUNTIME}.ldg_bf16_as_f32({address})"

    def pair_load(
        self, address: str, dtype: torch.dtype, paired: bool
    ) -> tuple[str, str]:
        """Two adjacent elements at ``address`` as fp32 scalars."""
        lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
        if not paired:
            self.emit(f"{lo} = {self.scalar_load(address, dtype)}")
            self.emit(
                f"{hi} = {self.scalar_load(f'{address} + {dtype.itemsize}', dtype)}"
            )
        elif dtype == torch.float32:
            self.emit(f"{lo}, {hi} = {_RUNTIME}.ldg_v2_f32({address})")
        else:
            unpack = "unpack_f16x2" if dtype == torch.float16 else "unpack_bf16x2"
            self.emit(
                f"{lo}, {hi} = {_RUNTIME}.{unpack}({_RUNTIME}.ldg_b32({address}))"
            )
        return lo, hi

    def pair_store(
        self, address: str, lo: str, hi: str, dtype: torch.dtype, paired: bool
    ) -> None:
        if not paired:
            scalar = {
                torch.float32: "stg_f32",
                torch.float16: "stg_f16",
                torch.bfloat16: "stg_bf16",
            }[dtype]
            self.emit(f"{_RUNTIME}.{scalar}({address}, {lo})")
            self.emit(f"{_RUNTIME}.{scalar}({address} + {dtype.itemsize}, {hi})")
        elif dtype == torch.float32:
            self.emit(f"{_RUNTIME}.stg_v2_f32({address}, {lo}, {hi})")
        else:
            pack = "pack_f16x2" if dtype == torch.float16 else "pack_bf16x2"
            self.emit(f"{_RUNTIME}.stg_b32({address}, {_RUNTIME}.{pack}({lo}, {hi}))")

    def pack(self, lo: str, hi: str) -> str:
        pack = "pack_f16x2" if self.kind == "f16" else "pack_bf16x2"
        return f"{_RUNTIME}.{pack}(cutlass.Float32({lo}), cutlass.Float32({hi}))"

    def set_tile_leaf(self, load: Access, element: int, raw: str) -> None:
        """Bind element ``element`` of the loaded tile: ``raw`` is its exact
        fp32 value (an unpacked packet half or a widening load), the typed
        leaf the DSL value of the tile's dtype."""
        key = (load.node, element)
        self.evaluator.raw_leaves[key] = raw
        self.evaluator.tile_leaves[key] = self.typed(raw, load.dtype)

    def pairs_aligned(self, access: Access, cols_pos: int) -> bool:
        """Whether adjacent elements along tile dim ``cols_pos`` may move as
        one aligned pair (contiguous columns, pair-aligned strides and base)."""
        from .memory_ops import cute_tensor_base_is_aligned

        es = access.dtype.itemsize
        if access.tile_dim_strides[cols_pos] != 1:
            return False
        if not cute_tensor_base_is_aligned(self.env, access.fake, 2 * es):
            return False
        return all(
            (stride * es) % (2 * es) == 0 for stride in access.strides if stride != 1
        )

    # -- index expressions ---------------------------------------------------

    def uniform(self, expr: sympy.Expr, loop_value: str) -> str:
        """Render a uniform scalar subscript with the loop index as ``loop_value``."""
        replacements: dict[sympy.Symbol, sympy.Symbol] = {}
        for symbol in expr.free_symbols:
            symbol = cast("sympy.Symbol", symbol)
            if symbol == self.program.loop_symbol:
                replacements[symbol] = sympy.Symbol(loop_value, integer=True)
            else:
                replacements[symbol] = sympy.Symbol(
                    self.df.sympy_expr(symbol), integer=True
                )
        return self.env.backend.sympy_printer_expr(expr.xreplace(replacements))

    def bound(self, value: object) -> str:
        if isinstance(value, torch.fx.Node):
            val = value.meta["val"]
            if isinstance(val, int):
                return str(val)
            return self.df.sympy_expr(val._sympy_())
        return str(value)

    def axis_base(self, axis: Axis) -> str:
        """The CTA's first index along ``axis`` (loop axes: per iteration)."""
        return f"md_a{axis.index}"

    def define_axis_bases(self) -> None:
        for axis in self.program.axes:
            if axis.kind in (AXIS_LOOP, AXIS_RDIM):
                continue
            self.emit(
                f"{self.axis_base(axis)} = cutlass.Int32({self.cg.tile_begin_var(axis.block_id)})"
            )

    def define_bases(self) -> None:
        """Per-access byte bases: the tensor pointer plus the loop-invariant
        scalar subscripts and the root / leading tile offsets (Int64)."""
        for access in self.program.accesses:
            if access.node in self.base_vars:
                continue
            name = self.df.tensor_arg(access.fake, prefer_name=access.name).name
            self.df.placeholder_args.add(name)
            es = access.dtype.itemsize
            terms = [f"{name}.iterator.toint()"]
            for subscript, stride in zip(
                access.subscripts, access.strides, strict=True
            ):
                scale = stride * es
                if subscript.kind == "axis":
                    axis = subscript.axis
                    assert axis is not None
                    if axis.kind in (AXIS_LOOP, AXIS_RDIM):
                        continue
                    terms.append(f"cutlass.Int64({self.axis_base(axis)}) * {scale}")
                elif not subscript.loop_dependent:
                    assert subscript.expr is not None
                    terms.append(
                        f"cutlass.Int64({self.uniform(subscript.expr, self.loop_var)}) * {scale}"
                    )
            self.base_vars[access.node] = self.bind("md_base", " + ".join(terms))

    def iteration_base(self, access: Access, loop_value: str) -> str:
        """The access base for the iteration whose loop index is ``loop_value``."""
        if loop_value == self.loop_var and access.node in self.iter_vars:
            return self.iter_vars[access.node]
        terms = [self.base_vars[access.node]]
        es = access.dtype.itemsize
        loop_axis = self.program.loop_axis
        for subscript, stride in zip(access.subscripts, access.strides, strict=True):
            scale = stride * es
            if subscript.kind == "scalar" and subscript.loop_dependent:
                assert subscript.expr is not None
                terms.append(
                    f"cutlass.Int64({self.uniform(subscript.expr, loop_value)}) * {scale}"
                )
            elif subscript.kind == "axis" and subscript.axis is loop_axis:
                assert loop_axis is not None
                step = self.plan.size(loop_axis)
                terms.append(f"cutlass.Int64({loop_value}) * {step * scale}")
        if len(terms) == 1:
            return terms[0]
        return self.bind("md_p", " + ".join(terms))

    def loop_dependent(self, access: Access) -> bool:
        loop_axis = self.program.loop_axis
        return any(
            subscript.loop_dependent
            or (subscript.kind == "axis" and subscript.axis is loop_axis)
            for subscript in access.subscripts
        )

    def define_iteration_bases(self, *, in_loop: bool) -> None:
        """Per-access bases for the current scope: inside the loop every
        access, outside it only the loop-independent ones."""
        self.iter_vars = {}
        for access in self.program.accesses:
            if access.node in self.iter_vars:
                continue
            if not in_loop and self.loop_dependent(access):
                continue
            self.iter_vars[access.node] = self.iteration_base(access, self.loop_var)

    def address(self, access: Access, coords: Coords) -> str:
        """Byte address of the element at tile-local positional ``coords``."""
        terms = [self.iter_vars[access.node]]
        es = access.dtype.itemsize
        for position, axis in enumerate(access.dims):
            if axis.kind == AXIS_LEAD:
                continue
            stride = access.dim_strides[position] * es
            coord = coords[position]
            if coord == "0":
                continue
            terms.append(
                f"cutlass.Int64({coord}) * {stride}"
                if stride != 1
                else f"cutlass.Int64({coord})"
            )
        return " + ".join(terms)

    # -- fragment coordinates -----------------------------------------------------

    def frag_coords(
        self, node: Node, layout: _Layout, am: int, an: int, j: int, swap: bool = False
    ) -> Coords:
        """Positional coordinates of ``node`` for the register ``(am, an, j)``
        of a fragment with ``layout``; ``swap`` puts the fragment's row on
        the node's second tile dim (a transposed view of the fragment)."""
        shape = _shape(node)
        row, col = layout.row(am, j // 2), layout.col(an, j & 1)
        positions = [
            i
            for i, size in enumerate(shape)
            if not (isinstance(size, int) and size == 1)
        ]
        positions = [i for i in positions if not self._is_lead_dim(node, i)]
        if len(positions) != 2:
            raise _decline(f"{node.name} is not a two-dim fragment value")
        coords = ["0"] * len(shape)
        first, second = (col, row) if swap else (row, col)
        coords[positions[0]] = first
        coords[positions[1]] = second
        return tuple(coords)

    def _is_lead_dim(self, node: Node, position: int) -> bool:
        size = _shape(node)[position]
        block_id = self.env.resolve_block_id(size)
        if block_id is None:
            return False
        block_id = self.env.canonical_block_id(block_id)
        return any(
            axis.block_id == block_id and axis.kind == AXIS_LEAD
            for axis in self.program.axes
        )

    def tile_coords(self, operand: Operand, prow: str, col: str) -> Coords:
        """Positional coordinates of the operand VALUE for the source tile
        element at ``(prow, col)`` (rows / cols of the shared tile)."""
        source = operand.load.node if operand.load is not None else operand.frag
        assert source is not None
        value_shape = _shape(operand.value)
        source_shape = _shape(source)
        # The source's two tile dims in positional order and which is rows.
        positions = [
            i
            for i, size in enumerate(source_shape)
            if not (isinstance(size, int) and size == 1)
            and not self._is_lead_dim(source, i)
        ]
        assert len(positions) == 2
        # The shared tile's rows are the source's non-contiguous tile dim.
        rows_first = operand.load is None or operand.load.stride_at(positions[0]) != 1
        native = [prow, col] if rows_first else [col, prow]
        if operand.swap:
            native.reverse()
        coords = ["0"] * len(value_shape)
        value_positions = [
            i
            for i, size in enumerate(value_shape)
            if not (isinstance(size, int) and size == 1)
            and not self._is_lead_dim(operand.value, i)
        ]
        assert len(value_positions) == 2
        coords[value_positions[0]] = native[0]
        coords[value_positions[1]] = native[1]
        return tuple(coords)

    # -- shared tiles ---------------------------------------------------------------

    def own_packets(self, tile: _Tile, body: Callable[[str, str, str], None]) -> None:
        """Call ``body(idx, prow, pch)`` for every packet this thread owns of
        the tile (round-robin over the CTA, as the cp.async issue)."""
        packets, cpr = tile.packets, tile.cpr
        threads = self.plan.threads
        for rep in range(-(-packets // threads)):
            idx = f"({rep * threads} + md_tidx)" if rep else "md_tidx"
            guard = packets < (rep + 1) * threads
            nest = self.nest(f"if {idx} < {packets}:") if guard else None
            if nest is not None:
                nest.__enter__()
            self.evaluator.push()
            prow = self.bind("md_prow", f"{idx} // {cpr}")
            pch = self.bind("md_pch", f"{idx} % {cpr}")
            body(idx, prow, pch)
            self.evaluator.pop()
            if nest is not None:
                nest.__exit__(None, None, None)

    def ring_items(self) -> list[tuple[_Tile, Access, int]]:
        """The ring's tiles in issue order: the loop operands' source tiles
        (raw tiles of converted operands) and the derived values' tiles, as
        ``(tile, load, position of the tile's rows among the load's dims)``."""
        items: list[tuple[_Tile, Access, int]] = []
        for dot, operand in self.plan.ring_operands():
            tile = self.tiles[(dot.node, operand.role)]
            assert operand.load is not None
            items.append((tile.raw or tile, operand.load, operand.rows_pos))
        for _derived, access in self.plan.ring_derived():
            items.append((self.derived_tiles[access.node], access, 0))
        return items

    def emit_issue(
        self, items: list[tuple[_Tile, Access, int]], loop_value: str, stage: str
    ) -> None:
        """cp.async the tiles of iteration ``loop_value`` into ring slot
        ``stage`` (no commit)."""
        for tile, load, rows_pos in items:
            self.emit_issue_tile(tile, load, rows_pos, loop_value, stage)

    def emit_issue_operand(
        self, dot: Dot, operand: Operand, loop_value: str, stage: str
    ) -> None:
        load = operand.load
        assert load is not None
        tile = self.tiles[(dot.node, operand.role)]
        self.emit_issue_tile(
            tile.raw or tile, load, operand.rows_pos, loop_value, stage
        )

    def emit_issue_tile(
        self, tile: _Tile, load: Access, rows_pos: int, loop_value: str, stage: str
    ) -> None:
        row_stride = load.tile_dim_strides[rows_pos] * load.dtype.itemsize
        smem = tile.base(stage)
        gbase = self.iteration_base(load, loop_value)
        cpr, row_bytes = tile.cpr, tile.row_bytes

        def body(idx: str, prow: str, pch: str) -> None:
            swz = _swizzled_chunk(prow, pch, cpr)
            self.emit(
                f"{_RUNTIME}.cp_async_16({smem} + {prow} * {row_bytes} + {swz} * 16, "
                f"{gbase} + cutlass.Int64({prow}) * {row_stride} + cutlass.Int64({pch} * 16))"
            )

        self.own_packets(tile, body)

    def hoist_scalars(self, value: Node) -> None:
        """Evaluate the 0-d (uniform) values under ``value`` once."""
        for node in _scalar_nodes(self.program, value):
            self.evaluator.evaluate(node, ())

    def emit_inplace_pass(self, dot: Dot, operand: Operand, stage: str) -> None:
        """Rewrite the thread's own packets of the staged source tile through
        the operand's pointwise DAG (after this thread's cp.async wait; no
        barrier is needed before the CTA-wide one that precedes ldmatrix)."""
        load = operand.load
        assert load is not None
        tile = self.tiles[(dot.node, operand.role)]
        smem = tile.base(stage)
        unpack = "unpack_f16x2" if self.kind == "f16" else "unpack_bf16x2"
        self.hoist_scalars(operand.value)
        cpr, row_bytes = tile.cpr, tile.row_bytes

        def body(idx: str, prow: str, pch: str) -> None:
            swz = _swizzled_chunk(prow, pch, cpr)
            address = self.bind("md_pa", f"{smem} + {prow} * {row_bytes} + {swz} * 16")
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({address})")
            for q, word in enumerate(words):
                lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                self.emit(f"{lo}, {hi} = {_RUNTIME}.{unpack}({word})")
                self.set_tile_leaf(load, 2 * q, lo)
                self.set_tile_leaf(load, 2 * q + 1, hi)
            packed = []
            for q in range(4):
                values = []
                for e in (2 * q, 2 * q + 1):
                    self.evaluator.element = e
                    coords = self.tile_coords(operand, prow, f"({pch} * 8 + {e})")
                    values.append(self.evaluator.unrounded(operand.value, coords))
                self.evaluator.element = None
                packed.append(self.pack(values[0], values[1]))
            self.emit(f"{_RUNTIME}.sts_v4_b32({address}, {', '.join(packed)})")

        self.own_packets(tile, body)

    def emit_convert_pass(self, dot: Dot, operand: Operand, stage: str) -> None:
        """Rewrite the thread's own packets of the raw fp32 source tile
        through the operand's pointwise DAG into the 16-bit operand tile
        (four elements per raw packet -> one 8-byte pair of pairs)."""
        load = operand.load
        assert load is not None
        tile = self.tiles[(dot.node, operand.role)]
        raw = tile.raw
        assert raw is not None
        smem = tile.base(stage)
        raw_smem = raw.base(stage)
        self.hoist_scalars(operand.value)
        cpr, row_bytes = tile.cpr, tile.row_bytes

        def body(idx: str, prow: str, pch: str) -> None:
            swz = _swizzled_chunk(prow, pch, raw.cpr)
            values = [self.df.new_var("md_f") for _ in range(4)]
            self.emit(
                f"{', '.join(values)} = {_RUNTIME}.lds_v4_f32({raw_smem} + {prow} * {raw.row_bytes} + {swz} * 16)"
            )
            for e, value in enumerate(values):
                self.evaluator.tile_leaves[(load.node, e)] = value
            packed = []
            for q in range(2):
                pair = []
                for e in (2 * q, 2 * q + 1):
                    self.evaluator.element = e
                    coords = self.tile_coords(operand, prow, f"({pch} * 4 + {e})")
                    pair.append(self.evaluator.unrounded(operand.value, coords))
                self.evaluator.element = None
                packed.append(self.pack(pair[0], pair[1]))
            # Raw packet p of a row holds columns 4p..4p+3: 16-bit chunk p // 2,
            # byte offset 8 * (p % 2) within it.
            dst = _swizzled_chunk(prow, f"({pch} // 2)", cpr)
            self.emit(
                f"{_RUNTIME}.sts_v2_b32({smem} + {prow} * {row_bytes} + {dst} * 16 + ({pch} % 2) * 8, "
                f"{', '.join(packed)})"
            )

        self.own_packets(raw, body)

    def emit_sync_pass(self, dot: Dot, operand: Operand, stage: str) -> None:
        """Cooperative pass for an unaligned source: every thread evaluates
        element pairs along the contiguous dim straight from global memory
        and packs them into the swizzled shared tile."""
        load = operand.load
        assert load is not None
        tile = self.tiles[(dot.node, operand.role)]
        smem = tile.base(stage)
        pairs = tile.rows * tile.cols // 2
        cpp = tile.cols // 2
        threads = self.plan.threads
        paired = self.pairs_aligned(load, operand.cols_pos)
        load_es = load.dtype.itemsize
        row_stride = load.tile_dim_strides[operand.rows_pos] * load_es
        self.hoist_scalars(operand.value)
        for rep in range(-(-pairs // threads)):
            idx = f"({rep * threads} + md_tidx)" if rep else "md_tidx"
            guard = pairs < (rep + 1) * threads
            nest = self.nest(f"if {idx} < {pairs}:") if guard else None
            if nest is not None:
                nest.__enter__()
            self.evaluator.push()
            prow = self.bind("md_pr", f"{idx} // {cpp}")
            pcol = self.bind("md_pc", f"{idx} % {cpp}")
            address = (
                f"{self.iter_vars[load.node]} + cutlass.Int64({prow}) * {row_stride} "
                f"+ cutlass.Int64({pcol}) * {2 * load_es}"
            )
            lo, hi = self.pair_load(address, load.dtype, paired)
            self.set_tile_leaf(load, 0, lo)
            self.set_tile_leaf(load, 1, hi)
            values = []
            for half in range(2):
                self.evaluator.element = half
                col = f"(2 * {pcol} + {half})" if half else f"(2 * {pcol})"
                values.append(
                    self.evaluator.unrounded(
                        operand.value, self.tile_coords(operand, prow, col)
                    )
                )
            self.evaluator.element = None
            # Pair (2p, 2p + 1) sits in 16-byte chunk p // 4 at byte 4 * (p % 4).
            swz = _swizzled_chunk(prow, f"({pcol} // 4)", tile.cpr)
            self.emit(
                f"{_RUNTIME}.sts_b32({smem} + {prow} * {tile.row_bytes} + {swz} * 16 + ({pcol} % 4) * 4, "
                f"{self.pack(values[0], values[1])})"
            )
            self.evaluator.pop()
            if nest is not None:
                nest.__exit__(None, None, None)

    def emit_exchange(self, dot: Dot, operand: Operand) -> None:
        """Write the operand value (a pointwise DAG over a fragment) into its
        exchange tile at the fragment's own coordinates, as packed pairs."""
        assert operand.frag is not None
        tile = self.shared_tiles[operand.chain_key]
        base = tile.base(self.tile_buffers[(dot.node, operand.role)])
        slot = self.program.frag_slots[operand.frag]
        layout = self.slot_layouts[slot]
        es = operand.dtype.itemsize
        self.hoist_scalars(operand.value)
        for am in range(layout.atoms_m):
            for an in range(layout.atoms_n):
                for half in range(2):
                    values = []
                    for bit in range(2):
                        j = 2 * half + bit
                        self.evaluator.reg = (am, an, j)
                        coords = self.frag_coords(
                            operand.value, layout, am, an, j, operand.swap
                        )
                        values.append(self.evaluator.unrounded(operand.value, coords))
                    self.evaluator.reg = None
                    row, col = layout.row(am, half), layout.col(an)
                    chunk = f"(({col}) * {es} // 16)"
                    swz = _swizzled_chunk(row, chunk, tile.cpr)
                    self.emit(
                        f"{_RUNTIME}.sts_b32({base} + {row} * {tile.row_bytes} + {swz} * 16 + ({col}) * {es} % 16, "
                        f"{self.pack(values[0], values[1])})"
                    )

    # -- carry snapshots (loop stores) ---------------------------------------

    def emit_stage_in(self, store: Store, parity: str) -> None:
        """Write the fragment pairs into the staging buffer of ``parity``."""
        tile = self.staging[store.access.node]
        slot = self.program.frag_slots[store.frag]
        layout = self.slot_layouts[slot]
        es = store.access.dtype.itemsize
        base = tile.base(parity)
        pack = "pack_f16x2" if store.access.dtype == torch.float16 else "pack_bf16x2"
        for am in range(layout.atoms_m):
            for an in range(layout.atoms_n):
                for half in range(2):
                    row, col = layout.row(am, half), layout.col(an)
                    chunk = f"(({col}) * {es} // 16)"
                    swz = _staging_swizzle(row, chunk, tile.cpr, es)
                    address = f"{base} + {row} * {tile.row_bytes} + {swz} * 16 + ({col}) * {es} % 16"
                    lo = self.reg(slot, am, an, 2 * half)
                    hi = self.reg(slot, am, an, 2 * half + 1)
                    if store.access.dtype == torch.float32:
                        self.emit(f"{_RUNTIME}.sts_v2_f32({address}, {lo}, {hi})")
                    else:
                        self.emit(
                            f"{_RUNTIME}.sts_b32({address}, {_RUNTIME}.{pack}({lo}, {hi}))"
                        )

    def emit_stage_out(self, store: Store, parity: str, loop_value: str) -> None:
        """Copy the staging buffer of ``parity`` to global memory as 16-byte
        packets, every thread its round-robin share (after the CTA barrier
        that follows its writes); the global stores are deferred past the
        first fragment loads so the shared-memory latency is not exposed."""
        tile = self.staging[store.access.node]
        access = store.access
        es = access.dtype.itemsize
        row_stride = access.tile_dim_strides[0] * es
        gbase = self.iteration_base(access, loop_value)
        base = tile.base(parity)
        threads = self.plan.threads
        for rep in range(-(-tile.packets // threads)):
            idx = f"({rep * threads} + md_tidx)" if rep else "md_tidx"
            guard = tile.packets < (rep + 1) * threads
            nest = self.nest(f"if {idx} < {tile.packets}:") if guard else None
            if nest is not None:
                nest.__enter__()
            prow = self.bind("md_orow", f"{idx} // {tile.cpr}")
            pch = self.bind("md_och", f"{idx} % {tile.cpr}")
            swz = _staging_swizzle(prow, pch, tile.cpr, es)
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(
                f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({base} + {prow} * {tile.row_bytes} + {swz} * 16)"
            )
            line = (
                f"{_RUNTIME}.stg_v4_b32({gbase} + cutlass.Int64({prow}) * {row_stride} "
                f"+ cutlass.Int64({pch} * 16), {', '.join(words)})"
            )
            if nest is not None:
                self.emit(line)
                nest.__exit__(None, None, None)
            else:
                self.deferred.append(line)

    def flush_deferred(self) -> None:
        for line in self.deferred:
            self.emit(line)
        self.deferred = []

    def emit_store(self, store: Store) -> None:
        """Pair stores of a fragment-derived value at the store's coordinates."""
        access = store.access
        slot = self.program.frag_slots[store.frag]
        layout = self.slot_layouts[slot]
        positions = access.tile_positions
        paired = self.pairs_aligned(access, 1) and not store.swap
        self.hoist_scalars(store.value)
        for am in range(layout.atoms_m):
            for an in range(layout.atoms_n):
                for half in range(2):
                    values = []
                    for bit in range(2):
                        j = 2 * half + bit
                        self.evaluator.reg = (am, an, j)
                        coords = self.frag_coords(
                            store.value, layout, am, an, j, store.swap
                        )
                        # Paired 16-bit stores pack (and so round) the pair.
                        if paired and access.dtype != torch.float32:
                            values.append(self.evaluator.unrounded(store.value, coords))
                        else:
                            values.append(self.evaluator.evaluate(store.value, coords))
                    self.evaluator.reg = None
                    row, col = layout.row(am, half), layout.col(an)
                    # The store's positional coordinates of the fragment element.
                    coords = ["0"] * len(access.dims)
                    first, second = (col, row) if store.swap else (row, col)
                    coords[positions[0]] = first
                    coords[positions[1]] = second
                    if paired:
                        address = self.address(access, tuple(coords))
                        self.pair_store(
                            address, values[0], values[1], access.dtype, True
                        )
                    else:
                        for bit in range(2):
                            element = list(coords)
                            element[positions[1 if not store.swap else 0]] = (
                                f"({col} + {bit})" if bit else col
                            )
                            address = self.address(access, tuple(element))
                            scalar = {
                                torch.float32: "stg_f32",
                                torch.float16: "stg_f16",
                                torch.bfloat16: "stg_bf16",
                            }[access.dtype]
                            self.emit(f"{_RUNTIME}.{scalar}({address}, {values[bit]})")

    # -- fragments ----------------------------------------------------------------

    def emit_carry_init(self) -> None:
        for carry in self.program.carries:
            layout = self.slot_layouts[carry.slot]
            if carry.init is None:
                for am in range(layout.atoms_m):
                    for an in range(layout.atoms_n):
                        self.emit(
                            "; ".join(
                                f"{self.reg(carry.slot, am, an, j)} = cutlass.Float32(0.0)"
                                for j in range(4)
                            )
                        )
                continue
            init = carry.init
            self.iter_vars[init.node] = self.base_vars[init.node]
            paired = self.pairs_aligned(init, 1)
            positions = [
                i for i, axis in enumerate(init.dims) if axis.kind != AXIS_LEAD
            ]
            for am in range(layout.atoms_m):
                for an in range(layout.atoms_n):
                    for half in range(2):
                        coords = ["0"] * len(init.dims)
                        coords[positions[0]] = layout.row(am, half)
                        coords[positions[1]] = layout.col(an)
                        address = self.address(init, tuple(coords))
                        lo, hi = self.pair_load(address, init.dtype, paired)
                        self.emit(f"{self.reg(carry.slot, am, an, 2 * half)} = {lo}")
                        self.emit(
                            f"{self.reg(carry.slot, am, an, 2 * half + 1)} = {hi}"
                        )

    def emit_acc(self, dot: Dot) -> None:
        """Materialize the dot's accumulator registers: zeros, the fragment
        leaf in place, or a pointwise DAG over the leaf evaluated per register."""
        layout = self.slot_layouts[dot.slot]
        if dot.acc is None:
            for am, an, j in layout.registers():
                self.emit(f"{self.reg(dot.slot, am, an, j)} = cutlass.Float32(0.0)")
            return
        if dot.acc_frag is not None and strip_new_vars(dot.acc) is dot.acc_frag:
            if dot.slot != dot.acc_slot:
                for am, an, j in layout.registers():
                    self.emit(
                        f"{self.reg(dot.slot, am, an, j)} = {self.reg(cast('int', dot.acc_slot), am, an, j)}"
                    )
            return
        self.hoist_scalars(dot.acc)
        updates = []
        for am, an, j in layout.registers():
            self.evaluator.reg = (am, an, j)
            coords = self.frag_coords(dot.acc, layout, am, an, j)
            updates.append(
                (
                    self.reg(dot.slot, am, an, j),
                    self.evaluator.evaluate(dot.acc, coords),
                )
            )
        self.evaluator.reg = None
        for name, value in updates:
            self.emit(f"{name} = {value}")

    def emit_frag_words(
        self, dot: Dot, operand: Operand
    ) -> dict[tuple[int, int], list[str]]:
        """The operand words of every (atom, K step) from the producer
        fragment's registers (see ``frag_register_route``).

        The producer's C layout holds, per 8x8 block ``(am, half)`` x ``an``,
        row ``g`` / columns ``2c, 2c + 1`` in each lane.  An ``[M, K]`` A
        operand's four words are those blocks packed (rows ``g`` / ``g + 8``,
        K blocks ``2 st`` / ``2 st + 1``); an ``[N, K]`` B operand's two words
        are the packed blocks (n-block ``an``, K block ``2 st`` / ``2 st +
        1``); a ``[K, N]`` B operand needs each block transposed
        (``movmatrix``): K block ``kb`` is the producer's rows ``(kb // 2, kb
        % 2)``.
        """
        assert operand.frag is not None
        slot = self.program.frag_slots[operand.frag]
        layout = self.slot_layouts[slot]
        self.hoist_scalars(operand.value)
        values: dict[tuple[int, int, int], str] = {}
        for am, an, j in layout.registers():
            self.evaluator.reg = (am, an, j)
            coords = self.frag_coords(operand.value, layout, am, an, j, operand.swap)
            values[(am, an, j)] = self.evaluator.unrounded(operand.value, coords)
        self.evaluator.reg = None
        route = self.plan.route(dot, operand)

        def block(am: int, an: int, half: int) -> str:
            packed = self.pack(
                values[(am, an, 2 * half)], values[(am, an, 2 * half + 1)]
            )
            if route == ROUTE_MOVMATRIX:
                return f"{_RUNTIME}.movmatrix_trans({packed})"
            return packed

        words: dict[tuple[int, int], list[str]] = {}
        k = self.plan.size(dot.k)
        consumer = self.slot_layouts[dot.slot]
        atoms = consumer.atoms_m if operand.role == "a" else consumer.atoms_n
        prefix = "md_aw" if operand.role == "a" else "md_bw"
        for atom in range(atoms):
            for st in range(k // 16):
                words[(atom, st)] = [
                    self.bind(prefix, block(*blk))
                    for blk in frag_word_blocks(operand.role, route, atom, st)
                ]
        return words

    def emit_dot(self, dot: Dot, stage: str) -> None:
        """K / 16 steps of ldmatrix + mma.sync accumulating into the slot."""
        plan = self.plan
        layout = self.slot_layouts[dot.slot]
        k = plan.size(dot.k)
        steps = k // 16
        mma = f"{_RUNTIME}.mma_{self.kind}"
        a_words = (
            self.emit_frag_words(dot, dot.a)
            if plan.route(dot, dot.a) in REGISTER_ROUTES
            else None
        )
        b_words = (
            self.emit_frag_words(dot, dot.b)
            if plan.route(dot, dot.b) in REGISTER_ROUTES
            else None
        )
        a_tile = self.tiles.get((dot.node, "a"))
        b_tile = self.tiles.get((dot.node, "b"))
        # Per-lane ldmatrix row / chunk terms (see ``ldmatrix_lane_address``);
        # the warp's M / N offset follows the slot's warp grid.
        wmi, wni = f"md_L{layout.index}_wmi", f"md_L{layout.index}_wni"
        wtm, wtn = layout.wtm, layout.wtn
        arow = achunk = a_base = brow = bchunk = b_base = ""
        if a_tile is not None:
            if dot.a.trans:
                arow = self.bind("md_arow", "md_l8 + 8 * (md_lq // 2)")
                achunk = self.bind("md_achunk", f"({wmi} * {wtm}) // 8 + md_lq % 2")
            else:
                arow = self.bind("md_arow", f"{wmi} * {wtm} + md_l8 + 8 * (md_lq % 2)")
                achunk = self.bind("md_achunk", "md_lq // 2")
            a_base = a_tile.base(self.tile_buffers[(dot.node, "a")])
        if b_tile is not None:
            if dot.b.trans:
                brow = self.bind("md_brow", "md_l8 + 8 * (md_lq % 2)")
                bchunk = self.bind("md_bchunk", f"({wni} * {wtn}) // 8 + md_lq // 2")
            else:
                brow = self.bind("md_brow", f"{wni} * {wtn} + md_l8 + 8 * (md_lq // 2)")
                bchunk = self.bind("md_bchunk", "md_lq % 2")
            b_base = b_tile.base(self.tile_buffers[(dot.node, "b")])
        for st in range(steps):
            for am in range(layout.atoms_m):
                if a_words is not None:
                    a0, a1, a2, a3 = a_words[(am, st)]
                    self.emit(
                        f"md_a_{am}_0, md_a_{am}_1, md_a_{am}_2, md_a_{am}_3 = {a0}, {a1}, {a2}, {a3}"
                    )
                    continue
                assert a_tile is not None
                if dot.a.trans:
                    row = f"({16 * st} + {arow})"
                    chunk = f"({achunk} + {2 * am})" if am else achunk
                    op = "ldmatrix_x4_trans"
                else:
                    row = f"({arow} + {16 * am})" if am else arow
                    chunk = f"({2 * st} + {achunk})"
                    op = "ldmatrix_x4"
                swz = _swizzled_chunk(row, chunk, a_tile.cpr)
                self.emit(
                    f"md_a_{am}_0, md_a_{am}_1, md_a_{am}_2, md_a_{am}_3 = {_RUNTIME}.{op}("
                    f"{a_base} + {row} * {a_tile.row_bytes} + {swz} * 16)"
                )
            if b_words is not None:
                for an in range(layout.atoms_n):
                    b0, b1 = b_words[(an, st)]
                    self.emit(f"md_b_{an}_0, md_b_{an}_1 = {b0}, {b1}")
            an = 0
            while b_words is None and an < layout.atoms_n:
                assert b_tile is not None
                pair = an + 1 < layout.atoms_n
                if dot.b.trans:
                    row = f"({16 * st} + {brow})"
                    if pair:
                        chunk = f"({bchunk} + {an})" if an else bchunk
                    else:
                        chunk = f"(({wni} * {wtn}) // 8 + {an})"
                    op = "ldmatrix_x4_trans" if pair else "ldmatrix_x2_trans"
                else:
                    if pair:
                        row = f"({brow} + {8 * an})" if an else brow
                    else:
                        row = f"({wni} * {wtn} + {8 * an} + md_l8)"
                    chunk = f"({2 * st} + {bchunk})"
                    op = "ldmatrix_x4" if pair else "ldmatrix_x2"
                swz = _swizzled_chunk(row, chunk, b_tile.cpr)
                address = f"{b_base} + {row} * {b_tile.row_bytes} + {swz} * 16"
                if pair:
                    self.emit(
                        f"md_b_{an}_0, md_b_{an}_1, md_b_{an + 1}_0, md_b_{an + 1}_1 = {_RUNTIME}.{op}({address})"
                    )
                else:
                    self.emit(f"md_b_{an}_0, md_b_{an}_1 = {_RUNTIME}.{op}({address})")
                an += 2 if pair else 1
            if st == 0:
                self.flush_deferred()
            for am in range(layout.atoms_m):
                for an in range(layout.atoms_n):
                    acc = ", ".join(self.reg(dot.slot, am, an, j) for j in range(4))
                    self.emit(
                        f"{acc} = {mma}(md_a_{am}_0, md_a_{am}_1, md_a_{am}_2, md_a_{am}_3, "
                        f"md_b_{an}_0, md_b_{an}_1, {acc})"
                    )

    # -- the body -----------------------------------------------------------------

    def render(self) -> str:
        plan = self.plan
        self.emit("md_tidx, _, _ = cute.arch.thread_idx()")
        self.emit("md_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())")
        self.emit("md_lane = md_tidx % 32")
        self.emit("md_g = md_lane // 4")
        self.emit("md_c = md_lane % 4")
        self.emit("md_l8 = md_lane % 8")
        self.emit("md_lq = md_lane // 8")
        for layout in self.layouts.values():
            # The lane's tile-local fragment coordinates in this warp grid:
            # rows g / g + 8 of each atom row, columns 2c / 2c + 1 of each
            # atom column.
            self.emit(f"md_L{layout.index}_wmi = md_warp // {layout.wn}")
            self.emit(f"md_L{layout.index}_wni = md_warp % {layout.wn}")
            self.emit(f"{layout.rowl} = md_L{layout.index}_wmi * {layout.wtm} + md_g")
            self.emit(
                f"{layout.coll} = md_L{layout.index}_wni * {layout.wtn} + 2 * md_c"
            )
        self.define_axis_bases()
        if self.smem_bytes:
            self.emit(
                f"md_smem = cute.arch.alloc_smem(cutlass.Uint8, {self.smem_bytes}, alignment=128)"
            )
            self.emit("md_s0 = cutlass.Int32(md_smem.toint())")
            for tile in (
                *self.tiles.values(),
                *self.derived_tiles.values(),
                *self.staging.values(),
            ):
                for each in (tile, tile.raw):
                    if each is not None and each.name not in self._defined_tiles:
                        self._defined_tiles.add(each.name)
                        self.emit(f"{each.name} = md_s0 + {each.offset}")
        self.define_bases()
        self.emit_carry_init()
        # Epilogue operand tiles land during the loop: one group, older than
        # every ring group, so any ring wait completes it as well.
        epilogue_async = list(
            plan.unique_operands(
                "epilogue", (ROUTE_DIRECT, ROUTE_INPLACE, ROUTE_CONVERT)
            )
        )
        self.define_iteration_bases(in_loop=False)
        if epilogue_async:
            for dot, operand in epilogue_async:
                self.emit_issue_operand(dot, operand, self.loop_var, "0")
            self.emit("cute.arch.cp_async_commit_group()")
        begin, end, trip_static = self.loop_bounds()
        ring = self.ring_items()
        if ring and plan.stages > 1:
            # Prologue of the operand ring: the first ``stages - 1``
            # iterations' tiles, one commit group each (an iteration past the
            # end commits an empty group so the wait depth stays uniform).
            for ahead in range(plan.stages - 1):
                value = f"({begin} + {ahead})" if ahead else begin
                if trip_static is not None and ahead < trip_static:
                    self.emit_issue(ring, value, str(ahead))
                elif trip_static is None:
                    with self.nest(f"if {value} < {end}:"):
                        self.emit_issue(ring, value, str(ahead))
                self.emit("cute.arch.cp_async_commit_group()")
        with self.nest(f"for {self.loop_var} in cutlass.range({begin}, {end}):"):
            self.emit_iteration(begin, end, ring)
        self.emit_epilogue(epilogue_async)
        return "\n".join(self.lines) + "\n"

    def loop_bounds(self) -> tuple[str, str, int | None]:
        """``(begin, end, static trip count)`` of the emitted loop: the grid
        loop's own bounds, or the tile loop's iteration count."""
        program = self.program
        if program.loop_axis is None:
            begin, end = self.bound(program.begin), self.bound(program.end)
            static = (
                program.end - program.begin
                if isinstance(program.begin, int) and isinstance(program.end, int)
                else None
            )
            return begin, end, static
        trip = program.loop_axis.extent // self.plan.size(program.loop_axis)
        return "0", str(trip), trip

    def emit_iteration(
        self, begin: str, end: str, ring: list[tuple[_Tile, Access, int]]
    ) -> None:
        plan, program = self.plan, self.program
        loop = self.loop_var
        stages = plan.stages
        if program.loop_axis is not None:
            step = plan.size(program.loop_axis)
            self.emit(f"{self.axis_base(program.loop_axis)} = {loop} * {step}")
        if ring and stages > 1:
            offset = f"{loop}" if begin == "0" else f"({loop} - {begin})"
            self.emit(f"md_stage = {offset} % {stages}")
        else:
            self.emit("md_stage = 0")
        if plan.staged_stores or "md_parity" in self.tile_buffers.values():
            self.emit(f"md_parity = {loop} % 2")
        self.define_iteration_bases(in_loop=True)
        self.evaluator = _Evaluator(self)
        if ring and stages == 1:
            self.emit_issue(ring, loop, "0")
            self.emit("cute.arch.cp_async_commit_group()")
        # Work that needs no shared data overlaps the copies in flight.
        for dot, operand in plan.unique_operands("loop", (ROUTE_SYNC,)):
            self.emit_sync_pass(dot, operand, "md_stage")
        for store in program.stores_in("loop"):
            if store.source == "carry_in":
                if store in plan.staged_stores:
                    self.emit_stage_in(store, "md_parity")
                else:
                    self.emit_store(store)
        # Accumulators that depend on nothing computed in the iteration
        # (zeros, the carry, a rescaled carry no earlier step reads).
        hoisted = {dot.node for dot in self.hoistable_accs()}
        for step in program.steps:
            if isinstance(step, Dot) and step.node in hoisted:
                self.emit_acc(step)
        # Exchange tiles of the carry at loop entry are written here, so the
        # iteration barrier below orders them before their ldmatrix reads.
        early = [
            (dot, operand)
            for dot, operand in plan.unique_operands("loop", (ROUTE_EXCHANGE,))
            if program.early_exchange(dot, operand)
        ]
        for dot, operand in early:
            self.emit_exchange(dot, operand)
        if ring:
            # The groups committed so far are the prologue's ``stages - 1``
            # plus one per earlier iteration, so this iteration's group is
            # followed by ``stages - 2`` younger ones (the iterations ahead;
            # the refill issued below comes after the wait): wait for all but
            # those and this iteration's tiles are in shared memory.
            self.emit(f"cute.arch.cp_async_wait_group({max(stages - 2, 0)})")
        for dot, operand in plan.unique_operands("loop", (ROUTE_INPLACE,)):
            self.emit_inplace_pass(dot, operand, "md_stage")
        for dot, operand in plan.unique_operands("loop", (ROUTE_CONVERT,)):
            self.emit_convert_pass(dot, operand, "md_stage")
        self.emit(self.sync)
        if ring and stages > 1:
            # Past the barrier every lane finished the previous iteration's
            # reads of the slot refilled here (``stages - 1`` iterations ahead
            # is the slot read one iteration ago); an iteration past the end
            # commits an empty group so the wait depth stays uniform.
            ahead = stages - 1
            with self.nest(f"if {loop} + {ahead} < {end}:"):
                self.emit_issue(
                    ring,
                    f"({loop} + {ahead})",
                    f"(md_stage + {ahead}) % {stages}",
                )
            self.emit("cute.arch.cp_async_commit_group()")
        # This iteration's snapshots (written into the staging buffer before
        # the barrier) stream out as 16-byte packets; the next iteration
        # writes the other buffer.
        for store in plan.staged_stores:
            self.emit_stage_out(store, "md_parity", loop)
        # The body in program order: a dot's accumulator (unless hoisted) and
        # the exchange tiles of fragments computed in this iteration (behind
        # their own barrier) right before it, derived values register by
        # register, stores at their program point.
        written: set[tuple[object, ...]] = {operand.chain_key for _d, operand in early}
        for step in program.steps:
            if isinstance(step, Dot):
                pending = [
                    operand
                    for operand in (step.a, step.b)
                    if plan.route(step, operand) == ROUTE_EXCHANGE
                    and operand.chain_key not in written
                ]
                for operand in pending:
                    written.add(operand.chain_key)
                    self.emit_exchange(step, operand)
                if pending:
                    self.emit(self.sync)
                if step.node not in hoisted:
                    self.emit_acc(step)
                self.emit_dot(step, "md_stage")
            elif isinstance(step, Derived):
                self.emit_derived(step)
            elif step.source == "body":
                self.emit_store(step)
        # The first dot flushed the snapshot stores behind its fragment loads.
        assert not self.deferred
        if not (ring and stages > 1):
            # Every lane finished reading the slot the next iteration refills.
            self.emit(self.sync)

    def hoistable_accs(self) -> list[Dot]:
        """The loop dots whose accumulator may be materialized at the
        iteration top: zeros, or a value of the carry at loop entry whose
        slot no earlier step reads (a carry used as an operand of an earlier
        dot must be read before its rescale overwrites the registers)."""
        program = self.program
        result = []
        for index, step in enumerate(program.steps):
            if not isinstance(step, Dot):
                continue
            if step.acc is None:
                result.append(step)
                continue
            if step.acc_frag not in program.carry_placeholders:
                continue
            slot = step.slot
            reads_slot = False
            for earlier in program.steps[:index]:
                if isinstance(earlier, Dot):
                    leaves = [o.frag for o in (earlier.a, earlier.b)] + [
                        earlier.acc_frag
                    ]
                elif isinstance(earlier, Derived):
                    leaves = list(earlier.frags)
                elif earlier.source == "carry_in":
                    # Snapshots are hoisted above every accumulator.
                    continue
                else:
                    leaves = [earlier.frag]
                if any(
                    leaf is not None and program.frag_slots[leaf] == slot
                    for leaf in leaves
                ):
                    reads_slot = True
                    break
            if not reads_slot:
                result.append(step)
        return result

    def emit_derived(self, step: Derived) -> None:
        """Materialize a derived fragment: evaluate its pointwise DAG per
        register over the input fragments' registers, with the 2-D tile
        inputs loaded as pairs at the fragment coordinates."""
        layout = self.slot_layouts[step.slot]
        self.hoist_scalars(step.node)
        updates = []
        for am in range(layout.atoms_m):
            for an in range(layout.atoms_n):
                for half in range(2):
                    row, col = layout.row(am, half), layout.col(an)
                    for access in step.loads:
                        tile = self.derived_tiles.get(access.node)
                        if tile is not None:
                            # The pair (col, col + 1) sits in 16-byte chunk
                            # col * es // 16 of the staged row (swizzled).
                            es = access.dtype.itemsize
                            chunk = f"(({col}) * {es} // 16)"
                            swz = _swizzled_chunk(row, chunk, tile.cpr)
                            lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                            unpack = (
                                "unpack_f16x2"
                                if access.dtype == torch.float16
                                else "unpack_bf16x2"
                            )
                            self.emit(
                                f"{lo}, {hi} = {_RUNTIME}.{unpack}({_RUNTIME}.lds_b32("
                                f"{tile.base('md_stage')} + {row} * {tile.row_bytes} + {swz} * 16 + ({col}) * {es} % 16))"
                            )
                        else:
                            positions = [
                                i
                                for i, axis in enumerate(access.dims)
                                if axis.kind != AXIS_LEAD
                            ]
                            coords = ["0"] * len(access.dims)
                            coords[positions[0]] = row
                            coords[positions[1]] = col
                            address = self.address(access, tuple(coords))
                            paired = self.pairs_aligned(access, 1)
                            lo, hi = self.pair_load(address, access.dtype, paired)
                        self.set_tile_leaf(access, 0, lo)
                        self.set_tile_leaf(access, 1, hi)
                    for bit in range(2):
                        j = 2 * half + bit
                        self.evaluator.reg = (am, an, j)
                        self.evaluator.element = bit
                        coords = self.frag_coords(step.node, layout, am, an, j)
                        updates.append(
                            (
                                self.reg(step.slot, am, an, j),
                                self.evaluator.define(step.node, coords),
                            )
                        )
                    self.evaluator.reg = None
                    self.evaluator.element = None
        for name, value in updates:
            self.emit(f"{name} = {value}")

    def emit_epilogue(self, epilogue_async: list[tuple[Dot, Operand]]) -> None:
        plan = self.plan
        dots = self.program.dots_in("epilogue")
        stores = self.program.stores_in("epilogue")
        if not dots and not stores:
            return
        self.evaluator = _Evaluator(self)
        self.define_iteration_bases(in_loop=False)
        needs_barrier = False
        if epilogue_async:
            self.emit("cute.arch.cp_async_wait_group(0)")
            needs_barrier = True
        for dot, operand in plan.unique_operands("epilogue", (ROUTE_SYNC,)):
            self.emit_sync_pass(dot, operand, "0")
            needs_barrier = True
        for dot, operand in plan.unique_operands("epilogue", (ROUTE_INPLACE,)):
            self.emit_inplace_pass(dot, operand, "0")
        for dot, operand in plan.unique_operands("epilogue", (ROUTE_CONVERT,)):
            self.emit_convert_pass(dot, operand, "0")
        # Exchanges of fragments the loop produced are written up front;
        # one of an epilogue dot's result waits for that dot and is written
        # (and fenced) right before the dot that reads it.
        produced = {dot.node for dot in dots}
        deferred: dict[Node, list[tuple[Dot, Operand]]] = {}
        for dot, operand in plan.unique_operands("epilogue", (ROUTE_EXCHANGE,)):
            assert operand.frag is not None
            if strip_new_vars(operand.frag) in produced:
                deferred.setdefault(dot.node, []).append((dot, operand))
                continue
            self.emit_exchange(dot, operand)
            needs_barrier = True
        if needs_barrier:
            self.emit(self.sync)
        by_node: dict[Node, Dot | Store] = {dot.node: dot for dot in dots}
        by_node.update({store.access.node: store for store in stores})
        for node in self.program.epilogue_order:
            statement = by_node[node]
            if isinstance(statement, Dot):
                if node in deferred:
                    for dot, operand in deferred[node]:
                        self.emit_exchange(dot, operand)
                    self.emit(self.sync)
                self.emit_acc(statement)
                self.emit_dot(statement, "0")
            else:
                self.emit_store(statement)


class _Nest:
    """Indented block: ``with emitter.nest("if x:"):``."""

    def __init__(self, emitter: _Emitter, header: str) -> None:
        self.emitter = emitter
        self.header = header

    def __enter__(self) -> None:
        self.emitter.emit(self.header)
        self.emitter.indent += "    "

    def __exit__(self, *exc_info: object) -> None:
        self.emitter.indent = self.emitter.indent[:-4]


def _staging_swizzle(row: str, chunk: str, chunks_per_row: int, elem_bytes: int) -> str:
    """Swizzled 16-byte chunk of the carry staging tile.

    fp32 fragment pairs are 8-byte lane writes: four lanes cover 32 bytes of
    one row, so XOR-ing the chunk by ``2 * (row % 8)`` keeps the sixteen
    lanes of each half-warp on distinct bank pairs; 16-bit pairs (16 bytes per
    four lanes) use the operand tiles' row swizzle.
    """
    if elem_bytes == 4 and chunks_per_row >= 2:
        return f"(({chunk}) ^ ((2 * ({row})) % {min(8, chunks_per_row)}))"
    return _swizzled_chunk(row, chunk, chunks_per_row)


def _scalar_nodes(program: object, value: Node) -> list[Node]:
    """The 0-d (uniform) loads and pointwise values under ``value``, in
    evaluation order (a load's tensor and subscript inputs are not values)."""
    from ...language import _tracing_ops
    from ...language import memory_ops

    order: list[Node] = []
    seen: set[Node] = set()
    frag_slots = cast("dict[Node, int]", program.frag_slots)  # pyrefly: ignore[missing-attribute]

    def visit(node: Node) -> None:
        if node in seen or node.op != "call_function" or node in frag_slots:
            return
        seen.add(node)
        if node.target is _tracing_ops._get_symnode:
            return
        if node.target is not memory_ops.load:
            for arg in node.all_input_nodes:
                visit(arg)
        if not _shape(node):
            order.append(node)

    visit(value)
    return order
