"""MPP matmuls whose computed operands are cooperative tensors in registers.

``metal_matmul="mpp"`` gives each matmul operand to ``matmul2d`` one of
three ways.  A device view is a tensor handle the op reads.  A computed
operand (a dequantized weight, a widened activation) is otherwise staged:
every K tile, the threads write it into a threadgroup tile, barrier, run
the op on the tile, barrier.  When the launch is one SIMD group, the op
runs at ``execution_simdgroup`` scope, the only scope whose inputs may be
cooperative tensors ("Input cooperative tensors require a single SIMD
group", MPPTensorOpsMatMul2dImpl.h), and staging buys nothing but
threadgroup traffic and two barriers per K tile.  So a computed operand
the lanes can recompute at any element (``OperandRecomputable``) is built
in registers instead (:func:`register_run_spec`): per step of the K tile,
each lane writes its elements of the op's left or right input cooperative
tensor, re-running the Scalar statements that compute each element with
the thread and lanes that hold it in the Scalar layout bound, as a
fragment element is computed (``operand_elements``), and the op runs on
them.  The accumulator and its element loops (stores, updates,
epilogues) are the MPP lowering's.  Tiles that take more SIMD groups keep
staging: they share each tile, which one decode serves.  Tiles of one SIMD
group can still run in several: a config's ``num_warps`` above 1 splits
the K loop among that many (:func:`register_split`), each over every
``num_warps``-th K tile from its own registers, the first adding the
others' accumulators after the loop.

Which elements a lane holds is the op's layout, probed on the GPU that runs
the kernel and factored into lane bases plus constants (``coop_layout``),
so each element's index arithmetic folds as a fragment element's does.

An activation widened from a 16-bit load (``x.to(float32)``) beside a
register operand is read as a device view of the 16-bit tensor when that
lets the op take fewer rows: with two cooperative inputs MPP takes M, N
and K of 16 or 32 only, one of them 32, while one cooperative input
leaves M a multiple of 8, so a decode of 8 rows or fewer isn't padded to
16.  A device view is read in K order, so beside one the register
operand's elements are too; between two register operands the K
positions are relabelled so each lane's elements of a row are adjacent
(``coop_layout.k_order``), which a view is read as when the GPU's layout
already keeps them adjacent.

Under ``fast_math``, where the GPU's matrix units run half inputs over
twice as fast as float (``coop_layout.fast_half_inputs``, by GPU family),
a dequantized weight ``codes * scale + bias`` whose scale and bias are
constant over each K tile is split (:func:`_affine_parts`,
:func:`affine_split_enabled`): x times the codes, exact in half, and each
tile's scale and bias applied in float to its products and to x's row
sums.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language.creation_ops import full
from ...language.memory_ops import load
from ...language.tile_ops import tile_index
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ._constants import MAX_THREADGROUP_MEMORY_BYTES
from ._constants import MAX_THREADS_PER_THREADGROUP
from ._constants import SIMD_WIDTH
from .affine_fold import Fact
from .coop_layout import INVALID
from .coop_layout import Atom
from .coop_layout import base_fact
from .coop_layout import factor
from .coop_layout import fast_half_inputs
from .coop_layout import k_order
from .coop_layout import lane_bits
from .coop_layout import lane_expr
from .coop_layout import layout
from .coop_layout import op_position
from .lane_fold import MSL_SCALAR_BYTES
from .lane_fold import round_bf16_casts
from .mpp_matmul import MAX_OP_TILE
from .mpp_matmul import PUT
from .mpp_matmul import REG_A
from .mpp_matmul import REG_B
from .mpp_matmul import REG_EPILOGUE
from .mpp_matmul import REG_RUN
from .mpp_matmul import RUN
from .mpp_matmul import WIDENING
from .mpp_matmul import Graphs
from .mpp_matmul import MppGroup
from .mpp_matmul import MppPlan
from .mpp_matmul import block_id_of
from .mpp_matmul import call_name
from .mpp_matmul import current_plan
from .mpp_matmul import group_sides
from .mpp_matmul import matmul_operands
from .mpp_matmul import names_read
from .mpp_matmul import op_dims
from .mpp_matmul import static_tile
from .mpp_matmul import storage_key
from .mpp_matmul import transposed_view
from .mpp_matmul import view_args
from .mpp_matmul import view_span
from .mpp_matmul import virtual_thread
from .operand_elements import SPLIT_BUFFER
from .operand_elements import OperandRecomputable
from .operand_elements import clamp_past_end
from .operand_elements import element_body
from .operand_elements import emit_split_adds
from .operand_elements import keyword
from .operand_elements import offset_fact
from .operand_elements import parse_expr
from .operand_elements import split_refusal
from .operand_elements import src
from .operand_elements import static_int
from .operand_elements import static_loop
from .operand_elements import tile_mask
from .operand_elements import tiles_in_range

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from torch.fx.node import Node

    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .coop_layout import Coords
    from .mpp_matmul import View
    from .msl_ast_walker import EmitState

#: The lane's index in its SIMD group, a kernel parameter of a kernel with
#: register operands (``metal_jit``).
LANE = "_mpp_lane"
#: Most steps in a K tile: they are unrolled, each element's statements
#: folded, so the source grows with them.  A bound on the code, not a
#: hardware limit: a K tile past it is staged.
MAX_STEPS = 16
#: Ops the affine split looks through to the rhs's value: they move or mask
#: its elements.
_SPLIT_MOVES = (
    torch.ops.aten.permute.default,
    torch.ops.aten.alias.default,
    _tracing_ops._mask_to,
)
#: The names an affine split's epilogue element reads: its accumulator
#: element, its tile's product element and x's row sum at its row.
_SPLIT_VALUES = ("_mr_av", "_mr_pv", "_mr_xv")


@dataclasses.dataclass(frozen=True)
class RegisterRun:
    """How a group's op takes its operands in registers: the op, each
    side's kind (``"reg"``, a cooperative tensor built in registers, or
    ``"view"``, a device view), its K tile's steps, whether the K positions
    are relabelled across steps (two register operands), and the affine
    split's FX values ``(codes, scale, bias)`` when the op takes it."""

    atom: Atom
    sides: tuple[str, str]
    steps: int
    interleave: bool
    affine: tuple[Node, Node, Node] | None


def _narrow_dtype(node: Node, dtype: torch.dtype) -> torch.dtype:
    """The 16-bit float type a value of ``dtype`` holds exactly, a tensor's
    of that type or one widened from it (``x.to(float32)``); else
    ``dtype``."""
    narrow = (torch.float16, torch.bfloat16)
    while dtype not in narrow and node.target in WIDENING:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            break
        val = source.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.dtype.itemsize >= dtype.itemsize:
            break
        node, dtype = source, val.dtype
    return dtype


def _coop_type(group: MppGroup, side: int) -> str:
    """The MSL element type of a register operand's cooperative tensor: half
    or bfloat where its values are exactly one, else float.  Each element
    holds the operand's value unrounded either way."""
    operand = (group.lhs, group.rhs)[side]
    node = matmul_operands(group.matmul)[1 + side]
    assert node is not None
    dtype = _narrow_dtype(node, operand.dtype)
    if dtype not in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return CompileEnvironment.current().backend.dtype_str(dtype)


def _exact_codes(node: object) -> bool:
    """Whether ``node`` is a float conversion of integers each exactly
    representable in half: an 8-bit integer, or an integer masked to at
    most 11 bits (``q & 0x1F``; half holds the integers up to 2048
    exactly)."""
    if not isinstance(node, torch.fx.Node) or node.target not in WIDENING:
        return False
    source = node.args[0]
    if not isinstance(source, torch.fx.Node):
        return False
    val = source.meta.get("val")
    if not isinstance(val, torch.Tensor) or val.dtype.is_floating_point:
        return False
    if val.dtype in (torch.int8, torch.uint8):
        return True
    mask = (
        source.args[1] if source.target is torch.ops.aten.bitwise_and.Scalar else None
    )
    return isinstance(mask, int) and 0 <= mask < 2048


def _tile_constant(
    node: object, k_block: int, k_tile: int, memo: dict[object, bool]
) -> bool:
    """Whether ``node``'s value is the same at every K position of each K
    tile: it reads the K block's positions only through ``k // g`` for a
    constant ``g`` the K tile (``k_tile``, from a begin it divides) divides,
    as a quantization group's scale index does.  Conservative: a value
    without inputs whose shape spans the K block (an ``hl.arange``) is
    taken to vary."""
    if isinstance(node, (list, tuple)):
        return all(_tile_constant(a, k_block, k_tile, memo) for a in node)
    if isinstance(node, torch.SymInt):
        return block_id_of(node) != k_block
    if not isinstance(node, torch.fx.Node):
        return True
    if node in memo:
        return memo[node]
    val = node.meta.get("val")
    if node.target is tile_index:
        assert isinstance(val, torch.Tensor)
        result = block_id_of(val.size(0)) != k_block
    elif (
        node.target is torch.ops.aten.div.Tensor_mode
        and node.kwargs.get("rounding_mode") == "floor"
        and isinstance(source := node.args[0], torch.fx.Node)
        and source.target is tile_index
        and isinstance(group_size := node.args[1], int)
        and group_size > 0
        and group_size % k_tile == 0
    ):
        result = True
    elif not node.all_input_nodes:
        sizes = val.size() if isinstance(val, torch.Tensor) else ()
        result = node.target is full or all(block_id_of(d) != k_block for d in sizes)
        if isinstance(val, torch.SymInt):
            result = block_id_of(val) != k_block
    else:
        result = all(
            _tile_constant(a, k_block, k_tile, memo)
            for a in (*node.args, *node.kwargs.values())
        )
    memo[node] = result
    return result


def _k_loop_tile(cg: GenerateAST, group: MppGroup) -> tuple[int, int] | None:
    """The begin and the K tile of the group's K loop when codegen knows
    them, inside the loop (the run) or before it (the accumulator)."""
    loop = static_loop(cg, group.k_block)
    if loop is not None:
        return loop[1], loop[2]
    if not group.levels:
        return None
    begins = group.levels[0].owner.args[1]
    step = cg.device_function.resolved_block_size(group.k_block)
    if (
        not isinstance(begins, list)
        or len(begins) != 1
        or not isinstance(begins[0], int)
        or not isinstance(step, int)
    ):
        return None
    return begins[0], step


def _affine_parts(cg: GenerateAST, group: MppGroup) -> tuple[Node, Node, Node] | None:
    """The FX values ``(codes, scale, bias)`` of a rhs ``codes * scale +
    bias`` (a dequantized weight, through element moves) whose codes are
    exact in half (:func:`_exact_codes`) and whose scale and bias are the
    same over each K tile (:func:`_tile_constant`) and fp16 values, with the
    product and the sum taken in float32, or None.  Over such a tile
    ``sum_k x (q s + b) = s sum_k x q + b sum_k x``, the reassociation MLX's
    quantized kernels use.  Its error grows with the cancellation between
    the two terms, to about eps * (|s| sum |x q| + |b| sum |x|) where the
    unsplit run's is eps * sum |x w|, since P and X round apart.  With x in
    half too, codes of at most 11 bits and at most 512 K positions a tile
    (:data:`MAX_STEPS` steps of at most 32), ``|s P|`` and ``|b X|`` stay
    below 2**52, so neither overflows float where the weights don't; a
    float32 or bf16 scale or bias, whose range is float's, isn't split, nor
    a weight rounded on the way (a product or sum in a narrower type)."""
    tile = _k_loop_tile(cg, group)
    if tile is None or tile[0] % tile[1]:
        return None
    node = matmul_operands(group.matmul)[2]
    while node.target in _SPLIT_MOVES:
        source = node.args[0]
        assert isinstance(source, torch.fx.Node)
        node = source
    if (
        node.target is not torch.ops.aten.add.Tensor
        or node.kwargs.get("alpha", 1) != 1
        or not _float32(node)
    ):
        return None
    memo: dict[object, bool] = {}
    for product, bias in itertools.permutations(node.args[:2]):
        if not (
            isinstance(product, torch.fx.Node)
            and product.target is torch.ops.aten.mul.Tensor
            and _float32(product)
            and isinstance(bias, torch.fx.Node)
        ):
            continue
        for codes, scale in itertools.permutations(product.args[:2]):
            if (
                _exact_codes(codes)
                and isinstance(scale, torch.fx.Node)
                and _fp16_values(scale)
                and _fp16_values(bias)
                and _tile_constant(scale, group.k_block, tile[1], memo)
                and _tile_constant(bias, group.k_block, tile[1], memo)
            ):
                assert isinstance(codes, torch.fx.Node)
                return codes, scale, bias
    return None


def _float32(node: Node) -> bool:
    """Whether ``node``'s value is a float32 tensor."""
    val = node.meta.get("val")
    return isinstance(val, torch.Tensor) and val.dtype == torch.float32


def _fp16_values(node: Node) -> bool:
    """Whether ``node``'s values are fp16 ones: an fp16 tensor's, or one
    widened from it."""
    val = node.meta.get("val")
    return (
        isinstance(val, torch.Tensor)
        and _narrow_dtype(node, val.dtype) == torch.float16
    )


def _on_gpu() -> bool:
    """Whether the kernel runs on the GPU here: its layouts are then probed,
    and its affine split decided, there."""
    return CompileEnvironment.current().device.type == "mps"


def affine_split_enabled() -> bool:
    """Whether register runs split dequantized weights: under ``fast_math``
    only, since the split reassociates the sum (its error grows where the
    scale's and the bias's terms cancel), and where half inputs pay
    (:func:`_half_pays`)."""
    return CompileEnvironment.current().settings.fast_math and _half_pays()


def _half_pays() -> bool:
    """Whether the GPU that runs the kernel runs ``matmul2d`` on half inputs
    over twice as fast as on float, as two half ops must to replace one
    half-by-float op: by its family (``coop_layout.fast_half_inputs``),
    which an M4 isn't."""
    return _on_gpu() and fast_half_inputs()


def register_run_spec(cg: GenerateAST, group: MppGroup) -> RegisterRun | None:
    """How the group's op takes its operands in registers, or None for the
    MPP lowering's views and staged tiles: when the launch is one SIMD
    group, an operand is a device view or a computed operand the lanes can
    recompute at any element (``registers``, set by
    :func:`check_register_operands`), at least one is the latter, and the
    op fits ``matmul2d`` with cooperative inputs.

    The op covers the tile: with two cooperative inputs its M, N and K are
    16 or 32 and one of them 32; with one, M and N are the MPP op's
    (``op_dims``) and K 16.  A widened activation beside a register operand
    is a device view when the two-input op would pad the tile's rows or
    columns (a decode of 8 rows or fewer) or the GPU's layout keeps each
    lane's K positions of a row adjacent in order (the M5's); else it is
    built in registers too, in the relabelled K order.  On an M4 at 8 rows
    the two tie: 8-row ops reading x as a view ran 12-15% faster than
    16-row ops with x in registers on the o, gate_up and lm_head
    projections and 8-20% slower on q/og and down, whose long K loops read
    a word of packed codes per element of a row in K order."""
    plan = current_plan()
    if math.prod(cg.launch_thread_axis_sizes().values()) != SIMD_WIDTH:
        assert plan.split == 1
        return None
    tile = static_tile(cg, (group.m_block, group.n_block, group.k_block))
    chosen = _register_op(group, plan, tile[0], tile[1], tile[2])
    if chosen is None:
        # The plan's split took every group's op (:func:`register_split`).
        assert plan.split == 1, group.gid
        return None
    atom, sides = chosen
    affine = None
    if _affine_candidate(atom, sides) and affine_split_enabled():
        parts = _affine_parts(cg, group)
        if parts is not None:
            affine = parts
            atom = dataclasses.replace(atom, right="half")
    return RegisterRun(atom, sides, tile[2] // atom.k, sides == ("reg", "reg"), affine)


def _affine_candidate(atom: Atom, sides: tuple[str, str]) -> bool:
    """Whether the op could take the affine split (:func:`_affine_parts`),
    which runs it on half inputs: only a float right input in registers
    beside a half left one gains (two half ops for one half-by-float op)."""
    return sides[1] == "reg" and atom.left == "half" and atom.right == "float"


def _register_op(
    group: MppGroup, plan: MppPlan, block_m: int, block_n: int, k: int
) -> tuple[Atom, tuple[str, str]] | None:
    """The op and each side's kind (:class:`RegisterRun`) of the group's
    register run over a ``block_m`` x ``block_n`` tile and a K tile of ``k``
    on one SIMD group, or None (:func:`register_run_spec`)."""
    if plan.registers is None:
        return None
    if (
        group.fold is not None
        or group.product is not None
        or group.batch_block is not None
        or group.k_factor != 1
        or not group.lhs.dtype.is_floating_point
    ):
        return None
    kinds: list[str] = []
    for operand, tile in ((group.lhs, (block_m, k)), (group.rhs, (k, block_n))):
        if operand.view is not None:
            if not _view_fits(operand.view, tile):
                # Staged, the view is refused as before (_codegen_matmul).
                return None
            kinds.append("view")
        elif operand.registers:
            widened = operand.widened is not None and _view_fits(operand.widened, tile)
            kinds.append("widened" if widened else "reg")
        else:
            return None
    if "reg" not in kinds and "widened" not in kinds:
        return None
    m, n = op_dims(group, block_m, block_n)
    backend = CompileEnvironment.current().backend

    def atom_of(sides: tuple[str, str]) -> Atom | None:
        types: list[str] = []
        flags: list[bool] = []
        for side, (operand, kind) in enumerate(
            zip((group.lhs, group.rhs), sides, strict=True)
        ):
            if kind == "view":
                view = operand.view if operand.view is not None else operand.widened
                assert view is not None
                types.append(backend.dtype_str(view.tensor.dtype))
                flags.append(transposed_view(view))
            else:
                types.append(_coop_type(group, side))
                flags.append(side == 1)
        if types[0] != types[1] and "float" not in types:
            # Two 16-bit types: the register sides take float.
            types = [
                t if kind == "view" else "float"
                for t, kind in zip(types, sides, strict=True)
            ]
        if sides == ("reg", "reg"):
            am, an = (16 if s <= 16 else 32 if s <= 32 else 0 for s in (m, n))
            if not am or not an or am > block_m or an > block_n:
                return None
            ak = 32 if (am, an) == (16, 16) else 16
        else:
            am, an, ak = m, n, 16
            if am % 8 or an % 8 or (am % 16 and an % 16):
                return None
            if am > MAX_OP_TILE[0] or an > MAX_OP_TILE[1]:
                return None
        if k % ak or k // ak > MAX_STEPS:
            return None
        return Atom(
            am,
            an,
            ak,
            types[0],
            types[1],
            "float",
            sides[0] == "view",
            sides[1] == "view",
            flags[0],
            flags[1],
        )

    sides = cast("tuple[str, str]", tuple(kinds))
    built = sides
    if "widened" in kinds:
        # A widened operand read as a view, the rhs taking registers if both
        # are; or built in registers.
        read = cast(
            "tuple[str, str]",
            tuple("view" if kind == "widened" else kind for kind in kinds),
        )
        if read == ("view", "view"):
            read = ("view", "reg") if kinds[1] == "widened" else ("reg", "view")
        built = cast(
            "tuple[str, str]",
            tuple("reg" if kind == "widened" else kind for kind in kinds),
        )
        sides = read
        if built == ("reg", "reg") and read != built:
            both = atom_of(built)
            if (
                both is not None
                and (both.m, both.n) == (m, n)
                and k_order(layout(both, _on_gpu()), both.k) != tuple(range(both.k))
            ):
                sides = built
    atom = atom_of(sides)
    if atom is None and sides != built:
        # The view's op doesn't fit (too many steps of its K): the op with
        # both operands in registers may.
        sides = built
        atom = atom_of(sides)
    return None if atom is None else (atom, sides)


def register_split(
    config: Config, plan: MppPlan, graphs: list[GraphInfo], threads: int
) -> int:
    """The SIMD groups that split the K loops of an MPP config's register
    runs, as a fragment config's do: its ``num_warps`` when its tiles take
    one SIMD group (``threads``), every group's op takes register operands
    there, the kernel admits a split (``operand_elements.split_refusal``)
    and the combine's threadgroup memory fits; else 1, the config's meaning
    before, where ``num_warps`` doesn't change a one-SIMD-group launch.
    Each SIMD group runs the tiles over every ``num_warps``-th K tile from
    its own register operands, and after the loop the first adds the
    others' accumulators to its own (:func:`emit_register_combine`) and
    alone writes memory."""
    split = int(cast("int", config.get("num_warps", 1)))
    if (
        split <= 1
        or threads != SIMD_WIDTH
        or split * SIMD_WIDTH > MAX_THREADS_PER_THREADGROUP
        or not plan.by_matmul
        or split_refusal(plan, graphs) is not None
    ):
        return 1
    env = CompileEnvironment.current()
    # The combine's threadgroup memory per K loop, which loops after one
    # another reuse (``split_slots``): each loop's accumulators' elements.
    elements: dict[object, int] = {}
    for group in plan.by_matmul.values():
        tile = [
            env.block_sizes[b].from_config(config)
            for b in (group.m_block, group.n_block, group.k_block)
        ]
        if not all(isinstance(t, int) for t in tile):
            return 1
        if (
            group.mats
            or group.stage_uses
            or any(g is group for g in plan.by_vec.values())
        ):
            # Its result read in the Scalar layout, or a row or column
            # vector its element loop reads, goes through threadgroup
            # memory, which every SIMD group would write.
            return 1
        chosen = _register_op(group, plan, *cast("list[int]", tile))
        if chosen is None:
            return 1
        # The affine split may run the op on half inputs instead
        # (register_run_spec), whose destination layout may differ.
        atoms = [chosen[0]]
        if _affine_candidate(*chosen) and affine_split_enabled():
            atoms.append(dataclasses.replace(chosen[0], right="half"))
        slots = max(_dest_slots(atom, _on_gpu()) for atom in atoms)
        # split_refusal refused groups outside a loop.
        loop = group.levels[0].owner
        elements[loop] = elements.get(loop, 0) + slots * SIMD_WIDTH
    most = max(elements.values())
    if (split - 1) * most * MSL_SCALAR_BYTES["float"] > MAX_THREADGROUP_MEMORY_BYTES:
        return 1
    return split


def _view_fits(view: View, tile: tuple[int, int]) -> bool:
    """Whether a device view's handles over an operand's ``tile`` stay
    within the int32 offsets MPP tensor handles index with, as the staged
    path requires of a view (``mpp_matmul._codegen_matmul``): each K
    step's handle covers part of the tile."""
    return view_span(view, tile) < 2**31


def _dest_slots(atom: Atom, on_gpu: bool) -> int:
    """The destination elements each lane holds of ``atom``'s op, read
    from its layout (``coop_layout.layout``)."""
    return len(layout(atom, on_gpu).dest[0])


def check_register_operands(group: MppGroup, plan: MppPlan, graphs: Graphs) -> None:
    """Mark the group's computed operands a lane can recompute at any
    element, over its M or N block and K (``OperandRecomputable``), from
    no load of a tensor the kernel writes (which a lane would re-run at the
    op, after a write the Scalar load preceded): staged at the op from the
    Scalar layout, not from another group's elements nor hoisted.  An
    operand widened from a view of a tensor the kernel writes isn't marked,
    so it stays staged: the op would read the view at the op."""
    _, lhs, rhs = matmul_operands(group.matmul)
    for operand, node, block in (
        (group.lhs, lhs, group.m_block),
        (group.rhs, rhs, group.n_block),
    ):
        if (
            not operand.staged
            or operand.elements is not None
            or operand.hoist is not None
            or group.k_factor != 1
        ):
            continue
        recomputable = OperandRecomputable(group, graphs, {block, group.k_block})
        if not recomputable(node):
            continue
        loads = [ld for ld in recomputable.nodes(node) if ld.target is load]
        if any(storage_key(ld) in plan.written for ld in loads):
            continue
        operand.registers = True


# Codegen.


def register_acc_keywords(spec: RegisterRun) -> str:
    """The ``registers`` keyword of a register group's ``_helion_mpp_acc``:
    its op (:class:`~coop_layout.Atom`'s fields)."""
    return f", registers={dataclasses.astuple(spec.atom)!r}"


def codegen_register_run(
    cg: GenerateAST,
    group: MppGroup,
    acc: ast.Name,
    values: tuple[object, object],
    env: Mapping[Node, object],
    spec: RegisterRun,
) -> ast.AST:
    """``_helion_mpp_run(acc, gid, lhs..., rhs..., registers=[lhs, rhs],
    ...)`` of a register group: a register operand as the name of its
    value, cast to the operand's dtype, with ``("reg", bindings, past)``,
    the bindings making a lane act as the thread holding the element at
    ``(_e_m or _e_n, _e_k)`` and ``past`` the M or N tile's mask, index and
    offset when the mask is live (``tile_mask``), for its loads past the
    tile's loop end; a device view as its nine run arguments
    (``view_args``) with ``("view", live)``, ``live`` whether each dim is
    live by construction.  ``steps`` is the K tile's steps, ``interleave``
    whether they relabel K, ``k_live`` the tile's live length, which masks
    a register operand's K tail and cuts a view's last steps (None when
    every tile is whole), ``facts`` the tile offsets' ``(offset, multiple,
    least)``, ``k_split`` the K loop's offset and the SIMD groups that split
    the loop (:func:`register_split`), None for one, and ``affine`` the
    split's codes, scale and bias names."""
    backend = CompileEnvironment.current().backend
    k = static_tile(cg, (group.k_block,))[0]
    parts: list[str] = []
    anchors: list[tuple[int, str | None]] = []
    outers: list[str] = []
    sides: list[str] = []
    for (operand, blocks, k_pos), value, e, kind in zip(
        group_sides(group), values, ("_e_m", "_e_n"), spec.sides, strict=True
    ):
        if kind == "reg":
            assert isinstance(value, ast.AST)
            name = cg.lift(backend.cast_ast(value, operand.dtype), prefix="_mpp_rv")
            binds = virtual_thread(
                cg, ((blocks[1 - k_pos], e), (group.k_block, "_e_k"))
            )
            past = tile_mask(cg, blocks[1 - k_pos])
            parts.append(name.id)
            outers.append("[]")
            sides.append(f"('reg', [{', '.join(binds)}], {past!r})")
            continue
        view = operand.view if operand.view is not None else operand.widened
        assert view is not None
        op_parts, op_anchors, live, view_outer = view_args(
            cg, view, group.k_block, value
        )
        anchors += [(2 + len(parts) + i, a) for i, a in op_anchors]
        parts += op_parts
        outers.append(
            "["
            + "".join(f"({x}, {s}, {st}, {a}), " for x, s, st, a in view_outer)
            + "]"
        )
        sides.append(f"('view', {[bool(f) for f in live]!r})")
    info = cg.active_device_loops[group.k_block][-1].block_id_to_info[group.k_block]
    k_live = "None"
    if not tiles_in_range(cg, group.k_block, k, static_int(info.end_expr)):
        k_live = f"{info.end_var_name} - {cg.offset_var(group.k_block)}"
    facts = [
        fact
        for block in (group.m_block, group.n_block, group.k_block)
        if (fact := offset_fact(cg, block)) is not None
    ]
    affine = ""
    if spec.affine is not None:
        names = [
            cg.lift(
                backend.cast_ast(cast("ast.AST", env[node]), torch.float32), prefix=p
            )
            for node, p in zip(
                spec.affine, ("_mpp_rq", "_mpp_rs", "_mpp_rb"), strict=True
            )
        ]
        affine = f", affine=({', '.join(name.id for name in names)})"
    split = current_plan().split
    k_split = None if split == 1 else (cg.offset_var(group.k_block), split)
    anchor_list = ", ".join(f"({p}, {a})" for p, a in anchors)
    return expr_from_string(
        f"{RUN}({acc.id}, {group.gid}, {', '.join(parts)}, "
        f"anchors=[{anchor_list}], outer=[{', '.join(outers)}], "
        f"registers=[{', '.join(sides)}], steps={spec.steps}, "
        f"interleave={spec.interleave}, k_live={k_live}, facts={facts!r}, "
        f"k_split={k_split!r}{affine})"
    )


def run_binds(call: ast.Call) -> list[list[ast.Tuple]]:
    """A ``_helion_mpp_run``'s bindings of each register operand; ``[]`` for
    a run without register operands."""
    sides = keyword(call, "registers")
    if sides is None:
        return []
    assert isinstance(sides, ast.List)
    out: list[list[ast.Tuple]] = []
    for side in sides.elts:
        assert isinstance(side, ast.Tuple)
        kind = ast.literal_eval(side.elts[0])
        if kind == "reg":
            binds = side.elts[1]
            assert isinstance(binds, ast.List)
            out.append(cast("list[ast.Tuple]", binds.elts))
    return out


# MSL assembly: elementize.


def register_run(
    call: ast.Call,
    acc_call: ast.Call,
    rerun: Callable[[Mapping[str, ast.expr], set[str], list[ast.AST]], list[ast.stmt]],
) -> ast.For:
    """The loop that replaces a register run ``call`` (after
    ``_Copies.rewrite_run``): over the K tile's steps
    (:func:`emit_register_run`), the loop over the elements each lane holds
    of each register operand's cooperative tensor, writing its value at the
    element: its value's name with the Scalar statements it reads re-run by
    ``rerun(binds, reads, checked)`` under its bindings, their loads past
    the M or N tile's loop end reading the tile's first position
    (``clamp_past_end``), 0 past the K tile's live length.  Under an affine
    split the rhs's elements are its codes, and a loop over the
    destination's elements applies the tile's scale and bias
    (:data:`_SPLIT_VALUES`)."""
    sides = keyword(call, "registers")
    k_live = keyword(call, "k_live")
    affine = keyword(call, "affine")
    assert isinstance(sides, ast.List)
    live = not (isinstance(k_live, ast.Constant) and k_live.value is None)
    pos = 2
    loops: list[ast.stmt] = []
    rhs: tuple[dict[str, ast.expr], object] | None = None

    def element_loop(marker: str, stmts: list[ast.stmt]) -> ast.For:
        return ast.For(
            ast.Name("_it", ast.Store()),
            ast.Call(ast.Name(marker, ast.Load()), [], []),
            stmts,
            [],
        )

    # The run's call keeps the views' arguments; a register operand's value
    # lives in the element loops alone, so the cooperative statement reads
    # no per-lane name.
    args = list(call.args)
    for i, side in enumerate(sides.elts):
        assert isinstance(side, ast.Tuple)
        if ast.literal_eval(side.elts[0]) == "view":
            pos += 9
            continue
        binds_list = side.elts[1]
        assert isinstance(binds_list, ast.List)
        binds = {
            cast("str", ast.literal_eval(cast("ast.Tuple", b).elts[0])): cast(
                "ast.Tuple", b
            ).elts[1]
            for b in binds_list.elts
        }
        past = ast.literal_eval(side.elts[2])
        source = call.args[pos]
        args[pos] = ast.Constant(None)
        pos += 1
        if affine is not None and i == 1:
            assert isinstance(affine, ast.Tuple)
            source = affine.elts[0]
            rhs = (binds, past)
        value = parse_expr(f"static_cast(float, {src(source)})")
        if live:
            assert k_live is not None
            value = parse_expr(f"({src(value)} if _e_k < {src(k_live)} else 0.0)")
        put = ast.Expr(ast.Call(ast.Name(PUT, ast.Load()), [value], []))
        stmts = rerun(binds, names_read(put), [value])
        if past is not None:
            mask, index, offset = past
            position = "_e_m" if i == 0 else "_e_n"
            stmts = clamp_past_end(stmts, mask, index, offset, binds, position)
        # Conversions to bfloat round as lane copies' do (BF16_CAST).
        loops.append(
            element_loop(REG_A if i == 0 else REG_B, round_bf16_casts([*stmts, put]))
        )
    assert pos == len(call.args)
    if affine is not None:
        assert isinstance(affine, ast.Tuple) and rhs is not None
        binds, past = rhs
        _, scale, bias = (src(e) for e in affine.elts)
        av, pv, xv = _SPLIT_VALUES
        # The tile's terms first: where they cancel, the accumulator keeps
        # its own precision.
        value = parse_expr(f"{av} + ({scale} * {pv} + {bias} * {xv})")
        put = ast.Expr(ast.Call(ast.Name(PUT, ast.Load()), [value], []))
        stmts = rerun(binds, names_read(put) - set(_SPLIT_VALUES), [value])
        if past is not None:
            mask, index, offset = cast("tuple[str, str, str]", past)
            stmts = clamp_past_end(stmts, mask, index, offset, binds, "_e_n")
        loops.append(element_loop(REG_EPILOGUE, round_bf16_casts([*stmts, put])))
    keywords = [
        ast.keyword(k.arg, ast.Constant(True)) if k.arg == "affine" else k
        for k in call.keywords
    ]
    loop = ast.For(
        ast.Name("_it", ast.Store()),
        ast.Call(ast.Name(REG_RUN, ast.Load()), args, keywords),
        loops,
        [],
    )
    ast.fix_missing_locations(loop)
    return loop


# MSL emission (msl_ast_walker).


def register_atom(call: ast.Call) -> Atom | None:
    """The op of a ``_helion_mpp_acc`` that declares a register group's
    accumulator (:func:`register_acc_keywords`), else None."""
    spec = keyword(call, "registers")
    return None if spec is None else Atom(*ast.literal_eval(spec))


#: One coordinate of an element a lane holds: its lane's base (a name the
#: kernel's preamble declares, None for 0) plus a constant.
Coord = tuple["str | None", int]


@dataclasses.dataclass(frozen=True)
class CoopLanes:
    """Where the elements a lane holds of one cooperative tensor lie
    (``coop_layout``): each one's row and column, and, where some element
    isn't valid on some lane, each one's validity, a name the preamble
    declares."""

    rows: tuple[Coord, ...]
    cols: tuple[Coord, ...]
    valid: tuple[str, ...] | None = None


def _lane_name(state: EmitState, base: tuple[int, ...]) -> str:
    """The preamble's name of a lane base, ``base[lane]``: the sum of the
    weights of the lane index's bits where it is one (``coop_layout``), else
    read from a table; and its fact."""
    name = state.lane_names.get(base)
    if name is not None:
        return name
    name = state.lane_names[base] = f"_mrl{len(state.lane_names)}"
    weights = lane_bits(base)
    if weights is not None:
        value = lane_expr(LANE, weights)
    else:
        table = state.tables.setdefault(
            ("short", tuple(map(str, base))), f"_mrl_table{len(state.tables)}"
        )
        value = f"{table}[{LANE}]"
    state.lane_decls.append(f"const short {name} = {value};")
    state.lane_facts[name] = base_fact(base)
    return name


def operand_lanes(
    state: EmitState, atom: Atom, operand: str, interleave: bool = False
) -> CoopLanes:
    """The elements a lane holds of ``atom``'s ``operand`` tensor: its
    layout probed on the GPU that runs the kernel (the M4's otherwise),
    each coordinate a lane base plus a constant where that is exact; else
    each element's coordinates, read from the op at run time into names the
    preamble declares (correct, but they fold over nothing but their
    range).  Under ``interleave`` an input's K positions are relabelled
    (``coop_layout.k_order``)."""
    key = (atom, operand, interleave)
    cached = state.lanes.get(key)
    if cached is not None:
        return cached
    probed = layout(atom, state.on_gpu)
    coords = probed.of(operand)
    if interleave and operand != "dest":
        coords = _relabelled(coords, k_order(probed, atom.k), operand)
    rows, cols = (factor(coords, d) for d in (0, 1))
    if rows is not None and cols is not None:
        out = []
        for f in (rows, cols):
            name = None if f.constant() else _lane_name(state, f.base)
            out.append(tuple((name, o) for o in f.offset))
        lanes = CoopLanes(out[0], out[1])
    else:
        lanes = _runtime_lanes(state, atom, operand, len(coords[0]))
    state.lanes[key] = lanes
    return lanes


def _relabelled(coords: Coords, order: tuple[int, ...], operand: str) -> Coords:
    """An input's ``coords`` with its K positions relabelled by ``order``
    (``coop_layout.k_order``): the left input's columns, the right's rows."""
    d = 1 if operand == "left" else 0
    return tuple(
        tuple(
            e if e == INVALID else (e[0], order[e[1]]) if d else (order[e[0]], e[1])
            for e in lane
        )
        for lane in coords
    )


def _relabels(state: EmitState, atom: Atom) -> bool:
    """Whether a run's two register inputs relabel their K positions
    (``coop_layout.k_order``): only when both inputs' relabelled
    coordinates factor, since an input whose coordinates are read at run
    time (:func:`_runtime_lanes`) keeps the op's K order, which the other
    must then keep too."""
    probed = layout(atom, state.on_gpu)
    order = k_order(probed, atom.k)
    for operand in ("left", "right"):
        coords = _relabelled(probed.of(operand), order, operand)
        if factor(coords, 0) is None or factor(coords, 1) is None:
            return False
    return True


def _runtime_lanes(state: EmitState, atom: Atom, operand: str, count: int) -> CoopLanes:
    """Each of the ``count`` elements' coordinates and validity read in the
    preamble from a cooperative tensor of ``atom``'s ``operand``."""
    j = len(state.lanes)
    op, tensor = f"_mrl_op{j}", f"_mrl_t{j}"
    extents = {
        "left": (atom.m, atom.k),
        "right": (atom.k, atom.n),
        "dest": (atom.m, atom.n),
    }[operand]
    # get_multidimensional_index lists the operand's innermost dimension
    # first: which of its indices each coordinate is.
    index_of = op_position(atom, operand, (0, 1))
    state.lane_decls += [
        f"{atom.op_type()} {op};",
        f"auto {tensor} = {atom.tensor(op, operand)};",
    ]
    rows: list[Coord] = []
    cols: list[Coord] = []
    valid: list[str] = []
    for e in range(count):
        names = []
        for d in (0, 1):
            name = f"_mrl{j}_{'rc'[d]}{e}"
            state.lane_decls.append(
                f"const short {name} = (short){tensor}.get_multidimensional_index("
                f"(ushort){e})[{index_of[d]}];"
            )
            state.lane_facts[name] = Fact(1, 0, extents[d] - 1)
            names.append(name)
        rows.append((names[0], 0))
        cols.append((names[1], 0))
        valid.append(f"_mrl{j}_v{e}")
        state.lane_decls.append(
            f"const bool {valid[-1]} = {tensor}.is_valid_element((ushort){e});"
        )
    return CoopLanes(tuple(rows), tuple(cols), tuple(valid))


def _at(const: int, coord: Coord, scale: int = 1) -> str:
    """``scale * (base + offset) + const`` of an element's coordinate."""
    base, offset = coord
    value = str(const + scale * offset)
    if base is None:
        return value
    return f"{value} + {base}" if scale == 1 else f"{value} + {scale} * {base}"


def emit_register_run(
    stmt: ast.For,
    parts: list[str],
    indent: int,
    state: EmitState,
    emit_body: Callable[[list[ast.stmt], list[str], int, EmitState], None],
    coop_view: Callable[..., tuple[list[str], list[str], str]],
) -> None:
    """A register run's loop (:func:`register_run`): per step of the K tile,
    each register operand's cooperative tensor, every lane writing its
    elements in a block with the step ``_fs`` a constant, each element's
    position bound and its statements folded over it
    (``operand_elements.element_body``), and each device view's handle of
    the step's K, then the op.  Each step's operands (``_mr_a1``,
    ``_mpp_a1``: suffixed by the step) of a run with one register operand
    are built before the previous step's op runs, so their reads overlap
    that op rather than wait for it; with two register operands or the
    affine split the steps keep their order.  Op K position ``kappa`` of step
    ``s`` is K position ``steps * order[kappa] + s`` of the tile between two
    register operands (``coop_layout.k_order``: a lane's elements of a row
    over the steps are then one run of adjacent K positions, a few words of
    packed codes), and ``s * K + kappa`` beside a view, which the op reads
    in order; a view's step past the tile's live K is skipped.

    Under an affine split the op's destination tensors start from 0 each
    tile and take x times the codes; a right input of ones gives x's row
    sums in a destination tensor of the same layout, whose element at each
    index lies in the same row; and each accumulator element adds the
    tile's scale times its product plus the bias times its row's sum."""
    pad = " " * indent
    inner = pad + "    "
    call = stmt.iter
    assert isinstance(call, ast.Call)
    acc = call.args[0]
    assert isinstance(acc, ast.Name)
    root = state.coop[acc.id]
    atom = state.register_atoms[root]
    gid = state.coop_group[root][0]
    op = f"_mpp_op_{gid}"
    sides_kw = keyword(call, "registers")
    outers = keyword(call, "outer")
    assert isinstance(sides_kw, ast.List) and isinstance(outers, ast.List)
    steps = cast("int", ast.literal_eval(cast("ast.expr", keyword(call, "steps"))))
    k_split = ast.literal_eval(cast("ast.expr", keyword(call, "k_split")))
    # Its K loop, which it lies in, splits it (``split_range``).
    assert k_split is None or k_split[0] in state.split_loops, k_split
    k_live = keyword(call, "k_live")
    # Every K tile whole: no view's step runs past the tile's live K.
    whole = isinstance(k_live, ast.Constant) and k_live.value is None
    interleave = bool(
        ast.literal_eval(cast("ast.expr", keyword(call, "interleave")))
    ) and _relabels(state, atom)
    affine = keyword(call, "affine") is not None
    facts = dict(state.lane_facts)
    for name, multiple, least in ast.literal_eval(
        cast("ast.expr", keyword(call, "facts"))
    ):
        facts[name] = Fact(multiple, least)
    loops = {
        call_name(loop.iter): loop for loop in stmt.body if isinstance(loop, ast.For)
    }
    # Each side: ("reg", lanes, element loop) or ("view", run args, live, outer).
    pos = 2
    sides: list[tuple[object, ...]] = []
    for i, side in enumerate(sides_kw.elts):
        assert isinstance(side, ast.Tuple)
        operand = ("left", "right")[i]
        if ast.literal_eval(side.elts[0]) == "reg":
            lanes = operand_lanes(state, atom, operand, interleave)
            facts.update(state.lane_facts)
            sides.append(("reg", lanes, loops[REG_A if i == 0 else REG_B]))
            pos += 1
        else:
            live = tuple(ast.literal_eval(side.elts[1]))
            outer = outers.elts[i]
            assert isinstance(outer, ast.List)
            sides.append(("view", call.args[pos : pos + 9], live, outer.elts))
            pos += 9
    saved = state.put, state.put_type
    dest = None
    if affine:
        dest = operand_lanes(state, atom, "dest")
        facts.update(state.lane_facts)
    if dest is not None:
        parts += [
            f"{pad}{{",
            *(
                line
                for tensor in ("_mr_p", "_mr_x")
                for line in (
                    f"{inner}auto {tensor} = {atom.tensor(op, 'dest')};",
                    f"{inner}#pragma unroll",
                    (
                        f"{inner}for (ushort _e = 0; _e < {len(dest.rows)}; _e++) "
                        f"{tensor}[_e] = 0.0f;"
                    ),
                )
            ),
            f"{inner}auto _mr_ones = {atom.tensor(op, 'right')};",
            f"{inner}#pragma unroll",
            (
                f"{inner}for (ushort _e = 0; _e < _mr_ones.get_capacity(); _e++) "
                f"_mr_ones[_e] = static_cast<{atom.right}>(1.0f);"
            ),
        ]
        pad, inner = inner, inner + "    "
    # Each step's operands (``fills``) and its ops (``ops``), in one block.
    fills: list[list[str]] = []
    ops: list[list[str]] = []
    for step in range(steps):
        lines: list[str] = []
        live_terms: list[str] = []
        step_names = ["", ""]
        for i, side in enumerate(sides):
            operand = ("left", "right")[i]
            if side[0] == "reg":
                lanes, loop = cast("tuple[CoopLanes, ast.For]", side[1:])
                step_names[i] = f"_mr_{'ab'[i]}{step}"
                lines += [
                    f"{inner}auto {step_names[i]} = {atom.tensor(op, operand)};",
                    f"{inner}{{",
                    f"{inner}    constexpr short _fs = {step};",
                ]
                state.put_type = (atom.left, atom.right)[i]
                for e in range(len(lanes.rows)):
                    row, col = lanes.rows[e], lanes.cols[e]
                    k_coord = col if i == 0 else row
                    if interleave:
                        k_pos = f"{_at(0, k_coord, steps)} + _fs"
                    else:
                        k_pos = f"{atom.k} * _fs + {_at(0, k_coord)}"
                    if i == 0:
                        decls = (f"_e_m = {_at(0, row)}", f"_e_k = {k_pos}")
                    else:
                        decls = (f"_e_k = {k_pos}", f"_e_n = {_at(0, col)}")
                    state.put = f"{step_names[i]}[{e}]"
                    el = inner + "    "
                    element = [f"{el}{{", *(f"{el}    int {decl};" for decl in decls)]
                    body, _, _ = element_body(loop, decls, step, e, facts, 0, state)
                    if lanes.valid is not None:
                        element.append(f"{el}    if ({lanes.valid[e]}) {{")
                        with state.isolated_scope():
                            emit_body(body, element, len(el) + 8, state)
                        element.append(f"{el}    }} else {state.put} = 0;")
                    else:
                        with state.isolated_scope():
                            emit_body(body, element, len(el) + 4, state)
                    lines += [*element, f"{el}}}"]
                lines.append(f"{inner}}}")
            else:
                args, live, outer = cast(
                    "tuple[list[ast.expr], tuple[bool, bool], list[ast.expr]]", side[1:]
                )
                blocks = (atom.m, atom.k) if i == 0 else (atom.k, atom.n)
                k_dim = 1 if i == 0 else 0
                dtype = (atom.left, atom.right)[i]
                step_names[i] = f"_mpp_{'ab'[i]}{step}"
                decls_v, live_v, handle = coop_view(
                    args,
                    dtype,
                    step_names[i],
                    live,
                    blocks,
                    outer,
                    k_dim,
                    str(atom.k * step),
                    k_tail=not whole,
                )
                lines += [f"{inner}{line}" for line in decls_v]
                live_terms += live_v
                lines.append(f"{inner}{handle}")
        target = "_mr_p" if affine else acc.id
        runs = [f"{op}.run({step_names[0]}, {step_names[1]}, {target});"]
        if affine:
            runs.append(f"{op}.run({step_names[0]}, _mr_ones, _mr_x);")
        if live_terms:
            # A view's step past the tile's live K adds nothing; every term
            # is uniform over the SIMD group.
            runs = [
                f"{inner}if ({' && '.join(live_terms)}) {{",
                *(f"{inner}    {run}" for run in runs),
                f"{inner}}}",
            ]
        else:
            runs = [f"{inner}{run}" for run in runs]
        fills.append(lines)
        ops.append(runs)
    # A run with one register operand builds each step's operands before the
    # previous step's ops, which keeps two of its cooperative tensors live.
    # With both inputs in registers (an M4 at 16 and 32 rows) that is four,
    # and on an M4 Max it slowed K-split configs at 16 rows by up to 1.8x.
    # The affine split holds a third right input, its tensor of ones, across
    # the steps; built ahead beside it, 5- and 6-bit weights at 32 rows with
    # 8 steps a tile summed wrong on an M5 Max (exact with the ones built
    # fresh each step, or with the steps in order).  Both keep their order.
    ahead = not affine and sum(side[0] == "reg" for side in sides) == 1
    parts.append(f"{pad}{{")
    for step in range(steps):
        parts += fills[step]
        if not ahead:
            parts += ops[step]
        elif step:
            parts += ops[step - 1]
    if ahead:
        parts += ops[-1]
    parts.append(f"{pad}}}")
    if dest is not None:
        loop = loops[REG_EPILOGUE]
        av, pv, xv = _SPLIT_VALUES
        state.put_type = None
        for e in range(len(dest.rows)):
            decls = (
                f"_e_m = {_at(0, dest.rows[e])}",
                f"_e_n = {_at(0, dest.cols[e])}",
                "_e_k = 0",
            )
            state.put = f"{acc.id}[{e}]"
            body, _, _ = element_body(loop, decls, 0, e, facts, 0, state)
            element = [
                f"{pad}{{",
                *(f"{pad}    int {decl};" for decl in decls),
                f"{pad}    float {av} = {acc.id}[{e}];",
                f"{pad}    float {pv} = _mr_p[{e}];",
                f"{pad}    float {xv} = _mr_x[{e}];",
            ]
            with state.isolated_scope():
                emit_body(body, element, len(pad) + 4, state)
            element.append(f"{pad}}}")
            if dest.valid is not None:
                # An element the lane doesn't hold keeps its accumulator.
                element = [
                    f"{pad}if ({dest.valid[e]}) {{",
                    *(f"    {line}" for line in element),
                    f"{pad}}}",
                ]
            parts += element
        parts.append(f"{' ' * indent}}}")
    state.put, state.put_type = saved


def register_split_runs(body: list[ast.stmt], loop_var: str) -> list[ast.For]:
    """The register runs in a K loop's ``body`` that SIMD groups split over
    the loop (``loop_var``): their ``k_split`` names its offset."""
    runs: list[ast.For] = []
    for stmt in body:
        if isinstance(stmt, ast.For) and call_name(stmt.iter) == REG_RUN:
            assert isinstance(stmt.iter, ast.Call)
            split = keyword(stmt.iter, "k_split")
            assert split is not None
            value = ast.literal_eval(split)
            if value is not None and value[0] == loop_var:
                runs.append(stmt)
    return runs


def emit_register_combine(
    runs: list[ast.For], parts: list[str], pad: str, state: EmitState
) -> None:
    """After a K loop the SIMD groups split, add their accumulators (those
    the loop's register ``runs`` update) into the first's
    (``operand_elements.emit_split_adds``), as fragment runs' are: a lane's
    elements of each destination tensor, each then reset to 0."""
    held: list[str] = []
    for run in runs:
        assert isinstance(run.iter, ast.Call)
        acc = run.iter.args[0]
        assert isinstance(acc, ast.Name)
        root = state.coop[acc.id]
        atom = state.register_atoms[root]
        count = len(operand_lanes(state, atom, "dest").rows)
        elements = [f"{root}[{e}]" for e in range(count)]
        if elements[0] not in held:
            held += elements
    count = len(held)
    state.split_slots = max(state.split_slots, (state.split - 1) * count * SIMD_WIDTH)

    def slot(simdgroup: str, e: int) -> str:
        return f"{SPLIT_BUFFER}[({simdgroup} * {count} + {e}) * {SIMD_WIDTH} + {LANE}]"

    emit_split_adds(
        held, [f"{element} = 0.0f;" for element in held], slot, state.split, parts, pad
    )


class RegisterLowering:
    """What :mod:`mpp_matmul` calls to build a group's computed operands in
    registers: the planner's check, the decision and the accumulator's
    keywords, the op's codegen and its loops in ``elementize``.
    ``MetalBackend`` gives it to the planner under an MPP config (codegen
    then takes it where the launch is one SIMD group) and ``metal_jit`` to
    ``elementize``, so :mod:`mpp_matmul`, which this module builds on,
    doesn't import it."""

    check_group = staticmethod(check_register_operands)
    spec = staticmethod(register_run_spec)
    acc_keywords = staticmethod(register_acc_keywords)
    codegen_run = staticmethod(codegen_register_run)
    run = staticmethod(register_run)
    run_binds = staticmethod(run_binds)


REGISTER_LOWERING = RegisterLowering()
