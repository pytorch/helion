"""Run a device loop's complete tiles apart from its partial last tile, and
load their lanes in packets.

Role: device-body pass (``cute_tail_peel``), before
``vectorize_affine_tile_lanes``.

A device loop ``range(0, n, tile)`` whose tile does not divide ``n`` masks
every lane load of every tile against ``n``: an odd vocabulary (50265
columns) turns the 16-byte loads of a row's tiles into masked 2-byte loads,
and every max and sum that reads them selects ``-inf`` or ``0`` for the
masked lanes.  Split into ``range(0, m, tile)`` and ``range(m, n, tile)``
with ``m = n - n % tile``, the first loop covers complete tiles, whose masks
``simplify_proven_loop_bounds`` proves true (with the definitions before the
loop in view); the second keeps them for the last tile.  In the first loop a
``vec_lane`` loop's load of ``base + vec_lane`` along a unit-stride dim then
becomes one packet loaded before the ``vec_lane`` loop, when ``base`` (an
affine form of the loop variable, constexpr lanes and thread indices), the
other dims' strides and the tensor's base pointer are aligned to it.  Both
loops run the same iterations, with the same lanes on the same threads, in
the same order, and each value comes from the same elements, so the results
are the original's except where LLVM, with the per-lane selects gone, fuses
a multiply and add into one FMA (as under ``cute_uniform_lane_compares``).

Unlike ``peel_partial_chunks`` (the rolled sweeps of ``cute_proven_bounds``),
the loop body may hold warp reductions: a warp shuffle claims no resource of
its own, so the second copy runs once more what the loop ran once per tile.
Shared-memory reduces and allocations still keep a loop whole.  For the sake
of such lane loads, a loop is split only at the top level of the kernel body
(or of its loop over a persistent launch's virtual program ids), and only
when a lane loop of its body loads under a mask against the loop's end and
its complete tiles gain a packet.

A nest ``cute_vloop_sink`` interchanged loads its lanes in packets already,
each under one predicate (``_cute_load_v16b_if``, zeros where it is false),
and selects every lane's value under the kernel's own index masks and the
tile mask.  A device loop that binds such packets is handled after the
splits above, whatever its block and at any depth of the loops, branches
and ``with`` blocks the bounds proof walks: the proof then also takes the
launch's block grid where it is exact and follows proven predicates through
the names they are bound to (``propagate_truths``).
Where it proves a packet's predicate for the complete tiles, their copy
loads the packet unpredicated under the same cache hint, and the selects
the proof discharged with it are gone; the partial last tile keeps the loop
as it was.  A loop none of whose packets is proven stays as it is, and
tiles that divide the extent are all complete: that loop is not split, only
unmasked, which is also how the complete tiles of a loop split for its lane
loads lose their packet predicates.  No statement is copied twice: a loop
nested in one that was split for its lane loads is unmasked only where its
tiles divide its extent, never split again, and a loop around one whose
packets were unmasked stays as it is.  The proof's facts are the loop's
literal bounds, the launch shape, the constexpr block sizes and the block
grid, which is the launch's own: a static flat launch's, or an XYZ launch's,
whose extents are static or input sizes that the bound kernel's cache key
fixes (``_block_grid_dims``).  So a mask the kernel states itself (``row <
rows``) is discharged only where these imply it.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from typing import TYPE_CHECKING

import torch

from ...language._tracing_ops import _for_loop
from ...language.memory_ops import load
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_expr
from ..device_ir import ForLoopGraphInfo
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_written
from .cache_policy_loads import CachePolicy
from .literal_offset_loads import add_terms
from .literal_offset_loads import stride_term
from .literal_offset_loads import strip_int_cast
from .memory_ops import tensor_proven_base_alignment
from .packet_access import PacketAccess
from .packet_access import call_write_roots
from .packet_access import match_packet_access
from .packet_access import render_packet_load
from .peel_partial_chunks import peelable
from .peel_partial_chunks import static_int
from .peel_partial_chunks import static_range
from .peel_partial_chunks import stays_whole
from .peel_partial_chunks import with_bounds
from .persistent_branch_vec import _binding_write_roots
from .proven_loop_bounds import simplify_proven_loop_bounds
from .proven_loop_bounds import with_proof_marker

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR


# The float dtypes ``tensor_facts`` gives a packet carrier.
_PACKET_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
# Set on both copies of a loop split for its lane loads (``_split_loops``).
_SPLIT_COPY = "_helion_tail_peel_copy"


def tail_peel_blocks(device_ir: DeviceIR, config_spec: ConfigSpec) -> frozenset[int]:
    """The blocks whose device loops open ``cute_tail_peel`` to the search:
    a one-axis ``hl.tile`` loop in a root graph (the kernel body's top level)
    over a static extent and a block that can have vector lanes, which loads
    a 16- or 32-bit float tensor along its tile.  Structure only: whether the
    last tile is partial and the lanes align to a packet are facts of the
    shape, and where they fail the pass leaves the loop whole.  In a kernel
    that has such a block the pass is not limited to its loops: it also
    unmasks the predicated packets of any device loop, whatever its block
    (the module docstring).  A dim is matched by its block's size symbol, so
    this needs no current host function."""
    env = CompileEnvironment.current()
    lane_blocks = set(config_spec.cute_vector_widths.valid_block_ids())
    blocks: set[int] = set()
    for root_id in device_ir.root_ids:
        root = device_ir.graphs[root_id].graph
        for node in root.find_nodes(op="call_function", target=_for_loop):
            info = device_ir.graphs[node.args[0]]
            # A rolled reduction's loop is a ``ReductionLoopGraphInfo``.
            if type(info) is not ForLoopGraphInfo or len(info.block_ids) != 1:
                continue
            block_id = info.block_ids[0]
            if block_id not in lane_blocks or not isinstance(
                env.block_sizes[block_id].size, int
            ):
                continue
            symbol = env.block_sizes[block_id].symbol()
            for read in info.graph.find_nodes(op="call_function", target=load):
                source = read.args[0]
                value = read.meta.get("val")
                if (
                    isinstance(source, torch.fx.Node)
                    and isinstance(tensor := source.meta.get("val"), torch.Tensor)
                    and tensor.dtype in _PACKET_DTYPES
                    and isinstance(value, torch.Tensor)
                    and any(
                        isinstance(size, torch.SymInt) and _symint_expr(size) == symbol
                        for size in value.shape
                    )
                ):
                    blocks.add(block_id)
                    break
    return frozenset(blocks)


def _masked_lane_load(body: list[ast.stmt], end: int) -> bool:
    """Whether a ``range_constexpr`` lane loop of ``body`` loads under a
    conditional expression whose test compares against ``end``."""
    masks: set[str] = set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and any(
                isinstance(compare, ast.Compare)
                and any(
                    isinstance(right, ast.Constant) and right.value == end
                    for right in compare.comparators
                )
                for compare in ast.walk(node.value)
            )
        ):
            masks.add(node.targets[0].id)
    if not masks:
        return False
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and call_path_str(node.iter.func) == "cutlass.range_constexpr"
        ):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.IfExp)
                and isinstance(inner.test, ast.Name)
                and inner.test.id in masks
                and isinstance(inner.body, ast.Call)
                and isinstance(inner.body.func, ast.Attribute)
                and inner.body.func.attr == "load"
            ):
                return True
    return False


# Calls that order memory between threads or across the launch; a load
# hoisted above one may read what it was meant to wait for.
_SYNCHRONIZATIONS = frozenset(
    {
        "cute.arch.barrier",
        "cute.arch.cluster_arrive",
        "cute.arch.cluster_arrive_relaxed",
        "cute.arch.cluster_wait",
        "cute.arch.griddepcontrol_launch_dependents",
        "cute.arch.griddepcontrol_wait",
        "cute.arch.sync_threads",
        "cute.arch.sync_warp",
    }
)


def _synchronizes(call: ast.Call) -> bool:
    """Whether ``call`` is a barrier, an mbarrier operation or a fence."""
    path = call_path_str(call.func)
    return path is not None and (
        path in _SYNCHRONIZATIONS
        or path.startswith(("cute.arch.mbarrier", "cute.arch.fence"))
    )


@dataclasses.dataclass(frozen=True)
class TensorFacts:
    """What a packet of a tensor argument needs: its scalar dtype and packet
    carrier (``cutlass.BFloat16`` / ``cutlass.Uint16``), the element size,
    the proven strides by dim and the proven base alignment in bytes (None
    when none is proven)."""

    dtype: str
    carrier: str
    itemsize: int
    strides: Mapping[int, int]
    alignment: int | None


_CARRIERS = {2: "cutlass.Uint16", 4: "cutlass.Uint32"}
_CARRIER_BYTES = {carrier: itemsize for itemsize, carrier in _CARRIERS.items()}


def tensor_facts(device_function: DeviceFunction) -> dict[str, TensorFacts]:
    """The packet facts of ``device_function``'s float tensor arguments: the
    proven strides and the base alignment the cache key backs
    (``tensor_proven_base_alignment``)."""
    from ..device_function import TensorArg

    env = CompileEnvironment.current()
    strides = device_function.proven_tensor_stride_values()
    facts: dict[str, TensorFacts] = {}
    for arg in device_function.arguments:
        if not isinstance(arg, TensorArg) or arg.fake_value.dtype not in _PACKET_DTYPES:
            continue
        tensor = arg.fake_value
        facts[arg.name] = TensorFacts(
            dtype=env.backend.dtype_str(tensor.dtype),
            carrier=_CARRIERS[tensor.element_size()],
            itemsize=tensor.element_size(),
            strides={
                dim: value for (name, dim), value in strides.items() if name == arg.name
            },
            alignment=tensor_proven_base_alignment(env, tensor),
        )
    return facts


def _affine(node: ast.expr) -> dict[str, int] | None:
    """``node`` as integer coefficients of names, thread indices (keys
    ``"thread_idx[0]"``.., which no name can be) and a constant (key ``""``),
    or ``None``."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return {"": node.value}
    if isinstance(node, ast.Name):
        return {node.id: 1}
    if (
        isinstance(node, ast.Call)
        and call_path_str(node.func) in ("cutlass.Int32", "cutlass.Int64")
        and len(node.args) == 1
        and not node.keywords
    ):
        return _affine(node.args[0])
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and call_path_str(node.value.func) == "cute.arch.thread_idx"
        and not node.value.args
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in (0, 1, 2)
    ):
        return {f"thread_idx[{node.slice.value}]": 1}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = _affine(node.left), _affine(node.right)
        if left is None or right is None:
            return None
        sign = 1 if isinstance(node.op, ast.Add) else -1
        result = dict(left)
        for key, value in right.items():
            result[key] = result.get(key, 0) + sign * value
        return result
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = _affine(node.left), _affine(node.right)
        if left is None or right is None:
            return None
        for scale, form in ((left, right), (right, left)):
            if set(scale) <= {""}:
                factor = scale.get("", 0)
                return {key: factor * value for key, value in form.items()}
    return None


def _multiple_of(
    form: dict[str, int], width: int, multiples: Mapping[str, int]
) -> bool:
    """Whether ``form`` is a multiple of ``width`` for every value of its
    names: a name in ``multiples`` is a multiple of that value, any other
    name or thread index an arbitrary integer."""
    for key, coefficient in form.items():
        unit = multiples.get(key, 1) if key else 1
        if (coefficient * unit) % width:
            return False
    return True


def _load_parts(value: ast.expr) -> tuple[ast.Call, ast.Name | None] | None:
    """``(PTR.load() call, mask name or None)`` of ``PTR.load()`` or
    ``PTR.load() if mask else <constant>``."""
    test: ast.Name | None = None
    if isinstance(value, ast.IfExp):
        if not isinstance(value.test, ast.Name):
            return None
        test = value.test
        value = value.body
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "load"
        and not value.args
        and not value.keywords
    ):
        return value, test
    return None


class _Packetizer:
    """Turn the lane loads of a complete-tile loop into packets: in a
    ``range_constexpr(V)`` loop over ``vec_lane_*``, a load whose index is
    ``base + vec_lane`` along a unit-stride dim, unmasked or under a mask the
    proof set to ``True``, becomes an extract of one ``V``-element packet
    loaded before the loop, when ``base``, the other dims' offsets and the
    tensor's base pointer are aligned to the packet."""

    def __init__(
        self,
        facts: Mapping[str, TensorFacts],
        multiples: dict[str, int],
        constexpr_values: Mapping[str, int],
        new_var: Callable[[str], str],
    ) -> None:
        self.facts = facts
        self.multiples = multiples
        self.constexpr_values = constexpr_values
        self.new_var = new_var
        self.count = 0

    def block(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        definitions: dict[str, ast.expr] = {}
        result: list[ast.stmt] = []
        for statement in statements:
            if isinstance(statement, (ast.For, ast.While)):
                # A name the loop binds again holds its multiple only until
                # that binding, which the loop's next iteration sees.
                for name in names_written(statement) & self.multiples.keys():
                    del self.multiples[name]
            if isinstance(statement, ast.For):
                self._loop(statement, definitions, result)
            else:
                for field in ("body", "orelse"):
                    inner = getattr(statement, field, None)
                    if isinstance(inner, list) and inner:
                        setattr(statement, field, self.block(inner))
                result.append(statement)
            # A name bound again, by any statement (a branch or loop that
            # binds it, an augmented or tuple assignment), no longer holds
            # its definition, nor does a name defined from it, and the loop
            # variable no longer holds a multiple of its step.
            written = names_written(statement)
            for name in [*definitions]:
                if name in written or names_read(definitions[name]) & written:
                    del definitions[name]
            for name in written & self.multiples.keys():
                del self.multiples[name]
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                definitions[statement.targets[0].id] = statement.value
        return result

    def _loop(
        self,
        loop: ast.For,
        definitions: dict[str, ast.expr],
        result: list[ast.stmt],
    ) -> None:
        call = loop.iter
        target = loop.target
        width = (
            static_int(call.args[0], self.constexpr_values)
            if isinstance(call, ast.Call)
            and call_path_str(call.func) == "cutlass.range_constexpr"
            and len(call.args) == 1
            else None
        )
        if (
            width is None
            or not isinstance(target, ast.Name)
            or not target.id.startswith("vec_lane_")
        ):
            loop.body = self.block(loop.body)
            result.append(loop)
            return
        if _binding_write_roots(loop) & self.facts.keys() or any(
            isinstance(node, ast.Call)
            and (
                _synchronizes(node)
                or call_write_roots(node, lambda address: None) != frozenset()
            )
            for node in ast.walk(loop)
        ):
            # A packet loads before the loop, ahead of every write in it (a
            # store, through a call or a subscript of a tensor argument, or a
            # call that may write memory it does not name, such as impure
            # inline asm) and of every synchronization in it (a barrier that
            # makes other threads' stores visible writes nothing itself).
            result.append(loop)
            return
        hoisted = self._packets(loop, target.id, width, definitions)
        result.extend(hoisted)
        result.append(loop)

    def _packets(
        self,
        loop: ast.For,
        lane: str,
        width: int,
        definitions: dict[str, ast.expr],
    ) -> list[ast.stmt]:
        bases: dict[str, ast.expr] = {}
        true_masks: set[str] = set()
        hoisted: list[ast.stmt] = []
        # A packet loads before the loop, so its pointer may read only names
        # bound before it, and no element or attribute of a name the loop
        # stores one of.
        assigned = _binding_write_roots(loop)
        for statement in loop.body:
            # A name bound again no longer holds the lane offset or the true
            # mask it was recorded for.
            for written in names_written(statement):
                bases.pop(written, None)
                true_masks.discard(written)
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                continue
            name, value = statement.targets[0].id, statement.value
            if isinstance(value, ast.Constant) and value.value is True:
                true_masks.add(name)
                continue
            form = _affine(value)
            if form is not None and form.get(lane) == 1:
                rest = {key: v for key, v in form.items() if key != lane and v}
                base = next(iter(rest), None)
                if base and rest == {base: 1} and base in definitions:
                    bases[name] = ast.Name(id=base, ctx=ast.Load())
                continue
            parts = _load_parts(value)
            if parts is None:
                continue
            load, mask = parts
            if mask is not None and mask.id not in true_masks:
                continue
            packet = self._packet(load, lane, width, bases, definitions)
            if packet is None or names_read(packet[0]) & assigned:
                continue
            pointer, facts = packet
            carrier_name = self.new_var("tail_packet")
            self.count += 1
            hoisted.append(
                ast.Assign(
                    targets=[ast.Name(id=carrier_name, ctx=ast.Store())],
                    value=ast.parse(
                        render_packet_load(ast.unparse(pointer), width, facts.carrier),
                        mode="eval",
                    ).body,
                )
            )
            statement.value = ast.parse(
                f"{facts.carrier}({carrier_name}[{lane}]).bitcast({facts.dtype})",
                mode="eval",
            ).body
        return hoisted

    def _packet(
        self,
        load: ast.Call,
        lane: str,
        width: int,
        bases: Mapping[str, ast.expr],
        definitions: Mapping[str, ast.expr],
    ) -> tuple[ast.expr, TensorFacts] | None:
        assert isinstance(load.func, ast.Attribute)
        terms = add_terms(load.func.value)
        roots = [
            term
            for term in terms
            if isinstance(term, ast.Attribute)
            and term.attr == "iterator"
            and isinstance(term.value, ast.Name)
        ]
        if len(roots) != 1:
            return None
        tensor = roots[0].value
        assert isinstance(tensor, ast.Name)
        facts = self.facts.get(tensor.id)
        if facts is None or facts.itemsize * width not in (8, 16):
            return None
        if facts.alignment is None or facts.alignment % (facts.itemsize * width):
            return None
        lane_term: ast.expr | None = None
        for term in terms:
            if term is roots[0]:
                continue
            parts = stride_term(term)
            if parts is None:
                return None
            coefficient, name, dim = parts
            index = strip_int_cast(coefficient)
            if name != tensor.id:
                return None
            if isinstance(index, ast.Name) and index.id in bases:
                if lane_term is not None or facts.strides.get(dim) != 1:
                    return None
                lane_term = term
                base = bases[index.id]
                assert isinstance(base, ast.Name)
                base_form = _affine(definitions[base.id])
                if base_form is None or not _multiple_of(
                    base_form, width, self.multiples
                ):
                    return None
                continue
            stride = facts.strides.get(dim)
            # Another dim's index that reads a lane offset differs per lane,
            # where the packet takes one address for all its lanes.
            if stride is None or stride % width or names_read(index) & bases.keys():
                return None
        if lane_term is None:
            return None
        pointer = deep_clone(load.func.value)
        for node in ast.walk(pointer):
            if isinstance(node, ast.Name) and node.id in bases:
                base_name = bases[node.id]
                assert isinstance(base_name, ast.Name)
                node.id = base_name.id
        return pointer, facts


def peel_vector_tails(
    body: list[ast.stmt],
    constexpr_values: Mapping[str, int],
    *,
    thread_block_dims: tuple[int, int, int] | None,
    tensor_facts: Mapping[str, TensorFacts],
    new_var: Callable[[str], str],
    persistent_loop: str | None,
    block_grid_dims: tuple[int, int, int] | None = None,
) -> list[ast.stmt]:
    """Split each qualifying device loop at the top level of ``body``, or of
    the body of its loop over a persistent launch's virtual program ids
    (whose variable is ``persistent_loop``), whose complete tiles, once
    their masks are proven, load at least one lane in packets (named by
    ``new_var``); then run the complete tiles of each device loop that loads
    predicated packets, at any depth, unpredicated where the masks are
    proven given the launch's exact block grid (``block_grid_dims``, None
    when it is not known).  Other loops, and ``body`` without such a loop,
    stay as they are."""
    if thread_block_dims is None:
        return body
    result, proven_any = _split_loops(
        body,
        [],
        constexpr_values,
        thread_block_dims,
        tensor_facts,
        new_var,
        persistent_loop,
    )
    # After the splits: a loop split for its lane loads keeps the packets
    # its complete tiles gained, and those tiles, which divide their extent,
    # are then unmasked in place.
    unmasker = _Unmasker(dict(constexpr_values), thread_block_dims, block_grid_dims)
    result = unmasker.block(result, [])
    if not (proven_any or unmasker.count):
        return body
    return with_proof_marker(result)


def _masked_packets(node: ast.AST) -> list[tuple[ast.Assign, PacketAccess]]:
    """The predicated packet loads ``node`` binds, with their statements."""
    return [
        (statement, access)
        for statement in ast.walk(node)
        if isinstance(statement, ast.Assign)
        and (access := match_packet_access(statement.value)) is not None
        and access.kind == "masked_packet"
    ]


def _unpredicated(access: PacketAccess) -> ast.expr | None:
    """The plain or hinted load of a predicated 8- or 16-byte packet under
    the same cache hint, or None when no unpredicated load carries it."""
    arguments = access.call.args
    priority = ""
    if len(arguments) == 4:
        hint = arguments[3]
        if not (isinstance(hint, ast.Constant) and isinstance(hint.value, str)):
            return None
        priority = hint.value
    if access.width is None or access.carrier not in _CARRIER_BYTES:
        return None
    packet_bytes = access.width * _CARRIER_BYTES[access.carrier]
    if packet_bytes not in (8, 16):
        return None
    policy = CachePolicy.from_masked_priority(priority)
    helper = policy.vector_helper(packet_bytes)
    if policy.is_l2_marker and helper is None:
        return None
    return ast.parse(
        render_packet_load(
            ast.unparse(access.pointer),
            access.width,
            access.carrier,
            helper=helper,
            kwargs=policy.load_kwargs,
        ),
        mode="eval",
    ).body


def _blocks(statement: ast.stmt) -> tuple[str, ...]:
    """The fields of ``statement`` that hold statements the bounds proof
    walks: the body and the ``else`` of a loop and of a branch, the body of
    a ``with`` block.  Any other compound statement stops the proof
    (``supported_statements``), so nothing in it or around it is proven."""
    if isinstance(statement, (ast.For, ast.If)):
        return ("body", "orelse")
    if isinstance(statement, ast.With):
        return ("body",)
    return ()


def _stand_in(holder: ast.stmt, field: str, statements: list[ast.stmt]) -> ast.stmt:
    """``holder`` for the bounds proof: clones of its header and of its
    other statements, with ``statements`` themselves as its ``field``."""
    lists = {
        name: statements
        if name == field
        else [deep_clone(statement) for statement in getattr(holder, name)]
        for name in _blocks(holder)
    }
    stand_in: ast.stmt
    if isinstance(holder, ast.For):
        stand_in = ast.For(
            target=deep_clone(holder.target),
            iter=deep_clone(holder.iter),
            body=lists["body"],
            orelse=lists["orelse"],
        )
    elif isinstance(holder, ast.If):
        stand_in = ast.If(
            test=deep_clone(holder.test), body=lists["body"], orelse=lists["orelse"]
        )
    else:
        assert isinstance(holder, ast.With)
        stand_in = ast.With(
            items=[deep_clone(item) for item in holder.items], body=lists["body"]
        )
    return ast.fix_missing_locations(ast.copy_location(stand_in, holder))


# Where a statement stands: each statement list around it, outermost first,
# with the index of its statement that holds the next list and the field of
# that statement which does (``_blocks``).
_Enclosing = list[tuple[list[ast.stmt], int, str]]


class _Unmasker:
    """Run the complete tiles of device loops that load predicated packets
    unpredicated (the module docstring's sunk nests)."""

    def __init__(
        self,
        constexpr_values: dict[str, int],
        thread_block_dims: tuple[int, int, int],
        block_grid_dims: tuple[int, int, int] | None,
    ) -> None:
        self.constexpr_values = constexpr_values
        self.thread_block_dims = thread_block_dims
        self.block_grid_dims = block_grid_dims
        # The packets unmasked so far, in split loops and in place.
        self.count = 0

    def block(
        self,
        statements: list[ast.stmt],
        enclosing: _Enclosing,
        copied: bool = False,
    ) -> list[ast.stmt]:
        """``statements`` with their qualifying loops replaced, those in
        the branches and ``with`` blocks among them included; ``enclosing``
        locates ``statements``.  ``copied``: they are in a copy of a loop
        split for its lane loads (``_SPLIT_COPY``), whose body may not be
        copied again."""
        result: list[ast.stmt] = []
        for index, statement in enumerate(statements):
            fields = _blocks(statement)
            if not fields or not _masked_packets(statement):
                result.append(statement)
                continue
            in_copy = copied or getattr(statement, _SPLIT_COPY, False)
            # The innermost device loop that gains a packet first: a loop
            # that holds one this pass unmasked, whether it was split for
            # that or unmasked in place, stays as it is.  A loop under a
            # branch or in a ``with`` block is settled where it stands, like
            # one directly in its enclosing loop's body.
            before = self.count
            for field in fields:
                setattr(
                    statement,
                    field,
                    self.block(
                        getattr(statement, field),
                        [*enclosing, (statements, index, field)],
                        in_copy,
                    ),
                )
            replaced = (
                self._tiles(statement, statements, index, enclosing, split=not in_copy)
                if isinstance(statement, ast.For) and self.count == before
                else None
            )
            if replaced is None:
                result.append(statement)
            else:
                result.extend(replaced)
        return result

    def _tiles(
        self,
        loop: ast.For,
        statements: list[ast.stmt],
        index: int,
        enclosing: _Enclosing,
        *,
        split: bool,
    ) -> list[ast.stmt] | None:
        """The complete tiles of ``loop``, which is ``statements[index]``
        under ``enclosing``, unmasked, then its partial last tile as it
        was, or None when the loop stays as it is.  Without ``split`` only
        a loop whose tiles are all complete is unmasked."""
        bounds = static_range(loop, self.constexpr_values)
        if bounds is None:
            return None
        start, end, step = bounds
        full_end = end - (end - start) % step
        if full_end <= start:
            # No complete tile.
            return None
        if full_end < end and not split:
            # The loop is, or is nested in, a copy of a loop split for its
            # lane loads: another split would copy its body a second time.
            return None
        if full_end < end and stays_whole(loop.body, warp_reduces_keep_whole=False):
            # The partial tile's copy would take the loop's shared memory
            # again (``peel_partial_chunks``).
            return None
        complete = with_bounds(loop, start, full_end, mark=False)
        self._prove(complete, statements, index, enclosing)
        gained = 0
        for statement, access in _masked_packets(complete):
            predicate = access.predicate
            if not (isinstance(predicate, ast.Constant) and predicate.value is True):
                continue
            load = _unpredicated(access)
            if load is not None:
                statement.value = load
                gained += 1
        if not gained:
            return None
        self.count += gained
        # The partial last tile keeps the loop as it was.  Neither copy is
        # split again: the loops nested in them gained no packet, the splits
        # for lane loads ran before, and the loops around them stay as they
        # are (``block``).
        if full_end < end:
            return [complete, with_bounds(loop, full_end, end, mark=False)]
        return [complete]

    def _prove(
        self,
        complete: ast.For,
        statements: list[ast.stmt],
        index: int,
        enclosing: _Enclosing,
    ) -> None:
        """Discharge in ``complete`` the predicates the bounds proof proves
        where the loop stands, at ``statements[index]`` under ``enclosing``.
        The proof walks clones of the statements around it: every enclosing
        loop, branch and ``with`` block whole (``_stand_in``), so a name a
        loop binds again, ahead of ``complete`` or behind it, is unknown
        where the loop reads it.  It rewrites the loops it walks in place,
        so ``complete`` itself holds the result wherever the proof leaves it
        among the clones (an ``if`` it resolves, ahead of the loop or around
        it, is replaced by its body, which moves the statements after
        it)."""
        node: ast.stmt = complete
        for outer_statements, outer_index, field in reversed(enclosing):
            node = _stand_in(
                outer_statements[outer_index],
                field,
                [
                    *(deep_clone(earlier) for earlier in statements[:index]),
                    node,
                    *(deep_clone(later) for later in statements[index + 1 :]),
                ],
            )
            statements, index = outer_statements, outer_index
        simplify_proven_loop_bounds(
            [*(deep_clone(earlier) for earlier in statements[:index]), node],
            enabled=True,
            thread_block_dims=self.thread_block_dims,
            constexpr_values=dict(self.constexpr_values),
            block_grid_dims=self.block_grid_dims,
            propagate_truths=True,
        )


def _unknown(name: str) -> ast.stmt:
    """``name`` bound to a call the bounds proof cannot evaluate."""
    return ast.parse(f"{name} = _tail_peel_unknown()").body[0]


def _split_loops(
    body: list[ast.stmt],
    before: list[ast.stmt],
    constexpr_values: Mapping[str, int],
    thread_block_dims: tuple[int, int, int],
    tensor_facts: Mapping[str, TensorFacts],
    new_var: Callable[[str], str],
    persistent_loop: str | None,
) -> tuple[list[ast.stmt], bool]:
    """``body`` with its qualifying device loops split (``peel_vector_tails``),
    both copies marked ``_SPLIT_COPY``, ``before`` the statements that run
    ahead of it, and whether one was."""
    result: list[ast.stmt] = []
    proven_any = False
    for index, statement in enumerate(body):
        if (
            isinstance(statement, ast.For)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == persistent_loop
        ):
            # Each virtual program id runs the loop's body as a block's
            # whole body would.  A statement ahead of the loop that binds a
            # name the loop binds again holds only on the first virtual id,
            # but it may still have changed any name it writes: in its place
            # the proof sees each of those names bound to a value it cannot
            # know.
            rebound = names_written(statement)
            ahead: list[ast.stmt] = []
            for earlier in [*before, *body[:index]]:
                written = names_written(earlier)
                if written & rebound:
                    ahead.extend(_unknown(name) for name in sorted(written))
                else:
                    ahead.append(earlier)
            statement.body, proven = _split_loops(
                statement.body,
                ahead,
                constexpr_values,
                thread_block_dims,
                tensor_facts,
                new_var,
                None,
            )
            result.append(statement)
            proven_any |= proven
            continue
        # A warp reduce in the body claims nothing of its own: the second
        # copy runs once more what the loop ran once per tile.
        split = (
            peelable(
                statement,
                constexpr_values,
                warp_reduces_keep_whole=False,
                needs=_masked_lane_load,
            )
            if isinstance(statement, ast.For)
            else None
        )
        if split is None:
            result.append(statement)
            continue
        assert isinstance(statement, ast.For)
        assert isinstance(statement.target, ast.Name)
        start, end, full_end, step = split
        complete = with_bounds(statement, start, full_end, mark=False)
        partial = with_bounds(statement, full_end, end, mark=False)
        # Prove the complete tiles' masks with the definitions before the
        # loop in view (clones: the proof may simplify them too, which this
        # pass leaves to the passes that own those statements).
        proven = simplify_proven_loop_bounds(
            [*(deep_clone(earlier) for earlier in [*before, *body[:index]]), complete],
            enabled=True,
            thread_block_dims=thread_block_dims,
            constexpr_values=dict(constexpr_values),
        )
        complete = proven[-1]
        assert isinstance(complete, ast.For)
        packetizer = _Packetizer(
            tensor_facts,
            {statement.target.id: math.gcd(start, step)},
            constexpr_values,
            new_var,
        )
        complete.body = packetizer.block(complete.body)
        if not packetizer.count:
            result.append(statement)
            continue
        for copy in (complete, partial):
            setattr(copy, _SPLIT_COPY, True)
        result.extend((complete, partial))
        proven_any = True
    return result, proven_any
