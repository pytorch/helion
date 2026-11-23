"""Select cache hints for already-proven full global store packets."""

from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING

from .cutedsl_compat import fixed_l2_evict_last_store_policy_supported
from .packet_access import match_packet_access
from .packet_flush_names import PACKET_FLUSH_BY_POLICY
from .persistent_branch_vec import PACKET_STORE_BYTES

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from ...autotuner.config_spec import ConfigSpec

# The narrowest packet a policy applies to (``lower_packet_store_policy``).
POLICY_PACKET_BYTES = 16
# The widest component packet (``memory_ops._cute_component_store_ctx``: one
# vector store of the widest kind a target has).
COMPONENT_PACKET_MAX_BYTES = 32


def store_policy_reaches_a_packet(
    config_spec: ConfigSpec, vector_widths: list[object]
) -> bool:
    """Whether a config with these ``cute_vector_widths`` may flush a packet
    that ``cute_store_policy`` applies to.

    ``lower_packet_store_policy`` renames the global packet flushes of 16
    bytes and splits those of 32; a narrower flush and a scalar store are the
    same code under every policy.  A flush writes vector lanes along the
    stored tensor's stride-1 dim (``memory_ops._cute_vector_load_ctx``:
    ``lane_on_stride1``), as many as the vector width of the block the lanes
    belong to: the tile strategy's (``LaneLoopContext.vec_width``, a flattened
    tile's last block) or the reduction strategy's
    (``_cute_reduction_vec_width``), which both read the block's
    ``cute_vector_widths`` entry.  Where a tile indexes the stride-1 dim, the
    lanes are that tile's block's; where a full slice or a computed index
    does, any block may lend them, and the config's widest vector stands for
    it.  A component packet (``memory_ops._cute_component_store_ctx``) is the
    one flush with lanes on another dim: a tile on a dim whose every element
    holds ``stride`` contiguous ones (``out[tile, c]`` on ``[elements, K]``)
    stores the ``K`` components of its vector lanes together, ``width * K``
    elements in one store of at most 32 bytes.  So a config whose stores all
    have fewer than 16 bytes of lanes on their stride-1 dim, and no such
    component packet of 16 bytes or more, stores nothing the policy changes,
    whatever its threads, alignment and masks: the answer is False, and
    config normalization makes the policy the default one
    (``search_surface._normalize_cute_lane_knobs``).

    Two facts of the stores narrow this.  A slice on the stride-1 dim
    (``out[tile_s, tile_r, :]``, the row a reduction's lanes store;
    ``cute_sliced_store_dims``) has the lanes of the slice's block, which is
    never an ``hl.tile`` block: the widest vector of the other blocks stands
    for it, whatever the tiles' widths are.  The block is not read off the
    store, because a partial slice (``out[tile, :half]``) runs on a block of
    its own and not on the dim's (``MemoryOpFact.indexed_block_ids``); a
    computed index (``out[row, col.index * 1]``) stores the lanes of the
    tile it reads and keeps the widest of all.  And a tensor whose base is
    on no 16-byte boundary (``cute_unaligned_store_tensors``: an output view
    that starts off it in its storage) is stored in no 16-byte packet at any
    width.

    The answer is True, leaving the policy to the pass, wherever this does
    not describe the stores: a kernel with a matmul (its epilogues store
    packets of their own), a kernel with a store under a new axis
    (``cute_new_axis_stores``: behind the ``None`` of ``out[None, tile_m,
    tile_n]`` the positions of the store's fact are a dim off) and a config
    spec without store facts (no kernel filled it in).  A stride-1 dim
    indexed by an integer has no lanes.
    """
    if config_spec.matmul_facts:
        return True
    if config_spec.cute_new_axis_stores:
        return True
    stores = [fact for fact in config_spec.memory_op_facts if fact.kind == "store"]
    if not stores:
        return True
    widest = max(
        (width for width in vector_widths if isinstance(width, int)), default=1
    )
    # The widest vector a slice may run in: of the blocks that are not tiles.
    tiles = {item.block_id for item in config_spec.block_sizes}
    slice_widths = [
        config_spec.cute_vector_widths.config_get(vector_widths, block_id, 1)
        for block_id in config_spec.cute_vector_widths.valid_block_ids()
        if block_id not in tiles
    ]
    known = [width for width in slice_widths if isinstance(width, int)]
    widest_slice = max(known) if known and len(known) == len(slice_widths) else widest
    sliced = config_spec.cute_sliced_store_dims
    for fact in stores:
        if fact.dtype is None:
            # A stack tensor: not a global store.
            continue
        if fact.tensor_name in config_spec.cute_unaligned_store_tensors:
            continue
        for position, stride in enumerate(fact.subscript_strides):
            block_id = fact.subscript_block_ids[position]
            if stride != 1:
                # The lane dim of a component packet: a tile's lanes, each
                # with the ``stride`` elements behind it.
                if block_id is None:
                    continue
                lanes = config_spec.cute_vector_widths.config_get(
                    vector_widths, block_id, 1
                )
                if not isinstance(lanes, int) or (
                    lanes > 1
                    and POLICY_PACKET_BYTES
                    <= lanes * stride * fact.dtype.itemsize
                    <= COMPONENT_PACKET_MAX_BYTES
                ):
                    return True
                continue
            if block_id is not None:
                width = config_spec.cute_vector_widths.config_get(
                    vector_widths, block_id, 1
                )
            elif (
                fact.indexed_block_ids[position] is not None
                and (fact.tensor_name, position) in sliced
            ):
                # A slice of a dim that has a block.
                width = widest_slice
            else:
                # A computed index: the lanes of any block it reads.
                width = widest
            if (
                not isinstance(width, int)
                or width * fact.dtype.itemsize >= POLICY_PACKET_BYTES
            ):
                return True
    return False


def _store_call(statement: ast.stmt) -> tuple[ast.Call, str, int | None] | None:
    """``(call, collector, lanes)`` of a plain packet flush statement.

    ``lanes`` is the number of collected lanes the flush writes when the
    values are a constant slice of the collector (``values[8:16]``: a lane
    loop wider than one packet flushes several), ``None`` for the whole list
    (``PacketAccess.flushed_values``).
    """
    if not isinstance(statement, ast.Expr):
        return None
    access = match_packet_access(statement.value)
    if access is None or access.helper not in PACKET_STORE_BYTES:
        return None
    flushed = access.flushed_values()
    if flushed is None:
        return None
    collector, lanes = flushed
    return access.call, collector, None if lanes is None else lanes[1] - lanes[0]


def _global_pointer(pointer: ast.expr, tensor_names: AbstractSet[str]) -> bool:
    tensors = {
        node.value.id
        for node in ast.walk(pointer)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }
    return len(tensors) == 1 and tensors <= tensor_names


def _constexpr_trips(loop: ast.For) -> int | None:
    """The trip count of a straight ``cutlass.range_constexpr(N)`` loop."""
    if not (
        isinstance(loop.iter, ast.Call)
        and ast.unparse(loop.iter.func) == "cutlass.range_constexpr"
        and len(loop.iter.args) == 1
        and not loop.iter.keywords
        and isinstance(loop.iter.args[0], ast.Constant)
        and type(loop.iter.args[0].value) is int
        and loop.iter.args[0].value > 0
        and not loop.orelse
        and not any(
            isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise))
            for node in ast.walk(loop)
        )
    ):
        return None
    return loop.iter.args[0].value


def _count_appends(
    statements: list[ast.stmt],
    trips: int,
    lanes: Counter[str],
    sites: Counter[str],
) -> None:
    """Add to ``lanes`` the values the ``name.append(value)`` statements of
    ``statements`` collect per list when the statements run ``trips`` times,
    and to ``sites`` the number of those statements.  The statements of a
    nested ``cutlass.range_constexpr`` loop count once per trip of it."""
    for statement in statements:
        if isinstance(statement, ast.For):
            nested = _constexpr_trips(statement)
            if nested is not None:
                _count_appends(statement.body, trips * nested, lanes, sites)
            continue
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        call = statement.value
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.attr == "append"
            and len(call.args) == 1
            and not call.keywords
        ):
            lanes[call.func.value.id] += trips
            sites[call.func.value.id] += 1


def _collector_widths(loop: ast.For, widths: dict[str, int]) -> dict[str, int]:
    """The lanes each fresh list of ``widths`` holds after ``loop``.

    A plain packet appends one value per iteration; a component packet's
    sibling stores append one each (``CuteComponentGroup``), directly or in a
    nested constexpr loop over the components.  A list the loop names in any
    other way has no known width.
    """
    trips = _constexpr_trips(loop)
    if trips is None:
        return {}
    names = Counter(node.id for node in ast.walk(loop) if isinstance(node, ast.Name))
    lanes: Counter[str] = Counter()
    sites: Counter[str] = Counter()
    _count_appends(loop.body, trips, lanes, sites)
    return {
        name: count
        for name, count in lanes.items()
        if widths.get(name) == 0 and names[name] == sites[name]
    }


def lower_packet_store_policy(
    body: list[ast.stmt],
    *,
    policy: str,
    global_tensor_names: AbstractSet[str],
    target_device_capability: tuple[int, int] | None,
    cuda_version: str | None,
) -> list[ast.stmt]:
    """Rename full 16-byte global packet flushes to the selected policy's form.

    The packet emitters already proved alignment, coverage and the scalar
    tail; this only swaps the final helper.  A flush's width comes from a
    fresh ``[]`` filled by the ``cutlass.range_constexpr(N)`` loop of the
    same statement list, which appends to it once per iteration, or several
    times for a component packet: one append per sibling store, or one in a
    nested constexpr loop over the components (``_collector_widths``).  It
    may also be a constant slice of such a list, the wide-lane emitter's
    several packets per site.  Any other use of the list keeps the default
    store.  Flushes the async-store policy already renamed are left alone.
    """
    helpers = PACKET_FLUSH_BY_POLICY.get(policy)
    if helpers is None or (
        policy == "l2_evict_last"
        and not fixed_l2_evict_last_store_policy_supported(
            target_device_capability, cuda_version
        )
    ):
        return body
    # A rebound tensor parameter no longer names its global argument.
    global_tensor_names = frozenset(global_tensor_names) - {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    def split(call: ast.Call, collector: str, lanes: int) -> list[ast.stmt]:
        """Two 16-byte policy flushes for a 32-byte collector (a lane loop of
        16 bf16 or 8 fp32 lanes flushed whole by the hoisted-vector emitter):
        ``helper(ptr, vals[:half])`` and ``helper(ptr + half, vals[half:])``,
        the pointer advanced in elements like the wide-lane emitter does."""
        assert isinstance(call.func, ast.Name)
        half = lanes // 2
        pointer = ast.unparse(call.args[0])
        helper = helpers[call.func.id]
        return [
            ast.parse(f"{helper}({pointer}, {collector}[0:{half}])").body[0],
            ast.parse(
                f"{helper}(({pointer}) + {half}, {collector}[{half}:{lanes}])"
            ).body[0],
        ]

    def rewrite(statements: list[ast.stmt], inherited: dict[str, int]) -> None:
        widths = dict(inherited)
        result: list[ast.stmt] = []
        # Only a list emptied in this statement list may seed a collector: an
        # enclosing unrolled loop could already have appended to an outer one.
        fresh: set[str] = set()
        for statement in statements:
            replacement: list[ast.stmt] = [statement]
            keep: set[str] = set()
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.List)
                and not any(
                    isinstance(value, ast.Starred) for value in statement.value.elts
                )
            ):
                name = statement.targets[0].id
                widths[name] = len(statement.value.elts)
                keep.add(name)
                fresh.add(name)
            if isinstance(statement, ast.For):
                collected = _collector_widths(
                    statement, {name: widths[name] for name in fresh & widths.keys()}
                )
                rewrite(statement.body, widths)
                rewrite(statement.orelse, widths)
                widths.update(collected)
                keep.update(collected)
            elif isinstance(statement, ast.If):
                rewrite(statement.body, widths)
                rewrite(statement.orelse, widths)
            elif isinstance(statement, ast.With):
                rewrite(statement.body, widths)
            store = _store_call(statement)
            if store is not None:
                call, collector, lanes = store
                assert isinstance(call.func, ast.Name)
                keep.add(collector)
                width = widths.get(collector, 0)
                if lanes is not None:
                    width = lanes if lanes <= width else 0
                itemsize = PACKET_STORE_BYTES[call.func.id]
                if width * itemsize == 16 and _global_pointer(
                    call.args[0], global_tensor_names
                ):
                    call.func.id = helpers[call.func.id]
                elif (
                    width * itemsize == 32
                    and lanes is None
                    and itemsize > 1
                    and _global_pointer(call.args[0], global_tensor_names)
                ):
                    replacement = split(call, collector, width)
            used = {
                node.id for node in ast.walk(statement) if isinstance(node, ast.Name)
            }
            for name in used - keep:
                widths.pop(name, None)
            result.extend(replacement)
        statements[:] = result

    rewrite(body, {})
    return body
