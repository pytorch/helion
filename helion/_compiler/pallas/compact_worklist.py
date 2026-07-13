"""Compile-time plan + resolver for the ``compact_worklist`` Pallas loop type.

Recognises the supported jagged loop nest, captures each axis's ``(base,
length)`` as resolvable host AST, and renders the per-kernel ``jnp`` gathers
that feed :func:`helion.runtime.compact_worklist.flatten_worklist`.

The supported loop shape is::

    for source in hl.grid(B):  # owner_grid / source coordinate (block size 1)
        compact_start = offsets[source]  # prologue scalar assigns
        compact_end = offsets[source + 1]
        for tile_m in hl.tile(compact_start, compact_end):  # compact_tile
            acc = init(...)  # optional carried-state init
            for tile_k in hl.tile(
                ordered_start, ordered_end
            ):  # ordered  (optional, carries acc)
                acc = update(acc, ...)
            out[tile_m] = finalize(acc)  # store in the compact region

Detection rejects anything outside this shape with ``exc.InvalidConfig`` so the
autotuner scores an offered-but-unmatched config ``inf`` and skips it (rather
than aborting), while an explicit hardcoded ``compact_worklist`` surfaces the
clear error.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..ast_read_writes import ReadWrites
from ..ast_read_writes import ast_rename

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from ..device_ir import GraphInfo
    from ..host_function import HostFunction


# ---------------------------------------------------------------------------
# Plan dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Axis:
    """One axis of the compacted loop nest (one type per role).

    ``base``/``length`` are host-evaluable ``ast.AST`` trees whose only free
    names are the owner loop var(s) and host tensors/constants -- they render,
    via :func:`resolve_for_worklist`, into the ``jnp`` ``base``/``length`` (and
    dependent ``dep_base``/``dep_len``) args of ``flatten_worklist``.
    """

    kind: str  # "owner_grid" (block size 1) | "compact_tile" | "ordered"
    block_id: int
    loop_var: str
    base: ast.AST
    length: ast.AST
    block_size_var: str  # "1" for owner_grid; the codegen block-size var otherwise
    # Name of the single offset tensor ``T`` when this axis's bounds are the exact
    # packed-consecutive idiom ``base=T[g], end=T[g+1]`` (so ``T[i+1]-T[i]`` is
    # EXACTLY the per-source length), else None.  Set from ``_packed_consecutive`` at
    # construction; resident caching uses it to prove the resident-window overflow
    # guard (``max(diff(T))``) can't under-count the range.
    packed_offset_arg: str | None = None


@dataclasses.dataclass(frozen=True)
class TensorPolicy:
    """Per-tensor access policy, derived from the indexing structure.

    ``kind`` is one of ``compact_aligned_load`` (input indexed by the compact
    tile), ``compact_exact_store`` (output indexed by the compact tile),
    ``owner_indexed`` (dense per-owner tensor, e.g. ``x[source]``),
    ``ordered_reduction`` (the reduction's reused operand, indexed by the ordered
    inner tile on the leading dim, e.g. ``x[tile_k]``), or
    ``static_full`` (unaffected).

    ``kind`` is *classification only*: ``ordered_reduction`` means "indexed by the
    ordered axis on the leading dim".  If the range/window proof succeeds, every
    such operand can be made resident in a per-range VMEM window.  Optional prep
    caching is detected later from the tiled FX graph.
    """

    arg_name: str
    # compact_aligned_load|compact_exact_store|owner_indexed|
    # ordered_reduction|static_full
    kind: str


@dataclasses.dataclass(frozen=True)
class CompactWorklistPlan:
    """The full compaction plan for one kernel.

    The metadata-arg order is *not* stored: it is derived on demand from
    ``axes`` + ``tensor_policies`` by :func:`metadata_arg_names`, the single
    source of truth shared by builder emission, BlockSpec index_map params, the
    kernel signature, and the launcher.
    """

    axes: tuple[Axis, ...]  # owner_grid + compact_tile + optional ordered
    tensor_policies: tuple[TensorPolicy, ...]
    upper_bound_expr: str  # static UPPER = cdiv(total, BLOCK) + owners - 1
    # jnp host expression for the owner count P (the grid size), e.g.
    # "offsets.shape[0] - 1" or "lo.shape[0]"; taken from the owner hl.grid
    # bound (NOT assumed to be base.shape[0] - 1, which only holds for the
    # offsets-array idiom).
    num_owners_expr: str = ""

    @property
    def owner_axis(self) -> Axis:
        return next(a for a in self.axes if a.kind == "owner_grid")

    @property
    def compact_axis(self) -> Axis:
        return next(a for a in self.axes if a.kind == "compact_tile")

    @property
    def ordered_axis(self) -> Axis | None:
        return next((a for a in self.axes if a.kind == "ordered"), None)


# ---------------------------------------------------------------------------
# Metadata arg order (single source of truth)
# ---------------------------------------------------------------------------


def _strip_tile_prefix(loop_var: str) -> str:
    """Derive a metadata-name prefix from a tile loop var (``tile_q`` -> ``q``)."""
    for prefix in ("tile_", "tile"):
        if loop_var.startswith(prefix) and len(loop_var) > len(prefix):
            return loop_var[len(prefix) :].lstrip("_") or loop_var
    return loop_var


def owner_ref_name(plan: CompactWorklistPlan) -> str:
    owner = plan.owner_axis.loop_var
    for suffix in ("_idx", "_id"):
        if owner.endswith(suffix) and len(owner) > len(suffix):
            owner = owner[: -len(suffix)]
            break
    return f"work_{owner}"


def compact_ref_names(plan: CompactWorklistPlan) -> tuple[str, str]:
    """(tile_starts ref, tile_extents ref), e.g. ``("m_begin", "m_extent")``."""
    prefix = _strip_tile_prefix(plan.compact_axis.loop_var)
    return f"{prefix}_begin", f"{prefix}_extent"


def ordered_ref_names(plan: CompactWorklistPlan) -> tuple[str, str]:
    """(range_start ref, range_len ref), e.g. ``("k_begin", "k_len")``."""
    ordered = plan.ordered_axis
    assert ordered is not None
    prefix = _strip_tile_prefix(ordered.loop_var)
    return f"{prefix}_begin", f"{prefix}_len"


def resident_ordered_entries(plan: CompactWorklistPlan) -> tuple[TensorPolicy, ...]:
    """Ordered operands eligible for a range-keyed resident VMEM window."""
    return tuple(p for p in plan.tensor_policies if p.kind == "ordered_reduction")


def ordered_resident_bound_arg(plan: CompactWorklistPlan) -> str | None:
    """The offset array whose consecutive differences are EXACTLY the per-source
    ordered (reduction) lengths -- i.e. the ordered bound is the proven
    packed-consecutive idiom ``T[g]``/``T[g+1]`` (recorded as
    ``Axis.packed_offset_arg``) -- or ``None`` otherwise.

    Resident caching requires this so the resident window can be bound-checked at
    launch: the launcher raises if ``max(T[i+1]-T[i])`` exceeds the compile-time
    window size.  A merely single-tensor-but-not-consecutive bound (e.g.
    ``T[g+1]+128``) does NOT qualify -- ``max(diff(T))`` would under-count the
    true range and could let it over-read the window -- so it returns None and the
    ordered loop streams.
    """
    ordered = plan.ordered_axis
    return None if ordered is None else ordered.packed_offset_arg


@dataclasses.dataclass(frozen=True)
class ResidentCacheRangeSpec:
    """The runtime proof for the resident ordered window.

    ``ordered_offset_arg`` supplies the ordered lengths. ``compact_offset_arg``
    supplies the active-owner mask: owners with no compact work produce no
    worklist item, so their ordered lengths are ignored.
    """

    ordered_offset_arg: str
    compact_offset_arg: str


@dataclasses.dataclass(frozen=True)
class ResidentCacheDecision:
    """The single 'is resident caching active, and on which operands' decision.

    Every codegen + runtime path that emits resident-window behavior must consume
    this cached decision and gate on ``active``.
    """

    resident_operands: tuple[str, ...]
    range_spec: ResidentCacheRangeSpec | None
    physical_window: int
    inactive_reason: str | None
    # Semantic metadata fields that identify the cached contents.  Store fields,
    # not generated ref names, so emitters resolve through the same metadata order
    # used by launcher args and BlockSpecs.
    resident_key_fields: tuple[str, ...] = ()
    prep_key_fields: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return self.inactive_reason is None


def build_resident_cache_decision(
    plan: CompactWorklistPlan,
    operands: list[tuple[tuple[int, ...], int]],
    *,
    physical_window: int,
) -> ResidentCacheDecision:
    """Finalize resident-cache eligibility for one concrete config.

    Detection records semantic proofs on the plan; backend setup computes the VMEM
    window once from concrete shapes/block sizes and stores the result here.  All
    later consumers read this decision instead of recomputing eligibility/window
    math from partially-overlapping inputs.
    """
    residents = tuple(p.arg_name for p in resident_ordered_entries(plan))
    if not residents:
        return ResidentCacheDecision(residents, None, 0, "no resident operands")
    ordered_arg = ordered_resident_bound_arg(plan)
    if ordered_arg is None:
        return ResidentCacheDecision(residents, None, 0, "ordered bound is not packed")
    compact_arg = plan.compact_axis.packed_offset_arg
    if compact_arg is None:
        return ResidentCacheDecision(residents, None, 0, "compact bound is not packed")
    if not operands:
        return ResidentCacheDecision(residents, None, 0, "no resident operands")
    if physical_window <= 0:
        return ResidentCacheDecision(
            residents,
            None,
            0,
            "VMEM budget cannot hold one ordered block",
        )
    return ResidentCacheDecision(
        resident_operands=residents,
        range_spec=ResidentCacheRangeSpec(
            ordered_offset_arg=ordered_arg,
            compact_offset_arg=compact_arg,
        ),
        physical_window=physical_window,
        inactive_reason=None,
        resident_key_fields=("range_start",),
        prep_key_fields=("range_start", "range_len"),
    )


@dataclasses.dataclass(frozen=True)
class ResidentPrepHoist:
    """Direct range-invariant prep of a resident ordered operand."""

    graph_id: int
    host_arg: str
    load_node_name: str
    prep_node_name: str
    perm: tuple[int, ...]


def _ordered_full_slice_load(
    node: torch.fx.Node,
    ordered_block_id: int,
) -> bool:
    """True for the strict PR1 load shape ``arg[ordered_tile, :, :, ...]``."""
    from .plan_tiling import ArbitrarySlicePattern
    from .plan_tiling import TilePattern

    load_val = node.meta.get("val")
    if not isinstance(load_val, torch.Tensor):
        return False
    patterns = node.meta.get("indexing_patterns")
    if not isinstance(patterns, list):
        return False
    if len(patterns) != load_val.ndim:
        return False
    first, *rest = patterns
    if not (isinstance(first, TilePattern) and first.block_id == ordered_block_id):
        return False
    return all(
        isinstance(pattern, ArbitrarySlicePattern)
        and pattern.slice == slice(None, None, None)
        for pattern in rest
    )


def _prep_perm_from_node(prep: torch.fx.Node, load_ndim: int) -> tuple[int, ...] | None:
    if prep.target is torch.ops.aten.permute.default:
        raw = prep.args[1]
        if not isinstance(raw, (list, tuple)):
            return None
        if not all(isinstance(v, int) for v in raw):
            return None
        raw_perm = cast("tuple[int, ...]", tuple(raw))
        perm = tuple(v % load_ndim for v in raw_perm)
    else:
        return None
    if len(perm) != load_ndim or sorted(perm) != list(range(load_ndim)):
        return None
    if perm.index(0) == 0:
        return None
    return perm


def detect_resident_prep_hoists(
    graphs: list[GraphInfo],
    plan: CompactWorklistPlan,
) -> tuple[ResidentPrepHoist, ...]:
    """Find direct transpose-like preps of resident ordered loads.

    This is semantic detection only: it must not perform VMEM budgeting or choose
    the resident window size.  ``build_resident_cache_admission`` owns admission
    of these descriptors after the resident-window budget is known.
    """
    from ...language import memory_ops
    from ...language._tracing_ops import _host_tensor
    from ..device_ir import ForLoopGraphInfo

    ordered_axis = plan.ordered_axis
    if ordered_axis is None:
        return ()
    ordered_block_id = ordered_axis.block_id
    resident_hosts = {p.arg_name for p in resident_ordered_entries(plan)}
    if not resident_hosts:
        return ()

    by_graph_host: dict[tuple[int, str], list[ResidentPrepHoist]] = {}
    for graph_info in graphs:
        if not (
            isinstance(graph_info, ForLoopGraphInfo)
            and graph_info.block_ids == [ordered_block_id]
        ):
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not memory_ops.load:
                continue
            tensor_node = node.args[0]
            if not (
                isinstance(tensor_node, torch.fx.Node)
                and tensor_node.op == "call_function"
                and tensor_node.target is _host_tensor
                and isinstance(tensor_node.args[0], str)
            ):
                continue
            host_arg = tensor_node.args[0]
            if host_arg not in resident_hosts:
                continue
            if not _ordered_full_slice_load(node, ordered_block_id):
                continue
            if len(node.users) != 1:
                continue
            prep = next(iter(node.users))
            if prep.op != "call_function":
                continue
            load_val = node.meta.get("val")
            if not isinstance(load_val, torch.Tensor):
                continue
            perm = _prep_perm_from_node(prep, load_val.ndim)
            if perm is None:
                continue
            hoist = ResidentPrepHoist(
                graph_id=graph_info.graph_id,
                host_arg=host_arg,
                load_node_name=node.name,
                prep_node_name=prep.name,
                perm=perm,
            )
            by_graph_host.setdefault((graph_info.graph_id, host_arg), []).append(hoist)

    admitted: list[ResidentPrepHoist] = []
    for hoists in by_graph_host.values():
        if len(hoists) == 1:
            admitted.append(hoists[0])
    return tuple(admitted)


@dataclasses.dataclass(frozen=True)
class ResidentCacheAdmission:
    """Resident-cache decision plus optional prep hoists admitted for a config."""

    decision: ResidentCacheDecision
    prep_hoists: tuple[ResidentPrepHoist, ...]


def _tensor_footprints(
    host_args: Mapping[str, object],
    arg_names: Iterable[str],
) -> list[tuple[tuple[int, ...], int]]:
    footprints: list[tuple[tuple[int, ...], int]] = []
    for arg_name in arg_names:
        arg = host_args[arg_name]
        assert isinstance(arg, torch.Tensor)
        footprints.append((tuple(int(s) for s in arg.shape), arg.dtype.itemsize))
    return footprints


def build_resident_cache_admission(
    graphs: list[GraphInfo],
    plan: CompactWorklistPlan,
    host_args: Mapping[str, object],
    *,
    ordered_block: int,
    vmem_bytes: int,
) -> ResidentCacheAdmission:
    """Admit resident caching and optional prep hoists for one concrete config.

    Backend setup supplies concrete config facts (host shapes, ordered block, VMEM
    budget). This helper owns the compact-worklist-specific policy: detect prep
    candidates, account for their scratch footprint, choose the one physical
    resident window, and drop optional prep when it cannot fit.
    """
    from ...runtime import compact_ordered_physical_window

    resident_operands = _tensor_footprints(
        host_args,
        tuple(policy.arg_name for policy in resident_ordered_entries(plan)),
    )
    prep_hoists = detect_resident_prep_hoists(graphs, plan)
    prep_operands = _tensor_footprints(
        host_args,
        tuple(hoist.host_arg for hoist in prep_hoists),
    )

    physical_window_no_prep = compact_ordered_physical_window(
        resident_operands,
        vmem_bytes,
        ordered_block,
        prep_operands=[],
    )
    physical_window = physical_window_no_prep
    admitted_hoists = prep_hoists
    if prep_hoists:
        physical_window_with_prep = compact_ordered_physical_window(
            resident_operands,
            vmem_bytes,
            ordered_block,
            prep_operands=prep_operands,
        )
        if physical_window_with_prep > 0:
            physical_window = physical_window_with_prep
        else:
            admitted_hoists = ()

    decision = build_resident_cache_decision(
        plan,
        resident_operands,
        physical_window=physical_window,
    )
    if not decision.active:
        admitted_hoists = ()
    return ResidentCacheAdmission(decision, tuple(admitted_hoists))


def elide_installed_prep_load_masks(
    graph: torch.fx.Graph,
    load_tail_fills: Mapping[str, float],
) -> None:
    """Drop the redundant per-tile OOB masks on loads whose prep cache was installed.

    ``load_tail_fills`` maps the load node name of each ACTUALLY-installed prep lowering
    to the value that lowering's refill writes into the cache's padded tail (its
    ``tail_fill_value``).  Because the refill already wrote that value there, a
    downstream ``_mask_to`` with the same fill is redundant; deleting it also lets
    Mosaic fold the transpose into the matmul push.

    Called from prep-lowering installation with only the lowerings that survived
    validation, so a prep that fell back to resident-only leaves its load's mask in
    place (correctness is never coupled to admission-time optimism).  Elision is keyed
    on the declared ``tail_fill_value``: a flash-style ``_mask_to(scores, -inf)`` (fill
    != the cache's tail fill, and downstream of the dot) is preserved automatically.
    Non-resident (streamed) deferred loads keep their unknown masked value untouched.
    """
    from ..host_function import HostFunction
    from ..node_masking import remove_unnecessary_masking

    if not load_tail_fills:
        return
    # ``remove_unnecessary_masking`` recomputes masked values, which for loop-carried
    # (phi) nodes walks the enclosing graphs via ``DeviceIR.current()``; make the device
    # IR current since prep-lowering install runs during codegen, outside the pass.
    with HostFunction.current().device_ir:
        for node in graph.nodes:
            fill = load_tail_fills.get(node.name)
            if fill is not None:
                # The refill wrote ``fill`` into the padded tail: declare it so the
                # matching-fill ``_mask_to`` downstream is judged redundant.
                node.meta["masked_value"] = fill
            elif (
                node.meta.get("masked_value") is None
                and "pallas_deferred_mask_block_ids" in node.meta
            ):
                # A non-resident deferred load: keep its unknown masked value so its
                # (still-needed) deferred mask is preserved.
                continue
            else:
                # Drop the stale cache so masked values recompute from the loads above.
                node.meta.pop("masked_value", None)
        remove_unnecessary_masking(graph)


def metadata_arg_names(plan: CompactWorklistPlan) -> list[str]:
    """The single source of metadata arg order.

    ``owner_ids`` is always included: the owner coordinate is recovered from it
    (``work_<owner>_ref[wid]``) so the owner-grid prologue scalar loads
    (``offsets[source]``) -- which are *not* DCE'd -- index a valid owner rather
    than the work id.  (Dropping ``owner_ids`` when nothing is ``owner_indexed``
    is a future optimization that first requires DCE'ing that prologue and the
    corresponding offsets device arg.)  Order mirrors ``CompactWorkMetadata``.
    """
    names: list[str] = [owner_ref_name(plan)]
    names.extend(compact_ref_names(plan))
    if plan.ordered_axis is not None:
        names.extend(ordered_ref_names(plan))
    return names


def metadata_field_names(plan: CompactWorklistPlan) -> list[str]:
    """``CompactWorkMetadata`` field names in ``metadata_arg_names`` order.

    Used by the launcher to select which metadata arrays become scalar-prefetch
    args; positionally aligned with :func:`metadata_arg_names`.
    """
    fields = ["owner_ids", "tile_starts", "tile_extents"]
    if plan.ordered_axis is not None:
        fields += ["range_start", "range_len"]
    return fields


def metadata_ref_for_field(plan: CompactWorklistPlan, field: str) -> str:
    """Return the emitted scalar-prefetch ref name for a metadata field."""
    try:
        index = metadata_field_names(plan).index(field)
    except ValueError as err:
        raise exc.InvalidConfig(
            "compact_worklist resident caching: active cache key metadata "
            f"{field!r} is missing."
        ) from err
    return f"{metadata_arg_names(plan)[index]}_ref"


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve_for_worklist(
    expr: ast.AST,
    subs: dict[str, str],
    *,
    leak_ok: set[str],
) -> ast.AST:
    """Substitute owner loop vars and leak-check a captured ``base``/``length``.

    ``subs`` maps each parallel loop var name to its coordinate-array name
    (``{"source": "parent"}`` for the builder, where ``parent = jnp.arange(P)``);
    written as a general transformer so the base-in-index idiom (grouped GEMM)
    is a purely additive follow-up.

    The leak-check defines the supported set: after substitution the only free
    names allowed are the substituted array names and the host names in
    ``leak_ok`` (host tensors / host constants, validated by the caller via
    origins).  Any reference to the ordered axis, a body-local value, or a
    device origin raises :class:`exc.InvalidConfig` -- the autotuner-skippable
    type -- so an offered-but-unsupported config is scored ``inf`` and skipped.

    Returns a fresh AST (the input is not mutated).  The rendered
    ``offsets[<index over owner_array>]`` form is valid in both torch and jnp,
    so it drops straight into the generated ``jnp`` ``_build_worklist``.
    """
    resolved = ast_rename(copy.deepcopy(expr), subs)
    reads = set(ReadWrites.from_ast(resolved).reads)
    allowed = set(subs.values()) | leak_ok
    leaked = reads - allowed
    if leaked:
        raise exc.InvalidConfig(
            "compact_worklist: loop bound is not host-evaluable; it references "
            f"{sorted(leaked)} which are neither the owner coordinate nor host "
            "tensors/constants. Only bounds affine in the parallel coordinates "
            "over host offset tensors are supported."
        )
    return resolved


# ---------------------------------------------------------------------------
# Generated jnp ``_build_worklist``
# ---------------------------------------------------------------------------


def _offset_tensor_name(resolved_base: ast.AST) -> str:
    """The offsets tensor name in a resolved bounds-carry base ``offsets[parent]``."""
    for node in ast.walk(resolved_base):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return node.value.id
    raise exc.InvalidConfig(
        "compact_worklist: could not identify the offsets tensor of the compact "
        "bound (expected a bounds-carry idiom like offsets[owner])."
    )


def _bound_tensors(plan: CompactWorklistPlan) -> set[str]:
    """Host offset tensor names appearing in any captured bound."""
    names: set[str] = set()
    for axis in plan.axes:
        names |= _names_read(axis.base) | _names_read(axis.length)
    return names


def _free_host_names_in_order(nodes: list[ast.AST], owner_array: str) -> list[str]:
    """Free names (excluding ``owner_array``) across ``nodes``, first-seen order.

    Used to collect *every* host name the builder must take as a parameter --
    not just the base's offsets tensor, so distinct begin/end tensors
    (``lo[g]``/``hi[g]``) and affine host scalars (``g*S``) are all included.
    """
    seen: list[str] = []
    for node in nodes:
        for name in _read_names_in_source_order(node):
            if name != owner_array and name not in seen:
                seen.append(name)
    return seen


def _read_names_in_source_order(node: ast.AST) -> list[str]:
    """Load-context Name ids in left-to-right source order."""
    # ast.walk is BFS; sort by source column for determinism where available.
    names_with_pos = [
        (getattr(s, "col_offset", 0), s.id)
        for s in ast.walk(node)
        if isinstance(s, ast.Name) and isinstance(s.ctx, ast.Load)
    ]
    names_with_pos.sort()
    out: list[str] = []
    for _, name in names_with_pos:
        if name not in out:
            out.append(name)
    return out


def render_build_worklist(
    plan: CompactWorklistPlan,
    *,
    block_expr: str,
    upper_expr: str,
    builder_name: str = "_build_worklist",
    owner_array: str = "parent",
) -> tuple[str, list[str]]:
    """Render the module-level ``jnp`` ``_build_worklist`` source.

    Returns ``(source, offset_params)``: the function source, and the ordered
    list of offsets-tensor parameter names it takes (one per axis in PR1's
    bounds-carry idiom).  The body resolves the captured ``base``/``length``
    (and dependent ``dep_base``/``dep_len``) over ``parent = jnp.arange(P)`` and
    calls the library :func:`flatten_worklist`; nothing is ``.item()``-ed, so
    the returned ``num_work`` stays a traced ``jax.Array``.

    ``block_expr``/``upper_expr`` are host expressions (static ints at codegen);
    they are textual so the caller can pass either literals (tests) or the
    codegen block-size var / ``program_id.num_pids_expr``.
    """
    leak_ok = _bound_tensors(plan)
    subs = {plan.owner_axis.loop_var: owner_array}
    compact = plan.compact_axis

    base_resolved = resolve_for_worklist(compact.base, subs, leak_ok=leak_ok)
    length_resolved = resolve_for_worklist(compact.length, subs, leak_ok=leak_ok)
    base_src = ast.unparse(base_resolved)
    length_src = ast.unparse(length_resolved)

    # The builder needs EVERY host name referenced by any bound as a parameter
    # (not just the base's offsets tensor) -- distinct begin/end tensors
    # (lo[g]/hi[g]) and affine host scalars (g*S) must all be passed in.
    resolved_nodes: list[ast.AST] = [base_resolved, length_resolved]
    # Owner count from the captured hl.grid bound (robust across idioms); fall
    # back to the offsets-array form only if a plan predates the field.
    num_owners_expr = (
        plan.num_owners_expr or f"{_offset_tensor_name(base_resolved)}.shape[0] - 1"
    )

    dep_lines: list[str] = []
    dep_kwargs = ""
    ordered = plan.ordered_axis
    if ordered is not None:
        dep_base_resolved = resolve_for_worklist(ordered.base, subs, leak_ok=leak_ok)
        dep_len_resolved = resolve_for_worklist(ordered.length, subs, leak_ok=leak_ok)
        resolved_nodes += [dep_base_resolved, dep_len_resolved]
        dep_lines = [
            f"    dep_base = {ast.unparse(dep_base_resolved)}",
            f"    dep_len = {ast.unparse(dep_len_resolved)}",
        ]
        dep_kwargs = ", dep_base=dep_base, dep_len=dep_len"

    # The owner-count expression (e.g. "offsets.shape[0] - 1" or "B") may
    # introduce additional free host names the builder must take as params.
    resolved_nodes.append(_plain(num_owners_expr))
    offset_params = _free_host_names_in_order(resolved_nodes, owner_array)

    lines = [
        f"def {builder_name}({', '.join(offset_params)}):",
        "    import jax.numpy as jnp",
        "    from helion.runtime.compact_worklist import flatten_worklist",
        f"    {owner_array} = jnp.arange({num_owners_expr}, dtype=jnp.int32)",
        f"    base = {base_src}",
        f"    length = {length_src}",
        *dep_lines,
        (
            "    return flatten_worklist("
            f"base, length, {block_expr}, {upper_expr}{dep_kwargs})"
        ),
    ]
    return "\n".join(lines), offset_params


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _to_plain(node: ast.AST) -> ast.AST:
    """Strip ExtendedAST/location by round-tripping through source.

    Captured host bounds (``offsets[source + 1]``) come from ``host_fn.body`` as
    ExtendedAST nodes, which are not ``copy.deepcopy``-able (they require a
    ``_location`` kwarg).  The resolver and builder need plain, copyable ASTs;
    the bounds are simple host expressions, so an unparse/parse round-trip is
    lossless.
    """
    return ast.parse(ast.unparse(node), mode="eval").body


def _plain(src: str) -> ast.AST:
    return ast.parse(src, mode="eval").body


def _torch_size_to_jnp_shape(node: ast.AST) -> ast.AST:
    """Rewrite torch ``x.size(i)`` calls to jnp-valid ``x.shape[i]`` subscripts.

    The owner count comes from the kernel's host ``hl.grid`` bound, which is
    written in torch (``offsets.size(0) - 1``); the builder runs on jax
    arrays, so it must read ``.shape[i]`` instead.
    """

    class _T(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "size"
                and len(node.args) == 1
            ):
                return ast.Subscript(
                    value=ast.Attribute(value=func.value, attr="shape", ctx=ast.Load()),
                    slice=node.args[0],
                    ctx=ast.Load(),
                )
            return node

    return ast.fix_missing_locations(_T().visit(node))


def _loop_iter_call(node: ast.For) -> ast.Call | None:
    """The ``hl.grid(...)``/``hl.tile(...)`` call driving a for-loop, or None."""
    it = node.iter
    if isinstance(it, ast.Call):
        return it
    return None


def _iter_func_name(call: ast.Call) -> str | None:
    """Return ``"grid"``/``"tile"``/... for an ``hl.<name>(...)`` iterator."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _names_read(node: ast.AST) -> set[str]:
    return set(ReadWrites.from_ast(node).reads)


def _names_written(nodes: list[ast.stmt]) -> set[str]:
    written: set[str] = set()
    for stmt in nodes:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                written.add(sub.id)
    return written


def _collect_prologue_assigns(
    body: list[ast.stmt], *, before: ast.stmt | None = None
) -> dict[str, ast.AST]:
    """Map ``name -> rhs AST`` for simple top-level ``name = expr`` assigns.

    Used to inline loop-prologue scalar loads (``start = offsets[source]``) so
    a captured bound resolves to a host expression over the owner var.

    Only statements that appear *before* ``before`` (the statement that is or
    contains the selected loop) are scanned -- a reassignment of a bound variable
    *after* the loop must NOT override the prologue binding the loop actually saw.
    """
    out: dict[str, ast.AST] = {}
    for stmt in body:
        if before is not None and any(node is before for node in ast.walk(stmt)):
            break
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            out[stmt.targets[0].id] = _to_plain(stmt.value)
    return out


def _inline(expr: ast.AST, prologue: dict[str, ast.AST]) -> ast.AST:
    """Recursively inline prologue-assigned names into ``expr`` (host bounds).

    ``expr`` and the prologue values must already be plain (copyable) ASTs.
    """
    expr = copy.deepcopy(expr)

    class _Inliner(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id in prologue:
                return self.visit(copy.deepcopy(prologue[node.id]))
            return node

    return _Inliner().visit(expr)


def _tile_loops(grid_body: list[ast.stmt]) -> list[ast.For]:
    """Direct-nested ``hl.tile`` for-loops, outermost first (PR1: chain of <=2)."""
    loops: list[ast.For] = []
    body = grid_body
    while True:
        found = None
        for stmt in body:
            if isinstance(stmt, ast.For):
                call = _loop_iter_call(stmt)
                if call is not None and _iter_func_name(call) == "tile":
                    found = stmt
                    break
        if found is None:
            break
        loops.append(found)
        body = found.body
    return loops


def _length_ast(begin: ast.AST, end: ast.AST) -> ast.AST:
    """Build the ``end - begin`` length expression (bounds-carry idiom)."""
    return _plain(f"({ast.unparse(end)}) - ({ast.unparse(begin)})")


def _packed_consecutive(begin: ast.AST, end: ast.AST, owner_var: str) -> bool:
    """True if ``(begin, end)`` are ``T[e]`` / ``T[e+1]`` on the SAME tensor ``T``.

    This is the packed-offsets idiom: owner ranges are contiguous and
    non-overlapping (``sum(length) == total``), so the tight megablocks UPPER
    bound ``cdiv(total, block) + owners - 1`` provably bounds ``num_work``.
    Distinct begin/end tensors or non-consecutive indices may overlap or exceed
    ``total``, so they must fall back to a conservative UPPER.

    Conservatively returns ``False`` on anything it can't prove (the caller then
    over-allocates metadata, which is safe; a too-small UPPER would not be).
    """

    def _same_ast(a: ast.AST, b: ast.AST) -> bool:
        return ast.dump(a, include_attributes=False) == ast.dump(
            b, include_attributes=False
        )

    def _owner_plus_const(node: ast.AST) -> int | None:
        """Return k for exactly ``owner_var + k`` forms, else None."""
        if isinstance(node, ast.Name) and node.id == owner_var:
            return 0
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                if (
                    isinstance(node.right, ast.Constant)
                    and type(node.right.value) is int
                ):
                    base = _owner_plus_const(node.left)
                    return None if base is None else base + node.right.value
                if isinstance(node.left, ast.Constant) and type(node.left.value) is int:
                    base = _owner_plus_const(node.right)
                    return None if base is None else node.left.value + base
            if (
                isinstance(node.op, ast.Sub)
                and isinstance(node.right, ast.Constant)
                and type(node.right.value) is int
            ):
                base = _owner_plus_const(node.left)
                return None if base is None else base - node.right.value
        return None

    if not (isinstance(begin, ast.Subscript) and isinstance(end, ast.Subscript)):
        return False
    if not _same_ast(begin.value, end.value):
        return False
    b = _owner_plus_const(begin.slice)
    e = _owner_plus_const(end.slice)
    return b is not None and e is not None and e == b + 1


def _block_id_of_loop(loop: ast.For) -> int | None:
    """Best-effort block id from the loop iterator's attached type info."""
    call = _loop_iter_call(loop)
    if call is None:
        return None
    type_info = getattr(call, "_type_info", None)
    inner = getattr(type_info, "inner", None)
    block_id = getattr(inner, "block_id", None)
    return block_id if isinstance(block_id, int) else None


def _find_grid_loop(body: list[ast.stmt]) -> ast.For | None:
    for stmt in body:
        if isinstance(stmt, ast.For):
            call = _loop_iter_call(stmt)
            if call is not None and _iter_func_name(call) == "grid":
                return stmt
    return None


def _classify_tensor_policies(
    grid_loop: ast.For,
    owner_var: str,
    compact_var: str,
    ordered_var: str | None,
    bound_tensors: set[str],
) -> tuple[TensorPolicy, ...]:
    """Derive per-tensor policies from how each subscript is indexed.

    A first-dim index by the compact var -> ``compact_aligned_load`` (read) /
    ``compact_exact_store`` (write); by the owner var -> ``owner_indexed``; by
    the ordered var -> handled by the fori body (no BlockSpec policy).  A
    compacted-axis access fitting no policy is rejected.

    ``bound_tensors`` (the host offset tensors that appear in the captured
    bounds) are skipped: they are consumed by the host builder, not the device
    kernel, so they carry no BlockSpec policy.
    """
    policies: dict[str, TensorPolicy] = {}
    for sub in ast.walk(grid_loop):
        if not isinstance(sub, ast.Subscript) or not isinstance(sub.value, ast.Name):
            continue
        name = sub.value.id
        if name in bound_tensors:
            continue
        index = sub.slice
        dims = list(index.elts) if isinstance(index, ast.Tuple) else [index]
        is_store = isinstance(sub.ctx, ast.Store)

        # Find which dim(s) each compacted loop var indexes.
        compact_dims = [d for d, e in enumerate(dims) if compact_var in _names_read(e)]
        owner_dims = [d for d, e in enumerate(dims) if owner_var in _names_read(e)]
        ordered_dims = (
            [d for d, e in enumerate(dims) if ordered_var in _names_read(e)]
            if ordered_var is not None
            else []
        )
        if is_store and ordered_dims:
            raise exc.InvalidConfig(
                f"compact_worklist: tensor {name!r} is stored with the ordered "
                "axis in its index; ordered-axis stores are not supported by "
                "the resident-window lowering."
            )

        if compact_dims:
            # PR1 only supports compaction on the leading dim (dim 0): the
            # synthetic tile scope + exact DMA store assume row-major leading
            # compaction. A compact var in a non-leading dim is unsupported.
            if compact_dims != [0]:
                raise exc.InvalidConfig(
                    f"compact_worklist: tensor {name!r} is indexed by the compact "
                    f"tile on dim(s) {compact_dims}; only leading-dim (0) "
                    "compaction is supported."
                )
            kind = "compact_exact_store" if is_store else "compact_aligned_load"
            policies[name] = TensorPolicy(arg_name=name, kind=kind)
        elif owner_dims:
            if owner_dims != [0]:
                raise exc.InvalidConfig(
                    f"compact_worklist: tensor {name!r} is owner-indexed on dim(s) "
                    f"{owner_dims}; only leading-dim (0) owner indexing is supported."
                )
            policies.setdefault(name, TensorPolicy(arg_name=name, kind="owner_indexed"))
        elif ordered_dims:
            # Ordered-range tensor = the reduction's reused operand.  A leading-dim
            # ordered access is CLASSIFIED ``ordered_reduction`` ("indexed by the
            # ordered axis"); whether it becomes resident is decided later by the
            # range/window proof.  Optional prep caching is detected later from
            # tiled FX nodes.  A non-leading ordered index has no resident-window
            # form, so it stays ``static_full``.
            kind = "ordered_reduction" if ordered_dims == [0] else "static_full"
            policies.setdefault(name, TensorPolicy(arg_name=name, kind=kind))
    return tuple(policies.values())


def _host_evaluable_names(host_fn: HostFunction) -> set[str]:
    """Names that resolve to host-available values (kernel inputs / host consts).

    The builder runs on the host (inside the launcher jit), so a captured bound
    may only reference host-origin values.  We take the kernel parameter names
    (host inputs) plus any local whose ``TypeInfo.origin`` is host
    (``Origin.is_host()``); a device-origin value (computed in the kernel body)
    is absent and therefore rejected.
    """
    names = {a.arg for a in host_fn.args.args}
    local_types = host_fn.local_types
    if local_types:
        for name, type_info in local_types.items():
            origin = getattr(type_info, "origin", None)
            if origin is not None and origin.is_host():
                names.add(name)
    return names


def _validate_host_bounds(
    host_fn: HostFunction,
    axes: list[Axis],
    owner_var: str,
    num_owners_expr: str,
) -> None:
    """Reject bounds that reference anything but the owner coord + host tensors.

    This is the real host/device check behind ``resolve_for_worklist``'s
    leak-set: a bound like ``hl.tile(0, device_value)`` (device-origin) or one
    over a body-local is rejected with ``exc.InvalidConfig``.

    Every remaining free name becomes a ``_build_worklist`` parameter, and the
    launcher maps builder args to *tensor* positions only -- so a name must be a
    **1-D host tensor**.  A scalar host arg (e.g. ``B`` in ``hl.grid(B)``, which
    would feed ``jnp.arange(B)``) has no tensor position and is rejected here
    rather than ``KeyError``-ing in the launcher.  ``num_owners_expr`` is checked
    too, since its free names are builder params via ``jnp.arange(...)``.
    """
    host_names = _host_evaluable_names(host_fn)
    params = host_fn.params.arguments
    nodes: list[ast.AST] = []
    for axis in axes:
        if axis.kind == "owner_grid":
            continue
        nodes += [axis.base, axis.length]
    if num_owners_expr:
        nodes.append(ast.parse(num_owners_expr, mode="eval").body)
    for node in nodes:
        for name in _names_read(node):
            if name == owner_var:
                continue
            if name not in host_names:
                raise exc.InvalidConfig(
                    f"compact_worklist: loop bound references {name!r}, which "
                    "is not host-evaluable (only the owner coordinate and "
                    "host inputs/constants are allowed). Data-dependent or "
                    "device-origin bounds are unsupported."
                )
            # Every builder param must be a 1-D offsets-like tensor: indexing it
            # by the source coordinate yields a per-source scalar.  Reject non-tensors (scalar
            # host args can't be a builder tensor position) and multi-dim data
            # tensors (e.g. q[g] -> [H, D] is not a valid bound source).
            value = params.get(name)
            if not isinstance(value, torch.Tensor):
                raise exc.InvalidConfig(
                    f"compact_worklist: the worklist builder would take {name!r} "
                    "as a parameter, but only 1-D host tensors are supported as "
                    "builder args (scalar host args / non-tensor bounds like "
                    "hl.grid(B) are unsupported)."
                )
            if value.ndim != 1:
                raise exc.InvalidConfig(
                    f"compact_worklist: loop bound indexes {name!r} "
                    f"(ndim={value.ndim}); only 1-D offsets-like tensors are "
                    "valid bound sources, not multi-dim data tensors."
                )


def detect_compact_worklist_plan(
    host_fn: HostFunction,
) -> CompactWorklistPlan:
    """Recognise the compactable nest and build a :class:`CompactWorklistPlan`.

    Raises :class:`exc.InvalidConfig` (autotuner-skippable) when the pattern is
    not the supported one: exactly one owner ``hl.grid``, one jagged compact
    ``hl.tile``, at most one ordered (loop-carried) inner ``hl.tile``, and a
    store in the compact region (not ordered-indexed).

    NOTE: detection runs off the host-source AST (``host_fn.body``), not the
    lowered device IR.  This is robust for the target nests but brittle to source
    variations (e.g. extra intermediate variables or a differently-written loop
    nest); such cases raise ``InvalidConfig`` and degrade to "compact not
    offered" rather than miscompiling -- which is also why the autotuner-gating
    safety net (only append compact when the pattern matches) matters.
    """
    device_ir = host_fn.device_ir
    if len(device_ir.grid_block_ids) != 1:
        raise exc.InvalidConfig(
            "compact_worklist requires exactly one owner hl.grid; found "
            f"{len(device_ir.grid_block_ids)}."
        )

    grid_loop = _find_grid_loop(host_fn.body)
    if grid_loop is None or not isinstance(grid_loop.target, ast.Name):
        raise exc.InvalidConfig(
            "compact_worklist could not locate the owner hl.grid loop in source."
        )
    owner_var = grid_loop.target.id

    # Owner count = the hl.grid bound (e.g. offsets.size(0) - 1 or lo.size(0)),
    # inlined over top-level assigns and translated to jnp .shape[i].
    grid_call = _loop_iter_call(grid_loop)
    if grid_call is None or not grid_call.args:
        raise exc.InvalidConfig("compact_worklist: owner hl.grid has no size arg.")
    top_level = _collect_prologue_assigns(host_fn.body, before=grid_loop)
    num_owners_ast = _torch_size_to_jnp_shape(
        _inline(_to_plain(grid_call.args[0]), top_level)
    )
    num_owners_expr = ast.unparse(num_owners_ast)

    tile_loops = _tile_loops(grid_loop.body)
    if not tile_loops:
        raise exc.InvalidConfig(
            "compact_worklist requires a jagged hl.tile inside the owner grid."
        )
    if len(tile_loops) > 2:
        raise exc.InvalidConfig(
            "compact_worklist supports at most a compact tile + one ordered "
            f"inner tile; found {len(tile_loops)} nested hl.tile loops "
            "(3+-level ragged products are out of scope)."
        )

    compact_loop = tile_loops[0]
    ordered_loop = tile_loops[1] if len(tile_loops) == 2 else None
    if not isinstance(compact_loop.target, ast.Name):
        raise exc.InvalidConfig("compact_worklist: compact tile var is not a Name.")
    compact_var = compact_loop.target.id
    ordered_var: str | None = None

    # An inner tile loop is only supported when it is the ordered (carried-state)
    # axis: a name assigned in its body, read in its body, and read after it.
    if ordered_loop is not None:
        if not isinstance(ordered_loop.target, ast.Name):
            raise exc.InvalidConfig("compact_worklist: ordered tile var is not a Name.")
        ordered_var = ordered_loop.target.id
        inner_written = _names_written(ordered_loop.body)
        inner_read = set()
        for stmt in ordered_loop.body:
            inner_read |= _names_read(stmt)
        after = _after_loop_reads(compact_loop.body, ordered_loop)
        carried = inner_written & inner_read & after
        if not carried:
            raise exc.InvalidConfig(
                "compact_worklist: the inner hl.tile carries no state, so it is "
                "a second parallel axis (multi-level ragged product) -- out of "
                "scope. Only an ordered carried-state inner loop is supported."
            )

    # Prologue scalar assigns in the grid body BEFORE the compact tile loop
    # (start = offsets[source], ...); a later reassignment must not leak in.
    prologue = _collect_prologue_assigns(grid_loop.body, before=compact_loop)

    # Compact axis (bounds-carry): base = begin, length = end - begin.
    compact_call = _loop_iter_call(compact_loop)
    if compact_call is None or len(compact_call.args) < 2:
        raise exc.InvalidConfig(
            "compact_worklist: compact hl.tile must have data-dependent "
            "(begin, end) bounds."
        )
    c_begin = _inline(_to_plain(compact_call.args[0]), prologue)
    c_end = _inline(_to_plain(compact_call.args[1]), prologue)
    # STORE-SAFETY GATE: the compact store is a masked full-block write; a partial
    # last tile spills masked-zero rows into a neighbor, repaired only because the
    # neighbor is a LATER work item under "arbitrary" serialization -- which holds
    # only when work-item order == row order, i.e. owner offsets are MONOTONIC.
    # The packed-offsets idiom (begin=T[g], end=T[g+1], same T) is the static
    # guarantee of that (contiguous cumsum).  Distinct/non-consecutive bounds
    # (e.g. lo[g]/hi[g]) can be non-monotonic, so one owner's spill could clobber
    # an EARLIER owner's valid rows with nothing to repair it -- reject them (PR1's
    # kernels are all packed; non-packed needs a different, exact store).
    if not _packed_consecutive(c_begin, c_end, owner_var):
        raise exc.InvalidConfig(
            "compact_worklist: compact tile bounds are not the packed-offsets "
            "idiom (begin=T[g], end=T[g+1] on one tensor). The masked full-block "
            "store is only correct for monotonic owner offsets; non-packed or "
            "possibly-overlapping bounds are unsupported (use fori_loop / "
            "emit_pipeline)."
        )
    compact_packed_arg = (
        c_begin.value.id
        if isinstance(c_begin, ast.Subscript) and isinstance(c_begin.value, ast.Name)
        else None
    )

    owner_block_id = device_ir.grid_block_ids[0][0]
    compact_block_id = _block_id_of_loop(compact_loop)
    if compact_block_id is None:
        raise exc.InvalidConfig(
            "compact_worklist: could not resolve the compact tile block id."
        )

    axes: list[Axis] = [
        Axis(
            kind="owner_grid",
            block_id=owner_block_id,
            loop_var=owner_var,
            base=_plain("0"),
            length=_plain("1"),
            block_size_var="1",
        ),
        Axis(
            kind="compact_tile",
            block_id=compact_block_id,
            loop_var=compact_var,
            base=c_begin,
            length=_length_ast(c_begin, c_end),
            block_size_var="",  # filled at codegen from the config block size
            packed_offset_arg=compact_packed_arg,
        ),
    ]

    if ordered_loop is not None:
        ordered_call = _loop_iter_call(ordered_loop)
        if ordered_call is None or len(ordered_call.args) < 2:
            raise exc.InvalidConfig(
                "compact_worklist: ordered hl.tile must have (begin, end) bounds."
            )
        # The ordered bounds (ordered_start = offsets[source], ...) may be assigned in
        # the grid body (captured by `prologue`) or inside the compact loop before
        # the ordered loop -- add the latter, again stopping before the loop.
        ordered_prologue = {
            **prologue,
            **_collect_prologue_assigns(compact_loop.body, before=ordered_loop),
        }
        o_begin = _inline(_to_plain(ordered_call.args[0]), ordered_prologue)
        o_end = _inline(_to_plain(ordered_call.args[1]), ordered_prologue)
        ordered_block_id = _block_id_of_loop(ordered_loop)
        if ordered_block_id is None:
            raise exc.InvalidConfig(
                "compact_worklist: could not resolve the ordered tile block id."
            )
        # Resident caching needs a bound whose consecutive diffs are EXACTLY the
        # per-source length; prove the packed-consecutive idiom (base=T[g],
        # end=T[g+1] on one Name tensor).  A bound like ``T[g+1]+128`` reads only
        # ``T`` but is NOT this shape, so it does not qualify.
        ordered_packed_arg = (
            o_begin.value.id
            if _packed_consecutive(o_begin, o_end, owner_var)
            and isinstance(o_begin, ast.Subscript)
            and isinstance(o_begin.value, ast.Name)
            else None
        )
        axes.append(
            Axis(
                kind="ordered",
                block_id=ordered_block_id,
                loop_var=ordered_var or "",
                base=o_begin,
                length=_length_ast(o_begin, o_end),
                block_size_var="",
                packed_offset_arg=ordered_packed_arg,
            )
        )

    # Reject bounds that aren't host-evaluable (device-origin / body-local) or
    # whose builder params aren't 1-D host tensors (incl. num_owners free names).
    _validate_host_bounds(host_fn, axes, owner_var, num_owners_expr)

    # Host offset tensors appearing in the captured bounds are builder inputs,
    # not device tensors -> exclude them from the BlockSpec policies.
    bound_tensors: set[str] = set()
    for axis in axes:
        bound_tensors |= _names_read(axis.base) | _names_read(axis.length)

    policies = _classify_tensor_policies(
        grid_loop, owner_var, compact_var, ordered_var, bound_tensors
    )

    # The store must be in the compact region (compact_exact_store), never
    # ordered-indexed.
    if not any(p.kind == "compact_exact_store" for p in policies):
        raise exc.InvalidConfig(
            "compact_worklist: no output store indexed by the compact tile; the "
            "store must live in the compact (parallel) region."
        )

    return CompactWorklistPlan(
        axes=tuple(axes),
        tensor_policies=policies,
        upper_bound_expr="",  # finalized at codegen via program_id.num_pids_expr
        num_owners_expr=num_owners_expr,
    )


def _after_loop_reads(body: list[ast.stmt], loop: ast.For) -> set[str]:
    """Names read in ``body`` *after* the given inner ``loop`` statement."""
    reads: set[str] = set()
    seen = False
    for stmt in body:
        if stmt is loop:
            seen = True
            continue
        if seen:
            reads |= _names_read(stmt)
    return reads
