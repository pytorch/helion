"""Host wrapper emitters of the tcgen05 GEMM family.

The A/B operand TMA atoms and tiled MMA of a tcgen05 GEMM (``tcgen05_ab_tma``
and its TF32 raw-RNA / TMA-RN grouped variants), the epilogue TMA store and
auxiliary-input load atoms (``tcgen05_d_tma`` / ``tcgen05_aux_tma``) and the
static persistent grouped scheduler's parameters and grid
(``tcgen05_grouped_static_persistent``).
"""

from __future__ import annotations

from typing import cast

from .... import exc
from ...._compiler.cute.device_state import Tcgen05GroupedSchedulerMode
from ...._compiler.cute.grouped_row_union import (
    ab_stages_supported as row_union_ab_stages_supported,
)
from ...._compiler.cute.grouped_row_union import (
    schedule_by_name as row_union_schedule_by_name,
)
from ...._compiler.cute.grouped_row_union import (
    with_ab_stages as row_union_with_ab_stages,
)
from ...._compiler.cute.strategies import tcgen05_default_epilogue_tile_expr
from ...._compiler.cute.strategies import tcgen05_explicit_d_store_tile_expr
from ...._compiler.cute.strategies import tcgen05_smem_layout_expr
from ...._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
)
from ..plan_facts import tcgen05_grouped_dynamic_ab_tensormap_rank
from ..plan_facts import tcgen05_grouped_scheduler_mode
from ..plan_facts import tcgen05_plan_orientation
from .context import HostWrapperContext
from .context import HostWrapperEmitter
from .grouped_static import grouped_static_active_clusters
from .grouped_static import grouped_static_quotas


def append_epilogue_tma_wrapper(
    ctx: HostWrapperContext,
    *,
    tensor_idx: int,
    bm: int,
    bn: int,
    stage_count: int,
    dtype: str,
    kernel_args: list[str],
    copy_op: str,
    epi_tile_m: int | None = None,
    epi_tile_n: int | None = None,
    d_store_box_n: int | None = None,
    epi_tile_raw_expr: str | None = None,
    tensor_name: str | None = None,
    rank3_mnl_tensor: bool = False,
    orientation: str = "mn",
    column_major: bool = False,
    bounded_output_shift: bool = False,
) -> None:
    assert len(kernel_args) == 2
    assert orientation in ("mn", "nm")
    worklist_nm_store = orientation == "nm"
    d_store_layout = (
        "cutlass.utils.layout.LayoutEnum.COL_MAJOR"
        if worklist_nm_store or column_major
        else "cutlass.utils.layout.LayoutEnum.ROW_MAJOR"
    )
    tensor_expr = tensor_name if tensor_name is not None else f"arg{tensor_idx}"
    explicit_epi_tile = any(
        value is not None for value in (epi_tile_m, epi_tile_n, d_store_box_n)
    )
    if epi_tile_raw_expr is not None:
        # The bm=128 CtaGroup.TWO family threads the device-exact (N-mode
        # permuted) epilogue-tile expression verbatim so the host TMA-store
        # atom is built from the same layout the device r2s copy writes
        # through. The plain ``epi_tile_m/n`` integer keys cannot express
        # the permutation. See ``tcgen05_two_cta_m128_epilogue_tile_expr``.
        assert not explicit_epi_tile
        epi_tile_expr = epi_tile_raw_expr
    elif explicit_epi_tile:
        checked_epi_tile_m = ctx.require_positive_int(epi_tile_m, "epi_tile_m")
        checked_epi_tile_n = ctx.require_positive_int(epi_tile_n, "epi_tile_n")
        checked_d_store_box_n = ctx.require_positive_int(d_store_box_n, "d_store_box_n")
        assert checked_epi_tile_n == checked_d_store_box_n
        epi_tile_expr = tcgen05_explicit_d_store_tile_expr(
            checked_epi_tile_m, checked_d_store_box_n
        )
    else:
        epi_tile_expr = tcgen05_default_epilogue_tile_expr(
            bm,
            bn,
            dtype,
            c_layout=d_store_layout,
        )
    tma_atom, tma_tensor = kernel_args
    epi_tile = f"{tma_atom}_epi_tile"
    smem_layout = f"{tma_atom}_smem_layout"
    cta_v_layout = f"{tma_atom}_cta_v_layout"
    gmem_tensor = (
        f"{tma_atom}_d_nm"
        if worklist_nm_store and rank3_mnl_tensor
        else f"{tma_atom}_gmem_mnl"
        if rank3_mnl_tensor
        else tensor_expr
    )
    cta_tiler_expr = (
        epi_tile
        if rank3_mnl_tensor
        else f"cute.composition(cute.make_identity_layout({gmem_tensor}.shape), {epi_tile})"
    )
    rank3_gmem_shape = (
        f"(arg{tensor_idx}_shape1, arg{tensor_idx}_shape0, 1)"
        if worklist_nm_store
        else f"(arg{tensor_idx}_shape0, arg{tensor_idx}_shape1, 1)"
    )
    rank3_gmem_stride = (
        f"(arg{tensor_idx}_stride1, arg{tensor_idx}_stride0, 0)"
        if worklist_nm_store
        else f"(arg{tensor_idx}_stride0, arg{tensor_idx}_stride1, 0)"
    )
    gmem_pointer = f"arg{tensor_idx}.iterator"
    if bounded_output_shift:
        if not rank3_mnl_tensor or column_major or ctx.descriptors is not None:
            raise exc.BackendUnsupported(
                "cute", "bounded D shift requires its proved rank-3 output view"
            )
        extra_extent = f"arg{tensor_idx}_shape0 + 1"
        rank3_gmem_shape = (
            f"(arg{tensor_idx}_shape1, arg{tensor_idx}_shape0, {extra_extent})"
            if worklist_nm_store
            else f"(arg{tensor_idx}_shape0, arg{tensor_idx}_shape1, {extra_extent})"
        )
        rank3_gmem_stride = (
            f"(arg{tensor_idx}_stride1, arg{tensor_idx}_stride0, arg{tensor_idx}_stride0)"
            if worklist_nm_store
            else f"(arg{tensor_idx}_stride0, arg{tensor_idx}_stride1, arg{tensor_idx}_stride0)"
        )
        gmem_pointer = (
            f"(arg{tensor_idx}.iterator - cutlass.Int64(arg{tensor_idx}_shape0) "
            f"* cutlass.Int64(arg{tensor_idx}_stride0))"
        )
    gmem_layout = (
        ctx.descriptors.layout(f"arg{tensor_idx}", output=True)
        if ctx.descriptors is not None
        else f"{rank3_gmem_shape}, stride={rank3_gmem_stride}"
    )
    # Keep these layout arguments in sync with the device-side
    # ``make_smem_layout_epi`` calls; the wrapper's TMA atom and the kernel's
    # SMEM staging must slice the same epilogue tile shape.
    ctx.body.extend(
        (
            f"    {epi_tile} = {epi_tile_expr}",
            (
                f"    {smem_layout} = cutlass.utils.blackwell_helpers."
                "make_smem_layout_epi("
                f"{dtype}, {d_store_layout}, "
                f"{epi_tile}, {stage_count})"
            ),
            *(
                (
                    (
                        f"    {gmem_tensor} = cute.make_tensor("
                        f"{gmem_pointer}, "
                        "layout=cute.make_layout("
                        f"{gmem_layout}))"
                    ),
                )
                if rank3_mnl_tensor
                else ()
            ),
            (f"    {cta_v_layout} = {cta_tiler_expr}"),
            (
                f"    {tma_atom}, {tma_tensor} = "
                "cute.nvgpu.cpasync.make_tiled_tma_atom("
                f"{copy_op}, "
                f"{gmem_tensor}, cute.slice_({smem_layout}, (None, None, 0)), "
                f"{cta_v_layout})"
            ),
        )
    )
    ctx.call_args.extend(kernel_args)


def emit_d_tma(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``tcgen05_d_tma``: the epilogue's TMA store atom over the output."""
    d_idx = ctx.plan_int("d_idx")
    bm = ctx.plan_int("bm")
    bn = ctx.plan_int("bn")
    c_stage_count = ctx.plan_int("c_stage_count")
    output_dtype = str(plan["output_dtype"])
    kernel_args = [str(arg) for arg in cast("list[object]", plan["kernel_args"])]
    d_tensor_name = None
    hier_layout = plan.get("d_hier_layout")
    if hier_layout is not None:
        # A fused view/permute store (``radix_store``): the TMA store
        # addresses the destination through its proved hierarchical
        # [M, N] layout instead of the argument's own shape.
        hier_shape, hier_stride = cast("list[str]", hier_layout)
        d_tensor_name = f"{kernel_args[0]}_d_hier"
        ctx.body.append(
            f"    {d_tensor_name} = cute.make_tensor(arg{d_idx}.iterator, "
            f"layout=cute.make_layout({hier_shape}, stride={hier_stride}))"
        )
    elif bool(plan.get("d_leading_passthrough")):
        d_tensor_name = f"{kernel_args[0]}_d_tma"
        ctx.append_permuted_cute_tensor_view(d_tensor_name, d_idx, (1, 2, 0))
    append_epilogue_tma_wrapper(
        ctx,
        tensor_idx=d_idx,
        bm=bm,
        bn=bn,
        stage_count=c_stage_count,
        dtype=output_dtype,
        kernel_args=kernel_args,
        copy_op="cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()",
        epi_tile_m=ctx.plan_optional_int("epi_tile_m"),
        epi_tile_n=ctx.plan_optional_int("epi_tile_n"),
        d_store_box_n=ctx.plan_optional_int("d_store_box_n"),
        epi_tile_raw_expr=ctx.plan_optional_str("epi_tile_raw_expr"),
        tensor_name=d_tensor_name,
        rank3_mnl_tensor=bool(plan.get("rank3_mnl_tensor")),
        orientation=tcgen05_plan_orientation(plan),
        column_major=bool(plan.get("d_column_major")),
        bounded_output_shift=bool(plan.get("bounded_output_shift")),
    )


def emit_aux_tma(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``tcgen05_aux_tma``: the epilogue's TMA load atom over an auxiliary input."""
    c_idx = ctx.plan_int("c_idx")
    bm = ctx.plan_int("bm")
    bn = ctx.plan_int("bn")
    stage_count = ctx.plan_int("stage_count")
    input_dtype = str(plan["input_dtype"])
    kernel_args = [str(arg) for arg in cast("list[object]", plan["kernel_args"])]
    append_epilogue_tma_wrapper(
        ctx,
        tensor_idx=c_idx,
        bm=bm,
        bn=bn,
        stage_count=stage_count,
        dtype=input_dtype,
        kernel_args=kernel_args,
        copy_op="cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()",
    )


def emit_grouped_static_persistent(
    ctx: HostWrapperContext, plan: dict[str, object]
) -> None:
    """``tcgen05_grouped_static_persistent``: the static persistent scheduler's parameters, quotas and grid."""
    assert ctx.num_sm is not None and ctx.num_sm > 0
    sched_params_arg = str(plan["sched_params_arg"])
    total_clusters_arg = str(plan["total_clusters_arg"])
    cluster_m = ctx.plan_int("cluster_m", 1)
    cluster_n = ctx.plan_int("cluster_n", 1)
    reserved_sms = ctx.plan_int(TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY, 0)
    max_active_clusters = grouped_static_active_clusters(
        num_sm=ctx.num_sm,
        cluster_m=max(1, cluster_m),
        reserved_sms=reserved_sms,
        num_sm_multiplier=ctx.plan_int("num_sm_multiplier", 1),
        cluster_n=cluster_n,
    )
    quota_args = tuple(
        str(arg)
        for arg in cast("tuple[object, ...]", plan.get("static_group_quota_args", ()))
    )
    quotas: tuple[int, ...] = ()
    scheduler_mode = tcgen05_grouped_scheduler_mode(plan)
    runtime_direct_tile_table = scheduler_mode in (
        Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT,
        Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
    )
    runtime_direct_clc = scheduler_mode is Tcgen05GroupedSchedulerMode.RUNTIME_CLC
    if quota_args:
        problem_shapes = cast(
            "tuple[tuple[int, int, int], ...]", plan["static_problem_shapes"]
        )
        assert len(quota_args) == len(problem_shapes)
        assert tcgen05_plan_orientation(plan) != "nm"
        assert cluster_m == 1 and cluster_n == 1
        quotas = grouped_static_quotas(
            problem_shapes,
            bm=ctx.plan_int("bm"),
            bn=ctx.plan_int("bn"),
            bk=ctx.plan_int("bk"),
            max_active_clusters=max_active_clusters,
        )
    if runtime_direct_clc:
        # The host table is exact: one real output record per z cluster, with
        # no padded/phantom tail. Launching the full request grid lets CLC
        # cancel each pending z cluster at most once and makes response bidz
        # the record index consumed identically by every CTA in the cluster.
        ctx.body.extend(
            (
                f"    grid_x = cutlass.Int32({cluster_m})",
                f"    grid_y = cutlass.Int32({cluster_n})",
                f"    grid_z = {total_clusters_arg}",
            )
        )
    else:
        ctx.body.append(
            f"    {sched_params_arg} = cutlass.utils.PersistentTileSchedulerParams("
            f"({cluster_m}, {cluster_n}, {total_clusters_arg}), "
            f"({cluster_m}, {cluster_n}, 1))"
        )
        ctx.body.extend(
            (
                (
                    "    _tcgen05_grouped_grid = "
                    "cutlass.utils.StaticPersistentGroupTileScheduler.get_grid_shape("
                    f"{sched_params_arg}, cutlass.Int32({max_active_clusters}))"
                ),
                "    grid_x = _tcgen05_grouped_grid[0]",
                "    grid_y = _tcgen05_grouped_grid[1]",
                "    grid_z = _tcgen05_grouped_grid[2]",
            )
        )
        if not runtime_direct_tile_table:
            ctx.call_args.append(sched_params_arg)
    if runtime_direct_tile_table:
        ctx.call_args.append(total_clusters_arg)
    ctx.call_args.extend(f"cutlass.Int32({quota})" for quota in quotas)


def emit_ab_tma(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``tcgen05_ab_tma`` / ``tcgen05_grouped_rna`` / ``tcgen05_grouped_tma_rn``:
    the tiled MMA, the A/B SMEM layouts and the A/B TMA atoms."""
    kind = plan["kind"]
    lhs_idx = ctx.plan_int("lhs_idx")
    rhs_idx = ctx.plan_int("rhs_idx")
    bm = ctx.plan_int("bm")
    bn = ctx.plan_int("bn")
    bk = ctx.plan_int("bk")
    cluster_m = ctx.plan_int("cluster_m", 1)
    cluster_n = ctx.plan_int("cluster_n", 1)
    input_dtype = str(plan["input_dtype"])
    acc_dtype = str(plan["acc_dtype"])
    # fp32 GMEM operands run the MMA as tf32: the A/B cute tensors keep their
    # torch-derived Float32 element type, and the TMA descriptors recast to
    # the plan's TFloat32 via ``internal_type`` (both are 4 bytes wide, so the
    # SMEM layout/byte math is unchanged). Mirrors quack's gemm_sm100 fp32
    # handling.
    if kind == "tcgen05_grouped_rna":
        # RNA conversion consumes raw FP32 words after TMA completion. Keep
        # the descriptor in FP32: TMA rounding would change tie/NaN semantics.
        if (
            input_dtype != "cutlass.TFloat32"
            or cluster_m != 1
            or cluster_n != 1
            or plan.get("operand_transform") != "tf32_rna"
        ):
            raise exc.BackendUnsupported(
                "cute", "raw RNA descriptors require one-CTA TF32 MMA"
            )
        tma_internal_type_arg = ", internal_type=cutlass.Float32"
    elif kind == "tcgen05_grouped_tma_rn":
        if (
            input_dtype != "cutlass.TFloat32"
            or cluster_m != 1
            or cluster_n != 1
            or plan.get("operand_transform") != "tf32_tma_rn"
            or type(plan.get("converter_warps")) is not int
            or plan["converter_warps"] != 0
            or plan.get("scheduler_self_consumer") is not False
        ):
            raise exc.BackendUnsupported(
                "cute", "TF32 TMA RN requires its converter-free one-CTA schedule"
            )
        tma_internal_type_arg = ", internal_type=cutlass.TFloat32"
    else:
        tma_internal_type_arg = (
            ", internal_type=cutlass.TFloat32"
            if input_dtype == "cutlass.TFloat32"
            else ""
        )
    ab_stage_count = ctx.plan_int("ab_stage_count", 2)
    # Optional ``smem_swizzle_*`` overrides recorded by the device-side
    # codegen when the user opts into a non-default A/B SMEM atom
    # swizzle. When absent the wrapper emits the legacy
    # ``make_smem_layout_a/b`` calls. The no-override wrapper markers
    # are covered by the focused tcgen05 SMEM-swizzle codegen test.
    smem_swizzle_a_raw = plan.get("smem_swizzle_a")
    smem_swizzle_b_raw = plan.get("smem_swizzle_b")
    smem_swizzle_a: int | None = (
        int(smem_swizzle_a_raw) if isinstance(smem_swizzle_a_raw, int) else None
    )
    smem_swizzle_b: int | None = (
        int(smem_swizzle_b_raw) if isinstance(smem_swizzle_b_raw, int) else None
    )
    a_k_major = bool(plan.get("a_k_major", True))
    # K-major (column-major / K-contiguous) B. Absent on the MN-major
    # (row-major B) default path.
    b_k_major = bool(plan.get("b_k_major"))
    lhs_tma_order = ctx.plan_optional_order("lhs_tma_order")
    rhs_tma_order = ctx.plan_optional_order("rhs_tma_order")
    rhs_rank3_grouped_nt = bool(plan.get("rhs_rank3_grouped_nt"))
    lhs_rank3_grouped_nt = bool(plan.get("lhs_rank3_grouped_nt"))
    shared_rhs = bool(plan.get("shared_rhs"))
    orientation = tcgen05_plan_orientation(plan)
    swapped_nm = orientation == "nm"
    dynamic_ab_tensormaps = bool(plan.get("dynamic_ab_tensormaps"))
    fixed_ab_tensormaps = bool(plan.get("fixed_ab_tensormaps"))
    full_allocation_a = bool(plan.get("full_allocation_a"))
    full_allocation_b = bool(plan.get("full_allocation_b"))
    full_allocation_b_rank3 = bool(plan.get("full_allocation_b_rank3"))
    fixed_grouped_b_rank3 = bool(plan.get("fixed_grouped_b_rank3"))
    if fixed_grouped_b_rank3 and not fixed_ab_tensormaps:
        raise exc.BackendUnsupported(
            "cute", "rank-3 immutable grouped B requires fixed A/B TensorMaps"
        )
    if fixed_ab_tensormaps and (not swapped_nm or dynamic_ab_tensormaps):
        raise exc.BackendUnsupported(
            "cute",
            "fixed full-allocation A/B TensorMaps require the N,M worklist "
            "orientation and cannot also be dynamic",
        )
    row_union_schedule = plan.get("row_union_schedule")
    row_profile = row_union_schedule_by_name(row_union_schedule)
    if row_union_schedule is not None and row_profile is None:
        raise exc.BackendUnsupported("cute", "unknown row-union physical descriptor")
    if row_profile is not None:
        row_union_ab_stages = plan.get("row_union_ab_stages", row_profile.ab_stages)
        if not row_union_ab_stages_supported(row_profile, row_union_ab_stages):
            raise exc.BackendUnsupported(
                "cute", "row-union physical descriptor ring depth"
            )
        row_profile = row_union_with_ab_stages(row_profile, row_union_ab_stages)
        expected = {
            "row_union_schedule": row_profile.name,
            "bm": row_profile.mma_m,
            "bn": row_profile.mma_n,
            "bk": row_profile.block_k,
            "orientation": "nm",
            "cluster_m": row_profile.cluster_m,
            "cluster_n": row_profile.cluster_n,
            "ab_stage_count": row_profile.ab_stages,
            "a_producer_partition": row_profile.producer_cluster,
            "lhs_tma_order": (1, 0),
            "rhs_tma_order": (0, 1),
            "input_dtype": "cutlass.BFloat16",
            "acc_dtype": "cutlass.Float32",
            "a_k_major": False,
            "b_k_major": True,
        }
        if any(plan.get(key) != value for key, value in expected.items()) or (
            dynamic_ab_tensormaps or fixed_ab_tensormaps or shared_rhs
        ):
            raise exc.BackendUnsupported(
                "cute", "row-union physical descriptor contract"
            )
    if swapped_nm and not (dynamic_ab_tensormaps or fixed_ab_tensormaps or row_profile):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 N,M-oriented A/B TensorMaps require dynamic per-group or "
            "fixed full-allocation descriptors",
        )
    dynamic_ab_tensormap_rank2 = (
        dynamic_ab_tensormaps and tcgen05_grouped_dynamic_ab_tensormap_rank(plan) == 2
    )
    immutable_mn_profile = (
        not swapped_nm
        and dynamic_ab_tensormap_rank2
        and not lhs_rank3_grouped_nt
        and rhs_rank3_grouped_nt
        and a_k_major
        and b_k_major
        and not shared_rhs
        and cluster_m == 2
        and cluster_n in (1, 2)
        and bm in (128, 256)
        and bn in (128, 192, 224, 256)
        and plan.get("source_m_tile") == bm
        and plan.get("device_split_sizes") is True
        and plan.get("device_layout_kind") in ("offsets", "split_sizes")
    )
    if full_allocation_a and not (
        immutable_mn_profile
        or (
            swapped_nm
            and dynamic_ab_tensormap_rank2
            and lhs_rank3_grouped_nt
            and not shared_rhs
            and a_k_major
            and ((cluster_m == 1 and bm in (64, 128)) or (cluster_m == 2 and bm == 256))
            and cluster_n == 1
        )
    ):
        raise exc.BackendUnsupported(
            "cute", "full-allocation A requires a K-major grouped N,M input"
        )
    if full_allocation_b and not immutable_mn_profile:
        raise exc.BackendUnsupported(
            "cute", "immutable grouped physical B requires the device M,N profile"
        )
    if full_allocation_b_rank3 and not (full_allocation_b and immutable_mn_profile):
        raise exc.BackendUnsupported(
            "cute", "rank-3 immutable physical B requires its proved M,N input mode"
        )
    if shared_rhs and (
        not swapped_nm
        or not dynamic_ab_tensormap_rank2
        or lhs_rank3_grouped_nt
        or rhs_rank3_grouped_nt
        or fixed_ab_tensormaps
    ):
        raise exc.BackendUnsupported(
            "cute", "shared rank-2 RHS requires the N,M dynamic rank-2 worklist route"
        )
    kernel_args = [str(arg) for arg in cast("list[object]", plan["kernel_args"])]
    assert len(kernel_args) == 4
    tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = kernel_args

    # CtaGroup.TWO is selected when ``cluster_m == 2 and bm == 256`` —
    # the V=2 path. ``cluster_n`` extends the cluster along the N axis
    # but does not change the V dimension. Cycle 26's
    # ``cluster_m * cluster_n == 2`` test happened to work for
    # cluster_m=2 cluster_n=1 but rejects the canonical Quack-best
    # cluster_m=2 cluster_n=2 4-CTA cluster (product=4). Use
    # ``cluster_m == 2`` directly so cluster_n=2 keeps CtaGroup.TWO.
    #
    # The bm=128 CtaGroup.TWO family (fp8 small-grid) cannot be derived from
    # ``bm == 256`` here, so the device codegen records the resolved decision
    # on the plan as ``use_2cta_instrs``. Honor it when present; fall back to
    # the legacy bm==256 derivation for golden-stable older plans.
    plan_use_2cta = plan.get("use_2cta_instrs")
    if plan_use_2cta is not None:
        assert isinstance(plan_use_2cta, bool)
        use_2cta_instrs = plan_use_2cta
    else:
        use_2cta_instrs = cluster_m == 2 and bm == 256
    cta_group = (
        "cute.nvgpu.tcgen05.CtaGroup.TWO"
        if use_2cta_instrs
        else "cute.nvgpu.tcgen05.CtaGroup.ONE"
    )
    cluster_shape = f"({cluster_m}, {cluster_n}, 1)"
    tiled_mma = f"{tma_atom_a}_tiled_mma"
    cluster_layout_vmnk = f"{tma_atom_a}_cluster_layout_vmnk"
    smem_a_layout = f"{tma_atom_a}_smem_layout"
    smem_b_layout = f"{tma_atom_b}_smem_layout"
    lhs_tma = f"{tma_atom_a}_lhs_tma"
    lhs_tma_arg = (
        lhs_tma
        if (
            dynamic_ab_tensormaps
            or fixed_ab_tensormaps
            or swapped_nm
            or lhs_tma_order is not None
        )
        else f"arg{lhs_idx}"
    )
    rhs_tma = f"{tma_atom_b}_rhs_tma"
    if swapped_nm:
        if not (lhs_rank3_grouped_nt or shared_rhs or row_profile):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 N,M-oriented A/B TensorMaps require grouped rank-3 logical A",
            )
        if shared_rhs or row_profile is not None:
            # Physical A is the one shared logical B[K,N]. Its immutable
            # rank-2 descriptor is reused for every group; only the packed
            # logical A and output descriptors require per-group updates.
            lhs_tma_layout = (
                f"(arg{lhs_idx}_shape1, arg{lhs_idx}_shape0), "
                f"stride=(arg{lhs_idx}_stride1, arg{lhs_idx}_stride0)"
            )
        elif fixed_ab_tensormaps or full_allocation_a:
            if fixed_grouped_b_rank3:
                lhs_tma_layout = (
                    f"(arg{lhs_idx}_shape1, arg{lhs_idx}_shape2, "
                    f"arg{lhs_idx}_shape0), "
                    f"stride=(arg{lhs_idx}_stride1, arg{lhs_idx}_stride2, "
                    f"arg{lhs_idx}_stride0)"
                )
            else:
                # K-major grouped B can flatten to one immutable [G*N,K]
                # allocation. Device coordinates add the group base.
                lhs_tma_layout = (
                    f"(arg{lhs_idx}_shape0 * arg{lhs_idx}_shape1, "
                    f"arg{lhs_idx}_shape2), stride=(arg{lhs_idx}_stride1, "
                    f"arg{lhs_idx}_stride2)"
                )
        elif dynamic_ab_tensormap_rank2:
            lhs_tma_layout = (
                f"(arg{lhs_idx}_shape1, arg{lhs_idx}_shape2), "
                f"stride=(arg{lhs_idx}_stride1, arg{lhs_idx}_stride2)"
            )
        else:
            lhs_tma_layout = (
                f"(arg{lhs_idx}_shape1, arg{lhs_idx}_shape2, "
                f"arg{lhs_idx}_shape0), "
                f"stride=(arg{lhs_idx}_stride1, arg{lhs_idx}_stride2, "
                f"arg{lhs_idx}_stride0)"
            )
        if (
            fixed_ab_tensormaps
            or dynamic_ab_tensormap_rank2
            or not dynamic_ab_tensormaps
        ):
            rhs_tma_layout = (
                f"(arg{rhs_idx}_shape0, arg{rhs_idx}_shape1), "
                f"stride=(arg{rhs_idx}_stride0, arg{rhs_idx}_stride1)"
            )
        else:
            rhs_tma_layout = (
                f"(arg{rhs_idx}_shape0, arg{rhs_idx}_shape1, 1), "
                f"stride=(arg{rhs_idx}_stride0, arg{rhs_idx}_stride1, 0)"
            )
        if ctx.descriptors is not None:
            rhs_tma_layout = ctx.descriptors.layout(f"arg{rhs_idx}")
        lhs_tma_setup = (
            (
                f"    {lhs_tma} = cute.make_tensor("
                f"arg{lhs_idx}.iterator, "
                f"layout=cute.make_layout({lhs_tma_layout}))"
            ),
        )
        rhs_tma_setup = (
            (
                f"    {rhs_tma} = cute.make_tensor("
                f"arg{rhs_idx}.iterator, "
                f"layout=cute.make_layout({rhs_tma_layout}))"
            ),
        )
    else:
        rhs_physical_n = f"arg{rhs_idx}_shape1"
        if full_allocation_b and not full_allocation_b_rank3:
            rhs_physical_n = f"arg{rhs_idx}_shape0 * arg{rhs_idx}_shape1"
        lhs_tma_setup = (
            (
                (
                    f"    {lhs_tma} = cute.make_tensor("
                    f"arg{lhs_idx}.iterator, "
                    "layout=cute.make_layout("
                    + (
                        f"(arg{lhs_idx}_shape0, arg{lhs_idx}_shape1), "
                        f"stride=(arg{lhs_idx}_stride0, arg{lhs_idx}_stride1)))"
                        if dynamic_ab_tensormap_rank2
                        else (
                            f"(arg{lhs_idx}_shape0, arg{lhs_idx}_shape1, 1), "
                            f"stride=(arg{lhs_idx}_stride0, "
                            f"arg{lhs_idx}_stride1, 0)))"
                        )
                    )
                ),
            )
            if dynamic_ab_tensormaps
            else ()
        )
        rhs_tma_setup = (
            (
                f"    {rhs_tma} = cute.make_tensor("
                f"arg{rhs_idx}.iterator, "
                "layout=cute.make_layout("
                + (
                    f"({rhs_physical_n}, arg{rhs_idx}_shape2), "
                    f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride2)))"
                    if dynamic_ab_tensormap_rank2 and not full_allocation_b_rank3
                    else (
                        f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape2, "
                        f"arg{rhs_idx}_shape0), "
                        f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride2, "
                        f"arg{rhs_idx}_stride0)))"
                    )
                )
                if rhs_rank3_grouped_nt
                else f"    {rhs_tma} = cute.make_tensor("
                f"arg{rhs_idx}.iterator, "
                "layout=cute.make_layout("
                f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape0), "
                f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride0)))"
            ),
        )
    smem_a_layout_expr = tcgen05_smem_layout_expr(
        tiled_mma=tiled_mma,
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_str=input_dtype,
        num_stages=ab_stage_count,
        operand="a",
        swizzle_override=smem_swizzle_a,
        k_major=a_k_major,
    )
    smem_b_layout_expr = tcgen05_smem_layout_expr(
        tiled_mma=tiled_mma,
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_str=input_dtype,
        num_stages=ab_stage_count,
        operand="b",
        swizzle_override=smem_swizzle_b,
        k_major=b_k_major,
    )
    if lhs_tma_order is not None:
        ctx.append_permuted_cute_tensor_view(lhs_tma, lhs_idx, lhs_tma_order)
    if rhs_tma_order is not None:
        ctx.append_permuted_cute_tensor_view(rhs_tma, rhs_idx, rhs_tma_order)
    lhs_tma_setup_lines = () if lhs_tma_order is not None else lhs_tma_setup
    rhs_tma_setup_lines = () if rhs_tma_order is not None else rhs_tma_setup
    ctx.body.extend(
        (
            (
                f"    {tiled_mma} = cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
                f"{input_dtype}, "
                f"{input_dtype}, "
                + (
                    "cute.nvgpu.OperandMajorMode.K, "
                    if a_k_major
                    else "cute.nvgpu.OperandMajorMode.MN, "
                )
                + (
                    "cute.nvgpu.OperandMajorMode.K, "
                    if b_k_major
                    else "cute.nvgpu.OperandMajorMode.MN, "
                )
                + f"{acc_dtype}, "
                f"{cta_group}, "
                f"({bm}, {bn}), "
                "cute.nvgpu.tcgen05.OperandSource.SMEM)"
            ),
            (
                f"    {cluster_layout_vmnk} = cute.tiled_divide("
                f"cute.make_layout({cluster_shape}), ({tiled_mma}.thr_id.shape,))"
            ),
            f"    {smem_a_layout} = {smem_a_layout_expr}",
            f"    {smem_b_layout} = {smem_b_layout_expr}",
            *lhs_tma_setup_lines,
            *rhs_tma_setup_lines,
            # ``make_tiled_tma_atom_A`` vs ``_B`` asymmetry:
            # - ``_B`` always passes ``cluster_layout_vmnk.shape`` as
            #   its trailing arg (CuTe's signature for B requires the
            #   cluster shape; the cluster_m=1 cluster_n=1 case still
            #   passes the 1x1x1 shape harmlessly).
            # - ``_A`` only adds the same trailing arg when
            #   ``cluster_n > 1``. For the validated cluster_n=1
            #   paths, A's atom is constructed without the cluster
            #   shape while B still receives it. The asymmetry is
            #   intentional: A only needs the cluster shape when N
            #   multicast is active (cluster_n>1). The cluster_n=1
            #   form is pinned by
            #   ``test_tcgen05_role_local_monolithic_codegen_markers``.
            (
                f"    {tma_atom_a}, {tma_tensor_a} = cute.nvgpu.make_tiled_tma_atom_A("
                "cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A("
                f"{cluster_shape}, {tiled_mma}.thr_id), "
                f"{lhs_tma_arg}, "
                f"cute.slice_({smem_a_layout}, (None, None, None, 0)), "
                f"({bm}, {bn}, {bk}), {tiled_mma}"
                + (
                    f", cute.tiled_divide(cute.make_layout({row_profile.producer_cluster!r}), ({tiled_mma}.thr_id.shape,)).shape"
                    if row_profile is not None
                    else f", {cluster_layout_vmnk}.shape"
                    if cluster_n > 1
                    else ""
                )
                + tma_internal_type_arg
                + ")"
            ),
            # See the asymmetry comment above ``make_tiled_tma_atom_A``
            # for why ``_B`` always passes the cluster shape and ``_A``
            # only does at cluster_n>1.
            (
                f"    {tma_atom_b}, {tma_tensor_b} = cute.nvgpu.make_tiled_tma_atom_B("
                "cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_B("
                f"{cluster_shape}, {tiled_mma}.thr_id), "
                f"{rhs_tma}, "
                + (
                    f"cute.select({smem_b_layout}, mode=[0, 1, 2]), "
                    if row_profile is not None
                    else f"cute.slice_({smem_b_layout}, (None, None, None, 0)), "
                )
                + f"({bm}, {bn}, {bk}), {tiled_mma}, {cluster_layout_vmnk}.shape"
                f"{tma_internal_type_arg})"
            ),
        )
    )
    ctx.call_args.extend(kernel_args)


# Keep transitive Helion device helpers with each wrapper that compiles them.
GROUPED_MAILBOX_DEPENDENCIES = (
    "language/_gelu_tanh_approx.py",
    "_compiler/cute/_flash_runtime.py",
    "_compiler/cute/_mlir_compat.py",
    "_compiler/cute/epilogue_helpers.py",
)
GROUPED_RNA_DEPENDENCIES = (
    *GROUPED_MAILBOX_DEPENDENCIES,
    "_compiler/cute/tcgen05_flattened_prefix.py",
    "_compiler/cute/tcgen05_operand_pipeline.py",
    "_compiler/cute/tcgen05_operand_transform.py",
    "_compiler/cute/tcgen05_storage.py",
)
GROUPED_TMA_RN_DEPENDENCIES = (
    *GROUPED_MAILBOX_DEPENDENCIES,
    "_compiler/cute/tcgen05_flattened_prefix.py",
)

EMITTERS = (
    HostWrapperEmitter("tcgen05_d_tma", emit_d_tma),
    HostWrapperEmitter("tcgen05_aux_tma", emit_aux_tma),
    HostWrapperEmitter(
        "tcgen05_grouped_static_persistent", emit_grouped_static_persistent
    ),
    HostWrapperEmitter("tcgen05_ab_tma", emit_ab_tma),
    HostWrapperEmitter(
        "tcgen05_grouped_rna", emit_ab_tma, source_dependencies=GROUPED_RNA_DEPENDENCIES
    ),
    HostWrapperEmitter(
        "tcgen05_grouped_tma_rn",
        emit_ab_tma,
        source_dependencies=GROUPED_TMA_RN_DEPENDENCIES,
    ),
)
