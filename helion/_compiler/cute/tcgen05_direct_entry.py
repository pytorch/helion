from __future__ import annotations

import dataclasses
from typing import Any

from .strategies import tcgen05_explicit_d_store_tile_expr
from .strategies import tcgen05_smem_layout_expr


@dataclasses.dataclass(frozen=True)
class Tcgen05DirectEntryPlan:
    """Metadata for the exact Target1 direct CuTe entrypoint.

    The generated module can consume this plan to emit a tensor-entry function
    that builds A/B/D TMA descriptors in generated source. The current direct
    entry still launches the generated kernel body, while non-target kernels
    keep the scalarized wrapper fallback.
    """

    lhs_name: str
    rhs_name: str
    d_name: str
    bm: int
    bn: int
    bk: int
    cluster_m: int
    cluster_n: int
    ab_stage_count: int
    c_stage_count: int
    input_dtype: str
    output_dtype: str
    tma_store_atom: str
    tma_store_tensor: str
    ab_kernel_args: tuple[str, str, str, str]
    d_kernel_args: tuple[str, str]

    def to_codegen_plan(self) -> dict[str, Any]:
        return {
            "kind": "tcgen05_target1_direct_entry",
            "lhs_name": self.lhs_name,
            "rhs_name": self.rhs_name,
            "d_name": self.d_name,
            "bm": self.bm,
            "bn": self.bn,
            "bk": self.bk,
            "cluster_m": self.cluster_m,
            "cluster_n": self.cluster_n,
            "ab_stage_count": self.ab_stage_count,
            "c_stage_count": self.c_stage_count,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "tma_store_atom": self.tma_store_atom,
            "tma_store_tensor": self.tma_store_tensor,
            "ab_kernel_args": list(self.ab_kernel_args),
            "d_kernel_args": list(self.d_kernel_args),
        }


def _plan_int(plan: dict[str, object], key: str) -> int:
    value = plan[key]
    assert isinstance(value, int), (key, value)
    return value


def _plan_kernel_args(plan: dict[str, object], count: int) -> list[str]:
    kernel_args = [str(arg) for arg in plan["kernel_args"]]  # type: ignore[index]
    assert len(kernel_args) == count
    return kernel_args


def _append_target1_ab_tma_descriptor_source(
    body: list[str],
    call_args: list[str],
    plan: dict[str, object],
) -> None:
    assert plan["kind"] == "tcgen05_ab_tma"
    lhs_idx = _plan_int(plan, "lhs_idx")
    rhs_idx = _plan_int(plan, "rhs_idx")
    bm = _plan_int(plan, "bm")
    bn = _plan_int(plan, "bn")
    bk = _plan_int(plan, "bk")
    cluster_m = _plan_int(plan, "cluster_m")
    cluster_n = _plan_int(plan, "cluster_n")
    ab_stage_count = _plan_int(plan, "ab_stage_count")
    input_dtype = str(plan["input_dtype"])
    acc_dtype = str(plan["acc_dtype"])
    assert cluster_m == 2 and cluster_n == 1
    assert bm == 256 and bn == 256 and bk == 64 and ab_stage_count in (3, 6)
    assert input_dtype == "cutlass.BFloat16"
    assert acc_dtype == "cutlass.Float32"
    if "smem_swizzle_a" in plan or "smem_swizzle_b" in plan:
        raise AssertionError("Target1 direct entry requires default A/B SMEM layouts")
    tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = _plan_kernel_args(plan, 4)
    tiled_mma = f"{tma_atom_a}_tiled_mma"
    cluster_layout_vmnk = f"{tma_atom_a}_cluster_layout_vmnk"
    smem_a_layout = f"{tma_atom_a}_smem_layout"
    smem_b_layout = f"{tma_atom_b}_smem_layout"
    rhs_tma = f"{tma_atom_b}_rhs_tma"
    smem_a_layout_expr = tcgen05_smem_layout_expr(
        tiled_mma=tiled_mma,
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_str=input_dtype,
        num_stages=ab_stage_count,
        operand="a",
        swizzle_override=None,
    )
    smem_b_layout_expr = tcgen05_smem_layout_expr(
        tiled_mma=tiled_mma,
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_str=input_dtype,
        num_stages=ab_stage_count,
        operand="b",
        swizzle_override=None,
    )
    body.extend(
        (
            (
                f"    {tiled_mma} = cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
                f"{input_dtype}, "
                "cute.nvgpu.tcgen05.OperandMajorMode.K, "
                "cute.nvgpu.tcgen05.OperandMajorMode.MN, "
                f"{acc_dtype}, "
                "cute.nvgpu.tcgen05.CtaGroup.TWO, "
                f"({bm}, {bn}), "
                "cute.nvgpu.tcgen05.OperandSource.SMEM)"
            ),
            (
                f"    {cluster_layout_vmnk} = cute.tiled_divide("
                f"cute.make_layout(({cluster_m}, {cluster_n}, 1)), "
                f"({tiled_mma}.thr_id.shape,))"
            ),
            f"    {smem_a_layout} = {smem_a_layout_expr}",
            f"    {smem_b_layout} = {smem_b_layout_expr}",
            (
                f"    {rhs_tma} = cute.make_tensor("
                f"arg{rhs_idx}.iterator, "
                "layout=cute.make_layout("
                f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape0), "
                f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride0)))"
            ),
            f"    {rhs_tma}.mark_layout_dynamic(leading_dim=0)",
            (
                f"    {tma_atom_a}, {tma_tensor_a} = cute.nvgpu.make_tiled_tma_atom_A("
                "cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A("
                f"({cluster_m}, {cluster_n}, 1), {tiled_mma}.thr_id), "
                f"arg{lhs_idx}, "
                f"cute.slice_({smem_a_layout}, (None, None, None, 0)), "
                f"({bm}, {bn}, {bk}), {tiled_mma})"
            ),
            (
                f"    {tma_atom_b}, {tma_tensor_b} = cute.nvgpu.make_tiled_tma_atom_B("
                "cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_B("
                f"({cluster_m}, {cluster_n}, 1), {tiled_mma}.thr_id), "
                f"{rhs_tma}, "
                f"cute.slice_({smem_b_layout}, (None, None, None, 0)), "
                f"({bm}, {bn}, {bk}), {tiled_mma}, {cluster_layout_vmnk}.shape)"
            ),
        )
    )
    call_args.extend([tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b])


def _append_target1_d_tma_descriptor_source(
    body: list[str],
    call_args: list[str],
    plan: dict[str, object],
) -> None:
    assert plan["kind"] == "tcgen05_d_tma"
    d_idx = _plan_int(plan, "d_idx")
    bm = _plan_int(plan, "bm")
    bn = _plan_int(plan, "bn")
    c_stage_count = _plan_int(plan, "c_stage_count")
    epi_tile_m = _plan_int(plan, "epi_tile_m")
    epi_tile_n = _plan_int(plan, "epi_tile_n")
    d_store_box_n = _plan_int(plan, "d_store_box_n")
    output_dtype = str(plan["output_dtype"])
    assert bm == 256 and bn == 256 and c_stage_count in (2, 4)
    assert (epi_tile_m, epi_tile_n, d_store_box_n) == (128, 32, 32)
    assert output_dtype == "cutlass.BFloat16"
    tma_atom, tma_tensor = _plan_kernel_args(plan, 2)
    epi_tile = f"{tma_atom}_epi_tile"
    smem_layout = f"{tma_atom}_smem_layout"
    cta_v_layout = f"{tma_atom}_cta_v_layout"
    body.extend(
        (
            f"    {epi_tile} = {tcgen05_explicit_d_store_tile_expr(epi_tile_m, d_store_box_n)}",
            (
                f"    {smem_layout} = cutlass.utils.blackwell_helpers."
                "make_smem_layout_epi("
                f"{output_dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, "
                f"{epi_tile}, {c_stage_count})"
            ),
            (
                f"    {cta_v_layout} = cute.composition("
                f"cute.make_identity_layout(arg{d_idx}.shape), {epi_tile})"
            ),
            (
                f"    {tma_atom}, {tma_tensor} = "
                "cute.nvgpu.cpasync.make_tiled_tma_atom("
                "cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), "
                f"arg{d_idx}, cute.slice_({smem_layout}, (None, None, 0)), "
                f"{cta_v_layout})"
            ),
        )
    )
    call_args.extend([tma_atom, tma_tensor])


def build_target1_direct_entry_source(
    *,
    function_name: str,
    kernel_name: str,
    direct_plan: dict[str, object],
    wrapper_plans: list[dict[str, object]],
) -> str:
    """Build a generated Target1 tensor-entry source function.

    This direct entry is compiler-emitted into the generated module instead of
    assembled dynamically by ``helion.runtime``. It still launches the generated
    CuTe kernel body, but descriptor construction now lives in generated code
    beside that body and the runtime direct-entry fallback is no longer needed
    for the validated Target1 path.
    """

    if direct_plan.get("kind") != "tcgen05_target1_direct_entry":
        raise AssertionError(f"unsupported direct-entry plan: {direct_plan!r}")
    if [plan.get("kind") for plan in wrapper_plans] != [
        "tcgen05_ab_tma",
        "tcgen05_d_tma",
    ]:
        raise AssertionError("Target1 direct entry requires A/B then D wrapper plans")
    if (
        _plan_int(direct_plan, "lhs_idx"),
        _plan_int(direct_plan, "rhs_idx"),
        _plan_int(direct_plan, "d_idx"),
        _plan_int(direct_plan, "bm"),
        _plan_int(direct_plan, "bn"),
        _plan_int(direct_plan, "bk"),
        _plan_int(direct_plan, "cluster_m"),
        _plan_int(direct_plan, "cluster_n"),
        _plan_int(direct_plan, "ab_stage_count"),
        _plan_int(direct_plan, "c_stage_count"),
    ) not in (
        (0, 1, 2, 256, 256, 64, 2, 1, 3, 2),
        (0, 1, 2, 256, 256, 64, 2, 1, 6, 4),
    ):
        raise AssertionError("Target1 direct entry plan does not match stage envelope")
    if _plan_int(wrapper_plans[0], "ab_stage_count") != _plan_int(
        direct_plan, "ab_stage_count"
    ):
        raise AssertionError("Target1 direct entry A/B wrapper stages are stale")
    if _plan_int(wrapper_plans[1], "c_stage_count") != _plan_int(
        direct_plan, "c_stage_count"
    ):
        raise AssertionError("Target1 direct entry D wrapper stages are stale")
    if (
        direct_plan.get("input_dtype") != "cutlass.BFloat16"
        or direct_plan.get("output_dtype") != "cutlass.BFloat16"
    ):
        raise AssertionError("Target1 direct entry requires bf16 input/output")
    body: list[str] = []
    for index in (0, 1, 2):
        body.extend(
            (
                f"    arg{index}_shape0 = arg{index}.shape[0]",
                f"    arg{index}_shape1 = arg{index}.shape[1]",
                f"    arg{index}_stride0 = arg{index}.stride[0]",
                f"    arg{index}_stride1 = arg{index}.stride[1]",
            )
        )
    call_args = ["arg0", "arg1", "arg2"]
    _append_target1_ab_tma_descriptor_source(body, call_args, wrapper_plans[0])
    _append_target1_d_tma_descriptor_source(body, call_args, wrapper_plans[1])
    launch_args = ["grid=(128, 1, 1)", "block=(256, 1, 1)", "cluster=[2, 1, 1]"]
    if any(plan.get("use_pdl") for plan in wrapper_plans):
        launch_args.append("use_pdl=True")
    body.append(
        "    "
        + kernel_name
        + "("
        + ", ".join(call_args)
        + ").launch("
        + ", ".join(launch_args)
        + ")"
    )
    return "\n".join(
        [
            "@cute.jit",
            f"def {function_name}(arg0, arg1, arg2) -> None:",
            *body,
        ]
    )
