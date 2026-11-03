"""Per-kernel launch facts of a compiled CuTe kernel, derived once.

The compiler renders a kernel's wrapper plans (``_helion_cute_wrapper_plans``)
and a few launch attributes (``_helion_cute_cluster_shape``,
``_helion_cute_use_pdl``, ...) into the generated module as plain assignments;
the launcher used to re-derive the constants it needs from the plan list on
every slow-path launch.  :class:`CuteLaunchDescriptor` computes the
plan-derived constants once per plan list -- which plans are present (and
that the singleton kinds are singletons), which operands the staged plan
kinds address with 16-byte instructions, whether the plans bake tensor
shapes -- and :func:`cute_launch_descriptor` caches it on the kernel keyed
on the identity of the plan list object (tests reassign the attribute on a
live kernel).  The launch attributes read live: tests set them after the
first use and ``_helion_cute_source_hash`` is assigned after module load.
Leaf module: imports ``helion.exc`` and ``plan_facts`` only.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ... import exc
from .plan_facts import tcgen05_plan_orientation

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence


def cluster_shape_from_wrapper_plans(
    wrapper_plans: Sequence[dict[str, object]],
) -> tuple[int, int, int] | None:
    cluster_m = 1
    cluster_n = 1
    for plan in wrapper_plans:
        if plan.get("kind") != "tcgen05_ab_tma":
            continue
        plan_cluster_m = plan.get("cluster_m", 1)
        plan_cluster_n = plan.get("cluster_n", 1)
        assert isinstance(plan_cluster_m, int)
        assert isinstance(plan_cluster_n, int)
        cluster_m = max(cluster_m, plan_cluster_m)
        cluster_n = max(cluster_n, plan_cluster_n)
    if cluster_m * cluster_n <= 1:
        return None
    return (cluster_m, cluster_n, 1)


def use_pdl(cute_kernel: object) -> bool:
    """Whether the generated kernel asked for programmatic dependent launch.

    The device function then waits (``griddepcontrol_wait``) before its first
    global memory access, so the launch and prologue overlap the previous
    kernel in the stream.
    """
    return getattr(cast("Any", cute_kernel), "_helion_cute_use_pdl", False) is True


def cluster_shape(
    cute_kernel: object, wrapper_plans: Sequence[dict[str, object]]
) -> tuple[int, int, int] | None:
    explicit_cluster_shape = getattr(
        cast("Any", cute_kernel), "_helion_cute_cluster_shape", None
    )
    if explicit_cluster_shape is not None:
        if (
            isinstance(explicit_cluster_shape, tuple)
            and len(explicit_cluster_shape) == 3
            and all(isinstance(dim, int) for dim in explicit_cluster_shape)
        ):
            return cast("tuple[int, int, int]", explicit_cluster_shape)
        raise exc.BackendUnsupported(
            "cute",
            f"invalid _helion_cute_cluster_shape: {explicit_cluster_shape!r}",
        )
    return cluster_shape_from_wrapper_plans(wrapper_plans)


def wrapper_plan_bakes_tensor_shapes(plan: dict[str, object]) -> bool:
    kind = str(plan.get("kind", ""))
    if kind in {
        "helion_small_biased_attention",
        "helion_flash_row_mma",
        "helion_warp_mma_gemm",
        "helion_register_subblocks",
        "chunk_prepare_tma",
        "chunk_recurrence_sm100",
        "chunk_recurrence_warp_dv4",
        "gdn_recurrence_sm100",
        "gathered_mma_tma",
        "block_scaled_mma",
    }:
        return True
    if not kind.startswith("tcgen05"):
        return False
    if kind != "tcgen05_ab_tma":
        return True
    # A row-union plan records its extents in MMA order already (``cute_mma``
    # swaps them with the profile), so only the plain N,M plan pairs the
    # problem extents crosswise with the MMA tile.
    extent_blocks = (
        (("n_size", "bm"), ("m_size", "bn"), ("k_total_size", "bk"))
        if tcgen05_plan_orientation(plan) == "nm" and "row_union_schedule" not in plan
        else (("m_size", "bm"), ("n_size", "bn"), ("k_total_size", "bk"))
    )
    for extent_key, block_key in extent_blocks:
        extent = plan.get(extent_key)
        block = plan.get(block_key)
        if type(extent) is not int or type(block) is not int or extent % block:
            return False
    return True


# Wrapper plans whose device bodies address the named tensors with 16-byte-wide
# instructions: TMA tensor maps over q/k/v (and the outputs the epilogues store
# or reduce through TMA), 128-bit copies of the output tile, the row programs'
# 16-byte packets. A TMA descriptor encodes its global address in 16-byte
# units, and the CuTe DSL builds one over a pointer whose assumed alignment
# is only 8 bytes without complaint: the base is rounded down to 16 bytes and
# every tile reads (or writes) 8 bytes before the tensor, giving a wrong
# result instead of a fault. ``default_cute_launcher`` therefore stages an
# under-aligned binding of these operands through a 16-byte-aligned copy and
# copies the outputs back after the launch. Values are (input index keys,
# output index keys); the flash kinds add their ``epi_aux{i}_idx`` inputs.
STAGED_PLAN_OPERANDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "helion_flash": (
        ("q_idx", "k_idx", "v_idx", "bias_idx", "alibi_idx", "document_idx"),
        ("o_idx", "lse_idx"),
    ),
    "helion_flash_row_mma": (
        ("q_idx", "k_idx", "v_idx", "bias_idx"),
        ("o_idx", "lse_idx"),
    ),
    "helion_flash_gated": (("q_idx", "k_idx", "v_idx"), ("o_idx",)),
    "helion_flash_gated_row_mma": (("q_idx", "k_idx", "v_idx"), ("o_idx",)),
    "helion_flash_bwd": (
        ("q_idx", "k_idx", "v_idx", "do_idx", "lse_idx", "delta_idx"),
        ("dq_idx", "dk_idx", "dv_idx"),
    ),
    "helion_flash_bwd_row_mma": (
        ("q_idx", "k_idx", "v_idx", "do_idx", "lse_idx", "delta_idx"),
        ("dq_idx", "dk_idx", "dv_idx"),
    ),
}
STAGED_OPERAND_ALIGNMENT = 16

# Whole-root plans that pass host-encoded TensorMap descriptor addresses as
# scalar wrapper arguments.  Their descriptor payload embeds the tensor data
# pointers, so patching only the ordinary tensor-pointer slots would leave a
# fast relaunch aimed at the first invocation's buffers; they stay on the
# pointer-keyed launch cache until the fastpath can also rebuild or patch raw
# TensorMap descriptors.
FASTPATH_EXCLUDED_KINDS = frozenset(
    {
        "chunk_prepare_tma",
        "chunk_recurrence_sm100",
        "chunk_recurrence_warp_dv4",
        "gdn_recurrence_sm100",
        "gathered_mma_tma",
        "block_scaled_mma",
    }
)


def block_scaled_aligned_inputs(plan: Mapping[str, object]) -> tuple[str, ...]:
    """Index keys of a block-scaled plan's inputs read with an alignment
    assumption: MXFP8 TMA reads the original FP8 allocations; the fused NVFP4
    consumer TMAs the packed sources and their scales; the linear preparation
    reads a side with a proved qword recipe as 8-byte words."""
    if plan.get("format") == "mxfp8":
        if plan.get("sf_tma") is True:
            return ("lhs_idx", "rhs_idx", "lhs_scale_idx", "rhs_scale_idx")
        return ("lhs_idx", "rhs_idx")
    if plan.get("prepare") == "fused":
        return ("lhs_idx", "rhs_idx", "lhs_scale_idx", "rhs_scale_idx")
    return tuple(
        f"{side}_idx"
        for side in ("lhs", "rhs")
        if plan.get(f"{side}_qword_address") is not None
    )


def _staged_operands(
    plans: Sequence[dict[str, object]],
) -> tuple[tuple[int, bool], ...]:
    """The (argument index, is_output) operands the kernel's plans stage."""
    operands: dict[int, bool] = {}
    for plan in plans:
        kind = str(plan.get("kind"))
        roles = STAGED_PLAN_OPERANDS.get(kind)
        if kind == "block_scaled_mma":
            roles = (block_scaled_aligned_inputs(plan), ())
            if plan.get("tma_output") is True:
                roles = (roles[0], ("out_idx",))
        if roles is None:
            continue
        input_keys, output_keys = roles
        if kind in ("helion_flash", "helion_flash_row_mma"):
            aux_count = plan.get("epi_aux_count", 0)
            assert isinstance(aux_count, int)
            input_keys += tuple(f"epi_aux{i}_idx" for i in range(aux_count))
        for keys, is_output in ((input_keys, False), (output_keys, True)):
            for key in keys:
                index = plan.get(key)
                if isinstance(index, int):
                    operands[index] = operands.get(index, False) or is_output
    return tuple(operands.items())


def _staged_strided_copies(plans: Sequence[dict[str, object]]) -> frozenset[int]:
    """Staged operands copied with their strides (``empty_strided`` + ``copy_``)
    rather than cloned: the MXFP8 TMA scale factors and TMA output."""
    indices: set[int] = set()
    for plan in plans:
        if plan.get("kind") != "block_scaled_mma" or plan.get("format") != "mxfp8":
            continue
        if plan.get("sf_tma") is True:
            for key in ("lhs_scale_idx", "rhs_scale_idx"):
                index = plan.get(key)
                if isinstance(index, int):
                    indices.add(index)
        if plan.get("tma_output") is True:
            index = plan.get("out_idx")
            if isinstance(index, int):
                indices.add(index)
    return frozenset(indices)


def _singleton(
    plans: Sequence[dict[str, object]], kind: str, message: str
) -> dict[str, object] | None:
    matching = [plan for plan in plans if plan.get("kind") == kind]
    if len(matching) > 1:
        raise exc.BackendUnsupported("cute", message)
    return matching[0] if matching else None


@dataclass(frozen=True)
class CuteLaunchDescriptor:
    """The plan-derived launch constants of one kernel (see the module doc)."""

    plans_source: object
    plans: tuple[dict[str, object], ...]
    grouped_static_plans: tuple[dict[str, object], ...]
    split_k_plan: dict[str, object] | None
    fixup_plan: dict[str, object] | None
    chunk_prepare_plan: dict[str, object] | None
    chunk_recurrence_plan: dict[str, object] | None
    sm100_recurrence_plan: dict[str, object] | None
    gdn_recurrence_plan: dict[str, object] | None
    alias_plan: dict[str, object] | None
    plans_bake_tensor_shapes: bool
    staged_operands: tuple[tuple[int, bool], ...]
    staged_strided_copies: frozenset[int]
    fastpath_excluded: bool

    @staticmethod
    def of(plans_source: object) -> CuteLaunchDescriptor:
        plans = tuple(
            cast("dict[str, object]", plan)
            for plan in (
                cast("Sequence[object]", plans_source)
                if plans_source is not None
                else ()
            )
        )
        return CuteLaunchDescriptor(
            plans_source=plans_source,
            plans=plans,
            grouped_static_plans=tuple(
                plan
                for plan in plans
                if plan.get("kind") == "tcgen05_grouped_static_persistent"
            ),
            split_k_plan=_singleton(
                plans,
                "helion_tcgen05_split_k",
                "multiple tcgen05 split-K wrapper plans",
            ),
            fixup_plan=_singleton(
                plans, "helion_mma_dag_fixup", "multiple mma_dag fixup wrapper plans"
            ),
            chunk_prepare_plan=_singleton(
                plans, "chunk_prepare_tma", "multiple chunk-prepare wrapper plans"
            ),
            chunk_recurrence_plan=_singleton(
                plans,
                "chunk_recurrence_warp_dv4",
                "multiple chunk-recurrence wrapper plans",
            ),
            sm100_recurrence_plan=_singleton(
                plans, "chunk_recurrence_sm100", "multiple SM100 recurrence plans"
            ),
            gdn_recurrence_plan=_singleton(
                plans, "gdn_recurrence_sm100", "multiple gdn recurrence plans"
            ),
            alias_plan=_singleton(
                plans, "helion_mma_dag_alias", "multiple mma_dag alias plans"
            ),
            plans_bake_tensor_shapes=not (
                bool(plans)
                and any(not wrapper_plan_bakes_tensor_shapes(plan) for plan in plans)
            ),
            staged_operands=_staged_operands(plans),
            staged_strided_copies=_staged_strided_copies(plans),
            fastpath_excluded=any(
                plan.get("kind") in FASTPATH_EXCLUDED_KINDS for plan in plans
            ),
        )


def cute_launch_descriptor(cute_kernel: object) -> CuteLaunchDescriptor:
    """The kernel's descriptor, rebuilt when its plan list object changes."""
    any_kernel = cast("Any", cute_kernel)
    plans_source = getattr(any_kernel, "_helion_cute_wrapper_plans", None)
    cached = getattr(any_kernel, "_helion_cute_launch_descriptor", None)
    if isinstance(cached, CuteLaunchDescriptor) and cached.plans_source is plans_source:
        return cached
    descriptor = CuteLaunchDescriptor.of(plans_source)
    with suppress(AttributeError, TypeError):
        any_kernel._helion_cute_launch_descriptor = descriptor
    return descriptor


def bake_tensor_shapes(cute_kernel: object) -> bool:
    """Whether the compiled wrapper bakes tensor sizes and strides: not when
    the kernel opted out or one of its plans propagates runtime layouts."""
    return not (
        getattr(
            cast("Any", cute_kernel), "_helion_cute_disable_bake_tensor_shapes", False
        )
        or not cute_launch_descriptor(cute_kernel).plans_bake_tensor_shapes
    )
