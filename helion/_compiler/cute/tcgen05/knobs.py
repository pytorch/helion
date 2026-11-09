"""Typed reads of the tcgen05 GEMM config knobs.

``Tcgen05Knobs.from_config`` copies every key ``plan_tcgen05_pipeline`` and the
``_emit_mma_pipeline`` emission read, with the default each site used. Keys
whose sites used different defaults (``grouped_resident_ctas``,
``grouped_panel_alternate``, ``grouped_panel_ratio``, ``aux_load_mode``,
``persistence_model``, ``sched_stage_count``) and keys whose default depends
on values resolved mid-plan (``ab_stages``, ``c_stages``, ``acc_stages``,
``l2_swizzle_size``, ``split_k``, ``epi_warp_groups``) are kept raw
(``None`` when absent); :meth:`Tcgen05Knobs.int_or` is the :func:`int_knob`
rule for them (a non-int or absent value takes the default). Knobs that
other modules also decode through a helper (``warp_spec_from_config``,
``layout_overrides_from_config``, ``l2_swizzle_size_from_config``,
``physical_schedule``, the consumer-register helpers) stay with those helpers.

The module-level readers are the single-knob reads the codegen, the launch
planning and the config layer share.  Each keeps the rule of the site it came
from; the rules differ on purpose and are not interchangeable:

* :func:`int_knob` -- any ``int`` (a ``bool`` is one), else the default; the
  rule of ``Tcgen05Knobs.int_or`` and of the cluster / stage reads in the
  planner and the SMEM prefilter.
* :func:`non_bool_int_knob` -- an ``int`` that is not a ``bool``, else the
  default (the warp-MMA GEMM's warp count).
* :func:`positive_int_knob` -- exactly ``int`` (no ``bool``) and ``> 0``, else
  the default (the config layer's stage and warp-group reads).
* :func:`parsed_int_knob` -- ``int(str(value))``: accepts digit strings and
  numpy integers and raises on anything else (the scheduler prelude's cluster
  fallback outside a registered matmul plan).
* :func:`str_knob` -- a ``str``, else the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from ..grouped_panel_raster import GROUPED_PANEL_ALTERNATE_KEY
from ..grouped_panel_raster import GROUPED_PANEL_RATIO_KEY
from ..grouped_row_union import CONFIG_KEY as GROUPED_ROW_UNION_KEY
from ..grouped_row_union import RESIDENT_CTAS_KEY as GROUPED_RESIDENT_CTAS_KEY
from ..grouped_row_union import STARTUP_PREFILL_KEY
from ..pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from ..strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from ..strategies import TCGEN05_NUM_EPI_WARPS_CONFIG_KEY
from ..strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from ..tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_DEFAULT
from ..tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from ..tcgen05_constants import TCGEN05_CLC_QUERY_PACING_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_CLC_QUERY_PACING_TILE
from ..tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPI_WARP_GROUP_CHOICES
from ..tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from ..tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY
from ..tcgen05_constants import TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping

GROUPED_WORKLIST_ORIENTATION_KEY = "tcgen05_grouped_worklist_orientation"
AB_STAGES_KEY = "tcgen05_ab_stages"
C_STAGES_KEY = "tcgen05_c_stages"
ACC_STAGES_KEY = "tcgen05_acc_stages"
MATERIALIZED_PDL_KEY = "tcgen05_materialized_pdl"
CLUSTER_M_KEY = "tcgen05_cluster_m"
CLUSTER_N_KEY = "tcgen05_cluster_n"


@dataclass(frozen=True)
class Tcgen05Knobs:
    """One config's tcgen05 pipeline knobs; see the module docstring."""

    # Grouped families.
    row_union: object
    grouped_worklist_orientation: object
    grouped_external_direct_pointers: object | None
    grouped_external_direct_strides: object | None
    grouped_static_problem_signature: object | None
    grouped_static_reserved_sms: int
    grouped_resident_ctas: object | None
    grouped_ab_tile_k: object | None
    grouped_panel_alternate: object | None
    grouped_panel_ratio: object | None
    grouped_runtime_direct: object
    grouped_full_allocation_a: object
    grouped_d_shift: object
    grouped_balanced_workers: object
    grouped_full_coverage: object | None
    # Schedule and roles.
    pid_type: object
    l2_groupings: list[int]
    num_sm_multiplier: object
    cta_group: str
    flat_role_coordinates: bool
    persistence_model: object | None
    sched_stage_count: object | None
    l2_swizzle_size: object | None
    split_k: object | None
    warp_register_reallocation: object
    epi_warp_groups: object | None
    epi_subtile_n: object | None
    # Pipelines and epilogue.
    ab_stages: object | None
    c_stages: object | None
    acc_stages: object | None
    aux_load_mode: object | None
    aux_load_placement: object | None
    clc_query_pacing: object
    c_store_mode: object
    epilogue_layout: object
    ab_startup_prefill: object
    # Launch.
    cute_pdl: object | None
    materialized_pdl: object

    @staticmethod
    def int_or(value: object, default: int) -> int:
        """A raw knob as an int: an absent or non-int value takes ``default``."""
        return value if isinstance(value, int) else default

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> Tcgen05Knobs:
        return cls(
            row_union=config.get(GROUPED_ROW_UNION_KEY, False),
            grouped_worklist_orientation=config.get(
                GROUPED_WORKLIST_ORIENTATION_KEY, "nm"
            ),
            grouped_external_direct_pointers=config.get(
                TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY
            ),
            grouped_external_direct_strides=config.get(
                TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY
            ),
            grouped_static_problem_signature=config.get(
                TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
            ),
            grouped_static_reserved_sms=int(
                cast(
                    "int", config.get(TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY, 0)
                )
            ),
            grouped_resident_ctas=config.get(GROUPED_RESIDENT_CTAS_KEY),
            grouped_ab_tile_k=config.get(TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY),
            grouped_panel_alternate=config.get(GROUPED_PANEL_ALTERNATE_KEY),
            grouped_panel_ratio=config.get(GROUPED_PANEL_RATIO_KEY),
            grouped_runtime_direct=config.get(
                TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY, False
            ),
            grouped_full_allocation_a=config.get(
                TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY, False
            ),
            grouped_d_shift=config.get(TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY, False),
            grouped_balanced_workers=config.get(
                TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY, False
            ),
            grouped_full_coverage=config.get(TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY),
            pid_type=config.get("pid_type", "flat"),
            l2_groupings=cast("list[int]", config.get("l2_groupings", [])),
            num_sm_multiplier=config.get("num_sm_multiplier", 1),
            cta_group=cast("str", config.get(TCGEN05_CTA_GROUP_CONFIG_KEY, "auto")),
            flat_role_coordinates=bool(
                config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY, False)
            ),
            persistence_model=config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY),
            sched_stage_count=config.get(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY),
            l2_swizzle_size=config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY),
            split_k=config.get(TCGEN05_SPLIT_K_CONFIG_KEY),
            warp_register_reallocation=config.get(
                TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY, True
            ),
            epi_warp_groups=config.get(TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY),
            epi_subtile_n=config.get(TCGEN05_EPI_SUBTILE_N_CONFIG_KEY),
            ab_stages=config.get(AB_STAGES_KEY),
            c_stages=config.get(C_STAGES_KEY),
            acc_stages=config.get(ACC_STAGES_KEY),
            aux_load_mode=config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY),
            aux_load_placement=config.get(TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY),
            clc_query_pacing=config.get(
                TCGEN05_CLC_QUERY_PACING_CONFIG_KEY, TCGEN05_CLC_QUERY_PACING_TILE
            ),
            c_store_mode=config.get(
                TCGEN05_C_STORE_MODE_CONFIG_KEY, TCGEN05_C_STORE_MODE_NORMAL
            ),
            epilogue_layout=config.get(
                TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY, TCGEN05_EPILOGUE_LAYOUT_NORMAL
            ),
            ab_startup_prefill=config.get(STARTUP_PREFILL_KEY, False),
            cute_pdl=config.get("cute_pdl"),
            materialized_pdl=config.get(MATERIALIZED_PDL_KEY, False),
        )


def int_knob(config: Mapping[str, object], key: str, default: int) -> int:
    """``config[key]`` as an int (a ``bool`` is one), else ``default``."""
    return Tcgen05Knobs.int_or(config.get(key, default), default)


def non_bool_int_knob(config: Mapping[str, object], key: str, default: int) -> int:
    """``config[key]`` as an int that is not a ``bool``, else ``default``."""
    value = config.get(key, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def positive_int_knob(config: Mapping[str, object], key: str, default: int) -> int:
    """``config[key]`` when it is exactly an ``int`` above zero, else ``default``."""
    value = config.get(key, default)
    if type(value) is int and value > 0:
        return value
    return default


def parsed_int_knob(config: Mapping[str, object], key: str, default: int) -> int:
    """``int(str(config[key]))``: digit strings and numpy ints parse, the rest raises."""
    return int(str(config.get(key, default)))


def str_knob(config: Mapping[str, object], key: str, default: str) -> str:
    """``config[key]`` as a str, else ``default``."""
    value = config.get(key, default)
    return value if isinstance(value, str) else default


def clamp_cluster_extent(value: int) -> int:
    """A cluster extent held to the one- or two-CTA range."""
    return max(1, min(2, value))


def cluster_m_knob(config: Mapping[str, object]) -> int:
    """``tcgen05_cluster_m`` (default 1) clamped to 1..2."""
    return clamp_cluster_extent(int_knob(config, CLUSTER_M_KEY, 1))


def cluster_n_knob(config: Mapping[str, object]) -> int:
    """``tcgen05_cluster_n`` (default 1) clamped to 1..2.

    cluster_n=2 only runs under the canonical Quack-best 4-CTA cluster
    (``cluster_m=2 use_2cta=True``); the ``cluster_n>1`` capability gate in
    the planner rejects unsupported pairings, so this returns the *requested*
    value and lets the caller demote.
    """
    return clamp_cluster_extent(int_knob(config, CLUSTER_N_KEY, 1))


def launch_epi_warp_groups_knob(config: Mapping[str, object]) -> int:
    """The epilogue warp-group count the role launch is sized for.

    ``tcgen05_epi_warp_groups`` when it is one of
    ``TCGEN05_EPI_WARP_GROUP_CHOICES``, else 1 (the planner refuses the
    out-of-range value later; the launch shape never widens for it).
    """
    groups = int_knob(config, TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY, 1)
    return groups if groups in TCGEN05_EPI_WARP_GROUP_CHOICES else 1


def epi_warp_groups_knob(config: Mapping[str, object]) -> int:
    """``tcgen05_epi_warp_groups`` as the config layer reads it: a positive int, else 1."""
    return positive_int_knob(config, TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY, 1)


def epi_subtile_n_knob(config: Mapping[str, object]) -> int | None:
    """``tcgen05_epi_subtile_n`` when it is exactly a positive ``int``, else ``None``."""
    value = config.get(TCGEN05_EPI_SUBTILE_N_CONFIG_KEY)
    if type(value) is int and value > 0:
        return value
    return None


def epi_warps_knob(config: Mapping[str, object]) -> int:
    """``tcgen05_num_epi_warps``: a positive int (a ``bool`` is one), else 4."""
    epi_warps = config.get(TCGEN05_NUM_EPI_WARPS_CONFIG_KEY, 4)
    if isinstance(epi_warps, int) and epi_warps > 0:
        return epi_warps
    return 4


def acc_stages_knob(config: Mapping[str, object]) -> int:
    """``tcgen05_acc_stages``: a positive int (a ``bool`` is one), else 2."""
    acc_stages = config.get(ACC_STAGES_KEY, 2)
    if isinstance(acc_stages, int) and acc_stages > 0:
        return acc_stages
    return 2


def aux_stages_knob(config: Mapping[str, object]) -> int:
    """``tcgen05_aux_stages`` when it is exactly a positive ``int``, else the default."""
    return positive_int_knob(
        config, TCGEN05_AUX_STAGES_CONFIG_KEY, TCGEN05_AUX_STAGE_COUNT_DEFAULT
    )


def aux_load_mode_knob(config: Mapping[str, object]) -> str:
    """``tcgen05_aux_load_mode`` as a str, else the SIMT default."""
    return str_knob(
        config, TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, TCGEN05_AUX_LOAD_MODE_SIMT
    )
