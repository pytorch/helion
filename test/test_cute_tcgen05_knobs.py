"""``tcgen05.knobs``: ``Tcgen05Knobs.from_config`` defaults and the typed knob readers.

The reader parity table re-states, verbatim, the rule of every reader the
module replaced (``cute_mma._tcgen05_config_int`` / ``_tcgen05_cluster_m`` /
``_tcgen05_cluster_n`` / ``_tcgen05_epi_warp_groups``, the config layer's
former ``_config_*`` readers, the scheduler prelude's
``int(str(...))`` cluster fallback and ``cute_warp_mma_gemm._config_int``) and
checks each new reader against it over ints, bools, strings, floats, lists,
numpy ints and absent / ``None`` values.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses

import numpy as np
import pytest

from helion._compiler.cute.strategies import TCGEN05_NUM_EPI_WARPS_CONFIG_KEY
from helion._compiler.cute.tcgen05 import knobs
from helion._compiler.cute.tcgen05.knobs import Tcgen05Knobs
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_DEFAULT
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_DIRECT
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from helion._compiler.cute.tcgen05_constants import TCGEN05_CLC_QUERY_PACING_TILE
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_WARP_GROUP_CHOICES
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from helion._compiler.cute.tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY

_DEFAULTS = {
    "row_union": False,
    "grouped_worklist_orientation": "nm",
    "grouped_external_direct_pointers": None,
    "grouped_external_direct_strides": None,
    "grouped_static_problem_signature": None,
    "grouped_static_reserved_sms": 0,
    "grouped_resident_ctas": None,
    "grouped_ab_tile_k": None,
    "grouped_panel_alternate": None,
    "grouped_panel_ratio": None,
    "grouped_runtime_direct": False,
    "grouped_full_allocation_a": False,
    "grouped_d_shift": False,
    "grouped_balanced_workers": False,
    "grouped_full_coverage": None,
    "pid_type": "flat",
    "l2_groupings": [],
    "num_sm_multiplier": 1,
    "cta_group": "auto",
    "flat_role_coordinates": False,
    "persistence_model": None,
    "sched_stage_count": None,
    "l2_swizzle_size": None,
    "split_k": None,
    "warp_register_reallocation": True,
    "epi_warp_groups": None,
    "epi_subtile_n": None,
    "ab_stages": None,
    "c_stages": None,
    "acc_stages": None,
    "aux_load_mode": None,
    "aux_load_placement": None,
    "clc_query_pacing": TCGEN05_CLC_QUERY_PACING_TILE,
    "c_store_mode": TCGEN05_C_STORE_MODE_NORMAL,
    "epilogue_layout": TCGEN05_EPILOGUE_LAYOUT_NORMAL,
    "ab_startup_prefill": False,
    "cute_pdl": None,
    "materialized_pdl": False,
}


def test_empty_config_takes_every_default() -> None:
    knob_values = Tcgen05Knobs.from_config({})
    assert dataclasses.asdict(knob_values) == _DEFAULTS
    assert {field.name for field in dataclasses.fields(Tcgen05Knobs)} == set(_DEFAULTS)


def test_set_keys_are_read_raw() -> None:
    knob_values = Tcgen05Knobs.from_config(
        {
            TCGEN05_C_STORE_MODE_CONFIG_KEY: TCGEN05_C_STORE_MODE_DIRECT,
            TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: TCGEN05_AUX_LOAD_MODE_TMA,
            TCGEN05_SPLIT_K_CONFIG_KEY: 4,
            "tcgen05_ab_stages": 3,
            "pid_type": "persistent_blocked",
            "l2_groupings": [8, 1],
            "cute_pdl": True,
            "tcgen05_grouped_ctas_per_sm": 2,
        }
    )
    assert knob_values.c_store_mode == TCGEN05_C_STORE_MODE_DIRECT
    assert knob_values.aux_load_mode == TCGEN05_AUX_LOAD_MODE_TMA
    assert knob_values.split_k == 4
    assert knob_values.ab_stages == 3
    assert knob_values.pid_type == "persistent_blocked"
    assert knob_values.l2_groupings == [8, 1]
    assert knob_values.cute_pdl is True
    assert knob_values.grouped_resident_ctas == 2


# ---- reader parity: the replaced rules, copied verbatim -------------------

_ABSENT = object()
_VALUES: tuple[object, ...] = (
    _ABSENT,
    None,
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    True,
    False,
    "2",
    "x",
    "",
    2.0,
    2.5,
    [2],
    np.int64(2),
)
_VALUE_IDS = [
    "absent",
    "None",
    "-1",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "True",
    "False",
    "str2",
    "strx",
    "empty",
    "2.0",
    "2.5",
    "list",
    "np64",
]


def _old_tcgen05_config_int(
    config: Mapping[str, object], key: str, default: int
) -> int:
    value = config.get(key, default)
    if not isinstance(value, int):
        return default
    return value


def _old_tcgen05_cluster(config: Mapping[str, object], key: str) -> int:
    return max(1, min(2, _old_tcgen05_config_int(config, key, 1)))


def _old_tcgen05_epi_warp_groups(config: Mapping[str, object]) -> int:
    groups = _old_tcgen05_config_int(config, TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY, 1)
    return groups if groups in TCGEN05_EPI_WARP_GROUP_CHOICES else 1


def _old_scheduler_cluster(config: Mapping[str, object], key: str) -> int:
    cluster = int(str(config.get(key, 1)))
    return max(1, min(cluster, 2))


def _old_warp_mma_config_int(
    config: Mapping[str, object], key: str, default: int
) -> int:
    value = config.get(key, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _old_config_positive_int(
    config: Mapping[str, object], key: str, default: int
) -> int:
    value = config.get(key, default)
    if type(value) is int and value > 0:
        return value
    return default


def _old_config_str(config: Mapping[str, object], key: str, default: str) -> str:
    value = config.get(key, default)
    return value if isinstance(value, str) else default


def _old_config_aux_load_mode(config: Mapping[str, object]) -> str:
    mode = config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, TCGEN05_AUX_LOAD_MODE_SIMT)
    return mode if isinstance(mode, str) else TCGEN05_AUX_LOAD_MODE_SIMT


def _old_config_aux_stages(config: Mapping[str, object]) -> int:
    stages = config.get(TCGEN05_AUX_STAGES_CONFIG_KEY, TCGEN05_AUX_STAGE_COUNT_DEFAULT)
    return (
        stages
        if type(stages) is int and stages > 0
        else TCGEN05_AUX_STAGE_COUNT_DEFAULT
    )


def _old_config_epi_warp_groups(config: Mapping[str, object]) -> int:
    groups = config.get(TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY, 1)
    if type(groups) is int and groups > 0:
        return groups
    return 1


def _old_config_epi_subtile_n(config: Mapping[str, object]) -> int | None:
    value = config.get(TCGEN05_EPI_SUBTILE_N_CONFIG_KEY)
    if type(value) is int and value > 0:
        return value
    return None


def _old_config_epi_warps(config: Mapping[str, object]) -> int:
    epi_warps = config.get("tcgen05_num_epi_warps", 4)
    if isinstance(epi_warps, int) and epi_warps > 0:
        return epi_warps
    return 4


def _old_config_acc_stages(config: Mapping[str, object]) -> int:
    acc_stages = config.get("tcgen05_acc_stages", 2)
    if isinstance(acc_stages, int) and acc_stages > 0:
        return acc_stages
    return 2


_Reader = Callable[[Mapping[str, object]], object]
# (key the value is stored under, new reader, replaced rule)
_PARITY: dict[str, tuple[str, _Reader, _Reader]] = {
    "int_knob": (
        "k",
        lambda c: knobs.int_knob(c, "k", 7),
        lambda c: _old_tcgen05_config_int(c, "k", 7),
    ),
    "non_bool_int_knob": (
        "k",
        lambda c: knobs.non_bool_int_knob(c, "k", 7),
        lambda c: _old_warp_mma_config_int(c, "k", 7),
    ),
    "positive_int_knob": (
        "k",
        lambda c: knobs.positive_int_knob(c, "k", 7),
        lambda c: _old_config_positive_int(c, "k", 7),
    ),
    "parsed_int_knob": (
        "k",
        lambda c: knobs.parsed_int_knob(c, "k", 7),
        lambda c: int(str(c.get("k", 7))),
    ),
    "str_knob": (
        "k",
        lambda c: knobs.str_knob(c, "k", "d"),
        lambda c: _old_config_str(c, "k", "d"),
    ),
    "cluster_m_knob": (
        knobs.CLUSTER_M_KEY,
        knobs.cluster_m_knob,
        lambda c: _old_tcgen05_cluster(c, knobs.CLUSTER_M_KEY),
    ),
    "cluster_n_knob": (
        knobs.CLUSTER_N_KEY,
        knobs.cluster_n_knob,
        lambda c: _old_tcgen05_cluster(c, knobs.CLUSTER_N_KEY),
    ),
    "scheduler_cluster_m": (
        knobs.CLUSTER_M_KEY,
        lambda c: knobs.clamp_cluster_extent(
            knobs.parsed_int_knob(c, knobs.CLUSTER_M_KEY, 1)
        ),
        lambda c: _old_scheduler_cluster(c, knobs.CLUSTER_M_KEY),
    ),
    "launch_epi_warp_groups_knob": (
        TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY,
        knobs.launch_epi_warp_groups_knob,
        _old_tcgen05_epi_warp_groups,
    ),
    "epi_warp_groups_knob": (
        TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY,
        knobs.epi_warp_groups_knob,
        _old_config_epi_warp_groups,
    ),
    "epi_subtile_n_knob": (
        TCGEN05_EPI_SUBTILE_N_CONFIG_KEY,
        knobs.epi_subtile_n_knob,
        _old_config_epi_subtile_n,
    ),
    "epi_warps_knob": (
        TCGEN05_NUM_EPI_WARPS_CONFIG_KEY,
        knobs.epi_warps_knob,
        _old_config_epi_warps,
    ),
    "acc_stages_knob": (
        knobs.ACC_STAGES_KEY,
        knobs.acc_stages_knob,
        _old_config_acc_stages,
    ),
    "aux_stages_knob": (
        TCGEN05_AUX_STAGES_CONFIG_KEY,
        knobs.aux_stages_knob,
        _old_config_aux_stages,
    ),
    "aux_load_mode_knob": (
        TCGEN05_AUX_LOAD_MODE_CONFIG_KEY,
        knobs.aux_load_mode_knob,
        _old_config_aux_load_mode,
    ),
}


def _outcome(reader: _Reader, config: Mapping[str, object]) -> object:
    """The reader's value, or the exception type it raised (``parsed_int_knob``)."""
    try:
        value = reader(config)
    except (TypeError, ValueError) as error:
        return type(error)
    return (type(value), value)


@pytest.mark.parametrize("value", _VALUES, ids=_VALUE_IDS)
@pytest.mark.parametrize("name", sorted(_PARITY))
def test_reader_matches_the_rule_it_replaced(name: str, value: object) -> None:
    key, reader, replaced = _PARITY[name]
    config: dict[str, object] = {} if value is _ABSENT else {key: value}
    assert _outcome(reader, config) == _outcome(replaced, config)


def test_reader_rules_are_distinct() -> None:
    """The rules differ where the replaced sites differed; none was unified."""
    assert knobs.int_knob({"k": True}, "k", 7) is True
    assert knobs.non_bool_int_knob({"k": True}, "k", 7) == 7
    assert knobs.positive_int_knob({"k": True}, "k", 7) == 7
    assert knobs.int_knob({"k": 0}, "k", 7) == 0
    assert knobs.non_bool_int_knob({"k": 0}, "k", 7) == 0
    assert knobs.positive_int_knob({"k": 0}, "k", 7) == 7
    assert knobs.parsed_int_knob({"k": "2"}, "k", 7) == 2
    assert knobs.int_knob({"k": "2"}, "k", 7) == 7
    with pytest.raises(ValueError, match="invalid literal"):
        knobs.parsed_int_knob({"k": "x"}, "k", 7)
    # The launch reader admits only the warp-group choices; the config
    # layer's reader is any positive int.
    assert knobs.launch_epi_warp_groups_knob({"tcgen05_epi_warp_groups": 8}) == 1
    assert knobs.epi_warp_groups_knob({"tcgen05_epi_warp_groups": 8}) == 8
    # Clamping, not refusing: 3 is read as the two-CTA extent.
    assert knobs.cluster_m_knob({"tcgen05_cluster_m": 3}) == 2
    assert knobs.cluster_n_knob({"tcgen05_cluster_n": 0}) == 1


@pytest.mark.parametrize(
    "value",
    [None, 3, 0, True, "x", 2.5],
    ids=["absent", "3", "0", "True", "str", "float"],
)
def test_int_or_is_the_int_knob_rule(value: object) -> None:
    config = {} if value is None else {TCGEN05_SPLIT_K_CONFIG_KEY: value}
    knob_values = Tcgen05Knobs.from_config(config)
    assert knob_values.int_or(knob_values.split_k, 1) == knobs.int_knob(
        config, TCGEN05_SPLIT_K_CONFIG_KEY, 1
    )
