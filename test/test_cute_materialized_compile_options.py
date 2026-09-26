from __future__ import annotations

import copy
from typing import cast

import pytest

import helion
from helion._compiler.cute.materialized_fission import MaterializedFissionPlan
from helion._compiler.cute.materialized_fission_codegen import _stage_config
from helion._compiler.cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY


def _plan(pointwise: tuple[int, ...]) -> MaterializedFissionPlan:
    return MaterializedFissionPlan(
        root_index=0,
        source_root_key="original body",
        materialized_names=("operand",),
        region_count=2,
        pointwise_region_indices=pointwise,
    )


def test_pointwise_compile_option_is_private_and_consumer_is_unchanged() -> None:
    config = helion.Config(
        block_sizes=[16, 32, 256, 256, 128],
        l2_groupings=[8, 2],
        loop_orders=[[0, 1], [1, 0]],
        tcgen05_tvm_ffi_launch=True,
    )
    saved = copy.deepcopy(config.config)
    producer = _stage_config(config, _plan((0,)), 0)
    consumer = _stage_config(config, _plan((0,)), 1)
    assert producer is not config and producer.config is not config.config
    assert producer.config == saved | {TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: False}
    assert consumer is config and consumer.config == saved
    producer.block_sizes[0] = 64
    cast("list[list[int]]", producer.config["loop_orders"])[0].reverse()
    assert config.config == saved


@pytest.mark.parametrize("value", [None, False, True, 1])
@pytest.mark.parametrize("pointwise", [(), (0,)])
def test_only_explicit_true_on_certified_pointwise_stage_is_projected(
    value: object, pointwise: tuple[int, ...]
) -> None:
    values: dict[str, object] = {"block_sizes": [16, 16, 256, 256, 128]}
    if value is not None:
        values[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY] = value
    config = helion.Config.from_dict(values)
    original = copy.deepcopy(config.config)
    result = _stage_config(config, _plan(pointwise), 0)
    if value is True and pointwise:
        assert result is not config
        assert result.config == original | {TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: False}
    else:
        assert result is config
    assert config.config == original


@pytest.mark.parametrize("pointwise_index", [0, 1])
@pytest.mark.parametrize("ffi", [False, True])
def test_flat_override_is_private_to_the_proved_pointwise_stage(
    pointwise_index: int, ffi: bool
) -> None:
    values = {
        "block_sizes": [1, 1024, 128, 128, 64],
        "pid_type": "persistent_blocked",
        "num_sm_multiplier": 2,
        "maxnreg": 128,
        "tcgen05_cta_group": "two",
        "tcgen05_tvm_ffi_launch": ffi,
    }
    config = helion.Config.from_dict(values | {"cute_pointwise_pid_type": "flat"})
    saved = copy.deepcopy(config.config)
    plan = _plan((pointwise_index,))
    producer = _stage_config(config, plan, pointwise_index)
    consumer = _stage_config(config, plan, 1 - pointwise_index)
    expected = values | {"pid_type": "flat", "tcgen05_tvm_ffi_launch": False}
    expected.pop("num_sm_multiplier")
    expected.pop("maxnreg")
    assert producer.config == expected
    assert consumer.config == values
    producer.block_sizes[0] = 8
    assert config.config == saved and consumer.config == values
