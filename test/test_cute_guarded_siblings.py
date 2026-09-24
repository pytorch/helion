from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from helion._compiler.autotuner_heuristics import cute_guarded_siblings
from helion.runtime.config import Config

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR


@pytest.mark.parametrize("parents", [[], [Config()], [Config(block_sizes=[1, 256])]])
def test_unrelated_search_does_not_access_host_or_discover(parents):
    env = cast(
        "CompileEnvironment",
        SimpleNamespace(
            config_spec=SimpleNamespace(
                cute_serial_lane_schedule_enabled=False,
                cute_chained_tcgen05_search_enabled=False,
            )
        ),
    )
    # No host_function exists: the disabled-family gate must precede even its
    # lookup, not merely suppress a later missing-environment exception.
    ir = cast("DeviceIR", object())
    before = [(id(parent), dict(parent.config)) for parent in parents]
    with (
        patch.object(cute_guarded_siblings, "discover") as discover,
        patch.object(cute_guarded_siblings, "detect_chained_matmul_search") as detect,
    ):
        assert cute_guarded_siblings.guarded_schedule_siblings(env, ir, parents) == []
        discover.assert_not_called()
        detect.assert_not_called()
    assert [(id(parent), parent.config) for parent in parents] == before


@pytest.mark.parametrize("chained_enabled", [False, True])
def test_enabled_serial_preserves_order_multiplicity_and_parent_objects(
    chained_enabled,
):
    spec = SimpleNamespace(
        cute_serial_lane_schedule_enabled=True,
        cute_chained_tcgen05_search_enabled=chained_enabled,
    )
    env = cast("CompileEnvironment", SimpleNamespace(config_spec=spec))
    ir = cast("DeviceIR", SimpleNamespace(host_function=nullcontext()))
    parent = Config(
        cute_serial_lane_schedule="step_major_vector",
        cute_serial_lane_load_schedule="prefetch4",
        cute_vector_widths=[4],
    )
    parents = [Config(), parent, parent]
    before = [(id(p), dict(p.config)) for p in parents]
    with patch.object(
        cute_guarded_siblings, "discover", return_value=SimpleNamespace(steps=8)
    ) as discover:
        result = cute_guarded_siblings.guarded_schedule_siblings(env, ir, parents)
    discover.assert_called_once_with(ir)
    peeled = parent.config | {cute_guarded_siblings.TAIL_KEY: "peel_final_group"}
    assert [p.config for p in result] == [
        peeled,
        peeled,
        parent.config | {cute_guarded_siblings.FAST_KEY: True},
        parent.config | {cute_guarded_siblings.FAST_KEY: True},
        peeled | {cute_guarded_siblings.FAST_KEY: True},
        peeled | {cute_guarded_siblings.FAST_KEY: True},
    ]
    assert [(id(p), p.config) for p in parents] == before


def test_enabled_chained_keeps_first_two_geometries_and_normalization():
    normalize = Mock()
    spec = SimpleNamespace(
        cute_serial_lane_schedule_enabled=False,
        cute_chained_tcgen05_search_enabled=True,
        normalize=normalize,
    )
    env = cast("CompileEnvironment", SimpleNamespace(config_spec=spec))
    ir = cast("DeviceIR", SimpleNamespace(host_function=nullcontext(), graphs=[]))
    common = {
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_pointwise_vectorize": True,
    }
    first = Config.from_dict(common | {"block_sizes": [128, 32, 128]})
    second = Config.from_dict(common | {"block_sizes": [128, 64, 128]})
    third = Config.from_dict(common | {"block_sizes": [128, 128, 128]})
    parents = [first, first, second, third]
    before = [(id(p), dict(p.config)) for p in parents]
    with (
        patch.object(cute_guarded_siblings, "discover", return_value=None),
        patch.object(
            cute_guarded_siblings, "detect_chained_matmul_search", return_value=True
        ),
    ):
        result = cute_guarded_siblings.guarded_schedule_siblings(env, ir, parents)
    assert [p.config for p in result] == [
        first.config | {cute_guarded_siblings.FAST_KEY: True},
        second.config | {cute_guarded_siblings.FAST_KEY: True},
    ]
    assert normalize.call_count == 2
    assert [(id(p), p.config) for p in parents] == before
