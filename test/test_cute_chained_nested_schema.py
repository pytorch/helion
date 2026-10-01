from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_nested_cache import _args
from .test_cute_chained_nested_cache import _nested_loop
from .test_cute_chained_nested_cache import _nested_root
from .test_cute_chained_prefill_search import _GROUP_OVERRIDES
from .test_cute_chained_prefill_search import _inputs
from .test_cute_chained_prefill_search import kda_prefill_native_math
from .test_cute_chained_prefill_search import kda_prefill_native_math_bt32
import helion
from helion import exc
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import (
    CUTE_CHAINED_POINTWISE_CACHE_BYTES_KEY as BUDGET,
)
from helion.autotuner.config_spec import (
    CUTE_CHAINED_POINTWISE_CACHE_ENTRIES_KEY as ENTRIES,
)
from helion.autotuner.config_spec import CUTE_CHAINED_POINTWISE_CACHE_NESTED_KEY as KEY

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


@pytest.mark.parametrize("chunk_size", [16, 32])
def test_legacy_prefill_has_no_nested_coordinate(chunk_size: int) -> None:
    kernel = (
        kda_prefill_native_math if chunk_size == 16 else kda_prefill_native_math_bt32
    )
    with _cpu_codegen():
        spec = kernel._bind_isolated(
            _inputs(heads=8, device=torch.device("cpu"))
        ).config_spec
        fields = spec._flat_fields()
        assert spec.cute_chained_pointwise_residency_search_enabled
        assert KEY not in fields and BUDGET not in fields and ENTRIES not in fields
        assert tuple(fields) == (
            "block_sizes",
            "cute_chunk_prefill_task_order",
            "cute_chunk_prefill_schedule",
        )
        default = spec.default_config()
        assert spec.normalized_config(default.config | {KEY: False}) == default
        generation = spec.create_config_generation()
        flat = generation.flatten(default)
        restored = generation.unflatten(flat)
        assert generation.flatten(restored) == flat
        # The original complete-root schema omits the fixed num_warps field.
        assert "num_warps" not in fields and default["num_warps"] == 4
        assert restored.config == {
            key: value for key, value in default.config.items() if key != "num_warps"
        }
        for repair in (False, True):
            with pytest.raises(exc.InvalidConfig, match="positive cache bytes"):
                spec.normalize(
                    helion.Config.from_dict({KEY: True}), _fix_invalid=repair
                )
            with pytest.raises(
                exc.InvalidConfig, match="explicit shared prefill family"
            ):
                spec.normalize(
                    helion.Config.from_dict({KEY: True, BUDGET: 4096, ENTRIES: 2}),
                    _fix_invalid=repair,
                )


def _assert_nested_schema(bound: BoundKernel, values: dict[str, object]) -> None:
    spec = bound.config_spec
    fields = spec._flat_fields()
    assert {BUDGET, ENTRIES, KEY} <= fields.keys()
    suffix = (KEY,)
    if "cute_chained_async_vector_store" in fields:
        suffix += ("cute_chained_async_vector_store",)
    if spec.cute_work_order_candidates:
        suffix += ("cute_grid_work_order",)
    suffix += ("cute_chained_fragment_epilogues",)
    assert tuple(fields)[-len(suffix) :] == suffix
    fragment = fields[KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment.default() is False
    default = spec.normalized_config(values)
    assert spec.normalized_config(values | {KEY: False}) == default
    assert all(
        seed.config.get(KEY, False) is False for seed in spec.compiler_seed_configs
    )
    selected = spec.normalized_config(values | {KEY: True, BUDGET: 16384, ENTRIES: 2})
    assert selected[KEY] is True
    generation = spec.create_config_generation()
    for config in (default, selected):
        assert generation.unflatten(generation.flatten(config)) == config


@pytest.mark.parametrize("loop", [False, True])
def test_real_root_and_loop_keep_nested_schema(loop: bool) -> None:
    kernel = _nested_loop if loop else _nested_root
    with _cpu_codegen():
        bound = kernel._bind_isolated(_args(loop, torch.bfloat16))
        _assert_nested_schema(
            bound, {"num_warps": 4, "cute_chained_mma_schedule": "coalesced"}
        )


@pytest.mark.parametrize("chunk_size", [16, 32])
def test_explicit_grouped_family_keeps_nested_schema(chunk_size: int) -> None:
    original = (
        kda_prefill_native_math if chunk_size == 16 else kda_prefill_native_math_bt32
    )
    grouped = helion.kernel(
        original.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides=_GROUP_OVERRIDES,
    )
    with _cpu_codegen():
        bound = grouped._bind_isolated(_inputs(heads=8, device=torch.device("cpu")))
        assert bound.config_spec.cute_chunk_prefill_task_order is None
        _assert_nested_schema(
            bound, {"block_sizes": [128], "num_warps": 4, **_GROUP_OVERRIDES}
        )


@pytest.mark.parametrize("missing", [BUDGET, ENTRIES])
def test_nested_requires_each_actual_schema_prerequisite(missing: str) -> None:
    with _cpu_codegen():
        spec = _nested_root._bind_isolated(_args(False, torch.bfloat16)).config_spec
        fields = spec._flat_fields_without_native_metadata()
        assert BUDGET in fields and ENTRIES in fields
        fields.pop(missing)
        with patch.object(
            spec, "_flat_fields_without_native_metadata", return_value=fields
        ):
            assert spec.cute_chained_pointwise_residency_search_enabled
            assert KEY not in spec._flat_fields()
