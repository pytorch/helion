from __future__ import annotations

from unittest.mock import patch

import pytest

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_register_emission import _register_config
from helion import exc
from helion._compiler.cute import chained_pipeline_storage as storage


@pytest.mark.parametrize("tma", [False, True])
def test_narrow_values_keep_exact_capacity_guard_and_admit_two_cohorts_cpu(tma):
    kernel, args = _kda_fixture()
    config = _register_config(
        3,
        block_sizes=[64],
        cute_chained_seed_tile_columns=32,
        cute_chained_pointwise_cache_layout="xor",
        cute_chained_vector_group=True,
        cute_chained_leaf_pipeline="rectangular_tma" if tma else "legacy",
    )
    original = storage.finalize_pipeline_storage
    observed = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        diagnostic = original(*args, **{**kwargs, "capacity_bytes": 1 << 20})
        assert result is None and diagnostic is not None
        observed.append((kwargs["capacity_bytes"], diagnostic.charged_bytes))
        return result

    with (
        patch.object(storage, "finalize_pipeline_storage", side_effect=observe),
        pytest.raises(
            exc.BackendUnsupported, match="complete post-transport allocation"
        ),
    ):
        _source(kernel, args, config)
    assert observed == [(232448, 248448)]
    config.config.update(
        cute_chained_preparation_cohorts=2,
        cute_chained_pipeline_consumer_warps=8,
    )
    source = _source(kernel, args, config)
    assert "chain_cohort = chain_thread // 128" in source
    assert "chain_thread < 256" in source
    assert "chain_generation = chain_iteration // 2" in source
