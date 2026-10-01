from __future__ import annotations

from benchmarks.cute.kda_prefill_fused import kda_prefill_native_math
from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32
import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chunk_prefill import _inputs
import helion
from helion import exc
from helion.autotuner.config_spec import CUTE_CHAINED_GROUP_CONTRACTIONS_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_PREPARATION_PIPELINE_KEY
from helion.autotuner.config_spec import CUTE_NATIVE_MATMUL_METADATA_KEY


@pytest.fixture(params=(16, 32))
def prefill_bound(request: pytest.FixtureRequest):
    kernel = (
        kda_prefill_native_math if request.param == 16 else kda_prefill_native_math_bt32
    )
    with _cpu_codegen():
        bound = kernel._bind_isolated(_inputs(heads=8, device=torch.device("cpu")))
        assert bound.config_spec.cute_chained_matmul_search_enabled
        assert bound.config_spec.cute_chunk_prefill_task_order is not None
        yield bound


def test_whole_root_has_only_effective_fields_and_exact_opt_out(prefill_bound):
    spec = prefill_bound.config_spec
    assert tuple(spec._flat_fields()) == (
        "block_sizes",
        "cute_chunk_prefill_task_order",
        "cute_chunk_prefill_schedule",
    )
    absent = helion.Config(block_sizes=[64])
    disabled = helion.Config(block_sizes=[64], cute_native_matmul_metadata=False)
    assert spec.normalized_config(absent) == spec.normalized_config(disabled)
    source = prefill_bound.to_code(absent)
    assert source == prefill_bound.to_code(disabled)
    assert "'kind': 'chunk_prefill_sm100'" in source
    assert "_helion_cute_native_metadata_specialization = True" not in source


@pytest.mark.parametrize("repair", (False, True))
def test_whole_root_metadata_is_rejected_not_silently_repaired(prefill_bound, repair):
    config = helion.Config(block_sizes=[64], cute_native_matmul_metadata=True)
    with pytest.raises(exc.InvalidConfig, match="not whole-root prefill"):
        prefill_bound.config_spec.normalize(config, _fix_invalid=repair)
    with pytest.raises(exc.InvalidConfig, match="not whole-root prefill"):
        prefill_bound.to_code(config)


@pytest.mark.parametrize(
    "selector",
    (CUTE_CHAINED_GROUP_CONTRACTIONS_KEY, CUTE_CHAINED_PREPARATION_PIPELINE_KEY),
)
def test_explicit_common_selection_preserves_metadata_obligation(
    prefill_bound, selector
):
    # Either selector bypasses whole-root planning in CuTeBackend. Other option
    # validation and the final native ledger still decide whether it can lower.
    config = {CUTE_NATIVE_MATMUL_METADATA_KEY: True, selector: True}
    prefill_bound.config_spec._normalize_cute_native_matmul_metadata(config)
    assert config == {CUTE_NATIVE_MATMUL_METADATA_KEY: True, selector: True}
    for inactive in (False, None, 0, 1):
        with pytest.raises(exc.InvalidConfig, match="not whole-root prefill"):
            prefill_bound.config_spec._normalize_cute_native_matmul_metadata(
                {CUTE_NATIVE_MATMUL_METADATA_KEY: True, selector: inactive}
            )


def test_explicit_common_incomplete_coverage_still_rejects(prefill_bound):
    config = helion.Config(
        block_sizes=[64],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_native_matmul_metadata=True,
    )
    normalized = prefill_bound.config_spec.normalized_config(config)
    assert normalized[CUTE_NATIVE_MATMUL_METADATA_KEY] is True
    with pytest.raises(exc.BackendUnsupported, match="incomplete committed native"):
        prefill_bound.to_code(normalized)
