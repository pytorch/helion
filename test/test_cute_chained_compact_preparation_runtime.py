from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_preparation_storage_runtime as storage_tests
from ._cute_aux import _cpu_codegen
from .test_cute_chained_compact_preparation import _twenty
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_scan_producer import _scan_sequence
from .test_cute_chained_scan_producer import _sequence_config
from helion._compiler.cute import chained_preparation_storage as storage_module
from helion._testing import skipUnlessBackends


def _compile_public(values, compact, *, source_only=False):
    config = _sequence_config()
    config.config["cute_chained_native_vector_reads"] = True
    if compact:
        config.config.update(_twenty().config)
    original_bind = storage_module.bind_preparation_storage
    bindings = []

    def observe(*args, **kwargs):
        result = original_bind(*args, **kwargs)
        assert result is not None
        bindings.append(result)
        return result

    # Only observe the existing physical binder. Selection is entirely public;
    # neither the emitter nor its compact flag is monkeypatched for this test.
    with patch.object(storage_module, "bind_preparation_storage", observe):
        bound = _scan_sequence._bind_isolated(values)
        if source_only:
            with bound.env.use_runtime_arg_values(
                _runtime_values(_scan_sequence, values)
            ):
                result = bound.to_code(config)
        else:
            result = bound.compile_config(config)
    assert len(bindings) == int(compact)
    if compact:
        physical = bindings[0]
        accepted = physical.physical.accepted
        assert physical._state.consumed
        assert physical.matches(accepted.revision.plan, accepted.pipeline)
        assert accepted.execution.threads == 128
        assert accepted.pipeline.slots == 4
        assert accepted.pipeline.cohorts is not None
        assert accepted.pipeline.cohorts.cta_threads == 640
        assert accepted.pipeline.cohorts.recurrence_threads == 128
        assert accepted.pipeline.cohorts.cohort_threads == 128
        assert len(physical.scan_transfers) == 1
        assert physical.scan_transfers[0].receipt in accepted.scan_transfers
        final = physical._state.finalized
        assert final is not None and final.preparation is physical
        assert dict(final.allocations)["frames"] == 4 * physical.stride
    return result


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("selected", (0, 17, 31))
@pytest.mark.parametrize("steps", (0, 1, 7, 9))
def test_public_compact_cohorts_source_preflight(dtype, selected, steps):
    values = storage_tests._fixture("cpu", dtype, steps, selected)
    with _cpu_codegen():
        source = _compile_public(values, True, source_only=True)
    assert isinstance(source, str)
    assert "block=(640, 1, 1)" in source
    assert "_helion_cute_min_blocks_per_mp = 1" in source
    assert "chain_generation = chain_iteration // 4" in source
    assert "chain_recurrence_thread = chain_thread - 512" in source
    assert "chain_preparation_workspace_" in source


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("selected", (0, 17, 31))
@pytest.mark.parametrize("steps", (0, 1, 7, 9))
def test_public_compact_cohorts_values_generations_and_replay_gpu(
    dtype, selected, steps
):
    # Preserve the complete original FP64, bitwise, two-allocation-generation,
    # eager repeat, poisoned graph replay and immutable-input oracle. Replace
    # only its compile factory with public16-warp/20-warp policy selection.
    with patch.object(storage_tests, "_compile", _compile_public):
        storage_tests.test_preparation_storage_original_values_generations_and_replay_gpu(
            dtype, selected, steps
        )
