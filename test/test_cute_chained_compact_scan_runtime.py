from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_preparation_storage_runtime as storage_tests
from helion._compiler.cute import chained_preparation_storage as storage_module
from helion._testing import skipUnlessBackends


@contextmanager
def _native_storage():
    original_config = storage_tests._sequence_config
    original_bind = storage_module.bind_preparation_storage
    bindings = []

    def config():
        selected = original_config()
        selected.config["cute_chained_native_vector_reads"] = True
        return selected

    def bind(*args, **kwargs):
        result = original_bind(*args, **kwargs)
        assert result is not None and len(result.scan_transfers) == 1
        transfer = result.scan_transfers[0]
        assert transfer.receipt.emission.native_sources
        assert transfer.receipt in result.physical.accepted.scan_transfers
        bindings.append(result)
        return result

    with (
        patch.object(storage_tests, "_sequence_config", config),
        patch.object(storage_module, "bind_preparation_storage", bind),
    ):
        yield
    assert len(bindings) == 1 and bindings[0]._state.consumed
    accepted = bindings[0].physical.accepted
    assert bindings[0].matches(accepted.revision.plan, accepted.pipeline)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("selected", (0, 17, 31))
@pytest.mark.parametrize("steps", (0, 1, 7))
def test_compact_native_scan_source_preflight(dtype, selected, steps):
    with _native_storage():
        storage_tests.test_preparation_storage_runtime_fixture_source_preflight(
            dtype, selected, steps
        )


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("selected", (0, 17, 31))
@pytest.mark.parametrize("steps", (0, 1, 7))
def test_compact_native_scan_values_generations_and_replay_gpu(dtype, selected, steps):
    # Reuse the original complete oracle, two allocation generations, eager
    # bitwise comparison, two independent poisoned graphs and immutable inputs.
    # Only the public native-read policy is added to both original configs.
    with _native_storage():
        storage_tests.test_preparation_storage_original_values_generations_and_replay_gpu(
            dtype, selected, steps
        )
