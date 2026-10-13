from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_scan_producer import _config
from .test_cute_chained_scan_producer import _scan_sequence
from .test_cute_chained_scan_producer import _sequence_args
from .test_cute_chained_scan_producer import _sequence_config
from helion import exc
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute import chained_scan_producer_emission as emission_module
from helion._compiler.cute.chained_matmul import _UnsupportedChain


def _native_config(enabled=True):
    config = _sequence_config()
    config.config["cute_chained_native_vector_reads"] = enabled
    return config


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_public_scan_native_reads_without_operand_retention(dtype):
    calls = []
    original = emission_module.emit_scan_producer

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.native_reads
        inputs = kwargs["native_inputs"]
        assert not inputs.activated
        assert len(inputs.inputs) == 1
        assert result.native_sources == inputs.inputs
        assert result.ownership is not None
        assert result.ownership.thread_order == "column_major"
        calls.append(inputs)
        return result

    with patch.object(emission_module, "emit_scan_producer", observe):
        source = _source(_scan_sequence, _sequence_args(dtype), _native_config())
    assert len(calls) == 1 and calls[0].activated
    assert "chain_scan_producer_" in source
    assert f"_input_{calls[0].inputs[0].index}_copy" in source
    assert "shuffle_sync_up" in source
    assert "cute_chained_operand_retention" not in _native_config().config


def test_native_off_does_not_attempt_binding_and_preserves_source():
    values = _sequence_args()
    expected = _source(_scan_sequence, values, _sequence_config())
    with patch(
        "helion._compiler.cute.chained_native_read_inputs.bind_preparation_leaf_native_inputs",
        side_effect=AssertionError("disabled native policy attempted binding"),
    ):
        assert _source(_scan_sequence, values, _native_config(False)) == expected


def test_declined_native_geometry_preserves_scalar_stage_without_activation():
    original = emission_module.emit_scan_producer
    calls = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and not result.native_reads
        assert result.native_sources == () and result.ownership is None
        assert "_input_" not in "\n".join(result.lines)
        calls.append(kwargs["native_inputs"])
        return result

    with (
        patch.object(emission_module, "emit_native_inputs", return_value=None),
        patch.object(emission_module, "emit_scan_producer", observe),
        pytest.raises(exc.BackendUnsupported, match="native"),
    ):
        _source(_scan_sequence, _sequence_args(), _native_config())
    assert len(calls) == 1 and not calls[0].activated


def test_failed_tail_does_not_activate_native_policy():
    original = pipeline_module._emit_scan_producer_stage
    calls = []

    def observe(*args, **kwargs):
        calls.append(kwargs["native_inputs"])
        return original(*args, **kwargs)

    with (
        patch.object(pipeline_module, "_emit_scan_producer_stage", observe),
        patch.object(
            emission_module,
            "emit_collectives_before",
            side_effect=_UnsupportedChain("test deferred publication failure"),
        ),
        pytest.raises(exc.BackendUnsupported),
    ):
        _source(_scan_sequence, _sequence_args(), _native_config())
    assert len(calls) == 1 and not calls[0].activated


def test_actual_kda_three_completed_leaves_use_same_public_lowering():
    kernel, values = _kda_fixture()
    config = _config()
    config.config["cute_chained_native_vector_reads"] = True
    original = emission_module.emit_scan_producer
    calls = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.native_reads
        assert len(kwargs["native_inputs"].inputs) == 3
        assert result.native_sources == kwargs["native_inputs"].inputs
        assert result.ownership is not None
        calls.append(kwargs["native_inputs"])
        return result

    with patch.object(emission_module, "emit_scan_producer", observe):
        source = _source(kernel, values, config)
    assert len(calls) == 1 and calls[0].activated
    assert "_input_2_copy" in source and "shuffle_sync_up" in source
