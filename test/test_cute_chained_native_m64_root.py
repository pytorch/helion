from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_m64 import _batched
from .test_cute_chained_m64 import _m64_config
from .test_cute_chained_m64 import _m64_single
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_root_stage as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._compiler.cute.chained_result_transport import load_operation


def _source(dtype=torch.bfloat16, width=128, *, scan=False, direct=True, unroll=2):
    arguments = (
        (
            torch.empty((2, 128, 128), dtype=dtype),
            torch.empty((2, 128, width), dtype=dtype),
            torch.empty((128,), dtype=torch.float32),
        )
        if scan
        else (
            torch.empty((128, 128), dtype=dtype),
            torch.empty((128, width), dtype=dtype),
        )
    )
    config = _m64_config(
        cute_chained_direct_output=direct,
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_unroll=unroll,
    )
    with _cpu_codegen():
        return (
            (_batched if scan else _m64_single)
            ._bind_isolated(arguments)
            .to_code(config)
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("width,unroll", ((32, 2), (96, 1), (128, 2)))
@pytest.mark.parametrize("scan,direct", ((False, False), (True, True)))
def test_native_m64_uses_common_stage_with_exact_original_source(
    dtype, width, unroll, scan, direct
):
    initialized = torch.cuda.is_initialized()
    with patch.object(roots, "supports_independent_root", return_value=False):
        before = _source(dtype, width, scan=scan, direct=direct, unroll=unroll)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = _source(dtype, width, scan=scan, direct=direct, unroll=unroll)
    assert after == before
    assert emitted.call_count == 1
    geometry = emitted.call_args.args[4]
    assert geometry.native_rows == 64
    assert geometry.physical == geometry.logical == (64, width, 128)
    assert emitted.call_args.kwargs["root_actions"].sequence.independent is not None
    assert emitted.call_args.kwargs["terminal_fragment"] is True
    assert load_operation((64, width)) in after
    assert ("chain_output_ptr =" not in after) == direct
    if scan:
        assert after.index("chain_scan_0_pointer =") < after.index("chain_0_mma =")
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("rows", (True, 0, 32, 64.0, 256))
def test_native_rows_require_an_explicit_supported_integer(rows):
    with pytest.raises(ValueError, match="native contraction rows"):
        stages.StageGeometry((64, 128, 128), False, native_rows=rows)


@pytest.mark.parametrize(
    "shape,transpose",
    (
        ((128, 128, 128), False),
        ((64, 48, 128), False),
        ((64, 128, 15), False),
        ((64, 128, 128), True),
    ),
)
def test_native_m64_rejects_padded_or_transposed_geometry(shape, transpose):
    with pytest.raises(ValueError, match="M64 geometry"):
        stages.StageGeometry(shape, transpose, native_rows=64)


def test_default_geometry_still_uses_original_m128_padding():
    assert stages.StageGeometry((64, 128, 128), False).physical == (128, 128, 128)
    assert stages.stage_geometry((64, 128, 128)) == stages.StageGeometry(
        (64, 128, 128), True
    )


@pytest.mark.parametrize(
    "case",
    (
        "without_action",
        "shape",
        "plan",
        "axes",
        "coupled_axes",
        "readiness",
        "boundaries",
        "options",
        "participants",
        "explicit_accumulator",
        "stage",
        "plan_axes",
        "scan_schedule",
        "loop",
    ),
)
def test_native_m64_rejects_stale_provenance_before_stage_publication(case):
    original = stages.emit_stage
    checked = []

    def inspect(*args, **kwargs):
        changed, options = list(args), dict(kwargs)
        action = kwargs["root_actions"]
        sequence = action.sequence
        axes, readiness = sequence.inner_axes, sequence.input_readiness
        inputs = sequence.independent
        old_plan_axes = args[1].axes
        old_early_scan = sequence.early_scan
        old_config = dict(args[0].device_function.config.config)
        before = (
            sequence.next_stage,
            list(sequence.staged),
            dict(args[2]),
            list(args[0].device_function.body),
        )
        try:
            if case == "without_action":
                options["root_actions"] = None
            elif case == "shape":
                changed[4] = stages.StageGeometry((64, 32, 128), False, native_rows=64)
            elif case == "plan":
                changed[1] = replace(args[1])
            elif case == "axes":
                sequence.inner_axes = ((1, 1),)
            elif case == "coupled_axes":
                assert inputs is not None
                sequence.inner_axes = ((1, 1),)
                sequence.independent = replace(inputs, axes=sequence.inner_axes)
            elif case == "readiness":
                sequence.input_readiness = ()
            elif case == "boundaries":
                changed[2] = {}
            elif case == "options":
                args[0].device_function.config.config[
                    "cute_chained_pointwise_unroll"
                ] = 4
            elif case == "participants":
                changed[1] = replace(args[1], threads=256)
            elif case == "explicit_accumulator":
                changed[1] = replace(args[1], initialized_accumulator=object())
            elif case == "plan_axes":
                object.__setattr__(args[1], "axes", ())
            elif case == "scan_schedule":
                sequence.early_scan = not sequence.early_scan
            elif case == "loop":
                changed[1] = replace(args[1], loop=object())
            else:
                changed[3] = 1
            with pytest.raises(chain._UnsupportedChain):
                original(*changed, **options)
            assert (
                sequence.next_stage,
                sequence.staged,
                args[2],
                args[0].device_function.body,
            ) == before
        finally:
            sequence.inner_axes, sequence.input_readiness = axes, readiness
            sequence.independent = inputs
            object.__setattr__(args[1], "axes", old_plan_axes)
            sequence.early_scan = old_early_scan
            args[0].device_function.config.config.clear()
            args[0].device_function.config.config.update(old_config)
        checked.append(case)
        return original(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=inspect):
        _source(scan=True)
    assert checked == [case]
