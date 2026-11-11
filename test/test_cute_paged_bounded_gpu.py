from __future__ import annotations

from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_optional_lse import initialize_paged_attention_inputs
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_bounded import bounded_merge_args
from helion._testing.cute_paged_bounded import bounded_partial_fixture
from helion._testing.cute_paged_bounded import bounded_partial_merge


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "rows,fast_math,query_tma", [(1, False, False), (4, True, True), (5, False, True)]
)
def test_raw_statistics_match_written_source_and_changed_graph(
    rows, fast_math, query_tma
):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the paired attention transport uses SM100")
    kernel, initial = bounded_partial_fixture(rows, DEVICE, reverse_pages=rows != 1)
    args = list(initial)
    args[7] = 31 if rows == 4 else None
    values = tuple(args)
    q, k, v, ks, vs, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    output, first, last = (cast("torch.Tensor", arg) for arg in args[10:13])
    initialize_paged_attention_inputs(values)
    lengths.copy_(lengths.new_tensor([256, 257]))
    bound = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        fast_math=fast_math,
    ).bind(values)
    reference = helion.kernel(
        kernel.fn, backend="cute", ref_mode=helion.RefMode.EAGER
    ).bind(values)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(rows * 8)],
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_mma_dag_query_tma=query_tma,
        cute_pdl=rows == 5,
        cute_pdl_capture_overlap=rows == 5,
    )
    assert "maximum_output=" in bound.to_code(config)
    compiled = bound.compile_config(config)
    compiled(*values)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            captured = compiled(*values)
        # One compile covers independent input updates, both empty orderings and
        # the complete exceptional decode path. Raw O is compared in unscaled
        # FP64 units so its 2^-22 representation cannot hide an absolute error.
        for case in (
            "ordinary",
            "changed",
            "empty",
            "inf_v",
            "nan_v",
            "nan_q",
            "max_v",
        ):
            initialize_paged_attention_inputs(values)
            lengths.copy_(lengths.new_tensor([256, 257]))
            first.zero_()
            last.fill_(3)
            if case == "changed":
                q.neg_()
                pages.copy_(pages.flip(1))
                first.copy_(first.new_tensor([0, 1]))
                last.copy_(last.new_tensor([2, 3]))
            elif case == "empty":
                first.copy_(first.new_tensor([2, 3]))
                last.copy_(last.new_tensor([1, 3]))
            elif case in ("inf_v", "nan_v", "max_v"):
                v.fill_(0 if case == "nan_v" else 127)
                vs.fill_(248 if case == "max_v" else 255)
                if case == "max_v":
                    q.zero_()
                    k.zero_()
            elif case == "nan_q":
                q.fill_(float("nan"))
            reference_args = (*values[:10], torch.empty_like(output), *values[11:])
            expected = reference.run_ref(*reference_args)
            direct = compiled(*values)
            for actual, wanted in zip(direct, expected, strict=True):
                factor = 2.0**22 if actual is direct[0] else 1.0
                torch.testing.assert_close(
                    actual.double() * factor,
                    wanted.double() * factor,
                    rtol=0.002,
                    atol=0.002,
                    equal_nan=True,
                    msg=f"{case}, direct, gain={factor}: {{}}".format,
                )
            for tensor in captured:
                tensor.fill_(float("nan"))
            graph.replay()
            for actual, wanted in zip(captured, expected, strict=True):
                factor = 2.0**22 if actual is captured[0] else 1.0
                torch.testing.assert_close(
                    actual.double() * factor,
                    wanted.double() * factor,
                    rtol=0.002,
                    atol=0.002,
                    equal_nan=True,
                    msg=f"{case}, graph, gain={factor}: {{}}".format,
                )


@skipIfNotCUDA()
def test_raw_merge_preserves_mass_empty_nonfinite_and_finite_clamp():
    args = bounded_merge_args(DEVICE)
    raw0, maximum0, denominator0, raw1, maximum1, denominator1 = args
    bound = bounded_partial_merge.bind(args)
    reference = helion.kernel(
        bounded_partial_merge.fn,
        backend="cute",
        static_shapes=True,
        ref_mode=helion.RefMode.EAGER,
    ).bind(args)
    compiled = bound.compile_config(helion.Config(block_sizes=[1]))
    compiled(*args)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            captured = compiled(*args)
        for changed in (False, True):
            if changed:
                raw0[0].neg_()
                denominator1[0].fill_(7)
            expected = reference.run_ref(*args)
            direct = compiled(*args)
            for tensor in captured:
                tensor.fill_(float("nan"))
            graph.replay()
            for actual in (direct, captured):
                torch.testing.assert_close(
                    actual[0], expected[0], rtol=0, atol=0, equal_nan=True
                )
                torch.testing.assert_close(
                    actual[1], expected[1], rtol=0.002, atol=0.002, equal_nan=True
                )
                assert (actual[0][0] == (-8 if changed else 16)).all()
                assert torch.isneginf(actual[1][1]).all()
