from __future__ import annotations

from typing import cast

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_optional_lse import initialize_paged_attention_inputs
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_reference import check_reference_attention_fixture
from helion._testing.cute_paged_reference import reverse_attention_fixture

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
driver = pytest.importorskip("cuda.bindings.driver")
attention_page_coordinates = pytest.importorskip(
    "helion._compiler.cute._paged_attention_partial_pipeline"
).attention_page_coordinates


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,query_len,fast_math,lse,scoped,occupancy,pdl,index_dtype,reference_scores",
    [
        ("mapped", 1, False, True, False, 1, False, torch.int32, True),
        ("mapped", 4, True, False, True, 2, True, torch.int64, True),
        ("mapped", 5, False, True, True, 2, False, torch.int32, True),
        ("per_key", 1, True, False, False, 2, False, torch.int64, True),
        ("per_key", 4, False, True, False, 1, True, torch.int32, True),
        ("per_key", 5, True, True, True, 2, False, torch.int64, True),
        ("mapped", 4, True, False, False, 1, False, torch.int64, False),
        ("per_key", 5, False, True, True, 2, False, torch.int32, False),
    ],
)
def test_reverse_attention_source_and_changed_input_replay(
    form,
    query_len,
    fast_math,
    lse,
    scoped,
    occupancy,
    pdl,
    index_dtype,
    reference_scores,
):
    # Preserve the six reference-score cases and add each ordinary-score layout.
    # Policy axes use pairwise coverage instead of a Cartesian product.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    kernel, args = reverse_attention_fixture(
        form, query_len, lse, DEVICE, reference_scores=reference_scores
    )
    check_reference_attention_fixture(
        kernel,
        args,
        form,
        query_len,
        fast_math,
        lse,
        "attention_scoped" if scoped else "attention",
        occupancy,
        pdl,
        reference_scores=reference_scores,
        reverse_pages=True,
        index_dtype=index_dtype,
    )


@cute.kernel
def _coordinates_kernel(inputs, output, dtype: cutlass.Constexpr):
    case = cute.arch.thread_idx()[0]
    if case < inputs.shape[0]:
        page, key_base = attention_page_coordinates(
            inputs[case, 0], inputs[case, 1], inputs[case, 2], dtype
        )
        output[case, 0] = page
        output[case, 1] = key_base


@cute.jit
def _coordinates_launch(
    inputs, output, dtype: cutlass.Constexpr, stream: driver.CUstream
):
    _coordinates_kernel(inputs, output, dtype).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream
    )


@skipIfNotCUDA()
@pytest.mark.parametrize("bits", [32, 64])
def test_reverse_integer_operation_order_and_key_product(bits):
    # Address-free proof includes signed overflow that cannot safely be turned
    # into an attention table allocation, plus changed-input graph replay.
    cases = [
        (1, 4, 0),
        (1, 4, 2),
        (2**31 - 64, 2**31 - 1, 17),
        (-(2**31), -(2**31) + 128, 0),
        (2**63 - 64, 2**63 - 1, 1),
        (-(2**63), -(2**63) + 64, 0),
        (2**32 + 1, 2**32 + 4, 1),
        (0, 1, 0),
    ]
    inputs = torch.tensor(cases, dtype=torch.int64, device=DEVICE)
    output = torch.empty((len(cases), 2), dtype=torch.int64, device=DEVICE)
    tensors = tuple(cute.runtime.from_dlpack(x) for x in (inputs, output))
    dtype = cutlass.Int32 if bits == 32 else cutlass.Int64
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(_coordinates_launch, *tensors, dtype, stream)
    compiled(*tensors, stream)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            compiled(*tensors, driver.CUstream(torch.cuda.current_stream().cuda_stream))

        def wrap(value, width=bits):
            sign = 1 << (width - 1)
            return (value + sign) % (2 * sign) - sign

        for changed in (False, True):
            if changed:
                inputs[:, 2].add_(1)
            graph.replay()
            expected = []
            for first, last, step in inputs.cpu().tolist():
                prefix = wrap(wrap(wrap(first) + wrap(last)) - 1)
                token = wrap(prefix - wrap(first + step, 64))
                expected.append((token, wrap(token * 128)))
            torch.testing.assert_close(
                output,
                torch.tensor(expected, dtype=torch.int64, device=DEVICE),
                rtol=0,
                atol=0,
            )


@skipIfNotCUDA()
@pytest.mark.parametrize("schedule", ["attention", "attention_paired"])
def test_reversed_int64_empty_then_nonempty_interval_replay(schedule):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    kernel, args = reverse_attention_fixture("per_key", 4, True, DEVICE)
    args = list(args)
    initialize_paged_attention_inputs(tuple(args))
    q, k, v, ks, vs = (cast("torch.Tensor", arg) for arg in args[:5])
    q.zero_()
    k.zero_()
    v.fill_(1)
    ks.fill_(127)
    vs.fill_(127)
    args[9] = -1  # full context; isolate typed traversal from window clipping
    bounds = torch.tensor([[1, -(2**63)], [2**63 - 1, -1]], device=DEVICE)
    args[10] = bounds
    args = tuple(args)
    bound = kernel.bind(args)
    config = optional_lse_config()
    config.config["cute_mma_dag_pass_warps"] = schedule
    source = bound.to_code(config)
    assert "reflected_dtype=cutlass.Int64" in source
    if schedule == "attention_paired":
        assert "paged_attention_paired_body(" in source
    compiled = bound.compile_config(config)
    compiled(*args)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            captured = compiled(*args)
        for changed in (False, True):
            if changed:
                bounds.copy_(bounds.new_tensor([[0, 0], [0, 2]]))
            direct = compiled(*args)
            graph.replay()
            for output, lse in (direct, captured):
                if not changed:
                    assert torch.count_nonzero(output).item() == 0
                    assert torch.isneginf(lse).all().item()
                else:
                    assert torch.count_nonzero(output[:2]).item() == 0
                    assert torch.isneginf(lse[:2]).all().item()
                    torch.testing.assert_close(
                        output[2:], torch.ones_like(output[2:]), rtol=0, atol=0
                    )
                    # Request 1 length 257, Q4/group8: row positions 253..256.
                    # Two reversed pages expose keys 0..255; count each causal row.
                    count = (254 + torch.arange(32, device=DEVICE) // 8).clamp_max(256)
                    expected = count.float().log().expand(2, -1)
                    torch.testing.assert_close(lse[2:], expected, rtol=0, atol=1e-6)
