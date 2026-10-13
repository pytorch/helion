from __future__ import annotations

import pytest
import torch

from .test_cute_chained_loop_tmem_carry_transport import _resident_carry_sequence
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_preparation_pipeline import _args
from .test_cute_chained_preparation_pipeline import _config
from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("steps", [0, 1, 4, 7])
@pytest.mark.parametrize("late", [False, True])
def test_three_cohorts_preserve_typed_recurrence_and_ragged_generations_gpu(
    steps: int, late: bool
) -> None:
    args = _args(DEVICE, steps, late)
    inputs = tuple(value for value in args if isinstance(value, torch.Tensor))
    assert len(inputs) == 4
    saved = tuple(value.clone() for value in inputs)
    compiled = []
    for count in (1, 3):
        config = _config(16, pipeline=True)
        if count != 1:
            config.config.update(
                cute_chained_preparation_cohorts=count,
                cute_chained_preparation_unroll=1,
            )
        bound = _typed_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)):
            source = bound.to_code(config)
            assert ("chain_cohort =" in source) is (count != 1)
            compiled.append(bound.compile_config(config))
    expected, actual = compiled[0](*args), compiled[1](*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(compiled[1](*args), actual, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled[1](*args)
    graph.replay()
    torch.testing.assert_close(captured, actual, atol=0, rtol=0)
    torch.testing.assert_close(inputs, saved, atol=0, rtol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "bt32,value_tile,consumer_warps,cohorts",
    [(False, 64, 4, 3), (True, 64, 8, 2), (False, 128, 4, 3), (True, 128, 4, 3)],
)
@pytest.mark.parametrize("tma", [False, True])
def test_cohort_prefill_keeps_original_ragged_fp32_and_replay_oracles_gpu(
    bt32: bool, value_tile: int, consumer_warps: int, cohorts: int, tma: bool
) -> None:
    _run_preparation_prefill(
        bt32,
        value_tile,
        consumer_warps,
        rectangular_leaf=tma,
        seed_columns=32,
        cache_layout="xor",
        vector_group=True,
        cohorts=cohorts,
        preparation_unroll=1,
    )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width,cohorts,consumers", [(32, 2, 8), (64, 3, 4)])
def test_cohort_resident_carry_selects_endpoint_storage_by_slab_size_gpu(
    dtype: torch.dtype,
    width: int,
    cohorts: int,
    consumers: int,
) -> None:
    torch.manual_seed(9358)
    steps = 7
    args = (
        *(
            torch.randn(shape, device=DEVICE, dtype=dtype) * 0.125
            for shape in (
                (steps, 128, 16),
                (steps, 16, width),
                (steps, width, 16),
                (steps, 16, width),
            )
        ),
        torch.randn((128, width), device=DEVICE) * 0.125,
    )
    saved = tuple(value.clone() for value in args)
    compiled = []
    for count in (1, cohorts):
        config = _config(16, pipeline=True, consumer_warps=consumers)
        config.config["cute_chained_warp_mma_rows"] = width
        if count != 1:
            config.config.update(
                cute_chained_preparation_cohorts=count,
                cute_chained_preparation_unroll=1,
            )
        bound = _resident_carry_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_resident_carry_sequence, args)
        ):
            source = bound.to_code(config)
            assert "chain_resident_carry_" in source
            if count != 1:
                assert ("chain_endpoints =" in source) is (width == 32)
            compiled.append(bound.compile_config(config))
    expected, actual = compiled[0](*args), compiled[1](*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(compiled[1](*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
