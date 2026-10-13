from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_computed_scan import _args
from .test_cute_computed_scan import _prefix
from .test_cute_computed_scan import _raw_scan
from .test_cute_computed_scan import _rejected
import helion
from helion import exc
from helion._compiler import autotuner_heuristics
from helion._compiler.cute import computed_scan_root
from helion._compiler.cute.computed_scan_staging import KEY
from helion._compiler.cute.computed_scan_staging import staged_scan_layout
from helion._compiler.cute.positional_root import _Unsupported
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.runtime import default_cute_launcher


@pytest.mark.parametrize("shape", [(64, 32), (48, 16), (17, 35)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("threads", [32, 128])
def test_staged_image_has_unique_complete_producers_and_consumers(
    shape: tuple[int, int], axis: int, threads: int
) -> None:
    layout = staged_scan_layout(shape, axis, shape[axis] - 1, threads)
    layout.check()
    physical = {
        (i, j): i * layout.strides[0] + j * layout.strides[1]
        for i in range(shape[0])
        for j in range(shape[1])
    }
    assert len(set(physical.values())) == shape[0] * shape[1]
    assert 0 <= min(physical.values()) <= max(physical.values()) < layout.elements
    assert layout.allocated_bytes >= 4 * layout.elements
    assert layout.allocated_bytes % 128 == 0
    producers = [
        (element // shape[1], element % shape[1])
        for thread in range(threads)
        for element in range(thread, shape[0] * shape[1], threads)
    ]
    consumers = []
    for warp in range(threads // 32):
        for vector in range(warp, shape[1 - axis], threads // 32):
            for part in range((shape[axis] + 31) // 32):
                group = []
                for lane in range(32):
                    position = part * 32 + lane
                    if position < shape[axis]:
                        coordinate = (
                            (position, vector) if axis == 0 else (vector, position)
                        )
                        group.append(physical[coordinate] % 32)
                        consumers.append(coordinate)
                assert len(set(group)) == len(group)
    assert sorted(producers) == sorted(consumers) == sorted(physical)


def test_staging_layout_rejects_stale_or_unbounded_facts() -> None:
    layout = staged_scan_layout((64, 32), 0, 63, 128)
    for changed in (
        replace(layout, strides=(32, 1)),
        replace(layout, elements=2048),
        replace(layout, allocated_bytes=8192),
    ):
        with pytest.raises(_Unsupported, match="changed"):
            changed.check()
    for shape, axis, extent, threads in (
        ((64,), 0, 64, 128),
        ((64, 32), 0, 65, 128),
        ((64, 32), 2, 32, 128),
        ((64, 32), 0, 64, 31),
        ((2**30, 32), 0, 64, 128),
    ):
        with pytest.raises(_Unsupported):
            staged_scan_layout(shape, axis, extent, threads)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("extent", [48, 64])
def test_actual_staged_scan_codegen_and_search(transpose: bool, extent: int) -> None:
    plans = []
    original = computed_scan_root.plan_computed_scan_root

    def observe(graphs, df):
        plan = original(graphs, df)
        if plan is not None:
            plan.check()
            plans.append(plan)
        return plan

    with (
        _cpu_codegen(),
        patch.object(computed_scan_root, "plan_computed_scan_root", observe),
    ):
        arguments = (*_args(), extent, transpose)
        with patch.object(
            autotuner_heuristics,
            "append_computed_scan_seed",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            old_bound = _prefix._bind_isolated(arguments)
        bound = _prefix._bind_isolated(arguments)
        spec = bound.config_spec
        assert (
            spec.compiler_seed_configs[:-1]
            == old_bound.config_spec.compiler_seed_configs
        )
        assert spec.default_config() == old_bound.config_spec.default_config()
        assert spec.cute_computed_scan_search_enabled
        config = helion.Config(
            block_sizes=[16], num_warps=4, cute_computed_scan_schedule="staged_warp"
        )
        normalized = spec.normalized_config(config)
        generation = spec.create_config_generation()
        flattened = generation.flatten(normalized)
        roundtrip = generation.unflatten(flattened)
        assert roundtrip[KEY] == "staged_warp"
        assert generation.flatten(roundtrip) == flattened
        assert any(
            seed.config.get(KEY) == "staged_warp" for seed in spec.compiler_seed_configs
        )
        default = bound.to_code(helion.Config(block_sizes=[16], num_warps=4))
        explicit = bound.to_code(
            helion.Config(
                block_sizes=[16], num_warps=4, cute_computed_scan_schedule="warp"
            )
        )
        assert default == explicit
        source = bound.to_code(config)
        plan = plans[-1]
        assert plan.staging is not None
        assert plan.shared_bytes == plan.staging.allocated_bytes
        assert source.count("alloc_smem(") == 1
        assert f"alloc_smem(cutlass.Float32, {plan.staging.elements}," in source
        assert source.count("sync_threads()") == 3
        assert default.count("sync_threads()") == 2
        assert (
            source.count("shuffle_sync_up(") == default.count("shuffle_sync_up(") == 5
        )
        assert source.index("sync_threads()") < source.index(
            "_vector in cutlass.range("
        )


@pytest.mark.parametrize(
    "kind",
    [
        "alias",
        "dtype",
        "reverse",
        "multiply",
        "omitted_axis",
        "extra_store",
        "output_read",
    ],
)
def test_staging_cannot_admit_an_unsafe_root(kind: str) -> None:
    with _cpu_codegen():
        bound = _rejected._bind_isolated((torch.empty((3, 48, 35)), kind))
        with pytest.raises((exc.BackendUnsupported, exc.InvalidConfig)):
            bound.to_code(
                helion.Config(
                    block_sizes=[16],
                    num_warps=4,
                    cute_computed_scan_schedule="staged_warp",
                )
            )


def test_staging_charges_padding_and_does_not_silently_fall_back() -> None:
    with (
        _cpu_codegen(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=4096
        ),
    ):
        bound = _prefix._bind_isolated((*_args(), 64, False))
        assert "ptp_computed_scan_" in bound.to_code(
            helion.Config(block_sizes=[16], num_warps=4)
        )
        with pytest.raises(exc.BackendUnsupported):
            bound.to_code(
                helion.Config(
                    block_sizes=[16],
                    num_warps=4,
                    cute_computed_scan_schedule="staged_warp",
                )
            )


def test_raw_scan_does_not_gain_the_staging_search_dimension() -> None:
    with _cpu_codegen():
        bound = _raw_scan._bind_isolated((torch.empty((3, 32)),))
        assert not bound.config_spec.cute_computed_scan_search_enabled
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(helion.Config(cute_computed_scan_schedule="staged_warp"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
def test_gpu_staged_scan_guards_and_mutated_graphs(
    dtype: torch.dtype, transpose: bool
) -> None:
    torch.manual_seed(915)
    backing = torch.full((3, 41, 40), float("nan"), dtype=dtype, device="cuda")
    x = backing[:, :37, :35]
    bias = torch.linspace(-1, 1, 35, device="cuda")
    rate = torch.tensor([-0.01, -0.03, -0.07], device="cuda")
    output_shape = (3, 64, 35)
    count = 3 * 64 * 35
    guard = torch.full((count + 256,), -29.0, device="cuda")
    output = guard[128 : 128 + count].view(output_shape)
    arguments = (x, bias, rate, 64, transpose)
    bound = _prefix._bind_isolated(arguments)
    config = helion.Config(
        block_sizes=[16], num_warps=4, cute_computed_scan_schedule="staged_warp"
    )
    assert bound.to_code(config).count("sync_threads()") == 3
    compiled = bound.compile_config(config)

    def protected(implementation, grid, *args, **kwargs):
        assert args[0] is rate and args[1] is x and args[2] is bias
        assert args[3].shape == output.shape and args[3].stride() == output.stride()
        return default_cute_launcher(implementation, grid, *args[:3], output, **kwargs)

    def invoke() -> torch.Tensor:
        compiled(*arguments, _launcher=protected)
        return output

    x.copy_(torch.randn_like(x))
    invoke()
    replay = _make_cudagraph_replay(invoke)
    for generation in range(4):
        x.copy_(torch.randn_like(x) * (generation + 1))
        saved = backing.clone()
        bias_before, rate_before = bias.clone(), rate.clone()
        steps = (
            torch.nn.functional.softplus(x.double() + bias.double())
            * rate.double()[:, None, None]
        )
        expected = torch.zeros(output_shape, device="cuda", dtype=torch.float64)
        expected[:, :37] = steps
        expected = expected.cumsum(1)
        output.fill_(float("nan"))
        (invoke if generation < 2 else replay)()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.double(), expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(backing, saved, atol=0, rtol=0, equal_nan=True)
        assert torch.equal(bias, bias_before) and torch.equal(rate, rate_before)
        assert torch.all(guard[:128] == -29.0) and torch.all(
            guard[128 + count :] == -29.0
        )
