"""CPU checks of the split-K search surface of tcgen05 GEMMs.

A knob the search can sample but codegen rejects wastes autotune compiles,
and a seed the search normalization rewrites never reaches the population.
Both are checked here on a fake SM100 target with 148 SMs: every sampled
``tcgen05_split_k > 1`` sits on the flat one-CTA tcgen05 envelope the
pipeline splits, the split-K seeds survive the strict seed transfer, and a
grid that fills the machine has no split surface.
"""

from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.cute_population_contracts import _cpu_target

import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._hardware import HardwareInfo
from helion._testing import default_cute_mma_support
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])


@contextmanager
def _sm100_target() -> Iterator[None]:
    """A 148-SM B200 without CUDA work (tcgen05 on for 16-bit and fp8)."""
    with ExitStack() as stack:
        stack.enter_context(_cpu_target())
        stack.enter_context(patch_cute_mma_support(default_cute_mma_support()))
        stack.enter_context(
            patch.object(
                CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
            )
        )
        stack.enter_context(
            patch(
                "helion._hardware.get_hardware_info",
                return_value=HardwareInfo(
                    device_kind="cuda",
                    hardware_name="NVIDIA B200",
                    runtime_version="13.0",
                    compute_capability="sm100",
                ),
            )
        )
        stack.enter_context(
            patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=148)
        )
        yield


@helion.kernel(backend="cute", static_shapes=True)
def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _gemm_fp8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _gemm_with_counter(
    a: torch.Tensor, b: torch.Tensor, counter: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        hl.atomic_add(counter, [0], 1)
    return out


def _bind(m: int, n: int, k: int, dtype: torch.dtype = torch.bfloat16):
    a = torch.empty(m, k, dtype=dtype)
    if dtype is torch.float8_e4m3fn:
        # fp8 tcgen05 operands are K-major on both sides.
        b = torch.empty(n, k, dtype=dtype).T
        return _gemm_fp8.bind((a, b))
    return _gemm.bind((a, torch.empty(k, n, dtype=dtype)))


def _samples(gen: ConfigGeneration, count: int) -> list[dict[str, object]]:
    return [dict(gen.unflatten(gen.random_flat()).config) for _ in range(count)]


def _is_flat_one_cta(config: dict[str, object]) -> bool:
    """The flat one-CTA tcgen05 envelope the pipeline splits: a 64- or
    128-row tile on one CTA with a flat pid."""
    block_sizes = config["block_sizes"]
    assert isinstance(block_sizes, list)
    return (
        config.get("pid_type", "flat") == "flat"
        and config.get("tcgen05_cluster_m", 1) == 1
        and config.get("tcgen05_cluster_n", 1) == 1
        and block_sizes[0] in (64, 128)
    )


@pytest.mark.parametrize(
    "shape,dtype",
    (
        ((512, 512, 65536), torch.bfloat16),
        # fp8 configs leave the search fixer through its one-CTA early
        # return; the split-K repair must still run for them.
        ((512, 512, 16384), torch.float8_e4m3fn),
    ),
    ids=("bf16-512x512x65536", "fp8-512x512x16384"),
)
def test_sampled_splits_stay_on_the_flat_one_cta_envelope(shape, dtype) -> None:
    with _sm100_target():
        bound = _bind(*shape, dtype=dtype)
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        assert tcgen05.split_k_search_choices() == (2, 4, 8, 16)
        gen = ConfigGeneration(spec)
        split = [
            config
            for config in _samples(gen, 300)
            if config.get("tcgen05_split_k", 1) > 1
        ]
        off_envelope = [config for config in split if not _is_flat_one_cta(config)]
        assert off_envelope == []
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.get("tcgen05_split_k", 1) > 1
        ]
        assert seeds
        for seed in seeds:
            _flat, normalized = gen.strict_config_pair(seed)
            assert (
                normalized.config["tcgen05_split_k"] == seed.config["tcgen05_split_k"]
            )
            assert _is_flat_one_cta(dict(normalized.config))


def test_a_full_grid_has_no_split_surface() -> None:
    with _sm100_target():
        bound = _bind(8192, 8192, 8192)
        spec = bound.config_spec
        assert spec._cute_tcgen05_config.split_k_search_choices() == ()
        gen = ConfigGeneration(spec)
        assert all("tcgen05_split_k" not in config for config in _samples(gen, 50))
        assert not any(
            seed.get("tcgen05_split_k", 1) > 1 for seed in spec.compiler_seed_configs
        )


def test_a_tile_body_with_another_side_effect_cannot_split() -> None:
    # Only the tcgen05 tile store runs under the last-arriver check: a body
    # that also counts its tiles would count once per split, so the knob is
    # neither searched nor accepted.
    with _sm100_target():
        a = torch.empty(512, 65536, dtype=torch.bfloat16)
        b = torch.empty(65536, 512, dtype=torch.bfloat16)
        counter = torch.zeros(1, dtype=torch.int32)
        bound = _gemm_with_counter.bind((a, b, counter))
        spec = bound.config_spec
        assert not spec.cute_tcgen05_split_k_plain_body
        assert spec._cute_tcgen05_config.split_k_search_choices() == ()
        assert not any(
            seed.get("tcgen05_split_k", 1) > 1 for seed in spec.compiler_seed_configs
        )
        plain = _bind(512, 512, 65536)
        assert plain.config_spec.cute_tcgen05_split_k_plain_body


def test_sampled_splits_render() -> None:
    # The envelope predicate above mirrors the fixer; rendering closes the
    # loop with codegen's own envelope check.
    from test._cute_dense_tcgen05 import cpu_codegen_target

    with cpu_codegen_target():
        bound = _bind(512, 512, 65536)
        gen = ConfigGeneration(bound.config_spec)
        # Random samples land on the split envelope a few percent of the time;
        # render the first four such samples (the renders are the slow part).
        rendered = 0
        for config in _samples(gen, 400):
            if config.get("tcgen05_split_k", 1) > 1:
                code = bound.to_code(helion.Config(**config))
                assert "tcgen05_sk_slab" in code
                rendered += 1
                if rendered == 4:
                    break
        assert rendered == 4
