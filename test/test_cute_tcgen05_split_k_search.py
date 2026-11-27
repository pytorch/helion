"""CPU checks of the split-K search surface of tcgen05 GEMMs.

A knob the search can sample but codegen rejects wastes autotune compiles,
and a seed the search normalization rewrites never reaches the population.
Both are checked here on a fake SM100 target with 148 SMs: every sampled
``tcgen05_split_k > 1`` sits on the flat one-CTA tcgen05 envelope the
pipeline splits, the split-K seeds survive the strict seed transfer, and a
grid that fills the machine has no split surface. The splits the seeds take
per tile are checked on the GB300 stand-in (152 SMs).
"""

from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.cute_population_contracts import _cpu_target

import helion
from helion._compiler.cute import target_facts
from helion._compiler.cute.mma_support import CuteMmaSupport
from helion._compiler.cute.tcgen05_config import surface
from helion._compiler.cute.tcgen05_constants import tcgen05_cluster_split_k
from helion._hardware import HardwareInfo
from helion._testing import default_cute_mma_support
from helion._testing import patch_cute_mma_support
from helion._testing import patch_target_facts
from helion._testing import skipUnlessBackends
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@contextmanager
def _sm100_target() -> Iterator[None]:
    """A 148-SM B200 without CUDA work (tcgen05 on for 16-bit and fp8)."""
    with ExitStack() as stack:
        stack.enter_context(_cpu_target())
        stack.enter_context(patch_cute_mma_support(default_cute_mma_support()))
        stack.enter_context(
            patch.object(
                target_facts, "per_cta_smem_capacity_bytes", return_value=232448
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
            patch(
                "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
                return_value=148,
            )
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


# torch's reduced-precision flags at (False, False), through helion.inductor.
_gemm_split_k_none = helion.kernel(
    _gemm.fn, backend="cute", static_shapes=True, split_k_reduction="none"
)


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
        assert surface.split_k_search_choices(tcgen05) == (2, 4, 8, 16)
        gen = ConfigGeneration(spec)
        split = [
            config
            for config in _samples(gen, 300)
            if config.get("tcgen05_split_k", 1) > 1
        ]
        # (or the CtaGroup.TWO pair's 4-CTA cluster splitting K,
        # ``tcgen05_cluster_split_k``)
        off_envelope = [
            config
            for config in split
            if not _is_flat_one_cta(config) and tcgen05_cluster_split_k(config) == 1
        ]
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
            assert _is_flat_one_cta(dict(normalized.config)) or (
                tcgen05_cluster_split_k(dict(normalized.config)) == 2
            )


def test_split_k_seeds_take_the_deepest_one_wave_split() -> None:
    # helion.inductor's bf16 [64, 25088] @ [25088, 4096] (b column-major) on
    # GB300's 152 SMs: beside the two smallest splits that reach a wave, a
    # tile takes the deepest split whose grid still fits in one, 2 for the
    # 64 [64, 64, *] tiles (128 CTAs) and 4 for the 32 [64, 128, *] tiles;
    # the 128 [64, 32, *] tiles fit no split in one wave. Those come after
    # every other seed, so they take no initial-population place from one.
    with gb300_target(), patch_target_facts(num_sm=152):
        a = torch.empty(64, 25088, dtype=torch.bfloat16)
        b = torch.empty(4096, 25088, dtype=torch.bfloat16).T
        spec = _gemm.bind((a, b)).config_spec
        assert spec._cute_tcgen05_config.device_sm_count == 152
        gen = ConfigGeneration(spec)
        splits: dict[tuple[int, int], set[int]] = {}
        for seed in spec.compiler_seed_configs:
            if seed.get("pid_type") != "flat" or seed.get("tcgen05_split_k", 1) == 1:
                continue
            _flat, normalized = gen.strict_config_pair(seed)
            assert normalized.config == {**normalized.config, **seed.config}
            bm, bn, _bk = seed.config["block_sizes"]
            splits.setdefault((bm, bn), set()).add(seed.config["tcgen05_split_k"])
        assert splits == {(64, 128): {4, 8, 16}, (64, 64): {2, 4, 8}, (64, 32): {2, 4}}
        tail = [
            (*seed.config["block_sizes"][:2], seed.config.get("tcgen05_split_k"))
            for seed in spec.compiler_seed_configs[-4:]
        ]
        assert tail == [(64, 128, 4), (64, 128, 4), (64, 64, 2), (64, 64, 2)]


def test_a_full_grid_has_no_split_surface() -> None:
    with _sm100_target():
        bound = _bind(8192, 8192, 8192)
        spec = bound.config_spec
        assert surface.split_k_search_choices(spec._cute_tcgen05_config) == ()
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
        assert not spec.tcgen05.split_k_plain_body
        assert surface.split_k_search_choices(spec._cute_tcgen05_config) == ()
        assert not any(
            seed.get("tcgen05_split_k", 1) > 1 for seed in spec.compiler_seed_configs
        )
        plain = _bind(512, 512, 65536)
        assert plain.config_spec.tcgen05.split_k_plain_body


@helion.kernel(backend="cute", static_shapes=True)
def _gemm_into_k16(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    m, k = a.size()
    _, n = b.size()
    block_k = hl.register_block_size(16, 16)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)


# A tile narrower than the tcgen05 MMA's N (8 columns for 16-bit operands, 32
# for TF32) runs the scalar fallback the search admits for a small-N GEMM,
# which codegen does not split: a sampled split is dropped, and the split-K
# default keeps such a tile unsplit. The transposed GEMVs (``out.T = w @ x.T``)
# pin 16-deep k-blocks: the 16-bit one's 64-block K loop is long enough for
# that default to split (the TF32 one keeps the base default).
@pytest.mark.parametrize(
    ("n", "dtype"),
    ((4, torch.bfloat16), (16, torch.float32)),
    ids=("bf16-n4", "tf32-n16"),
)
def test_a_tile_narrower_than_the_mma_cannot_split(n: int, dtype: torch.dtype) -> None:
    support = CuteMmaSupport(
        universal=True,
        warp_f16bf16=True,
        tcgen05_f16bf16=True,
        tcgen05_f8=True,
        tcgen05_tf32=True,
    )
    with cpu_codegen_target(), patch_cute_mma_support(support):
        weight = torch.empty(4096, 1024, dtype=dtype)
        x = torch.empty(n, 1024, dtype=dtype)
        out = torch.empty(n, 4096, dtype=dtype)
        bound = _gemm_into_k16._bind_isolated((weight, x.T, out.T))
        spec = bound.config_spec
        assert surface.split_k_search_choices(spec._cute_tcgen05_config)
        config = spec.default_config()
        # the registered K block comes first: K, M, N
        assert config.block_sizes == [16, 64, n]
        assert config.config["tcgen05_split_k"] == 1
        assert "cute.gemm(" not in bound.to_code(config)
        seeds = ConfigGeneration(spec).seed_flat_config_pairs()
        assert all(seed.config.get("tcgen05_split_k", 1) == 1 for _, seed in seeds)
        sample = {**config.config, "tcgen05_split_k": 4}
        spec.normalize(sample, _fix_invalid=True)
    assert sample["tcgen05_split_k"] == 1


def test_sampled_splits_render() -> None:
    # The envelope predicate above mirrors the fixer; rendering closes the
    # loop with codegen's own envelope check.
    with cpu_codegen_target():
        bound = _bind(512, 512, 65536)
        gen = ConfigGeneration(bound.config_spec)
        # Random samples land on the split envelope a few percent of the time;
        # render the first four such samples (the renders are the slow part).
        # The sampler draws from ``random``: a fixed seed makes the rendered
        # configs the same from run to run, restored so the process-wide RNG
        # is left as found.
        state = random.getstate()
        try:
            random.seed(0)
            samples = _samples(gen, 400)
        finally:
            random.setstate(state)
        rendered = 0
        for config in samples:
            if config.get("tcgen05_split_k", 1) > 1:
                code = bound.to_code(helion.Config(**config))
                assert "tcgen05_sk_slab" in code
                rendered += 1
                if rendered == 4:
                    break
        assert rendered == 4


def test_split_k_reduction_none_offers_no_split() -> None:
    # The grid above searches and seeds tcgen05_split_k, its default splits K
    # four ways and the split-K workspace takes its plain store; under
    # split_k_reduction="none" it gets none of them and its default runs the
    # same 64x64x64 tile over the whole K loop.
    with _sm100_target():
        a = torch.empty(512, 65536, dtype=torch.bfloat16)
        b = torch.empty(65536, 512, dtype=torch.bfloat16)
        split_bound = _gemm.bind((a, b))
        bound = _gemm_split_k_none.bind((a, b))
        split, spec = split_bound.config_spec, bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        assert tcgen05.split_k_reduction == "none"
        assert surface.split_k_search_choices(tcgen05) == ()
        assert surface.split_k_admissible_choices(tcgen05) == (2, 4, 8, 16)
        gen = ConfigGeneration(spec)
        assert all("tcgen05_split_k" not in config for config in _samples(gen, 50))
        for splits in (
            lambda seed: seed.get("tcgen05_split_k", 1) > 1,
            lambda seed: seed.get("cute_split_k_workspace", False),
        ):
            assert any(map(splits, split.compiler_seed_configs))
            assert not any(map(splits, spec.compiler_seed_configs))
        assert split.default_config().config["tcgen05_split_k"] == 4
        default = spec.default_config().config
        assert "tcgen05_split_k" not in default
        assert default["block_sizes"] == split.default_config().config["block_sizes"]
        assert split.cute_split_k_workspace_available
        assert not spec.cute_split_k_workspace_available
        # An explicit split is refused; the search's repair drops it.
        flat_split = {**split.default_config().config}
        with pytest.raises(InvalidConfig, match="split_k_reduction='none'"):
            spec.normalize(dict(flat_split))
        repaired = dict(flat_split)
        spec.normalize(repaired, _fix_invalid=True)
        assert "tcgen05_split_k" not in repaired
        with pytest.raises(InvalidConfig, match="split-K workspace"):
            spec.normalize({**default, "cute_split_k_workspace": True})
        # A printed config replays under the setting it was chosen under.
        config = helion.Config(**default)
        printed = bound.format_kernel_decorator(config, bound.settings)
        assert "split_k_reduction='none'" in printed
        printed = split_bound.format_kernel_decorator(config, split_bound.settings)
        assert "split_k_reduction" not in printed
