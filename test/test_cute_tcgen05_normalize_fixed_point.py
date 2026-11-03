"""The search repair of tcgen05 configs is a fixed point of strict normalization.

``ConfigSpec.normalize(_fix_invalid=True)`` is what the autotuner's population
builder runs on every sampled flat config; the compile path then runs the
strict ``normalize(_fix_invalid=False)`` on the repaired config and silently
drops a candidate the strict pass refuses (``has_compiled_config`` swallows
``InvalidConfig``).  So for every seed, the default and ``N`` random flat
samples of each GEMM kernel below, the repaired config has to (1) survive a
second repair unchanged and (2) be accepted, unchanged, by the strict pass.
The repair chain is an ordered list of projections (``fix_search_config``)
followed by the strategy normalizer; this test is what keeps a projection
from being judged on a tuple a later step moves.
"""

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING

import pytest
import torch

from test.cute_population_contracts import tcgen05_mma_support
from test.test_cute_tcgen05_m_pair_seeds import _bind as _pair_bind
from test.test_cute_tcgen05_m_pair_seeds import _make_plain_gemm
from test.test_dot_requirements import _cute_4096_matmul_kernel
from test.test_dot_requirements import _cute_strategy_matmul_force_persistent_kernel
from test.test_dot_requirements import _cute_strategy_matmul_kernel

import helion
from helion._compiler.cute.strategies import TCGEN05_PERSISTENT_GRID_CAP_DEFAULT
from helion._compiler.cute.tcgen05_config import surface
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import atomic_output_mm
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_dense_tcgen05 import extra_store_mm
from helion._testing.cute_dense_tcgen05 import plain_mm
from helion._testing.cute_dense_tcgen05 import scaled_args
from helion._testing.cute_dense_tcgen05 import scaled_mm
from helion._testing.cute_dense_tcgen05 import user_bound_mm
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel


SAMPLES = 64
BF16 = torch.bfloat16
F16 = torch.float16


@helion.kernel(backend="cute", static_shapes=True)
def _bias_gemm(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_m, tile_n]).to(x.dtype)
    return out


def _dense(
    kernel: helion.Kernel, make_args: Callable[[], tuple[object, ...]]
) -> Callable[[], BoundKernel]:
    """A ``_cute_dense_tcgen05`` kernel bound under ``cpu_codegen_target``.

    ``make_args`` builds the CPU argument tensors when the test runs, not at
    collection (the table below would otherwise allocate ~200 MB per worker).
    """

    def bind() -> BoundKernel:
        with cpu_codegen_target(), forbid_cuda_work():
            return kernel._bind_isolated(make_args())

    return bind


def _b200(
    kernel: helion.Kernel,
    make_args: Callable[[], tuple[object, ...]],
    *,
    matmul_num_sm: int = 148,
) -> Callable[[], BoundKernel]:
    """A kernel bound against the population contracts' B200 without CUDA."""

    def bind() -> BoundKernel:
        with cpu_render_target(
            mma_support=tcgen05_mma_support(), matmul_num_sm=matmul_num_sm
        ):
            return kernel._bind_isolated(make_args())

    return bind


def _square(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(n, n, dtype=dtype), torch.empty(n, n, dtype=dtype)


# Every kernel the fixed-point property is pinned on, by name.  The
# ``test_dot_requirements`` strategy kernels bind with the wave-quantization
# gate's SM count at 0, as their own tests do, which keeps the cluster_m=2
# search arm (and its projections) live on the 256x256 shape.
KERNELS: dict[str, Callable[[], BoundKernel]] = {
    "strategy_256": _b200(
        _cute_strategy_matmul_kernel, lambda: _square(256, F16), matmul_num_sm=0
    ),
    "strategy_256_force_persistent": _b200(
        _cute_strategy_matmul_force_persistent_kernel,
        lambda: _square(256, F16),
        matmul_num_sm=0,
    ),
    "matmul_4096": _b200(_cute_4096_matmul_kernel, lambda: _square(4096, F16)),
    "plain_mm_2048": _dense(
        plain_mm,
        lambda: (
            torch.empty(2048, 1024, dtype=BF16),
            torch.empty(1024, 2048, dtype=BF16),
        ),
    ),
    "extra_store_mm_4096": _dense(extra_store_mm, lambda: _square(4096, BF16)),
    "atomic_output_mm": _dense(
        atomic_output_mm,
        lambda: (*_square(1024, BF16), torch.empty(1, dtype=torch.int32)),
    ),
    "scaled_mm_2048": _dense(scaled_mm, lambda: scaled_args(m=2048, n=2048, mode=1)),
    "scaled_mm_64": _dense(scaled_mm, lambda: scaled_args()),
    "user_bound_mm": _dense(
        user_bound_mm,
        lambda: (
            torch.empty(512, 4096, dtype=BF16),
            torch.empty(4096, 4096, dtype=BF16),
            128,
        ),
    ),
    # The M-pair seeds' gradient GEMM (M-major A, the 512x256 pair family),
    # bound the way ``test_cute_tcgen05_m_pair_seeds`` binds it.
    "mat1T_grad_2048x4096x2048": lambda: _pair_bind(
        _make_plain_gemm(),
        (torch.empty(2048, 4096, dtype=F16).T, torch.empty(2048, 2048, dtype=F16)),
    ),
}

# Known gap, documented rather than hidden: a rowvec-bias GEMM whose
# repaired ``tcgen05_aux_load_mode="epi_tma"`` sample keeps ``tcgen05_ab_stages``
# above the ring's AB-stage limit (``_normalize_epi_tma`` refuses it strictly)
# and whose ``_fix_ab_stages_search_config`` demotes 3 -> 2 only on the
# second repair.  The strict xfail below fails loudly when the admission moves
# into the fix pass; the entry then moves from ``GAPS`` to ``KERNELS``.
GAPS: dict[str, Callable[[], BoundKernel]] = {
    "bias_gemm_2048x4096x2048": _b200(
        _bias_gemm,
        lambda: (
            torch.empty(2048, 4096, dtype=F16),
            torch.empty(4096, 2048, dtype=F16),
            torch.empty(2048, 2048, dtype=F16),
        ),
    ),
}


def _configs(spec: ConfigSpec, gen: ConfigGeneration) -> Iterator[tuple[str, dict]]:
    """The default, every compiler / tcgen05 seed and ``SAMPLES`` random flat
    configs, each already repaired once (``unflatten`` is the repairing
    decode the population builder runs)."""
    yield "default", dict(spec.default_config().config)
    seeds = [*spec.compiler_seed_configs, *spec.autotune_seed_configs()]
    for index, seed in enumerate(seeds):
        yield f"seed[{index}]", dict(gen.unflatten(gen.flatten(seed)).config)
    # The sampler draws from the process-wide ``random``; seed it for a
    # reproducible sample set and hand the state back afterwards so the
    # tests sharing this worker keep their own draws.
    state = random.getstate()
    random.seed(0)
    try:
        for index in range(SAMPLES):
            flat = gen.random_flat()
            yield f"sample[{index}]", dict(gen.unflatten(flat).config)
    finally:
        random.setstate(state)


def _changed(before: dict, after: dict) -> dict[str, tuple[object, object]]:
    return {
        key: (before.get(key), after.get(key))
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }


# The strict pass spells out knobs the repair pass leaves implicit (the
# validators ``setdefault`` them on the families they apply to); a key the
# strict pass *adds* at its default value is the same config in the other
# representation, not a repair the search missed.
_INERT_DEFAULTS: dict[str, object] = {
    "tcgen05_cta_group": "auto",
    "tcgen05_c_store_mode": "normal",
    "tcgen05_persistent_grid_cap": TCGEN05_PERSISTENT_GRID_CAP_DEFAULT,
    "tcgen05_epi_warp_groups": 1,
}


def _inert_defaults(spec: ConfigSpec) -> dict[str, object]:
    defaults = dict(_INERT_DEFAULTS)
    for fragments in (
        surface.optional_fragments(spec.tcgen05, for_search=False),
        surface.optional_fragments(spec.tcgen05, for_search=True),
    ):
        for key, fragment in fragments.items():
            defaults[key] = fragment.default()
    return defaults


def _strict_changed(
    repaired: dict, strict: dict, inert: dict[str, object]
) -> dict[str, tuple[object, object]]:
    return {
        key: change
        for key, change in _changed(repaired, strict).items()
        if not (key not in repaired and key in inert and strict[key] == inert[key])
    }


def _check_fixed_point(bind: Callable[[], BoundKernel]) -> list[str]:
    bound = bind()
    spec = bound.config_spec
    assert spec.tcgen05.search_enabled, "the kernel has no tcgen05 search surface"
    gen = ConfigGeneration(spec)
    inert = _inert_defaults(spec)
    failures: list[str] = []
    for name, repaired in _configs(spec, gen):
        again = copy.deepcopy(repaired)
        spec.normalize(again, _fix_invalid=True)
        if changed := _changed(repaired, again):
            failures.append(f"{name}: a second repair changed {changed}")
        strict = copy.deepcopy(repaired)
        try:
            spec.normalize(strict, _fix_invalid=False)
        except helion.exc.InvalidConfig as error:
            failures.append(f"{name}: strict normalization refused it: {error}")
            continue
        if changed := _strict_changed(repaired, strict, inert):
            failures.append(f"{name}: strict normalization changed {changed}")
    return failures


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_repair_is_a_fixed_point_of_strict_normalization(name: str) -> None:
    failures = _check_fixed_point(KERNELS[name])
    assert not failures, "\n".join(failures[:12])


@pytest.mark.parametrize("name", sorted(GAPS))
@pytest.mark.xfail(
    reason="epi_tma AB-stage admission and the ab=3 demotion run outside the "
    "fix pass (see GAPS)",
    strict=True,
)
def test_known_gaps(name: str) -> None:
    failures = _check_fixed_point(GAPS[name])
    assert not failures, "\n".join(failures[:12])
