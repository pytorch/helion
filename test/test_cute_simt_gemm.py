"""The exact fp32 SIMT GEMM (``cute_simt_mma``) and the ordered split-K phases."""

from __future__ import annotations

import pytest
import torch

from test._cute_binding import _cpu_bind

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _mm_ieee(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def _mm_seeded_split_k(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """A seeded accumulator over the split-K sub-range K loop (review case a)."""
    m, k = a.shape
    _, n = b.shape
    block_k = 64
    tmp = torch.empty((m, n, helion.cdiv(k, block_k)), device=a.device, dtype=a.dtype)
    for tile_m, tile_n, tile_k_outer in hl.tile(
        [m, n, k], block_size=[None, None, block_k]
    ):
        acc = bias[tile_m, tile_n]
        for tile_k in hl.tile(tile_k_outer.begin, tile_k_outer.end):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        tmp[tile_m, tile_n, tile_k_outer.id] = acc
    return tmp


def _mm_nested_k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A sub-range K loop under a device loop, zero-seeded (review case b)."""
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for k_outer in hl.tile(k, block_size=48):
            for tile_k in hl.tile(k_outer.begin, k_outer.end):
                acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def _tile_config(
    bound: helion.runtime.kernel.BoundKernel, **overrides: object
) -> helion.Config:
    """A config with 16x16 output tiles over 16x16 threads (the universal layout)."""
    default = bound.config_spec.default_config().config
    block_sizes = list(default["block_sizes"])
    block_sizes[:2] = [16, 16]
    block_sizes[2] = 32
    num_threads = [0] * sum(1 for _ in bound.config_spec.num_threads)
    num_threads[:2] = [16, 16]
    return _simt_config(
        bound, block_sizes=block_sizes, num_threads=num_threads, **overrides
    )


def _simt_config(
    bound: helion.runtime.kernel.BoundKernel, **overrides: object
) -> helion.Config:
    config = dict(bound.config_spec.default_config().config)
    config.update(overrides)
    return helion.Config(**config)


def _kernel(fn: object, **settings: object) -> helion.Kernel:
    return helion.kernel(
        fn,  # pyrefly: ignore[bad-argument-type]
        backend="cute",
        static_shapes=True,
        dot_precision="ieee",
        autotune_effort="none",
        **settings,
    )


def _assert_staged_gemm(code: str, *, root_lane: str = "lane_0") -> None:
    assert "_helion_pending_simt_gemm" not in code
    assert "cute.math.fma(" in code
    assert "simt_sA" in code and "simt_sC" in code
    # The staged K loop runs once per CTA, ahead of the per-lane epilogue.
    assert code.index("cute.math.fma(") < code.index(f"for {root_lane} in range(")


def test_simt_gemm_plain_matmul_codegen() -> None:
    kernel = _kernel(_mm_ieee)
    bound = _cpu_bind(kernel, (torch.ones(64, 256), torch.ones(256, 64)))
    config = _simt_config(
        bound,
        block_sizes=[64, 64, 8],
        num_threads=[16, 16, 0],
        cute_simt_mma=True,
    )
    code = bound.to_code(config)
    _assert_staged_gemm(code)
    # 64 x 64 over 16 x 16 threads: a 4 x 4 register block per thread.
    assert "cute.make_rmem_tensor(16, cutlass.Float32)" in code
    # Both operands reach the kernel although the scalar loads were replaced.
    assert "def _helion__mm_ieee(a, b, out" in code
    # Blocked rows read 128-bit words; strided rows read scalars.
    assert "_helion_mma_dag.lds_v4_f32(" in code
    strided = bound.to_code(
        _simt_config(
            bound,
            block_sizes=[64, 64, 8],
            num_threads=[16, 16, 0],
            cute_simt_mma=True,
            cute_simt_mma_layout="strided",
        )
    )
    _assert_staged_gemm(strided)
    assert "lds_v4_f32" not in strided
    assert "cutlass.Int32(_simt_r * 16)" in strided
    # The knob off keeps the scalar lowering byte-for-byte; the layout knob
    # is inert without it (it stays a valid flat field for coverage round
    # trips) and only an unknown value is rejected.
    off = bound.to_code(_simt_config(bound, block_sizes=[64, 64, 8]))
    assert "simt_sA" not in off and "cute.math.fma(" not in off
    inert = _simt_config(bound, block_sizes=[64, 64, 8], cute_simt_mma_layout="strided")
    bound.config_spec.normalize(inert)
    assert inert.config["cute_simt_mma_layout"] == "strided"
    assert "simt_sA" not in bound.to_code(inert)
    bogus = _simt_config(bound, block_sizes=[64, 64, 8], cute_simt_mma_layout="rows")
    with pytest.raises(exc.InvalidConfig, match="cute_simt_mma_layout"):
        bound.config_spec.normalize(bogus)


def test_simt_gemm_knob_requires_fp32_contraction() -> None:
    def _mm_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        _, n = b.shape
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(a.dtype)
        return out

    kernel = helion.kernel(
        _mm_bf16, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = _cpu_bind(
        kernel,
        (
            torch.ones(64, 256, dtype=torch.bfloat16),
            torch.ones(256, 64, dtype=torch.bfloat16),
        ),
    )
    assert not bound.config_spec.cute_simt_mma_enabled
    config = _simt_config(bound, cute_simt_mma=True)
    with pytest.raises(exc.InvalidConfig, match="cute_simt_mma"):
        bound.config_spec.normalize(config)


def test_split_k_barrier_ordered_phases_and_seeds() -> None:
    from examples.split_k_barrier import split_k_matmul

    kernel = _kernel(split_k_matmul.fn)
    bound = _cpu_bind(kernel, (torch.ones(128, 4096), torch.ones(4096, 128)))
    spec = bound.config_spec
    # ``hl.register_tunable`` and the ``cdiv`` / ``next_power_of_2`` prelude
    # are pure metadata: the barrier phases may launch in order.
    assert "flat" in spec.allowed_pid_types
    assert spec.cute_simt_mma_enabled
    seeds = [
        seed for seed in spec.compiler_seed_configs if seed.config.get("cute_simt_mma")
    ]
    assert seeds, "the SIMT GEMM must be seeded for the cold search"
    assert all(seed.config.get("pid_type") == "flat" for seed in seeds)
    assert {seed.config["cute_simt_mma_layout"] for seed in seeds} == {
        "blocked",
        "strided",
    }
    # The split-K factor is seeded at its default and sized to the SM count:
    # 128x128 tiles need split_k >= 0.75 * 148 and >= 1.5 * 148 CTAs.
    big = [s for s in seeds if s.config["block_sizes"][:2] == [128, 128]]
    assert {s.config["split_k"] for s in big} == {64, 128, 256}
    small = [s for s in seeds if s.config["block_sizes"][:2] == [64, 64]]
    assert {s.config["split_k"] for s in small} == {32, 64}
    primary = seeds[0].config
    assert primary["block_sizes"][:2] == [128, 128]
    assert primary["block_sizes"][2] in (8, 16)
    assert primary["num_threads"][:2] == [16, 16]
    assert primary["split_k"] == 64
    # The split reduction's extent is the ``split_k`` tunable: the seed runs
    # it as one pass over the tunable's default instead of the 4096-wide
    # masked loop its generic size hint would default to.
    assert primary["reduction_loops"] == [64]
    code = bound.to_code(seeds[0])
    # Two ordered launches: the staged GEMM and the split reduction.
    assert code.count("PyCodeCache.load(") == 2
    assert "_helion_pending_simt_gemm" not in code
    assert "cute.math.fma(" in code
    # Only the GEMM's K loop is masked against its own outer tile's end.
    assert "tile_end" in code


def test_simt_gemm_root_layout_predicate() -> None:
    from helion._compiler.cute.simt_gemm import simt_gemm_root_layout_supported

    assert simt_gemm_root_layout_supported(
        bm=128,
        bn=128,
        root_m_threads=16,
        root_n_threads=16,
        root_m_threads_explicit=True,
        root_n_threads_explicit=True,
    )
    # 128 x 128 over 8 x 8 threads is a 16 x 16 register block: too large.
    assert not simt_gemm_root_layout_supported(
        bm=128,
        bn=128,
        root_m_threads=8,
        root_n_threads=8,
        root_m_threads_explicit=True,
        root_n_threads_explicit=True,
    )
    assert not simt_gemm_root_layout_supported(
        bm=100,
        bn=64,
        root_m_threads=16,
        root_n_threads=16,
        root_m_threads_explicit=True,
        root_n_threads_explicit=True,
    )
    # Automatic counts are resolved by the scheduler to a divisor.
    assert simt_gemm_root_layout_supported(
        bm=100,
        bn=64,
        root_m_threads=100,
        root_n_threads=64,
        root_m_threads_explicit=False,
        root_n_threads_explicit=False,
    )


@skipIfNotCUDA()
@pytest.mark.parametrize("layout", ["blocked", "strided"])
@pytest.mark.parametrize("shape", [(64, 256, 64), (65, 130, 70)])
def test_simt_gemm_matches_reference(shape: tuple[int, int, int], layout: str) -> None:
    m, k, n = shape
    a = torch.randn(m, k, device=DEVICE)
    b = torch.randn(n, k, device=DEVICE).T
    kernel = _kernel(_mm_ieee)
    bound = kernel.bind((a, b))
    config = _simt_config(
        bound,
        block_sizes=[64, 64, 8],
        num_threads=[16, 16, 0],
        cute_simt_mma=True,
        cute_simt_mma_layout=layout,
    )
    compiled = bound.compile_config(config)
    for _ in range(3):
        a.normal_()
        b.normal_()
        torch.testing.assert_close(compiled(a, b), a @ b, atol=1e-3, rtol=1e-3)


@skipIfNotCUDA()
def test_split_k_barrier_simt_gemm_matches_reference() -> None:
    from examples.split_k_barrier import split_k_matmul

    a = torch.randn(64, 8192, device=DEVICE)
    b = torch.randn(64, 8192, device=DEVICE).T
    kernel = _kernel(split_k_matmul.fn)
    bound = kernel.bind((a, b))
    seed = next(
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_simt_mma")
    )
    compiled = bound.compile_config(seed)
    for _ in range(3):
        a.normal_()
        b.normal_()
        torch.testing.assert_close(compiled(a, b), a @ b, atol=1e-2, rtol=1e-3)


def _assert_scalar_dot(code: str) -> None:
    assert "MmaUniversalOp" not in code
    assert "_helion_pending_simt_gemm" not in code
    assert "simt_sA" not in code
    assert "cute.math.fma(" not in code


def test_universal_path_refuses_subrange_k_with_seeded_accumulator() -> None:
    # Review case (a): the knob on, the accumulator seeded, so the SIMT GEMM
    # does not apply; the universal MmaUniversalOp staging masks against the
    # operand's full K and must not take the sub-range loop either.
    kernel = _kernel(_mm_seeded_split_k)
    bound = _cpu_bind(
        kernel, (torch.ones(32, 256), torch.ones(256, 32), torch.ones(32, 32))
    )
    code = bound.to_code(_tile_config(bound, cute_simt_mma=True))
    _assert_scalar_dot(code)


def test_universal_path_refuses_subrange_k_under_a_device_loop() -> None:
    # Review case (b): the knob off, a sub-range K loop under a device loop;
    # the operand analysis now admits the block, the lowering must not.
    kernel = _kernel(_mm_nested_k)
    bound = _cpu_bind(kernel, (torch.ones(32, 256), torch.ones(256, 32)))
    _assert_scalar_dot(bound.to_code(_tile_config(bound)))
    # Nor the SIMT GEMM: its accumulator lives ahead of the K loop, so the
    # sub-range must subdivide a root tile, not a loop that re-enters it.
    _assert_scalar_dot(bound.to_code(_tile_config(bound, cute_simt_mma=True)))


def test_simt_knob_unavailable_for_tf32_contractions() -> None:
    # fp32 matmuls under the default dot_precision ("tf32") belong to the
    # tensor cores: no knob, no field, no inactive-root relaxation.
    kernel = helion.kernel(
        _mm_ieee, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = _cpu_bind(kernel, (torch.ones(64, 256), torch.ones(256, 64)))
    spec = bound.config_spec
    assert not spec.cute_fp32_dot_exact
    assert not spec.cute_simt_mma_enabled
    assert "cute_simt_mma" not in spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="cute_simt_mma"):
        spec.normalize(_simt_config(bound, cute_simt_mma=True))
    exact = _cpu_bind(_kernel(_mm_ieee), (torch.ones(64, 256), torch.ones(256, 64)))
    assert (
        exact.config_spec.cute_fp32_dot_exact
        and exact.config_spec.cute_simt_mma_enabled
    )


@skipIfNotCUDA()
def test_seeded_subrange_k_matches_reference() -> None:
    a = torch.randn(32, 4096, device=DEVICE)
    b = torch.randn(32, 4096, device=DEVICE).T
    bias = torch.randn(32, 32, device=DEVICE)
    bound = _kernel(_mm_seeded_split_k).bind((a, b, bias))
    compiled = bound.compile_config(_tile_config(bound, cute_simt_mma=True))
    tmp = compiled(a, b, bias)
    expected = a @ b + bias * tmp.shape[-1]
    torch.testing.assert_close(tmp.sum(-1), expected, atol=1e-2, rtol=1e-3)


@skipIfNotCUDA()
@pytest.mark.parametrize("knob", [False, True])
def test_nested_subrange_k_matches_reference(knob: bool) -> None:
    a = torch.randn(32, 480, device=DEVICE)
    b = torch.randn(32, 480, device=DEVICE).T
    bound = _kernel(_mm_nested_k).bind((a, b))
    compiled = bound.compile_config(_tile_config(bound, cute_simt_mma=knob))
    torch.testing.assert_close(compiled(a, b), a @ b, atol=1e-3, rtol=1e-3)
