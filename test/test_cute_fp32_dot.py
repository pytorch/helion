"""``cute_fp32_dot``: tf32 ``mma.sync`` for fp32 dots in computed fragments.

The CPU tests pin the search surface and the generated contraction; the GPU
tests check the numerics of whole, partial and accumulated tiles.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl

KEY = "cute_fp32_dot"
MMA = "_helion_fragment_mma.mma_tf32("

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunk_cumsum(g: torch.Tensor) -> torch.Tensor:
    """Prefix sums over the chunk axis as ``L @ g`` (the linear-attention
    cumsum): a 0/1 operand and a bf16-valued one, exact in tf32."""
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        idx = hl.arange(chunk)
        ltri = (idx[:, None] >= idx[None, :]).to(torch.float32)
        lower = ltri[None, :, :].broadcast_to([tile_b, chunk, chunk])  # pyrefly: ignore[no-matching-overload]
        values = g[tile_b, :, tile_d].to(torch.float32)
        out[tile_b, :, tile_d] = hl.dot(lower, values)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_dot(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """``x @ y + z`` per batch with fp32 operands and a seeded accumulator."""
    batches = x.size(0)
    rows = hl.specialize(x.size(1))
    cols = hl.specialize(y.size(2))
    out = torch.empty([batches, rows, cols], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(batches, block_size=1):
        out[tile_b, :, :] = hl.dot(
            x[tile_b, :, :], y[tile_b, :, :], acc=z[tile_b, :, :]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sixteen_row_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One ``[16, 128] x [128, 128]`` fp32 dot per batch: 64^3 multiply-adds
    although no axis reaches 64."""
    out = torch.empty([x.size(0), 16, 128], dtype=torch.float32, device=x.device)
    for tile_b in hl.tile(x.size(0), block_size=1):
        out[tile_b, :, :] = hl.dot(x[tile_b, :, :], y[tile_b, :, :])
    return out


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", dot_precision="ieee"
)
def _chunk_cumsum_ieee(g: torch.Tensor) -> torch.Tensor:
    batches = g.size(0)
    chunk = hl.specialize(g.size(1))
    width = g.size(2)
    out = torch.empty([batches, chunk, width], dtype=torch.float32, device=g.device)
    for tile_b, tile_d in hl.tile([batches, width], block_size=[1, None]):
        idx = hl.arange(chunk)
        ltri = (idx[:, None] >= idx[None, :]).to(torch.float32)
        lower = ltri[None, :, :].broadcast_to([tile_b, chunk, chunk])  # pyrefly: ignore[no-matching-overload]
        values = g[tile_b, :, tile_d].to(torch.float32)
        out[tile_b, :, tile_d] = hl.dot(lower, values)
    return out


@pytest.fixture
def _cpu_only():
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        yield


def _config(bound: object, mode: str, **extra: object) -> helion.Config:
    spec = bound.config_spec  # type: ignore[attr-defined]
    return helion.Config.from_dict(dict(spec.default_config()) | {KEY: mode} | extra)


def test_fp32_dot_enum_defaults_to_simt(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    spec = bound.config_spec
    assert spec.cute_fp32_dot_root_ids
    assert spec.cute_fp32_dot_seed_worthy
    field = spec._flat_fields()[KEY]
    assert isinstance(field, EnumFragment)
    assert field.choices == ("simt", "tf32_mma")
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    default = spec.default_config()
    assert KEY not in default
    simt = _config(bound, "simt")
    spec.normalize(simt.config)
    assert simt == default
    generation = spec.create_config_generation()
    _, effective = generation.strict_config_pair(_config(bound, "tf32_mma"))
    assert effective[KEY] == "tf32_mma"
    for invalid in ("tf32", "TF32_MMA", True, 1, None):
        with pytest.raises(exc.InvalidConfig, match="fp32 hl.dot"):
            spec.normalize(_config(bound, invalid).config)  # type: ignore[arg-type]
    # The 64x64x64 cumsum seeds the mode as a coverage witness.
    assert any(
        group.key == KEY and any(w.value == "tf32_mma" for w in group.witnesses)
        for group in spec.compiler_coverage_groups
    )


def test_fp32_dot_lowers_the_fragment_dot_through_mma_sync(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    simt = bound.to_code(_config(bound, "simt"))
    assert "fragment_dot" in simt
    assert MMA not in simt
    mma = bound.to_code(_config(bound, "tf32_mma"))
    assert "fragment_dot" not in mma
    assert (
        "import helion._compiler.cute._mma_dag_runtime as _helion_fragment_mma" in mma
    )
    assert mma.count(MMA) == 1
    assert "for fragment_mma_step in cutlass.range_constexpr(8):" in mma
    # 64 x 64 x 64 tiles exactly: no predicates on the operand reads or stores.
    assert "fragment_mma_operand = cutlass.Float32(0)" not in mma
    assert mma.count("_helion_fragment_mma.f32_bits(") == 6


def test_fp32_dot_partial_tiles_are_predicated(_cpu_only: None) -> None:
    x = torch.ones(2, 24, 20)
    y = torch.ones(2, 20, 12)
    z = torch.ones(2, 24, 12)
    bound = _cpu_bind(_batched_dot, (x, y, z))
    assert bound.config_spec.cute_fp32_dot_root_ids
    # Nothing of a 24 x 12 x 20 dot reaches 64: the field is searchable (the
    # group owns it) but no witness seeds the mode.
    assert not bound.config_spec.cute_fp32_dot_seed_worthy
    (group,) = [g for g in bound.config_spec.compiler_coverage_groups if g.key == KEY]
    assert group.witnesses == () and group.supplemental_witnesses == ()
    assert isinstance(bound.config_spec._flat_fields()[KEY], EnumFragment)
    code = bound.to_code(_config(bound, "tf32_mma"))
    assert code.count(MMA) == 1
    assert "for fragment_mma_step in cutlass.range_constexpr(3):" in code
    assert "fragment_mma_operand = cutlass.Float32(0)" in code
    assert "< 24" in code and "< 12" in code and "< 20" in code


def test_the_seed_counts_the_roots_multiply_adds(_cpu_only: None) -> None:
    # No axis of a [16, 128] x [128, 128] dot reaches 64, yet it is one
    # 64^3 of multiply-adds: the anchored chunk grams issue eight such
    # 16-row dots per tile, and the seed must offer them the tensor cores.
    bound = _cpu_bind(
        _sixteen_row_dot, (torch.ones(4, 16, 128), torch.ones(4, 128, 128))
    )
    assert bound.config_spec.cute_fp32_dot_root_ids
    assert bound.config_spec.cute_fp32_dot_seed_worthy
    assert any(
        group.key == KEY and any(w.value == "tf32_mma" for w in group.witnesses)
        for group in bound.config_spec.compiler_coverage_groups
    )
    # Half the contraction is half the work: below the seed.
    half = _cpu_bind(_sixteen_row_dot, (torch.ones(4, 16, 64), torch.ones(4, 64, 128)))
    assert half.config_spec.cute_fp32_dot_root_ids
    assert not half.config_spec.cute_fp32_dot_seed_worthy


def test_bf16_packet_loads_feed_the_tf32_dot(_cpu_only: None) -> None:
    # The cumsum's bf16 chunk tile is staged through 8-byte packets (four
    # bf16 per load) instead of one 2-byte load per element; with the dot on
    # tensor cores that load phase was the kernel's remaining bottleneck.
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum, (g,))
    assert "cute_fragment_packet_loads" in bound.config_spec._flat_fields()
    code = bound.to_code(_config(bound, "tf32_mma", cute_fragment_packet_loads=True))
    assert MMA in code
    assert "cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.BFloat16)" in code
    assert "num_bits_per_copy=64" in code


def test_ieee_dot_precision_never_offers_tf32(_cpu_only: None) -> None:
    g = torch.ones(4, 64, 128, dtype=torch.bfloat16)
    bound = _cpu_bind(_chunk_cumsum_ieee, (g,))
    assert not bound.config_spec.cute_fp32_dot_root_ids
    assert KEY not in bound.config_spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="fp32 hl.dot"):
        bound.config_spec.normalize(_config(bound, "tf32_mma").config)
    assert "fragment_dot" in bound.to_code(bound.config_spec.default_config())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("threads", "packets"), [(64, False), (128, False), (256, False), (256, True)]
)
def test_cumsum_dot_matches_reference(threads: int, packets: bool) -> None:
    torch.manual_seed(0)
    g = torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    bound = _chunk_cumsum.bind((g,))
    config = _config(
        bound,
        "tf32_mma",
        cute_fragment_threads=threads,
        cute_fragment_packet_loads=packets,
    )
    code = bound.to_code(config)
    assert MMA in code
    assert ("num_bits_per_copy=64" in code) == packets
    out = bound.compile_config(config)(g)
    torch.cuda.synchronize()
    expected = g.float().cumsum(1)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("rows", "cols", "contraction"),
    [(32, 16, 16), (24, 12, 20), (16, 8, 8), (48, 40, 64)],
)
def test_batched_dot_with_accumulator_matches_reference(
    rows: int, cols: int, contraction: int
) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, rows, contraction, device=DEVICE)
    y = torch.randn(3, contraction, cols, device=DEVICE)
    z = torch.randn(3, rows, cols, device=DEVICE)
    bound = _batched_dot.bind((x, y, z))
    config = _config(bound, "tf32_mma")
    assert MMA in bound.to_code(config)
    out = bound.compile_config(config)(x, y, z)
    torch.cuda.synchronize()
    expected = x @ y + z
    # tf32 operands: ~1e-3 relative per product.
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=2e-2)
    exact = bound.compile_config(_config(bound, "simt"))(x, y, z)
    torch.testing.assert_close(exact, expected, rtol=1e-5, atol=1e-5)
