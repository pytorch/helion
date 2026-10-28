"""Real CUDA coverage for two-CTA persistent GEMM launches of a barrier chain.

``tcgen05_region_pid_types`` runs chosen GEMM phases of an ordered
``hl.barrier()`` chain (the two-GEMM ``squeeze_and_excitation_net`` backward
chains of ``test._cute_se_bwd_chains``, and the one-GEMM ``bwd_dc`` chain of
the example itself) as the static persistent CtaGroup.TWO pipeline while the
chain and its pointwise phase stay flat; the exact-shape auxiliary epilogues
(the relu-backward mask, ``grad_out * d + acc``) ride along, and a launch
left flat keeps the one-CTA pipeline at the AB depth its tile fits.  Each
render is checked against the fp32 torch reference and must match the flat
chain's numerics bitwise (same tiles, same K order, same epilogue
arithmetic).
"""

from __future__ import annotations

from typing import cast

from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_bwd_dc
import pytest
import torch

from test._cute_se_bwd_chains import se_bwd_da_chain
from test._cute_se_bwd_chains import se_bwd_dx_chain

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _requires_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")


def _inputs(
    shape: tuple[int, int, int] = (512, 512, 256),
) -> dict[str, torch.Tensor]:
    m, n, k = shape
    generator = torch.Generator().manual_seed(1105)
    x = torch.randn((m, n), generator=generator, dtype=torch.float16)
    a = torch.randn((n, k), generator=generator, dtype=torch.float16)
    b = torch.randn((k, n), generator=generator, dtype=torch.float16)
    grad_out = torch.randn((m, n), generator=generator, dtype=torch.float16)
    tensors = {"x": x, "a": a, "b": b, "grad_out": grad_out}
    tensors["c"] = torch.relu(x.float() @ a.float()).to(torch.float16)
    tensors["d"] = torch.sigmoid(tensors["c"].float() @ b.float()).to(torch.float16)
    return {name: value.to(DEVICE) for name, value in tensors.items()}


def _reference(t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    grad_cb = (t["grad_out"].float() * t["x"].float() * t["d"].float()) * (
        1.0 - t["d"].float()
    )
    grad_c = (grad_cb.to(torch.float16).float() @ t["b"].float().T) * (t["c"] > 0)
    grad_c16 = grad_c.to(torch.float16).float()
    grad_x = t["grad_out"].float() * t["d"].float() + grad_c16 @ t["a"].float().T
    grad_a = t["x"].float().T @ grad_c16
    return grad_x, grad_a


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    return ((actual - expected).norm() / expected.norm()).item()


def _chain_config(
    bound: helion.runtime.kernel.BoundKernel,  # type: ignore[name-defined]
    region_pid_types: list[str] | None,
    *,
    tiles: list[int],
    region_ab_stages: list[int],
    pdl: bool,
    **overrides: object,
) -> helion.Config:
    spec = bound.config_spec
    config = dict(spec.default_config().config)
    blocks = list(config["block_sizes"])
    blocks[2:8] = tiles
    config.update(block_sizes=blocks, pid_type="flat", cute_pdl=pdl)
    if region_pid_types is not None:
        config.update(
            tcgen05_region_pid_types=region_pid_types,
            tcgen05_cta_group="two",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_region_ab_stages=region_ab_stages,
        )
    config.update(overrides)
    result = helion.Config.from_dict(config)
    spec.normalize(result)
    return result


@pytest.mark.parametrize("pdl", [False, True], ids=["serial", "pdl"])
@pytest.mark.parametrize(
    "region_pid_types",
    [
        ["persistent_blocked", "persistent_blocked"],
        ["persistent_interleaved", "persistent_interleaved"],
        ["persistent_blocked", "inherit"],
        ["inherit", "persistent_blocked"],
    ],
    ids=["blocked", "interleaved", "first_only", "second_only"],
)
def test_bwd_dx_two_cta_region_launches(region_pid_types: list[str], pdl: bool) -> None:
    _check_bwd_dx(region_pid_types, pdl, {})


def test_bwd_dx_inherit_launch_keeps_the_flat_recipe() -> None:
    """The campaign's flat winner recipe (a scheduler warp, a one-slot
    accumulator ring) stays on the ``inherit`` launch while grad_c runs the
    projected two-CTA pipeline; numerics stay bitwise."""
    _check_bwd_dx(
        ["persistent_blocked", "inherit"],
        True,
        {
            "tcgen05_strategy": "role_local_with_scheduler",
            "tcgen05_warp_spec_scheduler_warps": 1,
            "tcgen05_acc_stages": 1,
        },
    )


def _check_bwd_dx(
    region_pid_types: list[str], pdl: bool, recipe: dict[str, object]
) -> None:
    _requires_sm100()
    t = _inputs()
    args = (t["grad_out"], t["x"], t["a"], t["b"], t["c"], t["d"])
    grad_x_ref, _grad_a_ref = _reference(t)
    kernel = helion.kernel(
        se_bwd_dx_chain,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    assert bound.config_spec._cute_tcgen05_config.region_pid_type_choices()
    # grad_c: 512x256x512 (relu-backward mask), grad_x: 512x512x256
    # (grad_out * d + acc); 128x128x64 CTA-pair tiles, 4-deep AB rings.
    tiles = [128, 128, 64, 128, 128, 64]
    flat = _chain_config(
        bound, None, tiles=tiles, region_ab_stages=[], pdl=pdl, **recipe
    )
    # A launch left flat keeps the shared ring depth (entry 0).
    two_cta = _chain_config(
        bound,
        region_pid_types,
        tiles=tiles,
        region_ab_stages=[
            0 if pid_type == "inherit" else 4 for pid_type in region_pid_types
        ],
        pdl=pdl,
        tcgen05_aux_load_placement="pre_acc_wait",
        **recipe,
    )
    assert two_cta.config["tcgen05_region_pid_types"] == region_pid_types
    for key, value in recipe.items():
        assert two_cta.config[key] == value, key
    flat_out = bound.compile_config(flat, allow_print=False)(*args)
    two_cta_out = bound.compile_config(two_cta, allow_print=False)(*args)
    torch.cuda.synchronize()
    assert _rel_l2(flat_out, grad_x_ref) < 2e-3
    # Same tiles, K order and epilogue arithmetic: the schedule is bitwise.
    torch.testing.assert_close(two_cta_out, flat_out, atol=0, rtol=0)


@pytest.mark.parametrize("pdl", [False, True], ids=["serial", "pdl"])
def test_bwd_da_two_cta_region_launches(pdl: bool) -> None:
    _requires_sm100()
    t = _inputs()
    args = (t["grad_out"], t["x"], t["b"], t["c"], t["d"])
    _grad_x_ref, grad_a_ref = _reference(t)
    kernel = helion.kernel(
        se_bwd_da_chain,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    # grad_c: 512x256x512 (relu-backward mask); grad_a: 512x256x512 with a
    # transposed (column-major) A operand and a plain store: the deep ring of
    # the standalone long-K seed (128x64x128, 8 stages).
    tiles = [128, 128, 64, 128, 64, 128]
    flat = _chain_config(bound, None, tiles=tiles, region_ab_stages=[], pdl=pdl)
    two_cta = _chain_config(
        bound,
        ["persistent_blocked", "persistent_blocked"],
        tiles=tiles,
        region_ab_stages=[4, 4],
        pdl=pdl,
    )
    flat_out = bound.compile_config(flat, allow_print=False)(*args)
    two_cta_out = bound.compile_config(two_cta, allow_print=False)(*args)
    torch.cuda.synchronize()
    assert _rel_l2(flat_out, grad_a_ref) < 2e-3
    torch.testing.assert_close(two_cta_out, flat_out, atol=0, rtol=0)


@pytest.mark.parametrize("pdl", [False, True], ids=["serial", "pdl"])
def test_bwd_dc_one_gemm_chain_region_launches(pdl: bool) -> None:
    """One GEMM phase beside a pointwise phase is planned per phase: grad_c
    runs as a CtaGroup.TWO persistent launch while grad_cb renders 16-byte
    packets, and the flat launch's ring reaches the depth its tile fits."""
    _requires_sm100()
    t = _inputs()
    args = (t["grad_out"], t["x"], t["b"], t["c"], t["d"])
    grad_cb = (t["grad_out"].float() * t["x"].float() * t["d"].float()) * (
        1.0 - t["d"].float()
    )
    grad_c_ref = (grad_cb.to(torch.float16).float() @ t["b"].float().T) * (t["c"] > 0)
    kernel = helion.kernel(
        squeeze_and_excitation_net_bwd_dc.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    state = bound.config_spec._cute_tcgen05_config
    # grad_cb: 512x512 pointwise phase; grad_c: 512x256x512 (relu mask).
    assert state.materialized_matmul_shapes == ((512, 256, 512),)
    assert state.region_pid_type_choices()
    tiles = [128, 128, 64]
    flat = _chain_config(bound, None, tiles=tiles, region_ab_stages=[], pdl=pdl)
    assert flat.config["tcgen05_ab_stages"] == 2
    # The one-CTA fit of a 128x128x64 fp16 tile beside a 4-stage output
    # ring: 6 AB stages (the flat chain used to validate 3 at most).
    deep_flat = _chain_config(
        bound,
        None,
        tiles=tiles,
        region_ab_stages=[],
        pdl=pdl,
        tcgen05_ab_stages=6,
        tcgen05_c_stages=4,
    )
    assert deep_flat.config["tcgen05_ab_stages"] == 6
    with pytest.raises(helion.exc.InvalidConfig, match="maximum AB stages: 6"):
        _chain_config(
            bound,
            None,
            tiles=tiles,
            region_ab_stages=[],
            pdl=pdl,
            tcgen05_ab_stages=7,
            tcgen05_c_stages=4,
        )
    pointwise = {
        "num_threads": [1, 64, 0, 0, 0],
        "cute_vector_widths": [1, 8, 1, 1, 1],
    }
    two_cta = _chain_config(
        bound,
        ["persistent_blocked"],
        tiles=tiles,
        region_ab_stages=[4],
        pdl=pdl,
        tcgen05_aux_load_placement="pre_acc_wait",
        **pointwise,
    )
    blocks = list(two_cta.block_sizes)
    blocks[:2] = [1, 512]
    # The chain's launches raster under the shared L2 swizzle (two N tiles
    # of 128 at N=256 admit a swizzle of 2).
    two_cta = _chain_config(
        bound,
        ["persistent_blocked"],
        tiles=tiles,
        region_ab_stages=[4],
        pdl=pdl,
        tcgen05_aux_load_placement="pre_acc_wait",
        tcgen05_l2_swizzle_size=2,
        block_sizes=blocks,
        **pointwise,
    )
    assert two_cta.config["tcgen05_region_pid_types"] == ["persistent_blocked"]
    assert two_cta.config["cute_vector_widths"] == [1, 8, 1, 1, 1]
    assert two_cta.config["tcgen05_l2_swizzle_size"] == 2
    assert two_cta.block_sizes[:2] == [1, 512]
    source = bound.to_code(two_cta)
    assert "CtaGroup.TWO" in source
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in source
    # The kernel returns (grad_cb, grad_c).
    flat_out = bound.compile_config(flat, allow_print=False)(*args)
    deep_out = bound.compile_config(deep_flat, allow_print=False)(*args)
    two_cta_out = bound.compile_config(two_cta, allow_print=False)(*args)
    torch.cuda.synchronize()
    assert _rel_l2(flat_out[1], grad_c_ref) < 2e-3
    torch.testing.assert_close(deep_out, flat_out, atol=0, rtol=0)
    torch.testing.assert_close(two_cta_out, flat_out, atol=0, rtol=0)


def test_bwd_dx_flat_ring_bound_at_bk128() -> None:
    """At bm=128/bk=128 a flat launch's 16-bit ring stops at the standalone
    surface's admission (4 stages of 128x32x128, a 160 KiB ring), as the
    kernel's knob and as an ``inherit`` launch's region entry; the admitted
    depth is bitwise equal to the shallow ring.  The deeper rings the bare
    one-CTA capacity would hold (ab5, 200 KiB) changed grad_x's rounding in
    about 0.1 % of its elements and are refused."""
    _requires_sm100()
    t = _inputs((1024, 512, 256))
    args = (t["grad_out"], t["x"], t["a"], t["b"], t["c"], t["d"])
    grad_x_ref, _grad_a_ref = _reference(t)
    kernel = helion.kernel(
        se_bwd_dx_chain,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    # grad_c 128x128x64 (one K tile of 64 fits 6), grad_x 128x32x128.
    tiles = [128, 128, 64, 128, 32, 128]
    flat = _chain_config(bound, None, tiles=tiles, region_ab_stages=[], pdl=False)
    deep_flat = _chain_config(
        bound, None, tiles=tiles, region_ab_stages=[], pdl=False, tcgen05_ab_stages=4
    )
    assert deep_flat.config["tcgen05_ab_stages"] == 4
    with pytest.raises(helion.exc.InvalidConfig, match="maximum AB stages: 4"):
        _chain_config(
            bound,
            None,
            tiles=tiles,
            region_ab_stages=[],
            pdl=False,
            tcgen05_ab_stages=5,
        )
    mixed = _chain_config(
        bound,
        ["persistent_blocked", "inherit"],
        tiles=tiles,
        region_ab_stages=[4, 4],
        pdl=False,
        tcgen05_aux_load_placement="pre_acc_wait",
    )
    assert mixed.config["tcgen05_region_ab_stages"] == [4, 4]
    with pytest.raises(helion.exc.InvalidConfig, match="per-region SMEM"):
        _chain_config(
            bound,
            ["persistent_blocked", "inherit"],
            tiles=tiles,
            region_ab_stages=[4, 5],
            pdl=False,
            tcgen05_aux_load_placement="pre_acc_wait",
        )
    flat_out = bound.compile_config(flat, allow_print=False)(*args)
    deep_out = bound.compile_config(deep_flat, allow_print=False)(*args)
    mixed_out = bound.compile_config(mixed, allow_print=False)(*args)
    torch.cuda.synchronize()
    assert _rel_l2(flat_out, grad_x_ref) < 2e-3
    torch.testing.assert_close(deep_out, flat_out, atol=0, rtol=0)
    torch.testing.assert_close(mixed_out, flat_out, atol=0, rtol=0)


def test_chain_seeds_compile_and_match() -> None:
    """The heuristic's two-CTA chain seeds (``region_two_cta_seed_configs``)
    strict-normalize, compile and match the flat default bitwise."""
    _requires_sm100()
    t = _inputs()
    args = (t["grad_out"], t["x"], t["a"], t["b"], t["c"], t["d"])
    kernel = helion.kernel(
        se_bwd_dx_chain,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    spec = bound.config_spec
    seeds = spec._cute_tcgen05_config.region_two_cta_seed_configs()
    # Two K tiles x (all persistent, grad_c flat, grad_x flat).
    assert len(seeds) == 6
    expected = bound.compile_config(spec.default_config(), allow_print=False)(*args)
    for seed in seeds:
        config = helion.Config.from_dict(spec.default_config().config | seed.config)
        spec.normalize(config)
        assert (
            config.config["tcgen05_region_pid_types"]
            == seed.config["tcgen05_region_pid_types"]
        )
        actual = bound.compile_config(config, allow_print=False)(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chain_with_vector_phase(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The dx chain behind a 1-D vector phase: the GEMM roots (2, 3) and the
    ``loop_orders`` entries (one per multi-dim tile loop: 1, 2) differ."""
    m, n = x.size()
    k = a.size(1)
    grad_cb = torch.empty([m, n], dtype=x.dtype, device=x.device)
    grad_c = torch.empty([m, k], dtype=x.dtype, device=x.device)
    grad_x = torch.empty([m, n], dtype=x.dtype, device=x.device)
    w = torch.empty_like(v)
    for pv in hl.tile(v.size(0)):
        w[pv] = v[pv] * 2.0
    hl.barrier()
    for pm, pn in hl.tile((m, n)):
        grad_cb[pm, pn] = grad_out[pm, pn] * x[pm, pn] * d[pm, pn] * (1.0 - d[pm, pn])
    hl.barrier()
    for rc, cc in hl.tile((m, k)):
        acc = hl.zeros([rc, cc], dtype=torch.float32)
        for rn in hl.tile(n):
            acc = torch.addmm(acc, grad_cb[rc, rn], b[cc, rn].T)
        grad_c[rc, cc] = acc.to(x.dtype) * (c[rc, cc] > 0)
    hl.barrier()
    for rx, cx in hl.tile((m, n)):
        acc = hl.zeros([rx, cx], dtype=torch.float32)
        for rk in hl.tile(k):
            acc = torch.addmm(acc, grad_c[rx, rk], a[cx, rk].T)
        grad_x[rx, cx] = grad_out[rx, cx] * d[rx, cx] + acc.to(x.dtype)
    return grad_x, w


def test_chain_with_a_vector_phase_binds_clustered_rasters_by_block_ids() -> None:
    """Behind a 1-D phase the clustered GEMM launches' ``loop_orders`` entries
    are found by block ids: a swapped GEMM raster is refused and repaired to
    the default, the pointwise phase keeps its own order, and the chain runs
    bitwise-equal to the flat chain."""
    _requires_sm100()
    t = _inputs()
    v = torch.randn(
        777, generator=torch.Generator().manual_seed(9), dtype=torch.float16
    )
    v = v.to(DEVICE)
    args = (t["grad_out"], t["x"], t["a"], t["b"], t["c"], t["d"], v)
    grad_x_ref, _grad_a_ref = _reference(t)
    bound = _chain_with_vector_phase._bind_isolated(args)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.materialized_matmul_region_indices == (2, 3)
    assert [tuple(item.block_ids) for item in spec.loop_orders] == [
        (1, 2),
        (3, 4),
        (6, 7),
    ]
    base = dict(spec.default_config().config)
    blocks = list(cast("list[int]", base["block_sizes"]))
    blocks[3:9] = [128, 128, 64, 128, 128, 64]
    base.update(block_sizes=blocks, pid_type="flat", cute_pdl=True)
    flat = helion.Config.from_dict(base | {"loop_orders": [[1, 0], [0, 1], [0, 1]]})
    two_cta = dict(
        base,
        tcgen05_region_pid_types=["persistent_blocked", "persistent_blocked"],
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_region_ab_stages=[4, 4],
        tcgen05_aux_load_placement="pre_acc_wait",
    )
    swapped = helion.Config.from_dict(
        two_cta | {"loop_orders": [[1, 0], [1, 0], [0, 1]]}
    )
    with pytest.raises(helion.exc.InvalidConfig, match="clustered GEMM launch"):
        spec.normalize(swapped)
    spec.normalize(swapped, _fix_invalid=True)
    assert swapped.config["loop_orders"] == [[1, 0], [0, 1], [0, 1]]
    assert swapped.config["tcgen05_region_pid_types"] == ["persistent_blocked"] * 2
    spec.normalize(flat)
    flat_out, flat_w = bound.compile_config(flat, allow_print=False)(*args)
    out, w = bound.compile_config(swapped, allow_print=False)(*args)
    torch.cuda.synchronize()
    assert _rel_l2(flat_out, grad_x_ref) < 2e-3
    torch.testing.assert_close(flat_w, v * 2.0, atol=0, rtol=0)
    torch.testing.assert_close(out, flat_out, atol=0, rtol=0)
    torch.testing.assert_close(w, flat_w, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_gemm(
    grad_cb: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, n = grad_cb.size()
    k = b.size(0)
    grad_c = torch.empty([m, k], dtype=grad_cb.dtype, device=grad_cb.device)
    for row, col in hl.tile((m, k)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(n):
            acc = torch.addmm(acc, grad_cb[row, inner], b[col, inner].T)
        grad_c[row, col] = acc.to(grad_cb.dtype) * (c[row, col] > 0)
    return grad_c


@pytest.mark.parametrize("placement", ["post_acc_wait", "pre_acc_wait"])
def test_standalone_masked_gemm_paired_pipeline(placement: str) -> None:
    """A single 16-bit GEMM with an exact-shape auxiliary epilogue now owns
    the complete allocation proof: the paired two-CTA pipeline renders and
    matches the one-CTA flat schedule bitwise."""
    _requires_sm100()
    t = _inputs()
    grad_cb = (t["grad_out"].float() * t["x"].float()).to(torch.float16)
    args = (grad_cb, t["b"], t["c"])
    bound = _masked_gemm._bind_isolated(args)
    state = bound.config_spec._cute_tcgen05_config
    assert state.pipeline_smem_facts is not None
    assert state.pipeline_smem_facts.has_auxiliary_loads
    flat = helion.Config(block_sizes=[128, 128, 64], pid_type="flat")
    paired = helion.Config(
        block_sizes=[128, 128, 64],
        pid_type="persistent_blocked",
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        tcgen05_ab_stages=8,
        tcgen05_persistence_model="static_persistent",
        tcgen05_aux_load_placement=placement,
    )
    for config in (flat, paired):
        bound.config_spec.normalize(config)
    assert paired.config["tcgen05_cta_group"] == "two"
    expected = ((grad_cb.float() @ t["b"].float().T) * (t["c"] > 0)).to(torch.float16)
    flat_out = bound.compile_config(flat, allow_print=False)(*args)
    paired_out = bound.compile_config(paired, allow_print=False)(*args)
    torch.cuda.synchronize()
    assert _rel_l2(flat_out, expected) < 2e-3
    torch.testing.assert_close(paired_out, flat_out, atol=0, rtol=0)


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", cute_region_fission=True
)
def _plain_fission_pair(
    left: torch.Tensor, hidden: torch.Tensor, expansion: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two plain single-output GEMMs sharing a row tile: a fission plan
    without a shared epilogue fanout."""
    rows, inner = left.size()
    mid = hidden.size(1)
    cols = expansion.size(1)
    middle = torch.empty((rows, mid), dtype=left.dtype, device=left.device)
    out = torch.empty((rows, cols), dtype=left.dtype, device=left.device)
    for row in hl.tile(rows):
        for col in hl.tile(mid):
            acc = hl.zeros([row, col], dtype=torch.float32)
            for red in hl.tile(inner):
                acc = torch.addmm(acc, left[row, red], hidden[red, col])
            middle[row, col] = acc
        for col in hl.tile(cols):
            acc = hl.zeros([row, col], dtype=torch.float32)
            for red in hl.tile(mid):
                acc = torch.addmm(acc, middle[row, red], expansion[red, col])
            out[row, col] = acc
    return middle, out


def test_plain_fission_plan_two_cta_schedule() -> None:
    """A fission plan of plain single-output GEMMs (no shared fanout) is
    budgeted per region from its complete facts: the static persistent
    two-CTA schedule is admitted, each region renders the CtaGroup.TWO
    cluster pipeline, and the plan matches the flat plan bitwise."""
    _requires_sm100()
    generator = torch.Generator().manual_seed(3)
    rows, inner, mid, cols = 1024, 512, 512, 1024
    left = torch.randn((rows, inner), generator=generator, dtype=torch.float16)
    hidden = torch.randn((inner, mid), generator=generator, dtype=torch.float16)
    expansion = torch.randn((mid, cols), generator=generator, dtype=torch.float16)
    args = (left.to(DEVICE), hidden.to(DEVICE), expansion.to(DEVICE))
    bound = _plain_fission_pair._bind_isolated(args)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert bound.env.cute_fission_plan is not None
    assert len(state.materialized_matmul_block_ids) == 2
    default = dict(spec.default_config().config)
    blocks = list(cast("list[int]", default["block_sizes"]))
    for block_ids in state.materialized_matmul_block_ids:
        for block_id, size in zip(block_ids, (128, 128, 64), strict=True):
            blocks[spec.block_sizes.block_id_to_index(block_id)] = size
    flat = helion.Config.from_dict(
        default | {"block_sizes": blocks, "pid_type": "flat"}
    )
    paired = helion.Config.from_dict(
        default
        | {
            "block_sizes": blocks,
            "pid_type": "persistent_blocked",
            "tcgen05_cta_group": "two",
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_strategy": "role_local_monolithic",
            "tcgen05_ab_stages": 3,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
        }
    )
    for config in (flat, paired):
        spec.normalize(config)
    assert paired.pid_type == "persistent_blocked"
    assert paired.config["tcgen05_cta_group"] == "two"
    assert paired.config["tcgen05_cluster_m"] == 2
    code = bound.to_code(paired)
    assert "CtaGroup.TWO" in code
    assert "_helion_cute_cluster_shape = (2, 1, 1)" in code
    middle_flat, out_flat = bound.compile_config(flat, allow_print=False)(*args)
    middle_paired, out_paired = bound.compile_config(paired, allow_print=False)(*args)
    torch.cuda.synchronize()
    assert _rel_l2(middle_flat, args[0].float() @ args[1].float()) < 2e-3
    assert _rel_l2(out_flat, middle_flat.float() @ args[2].float()) < 2e-3
    torch.testing.assert_close(middle_paired, middle_flat, atol=0, rtol=0)
    torch.testing.assert_close(out_paired, out_flat, atol=0, rtol=0)


# The exact-shape auxiliary mechanisms on the chain's persistent pair launch
# (``test_cute_chain_aux_epilogue.py`` pins the admission and the renders):
# each moves the relu mask read of ``grad_c = acc.to(fp16) * (c[tile] > 0)``
# out of the per-subtile epilogue path without touching the tiles, the K
# order or the arithmetic, so every render is bitwise the plain pair's.
_AUX_MECHANISMS: dict[str, dict[str, object]] = {
    "prefetch_ring": {"tcgen05_aux_prefetch_subtiles": 2},
    "fragment_hoist": {"tcgen05_aux_fragment_hoist": True},
    "epi_tma": {"tcgen05_aux_load_mode": "epi_tma"},
    "epi_warp_groups": {"tcgen05_epi_warp_groups": 2},
    "epi_subtile": {"tcgen05_epi_subtile_n": 32},
    "ring_groups_subtile": {
        "tcgen05_aux_prefetch_subtiles": 1,
        "tcgen05_epi_warp_groups": 2,
        "tcgen05_epi_subtile_n": 32,
    },
}


@pytest.mark.parametrize("mechanism", sorted(_AUX_MECHANISMS))
def test_bwd_dc_pair_launch_aux_mechanisms_match_bitwise(mechanism: str) -> None:
    _requires_sm100()
    t = _inputs()
    args = (t["grad_out"], t["x"], t["b"], t["c"], t["d"])
    grad_cb = (t["grad_out"].float() * t["x"].float() * t["d"].float()) * (
        1.0 - t["d"].float()
    )
    grad_c_ref = (grad_cb.to(torch.float16).float() @ t["b"].float().T) * (t["c"] > 0)
    kernel = helion.kernel(
        squeeze_and_excitation_net_bwd_dc.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    # grad_c: 512x256x512 on the 256x128 CtaGroup.TWO pair (2 x 2 pairs).
    recipe: dict[str, object] = {"tcgen05_aux_load_placement": "pre_acc_wait"}
    plain = _chain_config(
        bound,
        ["persistent_blocked"],
        tiles=[256, 128, 64],
        region_ab_stages=[4],
        pdl=True,
        **recipe,
    )
    staged = _chain_config(
        bound,
        ["persistent_blocked"],
        tiles=[256, 128, 64],
        region_ab_stages=[4],
        pdl=True,
        **recipe,
        **_AUX_MECHANISMS[mechanism],
    )
    for key, value in _AUX_MECHANISMS[mechanism].items():
        assert staged.config[key] == value, key
    # (the bundle module embeds the stage sources as string constants, so the
    # launch attribute appears with escaped quotes)
    source = bound.to_code(staged)
    assert "CtaGroup.TWO" in source and "use_pdl" in source
    plain_out = bound.compile_config(plain, allow_print=False)(*args)
    staged_out = bound.compile_config(staged, allow_print=False)(*args)
    torch.cuda.synchronize()
    assert _rel_l2(plain_out[1], grad_c_ref) < 2e-3
    torch.testing.assert_close(staged_out, plain_out, atol=0, rtol=0)


@pytest.mark.parametrize("mechanism", sorted(_AUX_MECHANISMS))
def test_bwd_dc_pair_launch_aux_mechanisms_wrap_rings_across_tiles(
    mechanism: str,
) -> None:
    """The same mechanisms under persistent reuse: more pair tiles than CTA
    pairs on the device, at the chain seeds' deepest bk=64 region ring, so
    every ring (AB, C, the aux ring) wraps its phase across tiles."""
    _requires_sm100()
    t = _inputs((8192, 512, 512))
    args = (t["grad_out"], t["x"], t["b"], t["c"], t["d"])
    grad_cb = (t["grad_out"].float() * t["x"].float() * t["d"].float()) * (
        1.0 - t["d"].float()
    )
    grad_c_ref = (grad_cb.to(torch.float16).float() @ t["b"].float().T) * (t["c"] > 0)
    # grad_c: 8192x512x512 on the 256x128 pair: 32 x 4 = 128 pair tiles.
    tiles = [256, 128, 64]
    pair_tiles = (8192 // tiles[0]) * (512 // tiles[1])
    sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    assert pair_tiles > sm_count // 2, (pair_tiles, sm_count)
    kernel = helion.kernel(
        squeeze_and_excitation_net_bwd_dc.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    recipe: dict[str, object] = {"tcgen05_aux_load_placement": "pre_acc_wait"}
    plain = _chain_config(
        bound,
        ["persistent_blocked"],
        tiles=tiles,
        region_ab_stages=[8],
        pdl=True,
        **recipe,
    )
    staged = _chain_config(
        bound,
        ["persistent_blocked"],
        tiles=tiles,
        region_ab_stages=[8],
        pdl=True,
        **recipe,
        **_AUX_MECHANISMS[mechanism],
    )
    for key, value in _AUX_MECHANISMS[mechanism].items():
        assert staged.config[key] == value, key
    assert staged.config["tcgen05_region_ab_stages"] == [8]
    plain_out = bound.compile_config(plain, allow_print=False)(*args)
    staged_out = bound.compile_config(staged, allow_print=False)(*args)
    torch.cuda.synchronize()
    assert _rel_l2(plain_out[1], grad_c_ref) < 2e-3
    torch.testing.assert_close(staged_out, plain_out, atol=0, rtol=0)


def test_bwd_dc_128_row_pair_fragment_hoist_matches_bitwise() -> None:
    """The hoist on the chain's own 128x256 pair tile (a 64-row per-CTA
    fragment of the relu mask, 256 B per epilogue thread)."""
    _requires_sm100()
    t = _inputs()
    args = (t["grad_out"], t["x"], t["b"], t["c"], t["d"])
    kernel = helion.kernel(
        squeeze_and_excitation_net_bwd_dc.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    recipe: dict[str, object] = {"tcgen05_aux_load_placement": "pre_acc_wait"}
    plain = _chain_config(
        bound,
        ["persistent_blocked"],
        tiles=[128, 256, 64],
        region_ab_stages=[4],
        pdl=True,
        **recipe,
    )
    hoisted = _chain_config(
        bound,
        ["persistent_blocked"],
        tiles=[128, 256, 64],
        region_ab_stages=[4],
        pdl=True,
        tcgen05_aux_fragment_hoist=True,
        **recipe,
    )
    assert hoisted.config["tcgen05_aux_fragment_hoist"] is True
    assert "aux_rmem_full" in bound.to_code(hoisted)
    plain_out = bound.compile_config(plain, allow_print=False)(*args)
    hoisted_out = bound.compile_config(hoisted, allow_print=False)(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(hoisted_out, plain_out, atol=0, rtol=0)


def test_bwd_dc_flat_launch_epi_tma_matches_bitwise() -> None:
    """A GEMM launch left flat (the one-CTA pipeline) takes the
    epilogue-issued aux ring as well, priced by the one-CTA model; the
    register prefetch ring and the warp groups are refused there."""
    _requires_sm100()
    t = _inputs()
    args = (t["grad_out"], t["x"], t["b"], t["c"], t["d"])
    kernel = helion.kernel(
        squeeze_and_excitation_net_bwd_dc.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    tiles = [128, 128, 64]
    flat = _chain_config(
        bound,
        None,
        tiles=tiles,
        region_ab_stages=[],
        pdl=True,
        tcgen05_ab_stages=4,
        tcgen05_aux_load_placement="pre_acc_wait",
    )
    staged = _chain_config(
        bound,
        None,
        tiles=tiles,
        region_ab_stages=[],
        pdl=True,
        tcgen05_ab_stages=4,
        tcgen05_aux_load_placement="pre_acc_wait",
        tcgen05_aux_load_mode="epi_tma",
    )
    assert staged.config["tcgen05_aux_load_mode"] == "epi_tma"
    with pytest.raises(helion.exc.InvalidConfig, match="launch left flat"):
        _chain_config(
            bound,
            None,
            tiles=tiles,
            region_ab_stages=[],
            pdl=True,
            tcgen05_ab_stages=4,
            tcgen05_aux_load_placement="pre_acc_wait",
            tcgen05_aux_prefetch_subtiles=1,
        )
    flat_out = bound.compile_config(flat, allow_print=False)(*args)
    staged_out = bound.compile_config(staged, allow_print=False)(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(staged_out, flat_out, atol=0, rtol=0)
