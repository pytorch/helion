"""Real CUDA coverage for two-CTA persistent GEMM launches of a barrier chain.

``tcgen05_region_pid_types`` runs chosen GEMM phases of an ordered
``hl.barrier()`` chain (``squeeze_and_excitation_net_bwd_dx`` / ``_bwd_da``)
as the static persistent CtaGroup.TWO pipeline while the chain and its
pointwise phase stay flat; the exact-shape auxiliary epilogues (the
relu-backward mask, ``grad_out * d + acc``) ride along.  Each render is
checked against the fp32 torch reference and must match the flat chain's
numerics bitwise (same tiles, same K order, same epilogue arithmetic).
"""

from __future__ import annotations

from typing import cast

from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_bwd_da
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_bwd_dx
import pytest
import torch

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
        squeeze_and_excitation_net_bwd_dx.fn,
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
        squeeze_and_excitation_net_bwd_da.fn,
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


def test_chain_seeds_compile_and_match() -> None:
    """The heuristic's two-CTA chain seeds (``region_two_cta_seed_configs``)
    strict-normalize, compile and match the flat default bitwise."""
    _requires_sm100()
    t = _inputs()
    args = (t["grad_out"], t["x"], t["a"], t["b"], t["c"], t["d"])
    kernel = helion.kernel(
        squeeze_and_excitation_net_bwd_dx.fn,
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
