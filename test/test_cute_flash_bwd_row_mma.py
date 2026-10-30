"""The ``row_mma`` family of the fused flash-attention backward
(``cute_flash_bwd_row_mma``): register-MMA row programs for small grids.

One CTA per (KV group, ``kv_rows``) tile, warps arranged as ``kv_rows // 16``
row groups x ``q_split`` Q ranges; a warp owns 16 KV rows and walks its Q
range in ``q_chunk``-row chunks with warp-level ``mma.sync.m16n8k16`` (S^T /
dP^T with K / V as the A operands, dV / dK from the packed P^T / dS^T
fragments, dQ through ``movmatrix`` and one ``cp.reduce.async.bulk`` per
16-row tile).  Measured at ``(1, 4, 256, 64)`` on a B200: 5.3 us device
time against 7.4 us for the 128x128 tcgen05 family (8 CTAs, ~4.7 us of
per-CTA fixed cost) and 6.1 us for the Triton backward's 16x32 tiles.
"""

from __future__ import annotations

import math

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.autotuner.config_fragment import EnumFragment

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._compiler.cute import cute_flash_bwd_row_mma as row

FAMILY = row.FLASH_BWD_FAMILY_KEY
KV = row.FLASH_BWD_ROW_KV_KEY
SPLIT = row.FLASH_BWD_ROW_Q_SPLIT_KEY
CHUNK = row.FLASH_BWD_ROW_Q_CHUNK_KEY
KIND = row.FLASH_BWD_ROW_PLAN_KIND
RUNTIME = "_helion_flash_rowmma"
_ROW = {"block_sizes": [128, 128], FAMILY: "row_mma"}
RCP_LN2 = 1.4426950408889634


def _knobs(kv_rows: int, q_split: int, q_chunk: int) -> dict[str, object]:
    return {**_ROW, KV: kv_rows, SPLIT: q_split, CHUNK: q_chunk}


def _kernel() -> helion.Kernel:
    from examples.attention import attention_backward

    return helion.kernel(
        attention_backward.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_accuracy_check=False,
    )


def _args(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype = torch.bfloat16,
    q_seq: int | None = None,
) -> tuple[torch.Tensor | float, ...]:
    """Kernel arguments of the example backward (K pre-scaled) with random
    o / lse / delta: enough to bind and render."""
    torch.manual_seed(0)
    q_shape = (shape[0], shape[1], q_seq or shape[2], shape[3])
    q, do, o = (torch.randn(q_shape, device=DEVICE, dtype=dtype) for _ in range(3))
    k, v = (torch.randn(shape, device=DEVICE, dtype=dtype) for _ in range(2))
    lse, delta = (
        torch.randn(q_shape[:-1], device=DEVICE, dtype=torch.float32) for _ in range(2)
    )
    sm_scale = 1.0 / math.sqrt(shape[-1])
    return (q, k * (sm_scale * RCP_LN2), v, o, lse, do, delta, sm_scale)


def _inputs(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype = torch.bfloat16,
    q_seq: int | None = None,
) -> tuple[tuple[torch.Tensor | float, ...], tuple[torch.Tensor, ...]]:
    """(kernel args with K pre-scaled, float64 reference dq / dk / dv); a
    ``q_seq`` larger than the KV length is the pre-reshaped grouped-query
    form (``q_seq // seq`` query heads per KV head)."""
    torch.manual_seed(0)
    q_shape = (shape[0], shape[1], q_seq or shape[2], shape[3])
    q, do = (torch.randn(q_shape, device=DEVICE, dtype=dtype) for _ in range(2))
    k, v = (torch.randn(shape, device=DEVICE, dtype=dtype) for _ in range(2))
    sm_scale = 1.0 / math.sqrt(shape[-1])
    q64, k64, v64 = (t.double().requires_grad_() for t in (q, k, v))
    scores = q64 @ k64.transpose(-1, -2) * sm_scale
    out = scores.softmax(-1) @ v64
    out.backward(do.double())
    o = out.detach().to(q.dtype)
    lse = (torch.logsumexp(scores.detach(), -1) * RCP_LN2).float()
    delta = (o.float() * do.float()).sum(-1)
    args = (q, k * (sm_scale * RCP_LN2), v, o, lse, do, delta, sm_scale)
    grads = tuple(t.grad for t in (q64, k64, v64))
    return args, grads  # pyrefly: ignore [bad-return]


def _facts(
    head_dim: int = 64,
    mm_dim: int = 256,
    n_dim: int = 256,
    total_kv_rows: int = 1024,
    admitted: bool = True,
) -> row.FlashBwdRowFacts:
    return row.FlashBwdRowFacts(
        admitted=admitted,
        head_dim=head_dim,
        mm_dim=mm_dim,
        n_dim=n_dim,
        total_kv_rows=total_kv_rows,
    )


def _code(bound: object, **config: object) -> str:
    return bound.to_code(bound._normalized_config_copy(helion.Config(**config)))  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Legality (no device)
# --------------------------------------------------------------------------


def test_row_plan_rules() -> None:
    facts = _facts()
    # One chunk per warp: one stage; several chunks: two stages when they fit.
    assert row.flash_bwd_row_plan(facts, kv_rows=16, q_split=4, q_chunk=64) == 1
    assert row.flash_bwd_row_plan(facts, kv_rows=16, q_split=2, q_chunk=32) == 2
    assert row.flash_bwd_row_plan(facts, kv_rows=64, q_split=2, q_chunk=32) == 1
    assert row.flash_bwd_row_plan(facts, kv_rows=32, q_split=2, q_chunk=32) == 2
    # Whole chunks divide each warp's Q range, whole 16-row groups the KV rows.
    assert row.flash_bwd_row_plan(facts, kv_rows=16, q_split=4, q_chunk=128) is None
    assert row.flash_bwd_row_plan(facts, kv_rows=16, q_split=16, q_chunk=32) is None
    assert row.flash_bwd_row_plan(facts, kv_rows=48, q_split=2, q_chunk=32) is None
    assert row.flash_bwd_row_plan(facts, kv_rows=24, q_split=2, q_chunk=32) is None
    assert row.flash_bwd_row_plan(facts, kv_rows=16, q_split=2, q_chunk=24) is None
    # At most 16 warps; the staging must fit the shared-memory budget.
    assert row.flash_bwd_row_plan(facts, kv_rows=128, q_split=4, q_chunk=16) is None
    assert row.flash_bwd_row_plan(facts, kv_rows=32, q_split=8, q_chunk=16) is None
    assert row.flash_bwd_row_plan(facts, kv_rows=32, q_split=4, q_chunk=16) == 2
    # The 128-wide head: one stage is the last resort for a two-chunk walk.
    wide = _facts(head_dim=128)
    assert row.flash_bwd_row_plan(wide, kv_rows=16, q_split=4, q_chunk=32) == 1
    assert row.flash_bwd_row_plan(wide, kv_rows=16, q_split=8, q_chunk=32) is None
    # Plans the family does not admit (causal, other dtypes / head dims).
    assert (
        row.flash_bwd_row_plan(
            _facts(admitted=False), kv_rows=16, q_split=4, q_chunk=64
        )
        is None
    )
    assert row.flash_bwd_row_smem_bytes(
        head_dim=64, kv_rows=16, q_split=4, q_chunk=64, stages=1
    ) == 4 * (4096 + 16384 + 512 + 8704)
    assert row.flash_bwd_row_region_bytes(64, 1) == 8192
    assert row.flash_bwd_row_region_bytes(64, 2) == 8704


def test_row_legalize_keeps_one_identity_per_kernel() -> None:
    facts = _facts()
    # Off the family the row knobs read their defaults.
    config: dict[str, object] = {FAMILY: "fa4", KV: 64, SPLIT: 8, CHUNK: 16}
    row.flash_bwd_row_legalize(facts, config)
    assert config == {FAMILY: "fa4", KV: 16, SPLIT: 4, CHUNK: 64}
    # Absent knobs stay absent (the tcgen05 render is untouched).
    config = {"cute_flash_bwd_persistent": 0}
    row.flash_bwd_row_legalize(facts, config)
    assert config == {"cute_flash_bwd_persistent": 0}
    # On it, an illegal triple moves to the nearest legal one: fewer splits,
    # then smaller chunks, then fewer KV rows.
    config = {FAMILY: "row_mma", KV: 32, SPLIT: 8, CHUNK: 16}
    row.flash_bwd_row_legalize(facts, config)
    assert config == {FAMILY: "row_mma", KV: 32, SPLIT: 4, CHUNK: 16}
    config = {FAMILY: "row_mma", KV: 16, SPLIT: 4, CHUNK: 64}
    row.flash_bwd_row_legalize(_facts(mm_dim=128), config)
    assert config == {FAMILY: "row_mma", KV: 16, SPLIT: 2, CHUNK: 64}
    # Defaults fill in for a config that names the family alone.
    config = {FAMILY: "row_mma"}
    row.flash_bwd_row_legalize(facts, config)
    assert config == {FAMILY: "row_mma", KV: 16, SPLIT: 4, CHUNK: 64}
    # A plan the family does not admit, or no facts at all: the tcgen05
    # family (fail closed).
    config = {FAMILY: "row_mma", KV: 32, SPLIT: 2, CHUNK: 32}
    row.flash_bwd_row_legalize(_facts(admitted=False), config)
    assert config == {FAMILY: "fa4", KV: 16, SPLIT: 4, CHUNK: 64}
    config = {FAMILY: "row_mma"}
    row.flash_bwd_row_legalize(None, config)
    assert config == {FAMILY: "fa4"}
    # Nothing legal at all (a 16-row group cannot divide 8 KV rows).
    config = {FAMILY: "row_mma", KV: 16, SPLIT: 1, CHUNK: 16}
    row.flash_bwd_row_legalize(_facts(n_dim=8, total_kv_rows=8), config)
    assert config == {FAMILY: "fa4", KV: 16, SPLIT: 4, CHUNK: 64}


def test_row_search_grid_bound() -> None:
    assert row.flash_bwd_row_search_grid(_facts())
    assert row.flash_bwd_row_search_grid(_facts(total_kv_rows=128 * 64))
    assert not row.flash_bwd_row_search_grid(_facts(total_kv_rows=128 * 65))
    assert not row.flash_bwd_row_search_grid(_facts(admitted=False))
    assert row.flash_bwd_row_warps(64, 2) == 8


# --------------------------------------------------------------------------
# Surface
# --------------------------------------------------------------------------


@onlyBackends(["cute"])
def test_row_surface_is_searched_and_seeded_on_small_grids() -> None:
    bound = _kernel().bind(_args((1, 2, 256, 64)))
    spec = bound.config_spec
    assert spec.cute_flash_bwd_search_enabled
    fields = spec._flat_fields()
    family = fields[FAMILY]
    assert isinstance(family, EnumFragment)
    assert family.choices == ("fa4", "row_mma")
    assert family._active_choices() == ("fa4", "row_mma")
    assert fields[KV].choices == (16, 32, 64, 128)
    assert fields[SPLIT].choices == (4, 8, 2, 1)
    assert fields[CHUNK].choices == (64, 32, 16)
    assert fields[CHUNK]._active_choices() == (64, 32, 16)
    # The default config renders the tcgen05 family; the seed list adds the
    # row programs over the default row knobs.
    assert spec.default_config().config[FAMILY] == "fa4"
    seeds = [
        s for s in spec.autotune_seed_configs() if s.config.get(FAMILY) == "row_mma"
    ]
    assert len(seeds) == 1
    assert (seeds[0].config[KV], seeds[0].config[SPLIT], seeds[0].config[CHUNK]) == (
        16,
        4,
        64,
    )
    # Normalization legalizes the triple on the family and pins the knobs off it.
    on = bound._normalized_config_copy(helion.Config(**_knobs(32, 8, 16)))
    assert (on.config[FAMILY], on.config[KV], on.config[SPLIT], on.config[CHUNK]) == (
        "row_mma",
        32,
        4,
        16,
    )
    off = bound._normalized_config_copy(
        helion.Config(
            block_sizes=[128, 128], **{FAMILY: "fa4", KV: 64, SPLIT: 8, CHUNK: 16}
        )
    )
    assert (off.config[KV], off.config[SPLIT], off.config[CHUNK]) == (16, 4, 64)


@onlyBackends(["cute"])
def test_row_surface_stays_explicit_on_large_grids() -> None:
    # 128 KV tiles of the tcgen05 family: the knobs exist for explicit configs
    # but the search and the seeds stay on the tcgen05 family.
    bound = _kernel().bind(_args((2, 8, 1024, 64)))
    fields = bound.config_spec._flat_fields()
    assert fields[FAMILY].choices == ("fa4", "row_mma")
    assert fields[FAMILY]._active_choices() == ("fa4",)
    assert fields[KV]._active_choices() == (16,)
    assert not [
        s
        for s in bound.config_spec.autotune_seed_configs()
        if s.config.get(FAMILY) == "row_mma"
    ]
    code = _code(bound, **_knobs(64, 2, 32))
    assert f"'kind': '{KIND}'" in code
    assert "'kv_rows': 64, 'q_split': 2, 'q_chunk': 32, 'stages': 1" in code


# --------------------------------------------------------------------------
# Generated code
# --------------------------------------------------------------------------


@onlyBackends(["cute"])
def test_row_renders_the_default_program() -> None:
    bound = _kernel().bind(_args((1, 4, 256, 64)))
    code = _code(bound, **_ROW)
    assert f"'kind': '{KIND}'" in code
    assert (
        "'kv_rows': 16, 'q_split': 4, 'q_chunk': 64, 'stages': 1, 'row_warps': 4, "
        "'total_tiles': 64" in code
    )
    for name in ("q", "k", "v", "do", "lse", "delta", "dq", "dk", "dv"):
        assert f"'{name}_idx': " in code
    assert "block=(128, 1, 1)" in code
    # One row group of four Q ranges per CTA: the warp's KV rows and Q range.
    assert "rb_kv_row = rb_grp * 256 + rb_tile * 16 + rb_rg * 16" in code
    assert "rb_q_row0 = rb_grp * 256 + rb_qs * 64" in code
    # One 64-row chunk: S^T / dP^T 8 octets x 4 k-steps each, dV / dK 4
    # k-steps x 4 octet pairs x 2, dQ 4 m-tiles x 4 pairs x 2 = 160 mma.sync.
    assert code.count(f"{RUNTIME}.mma_bf16(") == 160
    assert code.count(f"{RUNTIME}.ldmatrix_x4(") == 8 + 32
    assert code.count(f"{RUNTIME}.ldmatrix_x4_trans(") == 32 + 16
    assert code.count(f"{RUNTIME}.movmatrix_trans(") == 16
    # cp.async: K 4 + V 4 + Q 16 + dO 16 packets per lane, LSE / delta guarded.
    assert code.count(f"{RUNTIME}.cp_async_16(") == 42
    assert code.count("if rb_lane < 16:") == 2
    # dQ: one bulk reduce-add per 16-row tile from the two-slot ring.
    assert code.count(f"{RUNTIME}.bulk_reduce_add_f32(") == 4
    assert code.count("cute.arch.cp_async_bulk_wait_group(1, read=True)") == 4
    assert "cute.arch.cp_async_bulk_wait_group(0, read=True)" in code
    assert code.count(f"{RUNTIME}.red_global_add") == 0
    # The softmax recompute and dS of the fused body, exact fp32.
    assert (
        "cute.arch.fma_packed_f32x2((rb_s_0_0, rb_s_0_1), (rb_scale2, rb_scale2), (rb_nl_a, rb_nl_b))"
        in code
    )
    assert "cute.arch.exp2(rb_x0)" in code
    assert (
        "cute.arch.sub_packed_f32x2((rb_dp_0_0, rb_dp_0_1), (rb_dl_a, rb_dl_b))" in code
    )
    # Four Q ranges: the partials meet after one CTA barrier; no chunk loop.
    assert "cute.arch.barrier()" in code
    assert f"{RUNTIME}.lds_v4_f32(" in code
    assert "cutlass.range(" not in code
    assert "rb_dvt" not in code
    body = code.split("_helion_cute_wrapper_plans")[0].split("def _helion_attention")[1]
    assert "tcgen05" not in body
    assert "tma" not in body.lower()


@onlyBackends(["cute"])
def test_row_renders_the_double_buffered_chunk_loop() -> None:
    bound = _kernel().bind(_args((1, 2, 512, 64)))
    code = _code(bound, **_knobs(16, 2, 32))
    assert "'kv_rows': 16, 'q_split': 2, 'q_chunk': 32, 'stages': 2" in code
    assert "block=(64, 1, 1)" in code
    # 256 Q rows per warp as eight 32-row chunks through two stages: the next
    # chunk is issued before this one's compute, the carried accumulators
    # live in register tensors across the dynamic loop.
    assert "for rb_ci in cutlass.range(1, 8):" in code
    assert "if rb_ci + 1 < 8:" in code
    assert "cute.arch.cp_async_wait_group(1)" in code
    assert "rb_dvt = cute.make_rmem_tensor((32,), cutlass.Float32)" in code
    assert "rb_dkt = cute.make_rmem_tensor((32,), cutlass.Float32)" in code
    assert "rb_sb = rb_stage_w + (rb_ci + 1) % 2 * 8192" in code
    assert "rb_slot = rb_dq_w + (rb_ci * 2 + 1) % 2 * 4096" in code


@onlyBackends(["cute"])
def test_row_renders_the_single_stage_loop_and_direct_stores() -> None:
    # The 128-wide head with four Q ranges: two 32-row chunks through one
    # stage (two do not fit the budget), refilled after each chunk's reads.
    bound = _kernel().bind(_args((1, 2, 256, 128)))
    code = _code(bound, **_knobs(16, 4, 32))
    assert "'kv_rows': 16, 'q_split': 4, 'q_chunk': 32, 'stages': 1" in code
    assert "for rb_ci in cutlass.range(1, 2):" in code
    assert "if rb_ci + 1 < 2:" not in code
    assert code.count("cute.arch.cp_async_wait_group(0)") >= 1
    # One Q range per row group: dK / dV go straight from the fragments.
    bound = _kernel().bind(_args((2, 2, 128, 64)))
    code = _code(bound, **_knobs(16, 1, 32))
    assert "'kv_rows': 16, 'q_split': 1, 'q_chunk': 32, 'stages': 2" in code
    assert f"{RUNTIME}.stg_b32(" in code
    assert "cute.arch.barrier()" not in code
    assert f"{RUNTIME}.lds_v4_f32(" not in code
    assert (
        "rb_s_part = cute.arch.alloc_smem(cutlass.Float32, 2048, alignment=128)" in code
    )


@onlyBackends(["cute"])
def test_tcgen05_render_ignores_the_row_knobs() -> None:
    bound = _kernel().bind(_args((1, 2, 256, 64)))
    plain = _code(bound, block_sizes=[128, 128])
    pinned = _code(
        bound, block_sizes=[128, 128], **{FAMILY: "fa4", KV: 64, SPLIT: 8, CHUNK: 16}
    )
    assert "'kind': 'helion_flash_bwd'" in plain
    assert plain == pinned
    assert KIND not in plain


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("shape", "dtype", "q_seq", "knobs"),
    (
        ((1, 4, 256, 64), torch.bfloat16, None, (16, 4, 64)),
        ((1, 4, 256, 64), torch.float16, None, (16, 8, 32)),
        ((1, 2, 512, 64), torch.bfloat16, None, (16, 4, 32)),
        ((1, 2, 256, 128), torch.bfloat16, None, (16, 4, 32)),
        ((2, 2, 128, 64), torch.bfloat16, None, (16, 1, 32)),
        ((1, 2, 256, 64), torch.bfloat16, 512, (16, 4, 64)),
        ((1, 1, 1024, 64), torch.bfloat16, None, (64, 2, 32)),
    ),
)
def test_row_matches_reference(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    q_seq: int | None,
    knobs: tuple[int, int, int],
) -> None:
    args, expected = _inputs(shape, dtype, q_seq)
    code, grads = code_and_output(_kernel(), args, **_knobs(*knobs))
    assert f"'kind': '{KIND}'" in code
    assert (
        f"'kv_rows': {knobs[0]}, 'q_split': {knobs[1]}, 'q_chunk': {knobs[2]}" in code
    )
    for actual, reference in zip(grads, expected, strict=True):
        torch.testing.assert_close(
            actual.double(), reference, atol=2e-2, rtol=2e-2, check_device=False
        )


@onlyBackends(["cute"])
def test_row_dk_dv_are_deterministic() -> None:
    args, expected = _inputs((1, 4, 256, 64))
    kernel = _kernel()
    bound = kernel.bind(args)
    config = bound._normalized_config_copy(helion.Config(**_knobs(16, 4, 64)))
    compiled = bound.compile_config(config)
    first = [t.clone() for t in compiled(*args)]
    second = [t.clone() for t in compiled(*args)]
    # dK / dV: fixed-order sums; dQ: order-free bulk reduce-adds (tolerance).
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    torch.testing.assert_close(first[0], second[0], atol=1e-3, rtol=1e-3)
    for actual, reference in zip(first, expected, strict=True):
        torch.testing.assert_close(
            actual.double(), reference, atol=2e-2, rtol=2e-2, check_device=False
        )
