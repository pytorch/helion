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

import ast
import math
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.pdl_release import ZERO_FILL_KERNEL_PREFIX
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test.cute_population_contracts import _target
from test.test_cute_pdl import _assert_only_the_dependent_launch_differs
from test.test_cute_pdl import _last_wave_guard
from test.test_cute_pdl import _on_and_off

from helion._compiler.cute import cute_flash_bwd_row_mma as row
from helion._testing import import_example
from helion._testing.cute_codegen import mock_cuda_unavailable

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    attention_backward = import_example("attention").attention_backward

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
    # row programs over the default row knobs, plain and as a programmatic
    # dependent (``cute_pdl``).
    assert spec.default_config().config[FAMILY] == "fa4"
    seeds = [
        s for s in spec.autotune_seed_configs() if s.config.get(FAMILY) == "row_mma"
    ]
    assert [s.config.get("cute_pdl") for s in seeds] == [False, True]
    for seed in seeds:
        assert (seed.config[KV], seed.config[SPLIT], seed.config[CHUNK]) == (
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


# --------------------------------------------------------------------------
# Dependent launch (cute_pdl)
# --------------------------------------------------------------------------


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with (
        mock_cuda_unavailable(),
        _target(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _cpu_args(shape: tuple[int, int, int, int]) -> tuple[torch.Tensor | float, ...]:
    """CPU tensors of the example backward's arguments: enough to bind."""
    torch.manual_seed(0)
    q, k, v, o, do = (torch.randn(shape, dtype=torch.bfloat16) for _ in range(5))
    lse, delta = (torch.randn(shape[:-1], dtype=torch.float32) for _ in range(2))
    sm_scale = 1.0 / math.sqrt(shape[-1])
    return (q, k * (sm_scale * RCP_LN2), v, o, lse, do, delta, sm_scale)


# (config, threads per CTA) of the two dense families at (1, 4, 256, 64).
_FAMILY_CONFIGS = (
    (_knobs(16, 4, 64), 128),
    ({"block_sizes": [128, 128]}, 512),
)
_WAIT = "cute.arch.griddepcontrol_wait()"
_RELEASE = "cute.arch.griddepcontrol_launch_dependents()"
_MAIN = "_helion_attention_backward"
_FILL = f"{ZERO_FILL_KERNEL_PREFIX}0_helion_attention_backward"
_ZEROS = "dq = torch.zeros((q.size(0), head_dim), device=q.device, dtype=torch.float32)"
_EMPTY = "dq = torch.empty((q.size(0), head_dim), device=q.device, dtype=torch.float32)"


def _kernel_def(code: str, name: str) -> ast.FunctionDef:
    (kernel,) = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
    ]
    return kernel


def _host_lines(code: str) -> list[str]:
    (host,) = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == "attention_backward"
    ]
    return [ast.unparse(statement) for statement in host.body]


def _assert_fill_launch(on: str, off: str) -> None:
    """The dq fill's form, spelled out: ``_assert_only_the_dependent_launch_differs``
    compares the whole knob-on module against the knob-off one (the kernel
    proper's wait, release and launch attribute, the fill kernel, its
    attribute and its launch stripped, the ``empty`` read as ``zeros``
    again), so everything else is known equal; what it does not pin down is
    the fill itself: which tensor (dq: the accumulator the memset node sat
    on between the preprocess kernel and this one), that its kernel waits
    and releases at entry, and that its launch follows the allocation and
    precedes the kernel's own launch."""
    fill = [ast.unparse(s) for s in _kernel_def(on, _FILL).body]
    assert fill[:2] == [_WAIT, _RELEASE]
    assert f"{_FILL}._helion_cute_use_pdl = True" in on
    assert _ZEROS in _host_lines(off)
    lines = _host_lines(on)
    allocation = lines.index(_EMPTY)
    assert lines[allocation + 1].startswith(
        f"_launcher({_FILL}, (max(1, min((dq.numel() + 1023) // 1024, "
    )
    assert lines[allocation + 1].endswith("),), dq.view(-1), block=(256, 1, 1))")
    (launch,) = [
        i for i, line in enumerate(lines) if line.startswith(f"_launcher({_MAIN},")
    ]
    assert allocation + 1 < launch


def test_the_surface_offers_and_seeds_the_dependent_launch(cpu_only: None) -> None:
    bound = _kernel()._bind_isolated(_cpu_args((1, 4, 256, 64)))
    spec = bound.config_spec
    assert spec.cute_flash_bwd_search_enabled
    fields = spec._flat_fields()
    pdl = fields["cute_pdl"]
    assert isinstance(pdl, EnumFragment)
    assert pdl.choices == (False, True)
    assert pdl._active_choices() == (False, True)
    release = fields["cute_pdl_release"]
    assert isinstance(release, EnumFragment)
    assert release._active_choices() == ("auto",)
    # A small grid seeds the row family twice: plain and as a dependent.
    seeds = [
        s for s in spec.autotune_seed_configs() if s.config.get(FAMILY) == "row_mma"
    ]
    assert [s.config.get("cute_pdl") for s in seeds] == [False, True]
    assert {(s.config[KV], s.config[SPLIT], s.config[CHUNK]) for s in seeds} == {
        (16, 4, 64)
    }
    # The knob survives the surface's flat round trip (a searched key the
    # normalizer dropped would be benchmarked as its knob-off kernel).
    with bound.env:
        generation = ConfigGeneration(spec)
        flat = generation.flatten(seeds[1])
        normalized = generation.unflatten(flat)
        assert normalized.config.get("cute_pdl") is True
        assert generation.flatten(normalized) == flat


def test_the_spread_pad_stays_off_both_families_here(cpu_only: None) -> None:
    # The surface offers ``cute_pdl_spread`` with the knob (the row family
    # allocates literally and can take the pad on a smaller tile), but at
    # (1, 4, 256, 64) neither family renders it: the row kernel's own
    # 118784 bytes already hold one 128-thread CTA per SM (past the 115713
    # byte footprint of one wave), and the tcgen05 kernel reserves its
    # storage through the DSL's ``SmemAllocator`` at the per-block maximum,
    # which the pad's literal sum cannot see and must not pad on top of (a
    # pad there failed every launch).  Knob on with the spread renders the
    # knob-on module.
    bound = _kernel()._bind_isolated(_cpu_args((1, 4, 256, 64)))
    spec = bound.config_spec
    spread = spec._flat_fields()["cute_pdl_spread"]
    assert isinstance(spread, EnumFragment)
    assert spread._active_choices() == (False, True)
    for config, _threads in _FAMILY_CONFIGS:
        on = bound.to_code(
            spec.normalized_config(helion.Config(**config, cute_pdl=True))
        )
        padded = bound.to_code(
            spec.normalized_config(
                helion.Config(**config, cute_pdl=True, cute_pdl_spread=True)
            )
        )
        assert "pdl_spread_pad" not in padded
        assert padded == on
    assert "smem.allocate(_fbwd_storage_cls)" in bound.to_code(
        spec.normalized_config(helion.Config(**_FAMILY_CONFIGS[1][0], cute_pdl=True))
    )
    assert "rb_s_stage = cute.arch.alloc_smem(cutlass.BFloat16, 32768" in bound.to_code(
        spec.normalized_config(helion.Config(**_FAMILY_CONFIGS[0][0], cute_pdl=True))
    )


def test_both_families_render_the_generic_dependent_launch(cpu_only: None) -> None:
    bound = _kernel()._bind_isolated(_cpu_args((1, 4, 256, 64)))
    for config, threads in _FAMILY_CONFIGS:
        on, off = _on_and_off(bound, config)
        # Whole module: only the dependent launch and the dq fill differ.
        # The wait is the body's first statement, the release follows it
        # (one wave: 64 CTAs of 128 threads for the rows, 8 of 512 for the
        # tcgen05 kernel), the launch attribute is set, the dq fill launches
        # as a dependent kernel.
        assert _assert_only_the_dependent_launch_differs(on, off) == []
        assert _last_wave_guard(on) is None
        _assert_fill_launch(on, off)
        assert f"block=({threads}, 1, 1)" in on


def test_the_row_launch_counts_its_own_ctas(cpu_only: None) -> None:
    # The launcher sizes the row kernel's grid from its plan (64 CTAs of one
    # 16-row KV tile each), not from the 128-row program-id grid (8 CTAs):
    # on a device of two SMs (32 resident 128-thread CTAs) the release comes
    # from the last wave, guarded against the plan's count.
    with patch("helion.runtime.get_num_sm", return_value=2):
        bound = _kernel()._bind_isolated(_cpu_args((1, 4, 256, 64)))
        on, off = _on_and_off(bound, _knobs(16, 4, 64))
    assert "'total_tiles': 64" in on
    # (The fill kernel's launch is capped at four CTAs per SM: eight here.)
    assert "(dq.numel() + 1023) // 1024, 8)" in on
    assert _assert_only_the_dependent_launch_differs(on, off) == []
    assert _last_wave_guard(on) == "if pdl_cta >= 32:"
    _assert_fill_launch(on, off)


@onlyBackends(["cute"])
@pytest.mark.parametrize("config", [config for config, _ in _FAMILY_CONFIGS])
def test_dependent_launch_matches_reference(config: dict[str, object]) -> None:
    args, expected = _inputs((1, 4, 256, 64))
    code, grads = code_and_output(_kernel(), args, **config, cute_pdl=True)
    assert "cute.arch.griddepcontrol_wait()" in code
    assert "cute.arch.griddepcontrol_launch_dependents()" in code
    assert f"{_MAIN}._helion_cute_use_pdl = True" in code
    # The dq accumulator's zero fill launches as a dependent fill kernel.
    assert f"{_FILL}._helion_cute_use_pdl = True" in code
    assert _EMPTY in code and _ZEROS not in code
    for actual, reference in zip(grads, expected, strict=True):
        torch.testing.assert_close(
            actual.double(), reference, atol=2e-2, rtol=2e-2, check_device=False
        )
