"""Register-MMA (``mma.sync``) row programs for gated jagged attention
(``cute_flash_gated_row_mma``): the 16- and 32-row query tiles of the gated
surface.

CPU tests drive codegen with the tcgen05 support probe patched on (the row
programs need none of it, but the surface is detected with it) and compile
the device kernel through the CuTe DSL without a GPU.  GPU tests check the
numerics against an fp32 reference at jagged lengths (1, the maximum, lengths
that are not multiples of any tile, a last sequence shorter than the maximum
so the tail rows of its last tile lie past the tensor end), bitwise
determinism of the fixed-order warp combine, and the exact-division twin.
"""

from __future__ import annotations

import operator
import os
import types
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test.test_cute_flash_gated import _assert_matches
from test.test_cute_flash_gated import _bind_hstu
from test.test_cute_flash_gated import _causal_keep
from test.test_cute_flash_gated import _compile_only_launcher
from test.test_cute_flash_gated import _cpu_codegen_patches
from test.test_cute_flash_gated import _gated_markers
from test.test_cute_flash_gated import _gated_reference
from test.test_cute_flash_gated import _hstu_args
from test.test_cute_flash_gated import _hstu_module
from test.test_cute_flash_gated import _jagged_inputs
from test.test_cute_flash_gated import _with_fast_math

import helion
from helion._compiler.cute import cute_flash_gated
from helion._compiler.cute import cute_flash_gated_row_mma as row_mma
from helion._testing import onlyBackends
from helion._testing import skipUnlessBackends
from helion._testing.cute_flash import compiled_source
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = skipUnlessBackends(["cute"])

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

ROW_KIND = "'kind': 'helion_flash_gated_row_mma'"


def _row_config(q_tile: int, kv_tile: int, kv_stage: int, warps: int) -> helion.Config:
    return helion.Config(
        block_sizes=[q_tile, kv_tile],
        cute_flash_kv_stage=kv_stage,
        cute_flash_gated_row_warps=warps,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_row_split_and_budgets() -> None:
    assert row_mma.gated_row_split(16, 1) == (1, 1)
    assert row_mma.gated_row_split(16, 8) == (1, 8)
    assert row_mma.gated_row_split(32, 2) == (2, 1)
    assert row_mma.gated_row_split(32, 4) == (2, 2)
    assert row_mma.gated_row_split(32, 8) == (2, 4)
    # One 16-row m-tile at head dim 128 and a 128-key tile: 32 Q + 64 acc +
    # 64 S + 32 packed P + 8 PV partial + 48 gate registers: past the budget
    # (it compiles to 255 registers with 188 spill instructions).
    assert row_mma.gated_row_fragment_registers(128, 16, 128, 1) == 248
    assert row_mma.gated_row_warp_choices(128, 16, 128) == ()
    assert row_mma.gated_row_warp_choices(128, 32, 128) == ()
    # Head dim 128 x 64 keys (200) fits at one m-tile per warp; two m-tiles
    # per warp (32 rows, one warp) do not.
    assert row_mma.gated_row_fragment_registers(128, 16, 64, 1) == 200
    assert row_mma.gated_row_warp_choices(128, 32, 64) == (2, 4, 8)
    # The s0 class: 8 Q + 16 acc + 16 S + 8 P + 8 + 48 (eight-element passes).
    assert row_mma.gated_row_fragment_registers(32, 16, 32, 4) == 104
    assert row_mma.gated_row_warp_choices(32, 16, 16) == (1, 2, 4, 8)
    assert row_mma.gated_row_warps_default((1, 2, 4, 8)) == 2
    assert row_mma.gated_row_warps_default((4, 8)) == 8
    # Shared memory: per warp the Q rows and ``kv_stage`` (K, V) chunks of
    # 16-bit rows; the fp32 partial exchange only when warps split the keys.
    one_warp = row_mma.gated_row_smem_bytes(
        head_dim=32, q_tile=16, kv_tile=16, warps=1, kv_stage=2
    )
    assert one_warp == (16 * 32 + 2 * 2 * 16 * 32) * 2 + 1024
    four_warps = row_mma.gated_row_smem_bytes(
        head_dim=32, q_tile=16, kv_tile=16, warps=4, kv_stage=2
    )
    assert four_warps == 4 * (16 * 32 + 2 * 2 * 16 * 32) * 2 + 4 * 16 * 36 * 4 + 1024
    # A 128-key, 128-wide chunk is 64 KB per warp per stage: two warps fit
    # one stage, one warp two, four warps none (the budget is the sm_100
    # opt-in maximum).
    assert row_mma.gated_row_kv_stage_choices(
        128, 16, 128, 2, smem_capacity=232448
    ) == (1,)
    assert row_mma.gated_row_kv_stage_choices(
        128, 16, 128, 1, smem_capacity=232448
    ) == (1, 2)
    assert (
        row_mma.gated_row_kv_stage_choices(128, 16, 128, 4, smem_capacity=232448) == ()
    )
    assert row_mma.gated_row_kv_stage_choices(64, 16, 16, 8, smem_capacity=232448) == (
        1,
        2,
        4,
    )
    assert row_mma.gated_row_kv_stage_default((1, 2, 4, 8)) == 4
    assert row_mma.gated_row_kv_stage_default((1, 2)) == 2
    assert row_mma.gated_row_kv_stage_default((8,)) == 8


def _eval_mask(expr: str, n: int) -> int:
    """Evaluate a ``_col_mask`` expression with Python ints standing in for
    the DSL scalars (``~`` on a 32-bit pattern is taken modulo 2**32)."""
    fake = types.SimpleNamespace(
        Int32=int,
        Int64=int,
        min=min,
        max=max,
        select_=lambda cond, a, b: a if cond else b,
    )
    value = eval(expr, {"cutlass": fake, "n": n})
    return value & 0xFFFFFFFF


@pytest.mark.parametrize("width", [4, 8, 16, 32])
def test_paired_column_masks(width: int) -> None:
    """The ``mma.sync`` fragment rows hold columns ``8 * (e // 2) + e % 2``:
    every comparison against a bound matches the brute-force bitmask."""
    columns = cute_flash_gated.GatedChunkGeometry(width, 1, 1, paired=True)
    offsets = columns.column_offsets()
    assert offsets[:4] == [0, 1, 8, 9]
    assert columns.chunk_span == 4 * width
    compare = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
    }
    for op, fn in compare.items():
        expr = cute_flash_gated._col_mask_paired(op, "n", width)
        for n in range(-20, 4 * width + 20):
            expected = sum(1 << e for e, col in enumerate(offsets) if fn(col, n))
            got = _eval_mask(expr, n) & ((1 << width) - 1)
            assert got == expected, (op, n, width)


def test_row_launcher_refuses_an_under_aligned_schema_and_stages_o() -> None:
    """The body moves 16-byte cp.async packets of q/k/v and 4-byte output
    packets: the launcher stages an under-aligned q/k/v/o binding through a
    16-byte copy (o copied back), and a schema that still proves a base
    under-aligned for its access width is refused."""
    from helion import exc
    from helion.runtime.cute import launcher

    plan: dict[str, object] = {
        "kind": row_mma.GATED_ROW_WRAPPER_KIND,
        "head_dim": 32,
        "q_tile": 16,
        "kv_tile": 32,
        "kv_stage": 2,
        "row_warps": 4,
        "q_idx": 0,
        "k_idx": 1,
        "v_idx": 2,
        "o_idx": 3,
    }
    aligned: tuple[object, ...] = (
        "tensor",
        "torch.bfloat16",
        3,
        (214, 4, 32),
        (128, 32, 1),
    )
    schema = [aligned] * 4
    body: list[str] = []
    launcher._append_cute_wrapper_plan(body, [], plan, schema_key=tuple(schema))
    assert body == []
    # q/k/v proven 8-byte aligned are refused; o needs 4 bytes, so an 8-byte
    # o passes and a 2-byte o is refused.
    for index in range(3):
        under = list(schema)
        under[index] = (*under[index], 8)
        with pytest.raises(exc.BackendUnsupported, match="16-byte-aligned q/k/v"):
            launcher._append_cute_wrapper_plan([], [], plan, schema_key=tuple(under))
    under = list(schema)
    under[3] = (*under[3], 8)
    launcher._append_cute_wrapper_plan([], [], plan, schema_key=tuple(under))
    under[3] = (*schema[3], 2)
    with pytest.raises(exc.BackendUnsupported, match="4-byte-aligned o"):
        launcher._append_cute_wrapper_plan([], [], plan, schema_key=tuple(under))
    # Every one of q/k/v/o is a staged operand; o is copied back.
    kernel = type("Kernel", (), {"_helion_cute_wrapper_plans": [plan]})()
    assert dict(launcher._cute_staged_plan_operands(kernel)) == {
        0: False,
        1: False,
        2: False,
        3: True,
    }
    numel = 214 * 4 * 32
    operands = [torch.randn(214, 4, 32, dtype=torch.bfloat16) for _ in range(3)]
    out = torch.zeros(numel + 8, dtype=torch.bfloat16)[1 : 1 + numel].view(214, 4, 32)
    assert out.data_ptr() % 16 == 2
    staging = launcher._cute_stage_under_aligned_operands(kernel, (*operands, out))
    assert staging is not None
    launch_args, copy_backs = staging
    assert all(a is b for a, b in zip(launch_args[:3], operands, strict=True))
    staged_out = launch_args[3]
    assert isinstance(staged_out, torch.Tensor)
    assert staged_out is not out and staged_out.data_ptr() % 16 == 0
    assert copy_backs == [(out, staged_out)]


# ---------------------------------------------------------------------------
# CPU codegen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_row_tiles_take_the_row_program(head_dim: int) -> None:
    with _cpu_codegen_patches():
        bound = _bind_hstu(_hstu_args(head_dim=head_dim))
        spec = bound.config_spec
        assert spec.cute_flash_gated_search_enabled
        assert list(spec._flat_fields()) == [
            "block_sizes",
            "cute_flash_kv_stage",
            "cute_flash_gate_warpgroups",
            "cute_flash_gated_row_warps",
        ]
        code = bound.to_code(_row_config(16, 16, 2, 4))
        assert ROW_KIND in code
        assert not _gated_markers(code)
        # Four warps on one 16-row m-tile: every warp walks every fourth KV
        # chunk and the key groups combine through shared memory.
        assert "block=(128, 1, 1)" in code
        assert "grm_ks = warp_idx // 1" in code
        assert (
            "grm_nchunks = cutlass.max((flash_num_kv_active - grm_ks + cutlass.Int32(3)) // cutlass.Int32(4), cutlass.Int32(0))"
            in code
        )
        assert "cute.arch.barrier()" in code
        assert "_helion_flash_rowmma.sts_v2_f32(" in code
        assert code.count("_helion_flash_rowmma.lds_v2_f32(") == 3 * 2 * (head_dim // 8)
        # Zero-filling cp.async packets, ldmatrix operands, mma.sync.  Q rows
        # are read below the query extent, the tensor end and the jagged
        # length the store-skip term names (``flash_g3 = ends - starts``).
        assert "_helion_flash_rowmma.cp_async_16_zfill(" in code
        assert (
            "grm_q_lim = cutlass.min(cutlass.min(flash_q_extent, grm_q_rows - "
            "flash_row_base), flash_g3) - grm_row0"
        ) in code
        assert "_helion_flash_rowmma.ldmatrix_x4(" in code
        assert "_helion_flash_rowmma.ldmatrix_x4_trans(" in code
        assert "_helion_flash_rowmma.mma_bf16(" in code
        # Two chunks in flight: the ring is two slots deep and the loop waits
        # for all but the most recent group.
        assert "cute.arch.cp_async_wait_group(1)" in code
        assert "grm_stage = grm_i % cutlass.Int32(2)" in code
        assert "if grm_i + cutlass.Int32(2) < grm_nchunks:" in code
        # The gate program replays the frontend chain with its bf16 rounding
        # points on 4-wide fragment-row vectors (one pass per 16-key chunk).
        assert (
            "flash_g10 = _helion_flash_rt.gated_round_vec(flash_s_v, cutlass.BFloat16)"
            in code
        )
        assert "grm_sv = cute.make_rmem_tensor((4,), cutlass.Float32)" in code
        assert "cute.math.exp2(" in code
        assert "approx=True" in code
        # The per-chunk ``P @ V`` product is rounded to bf16 before the fp32 add.
        assert "_helion_flash_rt._cvt_rn_bf16_hi(grm_o_0_0_0)" in code
        # Rows the frontend masks at the store (the jagged length) are skipped;
        # the output is packed and stored as 4-byte words.
        assert (
            "grm_store_ok = flash_row_in_tile & flash_g5 & (flash_o_row < grm_o_rows)"
            in code
        )
        assert "_helion_flash_rowmma.stg_b32(" in code
        # No TMA / TMEM / tcgen05 and no wrapper-built kernel parameters.
        assert "_flash_qk_mma" not in code
        assert (
            "tcgen05"
            not in code.split("def _helion__helion_jagged_attention_kernel")[1].split(
                "def _helion_jagged_attention_kernel"
            )[0]
        )

        # A single warp: no combine; 32 rows over two warps: one m-tile each.
        one = bound.to_code(_row_config(16, 64, 1, 1))
        assert ROW_KIND in one
        assert "block=(32, 1, 1)" in one
        assert "cute.arch.barrier()" not in one
        assert "grm_s_part" not in one
        assert "grm_stage = cutlass.Int32(0)" in one
        # A 64-key chunk: the gate runs two 8-element passes per row half,
        # each packing its P values into the A-fragments of its two k-steps.
        assert "grm_sv = cute.make_rmem_tensor((8,), cutlass.Float32)" in one
        assert (
            one.count("flash_chunk_base = flash_kv_col0 + cutlass.Int32(32) + grm_c")
            == 2
        )
        assert (
            "grm_pa_0_2_0 = _helion_flash_rowmma.pack_bf16x2(grm_pv[0], grm_pv[1])"
            in one
        )
        assert (
            "grm_pa_0_3_1 = _helion_flash_rowmma.pack_bf16x2(grm_pv[4], grm_pv[5])"
            in one
        )
        rows32 = bound.to_code(_row_config(32, 32, 4, 2))
        assert ROW_KIND in rows32
        assert "block=(64, 1, 1)" in rows32
        assert "grm_ms = warp_idx % 2" in rows32
        assert "grm_ks = cutlass.Int32(0)" in rows32
        assert "cute.arch.barrier()" not in rows32
        assert "grm_row0 = flash_q_row0 + grm_ms * cutlass.Int32(16)" in rows32
        # Eight warps on 32 rows: two row groups of four key groups.
        rows32_8 = bound.to_code(_row_config(32, 16, 2, 8))
        assert "grm_ms = warp_idx % 2" in rows32_8
        assert "grm_ks = warp_idx // 2" in rows32_8
        assert "grm_part_p = grm_part + (grm_ms + cutlass.Int32(6)) * " in rows32_8


def test_row_knobs_are_validated_and_repaired() -> None:
    with _cpu_codegen_patches():
        bound = _bind_hstu(_hstu_args(head_dim=128))
        spec = bound.config_spec
        with pytest.raises(InvalidConfig, match="cute_flash_gated_row_warps"):
            spec.normalize(_row_config(16, 16, 2, 3))
        # One warp cannot hold two 128-wide m-tiles at head dim 128 (the
        # 128-key tile does not exist there at all: an unlisted tile is not
        # the gated surface and normalizes as the scalar path).
        with pytest.raises(InvalidConfig, match="cute_flash_gated_row_warps"):
            spec.normalize(_row_config(32, 64, 1, 1))
        spec.normalize(_row_config(32, 128, 1, 1))
        assert (32, 128) not in spec._cute_flash_gated_row_warp_choices_by_tile
        # 64-key, 128-wide chunks (32 KB per warp per stage): eight warps fit
        # no ring slot, two warps fit two, four warps one.
        with pytest.raises(InvalidConfig, match="cute_flash_gated_row_warps"):
            spec.normalize(_row_config(16, 64, 1, 8))
        with pytest.raises(InvalidConfig, match="cute_flash_kv_stage"):
            spec.normalize(_row_config(16, 64, 2, 4))
        spec.normalize(_row_config(16, 64, 1, 4))
        spec.normalize(_row_config(16, 64, 2, 2))
        # Defaults fill in: two warps, the deepest ring at or below four slots.
        filled = spec.normalized_config(helion.Config(block_sizes=[16, 16])).config
        assert filled["cute_flash_gated_row_warps"] == 2
        assert filled["cute_flash_kv_stage"] == 4
        # The gate warpgroup count has no consumer at a row tile and the warp
        # count none at a fused tile: both read one value whatever is asked
        # (one identity, no duplicate configs in the search).
        fused = [
            spec.normalized_config(
                helion.Config(
                    block_sizes=[128, 128],
                    cute_flash_kv_stage=3,
                    cute_flash_gated_row_warps=warps,
                )
            ).config
            for warps in (1, 8)
        ]
        assert fused[0] == fused[1]
        assert fused[0]["cute_flash_gated_row_warps"] == 2
        row = [
            spec.normalized_config(
                helion.Config(
                    block_sizes=[16, 32],
                    cute_flash_kv_stage=2,
                    cute_flash_gated_row_warps=4,
                    cute_flash_gate_warpgroups=warpgroups,
                )
            ).config
            for warpgroups in (1, 4)
        ]
        assert row[0] == row[1]
        # Autotune (repair) mode snaps an illegal (query, KV) pair to the
        # nearest legal KV tile of the same query tile and a stage the warp
        # count admits.
        repaired: dict[str, object] = {
            "block_sizes": [64, 16],
            "cute_flash_kv_stage": 2,
            "cute_flash_gate_warpgroups": 2,
        }
        spec.normalize(repaired, _fix_invalid=True)
        assert repaired["block_sizes"] == [64, 64]
        # The 128-key tile is past the register budget at head dim 128: the
        # request moves to the nearest legal KV tile of the same query tile,
        # the default warps and the deepest ring they admit.
        repaired = {
            "block_sizes": [16, 128],
            "cute_flash_kv_stage": 8,
            "cute_flash_gated_row_warps": 8,
        }
        spec.normalize(repaired, _fix_invalid=True)
        assert repaired["block_sizes"] == [16, 64]
        assert repaired["cute_flash_gated_row_warps"] == 2
        assert repaired["cute_flash_kv_stage"] == 2
        assert row_mma.GATED_ROW_WARPS_KEY in spec._flat_fields()


def _small_hstu_args(head_dim: int) -> tuple[Any, ...]:
    """s0-class arguments: 2 sequences x 4 heads x 8 query tiles of 16 rows =
    64 row programs (the seeds are capped at ``GATED_ROW_SEED_MAX_PROGRAMS``)."""
    return _hstu_args(batch=2, seq=128, heads=4, head_dim=head_dim, total_rows=214)


@pytest.mark.parametrize("head_dim", [32, 128])
def test_row_seed_configs(head_dim: int) -> None:
    with _cpu_codegen_patches():
        # The default test shape (8 sequences x 8 heads x 32 tiles = 2048 row
        # programs) is past the seed cap: the row tiles stay search choices
        # (their knob is a flat field) without a seed.
        large = _bind_hstu(_hstu_args(head_dim=head_dim))
        assert row_mma.GATED_ROW_WARPS_KEY in large.config_spec._flat_fields()
        assert large.config_spec._cute_flash_gated_row_seed_programs == 2048
        assert not [
            seed
            for seed in large.config_spec.autotune_seed_configs()
            if seed.config["block_sizes"][0] < 64
        ]
        bound = _bind_hstu(_small_hstu_args(head_dim))
        assert bound.config_spec._cute_flash_gated_row_seed_programs == 64
        seeds = [
            seed.config
            for seed in bound.config_spec.autotune_seed_configs()
            if seed.config["block_sizes"][0] < 64
        ]
    # The 16-row tiles at four warps with two chunks in flight (the measured
    # small-shape winner), at two warps and at one; the 32-row tiles at four.
    # At head dim 128 the 128-key tile holds 64 KB per warp per stage: it
    # admits one or two warps and one stage at two, so its seeds collapse to
    # two.
    assert all("cute_flash_gated_row_warps" in seed for seed in seeds)
    seeded = {
        (
            *seed["block_sizes"],
            seed["cute_flash_gated_row_warps"],
            seed["cute_flash_kv_stage"],
        )
        for seed in seeds
    }
    if head_dim == 32:
        assert len(seeds) == 16
        assert seeded == {
            *((16, kv, 4, 2) for kv in (16, 32, 64, 128)),
            *((16, kv, 2, 4) for kv in (16, 32, 64, 128)),
            *((16, kv, 1, 4) for kv in (16, 32, 64, 128)),
            *((32, kv, 4, 2) for kv in (16, 32, 64, 128)),
        }
    else:
        # No 128-key tile at head dim 128 (register budget); 32 KB per warp
        # per stage at 64 keys: the 4-warp seeds there take the one ring slot
        # that fits.
        assert len(seeds) == 12
        assert {
            (16, 16, 4, 2),
            (16, 32, 4, 2),
            (16, 64, 4, 1),
            (32, 64, 4, 1),
        } <= seeded
        assert not [s for s in seeded if s[1] == 128]
        assert all(
            stage
            in row_mma.gated_row_kv_stage_choices(
                head_dim, q_tile, kv_tile, warps, smem_capacity=232448
            )
            for q_tile, kv_tile, warps, stage in seeded
        )
    # Every seed is legal for its own tile (the normalizer keeps it).
    with _cpu_codegen_patches():
        spec = _bind_hstu(_small_hstu_args(head_dim)).config_spec
        for seed in seeds:
            assert spec.normalized_config(helion.Config(**seed)).config == {
                **spec.normalized_config(helion.Config(**seed)).config,
                **seed,
            }


# The configurations the numerics test below launches are compiled there; this
# keeps the one row program (two m-tiles and a two-way key split behind the
# ring at head dim 64) that no numerics case runs.
@pytest.mark.parametrize(
    ("head_dim", "q_tile", "kv_tile", "kv_stage", "warps", "fast_math"),
    [(64, 32, 64, 2, 4, True)],
)
def test_row_kernel_compiles_without_gpu(
    head_dim: int,
    q_tile: int,
    kv_tile: int,
    kv_stage: int,
    warps: int,
    fast_math: bool,
    tmp_path: Path,
) -> None:
    """Compile the generated host wrapper and device kernel through the CuTe DSL."""
    env_patch = (
        {}
        if torch.cuda.is_available() or "CUTE_DSL_ARCH" in os.environ
        else {"CUTE_DSL_ARCH": "sm_100a"}
    )
    args = _hstu_args(head_dim=head_dim)
    kernel = _with_fast_math(_hstu_module()._helion_jagged_attention_kernel, fast_math)
    with _cpu_codegen_patches(), patch.dict(os.environ, env_patch):
        kernel.reset()
        bound = kernel.bind(args)
        code = bound.to_code(_row_config(q_tile, kv_tile, kv_stage, warps))
        assert ROW_KIND in code
        assert ("approx=True" in code) is fast_math
        source_path = tmp_path / "gated_row_hstu.py"
        source_path.write_text(code)
        namespace: dict[str, object] = {}
        exec(compile(code, str(source_path), "exec"), namespace)
        host_fn = namespace["_helion_jagged_attention_kernel"]
        assert callable(host_fn)
        host_fn(*args, _launcher=_compile_only_launcher)


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------


def _run_row(
    kernel: Any, args: tuple[Any, ...], config: helion.Config
) -> tuple[torch.Tensor, str]:
    """Compile and launch ``config``; returns the output and the compiled source
    (read back from the compiled module rather than rendered a second time)."""
    kernel.reset()
    bound = kernel.bind(args)
    code = compiled_source(bound, config)
    assert ROW_KIND in code
    out = bound(*args)
    torch.cuda.synchronize()
    return out, code


# The head-dim 64/128 row programs at 64- and 128-wide KV tiles take about a
# minute to generate and compile (CPU-bound codegen over the unrolled body),
# past the default per-test CI timeout under xdist contention.
@pytest.mark.timeout(300)
# pyrefly: ignore [bad-argument-type]
@onlyBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# Every row is a distinct kernel: the head dim sets the fragment counts; the
# query tile one or two m-tiles per warp (``grm_ms``); the KV tile the 4- or
# 8-element gate passes; ``kv_stage`` no ring (1) or a two- or four-slot
# cp.async ring; ``warps`` a single warp (no combine), a key split with the
# shared-memory combine (``grm_ks``), or both splits at once. ``fast_math``
# swaps the one gate-division line, so a single exact-division row covers it.
@pytest.mark.parametrize(
    ("head_dim", "q_tile", "kv_tile", "kv_stage", "warps", "fast_math"),
    [
        (32, 16, 16, 2, 4, True),
        (32, 16, 32, 4, 2, False),
        (64, 16, 64, 2, 1, True),
        (64, 16, 16, 4, 8, True),
        (64, 32, 32, 4, 2, True),
        (64, 32, 16, 2, 8, True),
        (64, 16, 128, 1, 1, True),
        (128, 16, 64, 2, 2, True),
        (128, 32, 64, 1, 4, True),
    ],
)
def test_row_program_matches_reference_at_jagged_lengths(
    head_dim: int,
    q_tile: int,
    kv_tile: int,
    kv_stage: int,
    warps: int,
    fast_math: bool,
) -> None:
    """Lengths 300 (the maximum, not a tile multiple: its last tile's tail
    rows alias the next sequence), 173, an empty sequence, 261, 5 and a last
    sequence of one row, shorter than the maximum, so the tail rows of its
    last tile lie past the tensor end (zero-filled loads, no store)."""
    torch.manual_seed(2026)
    max_seq_len = 300
    q, k, v, seq_offsets = _jagged_inputs([300, 173, 0, 261, 5, 1], 2, head_dim)
    args = (max_seq_len, 0.05, q, k, v, seq_offsets)
    expected = _gated_reference(
        q,
        k,
        v,
        seq_offsets,
        alpha=0.05,
        max_seq_len=max_seq_len,
        keep=_causal_keep,
        kv_end="tile_end",
    )
    kernel = _with_fast_math(_hstu_module()._helion_jagged_attention_kernel, fast_math)
    config = _row_config(q_tile, kv_tile, kv_stage, warps)
    out, code = _run_row(kernel, args, config)
    assert ("approx=True" in code) is fast_math
    _assert_matches(out, expected, 2e-2)
    # The warp combine runs in fixed order: repeated launches of the compiled
    # kernel (``bind`` returns the bound kernel ``_run_row`` compiled) are
    # bit-identical.
    again = kernel.bind(args)(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, again)
