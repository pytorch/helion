"""``cute_flash_tmem_cols``: the tensor-memory allocation of the one-warpgroup
single-tile flash body, and that body's load order.

Every tcgen05 flash body allocates all 512 tensor-memory columns of its
SM, so an SM runs one CTA at a time. The one-warpgroup single-tile body (``ws_overlap``
with one score stage) needs one 128-column score tile, which the P tile is
stored over, and a ``head_dim``-column output accumulator: with the knob at
256 it allocates that half and gives up its allocation permit as soon as it
has its columns (with 512 it keeps the permit to its last MMA), so a second
CTA can run on the SM beside it where the SM's shared memory holds both.
That CTA's columns start at 256, so the P operand of its PV MMA is rebuilt
over the score tile's pointer.

Warp 0 of the body also issues Q and every prologue K/V load before it waits
for Q, whatever the allocation. Everything but the ``cuda``-marked tests
renders on the CPU.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute import flash_knobs
from helion._compiler.cute.attention_plan import TENSOR_BIAS_KIND
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._compiler.cute.autotune import config_generation_ext
from helion._compiler.cute.autotune.config_generation_ext import (
    cute_generation_extension,
)
from helion._compiler.cute.autotune.flash_surface import enable_cute_flash_search
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing.cute_flash import compile_and_run
from helion._testing.cute_flash import shared_flash_generation
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import AutotuneError
from helion.exc import InvalidConfig
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction
    from helion.autotuner.config_fragment import ConfigSpecFragment

pytestmark = pytest.mark.cpu_only

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

KEY = cute_flash.FLASH_TMEM_COLS_KEY
FAMILY_KEY = cute_flash.FLASH_PIPELINE_FAMILY_KEY
S_STAGE_KEY = cute_flash.FLASH_S_STAGE_KEY
# The one-warpgroup single-tile body: the knob's dependency witness.
ONE_WARPGROUP: dict[str, object] = {FAMILY_KEY: "ws_overlap", S_STAGE_KEY: 1}
ALLOCATE = "flash_tmem.allocate({})"
WAIT = "flash_tmem.wait_for_alloc()"
RELEASE = "flash_tmem.relinquish_alloc_permit()"
RETRIEVE = "flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)"
Q_WAIT = "flash_q_full = flash_q_cons.wait_and_advance()"


def _plan_facts(small_biased_candidate: bool) -> dict[str, object]:
    """The surface facts of the plain dense plan, or of a small biased plan
    as the detector reports it: one tensor bias at the standard scales, so a
    score modifier and no standard dense output."""
    if not small_biased_candidate:
        return {"standard_dense_output": True}
    return {
        "small_biased_candidate": True,
        "plain_row_body": False,
        "has_score_modifiers": True,
        "score_modifier_kinds": (TENSOR_BIAS_KIND,),
    }


def _resolve(
    head_dim: int,
    num_kv: int,
    config: dict[str, object],
    *,
    dtype: torch.dtype = torch.bfloat16,
    num_bh: int = 192,
    small_biased_candidate: bool = False,
) -> cute_flash.FlashAttentionConfig:
    return cute_flash.resolve_flash_config(
        head_dim,
        num_kv,
        config,
        dtype=dtype,
        num_bh=num_bh,
        **_plan_facts(small_biased_candidate),
    )


def _fragment(
    head_dim: int,
    num_kv: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    **surface: object,
) -> ConfigSpecFragment:
    return cute_flash.flash_autotune_fragments(
        head_dim,
        num_kv,
        dtype=dtype,
        num_bh=64,
        standard_dense_output=True,
        **surface,  # pyrefly: ignore[bad-argument-type]
    )[KEY]


def _active(fragment: ConfigSpecFragment) -> tuple[object, ...]:
    choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
    search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
    return choices if search is None else tuple(search)


def _render(
    cfg: cute_flash.FlashAttentionConfig,
    *,
    head_dim: int = 64,
    num_kv: int = 1,
    io_dtype: str = "cutlass.BFloat16",
) -> str:
    body = cute_flash.emit_flash_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=num_kv,
        cfg=cfg,
        has_lse=False,
        io_dtype=io_dtype,
        score_plan=dense_score_plan(head_dim),
    )
    return "\n".join(ast.unparse(stmt) for stmt in body)


# --------------------------------------------------------------------------
# schema, legality and resolver
# --------------------------------------------------------------------------


def test_knob_is_an_appended_schema_axis() -> None:
    keys = flash_knobs.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(KEY) == 1
    assert keys.index(KEY) > keys.index(cute_flash.FLASH_WS_K_STAGE_EXTRA_KEY)
    assert flash_knobs.FLASH_EFFECTIVE_CONFIG_KEYS.count(KEY) == 1
    assert flash_knobs.FLASH_TMEM_COLS_CHOICES == (512, 256)
    # A config saved before the knob existed keeps every column.
    assert cute_flash.FlashAttentionConfig.__dataclass_fields__[
        "tmem_cols"
    ].default == (cute_flash.FLASH_TMEM_COLUMNS)
    assert _resolve(64, 1, {}).tmem_cols == 512
    # The design reaches the 256-column value on the body that honors it.
    assert flash_knobs.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, 256)] == {
        **ONE_WARPGROUP,
        cute_flash.FLASH_SMALL_BIASED_KEY: False,
    }
    assert (KEY, 512) not in flash_knobs.FLASH_AUTOTUNE_VALUE_DEPENDENCIES


def test_legality_is_the_one_warpgroup_body_with_room_for_its_tiles() -> None:
    legal = cute_flash._flash_tmem_cols_legal
    one_warpgroup = {"topology": "ws_overlap", "s_stage": 1, "small_biased": False}
    # All the columns are legal on every body.
    assert legal(512, topology="fa4", s_stage=2, small_biased=False, head_dim=128)
    assert legal(512, topology="ws_overlap", s_stage=2, small_biased=True, head_dim=256)
    # 256 columns hold the 128-column score tile and the output accumulator
    # up to head dim 128 (an exact fit).
    assert legal(256, **one_warpgroup, head_dim=64)
    assert legal(256, **one_warpgroup, head_dim=128)
    assert not legal(256, **one_warpgroup, head_dim=160)
    assert not legal(256, **one_warpgroup, head_dim=256)
    # The two-warpgroup body buffers two score tiles, the pair bodies hold
    # two query tiles, and the small biased plan's SIMT body allocates none.
    assert not legal(256, **{**one_warpgroup, "s_stage": 2}, head_dim=64)
    assert not legal(256, **{**one_warpgroup, "topology": "fa4"}, head_dim=64)
    assert not legal(256, **{**one_warpgroup, "small_biased": True}, head_dim=64)
    # No other column count has a body.
    assert not legal(128, **one_warpgroup, head_dim=64)
    assert not legal(384, **one_warpgroup, head_dim=64)


@pytest.mark.parametrize(
    ("head_dim", "num_kv", "dtype"),
    (
        (64, 1, torch.bfloat16),
        (128, 1, torch.float16),
        (64, 3, torch.bfloat16),
        (64, 1, torch.float8_e4m3fn),
    ),
    ids=("d64_one_tile", "d128_one_tile", "d64_three_tiles", "fp8_one_tile"),
)
def test_resolver_honors_256_columns_on_the_one_warpgroup_body(
    head_dim: int, num_kv: int, dtype: torch.dtype
) -> None:
    cfg = _resolve(head_dim, num_kv, {**ONE_WARPGROUP, KEY: 256}, dtype=dtype)
    assert cfg.pipeline_family == "ws_overlap" and cfg.s_stage == 1
    assert cfg.tmem_cols == 256
    assert cute_flash.flash_effective_config_values(cfg)[KEY] == 256


@pytest.mark.parametrize(
    ("head_dim", "num_kv", "config"),
    (
        (64, 3, {FAMILY_KEY: "ws_overlap", S_STAGE_KEY: 2}),
        (64, 16, {FAMILY_KEY: "fa4"}),
        (64, 16, {FAMILY_KEY: "fa4_2cta"}),
        (256, 1, dict(ONE_WARPGROUP)),
    ),
    ids=("two_warpgroups", "fa4_pair", "two_cta", "d256_accumulator"),
)
def test_resolver_keeps_all_columns_off_that_body(
    head_dim: int, num_kv: int, config: dict[str, object]
) -> None:
    # One identity per body that cannot run in 256 columns: the request
    # normalizes to the config without it.
    cfg = _resolve(head_dim, num_kv, {**config, KEY: 256})
    assert cfg.tmem_cols == 512
    assert cute_flash.flash_effective_config_values(cfg)[KEY] == 512
    assert cfg == _resolve(head_dim, num_kv, config)


def test_resolver_rejects_an_unknown_column_count() -> None:
    with pytest.raises(InvalidConfig, match=f"{KEY} must be one of"):
        _resolve(64, 1, {**ONE_WARPGROUP, KEY: 384})


# --------------------------------------------------------------------------
# search surface and seeds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head_dim", "num_kv", "surface"),
    (
        (64, 1, {}),
        (128, 1, {}),
        (64, 3, {}),
        (64, 48, {"requires_ws_overlap": True}),
        (64, 48, {"pipeline_family_override": "ws_overlap"}),
        (64, 1, {"dtype": torch.float8_e4m3fn}),
    ),
    ids=(
        "d64_one_tile",
        "d128_one_tile",
        "d64_three_tiles",
        "required_single_tile",
        "pinned_single_tile",
        "fp8_one_tile",
    ),
)
def test_single_tile_surfaces_search_both_allocations(
    head_dim: int, num_kv: int, surface: dict[str, object]
) -> None:
    fragment = _fragment(head_dim, num_kv, **surface)
    assert tuple(fragment.choices) == (512, 256)  # pyrefly: ignore[missing-attribute]
    assert _active(fragment) == (512, 256)
    assert fragment.default() == 512


@pytest.mark.parametrize(
    ("head_dim", "num_kv", "surface"),
    (
        (64, 2, {}),
        (64, 16, {}),
        (64, 2, {"dtype": torch.float8_e4m3fn}),
        (192, 3, {}),
        (256, 1, {}),
    ),
    ids=("mixed_two_tiles", "mixed_16_tiles", "fp8_pair", "d192", "d256"),
)
def test_other_surfaces_keep_all_columns(
    head_dim: int, num_kv: int, surface: dict[str, object]
) -> None:
    # A surface with a pair family keeps its size (the pair bodies need all
    # the columns), and a head dim whose accumulator does not fit beside the
    # score tile has no 256-column body.
    fragment = _fragment(head_dim, num_kv, **surface)
    assert _active(fragment) == (512,)
    assert fragment.default() == 512


def _seeds(
    num_kv: int,
    num_bh: int,
    *,
    device_sm_count: int = 148,
    dtype: torch.dtype = torch.bfloat16,
    small_biased_candidate: bool = False,
) -> tuple[helion.Config, ...]:
    return cute_flash.flash_attention_seed_configs(
        64,
        num_kv,
        dtype=dtype,
        num_bh=num_bh,
        device_sm_count=device_sm_count,
        **_plan_facts(small_biased_candidate),
    )


@pytest.mark.parametrize(
    ("num_kv", "num_bh", "dtype", "small_biased_candidate"),
    (
        (1, 192, torch.bfloat16, False),
        (3, 64, torch.bfloat16, False),
        (1, 192, torch.float8_e4m3fn, False),
        (1, 192, torch.bfloat16, True),
    ),
    ids=("one_tile", "three_tiles", "fp8_one_tile", "small_biased_plan"),
)
def test_a_grid_of_more_tiles_than_sms_seeds_256_columns(
    num_kv: int, num_bh: int, dtype: torch.dtype, small_biased_candidate: bool
) -> None:
    seeds = _seeds(
        num_kv, num_bh, dtype=dtype, small_biased_candidate=small_biased_candidate
    )
    shared = [seed for seed in seeds if seed.config[KEY] == 256]
    assert len(shared) == 1
    # The seed is the knob's dependency witness: the body the resolver
    # honors the allocation on, which on a small biased plan's surface is
    # not the default (the SIMT body, which allocates no tensor memory).
    for key, value in flash_knobs.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, 256)].items():
        assert shared[0].config[key] == value
    cfg = _resolve(
        64,
        num_kv,
        dict(shared[0].config),
        dtype=dtype,
        num_bh=num_bh,
        small_biased_candidate=small_biased_candidate,
    )
    assert cfg.tmem_cols == 256


def _small_biased_spec(small_biased_candidate: bool) -> ConfigSpec:
    """The search surface of one KV tile at head dim 64 on a grid of more
    tiles than SMs: of a small biased plan, whose default body is the SIMT
    one, or of the plain dense plan."""
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=None)
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    enable_cute_flash_search(
        spec,
        head_dim=64,
        num_kv=1,
        num_bh=192,
        dtype=torch.bfloat16,
        block_size_targets={0: 1, 1: 128, 2: 128},
        **_plan_facts(small_biased_candidate),  # pyrefly: ignore[bad-argument-type]
    )
    return spec


SMALL_BIASED_KEY = cute_flash.FLASH_SMALL_BIASED_KEY


def test_a_small_biased_pin_leaves_256_uncovered_without_a_defect() -> None:
    """``cute_flash_small_biased=True`` pinned on a small biased plan's
    surface keeps every candidate on the SIMT body, which allocates no
    tensor memory. No candidate of the design reaches the 256-column
    allocation, which is reported as uncovered and is no defect of the
    design (the knob's parents in ``FLASH_COVERAGE_CHILD_PARENTS`` are its
    witness's requirements), so the search starts and keeps all the columns.
    Without the entry the check refuses the design. (A pinned score-stage
    count or family never gets this far: the check steps aside for the keys
    that select a body.)"""
    assert (
        dict(flash_knobs.FLASH_COVERAGE_CHILD_PARENTS[(KEY, 256)])
        == (flash_knobs.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, 256)])
    )
    generation = shared_flash_generation(
        _small_biased_spec, True, overrides={SMALL_BIASED_KEY: True}
    )
    extension = cute_generation_extension(generation)
    assert (KEY, 256) in extension.flash_structural_coverage_uncovered_values()
    extension.validate_flash_structural_coverage()
    configs = extension.deterministic_population_configs()
    assert {config[KEY] for config in configs} == {512}
    parents = dict(flash_knobs.FLASH_COVERAGE_CHILD_PARENTS)
    del parents[(KEY, 256)]
    with (
        patch.object(config_generation_ext, "FLASH_COVERAGE_CHILD_PARENTS", parents),
        pytest.raises(AutotuneError, match="uncovered values"),
    ):
        extension.validate_flash_structural_coverage()


def test_a_small_biased_pin_excuses_nothing_on_a_plain_surface() -> None:
    """Off a small biased plan the resolver ignores the pin (the plan has no
    SIMT body), so the one-warpgroup body is still there: the design covers
    256 columns under it and the entry is not consulted."""
    generation = shared_flash_generation(
        _small_biased_spec, False, overrides={SMALL_BIASED_KEY: True}
    )
    extension = cute_generation_extension(generation)
    assert (KEY, 256) not in extension.flash_structural_coverage_uncovered_values()
    assert not extension._flash_coverage_parent_overridden((KEY, 256))
    extension.validate_flash_structural_coverage()
    configs = extension.deterministic_population_configs()
    assert {config[KEY] for config in configs} == {256, 512}


@pytest.mark.parametrize(
    ("num_kv", "num_bh", "device_sm_count"),
    ((1, 64, 148), (1, 148, 148), (1, 192, 0), (16, 64, 148)),
    ids=("fewer_tiles", "one_tile_per_sm", "unknown_device", "mixed_surface"),
)
def test_other_grids_seed_all_columns(
    num_kv: int, num_bh: int, device_sm_count: int
) -> None:
    # One wave already holds the grid, the device is unknown, or the surface
    # does not search the allocation.
    seeds = _seeds(num_kv, num_bh, device_sm_count=device_sm_count)
    assert seeds
    assert {seed.config[KEY] for seed in seeds} == {512}


# --------------------------------------------------------------------------
# generated code
# --------------------------------------------------------------------------


def test_render_with_all_columns_keeps_the_permit_to_the_last_mma() -> None:
    code = _render(_resolve(64, 1, dict(ONE_WARPGROUP)))
    assert ALLOCATE.format(512) in code
    assert "tOtO = cute.make_tensor(flash_tmem_ptr + 256, tOtO.layout)" in code
    # The P operand is make_fragment_A's (column 0, the allocation's base
    # when a CTA has all the columns).
    assert "tOrP = flash_pvt.make_fragment_A(tP)" in code
    assert "cute.recast_ptr(tStS.iterator" not in code
    assert code.count(RELEASE) == 1
    assert code.index("flash_q_full.release()") < code.index(RELEASE)


@pytest.mark.parametrize(
    ("head_dim", "num_kv", "io_dtype"),
    (
        (64, 1, "cutlass.BFloat16"),
        (128, 1, "cutlass.Float16"),
        (64, 3, "cutlass.Float8E4M3FN"),
    ),
    ids=("d64_one_tile", "d128_one_tile", "fp8_three_tiles"),
)
def test_render_with_256_columns_shares_the_sm(
    head_dim: int, num_kv: int, io_dtype: str
) -> None:
    all_columns = _render(
        _resolve(head_dim, num_kv, dict(ONE_WARPGROUP)),
        head_dim=head_dim,
        num_kv=num_kv,
        io_dtype=io_dtype,
    )
    cfg = _resolve(head_dim, num_kv, {**ONE_WARPGROUP, KEY: 256})
    assert cfg.tmem_cols == 256
    code = _render(cfg, head_dim=head_dim, num_kv=num_kv, io_dtype=io_dtype)
    rebase = (
        "tOrP = cute.make_tensor(cute.recast_ptr(tStS.iterator, "
        f"dtype={io_dtype}), tOrP.layout)"
    )
    # Half the columns; the permit goes as soon as the columns are there,
    # before anything else of the CTA runs.
    assert ALLOCATE.format(256) in code
    assert code.count(RELEASE) == 1
    assert f"{WAIT}\n{RELEASE}\n{RETRIEVE}" in code
    # The score tile at the allocation's base, the accumulator behind it,
    # and the PV MMA's P operand over the score tile's own pointer.
    assert "tStS = cute.make_tensor(flash_tmem_ptr, tStS.layout)" in code
    assert "tOtO = cute.make_tensor(flash_tmem_ptr + 128, tOtO.layout)" in code
    assert f"tOrP = flash_pvt.make_fragment_A(tP)\n{rebase}" in code
    # Nothing else differs from the body on all the columns.
    before = all_columns.splitlines()
    after = code.splitlines()
    assert [line for line in before if line not in after] == [
        ALLOCATE.format(512),
        "tOtO = cute.make_tensor(flash_tmem_ptr + 256, tOtO.layout)",
    ]
    assert [line for line in after if line not in before] == [
        ALLOCATE.format(256),
        "tOtO = cute.make_tensor(flash_tmem_ptr + 128, tOtO.layout)",
        rebase,
    ]
    assert len(after) == len(before) + 1


@pytest.mark.parametrize("tmem_cols", (512, 256))
@pytest.mark.parametrize("num_kv", (1, 3))
def test_render_issues_the_prologue_kv_loads_before_the_q_wait(
    tmem_cols: int, num_kv: int
) -> None:
    # Warp 0 issues Q and the prologue's K/V tiles back to back; the wait
    # for Q comes after all of them, so their latencies overlap.
    code = _render(
        _resolve(64, num_kv, {**ONE_WARPGROUP, KEY: tmem_cols}), num_kv=num_kv
    )
    assert f"for flash_pf in cutlass.range_constexpr({num_kv}):" in code
    q_issue = code.index("cute.copy(_flash_tma_q,")
    k_issue = code.index("cute.copy(_flash_tma_k,")
    v_issue = code.index("cute.copy(_flash_tma_v,")
    assert code.count(Q_WAIT) == 1
    q_wait = code.index(Q_WAIT)
    assert q_issue < k_issue < v_issue < q_wait < code.index("for flash_kv_d in")


def test_emitter_rejects_columns_outside_the_legality() -> None:
    cfg = _resolve(64, 1, {**ONE_WARPGROUP, KEY: 256})
    with pytest.raises(AssertionError):
        _render(dataclasses.replace(cfg, tmem_cols=128))
    # A 256-column accumulator leaves no room for the score tile.
    wide = _resolve(256, 1, dict(ONE_WARPGROUP))
    assert wide.tmem_cols == 512
    with pytest.raises(AssertionError):
        _render(dataclasses.replace(wide, tmem_cols=256), head_dim=256)


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------


def _attention_kernel() -> Kernel:
    """``examples/attention.py``'s output kernel under the CuTe backend."""
    example = import_path(EXAMPLES_DIR / "attention.py")
    return Kernel(
        example.attention_output.fn,
        settings=example.attention_output.settings.copy(backend="cute"),
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("batch", "heads", "seq", "head_dim", "dtype"),
    (
        (4, 48, 128, 64, torch.bfloat16),
        (4, 48, 128, 128, torch.float16),
        (2, 50, 384, 64, torch.bfloat16),
    ),
    ids=("d64_one_tile", "d128_one_tile", "d64_three_tiles"),
)
def test_256_columns_match_all_columns_on_device(
    batch: int, heads: int, seq: int, head_dim: int, dtype: torch.dtype
) -> None:
    """Grids of more CTAs than a B200 has SMs (192 and 300 on 148), so two
    CTAs hold tensor memory on one SM: the CTA whose columns start at 256
    must read its own P tile. The two allocations run the same arithmetic,
    so their outputs agree bit for bit."""
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("the fused tcgen05 flash bodies need an sm100 device")
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(batch, heads, seq, head_dim, device=DEVICE).to(dtype)
        for _ in range(3)
    )
    args = (q, k, v)
    kernel = _attention_kernel()
    spec = kernel.bind(args).config_spec
    assert spec.cute_flash_search_enabled
    base = {**spec.default_config().config, **ONE_WARPGROUP}
    code, all_columns = compile_and_run(
        kernel, args, helion.Config(**{**base, KEY: 512})
    )
    assert ALLOCATE.format(512) in code
    code, shared = compile_and_run(kernel, args, helion.Config(**{**base, KEY: 256}))
    assert ALLOCATE.format(256) in code
    assert torch.equal(shared, all_columns)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()
    )
    torch.testing.assert_close(shared.float(), reference, atol=2e-2, rtol=2e-2)
