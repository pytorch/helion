"""The plain causal fa4 pair under ``cute_flash_persistent``.

A causal work item is a pair of 128-row query tiles of one batch head, and
its KV tile count is the pair's place on the diagonal, so the items of a grid
differ in length by up to the sequence. One CTA per item (the default) pays
the CTA setup per item. With ``cute_flash_persistent`` a CTA walks several
items, and every walking role decodes each item's place in the longest-first
order and its KV count at the top of the item. ``cute_flash_persistent_loop``
names the scheduler:

* ``while``: the static strided loop. CTA ``c`` of ``G`` takes the places
  ``c``, ``c + G``, ... and mirrors every other round, so long and short
  items alternate on each CTA.
* ``clc``: the CLC scheduler. The launch grid is the whole problem, a CTA
  that finishes an item takes the next one that has not started, and the
  place is the stolen CTA's linear block index. The load warp asks for the
  next item after the current item's last load. The grid is (pairs, heads,
  batches) while CUDA accepts it and flat in x otherwise.
"""

from __future__ import annotations

import ast
import functools
import math
import os
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute import flash_knobs
from helion._compiler.cute.attention_plan import SLIDING_WINDOW_MASK_KIND
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing.cute_flash import bind_flash_kernel
from helion._testing.cute_flash import compile_and_run
from helion._testing.cute_flash import flash_config_spec
from helion._testing.cute_flash import memoized_flash_fragments
from helion._testing.cute_flash import render_flash_config
import helion.language as hl
from helion.runtime.cute.launcher import _append_cute_wrapper_plan

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction
    from helion.autotuner.config_fragment import ConfigSpecFragment

pytestmark = pytest.mark.cpu_only

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

PERSISTENT_KEY = cute_flash.FLASH_PERSISTENT_KEY
LOOP_KEY = cute_flash.FLASH_PERSISTENT_LOOP_KEY
FAMILY_KEY = cute_flash.FLASH_PIPELINE_FAMILY_KEY
GROUP_KEY = cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY

STATIC = {FAMILY_KEY: "fa4", PERSISTENT_KEY: True}
CLC = {**STATIC, LOOP_KEY: "clc"}

# The exact (hardware exp2) packet of the causal hd128 resident softmax
# schedule, which needs no ``fast_math`` opt-in.
RESIDENT_PACKET = "causal_hd128_resident3_013_prefetch2_xu_early_acquire"
RESIDENT: dict[str, object] = {cute_flash.FLASH_EXP2_PACKET_KEY: RESIDENT_PACKET}
RESIDENT_PASS = "fa4_disc_exp_convert_store_resident3_013_prefetch2("

_IO_DTYPES = {
    torch.bfloat16: "cutlass.BFloat16",
    torch.float8_e4m3fn: "cutlass.Float8E4M3FN",
}

LOAD_GUARD = "warp_idx == 14"
EMPTY_GUARD = "warp_idx == 15"
STRIDED_LOOP = "while flash_tile_id <"
CLC_LOOP = "while flash_clc_work.is_valid_tile:"
CLC_QUERY = "cute.arch.issue_clc_query("


def _resolve(
    config: dict[str, object],
    *,
    head_dim: int = 64,
    num_kv: int = 16,
    num_bh: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    env: dict[str, str] | None = None,
    **facts: object,
) -> cute_flash.FlashAttentionConfig:
    options: dict[str, object] = {
        "standard_causal_output": True,
        "plain_row_body": True,
        "score_modifier_kinds": (cute_flash.CAUSAL_MASK_KIND,),
        **facts,
    }
    with patch.dict(os.environ, env or {}, clear=True):
        return cast(
            "cute_flash.FlashAttentionConfig",
            cute_flash.resolve_flash_config(
                head_dim,
                num_kv,
                config,
                dtype=dtype,
                num_bh=num_bh,
                is_causal=True,
                **options,  # pyrefly: ignore[bad-argument-type]
            ),
        )


def _fragments(*, head_dim: int = 64, **facts: object) -> dict[str, ConfigSpecFragment]:
    options: dict[str, object] = {
        "standard_causal_output": True,
        "plain_row_body": True,
        "score_modifier_kinds": (cute_flash.CAUSAL_MASK_KIND,),
        **facts,
    }
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.flash_autotune_fragments(
            head_dim,
            16,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=True,
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _active(fragment: ConfigSpecFragment) -> tuple[object, ...]:
    choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
    search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
    return choices if search is None else tuple(search)


def _render(
    config: dict[str, object],
    *,
    head_dim: int = 64,
    num_kv: int = 16,
    num_bh: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    cfg = _resolve(config, head_dim=head_dim, num_kv=num_kv, num_bh=num_bh, dtype=dtype)
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=num_kv,
        sequence_extent=num_kv * 128,
        num_bh=num_bh,
        total_tiles=num_bh * (num_kv // 2),
        cfg=cfg,
        has_lse=False,
        io_dtype=_IO_DTYPES[dtype],
        score_plan=causal_score_plan(head_dim),
    )
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _launch_grid(code: str) -> tuple[int, ...]:
    """The grid the launcher builds from the plan of a generated kernel."""
    marker = "._helion_cute_wrapper_plans = "
    payload = next(line for line in code.splitlines() if marker in line)
    (plan,) = ast.literal_eval(payload.split(marker, 1)[1])
    body: list[str] = []
    _append_cute_wrapper_plan(body, [], plan, num_sm=148)
    extents = re.findall(r"grid_[xyz] = cutlass\.Int32\((\d+)\)", "\n".join(body))
    return tuple(int(extent) for extent in extents)


def _role(code: str, guard: str) -> ast.If:
    (role,) = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.If) and ast.unparse(node.test) == guard
    ]
    return role


def _item_loop(role: ast.If) -> tuple[list[ast.stmt], ast.While]:
    """The role's statements ahead of its work-item loop, and the loop."""
    (index,) = [
        index for index, stmt in enumerate(role.body) if isinstance(stmt, ast.While)
    ]
    loop = role.body[index]
    assert isinstance(loop, ast.While)
    return role.body[:index], loop


# The names of an item decode: the loop's carried state and the decoded place.
_DECODE_NAMES = frozenset(
    {
        "flash_tile_id",
        "flash_grid_dim",
        "flash_lpt_round",
        "flash_lpt_mirror",
        "flash_lpt_place",
        "flash_pid",
        "flash_lpt_group",
        "flash_lpt_mod",
        "flash_m_pair_raw",
        "flash_bh",
        "flash_m_pair",
    }
)


def _is_decode(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Assign):
        return all(
            isinstance(target, ast.Name) and target.id in _DECODE_NAMES
            for target in stmt.targets
        )
    if isinstance(stmt, ast.If):
        return all(_is_decode(inner) for inner in (*stmt.body, *stmt.orelse))
    return False


def _decode_statements(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """The item decode of a role's statements, with the decoded item recorded
    behind the pair's assignment."""
    kept: list[ast.stmt] = []
    for stmt in stmts:
        if not _is_decode(stmt):
            continue
        kept.append(stmt)
        if (
            isinstance(stmt, ast.Assign)
            and cast("ast.Name", stmt.targets[0]).id == "flash_m_pair"
        ):
            kept.extend(ast.parse("flash_items.append((flash_m_pair, flash_bh))").body)
    return kept


def _run(stmts: list[ast.stmt], scope: dict[str, object]) -> None:
    module = ast.Module(body=stmts, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<item decode>", "exec"), scope)


def _scope(**names: object) -> dict[str, object]:
    return {
        "cutlass": SimpleNamespace(Int32=int, min=min),
        "flash_items": [],
        **names,
    }


def _static_items(code: str, *, grid: int) -> list[list[tuple[int, int]]]:
    """The ``(m_pair, bh)`` items each CTA of a ``grid``-wide launch walks, by
    running the load role's decode as plain Python."""
    setup, loop = _item_loop(_role(code, LOAD_GUARD))
    walk = ast.While(test=loop.test, body=_decode_statements(loop.body), orelse=[])
    per_cta = []
    for block in range(grid):
        arch = SimpleNamespace(
            block_idx=lambda block=block: (block, 0, 0),
            grid_dim=lambda: (grid, 1, 1),
        )
        scope = _scope(cute=SimpleNamespace(arch=arch))
        _run([*_decode_statements(setup), walk], scope)
        per_cta.append(cast("list[tuple[int, int]]", scope["flash_items"]))
    return per_cta


def _clc_items(code: str, *, grid: tuple[int, int, int]) -> list[tuple[int, int]]:
    """The ``(m_pair, bh)`` item of every CTA of the CLC launch ``grid``, in
    the grid's linear order (x fastest)."""
    _setup, loop = _item_loop(_role(code, LOAD_GUARD))
    decode = _decode_statements(loop.body)
    items = []
    for z in range(grid[2]):
        for y in range(grid[1]):
            for x in range(grid[0]):
                scope = _scope(flash_clc_work=SimpleNamespace(tile_idx=(x, y, z)))
                _run(decode, scope)
                (item,) = cast("list[tuple[int, int]]", scope["flash_items"])
                items.append(item)
    return items


# --------------------------------------------------------------------------
# resolver and surface
# --------------------------------------------------------------------------


def test_a_causal_config_without_the_key_keeps_one_item_per_cta() -> None:
    cfg = _resolve({FAMILY_KEY: "fa4"})
    assert not cfg.persistent
    assert not cfg.use_clc_scheduler
    assert cfg.persistent_loop == "while"


@pytest.mark.parametrize("head_dim", (64, 128))
def test_the_plain_causal_pair_runs_either_loop_form(head_dim: int) -> None:
    static = _resolve(STATIC, head_dim=head_dim)
    assert static.persistent and not static.use_clc_scheduler
    assert static.persistent_loop == "while"
    clc = _resolve(CLC, head_dim=head_dim)
    assert clc.persistent and clc.use_clc_scheduler
    assert clc.persistent_loop == "clc"
    # The CLC form is the plain family's loop, not the ``fa4_clc`` family:
    # the causal packets and knobs keyed on ``fa4`` stay legal under it.
    assert static.pipeline_family == clc.pipeline_family == "fa4"
    # One identity per render: the grid's factoring and the ring depth of
    # the dense CLC family are pinned.
    assert clc.clc_heads_per_batch == 0
    assert clc.clc_stages == 2
    assert _resolve({**CLC, cute_flash.FLASH_CLC_STAGES_KEY: 3}).clc_stages == 2
    assert (
        _resolve({**CLC, cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY: 8})
    ).clc_heads_per_batch == 0


# The legacy structural keys that ask for an fa4 variant, and the environment
# variable each one defaults to.
_LEGACY_VARIANTS = (
    (cute_flash.FLASH_USE_2CTA_KEY, "HELION_CUTE_FLASH_USE_2CTA"),
    (cute_flash.FLASH_CGA2_LOCAL_KEY, "HELION_CUTE_FLASH_CGA2_LOCAL"),
    (cute_flash.FLASH_CLC_KEY, "HELION_CUTE_FLASH_CLC"),
    (cute_flash.FLASH_LOCAL_TMA_PARTITION_KEY, "HELION_CUTE_FLASH_LOCAL_TMA"),
    (cute_flash.FLASH_TENSOR_4D_TMA_KEY, "HELION_CUTE_FLASH_TENSOR_4D"),
)


# Family names that ask for an fa4 variant no causal plan takes (the resolver
# clears each), and the environment variables of the family and of the loop.
_CLEARED_FAMILIES = ("fa4_tma_4d", "fa4_local_tma", "fa4_clc", "fa4_cga2_local")
_FAMILY_ENV = "HELION_CUTE_FLASH_PIPELINE_FAMILY"
_LOOP_ENV = "HELION_CUTE_FLASH_PERSISTENT_LOOP"


def test_the_pair_needs_no_family_name() -> None:
    """A config that names no pipeline family resolves to the plain fa4 pair
    at an even KV tile count, so ``cute_flash_persistent`` is honored on it
    as on a config that names ``fa4`` (it used to be dropped: the check read
    the requested family, which such a config does not have). A legacy
    structural key, or its environment variable, that asks for an fa4
    variant changes nothing: no causal plan takes those variants, the
    resolver clears each, and the pair is the plain family's. An odd tile
    count keeps one item per CTA, where the topology is the single-tile
    body's."""
    for loop in ("while", "clc"):
        unnamed = _resolve({PERSISTENT_KEY: True, LOOP_KEY: loop})
        assert unnamed.persistent and unnamed.pipeline_family == "fa4"
        assert unnamed.persistent_loop == loop
        assert unnamed == _resolve({**STATIC, LOOP_KEY: loop})
    plain = _resolve(STATIC)
    # The legacy topology key names the pair's topology, not a variant.
    legacy_topology = {cute_flash.FLASH_TOPOLOGY_KEY: "fa4", PERSISTENT_KEY: True}
    assert _resolve(legacy_topology) == plain
    for key, variable in _LEGACY_VARIANTS:
        assert _resolve({PERSISTENT_KEY: True, key: True}) == plain
        assert _resolve({PERSISTENT_KEY: True}, env={variable: "1"}) == plain
        # One item per CTA resolves to the plain family as well: the
        # variant is cleared with or without the persistent loop.
        one_item = _resolve({PERSISTENT_KEY: False, key: True})
        assert one_item == _resolve({FAMILY_KEY: "fa4", PERSISTENT_KEY: False})
    # The same holds for a family name that no causal plan takes, in the
    # config or in the environment: the plan is the plain family's, which
    # is what the pair asks for.
    for family in _CLEARED_FAMILIES:
        assert _resolve({PERSISTENT_KEY: True, FAMILY_KEY: family}) == plain
        assert _resolve({PERSISTENT_KEY: True}, env={_FAMILY_ENV: family}) == plain
    assert not _resolve({PERSISTENT_KEY: True}, num_kv=15).persistent


def _normalized(
    config: dict[str, object],
    env: dict[str, str] | None = None,
    *,
    is_causal: bool = True,
) -> dict[str, object]:
    """``config`` as the config spec of a causal surface (or of a dense one)
    normalizes it. The surface is the detector's, without the 4-D TMA
    descriptors: they take fp16 operands without a mask, and these are bf16,
    under a mask on the causal plan
    (``_flash_values_support_tensor_4d_tma``). The spec is built under
    ``env`` as well: it captures the knob environment when it is built, and
    the normalization resolves under that capture."""
    with patch.dict(os.environ, env or {}, clear=True), memoized_flash_fragments():
        spec = flash_config_spec(
            head_dim=64,
            num_kv=16,
            dtype=torch.bfloat16,
            is_causal=is_causal,
            num_bh=64,
            supports_tensor_4d_tma=False,
        )
        return spec.normalized_config({"block_sizes": [1, 128, 128], **config}).config


def test_a_config_normalizes_to_what_it_resolves_to() -> None:
    """A bound kernel's config spec folds a config's legacy structural keys
    into the family they resolve to (``upgrade_legacy_flash_config``, which
    keeps the config's persistence), fills the keys the config lacks and
    resolves the result again. The config a kernel compiles is therefore the
    normalized one: it has to be the plan that resolving the config directly
    gives. Three inputs used to split the two. A variant key kept one item
    per CTA when resolved and the persistent loop when normalized, and so
    did a family that the environment names and a causal plan clears. A
    loop form asked for through the environment was resolved and then
    replaced: the default loop was filled in before the resolution read the
    environment. A config that names a family no causal plan takes is not
    legal on the surface, unless the name carries the 4-D TMA maps: a
    surface without the 4-D descriptors, which every causal one is, takes
    such a name for the family it resolves to without them, and on a causal
    plan that is the plain pair for each of them."""
    plain = _resolve(STATIC)

    def assert_the_plain_pair(normalized: dict[str, object], loop: str) -> None:
        assert normalized[FAMILY_KEY] == "fa4"
        assert normalized[PERSISTENT_KEY] is True
        assert normalized[LOOP_KEY] == loop

    for key, variable in _LEGACY_VARIANTS:
        legacy = {PERSISTENT_KEY: True, key: True}
        normalized = _normalized(legacy)
        assert key not in normalized
        assert_the_plain_pair(normalized, "while")
        assert _resolve(legacy) == plain
        assert_the_plain_pair(
            _normalized({PERSISTENT_KEY: True}, {variable: "1"}), "while"
        )
    # A surface without the 4-D descriptors takes a name that carries the
    # 4-D maps for what it resolves to without them: the plain pair on a
    # causal plan, whatever else the name asks for. Such a config used to
    # keep one item per CTA.
    tensor_4d_names = [
        name
        for name, flags in flash_knobs.FLASH_PIPELINE_FAMILY_FLAGS.items()
        if flags.tensor_4d_tma
    ]
    assert "fa4_tma_4d" in tensor_4d_names and "fa4_clc_tma_4d" in tensor_4d_names
    for family in tensor_4d_names:
        assert_the_plain_pair(
            _normalized({PERSISTENT_KEY: True, FAMILY_KEY: family}), "while"
        )
    for family in _CLEARED_FAMILIES:
        if family not in tensor_4d_names:
            # A config that names one of the others does not normalize at all.
            with pytest.raises(helion.exc.InvalidConfig, match="is not legal"):
                _normalized({PERSISTENT_KEY: True, FAMILY_KEY: family})
        env = {_FAMILY_ENV: family}
        assert_the_plain_pair(_normalized({PERSISTENT_KEY: True}, env), "while")
        assert _resolve({PERSISTENT_KEY: True}, env=env) == plain
    # The environment's loop form reaches a config that turns persistence
    # on and names no loop; a loop the config names outranks it.
    env = {_LOOP_ENV: "clc"}
    assert_the_plain_pair(_normalized({PERSISTENT_KEY: True}, env), "clc")
    assert _resolve({PERSISTENT_KEY: True}, env=env) == _resolve(CLC)
    assert_the_plain_pair(
        _normalized({PERSISTENT_KEY: True, LOOP_KEY: "while"}, env), "while"
    )
    # Without persistence the loop form is the one-item plan's.
    one_item = _normalized({PERSISTENT_KEY: False}, env)
    assert one_item[PERSISTENT_KEY] is False and one_item[LOOP_KEY] == "while"
    # A dense surface reads the environment's loop form the same way.
    env = {"HELION_CUTE_FLASH_PERSISTENT": "0", _LOOP_ENV: "counted"}
    dense = _normalized({FAMILY_KEY: "fa4", PERSISTENT_KEY: True}, env, is_causal=False)
    assert dense[PERSISTENT_KEY] is True and dense[LOOP_KEY] == "counted"


def test_the_loop_form_needs_the_persistent_pair() -> None:
    cfg = _resolve({FAMILY_KEY: "fa4", PERSISTENT_KEY: False, LOOP_KEY: "clc"})
    assert not cfg.persistent
    assert not cfg.use_clc_scheduler
    assert cfg.persistent_loop == "while"


@pytest.mark.parametrize("loop", ("while", "clc"))
def test_persistence_clears_the_one_item_schedules(loop: str) -> None:
    # The staggered walk, the pre-sync loads, the output over the dead Q
    # stages and the head pair are written for one work item per CTA.
    requested = {
        **STATIC,
        LOOP_KEY: loop,
        cute_flash.FLASH_CAUSAL_STAGGER_KEY: True,
        cute_flash.FLASH_EARLY_PROLOGUE_LOADS_KEY: True,
        cute_flash.FLASH_O_OVER_Q_KEY: True,
        cute_flash.FLASH_CAUSAL_HEAD_PAIR_KEY: True,
        cute_flash.FLASH_KV_STAGE_KEY: 4,
        cute_flash.FLASH_EPI_TMA_KEY: True,
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    }
    cfg = _resolve(requested, kv_group=2)
    assert cfg.persistent
    assert not cfg.causal_stagger
    assert not cfg.early_prologue_loads
    assert not cfg.o_over_q
    assert not cfg.causal_head_pair
    one_item = _resolve({**requested, PERSISTENT_KEY: False}, kv_group=2)
    assert one_item.causal_stagger or one_item.causal_head_pair


@pytest.mark.parametrize(
    ("config", "facts"),
    (
        # The other causal families have schedulers of their own (the
        # two-CTA family is fp16's; a plan that cannot take it is the plain
        # pair, test_the_pair_needs_no_family_name).
        ({FAMILY_KEY: "fa4_2cta_causal"}, {"dtype": torch.float16}),
        ({FAMILY_KEY: "ws_overlap"}, {}),
        # The stateful whole-row lowering is verified for one work item per
        # CTA.
        (
            {cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful", GROUP_KEY: 1},
            {"causal_resident_compatible": True},
        ),
        # A KV-tile-pruned mask binds per-item values in the CTA-wide setup.
        (
            {},
            {
                "has_kv_tile_pruning": True,
                "has_score_modifiers": True,
                "plain_row_body": False,
                "score_modifier_kinds": (SLIDING_WINDOW_MASK_KIND,),
            },
        ),
    ),
    ids=("two_cta", "single_tile", "stateful_lowering", "pruned_mask"),
)
@pytest.mark.parametrize("loop", ("while", "clc"))
def test_other_causal_bodies_keep_one_item_per_cta(
    config: dict[str, object], facts: dict[str, object], loop: str
) -> None:
    cfg = _resolve({**STATIC, LOOP_KEY: loop, **config}, **facts)
    assert not cfg.use_clc_scheduler
    if cfg.topology == "fa4":
        assert not cfg.persistent
    assert cfg.persistent_loop == "while"


@pytest.mark.parametrize("head_dim", (64, 128))
def test_plain_causal_surfaces_search_both_loop_forms(head_dim: int) -> None:
    fragments = _fragments(head_dim=head_dim)
    persistent = fragments[PERSISTENT_KEY]
    assert _active(persistent) == (False, True)
    assert persistent.default() is False
    loop = fragments[LOOP_KEY]
    assert _active(loop) == ("while", "clc")
    assert loop.default() == "while"
    # No CLC family on a causal surface: the scheduler is the loop form.
    families = _active(fragments[FAMILY_KEY])
    assert "fa4" in families
    assert not [family for family in families if "clc" in str(family)]


@pytest.mark.parametrize(
    "facts",
    (
        {
            "has_kv_tile_pruning": True,
            "has_score_modifiers": True,
            "plain_row_body": False,
            "score_modifier_kinds": (SLIDING_WINDOW_MASK_KIND,),
        },
        # The resolver, the search and the device body share one rule: the
        # causal mask once.
        {
            "score_modifier_kinds": (
                cute_flash.CAUSAL_MASK_KIND,
                cute_flash.CAUSAL_MASK_KIND,
            )
        },
    ),
    ids=("pruned_mask", "masked_twice"),
)
def test_other_causal_surfaces_search_neither(facts: dict[str, object]) -> None:
    fragments = _fragments(**facts)
    assert _active(fragments[PERSISTENT_KEY]) == (False,)
    assert _active(fragments[LOOP_KEY]) == ("while",)


def test_the_causal_coverage_design_reaches_the_persistent_pair() -> None:
    spec = flash_config_spec(
        head_dim=64, num_kv=16, dtype=torch.bfloat16, is_causal=True, num_bh=64
    )
    generation = spec.create_config_generation()
    with memoized_flash_fragments():
        configs = generation.extension.deterministic_population_configs()
    forms = {
        (config[FAMILY_KEY], config[PERSISTENT_KEY], config[LOOP_KEY])
        for config in configs
    }
    # The design covers every searched value once: the CLC form carries both
    # the persistent pair and its loop form (the seeds carry the static loop).
    assert ("fa4", True, "clc") in forms
    assert ("fa4", False, "while") in forms
    # Every other family normalizes the pair away.
    assert not [form for form in forms if form[0] != "fa4" and form[1]]


@pytest.mark.parametrize("head_dim", (64, 128))
def test_the_persistent_seeds_keep_their_parents_softmax_pass(head_dim: int) -> None:
    # The seeds repeat the surface's measured one-item bodies under both loop
    # forms. A loop form that withdrew its parent's exp2 packet would time
    # another softmax pass than the body it was seeded from (the causal hd128
    # resident schedule used to decline every CLC scheduler, which made the
    # CLC form 19% slower per KV step than the static loop).
    facts: dict[str, object] = {
        "num_bh": 64,
        "dtype": torch.bfloat16,
        "is_causal": True,
        "standard_causal_output": True,
        "target_device_capability": (10, 0),
    }
    with patch.dict(os.environ, {}, clear=True):
        seeds = cute_flash.flash_attention_seed_configs(
            head_dim,
            16,
            **facts,  # pyrefly: ignore[bad-argument-type]
        )
        persistent = [seed.config for seed in seeds if seed.config.get(PERSISTENT_KEY)]
        assert {config[LOOP_KEY] for config in persistent} == {"while", "clc"}
        packets: dict[str, set[object]] = {"while": set(), "clc": set()}
        for config in persistent:
            resolved = cute_flash.resolve_flash_config(
                head_dim,
                16,
                {key: value for key, value in config.items() if key != "block_sizes"},
                **facts,  # pyrefly: ignore[bad-argument-type]
            )
            assert resolved.persistent
            assert resolved.persistent_loop == config[LOOP_KEY]
            assert resolved.exp2_packet == config[cute_flash.FLASH_EXP2_PACKET_KEY]
            packets[resolved.persistent_loop].add(resolved.exp2_packet)
    # Both loop forms carry every measured packet of the surface.
    assert packets["while"] == packets["clc"]
    if head_dim == 128:
        assert packets["clc"] & cute_flash._FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS


@pytest.mark.parametrize("num_bh", (1, 64, None))
def test_the_static_form_is_seeded_wherever_the_grid_is_known(
    num_bh: int | None,
) -> None:
    """The static loop is seeded with the whole grid as one launch group, so
    the builder has to know the batch-head count. One batch head counts: its
    pairs are still a grid of items of every length, in several waves on a
    long sequence (the seed used to need two heads). Without the count the
    CLC form alone is seeded."""
    with patch.dict(os.environ, {}, clear=True):
        seeds = cute_flash.flash_attention_seed_configs(
            64,
            16,
            num_bh=num_bh,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 0),
        )
    persistent = [seed.config for seed in seeds if seed.config.get(PERSISTENT_KEY)]
    assert "clc" in {config[LOOP_KEY] for config in persistent}
    static_groups = {
        config[GROUP_KEY] for config in persistent if config[LOOP_KEY] == "while"
    }
    assert static_groups == (set() if num_bh is None else {num_bh})


# --------------------------------------------------------------------------
# codegen
# --------------------------------------------------------------------------

# Launch groups of the longest-first order: one that divides the batch heads
# and one that leaves a smaller last group (the residual decode).
_GROUPS = (
    pytest.param(6, 6, id="whole_grid"),
    pytest.param(6, 1, id="serial_heads"),
    pytest.param(6, 2, id="dividing_group"),
    pytest.param(6, 4, id="residual_group"),
)


@pytest.mark.parametrize(("num_bh", "group"), _GROUPS)
def test_the_static_loop_walks_every_item_once(num_bh: int, group: int) -> None:
    num_kv = 16
    code = _render({**STATIC, GROUP_KEY: group}, num_kv=num_kv, num_bh=num_bh)
    assert STRIDED_LOOP in code and CLC_LOOP not in code
    every = {(pair, bh) for pair in range(num_kv // 2) for bh in range(num_bh)}
    # The body does not depend on the grid's width, so one render is walked
    # at three: two that divide neither the 48 items nor a launch group and
    # end in a short round, mirrored (5: three places in round 9) and not
    # (7: six places in round 6), and one CTA per item (48).
    for grid in (5, 7, 48):
        per_cta = _static_items(code, grid=grid)
        walked = [item for items in per_cta for item in items]
        assert len(walked) == len(every)
        assert set(walked) == every


def test_the_static_loop_mirrors_every_other_round() -> None:
    """Six serial heads on a grid as wide as a head's eight pairs: round
    ``r`` is head ``r``, its pairs longest first. CTA ``c`` takes place ``c``
    of an even round (pair ``7 - c``) and the mirrored place of an odd one
    (pair ``c``), so every CTA walks the mean, 54 KV tiles. In one direction
    CTA 0 would take the longest pair of every head, 96 tiles."""
    num_kv, num_bh = 16, 6
    pairs = num_kv // 2
    code = _render({**STATIC, GROUP_KEY: 1}, num_kv=num_kv, num_bh=num_bh)
    per_cta = _static_items(code, grid=pairs)
    for cta, items in enumerate(per_cta):
        assert items == [
            (cta if head % 2 else pairs - 1 - cta, head) for head in range(num_bh)
        ]
    loads = {sum(2 * pair + 2 for pair, _bh in items) for items in per_cta}
    assert loads == {54}


@pytest.mark.parametrize(("num_bh", "group"), _GROUPS)
def test_the_clc_decode_orders_the_launch_grid_longest_first(
    num_bh: int, group: int
) -> None:
    num_kv = 16
    pairs = num_kv // 2
    code = _render({**CLC, GROUP_KEY: group}, num_kv=num_kv, num_bh=num_bh)
    assert CLC_LOOP in code and STRIDED_LOOP not in code
    # The launcher's grid: the pairs along x, the batch heads along y.
    items = _clc_items(code, grid=(pairs, num_bh, 1))
    every = {(pair, bh) for pair in range(pairs) for bh in range(num_bh)}
    assert len(items) == len(every)
    assert set(items) == every
    # Inside a launch group the pair descends (longest first) and the heads
    # are the fastest index.
    group_items = group * pairs
    for start in range(0, len(items), group_items):
        section = items[start : start + group_items]
        lengths = [pair for pair, _bh in section]
        assert lengths == sorted(lengths, reverse=True)
        heads = {bh for _pair, bh in section}
        assert heads == set(range(start // pairs, start // pairs + len(heads)))
    # A grid CUDA would reject (more than 65,535 on y or z; here the cap is
    # lowered under the six heads) launches the same CTAs flat in x: the
    # same items in the same order, decoded from x alone.
    with patch.object(cute_flash, "_CUDA_GRID_YZ_MAX", num_bh - 1):
        flat = _render({**CLC, GROUP_KEY: group}, num_kv=num_kv, num_bh=num_bh)
    assert "flash_clc_work.tile_idx[1]" in code
    assert "flash_clc_work.tile_idx[1]" not in flat
    assert f"problem_shape_ntile_mnl=({pairs * num_bh}, 1, 1)" in flat
    assert _clc_items(flat, grid=(pairs * num_bh, 1, 1)) == items


def test_the_automatic_launch_group_counts_the_operand_bytes() -> None:
    """The automatic launch group is the batch heads whose K and V fit the L2
    budget, so it reads the operands' element size: 32 heads of 4096 tokens
    at hd64 in bf16 and 64 in fp8, which it used to size by two bytes as
    well. The group is the one-item body's too."""
    pairs = 16
    for dtype, group in ((torch.bfloat16, 32), (torch.float8_e4m3fn, 64)):
        for config in (CLC, {FAMILY_KEY: "fa4"}):
            code = _render(config, num_kv=2 * pairs, num_bh=128, dtype=dtype)
            assert f"flash_lpt_group = flash_pid // {group * pairs}" in code


def test_every_walking_role_decodes_its_item() -> None:
    for config, loop in ((STATIC, STRIDED_LOOP), (CLC, CLC_LOOP)):
        code = _render(config)
        roles = [
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.If) and "warp_idx" in ast.unparse(node.test)
        ]
        walking = 0
        for role in roles:
            source = ast.unparse(role)
            if loop not in source:
                continue
            _setup, item_loop = _item_loop(role)
            body = ast.unparse(item_loop)
            if "flash_num_active_kv" in body:
                # A role that walks the KV tiles decodes the item first.
                assert body.index("flash_m_pair = ") < body.index(
                    "flash_num_active_kv = "
                )
                walking += 1
        # The load, MMA, two softmax and correction roles.
        assert walking >= 5


def test_the_clc_form_claims_from_the_load_warp() -> None:
    code = _render(CLC)
    load = ast.unparse(_role(code, LOAD_GUARD))
    # One query per item, behind the item's last load and ahead of the
    # item's response.
    assert load.count(CLC_QUERY) == 1
    assert load.index("for flash_kv_load_iter in") < load.index(CLC_QUERY)
    assert load.index(CLC_QUERY) < load.index("flash_clc_pipeline.consumer_wait(")
    assert "flash_clc_pipeline.producer_tail(" in load
    # The scheduler warp only consumes the responses.
    empty = ast.unparse(_role(code, EMPTY_GUARD))
    assert CLC_LOOP in empty
    assert CLC_QUERY not in empty
    assert "flash_clc_pipeline.consumer_release(" in empty
    assert code.count(CLC_QUERY) == 1


@pytest.mark.parametrize("config", (STATIC, CLC), ids=("static", "clc"))
def test_the_hd128_resident_schedule_renders_under_either_loop_form(
    config: dict[str, object],
) -> None:
    # The loop form replaces the item loop around the roles; the softmax
    # pass inside a KV step is the one-item body's.
    one_item = _render({FAMILY_KEY: "fa4", **RESIDENT}, head_dim=128)
    persistent = _render({**config, **RESIDENT}, head_dim=128)
    assert one_item.count(RESIDENT_PASS) == persistent.count(RESIDENT_PASS) == 2
    assert _resolve({**config, **RESIDENT}, head_dim=128).exp2_packet == (
        RESIDENT_PACKET
    )


def test_the_one_item_body_is_unchanged_by_the_loop_form() -> None:
    base = _render({FAMILY_KEY: "fa4"})
    assert STRIDED_LOOP not in base and CLC_LOOP not in base
    assert _render({FAMILY_KEY: "fa4", PERSISTENT_KEY: False, LOOP_KEY: "clc"}) == base


def test_the_generated_kernel_launches_the_whole_grid_under_clc() -> None:
    shape = (2, 4, 2048, 64)
    key = ("causal_persistent", *shape)

    def make_args() -> tuple[object, ...]:
        return tuple(torch.zeros(shape, dtype=torch.bfloat16) for _ in range(3))

    bound = bind_flash_kernel(_causal_attention, key, make_args)
    default = dict(bound.config_spec.default_config().config)
    clc = render_flash_config(_causal_attention, key, make_args, {**default, **CLC})
    assert CLC_LOOP in clc
    # The launcher sizes the grid from the plan: the whole problem, the pairs
    # along x and the batch heads along y.
    (plans,) = [
        line for line in clc.splitlines() if "_helion_cute_wrapper_plans" in line
    ]
    assert "'use_clc_scheduler': True" in plans
    assert "'total_tiles': 64" in plans and "'batch': 8" in plans
    assert "clc_flat_grid" not in plans
    assert _launch_grid(clc) == (8, 8, 1)
    static = render_flash_config(
        _causal_attention, key, make_args, {**default, **STATIC}
    )
    assert STRIDED_LOOP in static
    (plans,) = [
        line for line in static.splitlines() if "_helion_cute_wrapper_plans" in line
    ]
    assert "'use_clc_scheduler': False" in plans
    assert "'persistent': True" in plans


def test_the_clc_grid_above_the_cuda_cap_launches_flat() -> None:
    """65,536 batch heads of a kernel without a batch dimension: the grid of
    (pairs, heads, batches) has them all on y, one above what CUDA accepts,
    and the launch would be rejected where the one-item grid and the static
    loop, flat in x, run. The plan launches the same CTAs flat in x, and the
    place in the longest-first order is x."""
    shape = (65536, 512, 64)
    key = ("causal_persistent_wide", *shape)

    def make_args() -> tuple[object, ...]:
        return tuple(
            torch.empty(
                shape,
                dtype=torch.bfloat16,
                device="meta",  # @ignore-device-lint
            )
            for _ in range(3)
        )

    bound = bind_flash_kernel(_causal_attention, key, make_args)
    default = dict(bound.config_spec.default_config().config)
    clc = render_flash_config(_causal_attention, key, make_args, {**default, **CLC})
    assert CLC_LOOP in clc
    assert "'clc_flat_grid': True" in clc
    assert "problem_shape_ntile_mnl=(131072, 1, 1)" in clc
    assert "flash_lpt_place = cutlass.Int32(flash_clc_work.tile_idx[0])" in clc
    assert _launch_grid(clc) == (131072, 1, 1)


@helion.kernel(backend="cute", static_shapes=True)
def _causal_attention_masked_twice(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            visible = tile_m.index[None, :, None] >= tile_n.index[None, None, :]
            qk = torch.where(visible, qk, float("-inf"))
            qk = torch.where(visible, qk, float("-inf"))
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def test_a_kernel_that_masks_twice_renders_one_item_per_cta() -> None:
    """The matcher chains masks: a kernel that applies the causal mask twice
    has a plan with the kind twice. The pair takes the mask once (resolver,
    search and the device body's assertion agree), so a config that asks for
    the pair on such a kernel renders one item per CTA; it used to pass the
    resolver and fail the body's assertion."""
    shape = (4, 1024, 64)
    key = ("causal_masked_twice", *shape)

    def make_args() -> tuple[object, ...]:
        return tuple(torch.empty(shape, dtype=torch.bfloat16) for _ in range(3))

    bound = bind_flash_kernel(_causal_attention_masked_twice, key, make_args)
    assert bound.config_spec.flash_facts.score_modifier_kinds == (
        cute_flash.CAUSAL_MASK_KIND,
        cute_flash.CAUSAL_MASK_KIND,
    )
    default = dict(bound.config_spec.default_config().config)
    for pair in (STATIC, CLC):
        code = render_flash_config(
            _causal_attention_masked_twice, key, make_args, {**default, **pair}
        )
        assert "'persistent': False" in code
        assert STRIDED_LOOP not in code and CLC_LOOP not in code


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True)
def _causal_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _causal_attention_lse(q_in, k_in, v_in):
    """``_causal_attention`` with the log-sum-exp row as a second output."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
        lse[tile_b, tile_m] = m_i + torch.log2(l_i)
    return out.view(q_in.size()), lse.view(q_in.size()[:-1])


def _require_tcgen05() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("the fa4 pair needs tcgen05")


def _kernel(config: dict[str, object], *, lse: bool = False) -> helion.Kernel:
    return helion.kernel(
        (_causal_attention_lse if lse else _causal_attention).fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config(block_sizes=[1, 128, 128], **config),
    )


def _launch(config: dict[str, object], *args: torch.Tensor) -> torch.Tensor:
    out = _kernel(config)(*args)
    torch.cuda.synchronize()
    return out


def _launch_with_source(
    config: dict[str, object], *args: torch.Tensor, lse: bool = False
) -> tuple[str, Any]:
    """Compile a kernel of its own at ``config`` and launch it: the source
    that ran, and the result."""
    kernel = helion.kernel(
        (_causal_attention_lse if lse else _causal_attention).fn,
        backend="cute",
        static_shapes=True,
    )
    return compile_and_run(
        kernel, args, helion.Config(block_sizes=[1, 128, 128], **config)
    )


# The one-item body the persistent forms are compared with: the schedules
# that persistence clears are off, so both run the same arithmetic per row.
ONE_ITEM: dict[str, object] = {
    FAMILY_KEY: "fa4",
    PERSISTENT_KEY: False,
    cute_flash.FLASH_CAUSAL_STAGGER_KEY: False,
    cute_flash.FLASH_CAUSAL_HEAD_PAIR_KEY: False,
    cute_flash.FLASH_EARLY_PROLOGUE_LOADS_KEY: False,
    cute_flash.FLASH_O_OVER_Q_KEY: False,
}


@functools.cache
def _device_case(head_dim: int, packet: str | None) -> tuple[torch.Tensor, ...]:
    """``(q, k, v, one_item_output)`` of a grid of 1024-row batch heads (four
    work items each) whose items outnumber the device's SMs twice over: a
    persistent CTA walks two or three items and the CLC grid is stolen from.
    The one-item body's output (of the softmax ``packet``, or the default's)
    is checked against SDPA once."""
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    shape = (1, sms // 2 + 3, 1024, head_dim)
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE) for _ in range(3)
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )
    body = {} if packet is None else {cute_flash.FLASH_EXP2_PACKET_KEY: packet}
    one_item = _launch({**ONE_ITEM, **body}, q, k, v)
    torch.testing.assert_close(one_item, reference, atol=2e-2, rtol=2e-2)
    return q, k, v, one_item


# Nine launches, one compile each. The loop form is the scheduler under test
# (two forms). The launch group changes the item decode: the L2-sized default,
# one head per group and a group that leaves a smaller last one (the residual
# decode), on hd64 only (the decode does not depend on the head dim). The
# head dim changes the storage and the softmax body; hd128 also carries its
# resident softmax schedule, which each loop form has to keep.
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("head_dim", "config"),
    (
        pytest.param(64, STATIC, id="d64_static_l2_group"),
        pytest.param(64, {**STATIC, GROUP_KEY: 1}, id="d64_static_serial_heads"),
        pytest.param(64, {**STATIC, GROUP_KEY: 5}, id="d64_static_residual_group"),
        pytest.param(64, CLC, id="d64_clc_l2_group"),
        pytest.param(64, {**CLC, GROUP_KEY: 5}, id="d64_clc_residual_group"),
        pytest.param(128, {**STATIC, GROUP_KEY: 1}, id="d128_static"),
        pytest.param(128, CLC, id="d128_clc"),
        pytest.param(
            128, {**STATIC, GROUP_KEY: 1, **RESIDENT}, id="d128_static_resident"
        ),
        pytest.param(128, {**CLC, **RESIDENT}, id="d128_clc_resident"),
    ),
)
def test_the_persistent_pair_matches_the_one_item_body_on_device(
    head_dim: int, config: dict[str, object]
) -> None:
    _require_tcgen05()
    packet = config.get(cute_flash.FLASH_EXP2_PACKET_KEY)
    q, k, v, one_item = _device_case(head_dim, cast("str | None", packet))
    persistent = _launch({**ONE_ITEM, **config}, q, k, v)
    assert torch.equal(persistent, one_item)


@pytest.mark.cuda
def test_the_flat_clc_grid_matches_the_one_item_body_on_device() -> None:
    """The CLC grid's flat launch (a grid above CUDA's cap on y and z) with
    the cap lowered to four, so the case's batch heads take it: the CTAs
    decode their place from x and the result is the one-item body's."""
    _require_tcgen05()
    q, k, v, one_item = _device_case(64, None)
    with patch.object(cute_flash, "_CUDA_GRID_YZ_MAX", 4):
        code, flat = _launch_with_source({**ONE_ITEM, **CLC}, q, k, v)
    assert "'clc_flat_grid': True" in code
    assert CLC_LOOP in code
    assert torch.equal(flat, one_item)


@functools.cache
def _lse_device_case() -> tuple[torch.Tensor, ...]:
    """``(q, k, v, out, lse)`` of the hd64 device case with the kernel's LSE
    row: the one-item body's outputs, checked against SDPA and the masked
    scores' log-sum-exp (in the kernel's base-2 units) once."""
    q, k, v, _one_item = _device_case(64, None)
    out, lse = _kernel(ONE_ITEM, lse=True)(q, k, v)
    torch.cuda.synchronize()
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.size(-1))
    hidden = torch.ones_like(scores, dtype=torch.bool).triu(1)
    expected = torch.logsumexp(scores.masked_fill(hidden, float("-inf")), -1)
    torch.testing.assert_close(lse, expected * math.log2(math.e), atol=2e-2, rtol=2e-2)
    return q, k, v, out, lse


# The pair and its loop forms are searched on every plain causal fa4 surface,
# the ones with an LSE output among them: each item's LSE row is the item's
# own, as its output rows are.
@pytest.mark.cuda
@pytest.mark.parametrize("config", (STATIC, CLC), ids=("static", "clc"))
def test_the_persistent_pair_keeps_each_items_lse_row_on_device(
    config: dict[str, object],
) -> None:
    _require_tcgen05()
    q, k, v, one_out, one_lse = _lse_device_case()
    code, (out, lse) = _launch_with_source({**ONE_ITEM, **config}, q, k, v, lse=True)
    # The loop form that was asked for ran: one item per CTA would pass the
    # comparisons below as the reference's own kernel.
    assert (CLC_LOOP if config.get(LOOP_KEY) == "clc" else STRIDED_LOOP) in code
    assert torch.equal(out, one_out)
    assert torch.equal(lse, one_lse)
