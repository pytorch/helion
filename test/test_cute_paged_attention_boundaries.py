from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
from typing import TYPE_CHECKING
from typing import cast

import pytest

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.autotune.mma_dag_surface import _PAIRED_KNOB_PINS
from helion._compiler.cute.autotune.mma_dag_surface import enable_cute_mma_dag_search
from helion._compiler.cute.cute_mma_dag import (
    MMA_DAG_EARLY_TMEM_RETIREMENT_KEY as RETIREMENT,
)
from helion._compiler.cute.cute_mma_dag import MMA_DAG_KEY_CLIP_KEY as KEY_CLIP
from helion._compiler.cute.cute_mma_dag import MMA_DAG_V_LAG_CHOICES
from helion._compiler.cute.cute_mma_dag import (
    MMA_DAG_VALUE_METADATA_SKIP_KEY as METADATA,
)
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_transport import (
    paged_attention_schedule_facts,
)
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._compiler.cute.pdl_release import CUTE_PDL_CAPTURE_DEFAULT
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_paged_bounded import bounded_partial_fixture
from helion._testing.cute_paged_key_clip import expected_key_shift
from helion._testing.cute_paged_key_clip import unloaded_keys
from helion._testing.cute_paged_reference import reference_attention_fixture
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("constant", ["query_len", "group", "window"])
def test_attention_declines_promoting_integer_literals(constant: str) -> None:
    args = list(paged_attention_args())
    position = {"query_len": 7, "group": 8, "window": 9}[constant]
    # Keep the wrapped group divisor nonzero, so the test isolates literal
    # promotion rather than an invalid division.
    args[position] = 2**32 + (constant == "group")
    with cpu_render_target():
        bound = cpu_bind(cast("Kernel[object]", paged_online_attention), tuple(args))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            semantic = match_paged_attention(program)
            assert semantic is not None
            assert plan_paged_attention_transport(bound.env, semantic) is not None
            assert paged_attention_schedule_facts(bound.env, semantic) is None


@functools.cache
def _key_clip_bound(fast_math: bool, window: int | None) -> BoundKernel[object]:
    form = "mapped" if window is not None else "per_key"
    kernel, args = reference_attention_fixture(form, 4, True, window=window)
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(
            helion.kernel(
                kernel.fn,
                backend="cute",
                static_shapes=True,
                fast_math=fast_math,
                ignore_warnings=[helion.exc.FastMathUnused],
            ),
            args,
        )


def _paired_config(**options: object) -> helion.Config:
    config = optional_lse_config()
    config.config.update(
        block_sizes=[32], cute_mma_dag_pass_warps="attention_paired", **options
    )
    return config


@pytest.mark.parametrize(
    "fast_math,window,choices",
    [
        # The default clips the window side alone.
        (True, 31, ("leading", "both", "trailing", "off")),
        # A causal-only kernel has no leading keys and keeps whole tiles.
        (True, None, ("off", "trailing")),
        # Zero fill changes NaN propagation: only fast_math licenses it.
        (False, 31, None),
    ],
)
def test_key_clip_follows_fast_math_and_window(
    fast_math: bool, window: int | None, choices: tuple[str, ...] | None
) -> None:
    bound = _key_clip_bound(fast_math, window)
    fragment = bound.config_spec.cute_mma_dag_key_clip
    assert (None if fragment is None else fragment.choices) == choices
    assert (KEY_CLIP in bound.config_spec._flat_fields()) == (choices is not None)
    with cpu_render_target(fake_device_sm_count=148):
        with bound.env:
            normalized = bound.config_spec.normalized_config(_paired_config())
        assert normalized.get(KEY_CLIP) == (None if choices is None else choices[0])
        default_source = bound.to_code(_paired_config())
        assert "paged_attention_paired_body(" in default_source
        assert ("key_clip='leading'" in default_source) == (
            choices is not None and choices[0] == "leading"
        )
        assert default_source.count("key_clip=") == int("key_clip='" in default_source)
        for choice in ("off", "leading", "trailing", "both"):
            config = _paired_config(**{KEY_CLIP: choice})
            # An explicit whole tile is every kernel's identity.
            if choice in (choices or ("off",)):
                source = bound.to_code(config)
                assert (f"key_clip={choice!r}" in source) == (choice != "off")
            else:
                # A clip the kernel offers under no schedule is an error, not
                # a silently whole tile.
                with (
                    bound.env,
                    pytest.raises(helion.exc.InvalidConfig, match=KEY_CLIP),
                ):
                    bound.config_spec.normalized_config(config)
        # Other bodies have no shifted loader. There a clip the kernel
        # offers is not an error: it is pinned to whole tiles with the other
        # paired knobs, so equal kernels keep one identity.
        pins = dict(_PAIRED_KNOB_PINS)
        for choice in choices or ():
            single = _paired_config(
                **{KEY_CLIP: choice, "cute_mma_dag_v_lag": 2, RETIREMENT: True}
            )
            single.config["cute_mma_dag_pass_warps"] = "attention"
            with bound.env:
                normalized = bound.config_spec.normalized_config(single)
            assert normalized.get("cute_mma_dag_pass_warps") == "attention"
            assert {key: normalized.get(key, pins[key]) for key in pins} == pins
            assert normalized.get(KEY_CLIP) == "off"
            if choice == choices[-1]:
                source = bound.to_code(single)
                assert "paged_attention_partial_body(" in source
                assert "key_clip=" not in source


@contextlib.contextmanager
def _value_metadata_spec() -> Iterator[ConfigSpec]:
    """The windowed fast_math kernel's surface with value metadata on offer,
    and the compiler seeds of that surface."""
    bound = _key_clip_bound(True, 31)
    assert bound.host_function is not None
    spec = copy.copy(bound.config_spec)
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    # The metadata proof belongs to the transport; supply its fact alone.
    facts = dataclasses.replace(
        facts,
        attention=dataclasses.replace(
            facts.attention, value_metadata_skip_available=True
        ),
    )
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        enable_cute_mma_dag_search(spec, facts)
        spec.compiler_seed_configs = MmaDagSeedPlanner(
            facts, spec, occupancy_key="cute_min_blocks_per_mp"
        ).configs([])
        yield spec


def test_key_clip_keeps_whole_tiles_for_value_metadata() -> None:
    with _value_metadata_spec() as spec:
        config = _paired_config(**{METADATA: True})
        spec.normalize(config)
        assert config.get(KEY_CLIP) == "off"
        named = _paired_config(**{METADATA: True, KEY_CLIP: "both"})
        with pytest.raises(helion.exc.InvalidConfig, match=KEY_CLIP):
            spec.normalize(named)
        spec.normalize(named, _fix_invalid=True)
        assert named.get(KEY_CLIP) == "off" and named.get(METADATA) is True
        # Without metadata the same surface keeps the leading default.
        plain = _paired_config()
        spec.normalize(plain)
        assert plain.get(KEY_CLIP) == "leading"
        # Promoted over the base default, whose body pins the clip off, each
        # config still takes its own transport's default.
        for options, clip in (({}, "leading"), ({METADATA: True}, "off")):
            promoted = helion.Config.from_dict(
                spec.promoted_default_values(_paired_config(**options).config)
            )
            spec.normalize(promoted)
            assert promoted.get(KEY_CLIP) == clip
            assert promoted.get(METADATA) is bool(options)


# What a paired config says beyond its topology: the schedule's own options
# and the dependent-launch policy, each with the value an absent key has on
# another body.
_PAIRED_OPTIONS: dict[str, object] = {
    **dict(_PAIRED_KNOB_PINS),
    "cute_pdl": False,
    "cute_pdl_capture_overlap": False,
    "cute_pdl_release": "auto",
    "cute_pdl_spread": False,
}


def _paired(config: helion.Config) -> bool:
    return config.get("cute_mma_dag_pass_warps") == "attention_paired"


def _options(config: helion.Config) -> dict[str, object]:
    return {key: config.get(key, absent) for key, absent in _PAIRED_OPTIONS.items()}


def _explicit(
    spec: ConfigSpec, seed: helion.Config, override: dict[str, object]
) -> helion.Config | None:
    """What a config that names ``seed``'s options and ``override``
    normalizes to: as written, or, where the override contradicts a default
    the seed only spells out (the guarded dependent launch, early
    retirement), without that default. None when neither is a config."""
    named = dict(seed.config)
    unspelled = dict(named)
    if all(named.get(key) == value for key, value in CUTE_PDL_CAPTURE_DEFAULT.items()):
        for key in CUTE_PDL_CAPTURE_DEFAULT:
            del unspelled[key]
    if named.get(RETIREMENT) is True:
        del unspelled[RETIREMENT]
    for options in (named, unspelled):
        config = helion.Config.from_dict(copy.deepcopy({**options, **override}))
        try:
            spec.normalize(config)
        except helion.exc.InvalidConfig:
            continue
        return config
    return None


def test_single_overrides_keep_every_paired_seed_they_admit() -> None:
    # A seed is layered under ``autotune_config_overrides`` as a complete
    # config: every option it does not name already holds a default, and the
    # paired schedule derives three of them from other options (early
    # retirement, the guarded dependent launch, the key clip). An override
    # that such a default cannot coexist with must re-derive it, not leave
    # it behind for strict validation to reject: the paired schedule would
    # drop out of the search. Sweep every value of every enumerable option
    # as the only override, on a kernel that has all three defaults.
    with _value_metadata_spec() as spec:
        seeds = [seed for seed in spec.compiler_seed_configs if _paired(seed)]
        assert any(seed.get(METADATA) for seed in seeds)
        assert any(KEY_CLIP in seed.config for seed in seeds)
        assert not any(seed.get(METADATA) and KEY_CLIP in seed.config for seed in seeds)
        plain = ConfigGeneration(spec)
        own = []
        for seed in seeds:
            # Flattened before it is normalized, a seed keeps its meaning,
            # and its row decodes without repair.
            expected = helion.Config.from_dict(copy.deepcopy(seed.config))
            spec.normalize(expected)
            assert _paired(expected)
            assert expected.get(RETIREMENT) is not (seed.get(RETIREMENT) is False)
            assert expected.get(KEY_CLIP) == (
                "off" if seed.get(METADATA) else seed.get(KEY_CLIP, "leading")
            )
            assert {key: expected.get(key) for key in CUTE_PDL_CAPTURE_DEFAULT} == (
                dict(CUTE_PDL_CAPTURE_DEFAULT)
            )
            row = plain.flatten(seed)
            assert _options(plain.strict_unflatten(row)) == _options(expected)
            own.append(_options(expected))
        # The sweep takes one seed per set of named options: seeds that
        # differ only in their cache hints, which no paired option reads,
        # share one.
        distinct = {
            repr(
                sorted(
                    (key, value)
                    for key, value in seed.config.items()
                    if key != "load_eviction_policies"
                )
            ): index
            for index, seed in reversed(list(enumerate(seeds)))
        }
        assert 2 * len(distinct) > len(seeds) > len(distinct)
        overrides = [
            (key, choice)
            for key, field in spec._flat_fields().items()
            for choice in (
                field.choices
                if isinstance(field, EnumFragment)
                else (False, True)
                if isinstance(field, BooleanFragment)
                else ()
            )
        ]
        assert {*_PAIRED_OPTIONS, "cute_mma_dag_pass_warps"} <= {
            key for key, _choice in overrides
        }
        kept: dict[tuple[str, object], int] = {}
        for key, choice in overrides:
            override = {key: choice}
            generation = spec.create_config_generation(overrides=override)
            for index in distinct.values():
                seed, before = seeds[index], own[index]
                try:
                    row, config = generation.canonicalize_flat(generation.flatten(seed))
                except helion.exc.InvalidConfig as error:
                    # Dropped: then no config names this seed's options
                    # together with the override either.
                    explicit = _explicit(spec, seed, override)
                    assert explicit is None or not _paired(explicit), (
                        f"{override} drops a paired seed it admits: {error}"
                    )
                    continue
                if not _paired(config):
                    # Another body pins every paired knob, the clip the seed
                    # carried among them: nothing is left to reject.
                    assert all(
                        config.get(name, pin) == pin for name, pin in _PAIRED_KNOB_PINS
                    )
                    continue
                kept[key, choice] = kept.get((key, choice), 0) + 1
                options = _options(config)
                if options == before:
                    continue
                # The row a search starts from decodes to that config as is.
                assert _options(generation.strict_unflatten(row)) == options
                if options == {**before, **override}:
                    continue
                # Something else moved: only to what a config naming the
                # seed's options and the override derives, and the override
                # itself holds as it does there.
                explicit = _explicit(spec, seed, override)
                if explicit is None:
                    # The seed itself names what the override excludes (a
                    # clip of both sides under value metadata).
                    continue
                expected = _options(explicit)
                assert all(
                    options[name] in (expected[name], before[name])
                    for name in _PAIRED_OPTIONS
                ), (override, options, expected, before)
                if key in _PAIRED_OPTIONS:
                    assert options[key] == expected[key]
        # An override that names a clip keeps the seeds that can run one.
        assert kept[KEY_CLIP, "both"] == sum(
            not seeds[index].get(METADATA) for index in distinct.values()
        )
        # The overrides that contradict a derived default keep every paired
        # seed, cache hints included, each with that default re-derived.
        for (key, choice), policy in {
            ("cute_pdl", False): (False, False, "auto"),
            ("cute_pdl_capture_overlap", False): (False, False, "auto"),
            ("cute_pdl_spread", True): (False, False, "auto"),
            ("cute_pdl_release", "exit"): (False, False, "exit"),
            (METADATA, True): (True, True, "loader_tail"),
            (RETIREMENT, False): (True, True, "loader_tail"),
        }.items():
            assert kept[key, choice] == len(distinct)
            generation = spec.create_config_generation(overrides={key: choice})
            for seed in seeds:
                config = generation.canonicalize_flat(generation.flatten(seed))[1]
                assert _paired(config) and _options(config)[key] == choice
                assert (
                    config.get("cute_pdl", False),
                    config.get("cute_pdl_capture_overlap", False),
                    config.get("cute_pdl_release", "auto"),
                ) == policy
                if key == METADATA:
                    assert config.get(KEY_CLIP) == "off"
        # An entry that turns value metadata on and selects another schedule
        # as well: the transport check precedes that body's pin, so the clip
        # is re-derived there too.
        generation = spec.create_config_generation(
            overrides={METADATA: True, "cute_mma_dag_pass_warps": "attention"}
        )
        for index in distinct.values():
            config = generation.canonicalize_flat(generation.flatten(seeds[index]))[1]
            assert config.get("cute_mma_dag_pass_warps") == "attention"
            assert config.get(KEY_CLIP) == "off"


def test_key_clip_rederivation_cannot_see_an_overridden_option() -> None:
    # The hook that drops a displaced clip runs after the overrides are
    # layered over the candidate, so it cannot see what the candidate said
    # about an option an override names.
    with _value_metadata_spec() as spec:

        def layered(overrides: dict[str, object]) -> helion.Config:
            # An explicit config under the overrides, without the repair of
            # a flat transfer: the overrides, the hooks, strict
            # normalization.
            config = _paired_config(**{METADATA: True, KEY_CLIP: "both"})
            config.config.update(overrides)
            spec.prepare_override_normalization(config.config, overrides)
            spec.normalize(config)
            return config

        # A clip of both sides beside value metadata is an error on its own,
        # and under an override that names neither option.
        for overrides in ({}, {"cute_mma_dag_v_lag": 2}):
            with pytest.raises(helion.exc.InvalidConfig, match=KEY_CLIP):
                layered(overrides)
        # Under the override of the option at fault the candidate's own
        # value is gone: read as the value a shifted clip needs, the config
        # loses its clip like a seed whose clip the override displaced.
        config = layered({METADATA: True})
        assert config.get(METADATA) is True and config.get(KEY_CLIP) == "off"
        # An override of the clip itself is never dropped.
        with pytest.raises(helion.exc.InvalidConfig, match=KEY_CLIP):
            layered({METADATA: True, KEY_CLIP: "both"})


PASS_WARPS = "cute_mma_dag_pass_warps"


@pytest.mark.parametrize("window", [31, None])
def test_a_seed_an_override_puts_on_the_paired_schedule_takes_its_clip(
    window: int | None,
) -> None:
    # A seed of the plain attention schedule is decoded on its own body,
    # which pins every paired knob, the clip to whole tiles. Under an
    # override that selects the paired schedule that pin would read as the
    # seed's choice: every seed the override moves would start a windowed
    # kernel without the clip it defaults to.
    spec = _key_clip_bound(True, window).config_spec
    default = "leading" if window is not None else "off"
    seeds = spec.compiler_seed_configs
    others = [seed for seed in seeds if not _paired(seed)]
    assert others and len(others) < len(seeds)

    def moved(overrides: dict[str, object]) -> list[helion.Config]:
        generation = spec.create_config_generation(overrides=overrides)
        configs = []
        for seed in others:
            try:
                configs.append(
                    generation.canonicalize_flat(generation.flatten(seed))[1]
                )
            except helion.exc.InvalidConfig:
                # A seed the paired schedule rejects for its own options.
                continue
        return [config for config in configs if _paired(config)]

    paired = {PASS_WARPS: "attention_paired"}
    # The override alone, or beside the lowering the schedule needs.
    for overrides in (paired, {**paired, "cute_mma_dag": "tcgen05"}):
        configs = moved(overrides)
        assert configs
        assert {config.get(KEY_CLIP) for config in configs} == {default}
        # The other paired knobs are the body's pins still: only the clip
        # is read as displaced.
        for config in configs:
            assert all(
                config.get(name, pin) == pin
                for name, pin in _PAIRED_KNOB_PINS
                if name != KEY_CLIP
            )
    # An overridden clip is never replaced.
    for clip in ("off", default):
        configs = moved({**paired, KEY_CLIP: clip})
        assert configs and {config.get(KEY_CLIP) for config in configs} == {clip}
    # An override that selects no schedule moves nothing.
    assert not moved({"cute_mma_dag_v_lag": 0})

    # A candidate that is paired on its own keeps the whole tiles it names:
    # the overrides are layered over it, then the hooks run.
    def layered(options: dict[str, object], overrides: dict[str, object]) -> object:
        config = _paired_config(**options)
        config.config.update(overrides)
        spec.prepare_override_normalization(config.config, overrides)
        spec.normalize(config)
        return config.get(KEY_CLIP)

    chosen = {KEY_CLIP: "off", "cute_mma_dag_v_lag": 2}
    assert layered(chosen, paired) == "off"
    # Without a schedule override nothing is re-derived either, whatever
    # the candidate's other knobs hold.
    pinned = {
        name: pin for name, pin in _PAIRED_KNOB_PINS if name in spec._flat_fields()
    }
    assert layered(pinned, {"cute_min_blocks_per_mp": 1}) == "off"
    # Every paired knob at its pin under the schedule override is what a
    # row of another body decodes to, and takes the default.
    assert layered(pinned, paired) == default


def test_promoted_paired_default_keeps_the_default_key_clip() -> None:
    # Partial statistics admit the paired schedule alone, so this kernel's
    # no-config default is a promoted paired seed. It is layered over a base
    # default that runs another body and pins every paired knob for it, the
    # clip to whole tiles.
    kernel, args = bounded_partial_fixture(4)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            helion.kernel(
                kernel.fn, backend="cute", static_shapes=True, fast_math=True
            ),
            args,
        )
        spec = bound.config_spec
        seed = spec.compiler_default_config
        assert seed is not None and KEY_CLIP not in seed.config
        assert seed.get("cute_mma_dag_pass_warps") == "attention_paired"
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert spec.autotune_reference_config().get(KEY_CLIP) == "off"
            default = spec.default_config()
        assert default.get(KEY_CLIP) == "leading"
        # The kernel without a config is the seed's own render.
        source = bound.to_code(default)
        assert "key_clip='leading'" in source
        assert source == bound.to_code(helion.Config.from_dict(dict(seed.config)))


@pytest.mark.parametrize("window", [31, None])
def test_key_clip_seeds_follow_the_old_seed_order(window: int | None) -> None:
    bound = _key_clip_bound(True, window)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        seeds = MmaDagSeedPlanner(
            facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
        ).configs([])
        named = [seed for seed in seeds if KEY_CLIP in seed.config]
        if window is None:
            # The trailing clip alone is a search move only.
            assert not named
            return
        # Earlier seeds take the default leading clip when they normalize.
        assert named == seeds[-len(MMA_DAG_V_LAG_CHOICES) :]
        assert [
            (
                seed.get("cute_mma_dag_pass_warps"),
                seed.get("cute_mma_dag_v_lag", 0),
                seed.get(KEY_CLIP),
            )
            for seed in named
        ] == [("attention_paired", lag, "both") for lag in MMA_DAG_V_LAG_CHOICES]
        for seed in named:
            # Otherwise a paired seed like the rest: early retirement and the
            # guarded dependent launch.
            assert seed.get("cute_mma_dag_early_tmem_retirement") is True
            assert {key: seed.get(key) for key in CUTE_PDL_CAPTURE_DEFAULT} == dict(
                CUTE_PDL_CAPTURE_DEFAULT
            )
            normalized = bound.config_spec.normalized_config(seed)
            assert normalized.get(KEY_CLIP) == "both"
            assert normalized.get("cute_pdl_capture_overlap") is True
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, normalized
            )


# Each axis changes which side of a tile is hidden: the window (none, one key,
# inside a page, wider than a page), the query rows (one position, or five
# with padded columns) and every length across three pages.
@pytest.mark.parametrize("window", [None, 0, 31, 200])
@pytest.mark.parametrize("query_len,query_columns", [(1, 8), (5, 40)])
def test_expected_key_shift_drops_only_keys_no_row_sees(
    window: int | None, query_len: int, query_columns: int
) -> None:
    group = 8
    modes = ("trailing",) if window is None else ("leading", "trailing", "both")
    # A whole tile keeps every key, also where a clip would take all but four.
    assert not expected_key_shift(
        256,
        1,
        query_len=query_len,
        group=group,
        query_columns=query_columns,
        window=window,
        key_clip="off",
    )
    for length in range(300):
        positions = [
            length - query_len + row // group for row in range(0, query_columns, group)
        ]
        begin = max(length - query_len - window, 0) // 128 if window is not None else 0
        # One tile on each side of the source's range is hidden from every row.
        for tile in range(max(begin - 1, 0), (length + 127) // 128 + 1):
            keys = range(tile * 128, tile * 128 + 128)
            before_window = [
                window is not None and all(key < row - window for row in positions)
                for key in keys
            ]
            after_causal = [all(key > row for row in positions) for key in keys]
            visible = [
                any(
                    key <= row and (window is None or key >= row - window)
                    for row in positions
                )
                for key in keys
            ]
            for mode in modes:
                shift = expected_key_shift(
                    tile * 128,
                    length,
                    query_len=query_len,
                    group=group,
                    query_columns=query_columns,
                    window=window,
                    key_clip=mode,
                )
                assert not any(visible[key] for key in unloaded_keys(shift))
                # Scale boxes stay 16-byte aligned and one row stays loaded.
                assert shift % 4 == 0 and abs(shift) <= 124
                leading = sum(before_window) if mode != "trailing" else 0
                trailing = sum(after_causal) if mode != "leading" else 0
                assert abs(shift) == min(max(leading, trailing) & -4, 124)
                assert shift >= 0 or trailing > leading
