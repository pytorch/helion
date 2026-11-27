"""The ordinary PDL entry wait dominates paged attention's scalar prelude."""

from __future__ import annotations

import ast
import copy
import functools
from typing import TYPE_CHECKING
from typing import cast

import pytest

import helion
from helion._compiler.cute import paged_attention_codegen
from helion._compiler.cute.autotune.search_surface import (
    _normalize_cute_pdl_capture_overlap,
)
from helion._compiler.cute.autotune.search_surface import pdl_fields
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.pdl_release import CUTE_PDL_CAPTURE_DEFAULT
from helion._compiler.cute.pdl_release import generic_release_rendered
from helion._compiler.cute.pdl_release import loader_tail_release_rendered
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import ordered_paged_online_attention
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_pdl import WAIT_CALL
from helion._testing.cute_pdl import kernel_definition
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
from helion.language import memory_ops

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(per_key: bool, requests: int = 2) -> BoundKernel[object]:
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(
            cast("Kernel[object]", paged_online_attention),
            (
                *paged_attention_args(rows=8 if per_key else 64, requests=requests),
                None,
                per_key,
            ),
        )


@pytest.mark.parametrize(
    "per_key,paired", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("pdl", [False, True])
def test_paged_attention_uses_generic_entry_pdl(per_key, paired, pdl):
    bound = _bound(per_key)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[8 if per_key else 64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                cute_mma_dag_pass_warps="attention_paired" if paired else "attention",
                cute_min_blocks_per_mp=1,
                cute_pdl=pdl,
            )
        )
        assert config.get("cute_mma_dag_pass_warps") == (
            "attention_paired" if paired else "attention"
        )
        assert config.get("cute_pdl") is pdl
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    kernel = kernel_definition(source)
    wait = WAIT_CALL
    release = "cute.arch.griddepcontrol_launch_dependents()"
    calls = [
        ast.unparse(node) for node in ast.walk(kernel) if isinstance(node, ast.Call)
    ]
    assert ("._helion_cute_use_pdl = True" in source) is pdl
    assert calls.count(wait) == int(pdl)
    assert calls.count(release) == int(pdl)
    assert any("paged_attention" in call and "_body(" in call for call in calls)
    if pdl:
        # This precedes even the scalar tensor loads generated for lengths
        # and bounds, not just the opaque asynchronous attention body call.
        assert ast.unparse(kernel.body[0]) == wait
        assert "capture_pdl_wait_mode" not in source
        assert "capture_pdl_contract" not in source
        assert ast.unparse(kernel.body[1]) == release
        assert any(call.startswith("_helion_mma_dag.ldg_s32(") for call in calls)


@pytest.mark.parametrize(
    "paired,extra,expected,searched",
    [
        # No dependent-launch key: the proved paired kernel's guarded default.
        (True, {}, (True, True, "loader_tail"), None),
        # Naming any dependent-launch option keeps exactly the configured
        # policy, in which an absent ``cute_pdl`` is off: the two switches,
        (True, {"cute_pdl": False}, (False, False, "auto"), (False, False, "auto")),
        (
            True,
            {"cute_pdl_capture_overlap": False},
            (False, False, "auto"),
            (False, False, "auto"),
        ),
        (True, {"cute_pdl": True}, (True, False, "auto"), (True, True, "loader_tail")),
        # a placement (inert without ``cute_pdl``), even the default one
        # written out, and spread padding.
        (
            True,
            {"cute_pdl_release": "entry"},
            (False, False, "entry"),
            (True, True, "entry"),
        ),
        (
            True,
            {"cute_pdl_release": "auto"},
            (False, False, "auto"),
            (True, True, "auto"),
        ),
        (
            True,
            {"cute_pdl_spread": False},
            (False, False, "auto"),
            (True, True, "loader_tail"),
        ),
        # Only the paired schedule has the contract.
        (False, {}, (False, False, "auto"), None),
    ],
)
def test_paired_kernel_defaults_to_guarded_capture_overlap(
    paired, extra, expected, searched
):
    bound = _bound(True)
    assert bound.host_function is not None
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired" if paired else "attention",
        cute_min_blocks_per_mp=1,
        **extra,
    )
    with bound.env, bound.host_function:
        assert bound.config_spec.cute_pdl_capture_overlap_available
        normalized = bound.config_spec.normalized_config(config)
        assert (
            normalized.get("cute_pdl", False),
            normalized.get("cute_pdl_capture_overlap", False),
            normalized.get("cute_pdl_release", "auto"),
        ) == expected
        assert bound.config_spec.normalized_config(normalized) == normalized
        assert bound.env.backend.autotune_config_is_viable(
            bound.config_spec, normalized
        )
        # A seed flattened before it is normalized resolves the default
        # policy the same way; a search that does not sample the policy
        # gives it to every paired config, whatever the seed named.
        generation = ConfigGeneration(bound.config_spec)
        _, round_trip = generation.canonicalize_flat(generation.flatten(config))
        assert (
            round_trip.get("cute_pdl", False),
            round_trip.get("cute_pdl_capture_overlap", False),
            round_trip.get("cute_pdl_release", "auto"),
        ) == ((True, True, "loader_tail") if paired else expected)
        # The same key as a fixed override of that search states that one
        # option. An opt-out declines the policy; any other override leaves
        # the search the default for the options it does not name, where
        # the config that names the key means exactly what it says.
        if extra:
            fixed = ConfigGeneration(bound.config_spec, overrides=extra)
            _, round_trip = fixed.canonicalize_flat(fixed.flatten(config))
            assert (
                round_trip.get("cute_pdl", False),
                round_trip.get("cute_pdl_capture_overlap", False),
                round_trip.get("cute_pdl_release", "auto"),
            ) == searched
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
        assert source == bound.to_code(normalized)
    pdl, overlap, release = expected
    assert ("._helion_cute_use_pdl = True" in source) is pdl
    assert ("capture_pdl_contract" in source) is overlap
    assert ("capture_pdl_wait_mode" in source) is overlap
    assert ("capture_release_tail=True" in source) is (release == "loader_tail")


def test_decline_cannot_see_what_a_candidate_said_of_an_overridden_option():
    # The hook that declines the guarded policy under an override runs
    # after the overrides are layered over the candidate, and a transferred
    # candidate is flat: every option is named.
    bound = _bound(True)
    assert bound.host_function is not None
    spec = bound.config_spec
    paired = {
        "block_sizes": [8],
        "cute_mma_dag": "tcgen05",
        "cute_mma_dag_warps": 4,
        "cute_mma_dag_stages": 2,
        "cute_mma_dag_pass_warps": "attention_paired",
        "cute_min_blocks_per_mp": 1,
    }
    default = {
        "cute_pdl": True,
        "cute_pdl_capture_overlap": True,
        "cute_pdl_release": "loader_tail",
    }

    def policy(config: helion.Config) -> tuple[object, object, object]:
        return (
            config.get("cute_pdl", False),
            config.get("cute_pdl_capture_overlap", False),
            config.get("cute_pdl_release", "auto"),
        )

    def transferred(
        options: dict[str, object], overrides: dict[str, object]
    ) -> tuple[object, object, object]:
        generation = ConfigGeneration(spec, overrides=overrides)
        seed = helion.Config.from_dict({**paired, **options})
        return policy(generation.canonicalize_flat(generation.flatten(seed))[1])

    def layered(
        options: dict[str, object], overrides: dict[str, object]
    ) -> tuple[object, object, object]:
        # An explicit config under the overrides, without the repair of a
        # flat transfer: the overrides, the hooks, strict normalization.
        config = helion.Config.from_dict({**paired, **options, **overrides})
        spec.prepare_override_normalization(config.config, overrides)
        spec.normalize(config)
        return policy(config)

    opt_out = {"cute_pdl_capture_overlap": False}
    with bound.env, bound.host_function:
        # A seed that spells the default policy out is the compiler's seed
        # and the config that names none of the options, slot for slot: its
        # ``cute_pdl=True`` cannot be told from the default's, and the
        # opt-out of the overlap leaves it no dependent launch.
        assert transferred(default, opt_out) == (False, False, "auto")
        assert transferred({}, opt_out) == (False, False, "auto")
        # The overlap on another release is not that default: without the
        # overlap its ``cute_pdl`` and release hold, and stay.
        entry = {**default, "cute_pdl_release": "entry"}
        assert transferred(entry, opt_out) == (True, False, "entry")
        # ``cute_pdl=True`` beside the opt-out keeps ordinary dependent
        # launch for every candidate.
        ordinary = {**opt_out, "cute_pdl": True}
        assert transferred(default, ordinary) == (True, False, "auto")
        assert transferred({}, ordinary) == (True, False, "auto")

        # Capture overlap without ``cute_pdl`` is an error on its own, and
        # under an override that names neither option.
        inconsistent = {**default, "cute_pdl": False}
        for overrides in ({}, {"cute_min_blocks_per_mp": 1}):
            with pytest.raises(
                helion.exc.InvalidConfig, match="cute_pdl_capture_overlap=True"
            ):
                layered(inconsistent, overrides)
        # Under the override of the option at fault the candidate's own
        # value is gone: read as the value the policy needs, the config is
        # declined like a seed that carried the policy.
        assert layered(inconsistent, {"cute_pdl": False}) == (False, False, "auto")


_CAPTURE = {"cute_pdl": True, "cute_pdl_capture_overlap": True}


# Two grids, the largest of one wave and the smallest past it, under every
# policy that reaches the wave test by a rule of its own: the default and
# its spelling, the automatic release with the overlap (named and left out)
# and without it, and the guard a config has to name.
@pytest.mark.parametrize("requests", [148, 150])
@pytest.mark.parametrize(
    "policy",
    [
        # The default policy and the same policy written out.
        {},
        {**_CAPTURE, "cute_pdl_release": "loader_tail"},
        # The overlap on the automatic release, named or left out.
        {**_CAPTURE, "cute_pdl_release": "auto"},
        _CAPTURE,
        # The ordinary dependent launch, whose automatic release applies the
        # same wave test.
        {"cute_pdl": True},
        # The last wave's guard by name, with the overlap and without.
        {**_CAPTURE, "cute_pdl_release": "last_wave"},
        {"cute_pdl": True, "cute_pdl_release": "last_wave"},
    ],
)
def test_paired_release_follows_the_wave_count(requests, policy):
    # One paired CTA holds an SM's tensor memory, so 148 SMs hold 148 CTAs
    # whether or not the config names ``cute_min_blocks_per_mp`` (the thread
    # limit alone would call three 640-thread CTAs per SM resident). The
    # loader's tail releases only while the grid fits that one wave. Past it
    # every CTA releases at entry, under ``auto`` and under ``loader_tail``:
    # a CTA holds its SM, so the dependents launch as the last CTA enters
    # and take SMs only as this grid's CTAs leave them. The guard of the
    # last wave's CTAs, which waits for the first CTAs in particular, is
    # left to the config that names ``last_wave``.
    bound = _bound(True, requests)
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        **policy,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    kernel = kernel_definition(source)
    capture = policy.get("cute_pdl_capture_overlap", not policy) is True
    named = policy.get("cute_pdl_release", "loader_tail" if not policy else "auto")
    one_wave = requests <= 148
    tail = named == "loader_tail" and one_wave
    release = "cute.arch.griddepcontrol_launch_dependents()"
    assert ("capture_pdl_contract" in source) is capture
    assert ("capture_release_tail=True" in source) is tail
    if capture:
        assert ast.unparse(kernel.body[0].test) == "capture_pdl_wait_mode == 0"
    else:
        assert ast.unparse(kernel.body[0]) == WAIT_CALL
    guards = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("pdl_cta >=")
    ]
    if tail:
        # The body owns the tail release.
        assert not guards and release not in ast.unparse(kernel)
    elif named == "last_wave" and not one_wave:
        (guard,) = guards
        assert kernel.body[2] is guard
        assert ast.unparse(guard.test) == f"pdl_cta >= {requests - 148}"
        assert ast.unparse(guard.body[0]) == release
    else:
        # Every CTA releases right behind its wait, in one wave and past it.
        assert not guards and "pdl_cta" not in source
        assert ast.unparse(kernel.body[1]) == release
        assert ast.unparse(kernel).count(release) == 1
    assert generic_release_rendered(source) is not tail


def test_stored_attention_configs_already_name_their_occupancy():
    # The wave test above changes a release only for a config that leaves
    # ``cute_min_blocks_per_mp`` out. A config the autotuner stored is flat,
    # so it names the key, and an attention schedule accepts only the
    # occupancy it launches with: the search's 0 leaves the schedule.
    bound = _bound(True)
    assert bound.host_function is not None
    paired = {
        "block_sizes": [8],
        "cute_mma_dag": "tcgen05",
        "cute_mma_dag_warps": 4,
        "cute_mma_dag_stages": 2,
        "cute_mma_dag_pass_warps": "attention_paired",
        "cute_pdl": True,
    }
    with bound.env, bound.host_function:
        generation = ConfigGeneration(bound.config_spec)
        for occupancy, kept in ((1, True), (0, False)):
            config = helion.Config.from_dict(
                {**paired, "cute_min_blocks_per_mp": occupancy}
            )
            _, stored = generation.canonicalize_flat(generation.flatten(config))
            schedule = stored.get("cute_mma_dag_pass_warps")
            assert (schedule == "attention_paired") is kept
            assert stored.get("cute_min_blocks_per_mp") == occupancy
        with pytest.raises(
            helion.exc.InvalidConfig, match="requires cute_min_blocks_per_mp=1"
        ):
            bound.config_spec.normalized_config(
                helion.Config.from_dict({**paired, "cute_min_blocks_per_mp": 0})
            )


@pytest.mark.parametrize("requests", [75, 148, 150])
@pytest.mark.parametrize("release", [None, "last_wave"])
def test_paired_pdl_counts_physical_ctas(requests, release):
    # Two item/head tiles form each physical CTA. Check below, at and above
    # one resident wave for both automatic and explicit release placement.
    bound = _bound(True, requests)
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=True,
        cute_pdl_release=release,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    module = ast.parse(source)
    (launch,) = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert eval(
        compile(ast.Expression(launch.args[1]), "<paired-grid>", "eval"),
        {"__builtins__": {}},
    ) == (requests,)
    kernel = kernel_definition(source)
    guards = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("pdl_cta >=")
    ]
    if requests <= 148 or release is None:
        # One wave, and the automatic release of a grid past it: one unit
        # per CTA releases at entry in every CTA.
        assert not guards
        assert ast.unparse(kernel.body[1]) == (
            "cute.arch.griddepcontrol_launch_dependents()"
        )
    else:
        (guard,) = guards
        assert ast.unparse(guard.test) == f"pdl_cta >= {requests - 148}"
        assert ast.unparse(guard.body[0]) == (
            "cute.arch.griddepcontrol_launch_dependents()"
        )
    # The wait must still dominate both the release and all input loads.
    assert ast.unparse(kernel.body[0]) == WAIT_CALL
    assert "capture_pdl_wait_mode" not in source


# The other schedules of the same kernel, past one wave under the automatic
# release: 150 requests of two heads are 300 CTAs, one to an SM for the
# per-key and the reduced denominator and two to an SM for the scoped one.
@pytest.mark.parametrize(
    "per_key,schedule,first_released",
    [
        (True, "attention", 152),
        (True, "attention_scoped", 4),
        (False, "attention", 152),
    ],
)
def test_sibling_schedules_keep_the_last_wave_guard(per_key, schedule, first_released):
    # The entry release past one wave is the paired schedule's alone, whose
    # CTA holds an SM for its whole life. Every other schedule keeps the
    # guard that releases from the CTAs of the last wave.
    bound = _bound(per_key, 150)
    config = helion.Config(
        block_sizes=[8 if per_key else 64],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps=schedule,
        cute_pdl=True,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    kernel = kernel_definition(source)
    release = "cute.arch.griddepcontrol_launch_dependents()"
    assert ast.unparse(kernel.body[0]) == WAIT_CALL
    (guard,) = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("pdl_cta")
    ]
    assert ast.unparse(guard.test) == f"pdl_cta >= {first_released}"
    assert ast.unparse(guard.body[0]) == release
    assert ast.unparse(kernel).count(release) == 1


@pytest.mark.parametrize(
    "paired,pdl,overlap,release,active",
    [
        (True, True, True, "auto", True),
        (True, True, True, "entry", True),
        # The guard of the last wave's CTAs by name: plain entry in one wave.
        (True, True, True, "last_wave", True),
        (True, True, False, "auto", False),
        (True, False, True, "auto", False),
        (False, True, True, "auto", False),
        (True, True, True, "exit", False),
        (True, True, True, "loader_tail", True),
        (True, True, False, "loader_tail", False),
        (False, True, True, "loader_tail", False),
        (True, False, True, "loader_tail", False),
    ],
)
def test_capture_overlap_is_an_independent_proved_choice(
    paired, pdl, overlap, release, active
):
    # The ordinary enable/disable gates and the tail's independent overlap,
    # family and launch gates need distinct counterexamples, not a product.
    bound = _bound(True)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        config = helion.Config(
            block_sizes=[8],
            cute_mma_dag="tcgen05",
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
            cute_mma_dag_pass_warps="attention_paired" if paired else "attention",
            cute_min_blocks_per_mp=1,
            cute_pdl=pdl,
            cute_pdl_release=release,
            cute_pdl_capture_overlap=overlap,
        )
        if not active and (overlap or release == "loader_tail"):
            # Explicit configs are strict. Only search repair may discard an
            # inactive choice before the rendered-source checks below.
            with pytest.raises(helion.exc.InvalidConfig, match="capture_overlap"):
                bound.config_spec.normalized_config(config)
            assert config.get("cute_pdl") is pdl
            assert config.get("cute_pdl_release") == release
            assert config.get("cute_pdl_capture_overlap") is overlap
            bound.config_spec.normalize(config, _fix_invalid=True)
        else:
            config = bound.config_spec.normalized_config(config)
        if active and release == "loader_tail":
            generation = ConfigGeneration(bound.config_spec)
            _, round_trip = generation.canonicalize_flat(generation.flatten(config))
            assert round_trip.get("cute_pdl_release") == "loader_tail"
            assert round_trip.get("cute_pdl_capture_overlap") is True
    assert config.get("cute_pdl_capture_overlap", False) is active
    if not active and (not paired or not pdl or release == "exit"):
        assert "cute_pdl_capture_overlap" not in config
    else:
        assert "cute_pdl_capture_overlap" in config
    tail = active and release == "loader_tail"
    if release == "loader_tail":
        assert config.get("cute_pdl_release", "auto") == (
            "loader_tail" if tail else "auto"
        )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    kernel = kernel_definition(source)
    assert ("capture_pdl_contract" in source) is active
    assert ("capture_pdl_parameter_sizes" in source) is active
    assert ("capture_pdl_wait_mode" in source) is active
    assert ("capture_release_tail=True" in source) is tail
    if active:
        assert ast.unparse(kernel.body[0].test) == "capture_pdl_wait_mode == 0"
        assert ast.unparse(kernel.body[0].body[0]) == WAIT_CALL
        # The EARLY anchor finishes waiting before either release location.
        release_call = "cute.arch.griddepcontrol_launch_dependents()"
        if tail:
            assert release_call not in ast.unparse(kernel)
        else:
            assert ast.unparse(kernel.body[1]) == release_call
        assert kernel.args.args[-1].arg == "capture_pdl_wait_mode"
        assert ast.unparse(kernel.args.args[-1].annotation) == "cutlass.Int32"
        (plans,) = [
            ast.literal_eval(node.value)
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "_helion_cute_wrapper_plans"
        ]
        assert plans[0]["capture_pdl_parameter_sizes"] == (128, 128, 128, 128, 4)
        alias = plans[1]
        assert {
            name
            for name, loaded in zip(
                alias["tensor_names"], alias["capture_loaded"], strict=True
            )
            if loaded
        } == {"q", "k", "v", "k_scale", "v_scale", "pages", "lengths"}
        assert {
            name
            for name, stored in zip(alias["tensor_names"], alias["stored"], strict=True)
            if stored
        } == {"out", "lse"}
        assert generic_release_rendered(source) is not tail
        assert loader_tail_release_rendered(source) is tail
        # The body-owned tail release ranks with the generic entry release.
        assert CuteBackend.finalist_preference(config, lambda: source) == 1
        # A conditional wait without this exact audited contract is not ranked
        # as the generic entry release. Nor is its opposite wait-mode branch.
        assert not generic_release_rendered(
            source.replace("paired-output-stores-v1", "unknown-contract")
        )
        assert not generic_release_rendered(
            source.replace("capture_pdl_wait_mode == 0", "capture_pdl_wait_mode != 0")
        )
        # Nor does a tail flag outside the audited contract count as one.
        for other in ("paired-output-stores-v1", "capture_pdl_wait_mode == 0"):
            assert not loader_tail_release_rendered(source.replace(other, "other"))
    elif pdl:
        assert ast.unparse(kernel.body[0]) == WAIT_CALL


def test_capture_overlap_surface_and_unavailable_normalization():
    spec = _bound(True).config_spec
    fields = {}
    with cpu_render_target(fake_device_sm_count=148), _bound(True).env:
        pdl_fields(spec, fields)
    field = fields["cute_pdl_capture_overlap"]
    assert field.choices == (False, True)
    assert field.default() is False
    assert field._active_choices() == (False,)
    assert fields["cute_pdl_release"]._active_choices() == ("auto",)
    assert fields["cute_pdl_release"].default() == "auto"
    # Unmatched and multi-root bindings never receive the availability proof.
    unavailable = ConfigSpec(backend=_bound(True).env.backend)
    for selected in (False, True):
        config = {
            "cute_mma_dag": "tcgen05",
            "cute_mma_dag_pass_warps": "attention_paired",
            "cute_pdl": True,
            "cute_pdl_capture_overlap": selected,
        }
        if selected:
            with pytest.raises(helion.exc.InvalidConfig, match="capture_overlap"):
                _normalize_cute_pdl_capture_overlap(
                    unavailable, config, fix_invalid=False
                )
            assert config["cute_pdl_capture_overlap"] is True
            _normalize_cute_pdl_capture_overlap(unavailable, config, fix_invalid=True)
        else:
            _normalize_cute_pdl_capture_overlap(unavailable, config, fix_invalid=False)
        assert "cute_pdl_capture_overlap" not in config
    with pytest.raises(helion.exc.InvalidConfig, match="must be a boolean"):
        _normalize_cute_pdl_capture_overlap(
            spec, {"cute_pdl_capture_overlap": 1}, fix_invalid=False
        )


def test_capture_overlap_spread_is_rejected_before_generic_pad_insertion():
    bound = _bound(True)
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=True,
        cute_pdl_capture_overlap=True,
        cute_pdl_spread=True,
    )
    # This family already owns a shared-memory arena. Its contract rejects
    # generic padding before the pad insertion pass sees a conditional wait.
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.InvalidConfig, match="no PDL shared-memory padding"),
    ):
        bound.to_code(config)


def test_multiple_roots_do_not_offer_capture_overlap():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(ordered_paged_online_attention(), paged_attention_args())
    assert bound.host_function is not None
    assert len(bound.host_function.device_ir.root_ids) > 1
    assert not bound.config_spec.cute_pdl_capture_overlap_available
    fields = {}
    with cpu_render_target(fake_device_sm_count=148), bound.env:
        pdl_fields(bound.config_spec, fields)
    assert "cute_pdl" in fields
    assert "cute_pdl_capture_overlap" not in fields
    assert fields["cute_pdl_release"]._active_choices() == ("auto",)


def test_a_kernel_without_the_capture_proof_takes_no_guarded_default():
    """A paired config that names no dependent-launch option takes the
    guarded policy only on a kernel that has the capture proof
    (``capture_policy_available``), in normalization and in the default of
    a config flattened before it is normalized. The same kernel with
    explicit dependent-launch markers, which own the wait and the release,
    keeps ``cute_pdl`` off for that config and rejects one that names the
    overlap; a binding without the proof is left the config as it is."""
    bound = _bound(True)
    spec = bound.config_spec
    policy = tuple(CUTE_PDL_CAPTURE_DEFAULT)
    schedule = {
        "cute_mma_dag": "tcgen05",
        "cute_mma_dag_pass_warps": "attention_paired",
    }

    def paired(**named: object) -> helion.Config:
        return helion.Config(
            block_sizes=[8],
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
            cute_min_blocks_per_mp=1,
            **schedule,
            **named,
        )

    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        taken = spec.normalized_config(paired()).config
        assert {key: taken.get(key) for key in policy} == CUTE_PDL_CAPTURE_DEFAULT
        for key, value in CUTE_PDL_CAPTURE_DEFAULT.items():
            assert spec.flatten_missing_field_default(key, schedule) == (True, value)

        marked = copy.copy(spec)
        marked.has_explicit_pdl = True
        declined = marked.normalized_config(paired()).config
        assert declined["cute_mma_dag_pass_warps"] == "attention_paired"
        assert [declined.get(key) for key in policy] == [False, None, None]
        for key in policy:
            assert marked.flatten_missing_field_default(key, schedule) == (False, None)
        with pytest.raises(
            helion.exc.InvalidConfig, match="cute_pdl_capture_overlap=True requires"
        ):
            marked.normalized_config(
                paired(cute_pdl=True, cute_pdl_capture_overlap=True)
            )

        # Unmatched and multi-root bindings never receive the proof.
        unproved = ConfigSpec(backend=bound.env.backend)
        config = dict(schedule)
        _normalize_cute_pdl_capture_overlap(unproved, config, fix_invalid=False)
        assert config == schedule


def test_code_generation_asks_the_capture_rule_normalization_applies(monkeypatch):
    # The emitter keeps no copy of the rule that makes a config eligible for
    # capture overlap: it asks ``carries_capture_overlap``, which reads the
    # rule normalization applies, and adds only its own single-root and
    # effect checks. A rule that turns the config down therefore leaves the
    # body on the ordinary entry wait.
    asked = []

    def decline(available, config):
        asked.append(dict(config))
        return False

    monkeypatch.setattr(paged_attention_codegen, "carries_capture_overlap", decline)
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=True,
        cute_pdl_capture_overlap=True,
        cute_pdl_release="loader_tail",
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = _bound(True).to_code(config)
    assert len(asked) == 1
    assert asked[0]["cute_mma_dag_pass_warps"] == "attention_paired"
    assert asked[0]["cute_pdl_capture_overlap"] is True
    assert ast.unparse(kernel_definition(source).body[0]) == WAIT_CALL
    assert "capture_pdl_wait_mode" not in source
    assert "capture_pdl_contract" not in source


def test_declined_final_capture_effect_proof_restores_automatic_pdl(monkeypatch):
    original = paged_attention_codegen._is_call

    def reject_store_effect(node, target):
        # Simulate an effect proof that no longer recognizes the output store
        # after normalization accepted loader-tail release. The body stays valid.
        return target is not memory_ops.store and original(node, target)

    monkeypatch.setattr(paged_attention_codegen, "_is_call", reject_store_effect)
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=True,
        cute_pdl_capture_overlap=True,
        cute_pdl_release="loader_tail",
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = _bound(True).to_code(config)
    kernel = kernel_definition(source)
    assert ast.unparse(kernel.body[0]) == WAIT_CALL
    assert ast.unparse(kernel.body[1]) == (
        "cute.arch.griddepcontrol_launch_dependents()"
    )
    assert "capture_pdl_wait_mode" not in source
    assert "capture_release_tail=True" not in source
    assert "capture_pdl_contract" not in source
