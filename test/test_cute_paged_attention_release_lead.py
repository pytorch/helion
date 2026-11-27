"""Check the lead of the paired loader's tail release: what a config may
name, what it renders, the step the device releases at, and where a search
samples and seeds it."""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
from pathlib import Path
import re
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import pytest

import helion
from helion import exc
from helion._compiler.cute import _paged_attention_paired_pipeline as template
from helion._compiler.cute import cute_mma_dag
from helion._compiler.cute import paged_attention_release
from helion._compiler.cute.autotune.mma_dag_surface import enable_cute_mma_dag_search
from helion._compiler.cute.cute_mma_dag import MMA_DAG_RELEASE_LEAD_CHOICES
from helion._compiler.cute.cute_mma_dag import MMA_DAG_RELEASE_LEAD_KEY
from helion._compiler.cute.cute_mma_dag import paged_attention_release_lead_choices
from helion._compiler.cute.cute_mma_dag import paged_attention_release_lead_renders
from helion._compiler.cute.cute_mma_dag_seeds import RELEASE_LEAD_SEED_PAGES
from helion._compiler.cute.paged_attention_config import (
    paged_attention_release_lead_reason,
)
from helion._compiler.cute.paged_attention_release import MMA_DAG_RELEASE_LEAD_DEFAULT
from helion._testing import cpu_render_target
from helion._testing.cute_paged_stream import CAPTURE_POLICY
from helion._testing.cute_paged_stream import NO_WINDOW
from helion._testing.cute_paged_stream import PAIRED
from helion._testing.cute_paged_stream import SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import jit_host_body
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render
from helion._testing.cute_paged_stream import stream_seeds
from helion._testing.cute_paged_stream import stream_step
from helion._testing.cute_paged_stream import transferred_seeds
from helion._testing.cute_template_source import source_function
from helion.runtime.cute.host_wrappers import HOST_WRAPPER_EMITTERS

pytestmark = pytest.mark.cpu_only
KEY = MMA_DAG_RELEASE_LEAD_KEY
PARTITION = "cute_mma_dag_partition"
RING = "cute_mma_dag_value_ring"
TEMPLATE = Path(inspect.getfile(template))
# Units on 148 SMs: a grid of one wave, on which the tail release is the
# loader's own, and one of more than two.
ONE_WAVE = 128
WAVES = 300


def _default_step(total: int) -> int:
    """The step at which the default lead releases a stream of ``total``
    pages: four pages ahead of its end, or behind the loop, at ``total``,
    for a stream it is longer than."""
    return total - 4 if total >= 4 else total


def test_the_default_lead_renders_the_kernel_without_the_option():
    assert KEY == "cute_mma_dag_release_lead"
    assert MMA_DAG_RELEASE_LEAD_CHOICES == (4, 8, 16)
    assert MMA_DAG_RELEASE_LEAD_DEFAULT == MMA_DAG_RELEASE_LEAD_CHOICES[0] == 4
    # The template's own default is the config's: it names the constant
    # wherever it gives the lead a default or compares it, and no number of
    # its own, so the two cannot come apart.
    assert template.MMA_DAG_RELEASE_LEAD_DEFAULT == MMA_DAG_RELEASE_LEAD_DEFAULT
    # Both bind it by importing it from the one module that states it, which
    # imports nothing and is a source dependency of the kernel's wrapper.
    for module in (template, cute_mma_dag):
        assert module.__file__ is not None
        bound = ast.parse(Path(module.__file__).read_text())
        assert [
            node.module
            for node in bound.body
            if isinstance(node, ast.ImportFrom)
            and any(
                alias.name == "MMA_DAG_RELEASE_LEAD_DEFAULT" for alias in node.names
            )
        ] == ["paged_attention_release"]
        assert not any(
            isinstance(target, ast.Name) and target.id == "MMA_DAG_RELEASE_LEAD_DEFAULT"
            for node in ast.walk(bound)
            if isinstance(node, ast.Assign)
            for target in node.targets
        )
    stated = ast.parse(Path(inspect.getfile(paged_attention_release)).read_text())
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom)) and node.module != "__future__"
        for node in stated.body
    )
    dependencies = HOST_WRAPPER_EMITTERS["helion_paged_attention"].source_dependencies
    assert "_compiler/cute/paged_attention_release.py" in dependencies
    tree = ast.parse(TEMPLATE.read_text())
    defaults = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        arguments = function.args.args
        for argument, default in zip(
            arguments[len(arguments) - len(function.args.defaults) :],
            function.args.defaults,
            strict=True,
        ):
            if argument.arg == "capture_release_lead":
                defaults.append((function.name, ast.unparse(default)))
    assert defaults == [
        (name, "MMA_DAG_RELEASE_LEAD_DEFAULT")
        for name in ("_load_pages", "_load_units", "paged_attention_paired_body")
    ]
    compared = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and ast.unparse(node.left) == "capture_release_lead"
    ]
    assert sorted(compared) == [
        f"capture_release_lead {relation} MMA_DAG_RELEASE_LEAD_DEFAULT"
        for relation in ("!=", "==", ">=")
    ]
    bound = stream_bound(ONE_WAVE)
    plain = stream_render(bound, stream_config())
    assert stream_render(bound, stream_config(**{KEY: 4})) == plain
    keywords = body_keywords(plain)
    assert keywords["capture_release_tail"] == "True"
    assert "capture_release_lead" not in keywords
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        # Four pages and a missing key are one identity.
        named = spec.normalized_config(stream_config(**{KEY: 4}))
        assert KEY not in named.config
        assert named == spec.normalized_config(stream_config())
        assert KEY not in spec.default_config().config


@pytest.mark.parametrize("lead", MMA_DAG_RELEASE_LEAD_CHOICES[1:])
def test_a_lead_renders_on_the_loaders_own_release_alone(lead):
    bound = stream_bound(ONE_WAVE)
    plain = stream_render(bound, stream_config())
    source = stream_render(bound, stream_config(**{KEY: lead}))
    keywords = body_keywords(source)
    assert keywords["capture_release_tail"] == "True"
    assert keywords["capture_release_lead"] == str(lead)
    # Nothing else of the kernel changes.
    assert source.replace(f", capture_release_lead={lead}", "") == plain
    # A persistent partition is one wave of CTAs: the lead is that of each
    # CTA's whole stream.
    whole = body_keywords(
        stream_render(bound, stream_config(**{KEY: lead, PARTITION: "whole"}))
    )
    assert whole["partition"] == "'whole'"
    assert whole["capture_release_lead"] == str(lead)
    # Past one wave a ``loader_tail`` config renders the automatic release,
    # and with it the kernel it renders without a lead: the config keeps the
    # lead as it keeps the release's name.
    waves = stream_bound(WAVES)
    with cpu_render_target(fake_device_sm_count=SMS), waves.env, waves.host_function:
        kept = waves.config_spec.normalized_config(stream_config(**{KEY: lead}))
        assert kept.config[KEY] == lead
        assert kept.config["cute_pdl_release"] == "loader_tail"
    released = stream_render(waves, stream_config(**{KEY: lead}))
    assert released == stream_render(waves, stream_config())
    assert "capture_release" not in released


def test_a_stream_shorter_than_the_lead_releases_at_its_first_page():
    step = jit_host_body(template._tail_release_step)
    for lead in MMA_DAG_RELEASE_LEAD_CHOICES[1:]:
        for total in range(3 * lead):
            release = int(step(cutlass.Int64(total), lead))
            # ``lead`` pages ahead of the end, and at the first page of a
            # stream that has fewer: the bare difference is negative there
            # and no step would meet it, so the release would wait for the
            # end of the stream.
            assert release == max(total - lead, 0), (lead, total)
            # Never later than the default lead releases the same stream,
            # and at a step the stream has, unless it has none.
            assert release <= _default_step(total), (lead, total)
            assert total == 0 or 0 <= release < total, (lead, total)
    # The loader takes that step for every lead but the default, whose test
    # of the bare difference is the kernel's of old, in both V-lag forms.
    pages = source_function("_load_pages")
    (assigned,) = [
        node
        for node in ast.walk(pages)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "_tail_release_step"
    ]
    assert ast.unparse(assigned) == (
        "release_step = _tail_release_step(total, capture_release_lead)"
    )
    guarded = [
        node
        for node in ast.walk(pages)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(capture_release_tail)"
    ]
    assert [ast.unparse(node.body[0]) for node in guarded] == [
        "_release_at_tail(g0 + step, total, release_step)"
    ] * 2
    # No release of its own beside the helper's, which both forms call.
    assert "griddepcontrol_launch_dependents" not in ast.unparse(pages)
    release = source_function("_release_at_tail")
    (default,) = release.body[1:]
    assert isinstance(default, ast.If)
    assert ast.unparse(default.test) == "cutlass.const_expr(release_step is None)"
    released = "    cute.arch.griddepcontrol_launch_dependents()"
    assert [ast.unparse(branch[0]) for branch in (default.body, default.orelse)] == [
        f"if stream_step == total - MMA_DAG_RELEASE_LEAD_DEFAULT:\n{released}",
        f"if stream_step == release_step:\n{released}",
    ]
    # Every caller hands the lead on: the one-unit body and a persistent
    # CTA's walk, whose stream is all of its units.
    body = source_function("paged_attention_paired_body")
    units = source_function("_load_units")
    for caller, callee in (
        (body, "_load_pages"),
        (body, "_load_units"),
        (units, "_load_pages"),
    ):
        (call,) = [
            node
            for node in ast.walk(caller)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
        ]
        passed = dict(
            zip(
                [argument.arg for argument in source_function(callee).args.args],
                [ast.unparse(argument) for argument in call.args],
                strict=False,
            )
        )
        assert passed["capture_release_tail"] == "capture_release_tail"
        assert passed["capture_release_lead"] == "capture_release_lead"


def test_a_host_body_is_the_sources_code_whatever_a_trace_left_in_the_function():
    # A kernel's trace replaces a jit function's code with the DSL's
    # rewritten form. The host body is the code the module's source compiles
    # to: the same before and after a trace, and with nothing the DSL keeps
    # of the function's first code (it has kept it as ``_original_code``).
    function = template._tail_release_step
    plain = function.__wrapped__
    code = plain.__code__
    kept = plain.__dict__.pop("_original_code", None)
    try:
        plain.__code__ = (lambda total, release_lead: "rewritten").__code__
        assert plain(10, 8) == "rewritten"
        host = jit_host_body(function)
        assert int(host(cutlass.Int64(10), 8)) == 2
        assert host.__code__.co_name == "_tail_release_step"
        assert host.__code__.co_filename == str(TEMPLATE)
        assert host.__defaults__ == plain.__defaults__
        assert host.__globals__ is plain.__globals__
    finally:
        plain.__code__ = code
        if kept is not None:
            plain._original_code = kept
    # A name the module's source does not define at its top level has no
    # body here, rather than whatever a function object holds.
    renamed = SimpleNamespace(
        __wrapped__=SimpleNamespace(__module__=plain.__module__, __name__="_no_such")
    )
    with pytest.raises(KeyError):
        jit_host_body(renamed)


def test_the_release_helper_releases_once_at_the_step_of_its_lead(monkeypatch):
    # The helper on the host, over every step of a stream: the default lead
    # (no step of its own) releases four pages ahead of the end and never in
    # a stream shorter than that; any other lead releases at its clamped
    # step, so in every stream that has a page.
    released = []
    monkeypatch.setattr(
        cute.arch,
        "griddepcontrol_launch_dependents",
        lambda: released.append("released"),
    )
    release = jit_host_body(template._release_at_tail)
    step_of = jit_host_body(template._tail_release_step)
    for total in range(40):
        for lead in MMA_DAG_RELEASE_LEAD_CHOICES:
            release_step = None if lead == 4 else int(step_of(total, lead))
            at = []
            for at_step in range(total):
                released.clear()
                release(at_step, total, release_step)
                at += [at_step] * len(released)
            if lead == 4:
                assert at == ([total - 4] if total >= 4 else []), (lead, total)
            else:
                assert at == ([max(total - lead, 0)] if total else []), (lead, total)


def test_a_lead_needs_the_paired_schedule_and_the_tail_release():
    bound = stream_bound(ONE_WAVE)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    legal = {KEY: 16, "cute_mma_dag_pass_warps": PAIRED}
    assert paged_attention_release_lead_reason(facts, legal) is None
    assert paged_attention_release_lead_reason(facts, {}) is None
    assert paged_attention_release_lead_reason(facts, {KEY: 4}) is None
    for value in (2, 12, 32, 0, -8, True, "8", 8.0):
        reason = paged_attention_release_lead_reason(facts, {**legal, KEY: value})
        assert reason is not None and "(4, 8, 16)" in reason
    other = paged_attention_release_lead_reason(
        facts, {**legal, "cute_mma_dag_pass_warps": "attention"}
    )
    assert other is not None and other.startswith(f"{KEY} must be one of (4,) for")
    assert paged_attention_release_lead_choices(facts.attention, PAIRED) == (4, 8, 16)
    assert paged_attention_release_lead_choices(
        dataclasses.replace(facts.attention, paired_items=False), PAIRED
    ) == (4,)
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        kept = spec.normalized_config(stream_config(**{KEY: 16}))
        assert kept.config[KEY] == 16
        assert {key: kept.config[key] for key in CAPTURE_POLICY} == CAPTURE_POLICY
        # A value no schedule offers, another schedule, and every
        # dependent-launch policy that releases elsewhere: each is rejected,
        # and repaired to the same config without the lead.
        for lead, change in (
            (12, {}),
            (
                8,
                {
                    "cute_mma_dag_pass_warps": "attention",
                    "cute_mma_dag_query_tma": False,
                },
            ),
            (8, {"cute_pdl": False}),
            (8, {"cute_pdl": True}),
            (8, {"cute_pdl": True, "cute_pdl_capture_overlap": True}),
            (
                16,
                {
                    "cute_pdl": True,
                    "cute_pdl_capture_overlap": True,
                    "cute_pdl_release": "entry",
                },
            ),
            (16, {"cute_pdl": True, "cute_pdl_release": "last_wave"}),
        ):
            config = stream_config(**{KEY: lead}, **change)
            with pytest.raises(exc.InvalidConfig, match=re.escape(KEY)):
                spec.normalized_config(config)
            spec.normalize(config, _fix_invalid=True)
            assert KEY not in config.config
            assert config == spec.normalized_config(stream_config(**change))
        # The lead is one of the paired schedule's knobs: a config whose
        # schedule is decided by a mode that is off keeps the identity of
        # the config without it.
        off = spec.normalized_config(stream_config(**{KEY: 8}, cute_mma_dag="off"))
        assert KEY not in off.config
        assert off == spec.normalized_config(stream_config(cute_mma_dag="off"))
        # A kernel that does not take the guarded policy never releases from
        # its loader's tail, so no config of it can keep a lead.
        marked = copy.copy(spec)
        marked.has_explicit_pdl = True
        with pytest.raises(exc.InvalidConfig, match="loader's tail"):
            marked.normalized_config(stream_config(**{KEY: 8}, cute_pdl=False))


def test_the_lead_is_searched_and_seeded_only_where_the_release_is():
    eager = stream_bound(ONE_WAVE, window=NO_WINDOW)
    fragment = eager.config_spec._flat_fields()[KEY]
    assert fragment is eager.config_spec.cute_mma_dag_release_lead
    # Four pages lead and are all a search without bank cycles draws; every
    # search that draws the knob can draw them.
    assert fragment.choices == (4, 8, 16) and fragment.default() == 4
    assert fragment._active_choices() == (4,)
    assert not eager.config_spec.cute_pdl_capture_overlap_searched
    banks = stream_bound(ONE_WAVE, window=NO_WINDOW, input_banks=2)
    assert banks.config_spec.cute_pdl_capture_overlap_searched
    searched = banks.config_spec._flat_fields()[KEY]
    assert searched.choices == searched._active_choices() == (4, 8, 16)
    # No seed names the lead where it is not searched, and nothing the
    # autotune-off default.
    assert not any(KEY in seed.config for seed in stream_seeds(eager))
    assert RELEASE_LEAD_SEED_PAGES == 16
    seeds = stream_seeds(banks)
    head = seeds[0]
    assert head.get("cute_mma_dag_pass_warps") == PAIRED and KEY not in head.config
    # The leading paired family once more with the longer lead, behind every
    # other seed, and its counterpart on a deeper raw V ring behind it
    # (test_cute_paged_attention_value_ring.py).
    named = [seed for seed in seeds if KEY in seed.config]
    assert named == seeds[-2:]
    assert named[0].config == {**head.config, KEY: 16}
    assert named[1].config[KEY] == 16 and named[1].config[RING] == 3
    unsearched = copy.copy(banks.config_spec)
    unsearched.cute_pdl_capture_overlap_searched = False
    assert not any(KEY in seed.config for seed in stream_seeds(banks, unsearched))
    # A window's few pages lose to a longer lead, and past one wave the
    # release is not the loader's: no seed names a lead there, searched or
    # not, and the field stays.
    for other in (
        stream_bound(ONE_WAVE, input_banks=2),
        stream_bound(WAVES, window=NO_WINDOW, input_banks=2),
    ):
        assert other.config_spec._flat_fields()[KEY]._active_choices() == (4, 8, 16)
        assert not any(KEY in seed.config for seed in stream_seeds(other))
    facts = banks.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.one_wave_units and not facts.attention.windowed
    for bound in (eager, banks):
        spec = bound.config_spec
        with (
            cpu_render_target(fake_device_sm_count=SMS),
            bound.env,
            bound.host_function,
        ):
            assert KEY not in spec.default_config().config
            # A seed is a complete legal config, and a search's row keeps a
            # lead, searched or not.
            generation = spec.create_config_generation()
            normalized = spec.normalized_config(named[0])
            assert normalized.config[KEY] == 16
            _flat, transferred = generation.canonicalize_flat(
                generation.flatten(normalized)
            )
            assert transferred.config[KEY] == 16
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        leads = [
            config.config[KEY]
            for _flat, config in banks.config_spec.create_config_generation().seed_flat_config_pairs()
            if KEY in config.config
        ]
    assert leads == [16, 16]
    # A kernel that does not take the guarded policy has no field and no
    # seed: no config of it releases from its loader's tail.
    marked = copy.copy(banks.config_spec)
    marked.has_explicit_pdl = True
    marked.cute_pdl_capture_overlap_searched = False
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        enable_cute_mma_dag_search(marked, facts)
    assert marked.cute_mma_dag_release_lead is None
    assert KEY not in marked._flat_fields()
    assert not any(KEY in seed.config for seed in stream_seeds(banks, marked))


def test_a_step_onto_a_lead_takes_the_tail_release():
    bound = stream_bound(ONE_WAVE, window=NO_WINDOW, input_banks=2)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        plain = stream_config()
        automatic = stream_config(
            cute_pdl=True, cute_pdl_capture_overlap=True, cute_pdl_release="auto"
        )
        unlaunched = stream_config(cute_pdl=False)
        for parent in (plain, automatic, unlaunched):
            reached, alone = stream_step(spec, parent, **{KEY: 16})
            assert reached.config[KEY] == 16
            assert {key: reached.config[key] for key in CAPTURE_POLICY} == (
                CAPTURE_POLICY
            )
            assert reached.config["cute_mma_dag_v_lag"] == 2
            # Alone the step is an alias of where it started wherever the
            # parent releases elsewhere: normalization repairs the lead to
            # the release's own and changes nothing else.
            if parent is plain:
                assert alone == reached
            else:
                assert KEY not in alone.config
                _reached, unmoved = stream_step(spec, parent)
                assert alone == unmoved
        # A step back to four pages, and one between leads, is the knob's
        # own: nothing else moves.
        on = spec.normalized_config(stream_config(**{KEY: 16}))
        for lead in (4, 8):
            reached, alone = stream_step(spec, on, **{KEY: lead})
            assert reached == alone
            moved = spec.normalized_config(reached)
            assert moved.config.get(KEY, 4) == lead
            assert all(
                moved.config[key] == value
                for key, value in on.config.items()
                if key != KEY
            )
        # A step off the tail release leaves the lead behind.
        for move in (
            {"cute_pdl_release": "auto"},
            {"cute_pdl_capture_overlap": False},
            {"cute_pdl": False},
        ):
            reached, _alone = stream_step(spec, on, **move)
            moved = spec.normalized_config(reached)
            assert KEY not in moved.config, move
            assert moved.config.get("cute_pdl_release", "auto") != "loader_tail"


def test_a_search_does_not_sample_a_lead_that_renders_nothing():
    """Past one wave a ``loader_tail`` config with one unit per CTA renders
    the automatic release whatever its lead. The contract keeps such a lead
    (above); a search's rows of those configs are one row, so a draw or a
    step onto a lead there is an alias of a config the search has and takes
    no slot. A persistent partition of the same kernel is one wave of CTAs:
    there the lead renders and is sampled."""
    one = stream_bound(ONE_WAVE, window=NO_WINDOW, input_banks=2).config_spec
    attention = one.cute_mma_dag_facts.attention
    assert attention.one_wave_units and not attention.past_one_wave_units
    partitions = ("item", "whole", "balanced")
    assert attention.partition_choices == partitions
    assert all(paged_attention_release_lead_renders(attention, p) for p in partitions)
    assert not paged_attention_release_lead_renders(None, "item")
    waves = stream_bound(WAVES, window=NO_WINDOW, input_banks=2)
    spec = waves.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    attention = facts.attention
    assert attention.past_one_wave_units and not attention.one_wave_units
    assert attention.partition_choices == partitions
    assert [paged_attention_release_lead_renders(attention, p) for p in partitions] == [
        False,
        True,
        True,
    ]
    # The two facts at their bound: a grid of as many units as the device
    # has SMs is one wave, and one unit more is past it.
    for units, expected in ((SMS, (True, False)), (SMS + 1, (False, True))):
        edge = stream_bound(units, window=NO_WINDOW, input_banks=2).config_spec
        edge = edge.cute_mma_dag_facts.attention
        assert (edge.one_wave_units, edge.past_one_wave_units) == expected, units
    # A grid on a device of unknown SM count is known to be neither: its
    # ``loader_tail`` release applies, and a lead renders.
    unknown = dataclasses.replace(attention, past_one_wave_units=False)
    assert paged_attention_release_lead_renders(unknown, "item")
    # The bind of such a device (one SM stands for an unknown count) says
    # so itself, and its rows of one unit per CTA keep their lead.
    blind = stream_bound(WAVES, sms=1, window=NO_WINDOW, input_banks=2)
    blind_attention = blind.config_spec.cute_mma_dag_facts.attention
    assert not blind_attention.one_wave_units
    assert not blind_attention.past_one_wave_units
    assert paged_attention_release_lead_renders(blind_attention, "item")
    with cpu_render_target(fake_device_sm_count=1), blind.env, blind.host_function:
        generation = blind.config_spec.create_config_generation()
        kept = generation.canonicalize_flat(
            generation.flatten(stream_config(**{KEY: 16}))
        )[1]
        assert kept.config[KEY] == 16
        assert kept.config.get(PARTITION, "item") == "item"
    automatic = {
        "cute_pdl": True,
        "cute_pdl_capture_overlap": True,
        "cute_pdl_release": "auto",
    }
    with cpu_render_target(fake_device_sm_count=SMS), waves.env, waves.host_function:
        # The field keeps its choices: the persistent partitions render them.
        assert spec._flat_fields()[KEY]._active_choices() == (4, 8, 16)
        generation = spec.create_config_generation()

        def row(config: helion.Config) -> helion.Config:
            return generation.canonicalize_flat(generation.flatten(config))[1]

        for lead in MMA_DAG_RELEASE_LEAD_CHOICES[1:]:
            named = stream_config(**{KEY: lead})
            assert spec.normalized_config(named).config[KEY] == lead
            # One unit per CTA: the search's row is the config's without
            # the lead, and a step onto the lead is where it started, from
            # the tail release and from another policy, which it does not
            # leave for a release that would render the same kernel.
            assert KEY not in row(named).config
            assert row(named) == row(stream_config())
            for parent in (stream_config(), stream_config(**automatic)):
                reached, alone = stream_step(spec, parent, **{KEY: lead})
                _same, unmoved = stream_step(spec, parent)
                assert reached == alone == unmoved, (lead, parent)
            for partition in partitions[1:]:
                # A persistent partition: the row keeps the lead, a step
                # onto it takes the release it belongs to, and a step back
                # to one unit per CTA leaves it behind.
                persistent = stream_config(**{KEY: lead, PARTITION: partition})
                assert row(persistent).config[KEY] == lead
                reached, _alone = stream_step(
                    spec,
                    stream_config(**{PARTITION: partition}, **automatic),
                    **{KEY: lead},
                )
                assert reached.config[KEY] == lead
                assert reached.config[PARTITION] == partition
                assert {key: reached.config[key] for key in CAPTURE_POLICY} == (
                    CAPTURE_POLICY
                )
                reached, _alone = stream_step(spec, persistent, **{PARTITION: "item"})
                assert KEY not in reached.config
                assert reached.config.get(PARTITION, "item") == "item"
                # A draw that moves the partition and the lead together is
                # a config whose lead renders.
                reached, _alone = stream_step(
                    spec, stream_config(), **{KEY: lead, PARTITION: partition}
                )
                assert reached.config[KEY] == lead
                assert reached.config[PARTITION] == partition
                # From another release the same draw takes the tail release
                # too: the lead's step has a test of its own, which does not
                # leave a step that moves the partition to the partition, as
                # the other paired options' steps do.
                reached, _alone = stream_step(
                    spec,
                    stream_config(**automatic),
                    **{KEY: lead, PARTITION: partition},
                )
                assert reached.config[KEY] == lead
                assert reached.config[PARTITION] == partition
                assert {key: reached.config[key] for key in CAPTURE_POLICY} == (
                    CAPTURE_POLICY
                )
                # What counts is the partition a row decodes to, not the
                # one its slot names: a draw that moves the lead and takes
                # the query map from a persistent parent decodes to one
                # unit per CTA (the partition needs the map), and its lead
                # goes with the release that is no longer the loader's.
                reached, _alone = stream_step(
                    spec,
                    stream_config(**{PARTITION: partition}),
                    **{KEY: lead, "cute_mma_dag_query_tma": False},
                )
                assert reached.config.get(PARTITION, "item") == "item"
                assert reached.config["cute_mma_dag_query_tma"] is False
                assert KEY not in reached.config
        # A fixed override keeps what it names, and an overridden partition
        # is the one that says whether a lead renders.
        forced = spec.create_config_generation(overrides={KEY: 16})
        kept = forced.canonicalize_flat(forced.flatten(stream_config()))[1]
        assert kept.config[KEY] == 16 and kept.config.get(PARTITION, "item") == "item"
        whole = spec.create_config_generation(overrides={PARTITION: "whole"})
        kept = whole.canonicalize_flat(whole.flatten(stream_config(**{KEY: 16})))[1]
        assert kept.config[KEY] == 16 and kept.config[PARTITION] == "whole"
        single = spec.create_config_generation(overrides={PARTITION: "item"})
        dropped = single.canonicalize_flat(
            single.flatten(stream_config(**{KEY: 16, PARTITION: "whole"}))
        )[1]
        assert KEY not in dropped.config
        # A kernel past one wave without a persistent partition renders no
        # lead in any config: its search draws the default alone.
        narrowed = copy.copy(spec)
        enable_cute_mma_dag_search(
            narrowed,
            dataclasses.replace(
                facts,
                attention=dataclasses.replace(attention, partition_choices=("item",)),
            ),
        )
        assert narrowed.cute_mma_dag_release_lead is not None
        assert narrowed.cute_mma_dag_release_lead.choices == (4, 8, 16)
        assert narrowed.cute_mma_dag_release_lead._active_choices() == (4,)
    # On one wave every row keeps its lead, with one unit per CTA as well.
    banks = stream_bound(ONE_WAVE, window=NO_WINDOW, input_banks=2)
    with cpu_render_target(fake_device_sm_count=SMS), banks.env, banks.host_function:
        generation = banks.config_spec.create_config_generation()
        for partition in partitions:
            config = stream_config(**{KEY: 16, PARTITION: partition})
            kept = generation.canonicalize_flat(generation.flatten(config))[1]
            assert kept.config[KEY] == 16


def test_overrides_keep_the_paired_seeds_with_or_without_their_lead():
    bound = stream_bound(ONE_WAVE, window=NO_WINDOW, input_banks=2)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        seeded, failures = transferred_seeds(spec, None)
        assert not failures
        paired = [
            seed for seed in seeded if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert [seed.config[KEY] for seed in seeded if KEY in seed.config] == [16, 16]
        # A fixed lead is layered over complete candidates: every paired one
        # takes the release it belongs to and keeps its place, the seeds of
        # the other dependent-launch policies among them, which fold into
        # the seeds that had it; a candidate on another schedule is left to
        # that body's rule, which has no such release.
        forced, failures = transferred_seeds(spec, {KEY: 8})
        kept = [
            seed for seed in forced if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert kept and forced[0] == kept[0] and len(forced) == len(kept)
        assert all(seed.config[KEY] == 8 for seed in kept)
        assert all(
            {key: seed.config[key] for key in CAPTURE_POLICY} == CAPTURE_POLICY
            for seed in kept
        )
        assert len(failures) == len(seeded) - len(paired)
        assert all(KEY in failure for failure in failures)
        # An override that takes away the release a seed's own lead belongs
        # to takes its lead, and the seed stays in the search.
        for overrides, drops in (
            ({"cute_pdl": False}, True),
            ({"cute_pdl_capture_overlap": False}, True),
            ({"cute_pdl_release": "entry"}, True),
            ({"cute_pdl_release": "auto"}, True),
            ({"cute_pdl_spread": True}, True),
            ({"cute_pdl": True}, False),
            ({"cute_mma_dag_v_lag": 1}, False),
        ):
            moved, failures = transferred_seeds(spec, overrides)
            assert not failures, overrides
            leads = [seed.config[KEY] for seed in moved if KEY in seed.config]
            assert leads == ([] if drops else [16, 16]), overrides
            still = [
                seed for seed in moved if seed.get("cute_mma_dag_pass_warps") == PAIRED
            ]
            assert moved[0] == still[0]
        # An override that names both a lead and a release it does not
        # belong to is never repaired: the paired candidates are rejected.
        moved, failures = transferred_seeds(
            spec, {KEY: 8, "cute_pdl_capture_overlap": False}
        )
        assert not [
            seed for seed in moved if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert sum(KEY in failure for failure in failures) >= len(paired)


def test_a_selected_config_loses_its_lead_with_a_declined_policy():
    """The kernel also selects configs without a search (an autotune cache
    hit, AOT tuning data), and overrides that decline the guarded capture
    policy decline it for such a config too: its lead goes with the release
    it belongs to, as in a search candidate, and the config stays legal."""
    bound = stream_bound(ONE_WAVE, window=NO_WINDOW)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        selected = spec.normalized_config(stream_config(**{KEY: 16}))
        plain = spec.normalized_config(stream_config())
        for overrides in (
            {"cute_pdl": False},
            {"cute_pdl_capture_overlap": False},
        ):
            compiled = spec.unsearched_config(selected, overrides)
            assert compiled is not selected
            declined = spec.normalized_config(spec.unsearched_config(plain, overrides))
            assert declined.config["cute_pdl"] is False
            assert spec.normalized_config(compiled).config == declined.config
        # Overrides the policy keeps act on a search only, the knob's own
        # among them, and so does another schedule.
        for overrides in (
            {"cute_pdl": True},
            {KEY: 4},
            {KEY: 8},
            {"cute_mma_dag_pass_warps": "attention"},
        ):
            assert spec.unsearched_config(selected, overrides) is selected


def test_the_seed_is_a_config_of_its_own():
    # What the lead seed adds to the leading seed renders, on a grid of one
    # wave, and is one config after normalization.
    bound = stream_bound(ONE_WAVE, window=NO_WINDOW, input_banks=2)
    seed = stream_seeds(bound)[-2]
    assert seed.config[KEY] == 16 and RING not in seed.config
    assert (
        body_keywords(stream_render(bound, helion.Config.from_dict(seed.config)))[
            "capture_release_lead"
        ]
        == "16"
    )
