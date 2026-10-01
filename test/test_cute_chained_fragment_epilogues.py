from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_review import overlap_bound as overlap_bound
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_plain_root import _config as _plain_config
from .test_cute_chained_plain_root import _source as _plain_source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_tcgen05 import _tcgen_chain
from .test_cute_chained_tcgen05 import _tcgen_inputs
from .test_cute_chained_warp_bridge import _escaped_bridge
import helion
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_preparation_fragment as fragments
from helion._compiler.cute import chained_preparation_storage as storage
from helion._compiler.cute import chained_tcgen05 as tcgen
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._compiler.cute import chained_warp_bridge as bridges
from helion._compiler.cute import chained_warp_stage as warp
from helion.autotuner.config_fragment import EnumFragment

KEY = "cute_chained_fragment_epilogues"


def _root(dtype=torch.bfloat16, enabled=True, seed=False):
    with _cpu_codegen():
        args = tuple(
            torch.empty(shape, dtype=dtype) for shape in ((128, 32), (32, 32), (32, 32))
        )
        config = helion.Config(
            num_warps=4, cute_chained_mma_schedule="cp_async_register"
        )
        if enabled is not None:
            config.config[KEY] = enabled
        return _escaped_bridge._bind_isolated((*args, seed)).to_code(config)


def _grouped():
    kernel, args = _kda_fixture()
    config = _kda_config()
    config.config.update(
        cute_chained_scan_producer_retention=True,
        cute_chained_native_vector_reads=True,
        cute_chained_frontier_tile_columns=32,
        cute_chained_frontier_stmatrix=True,
        cute_chained_fragment_epilogues=True,
        cute_chained_compact_preparation=True,
    )
    bindings = []
    original = storage.bind_preparation_storage

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        bindings.append(result)
        return result

    with patch.object(storage, "bind_preparation_storage", observe):
        source = _source(kernel, args, config)
    assert len(bindings) == 1 and bindings[0] is not None
    bound = bindings[0]
    return source, bound.physical.accepted.lines, bound


@pytest.fixture(scope="module")
def completed():
    source, body, bound = _grouped()
    assert bound is not None
    accepted = bound.physical.accepted
    assert len(accepted.fragments) == 1
    fragment = accepted.fragments[0]
    assert fragment.accepted(accepted)
    return source, body, bound, fragment


def test_completed_group_member_has_no_fictitious_shared_publication(completed):
    source, body, bound, fragment = completed
    accepted = bound.physical.accepted
    assert not fragment.keep_shared
    producer = next(a for a in accepted.actions if a.fragment is fragment)
    target = next(a for a in accepted.actions if a.first == fragment.cache_event)
    assert fragment.source not in dict(producer.outputs)
    assert fragment.source_name not in producer.writes
    assert fragment.source_name not in producer.omitted
    assert fragment.source_name not in target.reads
    assert target.writes == (fragment.entry.name,)
    assert any(b.name == fragment.source_name for b in accepted.pipeline.frame.buffers)
    assert accepted.pipeline.frame.layout.region(fragment.source_name).byte_size > 0
    completion = fragment.completed()
    assert isinstance(completion.prepared.result, warp.WarpMemberResult)
    assert fragment._state.publication.owner is completion.prepared
    assert f"{fragment.source_name}[" not in source
    assert f"{fragment.entry.name}_fragment_result_index" in source
    assert body == accepted.lines
    assert bound.matches(accepted.revision.plan, accepted.pipeline)


@pytest.mark.parametrize(
    "change",
    (
        "drop",
        "copy",
        "payload",
        "join",
        "owner",
        "copy_image",
        "image_shape",
        "member",
        "cut",
        "frame",
        "config",
    ),
)
def test_original_completed_member_and_image_cannot_be_replaced(completed, change):
    _, _, bound, fragment = completed
    accepted = bound.physical.accepted
    state, completion = fragment._state, fragment.completed()
    target, field = state, "completion"
    old = state.completion
    value = None
    if change == "copy":
        value = replace(completion)
    elif change in ("payload", "join"):
        target, field, old = completion, "prefix", completion.prefix
        value = old[:-1] if change == "join" else (*old, "unemitted = 0")
    elif change in ("owner", "copy_image", "image_shape"):
        field, old = "publication", state.publication
        assert old is not None
        value = replace(
            old,
            **(
                {"owner": object()}
                if change == "owner"
                else {"shape": (16, 16)}
                if change == "image_shape"
                else {}
            ),
        )
    elif change == "member":
        target, field, old = fragment, "member", fragment.member
        value = (old[0], old[1], old[2] + 1)
    elif change == "cut":
        target, field, old = fragment, "cache_event", fragment.cache_event
        value = old - 1
    elif change == "frame":
        target, field, old = fragment.pipeline, "frame", fragment.pipeline.frame
        value = replace(old)
    elif change == "config":
        config = fragment.cg.device_function.config.config
        old = dict(config)
        try:
            config[KEY] = False
            with pytest.raises(chain._UnsupportedChain, match="fragment"):
                fragment.completed()
        finally:
            config.clear()
            config.update(old)
        assert fragment.accepted(accepted)
        return
    try:
        object.__setattr__(target, field, value)
        with pytest.raises(chain._UnsupportedChain, match="fragment"):
            fragment.completed()
    finally:
        object.__setattr__(target, field, old)
    assert fragment.accepted(accepted)


@pytest.mark.parametrize("change", ("cache_body", "source_body", "boundary", "writes"))
def test_actual_accepted_action_placement_is_not_recertified(completed, change):
    _, _, bound, fragment = completed
    accepted = bound.physical.accepted
    if change.endswith("body"):
        index = (
            fragment._state.cache_first
            if change == "cache_body"
            else fragment._state.stage_first
        )
        assert index is not None
        lines = list(accepted.lines)
        lines[index] = "unemitted = 0"
        altered = replace(accepted, lines=tuple(lines))
    else:
        action = next(a for a in accepted.actions if a.first == fragment.cache_event)
        changed = replace(
            action, **({"outputs": ()} if change == "boundary" else {"writes": ()})
        )
        altered = replace(
            accepted,
            actions=tuple(changed if a is action else a for a in accepted.actions),
        )
    assert not fragment.accepted(altered)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_additive_reader_keeps_complete_shared_c_and_uses_common_epilogue(dtype):
    stages = []
    original = warp.emit_prepared_warp_stage

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        stages.append(args[3])
        return result

    with patch.object(warp, "emit_prepared_warp_stage", observe):
        source = _root(dtype)
    assert len(stages) == 2
    result = stages[0].result
    assert isinstance(result, warp.WarpRegisterResult)
    assert result.shared is not None and result.bridge.keep_shared
    assert result.bridge.epilogue is not None
    assert result.fragment is not None and result.fragment._consumed
    assert "cute.autovec_copy(chain_0_acc, chain_0_sc)" in source
    assert "chain_0_c[chain_store" in source
    assert "chain_1_a_bridge_values" in source
    assert _root(dtype, None) == _root(dtype, False)


@pytest.mark.parametrize("change", ("publication", "shared", "boundary", "epilogue"))
def test_retained_shared_reader_cannot_replace_original_stage_authority(change):
    original = bridges.RootWarpBridgeSequence.emit
    seen = []

    def altered(self, cg, plan, boundaries, staged, scratch, prefix, stage):
        if stage == 1:
            fragment = self._state.fragment
            assert fragment is not None
            seen.append(True)
            if change == "publication":
                state = fragment.prepared.completion._state
                assert state.publication is not None
                state.publication = replace(state.publication)
            elif change == "shared":
                object.__setattr__(fragment.prepared.result, "shared", None)
            elif change == "boundary":
                boundaries[plan.dots[0]] = "unemitted"
            else:
                object.__setattr__(self.bridge.epilogue, "destination", "unemitted")
        return original(self, cg, plan, boundaries, staged, scratch, prefix, stage)

    with (
        patch.object(bridges.RootWarpBridgeSequence, "emit", altered),
        pytest.raises(exc.BackendUnsupported, match="warp"),
    ):
        _root()
    assert seen == [True]


@pytest.mark.parametrize("value", (0, 1, "true", None))
def test_experimental_switch_is_an_exact_boolean(value):
    if value is None:
        # None means absence to the helper, not a valid explicit Config value.
        with _cpu_codegen():
            args = tuple(
                torch.empty(s, dtype=torch.bfloat16)
                for s in ((128, 32), (32, 32), (32, 32))
            )
            bound = _escaped_bridge._bind_isolated((*args, False))
            with pytest.raises(exc.InvalidConfig, match="fragment_epilogues"):
                bound.to_code(helion.Config(**{KEY: None}))
    else:
        with pytest.raises(exc.InvalidConfig, match="fragment_epilogues"):
            _root(enabled=value)


def test_explicit_accumulator_is_not_repaired_into_an_additive_reader():
    assert _root(enabled=None, seed=True) == _root(enabled=False, seed=True)
    original = chain._register_bridges
    accepted = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        accepted.append(result)
        return result

    with (
        patch.object(chain, "_register_bridges", side_effect=observe),
        pytest.raises(
            exc.BackendUnsupported, match="chained matmul failed late validation"
        ),
    ):
        _root(seed=True)
    assert accepted == [{}]


def test_experimental_coordinate_appends_without_seed_or_default_adoption():
    with _cpu_codegen():
        args = tuple(
            torch.empty(shape, dtype=torch.bfloat16)
            for shape in ((128, 32), (32, 32), (32, 32))
        )
        spec = _escaped_bridge._bind_isolated((*args, False)).config_spec
        fields = spec._flat_fields()
        assert tuple(fields)[-1] == KEY
        field = fields[KEY]
        assert isinstance(field, EnumFragment)
        assert field.default() is False and field.search_values() == [False, True]
        default = spec.default_config()
        seeds = tuple(spec.compiler_seed_configs)
        assert KEY not in default.config
        assert all(seed.config.get(KEY, False) is False for seed in seeds)
        generation = spec.create_config_generation()
        pairs = generation.seed_flat_config_pairs()
        with patch.object(
            spec,
            "_flat_fields",
            return_value={k: v for k, v in fields.items() if k != KEY},
        ):
            old = spec.create_config_generation()
            assert generation.flatten(default)[:-1] == old.flatten(default)
            for (flat, config), (old_flat, old_config) in zip(
                pairs, old.seed_flat_config_pairs(), strict=True
            ):
                assert flat[:-1] == old_flat and flat[-1] is False
                assert config == old_config
        assert all(
            a is b for a, b in zip(seeds, spec.compiler_seed_configs, strict=True)
        )
        selected = spec.normalized_config(
            helion.Config.from_dict(
                {
                    "num_warps": 4,
                    "cute_chained_mma_schedule": "cp_async_register",
                    KEY: True,
                }
            )
        )
        assert generation.unflatten(generation.flatten(selected)) == selected
        assert spec.normalized_config(
            helion.Config.from_dict({KEY: False})
        ) == spec.normalized_config(helion.Config())


@pytest.mark.parametrize(
    "where,change",
    (("stage", "join"), ("stage", "payload"), ("cache", "join"), ("cache", "payload")),
)
def test_returned_fragment_segments_cannot_be_recertified(where, change):
    seen = []
    original = (
        stages.emit_stage
        if where == "stage"
        else fragments.PreparationFragment.emit_cache
    )

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if where == "cache" or kwargs.get("member_fragment") is not None:
            seen.append(True)
            result = list(result)
            if change == "join":
                result.pop()
            else:
                result.insert(-1, "unemitted_payload = cutlass.Float32(0)")
        return result

    target = stages if where == "stage" else fragments.PreparationFragment
    method = "emit_stage" if where == "stage" else "emit_cache"
    with (
        patch.object(target, method, changed),
        pytest.raises((exc.InternalError, exc.BackendUnsupported)) as error,
    ):
        _grouped()
    cause = error.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(
        cause, chain._UnsupportedChain if where == "stage" else exc.BackendUnsupported
    )
    assert (
        "grouped stage return changed"
        if where == "stage"
        else "fragment cache returned body or boundary changed"
    ) in str(cause)
    assert seen == [True]


def _check_existing_tail(spec):
    fields = spec._flat_fields()
    assert tuple(fields)[-1] == KEY
    generation = spec.create_config_generation()
    default = spec.default_config()
    pairs = generation.seed_flat_config_pairs()
    with patch.object(
        spec, "_flat_fields", return_value={k: v for k, v in fields.items() if k != KEY}
    ):
        original = spec.create_config_generation()
        assert generation.flatten(default)[:-1] == original.flatten(default)
        for (flat, config), (old, old_config) in zip(
            pairs, original.seed_flat_config_pairs(), strict=True
        ):
            assert flat[:-1] == old and flat[-1] is False
            assert config == old_config


def test_fragment_coordinate_follows_original_async_tail(retention_bound):
    assert (
        "cute_chained_async_vector_store" in retention_bound.config_spec._flat_fields()
    )
    _check_existing_tail(retention_bound.config_spec)


def test_fragment_coordinate_follows_original_legacy_grid_tail(overlap_bound):
    _check_existing_tail(overlap_bound.config_spec)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("rows", (128, 64))
def test_plain_and_tcgen_no_image_explicit_true_rejects(dtype, rows):
    args = tuple(
        torch.empty(shape, dtype=dtype)
        for shape in ((1, 2, rows, 3, 128), (1, 2, 32, 3, 128))
    )
    config = _plain_config()
    config.config["block_sizes"] = [rows, 32]
    absent = _plain_source(args, config)
    config.config[KEY] = False
    assert _plain_source(args, config) == absent
    config.config[KEY] = True
    with pytest.raises(
        exc.BackendUnsupported,
        match="fragment epilogues lack an admitted original result image",
    ) as error:
        _plain_source(args, config)
    assert isinstance(error.value.__cause__, chain._UnsupportedChain)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_actual_tcgen_fragment_remains_admitted_on_normal_route(dtype):
    args = (*_tcgen_inputs("cpu", dtype, n=64), "plain")
    config = helion.Config(
        block_sizes=[128, 64],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
    )
    with _cpu_codegen():
        absent = _tcgen_chain._bind_isolated(args).to_code(config)
        config.config[KEY] = False
        assert _tcgen_chain._bind_isolated(args).to_code(config) == absent
        config.config[KEY] = True
        with patch.object(tcgen, "_bridge", wraps=tcgen._bridge) as original_bridge:
            actual = _tcgen_chain._bind_isolated(args).to_code(config)
    assert actual == absent
    assert original_bridge.call_count == 1
    assert "chain_1_bridge_values" in actual and "OperandSource.TMEM" in actual
