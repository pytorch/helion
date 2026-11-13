"""Storage-proof contracts on actual CPU bindings; no device code executes."""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache
from types import SimpleNamespace

import pytest
import torch

import helion
from helion._compiler import runtime_tensor_guards as guards
from helion._compiler.cute.fragment_dot import FragmentDotEmitter
from helion._compiler.cute.register_loads import host_load_is_readonly
from helion._compiler.cute.selection.topk import CuteTopKPlan
from helion._compiler.cute.selection.topk import topk_tensors_are_proven_disjoint
from helion._compiler.device_ir import RootGraphInfo
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl
from helion.language._tracing_ops import _host_tensor
from helion.language.memory_ops import load
from helion.language.memory_ops import store

pytestmark = pytest.mark.cpu_only


def _opaque_replace(left, right):
    left.data = right
    return left


def _opaque_restore(left, right):
    original = left.data
    left.data = right
    selected = left[:, :]
    left.data = original
    return selected


def _input_pair(left: torch.Tensor, right: torch.Tensor, mode: hl.constexpr):
    if mode == "native_set":
        left.set_(right)
        selected = left
    elif mode == "native_restore":
        original = left.data
        left.set_(right)
        selected = left[:, :]
        left.set_(original)
    elif mode == "opaque_replace":
        selected = _opaque_replace(left, right)
    elif mode == "opaque_restore":
        selected = _opaque_restore(left, right)
    else:
        selected = left
    for rows, cols in hl.tile(left.shape):
        selected[rows, cols] = right[rows, cols] + 1
    return selected, right


def _fresh_join(left: torch.Tensor, right: torch.Tensor, select: int):
    first = torch.empty_like(left)
    second = torch.empty_like(right)
    if select > 0:
        selected = first
    else:
        selected = second
    for rows, cols in hl.tile(left.shape):
        first[rows, cols] = left[rows, cols] + 1
        selected[rows, cols] = right[rows, cols] + 2
    return first, selected


@cache
def _bound_case(mode):
    left, right = torch.empty(4, 8), torch.empty(4, 8)
    fn, third = (_fresh_join, 1) if mode == "join" else (_input_pair, mode)
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (left, right, third))
    return bound, (left, right, third)


@contextmanager
def _guarded_case(mode):
    bound, args = _bound_case(mode)
    env, host = bound.env, bound.host_function
    third_name = "select" if mode == "join" else "mode"
    with (
        env,
        host,
        env.use_runtime_arg_values(
            dict(zip(("left", "right", third_name), args, strict=True))
        ),
    ):
        guards.register_tensor_alias_specializations(env)
        key = guards._TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
        descriptor = env.runtime_input_specializations[key]
        # Actual original entry inputs and the real registered classifier;
        # this mirrors recording its result in the bound dispatch key.
        fact = descriptor.classifier(args[:2])
        assert fact == (True,)
        env.bound_runtime_input_specialization_results[key] = fact
        yield bound, args, descriptor


@pytest.mark.parametrize(
    "mode",
    ["untouched", "native_set", "native_restore", "opaque_replace", "opaque_restore"],
)
@forbid_cuda_work()
def test_direct_runtime_queries_reject_invalidated_input_provenance(mode):
    with _guarded_case(mode) as (bound, args, descriptor):
        env, host = bound.env, bound.host_function
        expected = mode == "untouched"
        for name in ("left", "selected"):
            left = host.local_types[name].fake_value
            right = host.local_types["right"].fake_value
            assert (env.tensor_storage_provenance(left) is not None) is expected
            assert (env.tensor_storage_provenance(right) is not None) is expected
            actual = {
                "bound": guards.runtime_tensors_are_proven_disjoint(env, left, right),
                "unbound": guards.runtime_tensors_are_proven_disjoint(
                    env, left, right, allow_unbound=True
                ),
                "provisional": guards.runtime_tensors_have_disjoint_storage(
                    env, left, right, descriptor.sources
                ),
            }
            assert actual == dict.fromkeys(actual, expected), (name, actual)
        assert (
            guards.runtime_tensor_sources_are_proven_disjoint(env, *descriptor.sources)
            is expected
        )
        assert (
            guards.runtime_tensor_sources_are_proven_disjoint(
                env, *descriptor.sources, allow_unbound=True
            )
            is expected
        )
        pairs = guards.tensor_pairs_with_proven_disjoint_storage(
            env, {n: host.local_types[n].fake_value for n in ("left", "right")}
        )
        assert (frozenset(("left", "right")) in pairs) is expected
    with cpu_render_target(fake_device_sm_count=148):
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        selected, right = entry(*args)
    # This executes only the generated host wrapper: native arithmetic is stubbed.
    assert (selected.untyped_storage() is right.untyped_storage()) is (not expected)


@forbid_cuda_work()
def test_fragment_dot_rejects_a_join_containing_its_fresh_operand():
    with _guarded_case("join") as (bound, args, _descriptor):
        env, host = bound.env, bound.host_function
        tensors = {
            n: host.local_types[n].fake_value for n in ("first", "second", "selected")
        }
        pairs = guards.tensor_pairs_with_proven_disjoint_storage(env, tensors)
        first = tensors["first"]
        for name, expected in (("second", True), ("selected", False)):
            assert (
                FragmentDotEmitter.host_storages_disjoint(
                    SimpleNamespace(env=env), first, tensors[name]
                )
                is expected
            )
            assert (frozenset(("first", name)) in pairs) is expected
    with cpu_render_target(fake_device_sm_count=148):
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        for select in (1, 0):
            first, selected = entry(*args[:2], select)
            assert (first.untyped_storage() is selected.untyped_storage()) is (
                select == 1
            )


@pytest.mark.parametrize("answer", [False, True])
@forbid_cuda_work()
def test_all_origin_input_proof_uses_the_leaf_callback(answer):
    with _guarded_case("untouched") as (bound, _args, _descriptor):
        env, host = bound.env, bound.host_function
        left, right = (host.local_types[n].fake_value for n in ("left", "right"))
        calls = []

        def leaf_proof(x, y):
            calls.append((x, y))
            return answer

        assert (
            guards.tensors_have_proven_disjoint_storage(
                env, left, right, runtime_proof=leaf_proof
            )
            is answer
        )
        assert len(calls) == 1
        assert calls[0][0] is left and calls[0][1] is right


@forbid_cuda_work()
def test_all_origin_fresh_join_proof_cannot_be_overridden_by_a_callback():
    with _guarded_case("join") as (bound, _args, _descriptor):
        env, host = bound.env, bound.host_function
        first, second, selected = (
            host.local_types[n].fake_value for n in ("first", "second", "selected")
        )
        calls = []

        def leaf_proof(x, y):
            calls.append((x, y))
            return True

        assert guards.tensors_have_proven_disjoint_storage(
            env, first, second, runtime_proof=leaf_proof
        )
        assert not guards.tensors_have_proven_disjoint_storage(
            env, first, selected, runtime_proof=leaf_proof
        )
        assert calls == []


@pytest.mark.parametrize("target, expected", [("second", True), ("selected", False)])
@forbid_cuda_work()
def test_topk_output_proof_checks_every_join_origin(target, expected):
    with _guarded_case("join") as (bound, _args, _descriptor):
        env, host = bound.env, bound.host_function
        tensors = {n: host.local_types[n].fake_value for n in ("left", "first", target)}
        plan = CuteTopKPlan(
            root_graph=host.device_ir.graphs[0].graph,
            row_block_id=0,
            n=8,
            k=1,
            largest=True,
            x=tensors["left"],
            values=tensors["first"],
            indices=tensors[target],
            selection_dtype=tensors["left"].dtype,
        )
        assert topk_tensors_are_proven_disjoint(plan, env) is expected


@pytest.mark.parametrize("target, expected", [("second", True), ("selected", False)])
@forbid_cuda_work()
def test_readonly_host_load_proof_checks_every_join_origin(target, expected):
    with _guarded_case("join") as (bound, _args, _descriptor):
        env, host = bound.env, bound.host_function
        graph = torch.fx.Graph()
        source = graph.call_function(_host_tensor, ("first",))
        source.meta["val"] = host.local_types["first"].fake_value
        output = graph.call_function(_host_tensor, (target,))
        output.meta["val"] = host.local_types[target].fake_value
        value = graph.call_function(load, (source, [slice(None)], None, None))
        graph.call_function(store, (output, [slice(None)], value, None))
        graph.output(None)
        assert host_load_is_readonly(value, env, [RootGraphInfo(0, graph)]) is expected


@forbid_cuda_work()
def test_all_origin_invalidated_proof_never_calls_the_leaf_callback():
    with _guarded_case("native_restore") as (bound, _args, _descriptor):
        env, host = bound.env, bound.host_function
        left, right = (host.local_types[n].fake_value for n in ("left", "right"))
        calls = []

        def leaf_proof(x, y):
            calls.append((x, y))
            return True

        assert env.tensor_storage_provenance(left) is None
        assert not guards.tensors_have_proven_disjoint_storage(
            env, left, right, runtime_proof=leaf_proof
        )
        assert calls == []
