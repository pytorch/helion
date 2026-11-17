from __future__ import annotations

from itertools import starmap
from operator import itemgetter
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sympy
import torch
from torch._dynamo.source import AttrSource
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._compiler._dynamo.variables import infer_output_spec
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.runtime_tensor_guards import _canonical_metadata_guard_factory
from helion._compiler.runtime_tensor_guards import _CanonicalLayoutClassifier
from helion._compiler.runtime_tensor_guards import _guard_canonical_layout
from helion._compiler.variable_origin import ArgumentOrigin
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl
from helion.runtime.kernel import _capture_safe_specialization
from helion.runtime.kernel import _fast_canonical_metadata_dispatch
from helion.runtime.kernel import _implied_by_exact_metadata
from helion.runtime.kernel import _make_prepared_arg_guard
from helion.runtime.kernel import _partition_prepared_extra_guards
from helion.runtime.kernel import _RuntimeInputSpecializationExtractor

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("maximum", [False, True])
@pytest.mark.parametrize("arity", [2, 3])
def test_stored_layout_classifier_replays_symbolic_extrema(maximum, arity):
    env = CompileEnvironment(torch.device("cpu"), helion.Settings(static_shapes=False))
    real_input = torch.empty(4, 6)
    extremum = sympy.Max if maximum else sympy.Min
    concrete_extremum = max if maximum else min
    with env, env.use_runtime_arg_values({"x": real_input}):
        fake_input = env.to_fake(real_input, ArgumentOrigin("x"))
        assert isinstance(fake_input, torch.Tensor)
        operands = [size.node.expr for size in fake_input.size()]
        if arity == 3:
            operands.append(sympy.Integer(3))
        expression = extremum(*operands)
        hint = concrete_extremum(4, 6, 3) if arity == 3 else concrete_extremum(4, 6)
        extent = env.shape_env.create_symintnode(expression, hint=hint)
        allocation = torch.empty((extent,))
        assert _guard_canonical_layout(env, allocation, require_input=False) == (2,)
        (specialization,) = env.runtime_input_specializations.values()

    # Exercise the stored source projection, not a separate extrema wrapper.
    def classify(tensor):
        values = []
        for source in specialization.sources:
            assert isinstance(source, TensorPropertySource)
            assert source.prop == TensorProperty.SIZE
            assert source.idx is not None
            values.append(tensor.size(source.idx))
        return specialization.classifier(values)

    for shape in ((0, 0), (1, 1), (0, 4), (4, 6)):
        extent = (
            concrete_extremum(*shape, 3) if arity == 3 else concrete_extremum(*shape)
        )
        assert classify(torch.empty(shape)) == (True, (min(extent, 2),))

    outer_shapes = ShapeEnv()
    with FakeTensorMode(shape_env=outer_shapes):
        left = outer_shapes.create_unbacked_symint()
        right = outer_shapes.create_unbacked_symint()
        outer_input = torch.empty_strided((left, right), (right, 1))
    guards_before = tuple(outer_shapes.guards)
    asserts_before = {
        key: tuple(value)
        for key, value in outer_shapes.deferred_runtime_asserts.items()
    }
    # max(3, u, v) proves its class; the other expressions remain unknown.
    expected = (True, (2,)) if maximum and arity == 3 else None
    assert classify(outer_input) == expected
    assert tuple(outer_shapes.guards) == guards_before
    assert {
        key: tuple(value)
        for key, value in outer_shapes.deferred_runtime_asserts.items()
    } == asserts_before


@helion.kernel(static_shapes=False, autotune_effort="none")
def _copy_layout(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _copy_static_rank_one(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def _fresh_shape_layout(x):
    created = torch.empty((4, x.size(0)), device=x.device)
    out = torch.empty_like(created)
    for tile in hl.tile(out.size()):
        out[tile] = 0
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def _fresh_product_layout(x):
    created = torch.empty((4, x.size(1) * x.size(2)), device=x.device)
    out = torch.empty_like(created)
    for tile in hl.tile(out.size()):
        out[tile] = 0
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def _copy_nested_layout(values):
    x = values["x"][0]
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile]
    return out


def test_canonical_layout_classifier_can_be_captured_without_eliding_metadata():
    real_input = torch.empty(4, 6)
    with cpu_render_target():
        bound = cpu_bind(_copy_layout, (real_input,))
    (specialization,) = (
        item
        for item in bound.env.runtime_input_specializations.values()
        if item.classifier_identity[0] == "canonical_layout_v2"
    )
    indices = tuple(
        source.idx
        for source in specialization.sources
        if isinstance(source, TensorPropertySource)
    )

    def classify(x):
        return specialization.classifier((x, *(x.size(index) for index in indices)))

    assert classify(real_input) == (True, (2,))
    assert classify(real_input.t()) == (False, ())
    extractor = _RuntimeInputSpecializationExtractor(
        (itemgetter(0), *(lambda args, i=i: args[0].size(i) for i in indices)),
        specialization.classifier,
        specialization.reusable_tensor_properties,
    )
    assert _capture_safe_specialization(extractor)
    assert not _implied_by_exact_metadata(extractor)
    captured = torch.compile(classify, backend="eager", fullgraph=True, dynamic=True)
    for value in (real_input, real_input.t(), real_input[:, ::2], real_input):
        assert captured(value) == classify(value)


def test_compound_outer_tensor_layout_uses_whole_metadata_source():
    outer_shapes = ShapeEnv()
    real_input = torch.empty(5, 8)
    with FakeTensorMode(shape_env=outer_shapes) as mode:
        outer_input = mode.from_tensor(real_input, static_shapes=False)
        sliced = outer_input[1:] * 2.0
    with cpu_render_target():
        bound = cpu_bind(_copy_layout, (sliced,))
        spec = infer_output_spec(_copy_layout, (sliced,))
    result = bound.host_function.local_types["out"].fake_value
    assert bound.env.tensor_layout_is_symbolically_exact(result)
    with bound.env:
        assert all(
            starmap(
                bound.env.known_equal,
                zip(result.stride(), bound.fake_args[0].stride(), strict=True),
            )
        )
    assert spec["leaf_specs"][0]["stride"] == list(sliced.stride())


def test_compound_outer_factory_extent_uses_whole_input_size():
    outer_shapes = ShapeEnv()
    real_input = torch.empty(5, 8)
    with FakeTensorMode(shape_env=outer_shapes) as mode:
        outer_input = mode.from_tensor(real_input, static_shapes=False)
        expanded = torch.cat((outer_input, outer_input[:1]), dim=0)
    with cpu_render_target():
        bound = cpu_bind(_fresh_shape_layout, (expanded,))
    result = bound.host_function.local_types["out"].fake_value
    assert bound.env.tensor_layout_is_symbolically_exact(result)
    (specialization,) = (
        item
        for item in bound.env.runtime_input_specializations.values()
        if item.classifier_identity[0] == "canonical_layout_v2"
    )
    assert len(specialization.sources) == 1
    (source,) = specialization.sources
    assert isinstance(source, TensorPropertySource)
    assert source.prop == TensorProperty.SIZE and source.idx == 0
    before = tuple(outer_shapes.guards)
    assert specialization.classifier((expanded.size(0),)) == (True, (2, 2))
    assert tuple(outer_shapes.guards) == before


def test_factory_extent_product_does_not_replay_an_inferred_input_stride():
    _fresh_product_layout.reset()
    with cpu_render_target():
        _fresh_product_layout.bind((torch.empty(4, 3, 2),))
        for shape in ((4, 3, 2), (4, 0, 2), (4, 1, 1), (4, 3, 2)):
            x = torch.empty_strided(shape, (100, 10, 2))
            width = shape[1] * shape[2]
            bound = _fresh_product_layout.bind((x,))
            specializations = tuple(
                item
                for item in bound.env.runtime_input_specializations.values()
                if type(item.classifier) is _CanonicalLayoutClassifier
            )
            assert specializations
            for specialization in specializations:
                assert all(
                    isinstance(source, TensorPropertySource)
                    and source.prop == TensorProperty.SIZE
                    for source in specialization.sources
                )
                values = tuple(x.size(source.idx) for source in specialization.sources)
                assert specialization.classifier(values) == (True, (2, min(width, 2)))
            host = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            out = host(x)
            assert out.shape == (4, width)
            assert out.stride() == (max(width, 1), 1)
            fake = bound.host_function.local_types["out"].fake_value
            assert bound.env.tensor_layout_is_symbolically_exact(fake)
            for dim, stride in enumerate(fake.stride()):
                if isinstance(stride, torch.SymInt):
                    expression = stride._sympy_()
                    stride = expression.subs(
                        {
                            symbol: eval(
                                bound.host_function.expr_to_origin[
                                    symbol
                                ].origin.host_str(),
                                {},
                                {"x": x, "out": out},
                            )
                            for symbol in expression.free_symbols
                        }
                    )
                assert int(stride) == out.stride(dim)


@pytest.mark.parametrize("rank", [1, 2])
def test_direct_canonical_leading_extent_keeps_normal_dispatch(rank):
    _copy_layout.reset()
    first = None
    with cpu_render_target():
        for leading in (4, 1, 0, 4):
            shape = (leading,) if rank == 1 else (leading, 8)
            x = torch.empty(shape)
            bound = _copy_layout.bind((x,))
            if first is None:
                first = bound
            elif leading == 4:
                assert bound is first
            host = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            out = host(x)
            assert out.shape == x.shape
            assert out.stride() == torch.empty_like(x).stride()


def test_noncanonical_input_does_not_partition_unused_extents():
    classifier = _CanonicalLayoutClassifier(True, ("v0",), 1)
    for size in (0, 1, 4):
        value = torch.empty_strided((size, 8), (24, 2))
        assert classifier((value, size)) == (False, ())


def test_symbolic_noncanonical_rank_one_compacts_only_nondegenerate_extents():
    _copy_layout.reset()
    first = None
    with cpu_render_target():
        for size in (4, 1, 0, 4):
            x = torch.empty_strided((size,), (2,))
            bound = _copy_layout.bind((x,))
            if first is None:
                first = bound
            elif size == 4:
                assert bound is first
            host = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            out = host(x)
            assert out.shape == x.shape
            assert out.stride() == torch.empty_like(x).stride()
            fake = bound.host_function.local_types["out"].fake_value
            assert bound.env.tensor_layout_is_symbolically_exact(fake) is (size > 1)


@pytest.mark.parametrize("size", [0, 1])
def test_static_rank_one_fallback_has_no_constant_layout_classifier(size):
    with cpu_render_target():
        x = torch.empty_strided((size,), (2,))
        bound = cpu_bind(_copy_static_rank_one, (x,))
        assert not any(
            type(item.classifier) is _CanonicalLayoutClassifier
            for item in bound.env.runtime_input_specializations.values()
        )
        host = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        assert host(x).stride() == torch.empty_like(x).stride() == (2,)


def test_dynamic_rank_one_degenerate_layouts_share_layout_fallback():
    _copy_layout.reset()
    fallback = None
    seen = {}
    with cpu_render_target():
        for size in (0, 1, 4, 1, 0):
            x = torch.empty_strided((size,), (2,))
            bound = _copy_layout.bind((x,))
            (specialization,) = (
                item
                for item in bound.env.runtime_input_specializations.values()
                if type(item.classifier) is _CanonicalLayoutClassifier
            )
            assert len(specialization.sources) == 1
            source = specialization.sources[0]
            assert isinstance(source, TensorPropertySource)
            assert source.prop == TensorProperty.SIZE and source.idx == 0
            assert specialization.classifier((size,)) == (
                True,
                (1 if size <= 1 else 2,),
            )
            if size <= 1:
                if fallback is None:
                    fallback = bound
                elif bound.settings.backend == "triton":
                    # CuTe's independent vector-alignment key includes shape.
                    assert bound is fallback
            else:
                assert bound is not fallback
            if size in seen:
                assert bound is seen[size]
            seen[size] = bound
            host = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            out = host(x)
            assert out.shape == (size,)
            assert out.stride() == torch.empty_like(x).stride()
            fake = bound.host_function.local_types["out"].fake_value
            assert bound.env.tensor_layout_is_symbolically_exact(fake) is (size > 1)


def test_rank_one_equal_output_layouts_keep_distinct_input_indexing():
    _copy_layout.reset()
    contiguous = torch.empty(4)
    stepped = torch.empty(8)[::2]
    with cpu_render_target():
        dense_bound = _copy_layout.bind((contiguous,))
        stepped_bound = _copy_layout.bind((stepped,))
        assert stepped_bound is not dense_bound
        # Publishing the second layout's guard can replace the earlier schema.
        # Repopulate both routes before checking within-class cache reuse.
        dense_bound = _copy_layout.bind((contiguous,))
        stepped_bound = _copy_layout.bind((stepped,))
        assert _copy_layout.bind((contiguous,)) is dense_bound
        assert _copy_layout.bind((stepped,)) is stepped_bound
        for bound, x in ((dense_bound, contiguous), (stepped_bound, stepped)):
            host = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            assert host(x).stride() == (1,)
            assert bound.env.tensor_layout_is_symbolically_exact(
                bound.host_function.local_types["out"].fake_value
            )
        assert dense_bound.fake_args[0].stride() == (1,)
        with stepped_bound.env:
            assert not stepped_bound.env.known_equal(
                stepped_bound.fake_args[0].stride(0), 1
            )


@pytest.mark.parametrize("nested", [False, True])
def test_canonical_metadata_fast_and_prepared_keep_layout_fallback(nested):
    kernel = _copy_nested_layout if nested else _copy_layout
    kernel.reset()
    x = torch.empty(4, 8)

    def pack(value):
        return ({"x": [value]},) if nested else (value,)

    with cpu_render_target():
        args = pack(x)
        kernel.bind(args)
        first = kernel._fast_dispatch_key_and_guards(args, is_distributed=False)
        assert first is not None
        extras = tuple(
            (extractor, result)
            for extractor, result in first.extra_guards
            if isinstance(extractor, _RuntimeInputSpecializationExtractor)
            and type(extractor.classifier) is _CanonicalLayoutClassifier
        )
        assert len(extras) == 1
        with patch.object(
            _CanonicalLayoutClassifier,
            "__call__",
            side_effect=AssertionError("fixed metadata repeated classifier"),
        ):
            assert (
                kernel._fast_dispatch_key(pack(x.clone()), is_distributed=False)
                == first.key
            )
            always_check, residual = _partition_prepared_extra_guards(args, extras)
            assert not always_check and len(residual) == 1
            assert residual[0](pack(x.clone()))

        metadata_guard = _make_prepared_arg_guard(kernel, args)
        assert metadata_guard(pack(x.clone()))
        assert not metadata_guard(pack(x.t()))
        assert not metadata_guard(pack(SimpleNamespace()))
        if nested:
            assert not metadata_guard(({"x": (x,)},))

        extractor, expected = extras[0]
        dispatch = _fast_canonical_metadata_dispatch(extractor, expected, args)
        sparse = x.to_sparse()
        assert not residual[0](pack(sparse))
        assert dispatch(pack(sparse)) == extractor(pack(sparse)) == (False, ())


def test_canonical_metadata_memoization_refuses_unpinned_paths():
    class CustomList(list):
        pass

    x = torch.empty(4, 8)
    env = CompileEnvironment(torch.device("cpu"), helion.Settings(static_shapes=False))
    local = LocalSource("x", is_input=True)
    with env.use_runtime_arg_values({"x": SimpleNamespace(tensor=x)}):
        assert (
            _canonical_metadata_guard_factory(env, (AttrSource(local, "tensor"),), True)
            is None
        )
    with env.use_runtime_arg_values({"x": CustomList([x])}):
        assert (
            _canonical_metadata_guard_factory(env, (GetItemSource(local, 0),), True)
            is None
        )
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(x)
    with env.use_runtime_arg_values({"x": fake}):
        assert _canonical_metadata_guard_factory(env, (local,), True) is None


def test_layout_classifier_without_extent_partition() -> None:
    classifier = _CanonicalLayoutClassifier(True, (), 0)
    for size in (0, 1, 7):
        assert classifier((torch.empty(size),)) == (True, ())
    assert classifier((torch.empty_strided((1,), (7,)),)) == (False, ())
