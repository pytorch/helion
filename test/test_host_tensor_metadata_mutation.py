from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _check_metadata(bound, names, values):
    with bound.env, bound.host_function:
        for name in names:
            fake = bound.host_function.local_types[name].fake_value
            actual = values[name]
            for symbolic, expected in zip(
                (*fake.size(), *fake.stride()),
                (*actual.size(), *actual.stride()),
                strict=True,
            ):
                if isinstance(symbolic, torch.SymInt):
                    expr = bound.env.shape_env.simplify(symbolic._sympy_())
                    substitutions = {
                        symbol: eval(
                            bound.host_function.expr_to_origin[
                                symbol
                            ].origin.host_str(),
                            {"torch": torch, "hl": hl},
                            values,
                        )
                        for symbol in expr.free_symbols
                    }
                    symbolic = int(expr.xreplace(substitutions))
                assert symbolic == expected, (name, symbolic, expected)


def _conditional_mutation(x, flag: int, loop: hl.constexpr, fresh: hl.constexpr):
    if fresh:
        value = torch.zeros_like(x)
    else:
        value = x
    if loop:
        while flag > 0:
            value.t_()
            flag -= 1
    else:
        if flag > 0:
            value.t_()
    out = torch.empty((value.size(0),), dtype=value.dtype, device=value.device)
    for row in hl.tile(value.size(0)):
        out[row] = value[row, :].sum(-1)
    return value, out


@pytest.mark.parametrize(
    "static_shapes,loop,fresh", [(True, False, False), (False, True, True)]
)
def test_conditional_metadata_uses_actual_shape_and_grid(static_shapes, loop, fresh):
    kernel = helion.kernel(_conditional_mutation, static_shapes=static_shapes)
    with cpu_render_target(fake_device_sm_count=148):
        # CuTe specializes a reduction's power-of-two bucket. Keep the positive
        # replay in one bucket; the separate test retains its 3x5 refusal.
        shape = (5, 7) if kernel.settings.backend == "cute" else (3, 5)
        x = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        if kernel.settings.backend == "pallas":
            # Pallas requires an explicitly specialized untiled device extent.
            # Mutation makes this full-row load runtime-sized even with static
            # entry shapes; it must refuse before generating a stale reduction.
            with pytest.raises(helion.exc.ShapeSpecializingAllocation):
                cpu_bind(kernel, (x, 1, loop, fresh))
            return
        bound = cpu_bind(kernel, (x, 1, loop, fresh))
        config = bound.config_spec.default_config()
        config.config["block_sizes"][0] = 1
        if kernel.settings.backend == "metal":
            # This post-mutation full-row reduction has no legal looped slot.
            # Metal must refuse its runtime extent rather than bake one shape.
            with pytest.raises(
                helion.exc.BackendUnsupported, match=":' dimension of runtime extent"
            ):
                bound.to_code(config)
            return
        launches = []
        entry = load_host_function(
            bound, config, lambda kernel, grid, *args, **kwargs: launches.append(grid)
        )
        for flag in (0, 1, 2, 0):
            runtime_x = x.clone()
            value, out = entry(runtime_x, flag, loop, fresh)
            expected_transpose = bool(flag % 2 if loop else flag > 0)
            assert value.shape == (
                tuple(reversed(shape)) if expected_transpose else shape
            )
            assert out.shape == (value.size(0),)
            assert launches.pop() == (value.size(0),)
            _check_metadata(
                bound, ("value", "out"), {"x": runtime_x, "value": value, "out": out}
            )


@onlyBackends(["cute"])
def test_conditional_metadata_refuses_changed_reduction_bucket_before_launch():
    kernel = helion.kernel(_conditional_mutation, static_shapes=True)
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (x, 1, False, False))
        launches = []
        entry = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda *args, **kwargs: launches.append(args),
        )
        with pytest.raises(RuntimeError, match="power-of-2 bucket"):
            entry(x, 0, False, False)
        assert not launches


def _owner_mutation(x, src, choose: int, mutation: hl.constexpr):
    first = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    second = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    if choose > 0:
        target = first
    else:
        target = second
    if mutation == "set":
        first.set_(src)
    else:
        first.t_()
    for i, j in hl.tile(target.size()):
        target[i, j] = 1.0
    return target, first, second


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("mutation", ["set", "transpose"])
def test_owner_mutation_refreshes_dependent_join_metadata(static_shapes, mutation):
    kernel = helion.kernel(_owner_mutation, static_shapes=static_shapes)
    x = torch.empty(4, 8)
    src = torch.empty(8, 4).t()
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (x, src, 1, mutation))
        config = bound.config_spec.default_config()
        entry = load_host_function(bound, config, lambda *args, **kwargs: None)
        for choose in (1, 0, 1):
            target, first, second = entry(x, src, choose, mutation)
            assert target is (first if choose > 0 else second)
            assert second.shape == (4, 8) and second.stride() == (8, 1)
            _check_metadata(
                bound,
                ("target", "first", "second"),
                {
                    "x": x,
                    "src": src,
                    "target": target,
                    "first": first,
                    "second": second,
                },
            )


def _joined_view_mutation(x, choose: int):
    first = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    second = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    if choose > 0:
        target = first
    else:
        target = second
    flattened = target.view(-1)
    flattened.as_strided_((14,), (1,))
    for i, j in hl.tile(first.size()):
        first[i, j] = 1.0
    return first, second, target, flattened


def test_view_metadata_mutation_does_not_resize_join_owners():
    kernel = helion.kernel(_joined_view_mutation, static_shapes=True)
    x = torch.empty(3, 5)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (x, 1))
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        for choose in (1, 0):
            first, second, target, flattened = entry(x, choose)
            assert first.shape == second.shape == target.shape == (3, 5)
            assert flattened.shape == (14,)
            _check_metadata(
                bound,
                ("first", "second", "target", "flattened"),
                {
                    "x": x,
                    "first": first,
                    "second": second,
                    "target": target,
                    "flattened": flattened,
                },
            )


def _expanded_input(x):
    x.as_strided_((4,), (1,))
    for tile in hl.tile(x.numel()):
        x[tile] = 1.0
    return x


def test_metadata_expansion_discards_entry_storage_span_proof():
    kernel = helion.kernel(_expanded_input, static_shapes=True)
    x = torch.empty_strided((2,), (4,))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (x,))
        with bound.env, bound.host_function:
            fake = bound.host_function.local_types["x"].fake_value
            assert bound.env.tensor_storage_provenance(fake) is None
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        assert entry(x).shape == (4,)
        _check_metadata(bound, ("x",), {"x": x})


def _conditional_rank(x, flag: int):
    if flag > 0:
        x.unsqueeze_(0)
    return x


def test_conditional_rank_mutation_is_refused():
    kernel = helion.kernel(_conditional_rank, static_shapes=True)
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.TypeInferenceError, match="cannot change its rank"),
    ):
        cpu_bind(kernel, (torch.empty(3, 5), 1))


def _or_conditional_rank(x, flag: int):
    result = flag > 0 or x.unsqueeze_(0) is None
    out = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.numel()):
        out[tile] = 0.0
    return x, out, result


def _and_conditional_rank(x, flag: int):
    result = flag > 0 and x.unsqueeze_(0) is None
    out = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.numel()):
        out[tile] = 0.0
    return x, out, result


def _comparison_conditional_rank(x, flag: int):
    result = 0 > flag > x.unsqueeze_(0).numel()
    out = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.numel()):
        out[tile] = 0.0
    return x, out, result


def _list_conditional_rank(x, flag: int):
    values = [x.unsqueeze_(0) for index in (0,) if flag > index]
    out = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.numel()):
        out[tile] = 0.0
    return x, out, values


def _generator_conditional_rank(x, flag: int):
    values = (x.unsqueeze_(0) for index in (0,) if flag > index)
    out = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.numel()):
        out[tile] = 0.0
    return x, out, values


def _dict_conditional_rank(x, flag: int):
    values = {index: x.unsqueeze_(0) for index in (0,) if flag > index}
    out = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.numel()):
        out[tile] = 0.0
    return x, out, values


@pytest.mark.parametrize(
    "fn",
    [
        _or_conditional_rank,
        _and_conditional_rank,
        _comparison_conditional_rank,
        _list_conditional_rank,
        _generator_conditional_rank,
        _dict_conditional_rank,
    ],
)
def test_expression_conditional_rank_mutation_is_refused(fn):
    kernel = helion.kernel(fn, static_shapes=True)
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.TypeInferenceError, match="cannot change its rank"),
    ):
        cpu_bind(kernel, (torch.empty(3, 5), 1))


def _or_unconditional_rank(x, flag: int):
    result = x.unsqueeze_(0) is None or flag > 0
    out = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.numel()):
        out[tile] = 0.0
    return x, out, result


def _comparison_unconditional_rank(x, flag: int):
    result = 0 < x.unsqueeze_(0).numel() < flag
    out = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.numel()):
        out[tile] = 0.0
    return x, out, result


def _comprehension_iterable_unconditional_rank(x, flag: int):
    values = [size for size in x.unsqueeze_(0).shape if flag > 0]
    out = torch.empty((x.numel(),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.numel()):
        out[tile] = 0.0
    return x, out, values


@pytest.mark.parametrize(
    "fn",
    [
        _or_unconditional_rank,
        _comparison_unconditional_rank,
        _comprehension_iterable_unconditional_rank,
    ],
)
def test_expression_unconditional_rank_mutation_replays(fn):
    kernel = helion.kernel(fn, static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (torch.empty(3, 5), 1))
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        for flag in (0, 1):
            x = torch.empty(3, 5)
            result = entry(x, flag)
            assert result[0] is x
            assert x.shape == (1, 3, 5)
            _check_metadata(bound, ("x",), {"x": x})


def _specialized_conditional_mutation(x, flag: int):
    hl.specialize(x.stride())
    if flag > 0:
        x.t_()
    out = torch.empty_like(x)
    for i, j in hl.tile(x.size()):
        out[i, j] = x[i, j]
    return x, out


def test_entry_stride_specialization_does_not_freeze_post_mutation_metadata():
    kernel = helion.kernel(_specialized_conditional_mutation, static_shapes=False)
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (x, 1))
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        for flag in (0, 1):
            runtime_x, out = entry(x.clone(), flag)
            assert out.shape == runtime_x.shape
            _check_metadata(bound, ("x", "out"), {"x": runtime_x, "out": out})


def _conditionally_transposed_mpp_operand(x, y, flag: int, mutate: hl.constexpr):
    hl.specialize(x.stride())
    hl.specialize(y.stride())
    if mutate:
        if flag > 0:
            x.t_()
    m, k = x.size()
    n = hl.specialize(y.size(1))
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for reduction in hl.tile(k):
            acc = torch.addmm(acc, x[rows, reduction], y[reduction, cols])
        out[rows, cols] = acc
    return out


@onlyBackends(["metal"])
def test_metal_mpp_view_requires_unmutated_specialized_strides():
    with cpu_render_target():
        kernel = helion.kernel(
            _conditionally_transposed_mpp_operand, backend="metal", static_shapes=False
        )
        for mutate in (False, True):
            x = torch.empty((32, 32), dtype=torch.float16)
            y = torch.empty((32, 32), dtype=torch.float16)
            bound = cpu_bind(kernel, (x, y, 1, mutate))
            code = bound.to_code(helion.Config(block_sizes=[16, 16, 16]))
            assert "_helion_mpp_run" in code
            if mutate:
                # MPP must stage the operand and load through runtime strides,
                # even though its entry strides were explicitly specialized.
                assert "spaces=[1, 0]" in code
                assert "x_stride_0" in code and "x_stride_1" in code
            else:
                # Unchanged specialized operands still use two device views.
                assert "spaces=[0, 0]" in code


def _metadata_between_roots(x):
    x.t_()
    first = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        first[row] = 1.0
    x.t_()
    second = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        second[row] = 1.0
    for row in hl.tile(first.size(0)):
        first[row] += 1.0
    return first, second


def test_metadata_mutation_between_device_roots_remains_unsupported():
    kernel = helion.kernel(_metadata_between_roots, static_shapes=True)
    x = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.TopLevelStatementBetweenLoops),
    ):
        cpu_bind(kernel, (x,))


def _metadata_before_root(x):
    x.t_()
    first = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    x.t_()
    for row in hl.tile(first.size(0)):
        first[row] = 1.0
    return first


def test_distinct_mutation_sites_keep_earlier_allocation_metadata():
    kernel = helion.kernel(_metadata_before_root, static_shapes=True)
    x = torch.empty(5, 7)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (x,))
        config = bound.config_spec.default_config()
        config.config["block_sizes"][0] = 1
        launches = []
        entry = load_host_function(
            bound,
            config,
            lambda kernel, grid, *args, **kwargs: launches.append((grid, kwargs)),
        )
        first = entry(x)
        assert first.shape == (7,)
        ((grid, launch_kwargs),) = launches
        block_size = config.config["block_sizes"][0]
        if bound.settings.backend == "pallas":
            # The emitted Pallas tile can exceed the requested block size.
            # Its actual output BlockSpec determines the launch grid.
            (output_index,) = launch_kwargs["_output_indices"]
            ((block_size,), (grid_dim,)) = launch_kwargs["_block_spec_info"][
                output_index
            ]
            assert grid_dim == 0
        assert grid == ((first.size(0) + block_size - 1) // block_size,)
        _check_metadata(bound, ("x", "first"), {"x": x, "first": first})
