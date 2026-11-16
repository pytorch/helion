from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.runtime_tensor_guards import _canonical_contiguous_layout
from helion._compiler.type_info import NumericType
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import scalar_additions
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _consume(values):
    print(values)


def _label(values):
    print(values)
    return "nonempty" if values else "empty"


@helion.kernel(autotune_effort="none")
def _host_calls(x, limit: int, kind: hl.constexpr):
    vals = [v for v in (1, 2, 3) if v > limit]
    if kind == "print":
        print(vals)
    elif kind == "print_starred":
        print(*vals)
    elif kind == "helper_none":
        _consume(vals)
    else:
        _label(vals)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile]
    return out


@pytest.mark.parametrize(
    "kind", ["print", "print_starred", "helper_none", "helper_unused_result"]
)
def test_unused_host_collection_call(kind, capsys):
    x = torch.ones(7)
    with cpu_render_target():
        bound = cpu_bind(_host_calls, (x, 0, kind))
        host = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        capsys.readouterr()
        for limit in (0, 3, 1):
            host(x, limit, kind)
            values = [v for v in (1, 2, 3) if v > limit]
            expected = (
                " ".join(map(str, values)) if kind == "print_starred" else str(values)
            )
            assert capsys.readouterr().out.strip() == expected


@helion.kernel(autotune_effort="none")
def _inline_length(x, limit: int, form: hl.constexpr):
    if form == "tuple":
        vals = tuple([v for v in (1, 2, 3) if v > limit])
    elif form == "dict":
        vals = {v: v for v in (1, 2, 3) if v > limit}
    else:
        vals = [v for v in (1, 2, 3) if v > limit]
    out = torch.empty_like(x)
    for tile in hl.tile(len(vals)):
        out[tile] = x[tile]
    return out


@pytest.mark.parametrize("form", ["list", "tuple", "dict"])
def test_inline_filtered_length_bound(form):
    with cpu_render_target():
        x = torch.ones(7)
        bound = cpu_bind(_inline_length, (x, 0, form))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append((grid, args)),
            scalar_values=True,
        )
        for limit in (0, 1, 3):
            host(x, limit, form)
            count = len([v for v in (1, 2, 3) if v > limit])
            assert count in [arg for arg in launches[-1][1] if type(arg) is int]
        assert len(launches) == 3


@helion.kernel(autotune_effort="none")
def _filtered_endpoints(x, limit: int):
    vals = [v for v in (1, 2, 3) if v > limit]
    a = vals[0]
    b = vals[-1]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + 10 * a + b
    return out


def test_filtered_indices_have_distinct_runtime_values():
    with cpu_render_target():
        x = torch.ones(7)
        bound = cpu_bind(_filtered_endpoints, (x, 0))
        a = bound.host_function.local_types["a"]
        b = bound.host_function.local_types["b"]
        assert isinstance(a, NumericType) and isinstance(b, NumericType)
        assert a.value._sympy_() != b.value._sympy_()
        assert [value.meta["val"]._sympy_() for value in scalar_additions(bound)] == [
            10 * a.value._sympy_(),
            b.value._sympy_(),
        ]
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
            scalar_values=True,
        )
        for limit in (0, 1, 0):
            host(x, limit)
            scalars = [arg for arg in launches[-1] if type(arg) is int]
            expected = [v for v in (1, 2, 3) if v > limit]
            assert expected[0] in scalars and expected[-1] in scalars


@helion.kernel(autotune_effort="none")
def _dict_collision(x, y, limit: int):
    vals = {0: value for key, value in ((0, x), (1, y)) if key <= limit}  # noqa: B035 - exercise colliding keys
    selected = vals[0]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = selected[tile]
    return out


@pytest.mark.parametrize("stride", [0, 1, 2])
def test_filtered_dict_collision_passes_actual_tensor(stride):
    x = torch.ones(7)
    y = torch.full((max(1, 7 * stride),), 3.0).as_strided((7,), (stride,))
    with cpu_render_target():
        bound = cpu_bind(_dict_collision, (x, y, 0))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
        )
        for limit in (0, 1, 0):
            host(x, y, limit)
            expected, other = (x, y) if limit == 0 else (y, x)
            assert any(arg is expected for arg in launches[-1])
            assert not any(arg is other for arg in launches[-1])
            if bound.settings.backend == "triton":
                assert [arg for arg in launches[-1] if type(arg) is int] == [
                    expected.stride(0)
                ]


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_filtered_dict_collision_rejects_incompatible_tensors(mismatch):
    x = torch.ones(7)
    y = torch.ones(8) if mismatch == "shape" else torch.ones(7, dtype=torch.int64)
    with cpu_render_target(), pytest.raises(helion.exc.ControlFlowTensorMismatch):
        cpu_bind(_dict_collision, (x, y, 0))


@helion.kernel(autotune_effort="none")
def _indexed_length(x, limit: int, index: hl.constexpr):
    vals = [v for v in (1, 2, 3) if v > limit]
    out = torch.empty_like(x)
    for tile in hl.tile(vals[index]):
        out[tile] = x[tile]
    return out


@pytest.mark.parametrize("index", [0, 1, -1])
def test_inline_filtered_index_bound(index):
    with cpu_render_target():
        x = torch.ones(7)
        bound = cpu_bind(_indexed_length, (x, 0, index))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
            scalar_values=True,
        )
        for limit in (0, 1, 0):
            host(x, limit, index)
            expected = [v for v in (1, 2, 3) if v > limit][index]
            assert expected in [arg for arg in launches[-1] if type(arg) is int]


@helion.kernel(autotune_effort="none")
def _nested_length(x, limit: int):
    vals = [v for v in (1, 2, 3) if v > limit]
    outer = (vals,)
    out = torch.empty_like(x)
    for tile in hl.tile(len(outer[0])):
        out[tile] = x[tile]
    return out


def test_nested_filtered_length_bound():
    with cpu_render_target():
        x = torch.ones(7)
        bound = cpu_bind(_nested_length, (x, 0))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
            scalar_values=True,
        )
        for limit in (0, 3, 1):
            host(x, limit)
            expected = len([v for v in (1, 2, 3) if v > limit])
            assert expected in [arg for arg in launches[-1] if type(arg) is int]


@helion.kernel(static_shapes=True, autotune_effort="none")
def _derived_layout(x, y, limit: int, form: hl.constexpr):
    vals = {0: value for key, value in ((0, x), (1, y)) if key <= limit}  # noqa: B035 - exercise colliding keys
    selected = vals[0]
    if form == "keyword":
        out = torch.empty_like(input=selected)
    elif form == "clone":
        out = selected.clone()
    elif form == "chain":
        copied = selected.clone()
        selected = copied.t()
        out = torch.empty_like(selected)
    elif form == "zeros":
        out = torch.zeros_like(selected)
    elif form == "binary":
        out = torch.empty_like(selected + 1)
    elif form == "unary":
        out = torch.empty_like(-selected)
    elif form == "branch":
        if limit > 0:
            out = selected.clone()
        else:
            out = torch.empty_like(selected)
    elif form == "shape":
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    else:
        out = torch.empty_like(selected)
    for row, column in hl.tile(x.shape):
        out[row, column] = selected[row, column]
    return out


@pytest.mark.parametrize(
    "form", ["like", "keyword", "clone", "chain", "zeros", "binary", "unary", "branch"]
)
def test_filtered_tensor_derived_allocation_uses_runtime_layout(form):
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    y = x.t()
    with cpu_render_target():
        bound = _derived_layout.bind((x, y, 0, form))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
        )
        output_type = bound.host_function.local_types["out"]
        assert not bound.env.tensor_layout_is_symbolically_exact(output_type.fake_value)
        for limit in (0, 1, 0):
            assert _derived_layout.bind((x, y, limit, form)) is bound
            out = host(x, y, limit, form)
            expected = x if limit == 0 else y
            if form == "chain":
                expected = expected.t()
            assert out.stride() == expected.stride()
            assert any(arg is out for arg in launches[-1])
            origins = bound.host_function.expr_to_origin
            for dim, stride in enumerate(output_type.fake_value.stride()):
                assert isinstance(stride, torch.SymInt), (dim, stride, out.stride())
                origin = origins[stride._sympy_()].origin
                assert eval(origin.host_str(), {}, {"out": out}) == out.stride(dim)
            if bound.settings.backend == "triton":
                # The emitted launch must pass both output strides to the
                # device; a dynamic load with a baked store is still wrong.
                assert [arg for arg in launches[-1] if type(arg) is int] == [
                    *out.stride(),
                    *expected.stride(),
                ]


def test_filtered_tensor_does_not_taint_independent_static_allocation():
    x = torch.empty(4, 4)
    with cpu_render_target():
        bound = _derived_layout.bind((x, x.t(), 0, "shape"))
        output = bound.host_function.local_types["out"].fake_value
        assert output.stride() == (4, 1)
        assert all(type(stride) is int for stride in output.stride())
        assert bound.env.tensor_layout_is_symbolically_exact(output)
        bound.to_code()


@pytest.mark.cuda
@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("form", ["like", "chain"])
def test_filtered_tensor_derived_allocation_copies_current_layout(form):
    x = torch.arange(16, dtype=torch.float32, device=DEVICE).reshape(4, 4)
    y = x.t()
    original = x.clone()
    bound = _derived_layout.bind((x, y, 0, form))
    compiled = bound.compile_config(bound.config_spec.default_config())
    for limit in (0, 1, 0):
        assert _derived_layout.bind((x, y, limit, form)) is bound
        expected = x if limit == 0 else y
        if form == "chain":
            expected = expected.t()
        out = compiled(x, y, limit, form)
        assert out.stride() == expected.stride()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@helion.kernel(static_shapes=True, autotune_effort="none")
def _direct_branch_layout(x, y, flag: int):
    if flag > 0:
        selected = x
    else:
        selected = y
    out = torch.empty_like(x)
    for row, column in hl.tile(x.shape):
        out[row, column] = selected[row, column]
    return out


@pytest.mark.parametrize("layout", ["transpose", "reverse", "same"])
def test_host_branch_introduces_only_required_layout_uncertainty(layout):
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    y = x.clone() if layout == "same" else x.t()
    if layout == "reverse":
        x, y = y, x
    with cpu_render_target():
        bound = _direct_branch_layout.bind((x, y, 1))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
        )
        strides = bound.host_function.local_types["selected"].fake_value.stride()
        if layout == "same":
            assert strides == (4, 1)
            assert all(type(stride) is int for stride in strides)
        for flag in (1, 0, 1):
            assert _direct_branch_layout.bind((x, y, flag)) is bound
            host(x, y, flag)
            expected = x if flag else y
            assert any(arg is expected for arg in launches[-1])
            if layout != "same":
                for dim, stride in enumerate(strides):
                    assert isinstance(stride, torch.SymInt), (layout, strides)
                    origin = bound.host_function.expr_to_origin[stride._sympy_()].origin
                    assert eval(
                        origin.host_str(), {}, {"selected": expected}
                    ) == expected.stride(dim)
                if bound.settings.backend == "triton":
                    assert [arg for arg in launches[-1] if type(arg) is int] == list(
                        expected.stride()
                    )


@helion.kernel(static_shapes=True, autotune_effort="none")
def _projected_layout(x, y, limit: int, form: hl.constexpr):
    vals = {0: value for key, value in ((0, x), (1, y)) if key <= limit}  # noqa: B035 - exercise colliding keys
    selected = vals[0]
    if form == "advanced":
        projected = selected[[0, 1], :, :]
    elif form == "comparison":
        projected = selected > 20
    elif form == "property":
        projected = selected.T
    elif form == "method":
        projected = selected.clone()
    else:
        projected = torch.neg(selected)
    out = torch.empty(projected.shape, dtype=projected.dtype, device=x.device)
    for row, column, depth in hl.tile(projected.shape):
        out[row, column, depth] = projected[row, column, depth]
    return out


def _projected_reference(selected, form):
    if form == "advanced":
        return selected[[0, 1], :, :]
    if form == "comparison":
        return selected > 20
    if form == "property":
        return selected.T
    if form == "method":
        return selected.clone()
    return torch.neg(selected)


@pytest.mark.parametrize(
    "form", ["advanced", "comparison", "property", "method", "function"]
)
def test_host_tensor_projection_keeps_actual_layout(form):
    x = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
    y = x.transpose(1, 2)
    with cpu_render_target():
        bound = _projected_layout.bind((x, y, 0, form))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
        )
        strides = bound.host_function.local_types["projected"].fake_value.stride()
        for limit in (0, 1, 0):
            assert _projected_layout.bind((x, y, limit, form)) is bound
            out = host(x, y, limit, form)
            projected = next(
                arg
                for arg in launches[-1]
                if isinstance(arg, torch.Tensor) and arg is not out
            )
            expected = _projected_reference(x if limit == 0 else y, form)
            torch.testing.assert_close(projected, expected, rtol=0, atol=0)
            assert projected.stride() == expected.stride()
            for dim, stride in enumerate(strides):
                assert isinstance(stride, torch.SymInt), (
                    form,
                    strides,
                    projected.stride(),
                )
                origin = bound.host_function.expr_to_origin[stride._sympy_()].origin
                assert eval(
                    origin.host_str(),
                    {},
                    {"projected": projected, "selected": x if limit == 0 else y},
                ) == projected.stride(dim)
            if bound.settings.backend == "triton":
                assert [arg for arg in launches[-1] if type(arg) is int] == list(
                    projected.stride()
                )


@pytest.mark.cuda
@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("form", ["branch", "advanced", "comparison"])
def test_host_tensor_layout_exits_copy_current_values(form):
    if form == "branch":
        x = torch.arange(16, dtype=torch.float32, device=DEVICE).reshape(4, 4)
        y = x.t()
        kernel = _direct_branch_layout
        tail = ()
    else:
        x = torch.arange(64, dtype=torch.float32, device=DEVICE).reshape(4, 4, 4)
        y = x.transpose(1, 2)
        kernel = _projected_layout
        tail = (form,)
    original = x.clone()
    bound = kernel.bind((x, y, 0, *tail))
    compiled = bound.compile_config(bound.config_spec.default_config())
    for selection in (0, 1, 0):
        assert kernel.bind((x, y, selection, *tail)) is bound
        expected = (
            (x if selection else y)
            if form == "branch"
            else _projected_reference(x if selection == 0 else y, form)
        )
        out = compiled(x, y, selection, *tail)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@helion.kernel(static_shapes=True, autotune_effort="none")
def _fixed_layout_destination(x, y, destination, limit: int, form: hl.constexpr):
    vals = {0: value for key, value in ((0, x), (1, y)) if key <= limit}  # noqa: B035 - exercise colliding keys
    selected = vals[0]
    if form == "copy":
        destination.copy_(selected)
        fixed = destination[:, :, :]
    elif form == "out":
        torch.add(selected, 0, out=destination)
        fixed = destination[:, :, :]
    elif form == "new_empty":
        fixed = selected.new_empty(x.shape)
    elif form == "new_strided":
        fixed = selected.new_empty_strided(x.shape, (16, 1, 4))
    elif form == "new_zeros":
        fixed = selected.new_zeros(x.shape)
    elif form == "new_ones":
        fixed = selected.new_ones(x.shape)
    else:
        fixed = selected.new_full(x.shape, 7)
    out = torch.empty_like(destination)
    for row, column, depth in hl.tile(x.shape):
        out[row, column, depth] = fixed[row, column, depth]
    return out


@pytest.mark.parametrize(
    "form",
    ["copy", "out", "new_empty", "new_strided", "new_zeros", "new_ones", "new_full"],
)
def test_host_fixed_storage_layout_does_not_inherit_selected_strides(form):
    x = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
    y = x.transpose(1, 2)
    destination = torch.empty_like(x)
    with cpu_render_target():
        bound = _fixed_layout_destination.bind((x, y, destination, 0, form))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
        )
        expected_stride = (16, 1, 4) if form == "new_strided" else (16, 4, 1)
        fixed_type = bound.host_function.local_types["fixed"]
        assert fixed_type.fake_value.stride() == expected_stride
        assert all(type(stride) is int for stride in fixed_type.fake_value.stride())
        for limit in (0, 1, 0):
            assert (
                _fixed_layout_destination.bind((x, y, destination, limit, form))
                is bound
            )
            out = host(x, y, destination, limit, form)
            fixed = next(
                arg
                for arg in launches[-1]
                if isinstance(arg, torch.Tensor) and arg is not out
            )
            assert fixed.stride() == expected_stride
            if form in ("copy", "out"):
                assert fixed.untyped_storage() is destination.untyped_storage()
                torch.testing.assert_close(
                    fixed, x if limit == 0 else y, rtol=0, atol=0
                )
            else:
                assert fixed.untyped_storage() is not destination.untyped_storage()
                assert bound.env.tensor_layout_is_symbolically_exact(
                    fixed_type.fake_value
                )
            if bound.settings.backend == "triton":
                assert not any(type(arg) is int for arg in launches[-1])


def test_host_marked_view_reuses_runtime_stride_expressions():
    x = torch.empty(4, 4, 4)
    with cpu_render_target():
        bound = _projected_layout.bind((x, x.transpose(1, 2), 0, "property"))
        selected = bound.host_function.local_types["selected"].fake_value
        projected = bound.host_function.local_types["projected"].fake_value
        assert selected.untyped_storage() is projected.untyped_storage()
        assert [stride._sympy_() for stride in projected.stride()] == [
            stride._sympy_() for stride in reversed(selected.stride())
        ]


@helion.kernel(static_shapes=True, autotune_effort="none")
def _resized_out_layout(x, y, limit: int, form: hl.constexpr):
    vals = {0: value for key, value in ((0, x), (1, y)) if key <= limit}  # noqa: B035 - exercise colliding keys
    selected = vals[0]
    destination = torch.empty(0, dtype=x.dtype, device=x.device)
    torch.add(selected, 0, out=destination)
    if form == "view":
        fixed = destination[:, :, :]
    else:
        fixed = destination
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    for row, column, depth in hl.tile(x.shape):
        out[row, column, depth] = fixed[row, column, depth]
    return out


@pytest.mark.parametrize("form", ["direct", "view"])
def test_host_resized_out_uses_current_layout(form):
    x = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
    y = x.transpose(1, 2)
    with cpu_render_target():
        bound = _resized_out_layout.bind((x, y, 0, form))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
        )
        strides = bound.host_function.local_types["destination"].fake_value.stride()
        assert all(isinstance(stride, torch.SymInt) for stride in strides)
        for limit in (0, 1, 0):
            assert _resized_out_layout.bind((x, y, limit, form)) is bound
            out = host(x, y, limit, form)
            destination = next(
                arg
                for arg in launches[-1]
                if isinstance(arg, torch.Tensor) and arg is not out
            )
            expected = x if limit == 0 else y
            assert destination.stride() == expected.stride()
            torch.testing.assert_close(destination, expected, rtol=0, atol=0)
            for dim, stride in enumerate(strides):
                origin = bound.host_function.expr_to_origin[stride._sympy_()].origin
                assert eval(
                    origin.host_str(),
                    {},
                    {"destination": destination, "fixed": destination},
                ) == expected.stride(dim)
            if bound.settings.backend == "triton":
                assert [arg for arg in launches[-1] if type(arg) is int] == list(
                    expected.stride()
                )


@pytest.mark.parametrize("transpose", [False, True])
def test_host_branch_preserves_common_nondense_layout(transpose):
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2]
    y = (torch.arange(32, dtype=torch.float32) + 100).reshape(4, 8)[:, ::2]
    if transpose:
        x, y = x.t(), y.t()
    with cpu_render_target():
        bound = _direct_branch_layout.bind((x, y, 1))
        selected = bound.host_function.local_types["selected"].fake_value
        assert selected.stride() == x.stride() == y.stride()
        assert all(type(stride) is int for stride in selected.stride())
        assert all(
            selected.untyped_storage() is not input_tensor.untyped_storage()
            for input_tensor in bound.env.input_sources
        )
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
        )
        for flag in (1, 0, 1):
            assert _direct_branch_layout.bind((x, y, flag)) is bound
            host(x, y, flag)
            assert any(arg is (x if flag else y) for arg in launches[-1])
            if bound.settings.backend == "triton":
                assert not any(type(arg) is int for arg in launches[-1])


@pytest.mark.cuda
@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("form", ["direct", "view", "common_nondense"])
def test_host_resized_and_common_branch_layouts_copy_values(form):
    if form != "common_nondense":
        x = torch.arange(64, dtype=torch.float32, device=DEVICE).reshape(4, 4, 4)
        y = x.transpose(1, 2)
        kernel = _resized_out_layout
        tail = (form,)
    else:
        x = torch.arange(32, dtype=torch.float32, device=DEVICE).reshape(4, 8)[:, ::2]
        y = (torch.arange(32, dtype=torch.float32, device=DEVICE) + 100).reshape(4, 8)[
            :, ::2
        ]
        kernel = _direct_branch_layout
        tail = ()
    original_x, original_y = x.clone(), y.clone()
    bound = kernel.bind((x, y, 0, *tail))
    compiled = bound.compile_config(bound.config_spec.default_config())
    for selection in (0, 1, 0):
        assert kernel.bind((x, y, selection, *tail)) is bound
        expected = (
            (x if selection == 0 else y)
            if form != "common_nondense"
            else (x if selection else y)
        )
        torch.testing.assert_close(
            compiled(x, y, selection, *tail), expected, rtol=0, atol=0
        )
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(y, original_y, rtol=0, atol=0)


@helion.kernel(static_shapes=False, autotune_effort="none")
def _shape_factory_layout(x, form: hl.constexpr):
    if form == "empty_strided":
        created = torch.empty_strided(
            x.shape, (16, 1, 4), dtype=x.dtype, device=x.device
        )
    elif form == "zeros":
        created = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    elif form == "ones":
        created = torch.ones(x.shape, dtype=x.dtype, device=x.device)
    else:
        created = torch.full(x.shape, 7, dtype=x.dtype, device=x.device)
    out = torch.empty_like(created)
    for row, column, depth in hl.tile(x.shape):
        out[row, column, depth] = created[row, column, depth]
    return out


@pytest.mark.parametrize("form", ["empty_strided", "zeros", "ones", "full"])
def test_host_shape_factories_preserve_exact_layout_with_dynamic_shapes(form):
    x = torch.empty(4, 4, 4)
    with cpu_render_target():
        bound = _shape_factory_layout.bind((x, form))
        created_type = bound.host_function.local_types["created"]
        assert bound.env.tensor_layout_is_symbolically_exact(created_type.fake_value)
        launches = []
        config = bound.config_spec.default_config()
        host = load_host_function(
            bound,
            config,
            lambda kernel, grid, *args, **kwargs: launches.append(args),
        )
        out = host(x, form)
        created = next(
            arg
            for arg in launches[-1]
            if isinstance(arg, torch.Tensor) and arg is not out
        )
        assert created.shape == x.shape
        assert created.stride() == (
            (16, 1, 4) if form == "empty_strided" else (16, 4, 1)
        )
        if form != "empty_strided":
            expected = {"zeros": 0, "ones": 1, "full": 7}[form]
            torch.testing.assert_close(
                created, torch.full_like(created, expected), rtol=0, atol=0
            )
        if bound.settings.backend == "triton":
            source = bound.to_code(config)
            if form == "empty_strided":
                assert "created.stride(" not in source
            else:
                assert "created.stride(2)" not in source


@helion.kernel(static_shapes=False, autotune_effort="none")
def _factory_like_layout(x, form: hl.constexpr):
    if form == "torch_runtime":
        created = torch.empty_strided(
            x.shape, x.stride(), dtype=x.dtype, device=x.device
        )
    elif form == "method_runtime":
        created = x.new_empty_strided(x.shape, x.stride())
    elif form == "torch_literal":
        created = torch.empty_strided(x.shape, (4, 1), dtype=x.dtype, device=x.device)
    elif form == "method_literal":
        created = x.new_empty_strided(x.shape, (4, 1))
    else:
        created = torch.empty_strided((4, 4), (4, 1), dtype=x.dtype, device=x.device)
    out = torch.empty_like(created)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile]
    return out


def _factory_like_inputs(form, device="cpu"):
    if form.endswith("runtime"):
        first = torch.arange(32, dtype=torch.float32, device=device).reshape(4, 8)[
            :, ::2
        ]
        second = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4).t()
    else:
        first = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        second = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
    return first, second


@pytest.mark.parametrize(
    "form", ["torch_runtime", "method_runtime", "torch_literal", "method_literal"]
)
def test_empty_like_replays_shape_and_stride_dependent_layout(form):
    first, second = _factory_like_inputs(form)
    with cpu_render_target():
        first_bound = _factory_like_layout.bind((first, form))
        hosts = {}
        outputs = []
        for x in (first, second, first):
            bound = _factory_like_layout.bind((x, form))
            # CuTe alignment guards and literal-stride density guards can
            # rebind. Only the Triton runtime-stride case promises reuse.
            if bound.settings.backend == "triton" and form.endswith("runtime"):
                assert bound is first_bound
            if bound not in hosts:
                launches = []
                config = bound.config_spec.default_config()
                host = load_host_function(
                    bound,
                    config,
                    lambda kernel, grid, *args, records=launches, **kwargs: (
                        records.append(args)
                    ),
                )
                hosts[bound] = host, launches
            host, launches = hosts[bound]
            out = host(x, form)
            expected = torch.empty_strided(
                x.shape, x.stride() if form.endswith("runtime") else (4, 1)
            )
            assert out.shape == x.shape
            assert out.stride() == torch.empty_like(expected).stride()
            assert any(arg is out for arg in launches[-1])
            outputs.append(out)
            created = bound.host_function.local_types["created"].fake_value
            output = bound.host_function.local_types["out"].fake_value
            assert bound.env.tensor_layout_is_symbolically_exact(created)
            exact = bound.env.tensor_layout_is_symbolically_exact(output)
            # The literal factory's width==4 guard proves canonical density;
            # its width==2 binding must still use the compacted runtime layout.
            if form.endswith("runtime") or x.shape[1] != 4:
                assert not exact
            else:
                assert exact
            if not exact:
                for dim, stride in enumerate(output.stride()):
                    assert isinstance(stride, torch.SymInt)
                    origin = bound.host_function.expr_to_origin[stride._sympy_()].origin
                    assert eval(origin.host_str(), {}, {"out": out}) == out.stride(dim)
                if bound.settings.backend == "triton":
                    source = bound.to_code(bound.config_spec.default_config())
                    assert "out.stride(0)" in source and "out.stride(1)" in source
        assert outputs[0].stride() != outputs[1].stride()


def test_empty_like_keeps_fully_literal_layout_exact():
    x = torch.empty(4, 4)
    with cpu_render_target():
        bound = _factory_like_layout.bind((x, "fixed"))
        output = bound.host_function.local_types["out"].fake_value
        assert output.stride() == (4, 1)
        assert bound.env.tensor_layout_is_symbolically_exact(output)
        if bound.settings.backend == "triton":
            assert "out.stride(" not in bound.to_code()


@pytest.mark.cuda
@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("form", ["torch_runtime", "method_runtime", "torch_literal"])
def test_empty_like_compiled_copy_tracks_factory_layout(form):
    first, second = _factory_like_inputs(form, DEVICE)
    originals = first.clone(), second.clone()
    first_bound = _factory_like_layout.bind((first, form))
    compiled = {}
    for x in (first, second, first):
        bound = _factory_like_layout.bind((x, form))
        if bound.settings.backend == "triton" and form.endswith("runtime"):
            assert bound is first_bound
        if bound not in compiled:
            compiled[bound] = bound.compile_config(bound.config_spec.default_config())
        out = compiled[bound](x, form)
        torch.testing.assert_close(out, x, rtol=0, atol=0)
        expected = torch.empty_strided(
            x.shape, x.stride() if form.endswith("runtime") else (4, 1)
        )
        assert out.stride() == torch.empty_like(expected).stride()
    for actual, original in zip((first, second), originals, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@helion.kernel(static_shapes=False, autotune_effort="none")
def _canonical_input_like(x, form: hl.constexpr):
    if form == "factory":
        created = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    elif form == "derived":
        created = torch.empty((4, x.shape[0] - 3), dtype=x.dtype, device=x.device)
    elif form == "view":
        created = x.view(-1)
    else:
        created = x
    out = torch.empty_like(created)
    for tile in hl.tile(created.shape):
        out[tile] = created[tile]
    return out


@pytest.mark.parametrize("form", ["factory", "input", "view"])
def test_empty_like_proves_symbolic_canonical_layout(form):
    x = torch.empty(4, 4)
    with cpu_render_target():
        bound = _canonical_input_like.bind((x, form))
        output = bound.host_function.local_types["out"].fake_value
        assert bound.env.tensor_layout_is_symbolically_exact(output)
        assert output.stride(-1) == 1
        source = bound.to_code(bound.config_spec.default_config())
        assert f"out.stride({output.ndim - 1})" not in source


def test_empty_like_input_layout_guard_changes_with_runtime_metadata():
    canonical = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    transpose = canonical.t()
    stepped = torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2]
    with cpu_render_target():
        first = _canonical_input_like.bind((canonical, "input"))
        for value in (canonical, transpose, stepped, canonical):
            bound = _canonical_input_like.bind((value, "input"))
            output = bound.host_function.local_types["out"].fake_value
            if value is canonical:
                assert bound is first
                assert bound.env.tensor_layout_is_symbolically_exact(output)
            else:
                assert bound is not first
                assert not bound.env.tensor_layout_is_symbolically_exact(output)
            host = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            assert host(value, "input").stride() == torch.empty_like(value).stride()


@onlyBackends(["cute"])
def test_empty_like_nondegenerate_rank_one_keeps_topk_output_layout():
    @helion.kernel(static_shapes=True, autotune_effort="none")
    def topk_sum(x, stride: int):
        tmp = torch.empty_strided(
            (x.shape[0],), (stride,), dtype=x.dtype, device=x.device
        )
        out = torch.empty_like(tmp)
        for tile in hl.tile(x.shape[0]):
            values, _ = torch.topk(x[tile, :], 4, dim=-1)
            out[tile] = values.sum(-1)
        return out

    with cpu_render_target():
        bound = cpu_bind(topk_sum, (torch.empty(40, 64), 2))
        out = bound.host_function.local_types["out"].fake_value
        assert out.stride() == (1,)
        assert bound.env.tensor_layout_is_symbolically_exact(out)
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [8]
        assert "topk" in bound.to_code(config).lower()


@pytest.mark.parametrize("form", ["factory", "input", "view"])
def test_empty_like_canonical_guard_includes_empty_and_singleton_strides(form):
    shapes = ((4, 4), (4, 0), (4, 1), (0, 4), (1, 4))
    with cpu_render_target():
        for shape in shapes:
            for unusual in (False, True):
                x = torch.empty(shape)
                if unusual and form != "view":
                    x = torch.empty_strided(shape, (12, 2))
                assert _canonical_contiguous_layout((x,)) is (
                    not unusual or form == "view"
                )
                bound = _canonical_input_like.bind((x, form))
                host = load_host_function(
                    bound,
                    bound.config_spec.default_config(),
                    lambda *args, **kwargs: None,
                )
                out = host(x, form)
                expected = (
                    torch.empty_like(x.view(-1))
                    if form == "view"
                    else torch.empty(x.shape)
                    if form == "factory"
                    else torch.empty_like(x)
                )
                assert out.stride() == expected.stride()
                fake = bound.host_function.local_types["out"].fake_value
                if bound.env.tensor_layout_is_symbolically_exact(fake):
                    for dim, stride in enumerate(fake.stride()):
                        if type(stride) is int:
                            assert stride == out.stride(dim)
                        else:
                            expression = stride._sympy_()
                            substitutions = {
                                symbol: eval(
                                    bound.host_function.expr_to_origin[
                                        symbol
                                    ].origin.host_str(),
                                    {},
                                    {"x": x, "out": out},
                                )
                                for symbol in expression.free_symbols
                            }
                            assert int(expression.subs(substitutions)) == out.stride(
                                dim
                            )


@pytest.mark.cuda
@onlyBackends(["triton", "cute"])
def test_empty_like_canonical_guard_dispatch_copies_runtime_layouts():
    x = torch.arange(16, device=DEVICE, dtype=torch.float32).reshape(4, 4)
    stepped = torch.arange(32, device=DEVICE, dtype=torch.float32).reshape(4, 8)[:, ::2]
    for value in (x, x.t(), stepped, x, torch.ones(4, 1, device=DEVICE)):
        out = _canonical_input_like(value, "input")
        torch.testing.assert_close(out, value, rtol=0, atol=0)
        assert out.stride() == torch.empty_like(value).stride()


def test_empty_like_derived_extent_class_is_guarded():
    with cpu_render_target():
        bindings = {}
        for n in (6, 5, 4, 3, 6):
            x = torch.empty(n)
            bound = _canonical_input_like.bind((x, "derived"))
            if n == 6 and 6 in bindings:
                assert bound is bindings[6]
            bindings[n] = bound
            host = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            out = host(x, "derived")
            assert out.shape == (4, n - 3)
            assert out.stride() == (max(n - 3, 1), 1)
            fake = bound.host_function.local_types["out"].fake_value
            assert bound.env.tensor_layout_is_symbolically_exact(fake)
            for dim, stride in enumerate(fake.stride()):
                expression = (
                    stride._sympy_() if isinstance(stride, torch.SymInt) else stride
                )
                if isinstance(stride, torch.SymInt):
                    expression = expression.subs(
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
                assert int(expression) == out.stride(dim)
        assert bindings[3] is not bindings[4] and bindings[4] is not bindings[6]


def test_empty_like_rank_one_preserves_requires_grad():
    @helion.kernel(static_shapes=True, autotune_effort="none")
    def copy_grad(x):
        out = torch.empty_like(x, requires_grad=True)
        for tile in hl.tile(x.shape):
            out[tile] = x[tile]
        return out

    with cpu_render_target(), torch.enable_grad():
        x = torch.ones(40)
        bound = copy_grad.bind((x,))
        fake = bound.host_function.local_types["out"].fake_value
        assert fake.requires_grad and fake.stride() == (1,)
        host = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        out = host(x)
        assert out.requires_grad and out.stride() == (1,)
        assert torch.is_grad_enabled()
