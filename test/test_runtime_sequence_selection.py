from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(static_shapes=True, autotune_effort="none")
def _select_tensor(x, y, index: int, form: hl.constexpr):
    if form == "list":
        selected = [x, y][index]
    elif form == "nested":
        selected = ((x,), (y,))[index][0]
    else:
        selected = (x, y)[index]
    out = torch.empty_like(x)
    for row, column in hl.tile(x.shape):
        out[row, column] = x[row, column] + selected[row, column]
    return out, selected


def _inputs(device="cpu"):
    x = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
    y = (x + 20).t()
    return x, y


@pytest.mark.parametrize("form", ["tuple", "list", "nested"])
def test_runtime_tensor_selection_launches_actual_pointer(form):
    x, y = _inputs()
    with cpu_render_target():
        bound = cpu_bind(_select_tensor, (x, y, 0, form))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
        )
        for index in (0, 1, 0):
            _, selected = host(x, y, index, form)
            expected = (x, y)[index]
            assert selected is expected
            assert any(arg is expected for arg in launches[-1])
            if index == 1:
                assert any(arg is y for arg in launches[-1])
            selected_type = bound.host_function.local_types["selected"]
            for dim, stride in enumerate(selected_type.fake_value.stride()):
                assert isinstance(stride, torch.SymInt)
                origin = bound.host_function.expr_to_origin[stride._sympy_()].origin
                assert eval(origin.host_str(), {}, {"selected": selected}) == (
                    expected.stride(dim)
                )


@pytest.mark.parametrize("alias", ["distinct", "view", "identity"])
def test_runtime_tensor_selection_preserves_possible_storage_owners(alias):
    x, y = _inputs()
    if alias == "view":
        y = x.t()
    elif alias == "identity":
        y = x
    with cpu_render_target():
        bound = cpu_bind(_select_tensor, (x, y, 0, "tuple"))
        env, types = bound.env, bound.host_function.local_types
        selected = types["selected"].fake_value
        owners = env.tensor_storage_provenance(selected)
        assert owners is not None
        assert {storage for storage, _ in owners} == {
            types["x"].fake_value.untyped_storage(),
            types["y"].fake_value.untyped_storage(),
        }
        assert not any(fresh for _, fresh in owners)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "rank"])
def test_runtime_tensor_selection_rejects_incompatible_candidates(mismatch):
    x, y = _inputs()
    if mismatch == "shape":
        y = torch.ones(3, 4)
    elif mismatch == "dtype":
        y = y.to(torch.int64)
    else:
        y = y.flatten()
    with cpu_render_target(), pytest.raises(helion.exc.ControlFlowTensorMismatch):
        cpu_bind(_select_tensor, (x, y, 0, "tuple"))


@helion.kernel(autotune_effort="none")
def _select_scalars(x, index: int):
    values = (2, 5)
    first = values[index]
    second = values[1 - index]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + first * 10 + second
    return out


def test_runtime_scalar_selections_have_independent_values():
    x = torch.ones(7)
    with cpu_render_target():
        bound = cpu_bind(_select_scalars, (x, 0))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
            scalar_values=True,
        )
        types = bound.host_function.local_types
        assert types["first"].value._sympy_() != types["second"].value._sympy_()
        for index in (0, 1, 0):
            host(x, index)
            assert [arg for arg in launches[-1] if type(arg) is int] == [
                (2, 5)[index],
                (2, 5)[1 - index],
            ]


@helion.kernel(static_shapes=True, autotune_effort="none")
def _select_specialized(x, y, index: int):
    selected = (x, y)[hl.specialize(index)]
    for tile in hl.tile(x.numel()):
        x[tile] = selected[tile]
    return selected


def test_specialized_sequence_index_keeps_selected_identity():
    x, y = torch.ones(7), torch.zeros(7)
    with cpu_render_target():
        for index, name in ((0, "x"), (1, "y")):
            bound = cpu_bind(_select_specialized, (x, y, index))
            types = bound.host_function.local_types
            assert types["selected"].fake_value is types[name].fake_value


@pytest.mark.cuda
@pytest.mark.parametrize("form", ["tuple", "list"])
def test_runtime_tensor_selection_native_replay(form):
    x, y = _inputs(DEVICE)
    bound = _select_tensor.bind((x, y, 0, form))
    compiled = bound.compile_config(bound.config_spec.default_config())
    for index in (0, 1, 0):
        assert _select_tensor.bind((x, y, index, form)) is bound
        out, selected = compiled(x, y, index, form)
        expected = (x, y)[index]
        assert selected is expected
        torch.testing.assert_close(out, x + expected, rtol=0, atol=0)
