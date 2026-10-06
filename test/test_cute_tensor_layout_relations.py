from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from helion._compiler.cute.tensor_layout_relations import TensorLayoutRelation
from helion._compiler.cute.tensor_layout_relations import prove_tensor_layout_relations


def _prove(prefix: str, *, rank: int = 2, arguments: str = "x, out, n"):
    body = ast.parse(
        prefix
        + "\nout = torch.empty_like(x)\n"
        + f"_launcher(kernel, (1,), {arguments}, block=(128, 1, 1))\n"
        + "return out\n"
    ).body
    return prove_tensor_layout_relations(
        body,
        kernel_name="kernel",
        parameter_names=["x", "out", "n"],
        tensor_parameters={"x": rank, "out": rank},
        integer_parameters={"n"},
        tensor_inputs={"x": rank, "other": rank},
        scalar_inputs={"independent"},
    )


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("property_name", ["size", "stride"])
@pytest.mark.parametrize("axis", [0, -1])
def test_exact_axis_and_negative_axis(rank, property_name, axis):
    result = _prove(f"n = x.{property_name}({axis})", rank=rank)
    assert result == (TensorLayoutRelation("n", "x", property_name, axis % rank),)


@pytest.mark.parametrize(
    "prefix",
    [
        "m, n = x.size()",
        "m, n = x.shape",
        "shape = x.size(); m, n = shape",
        "n = x.shape[1]",
        "n = x.size()[1]",
        "a = x; width = a.size(1); n = width",
        "n = x.size(1); other = x; x = other",
    ],
)
def test_aliases_and_shape_unpacking_keep_object_identity(prefix):
    assert _prove(prefix) == (TensorLayoutRelation("n", "x", "size", 1),)


@pytest.mark.parametrize(
    "prefix",
    [
        "n = independent",
        "n = 513",
        "n = other.size(1)",
        "n = x.size(1); x = other",
        "n = x.size(1) + 0",
        "n = x.size(1); x = torch.empty_like(x)",
    ],
)
def test_equal_hints_and_rebound_tensors_are_not_relations(prefix):
    assert _prove(prefix) == ()


@pytest.mark.parametrize(
    "prefix",
    [
        "n = x.size(1); x.resize_(1)",
        "n = x.size(1); unknown(x)",
        "n = x.size(1); ignored = unknown(x)",
        "n = x.size(1); ignored = unknown + 1",
        "n = x.size(1); ignored = x + 1",
        "n = x.size(1); ignored = unknown[0]",
        "n = x.size(1); ignored = x[0]",
        "n = x.size(1); x[0] = 1",
        "n = x.size(1); x.shape = (1, 1)",
        "if independent:\n    n = x.size(1)",
        "for i in range(2):\n    n = x.size(1)",
        "n = x.size(independent)",
        "n = x.size(2)",
        "n = x.size(-3)",
        "n = x.size(1); torch = unknown",
        "n = x.size(1); _launcher = unknown",
        "n = x.size(1); _launcher(other_kernel, (), x)",
    ],
)
def test_effects_control_flow_unknown_objects_and_invalid_axes_decline(prefix):
    assert _prove(prefix) == ()


def test_fresh_output_is_not_substituted_for_equal_shape_input():
    assert _prove("n = x.size(1)") == (TensorLayoutRelation("n", "x", "size", 1),)
    assert _prove("n = x.size(1)", arguments="out, out, n") == ()


def test_actual_launch_positions_are_resolved():
    assert _prove("n = x.size(1)", arguments="other, x, n") == (
        TensorLayoutRelation("n", "out", "size", 1),
    )


def test_metadata_only_host_work_is_allowed():
    assert _prove("n = x.size(1); q = (n + 255) // 256") == (
        TensorLayoutRelation("n", "x", "size", 1),
    )


def test_proof_size_is_bounded():
    prefix = "n = x.size(1)\n" + "q = independent + 1\n" * 1000
    assert _prove(prefix) == ()


def _prove_flatten(
    prefix: str,
    *,
    rank: int = 2,
    arguments: str = "flat, out, n",
    allow_flatten: bool = True,
):
    body = ast.parse(
        prefix
        + "\nout = torch.empty_like(flat)\n"
        + f"_launcher(kernel, (1,), {arguments}, block=(128, 1, 1))\n"
    ).body
    return prove_tensor_layout_relations(
        body,
        kernel_name="kernel",
        parameter_names=["x", "out", "n"],
        tensor_parameters={"x": 1, "out": 1},
        integer_parameters={"n"},
        tensor_inputs={"x": rank, "other": rank},
        scalar_inputs={"independent"},
        allow_flatten=allow_flatten,
    )


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "prefix",
    [
        "n = x.numel(); flat = x.view(-1)",
        "flat = x.view(-1); n = flat.numel()",
        "flat = x.view(-1); n = flat.size(0)",
        "alias = x; n = alias.numel(); flat = alias.view(-1)",
        "n = x.numel(); first = x.view(-1); flat = first.view(-1)",
        "flat = x.view(-1); n = x.numel(); alias = flat; flat = alias",
    ],
)
def test_numel_preserved_by_exact_flatten(rank, prefix):
    assert _prove_flatten(prefix, rank=rank) == (
        TensorLayoutRelation("n", "x", "size", 0),
    )


def test_rank_one_size_preserved_by_flatten_but_strides_are_not():
    assert _prove_flatten("n = x.size(0); flat = x.view(-1)", rank=1) == (
        TensorLayoutRelation("n", "x", "size", 0),
    )
    assert _prove_flatten("n = x.stride(0); flat = x.view(-1)", rank=1) == ()
    assert _prove_flatten("n = x.size(0); flat = x.view(-1)", rank=2) == ()


@pytest.mark.parametrize(
    "prefix",
    [
        "n = independent; flat = x.view(-1)",
        "n = other.numel(); flat = x.view(-1)",
        "n = x.numel(); flat = other.view(-1)",
        "n = x.numel(); flat = torch.empty_like(x).view(-1)",
        "n = x.numel() + 0; flat = x.view(-1)",
        "n = x.numel(0); flat = x.view(-1)",
        "n = x.numel(dim=0); flat = x.view(-1)",
        "n = x.numel(); flat = x.view((-1,))",
        "n = x.numel(); flat = x.view(-1.0)",
        "n = x.numel(); flat = x.view(size=-1)",
        "n = x.numel(); flat = x.view(-1, 1)",
        "n = x.numel(); flat = x.reshape(-1)",
        "n = x.numel(); flat = x.view(-1); flat.resize_(1)",
        "n = x.numel(); flat = x.view(-1); ignored = unknown(flat)",
        "n = x.numel(); flat = x.view(-1); flat = other.view(-1)",
    ],
)
def test_flatten_declines_independent_lengths_effects_and_other_spellings(prefix):
    assert _prove_flatten(prefix) == ()


def test_numel_does_not_relate_a_fresh_equal_extent_output():
    assert (
        _prove_flatten("n = x.numel(); flat = x.view(-1)", arguments="out, out, n")
        == ()
    )


def test_flatten_relations_are_opt_in_for_existing_consumers():
    assert _prove_flatten("n = x.numel(); flat = x.view(-1)", allow_flatten=False) == ()


@pytest.mark.parametrize("property_name", ["size", "stride"])
@pytest.mark.parametrize("value", [0, 1, 513, (1 << 31) + 1, (1 << 63) - 1])
def test_assignment_retains_int64_modular_arithmetic(property_name, value):
    class Int64(int):
        def __new__(cls, number):
            return super().__new__(
                cls, (int(number) + (1 << 63)) % (1 << 64) - (1 << 63)
            )

        def __mul__(self, rhs):
            return Int64(int(self) * int(rhs))

    relation = TensorLayoutRelation("n", "x", property_name, 0)
    namespace = {
        "cutlass": SimpleNamespace(Int64=Int64),
        "cute": SimpleNamespace(size=lambda tensor, mode: tensor.layout.shape[mode[0]]),
        "x": SimpleNamespace(layout=SimpleNamespace(shape=(value,), stride=(value,))),
    }
    module = ast.Module(
        body=[relation.assignment(), *ast.parse("result = n * 16").body],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<layout-relation>", "exec"),
        namespace,
    )
    assert type(namespace["n"]) is Int64
    assert namespace["result"] == Int64(value) * 16
