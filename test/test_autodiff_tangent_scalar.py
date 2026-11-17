"""CPU checks for autodiff tangent padding and singleton shape effects."""

from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch
from torch._inductor.codecache import PyCodeCache
from torch.fx.experimental.proxy_tensor import make_fx

import helion
from helion._compiler.compile_environment import FixedBlockSizeSource
from helion._compiler.device_ir import RootGraphInfo
from helion._testing import cpu_render_target
from helion._testing import matchesBackends
from helion._testing import skipUnlessBackends
from helion._testing.autodiff_shapes import ALIAS_SHAPE_CASES
from helion._testing.autodiff_shapes import alias_shape_backward
from helion._testing.autodiff_shapes import alias_shape_case
from helion._testing.cute_codegen import cpu_bind
from helion.experimental.autodiff import FXToHelionConverter
from helion.experimental.autodiff import GraphAnalyzer
from helion.experimental.autodiff import InputMapping
from helion.experimental.autodiff import _collect_reduce_dims
from helion.experimental.autodiff import _collect_store_reduce_dims
from helion.experimental.autodiff import _reduction_dims_and_keepdim
from helion.experimental.autodiff import differentiate_graph
import helion.language as hl
from helion.runtime.ref_mode import RefMode

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["triton", "cute"])]


# Each named case has a distinct coupled shape/dataflow proof; the rationale
# and complete list live beside ALIAS_SHAPE_CASES, rather than a product grid.
@pytest.mark.parametrize("case", ALIAS_SHAPE_CASES)
def test_alias_branches_preserve_compatible_physical_ranks(case: str) -> None:
    forward, inputs, blocks = alias_shape_case(case, "cpu")
    outputs = forward(*inputs)
    graph, converter, module, source = alias_shape_backward(forward, inputs, outputs)
    view_rewrites = case in (
        "scalar_mask",
        "broadcast_mask",
        "independent_outputs",
        "shared_nonlinear_outputs",
        "shared_scalar_gate",
        "reduced_expanded_mask",
        "counted_expanded_mask",
    )
    local_expand = case in (
        "reduced_expanded_mask",
        "counted_expanded_mask",
        "broadcast_product",
        "nonlinear_broadcast_product",
        "sum_then_expand",
        "sum_then_count",
        "unsqueeze_expand_mask",
        "unsqueeze_counted_mask",
        "sliced_unsqueeze_mask",
        "sliced_unsqueeze_count",
        "singleton_row_expand",
    )
    normalized = view_rewrites or local_expand
    assert (converter.backward_graph is not graph) == normalized
    if normalized:
        targets = (torch.ops.aten.unsqueeze.default, torch.ops.aten.squeeze.dim)
        added_views = sum(
            node.target in targets for node in converter.backward_graph.nodes
        ) > sum(node.target in targets for node in graph.nodes)
        assert added_views == view_rewrites
    assert sum(
        bool(node.meta.get("helion_local_expand"))
        for node in converter.backward_graph.nodes
    ) == int(local_expand)
    with cpu_render_target():
        bound = cpu_bind(
            cast("helion.Kernel[object]", module.backward_kernel),
            (*(torch.ones_like(output) for output in outputs), *inputs),
        )
        bound.to_code(helion.Config(block_sizes=blocks))

    if case in ("sliced_unsqueeze_mask", "sliced_unsqueeze_count"):
        assert "input_1_tile = input_1[:]" in source
    if case == "singleton_row_expand":
        assert "input_1_tile = input_1[0:1," in source


def test_unsupported_multi_axis_squeeze_is_rejected_before_compilation() -> None:
    def forward(value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value.squeeze().sin().view(value.shape),)

    inputs = (torch.randn(1, 8, 1, 16),)
    # The legacy alias path emits two extra axes and fails native compilation.
    # Preserve the explicit rejection instead of pretending this graph works.
    with pytest.raises(helion.exc.AutodiffNotSupported, match="squeeze without dim"):
        alias_shape_backward(forward, inputs, forward(*inputs))


def test_local_expand_rejects_unmappable_input() -> None:
    def forward(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        counts = mask.expand(8, 2).sum(1, keepdim=True)
        return (torch.where((counts > 1).unsqueeze(-1), value, 0),)

    inputs = (torch.randn(8, 16, 32), torch.ones(8, 1, dtype=torch.bool))
    # The singleton mask axis cannot map to any iteration axis. The loader
    # cannot establish a physical shape for the local replication proof.
    with pytest.raises(
        helion.exc.AutodiffNotSupported,
        match="local expand requires a logically shaped input",
    ):
        alias_shape_backward(forward, inputs, forward(*inputs))


def test_local_expand_rejects_tiled_singleton_reduction_input() -> None:
    def forward(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        counts = mask.expand(1, 2).sum(1, keepdim=True)
        return (torch.where(counts > 1, value, 0),)

    inputs = (torch.randn(8, 16), torch.ones(1, 1, dtype=torch.bool))
    # The full-rank reduction load still tiles the leading singleton using
    # the eight-row iteration coordinate. It cannot prove a literal row.
    with pytest.raises(
        helion.exc.AutodiffNotSupported,
        match="local expand requires a logically shaped input",
    ):
        alias_shape_backward(forward, inputs, forward(*inputs))


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("operation", ["sum", "mean", "scaled_mean", "negated_sum"])
def test_scalar_tangent_arithmetic_keeps_the_existing_broadcast_axis(
    dim: int, operation: str
) -> None:
    def forward(x: torch.Tensor) -> tuple[torch.Tensor]:
        if operation == "sum":
            return (x.sum(dim),)
        if operation == "mean":
            return (x.mean(dim),)
        if operation == "scaled_mean":
            return (3 * x.mean(dim),)
        return (-x.sum(dim),)

    _render_backward(forward, dim)


@pytest.mark.parametrize(
    "operation,axes,view_dim",
    [
        ("sum", (0, 1), None),
        ("sum", (0, 2), None),
        ("sum", (1, 2), None),
        ("mean", (0, 1), None),
        ("mean", (0, 2), None),
        ("mean", (1, 2), None),
        ("sum", (1, 2), -1),
        ("sum", (1, 2), 0),
    ],
    ids=[
        "sum-leading",
        "sum-separated",
        "sum-trailing",
        "mean-leading",
        "mean-separated",
        "mean-trailing",
        "trailing-view-roundtrip",
        "leading-view-roundtrip",
    ],
)
def test_multi_axis_tangent_scalar_views_preserve_gradient_values(
    operation: str, axes: tuple[int, int], view_dim: int | None
) -> None:
    def forward(x: torch.Tensor) -> tuple[torch.Tensor]:
        y = x.sum(axes) if operation == "sum" else x.mean(axes)
        if view_dim is None:
            return (y,)
        return (y + (y.unsqueeze(view_dim) * 3).squeeze(view_dim),)

    x = torch.zeros(8, 16, 32)
    # Different gradient entries expose accidental broadcasting across the
    # two padding axes; a uniform gradient would hide the wrong values.
    gradient = torch.arange(1, forward(x)[0].numel() + 1, dtype=x.dtype)
    graph = differentiate_graph(make_fx(forward)(x).graph, (x,))
    converter = FXToHelionConverter(
        graph,
        [InputMapping("x_1", "x", x)],
        (x,),
        (tuple(gradient.shape),),
        forward_reduce_dims=axes,
    )
    module = PyCodeCache.load(converter.convert())
    config = helion.Config(block_sizes=[2, 4, 8])
    with cpu_render_target():
        eager = helion.kernel(
            module.backward_kernel.fn,
            config=config,
            ref_mode=RefMode.EAGER,
            autotune_effort="none",
        )
        actual = eager(gradient, x)
        if isinstance(actual, tuple):
            (actual,) = actual
        x_ref = x.detach().requires_grad_()
        (expected,) = torch.autograd.grad(forward(x_ref), (x_ref,), (gradient,))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        bound = cpu_bind(
            cast("helion.Kernel[object]", module.backward_kernel), (gradient, x)
        )
        bound.to_code(config)


def _render_backward(
    forward: Callable[[torch.Tensor], tuple[torch.Tensor]],
    dim: int,
    cast_kind: str | None = None,
) -> None:
    x = torch.randn(8, 16, 32)
    graph = make_fx(forward)(x).graph
    gradient = torch.randn_like(forward(x)[0])
    backward_graph = differentiate_graph(graph, (x,))
    if cast_kind is not None:
        casts = [
            node
            for node in backward_graph.nodes
            if node.target
            in (
                torch.ops.aten._to_copy.default,
                torch.ops.prims.convert_element_type.default,
            )
        ]
        assert casts
        for node in casts:
            dtype = node.kwargs["dtype"] if node.kwargs else node.args[1]
            if cast_kind == "prims":
                node.target = torch.ops.prims.convert_element_type.default
                node.args = (node.args[0], dtype)
                node.kwargs = {}
            else:
                node.target = torch.ops.aten._to_copy.default
                node.args = (node.args[0],)
                node.kwargs = {"dtype": dtype}
    converter = FXToHelionConverter(
        backward_graph,
        [InputMapping("x_1", "x", x)],
        (x,),
        (tuple(gradient.shape),),
        forward_reduce_dims=(dim,),
    )
    source = converter.convert()
    # The load owns the None index. Arithmetic and casts must not insert it again.
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id.endswith(("_tile", "_val"))
        ):
            assert not any(
                isinstance(index, ast.Constant) and index.value is None
                for index in ast.walk(node.slice)
            )
    module = PyCodeCache.load(source)
    with cpu_render_target():
        bound = cpu_bind(
            cast("helion.Kernel[object]", module.backward_kernel), (gradient, x)
        )
        # Different block sizes distinguish tile coordinates from global shapes.
        bound.to_code(helion.Config(block_sizes=[2, 4, 8]))


@pytest.mark.parametrize("dim,cast_kind", [(1, "aten"), (2, "prims")])
def test_dtype_cast_preserves_the_tangent_broadcast_axis(
    dim: int, cast_kind: str
) -> None:
    def forward(x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x.mean(dim).to(torch.bfloat16),)

    _render_backward(forward, dim, cast_kind)


@pytest.mark.parametrize(
    "identity",
    [
        "plain",
        "unsqueeze_squeeze",
        "view",
        "scalar_between_views",
        "cast_between_views",
        "scalar_middle_views",
    ],
)
def test_mixed_outputs_do_not_prebroadcast_the_full_rank_tangent(identity: str) -> None:
    def forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = 2 * x
        if identity == "unsqueeze_squeeze":
            full = full.unsqueeze(-1).squeeze(-1)
        elif identity == "view":
            full = full.view(x.shape)
        elif identity == "scalar_between_views":
            full = (full.unsqueeze(-1) * 3).squeeze(-1)
        elif identity == "cast_between_views":
            full = (full.unsqueeze(-1).to(torch.bfloat16) * 3).squeeze(-1)
        elif identity == "scalar_middle_views":
            full = (full.unsqueeze(1) * 3).squeeze(1)
        return full, x.mean(-1)

    x = torch.randn(8, 16, 32)
    outputs = forward(x)
    backward_graph = differentiate_graph(make_fx(forward)(x).graph, (x,))
    converter = FXToHelionConverter(
        backward_graph,
        [InputMapping("x_1", "x", x)],
        (x,),
        tuple(tuple(output.shape) for output in outputs),
        forward_reduce_dims=(2,),
    )
    source = converter.convert()
    source_tree = ast.parse(source)
    for name, padded in (("grad_out_0", False), ("grad_out_1", True)):
        loads = [
            node
            for node in ast.walk(source_tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == name
        ]
        assert len(loads) == 1
        assert (
            any(
                isinstance(index, ast.Constant) and index.value is None
                for index in ast.walk(loads[0].slice)
            )
            == padded
        )
    if identity == "unsqueeze_squeeze":
        # Unlike the reduced tangent, this load has no padding to consume.
        assert any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "grad_out_0_tile"
            and any(
                isinstance(index, ast.Constant) and index.value is None
                for index in ast.walk(node.slice)
            )
            for node in ast.walk(source_tree)
        )
    module = PyCodeCache.load(source)
    with cpu_render_target():
        bound = cpu_bind(
            cast("helion.Kernel[object]", module.backward_kernel),
            (torch.ones_like(outputs[0]), torch.ones_like(outputs[1]), x),
        )
        bound.to_code(helion.Config(block_sizes=[2, 4, 8]))


# Distinct proof edges, not a Cartesian grid: endpoint kind, completed/nested
# pairs, padding orientation, and an unpaired consumer whose shape must stay intact.
@pytest.mark.parametrize(
    "case",
    [
        "vector_endpoint",
        "vector_scalar_endpoint",
        "identity_view",
        "nested",
        "reduced_last",
        "reduced_middle",
        "reduced_vector_leading",
        "reduced_vector_trailing",
        "unpaired_transpose",
    ],
)
def test_inverse_views_restore_existing_padding(case: str) -> None:
    if case in ("vector_endpoint", "vector_scalar_endpoint", "identity_view"):
        shape, blocks = (128,), [16]
    elif case == "unpaired_transpose":
        shape, blocks = (128, 1), [16, 1]
    elif case.startswith("reduced_vector"):
        shape, blocks = (8, 16), [2, 4]
    else:
        shape, blocks = (8, 16, 32), [2, 4, 8]
    dim = {
        "reduced_last": 2,
        "reduced_middle": 1,
        "reduced_vector_leading": 0,
        "reduced_vector_trailing": 1,
    }.get(case)

    def forward(x: torch.Tensor) -> tuple[torch.Tensor]:
        value = 2 * x
        if case == "vector_endpoint":
            value = value.view(-1, 1).squeeze(-1)
        elif case == "vector_scalar_endpoint":
            value = (3 * value.view(-1, 1)).squeeze(-1)
        elif case == "identity_view":
            value = (3 * value.unsqueeze(-1).view(128, 1)).squeeze(-1)
        elif case == "nested":
            value = value.unsqueeze(-1).unsqueeze(0).squeeze(0).squeeze(-1)
        elif case == "unpaired_transpose":
            value = value.t().squeeze(0)
        else:
            assert dim is not None
            value = value.mean(dim)
            if case == "reduced_vector_trailing":
                value = (3 * value.unsqueeze(0)).squeeze(0)
            elif case == "reduced_vector_leading":
                value = (3 * value.view(-1, 1)).squeeze(-1)
            else:
                value = (3 * value.unsqueeze(-1)).squeeze(-1)
        return (value,)

    x = torch.randn(shape)
    output = forward(x)[0]
    backward_graph = differentiate_graph(make_fx(forward)(x).graph, (x,))
    converter = FXToHelionConverter(
        backward_graph,
        [InputMapping("x_1", "x", x)],
        (x,),
        (tuple(output.shape),),
        forward_reduce_dims=() if dim is None else (dim,),
    )
    module = PyCodeCache.load(converter.convert())
    with cpu_render_target():
        bound = cpu_bind(
            cast("helion.Kernel[object]", module.backward_kernel),
            (torch.ones_like(output), x),
        )
        bound.to_code(helion.Config(block_sizes=blocks))


@pytest.mark.parametrize(
    "case",
    [
        "vector",
        "view",
        "nested",
        "reduced_middle",
        "reshaped_padding",
        "elided_padding",
        "incompatible_axes",
    ],
)
def test_inverse_views_join_compatible_tangent_branches(case: str) -> None:
    if case in ("vector", "view"):
        shape, blocks = (128,), [16]
    elif case == "reshaped_padding":
        shape, blocks = (8, 16), [2, 4]
    elif case == "incompatible_axes":
        shape, blocks = (8, 1), [2]
    else:
        shape, blocks = (8, 16, 32), [2, 4, 8]

    def forward(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if case == "incompatible_axes":
            value = (2 * x).unsqueeze(-1)
            return value.squeeze(-1), (3 * value).squeeze(1), (5 * x).expand(-1, 16)
        if case == "reshaped_padding":
            value = x.mean(1).unsqueeze(0)
            return value.squeeze(0), (3 * value).squeeze(0)
        if case == "elided_padding":
            value = x.mean(-1)
        else:
            value = x.mean(1) if case == "reduced_middle" else 2 * x
        value = value.unsqueeze(-1)
        if case == "view":
            return value.view(-1), (3 * value).view(-1)
        if case == "nested":
            value = value.unsqueeze(0)
            return value.squeeze(0).squeeze(-1), (3 * value).squeeze(0).squeeze(-1)
        return value.squeeze(-1), (3 * value).squeeze(-1)

    x = torch.randn(shape)
    outputs = forward(x)
    graph = differentiate_graph(make_fx(forward)(x).graph, (x,))
    converter = FXToHelionConverter(
        graph,
        [InputMapping("x_1", "x", x)],
        (x,),
        tuple(tuple(value.shape) for value in outputs),
        forward_reduce_dims=(1,)
        if case in ("reduced_middle", "reshaped_padding")
        else ((2,) if case == "elided_padding" else ()),
    )
    source = converter.convert()
    if case == "incompatible_axes":
        axes = {
            node.args[1]
            for node in graph.nodes
            if node.target is torch.ops.aten.unsqueeze.default
        }
        assert {1, 2}.issubset(axes)
        # The join has equal shapes but different inserted axes. It must keep
        # the ordinary reduction-path inverse, not select either branch's axis.
        assert ".squeeze(-1)" in source
    module = PyCodeCache.load(source)
    with cpu_render_target():
        bound = cpu_bind(
            cast("helion.Kernel[object]", module.backward_kernel),
            (*(torch.ones_like(value) for value in outputs), x),
        )
        bound.to_code(helion.Config(block_sizes=blocks))


# Every entry exercises a distinct primal/tangent shape effect: singleton
# placement, arbitrary reshape composition, loader padding or branch roundtrip.
# This is a graph-regression list rather than a Cartesian tuning grid.
@pytest.mark.parametrize(
    "case",
    [
        "single",
        "multiple",
        "middle",
        "mixed",
        "arbitrary",
        "composed_arbitrary",
        "restored_arbitrary",
        "noop_squeeze",
        "leading_singleton",
        "trailing_singleton",
        "leading_roundtrip",
        "trailing_roundtrip",
        "multiple_loader_padding",
        "expanded_padding",
        "intermediate_padding",
        "later_arbitrary",
    ],
)
def test_nonlinear_primal_and_tangent_views_share_shapes(case: str) -> None:
    # Each named graph covers a distinct shape effect; this is not a knob product.
    shape = (
        (128,)
        if case
        in (
            "single",
            "multiple",
            "arbitrary",
            "composed_arbitrary",
            "restored_arbitrary",
        )
        else (8, 16, 32)
    )
    if case in ("leading_singleton", "leading_roundtrip"):
        shape = (1, 128)
    elif case in ("trailing_singleton", "trailing_roundtrip"):
        shape = (128, 1)
    elif case in ("multiple_loader_padding", "intermediate_padding"):
        shape = (1, 8, 16, 1)
    elif case == "expanded_padding":
        shape = (8, 1, 16)
    elif case == "later_arbitrary":
        shape = (8, 16)
    blocks = (
        [1 if size == 1 else 16 for size in shape]
        if len(shape) == 2
        else ([16] if len(shape) == 1 else [2, 4, 8])
    )

    if case in ("multiple_loader_padding", "intermediate_padding"):
        blocks = [1, 2, 4, 1]
    elif case == "expanded_padding":
        blocks = [4, 1, 8]

    def forward(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if case == "later_arbitrary":
            return (x.view(8, 16, 1).sin().view(128).cos().view(8, 16),)
        if case == "multiple_loader_padding":
            return (x.view(8, 16).sin(),)
        if case == "intermediate_padding":
            return (x.view(8, 16).sin().view(8, 16, 1),)
        if case == "expanded_padding":
            return (x.view(8, 16).sin().view(8, 1, 16).sum(-1),)
        if case == "single":
            return (x.view(-1, 1).sin().squeeze(-1),)
        if case == "multiple":
            value = x.view(1, -1, 1, 1).sin()
            return (value.squeeze(0).squeeze(-1).squeeze(-1),)
        if case == "arbitrary":
            return (x.view(8, 16).sin().view(128),)
        if case == "composed_arbitrary":
            return ((2 * x.view(8, 16)).view(8, 16, 1).sin().view(128),)
        if case == "restored_arbitrary":
            return ((2 * x.view(8, 16)).view(128).view(128, 1).sin().view(128),)
        if case in ("leading_roundtrip", "trailing_roundtrip"):
            value = x.squeeze()
            return (value.sin(), (3 * value.unsqueeze(0)).sin().squeeze(0))
        if case == "leading_singleton":
            return (x.squeeze(0).sin(),)
        if case == "trailing_singleton":
            return (x.squeeze(-1).sin(),)
        if case == "noop_squeeze":
            return (x.squeeze().sin(),)
        value = x.view(8, 1, 16, 32).sin().squeeze(1)
        return (value, x.mean(-1)) if case == "mixed" else (value,)

    x = torch.randn(shape)
    outputs = forward(x)
    graph = differentiate_graph(make_fx(forward)(x).graph, (x,))
    before = str(graph)
    converter = FXToHelionConverter(
        graph,
        [InputMapping("x_1", "x", x)],
        (x,),
        tuple(tuple(value.shape) for value in outputs),
        forward_reduce_dims=(2,) if case == "mixed" else (),
    )
    source = converter.convert()
    if case == "expanded_padding":
        # Expand retains the loaded rank; the final view must not duplicate it.
        assert not any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id.endswith(("_tile", "_val"))
            and any(
                isinstance(index, ast.Constant) and index.value is None
                for index in ast.walk(node.slice)
            )
            for node in ast.walk(ast.parse(source))
        )
    module = PyCodeCache.load(source)
    assert str(graph) == before
    if case in (
        "arbitrary",
        "multiple_loader_padding",
        "intermediate_padding",
        "later_arbitrary",
    ):
        # Arbitrary reshapes and unsupported padding keep the previous graph.
        assert converter.backward_graph is graph
    with cpu_render_target():
        bound = cpu_bind(
            cast("helion.Kernel[object]", module.backward_kernel),
            (*(torch.ones_like(value) for value in outputs), x),
        )
        bound.to_code(helion.Config(block_sizes=blocks))


def test_local_expand_uses_literal_singleton_iteration_load() -> None:
    forward, inputs, blocks = alias_shape_case("literal_singleton_expand", "cpu")
    _graph, _converter, module, source = alias_shape_backward(
        forward, inputs, forward(*inputs)
    )
    tree = ast.parse(source)
    loads = {
        statement.targets[0].id: ast.unparse(statement.value)
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }
    assert loads["input_1_tile"] == "input_1[0:1, :]"
    assert loads["expand_val"] == "input_1_tile.expand(2, 2)"
    # CuTe's separate multidimensional-reduction limitation still applies;
    # the rank proof and generated singleton load are backend-independent.
    if matchesBackends(["triton"]):
        with cpu_render_target():
            bound = cpu_bind(
                cast("helion.Kernel[object]", module.backward_kernel),
                (torch.ones_like(inputs[0]), *inputs),
            )
            bound.to_code(helion.Config(block_sizes=blocks))


def test_local_expand_rejects_unmappable_direct_unsqueeze() -> None:
    def forward(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        counts = mask.unsqueeze(-1).expand(8, 1, 2).sum(-1)
        return (torch.where((counts > 1).unsqueeze(-1), value, 0),)

    inputs = (torch.randn(8, 16, 32), torch.ones(8, 1, dtype=torch.bool))
    # This direct-placeholder path is unmappable, not an actual padded load.
    # test_emitted_unsqueeze_does_not_hide_loader_padding checks that separately.
    with pytest.raises(
        helion.exc.AutodiffNotSupported,
        match="local expand requires a logically shaped input",
    ):
        alias_shape_backward(forward, inputs, forward(*inputs))


def test_emitted_unsqueeze_does_not_hide_loader_padding() -> None:
    def control(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        counts = mask.sum(-1)
        return (torch.where((counts > 1).unsqueeze(-1), value, 0),)

    def forward(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        counts = mask.sum(-1).unsqueeze(-1).expand(16, 2).sum(-1)
        return (torch.where((counts > 1).unsqueeze(-1), value, 0),)

    inputs = (torch.randn(8, 16, 32), torch.ones(16, 32, dtype=torch.bool))
    # The mask maps to axes 1 and 2, but is missing tiled axis 0. Check the
    # actual loader of the otherwise equivalent reduction-only control.
    _graph, _converter, _module, source = alias_shape_backward(
        control, inputs, control(*inputs)
    )
    assert "input_1_tile = input_1[None, tile_1, :]" in source
    with pytest.raises(
        helion.exc.AutodiffNotSupported,
        match="local expand requires a logically shaped input",
    ):
        alias_shape_backward(forward, inputs, forward(*inputs))


@pytest.mark.parametrize(
    "case",
    [
        "coinciding",
        "leading",
        "blocked",
        "rank_gate",
        "two_aliases",
        "singleton_operand",
        "sum_chain",
        "sum_chain_squeezed",
    ],
)
def test_aliased_expand_count_is_rejected(case: str) -> None:
    def forward(
        value: torch.Tensor, mask: torch.Tensor, *others: torch.Tensor
    ) -> tuple[torch.Tensor]:
        if case in ("two_aliases", "singleton_operand"):
            other = others[0]
            if case == "two_aliases":
                other = other.expand(8, 16)
            counts = (mask.expand(8, 16) & other).sum(-1, keepdim=True)
        elif case in ("sum_chain", "sum_chain_squeezed"):
            keepdim = case == "sum_chain"
            counts = mask.expand(8, 16).sum(0, keepdim=keepdim).sum(-1)
        elif case == "coinciding":
            # An iteration-sized extent is not a local literal expansion.
            counts = (mask.expand(8, 16).to(torch.int64) * 2).sum(-1, keepdim=True)
        elif case == "leading":
            counts = mask.expand(2, 16).sum(0)
        elif case == "blocked":
            # This later arbitrary reshape disables singleton normalization.
            counts = mask.expand(8, 2).sum(-1).view(4, 2).sin().view(8, 1)
        else:
            # A loader rank gap of two skips the whole normalization pass.
            counts = mask.expand(2).sum(0)
        return (torch.where(counts > 1, value, 0),)

    if case == "rank_gate":
        inputs = (torch.randn(8, 16, 32), torch.ones(1, dtype=torch.bool))
    elif case == "leading":
        inputs = (torch.randn(8, 16), torch.ones(16, dtype=torch.bool))
    else:
        inputs = (torch.randn(8, 16), torch.ones(8, 1, dtype=torch.bool))
    if case in ("two_aliases", "singleton_operand"):
        inputs = (*inputs, torch.ones(8, 1, dtype=torch.bool))
    with pytest.raises(
        helion.exc.AutodiffNotSupported,
        match="sum over an aliased expand requires explicit replication",
    ):
        alias_shape_backward(forward, inputs, forward(*inputs))


@pytest.mark.parametrize(
    "case", ["scalar_reduce", "scalar_reduce_negative", "scalar_reshape"]
)
def test_scalar_mask_aliases_still_convert(case: str) -> None:
    def forward(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        if case == "scalar_reshape":
            condition = mask.expand(8, 16).view(128)
        else:
            condition = mask.sum(-1 if case == "scalar_reduce_negative" else 0) > 0
        return (torch.where(condition, value, 0),)

    inputs = (torch.randn(128), torch.tensor(True))
    # Scalar dim=0/-1 reductions have no aliased axis to validate. The reshape
    # retains a scalar alias, which the full-sized where operand broadcasts.
    alias_shape_backward(forward, inputs, forward(*inputs))


def test_aliased_reshape_count_is_rejected() -> None:
    def forward(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        count = mask.expand(8, 16).view(128).sum(0)
        return (torch.where(count > 1, value, 0),)

    inputs = (torch.randn(128), torch.tensor(True))
    with pytest.raises(
        helion.exc.AutodiffNotSupported,
        match="sum over an aliased expand requires explicit replication",
    ):
        alias_shape_backward(forward, inputs, forward(*inputs))


def test_reduction_argument_parsing_preserves_collector_defaults() -> None:

    def store(target, index, value):
        pass

    # Each form changes argument spelling or axis semantics; no compilation.
    forms = [
        ((), {}, set(), False, {0, 1, 2}),
        ((None,), {}, set(), False, {0, 1, 2}),
        (([],), {}, set(), False, {0, 1, 2}),
        (((),), {}, set(), False, {0, 1, 2}),
        (([-1, 0],), {}, {-1, 0}, False, {0, 2}),
        ((), {"dim": -1, "keepdim": True}, {-1}, True, {2}),
        (([0, -1], True), {}, {0, -1}, True, {0, 2}),
        (([0, "unsupported"],), {}, {0}, False, {0}),
    ]
    for args, kwargs, raw, keepdim, normalized_all in forms:
        graph = torch.fx.Graph()
        source = graph.placeholder("source")
        target = graph.placeholder("target")
        reduction = graph.call_function(
            torch.ops.aten.sum.dim_IntList, (source, *args), kwargs
        )
        graph.call_function(store, (target, None, reduction))
        assert _reduction_dims_and_keepdim(reduction) == (raw, keepdim)
        assert _reduction_dims_and_keepdim(reduction, ndim=3, implicit_all=True) == (
            normalized_all,
            keepdim,
        )
        assert _collect_reduce_dims(graph) == raw
        assert _collect_reduce_dims(graph, iter_ndim=3) == {dim % 3 for dim in raw}
        assert _collect_reduce_dims(graph, skip_keepdim=True) == (
            set() if keepdim else raw
        )
        assert _collect_store_reduce_dims(graph) == {
            "target": frozenset() if keepdim else frozenset(raw)
        }

    # Filtering keepdim happens before normalization, including scalar dims.
    graph = torch.fx.Graph()
    source = graph.placeholder("scalar")
    graph.call_function(torch.ops.aten.sum.dim_IntList, (source, [0], True))
    assert _collect_reduce_dims(graph, iter_ndim=0, skip_keepdim=True) == set()


@helion.kernel(static_shapes=True, autotune_effort="none")
def _autodiff_semantic_operands(x: torch.Tensor, y: torch.Tensor, pruned: hl.constexpr):
    out = torch.empty_like(x)
    for rows in hl.tile(8, block_size=8):
        if pruned:
            out[rows] = torch.pow(1.0, x[rows]) + y[rows]
        else:
            out[rows] = torch.ops.aten.clamp.Tensor(x[rows], y[rows], None)
    return out


def test_autodiff_refuses_pruned_required_tensor():
    with cpu_render_target():
        bound = cpu_bind(
            _autodiff_semantic_operands, (torch.empty(8), torch.empty(8), True)
        )
        ir = bound.host_function.device_ir
        root = next(info for info in ir.graphs if isinstance(info, RootGraphInfo))
        before = str(root.graph)
        power = next(
            n for n in root.graph.nodes if n.target is torch.ops.aten.pow.Scalar
        )
        assert power.args == (1.0, None)
        assert not power.meta["lowering"].input_names
        analyzer = GraphAnalyzer(copy.deepcopy(root.graph), all_graphs=ir.graphs)
        # This was already unsupported: the extractor emitted pow(1,None)
        # and failed only when the extracted module executed.
        with pytest.raises(
            helion.exc.AutodiffNotSupported,
            match="required Tensor argument exponent was pruned",
        ):
            analyzer.extract_computation_graph()
        assert str(root.graph) == before


def test_autodiff_preserves_optional_tensor_none():
    with cpu_render_target():
        bound = cpu_bind(
            _autodiff_semantic_operands, (torch.empty(8), torch.empty(8), False)
        )
        ir = bound.host_function.device_ir
        root = next(info for info in ir.graphs if isinstance(info, RootGraphInfo))
        before = str(root.graph)
        # Helion canonicalizes optional-Tensor clamp to maximum. Restore its
        # equivalent legal ATen spelling on a copy at the extractor API.
        extraction_graph = copy.deepcopy(root.graph)
        clamp = next(
            n
            for n in extraction_graph.nodes
            if n.target
            in (
                torch.ops.aten.maximum.default,
                torch.ops.aten.clamp_min.Tensor,
                torch.ops.aten.clamp.Tensor,
            )
        )
        clamp.target = torch.ops.aten.clamp.Tensor
        clamp.args = (*clamp.args[:2], None)
        clamp.kwargs = {}
        analyzer = GraphAnalyzer(extraction_graph, all_graphs=ir.graphs)
        graph, mappings, _, _ = analyzer.extract_computation_graph()
        clamp = next(n for n in graph.nodes if n.target is torch.ops.aten.clamp.Tensor)
        assert clamp.args[2] is None
        values = {
            "x": torch.linspace(-2, 2, 8).requires_grad_(),
            "y": torch.zeros(8, requires_grad=True),
        }
        actual = torch.fx.GraphModule({}, graph)(
            *(values[m.tensor_name] for m in mappings)
        )[0]
        expected = torch.clamp(values["x"], min=values["y"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(actual.sum(), tuple(values.values()))
        expected_grad = torch.autograd.grad(expected.sum(), tuple(values.values()))
        for a, b in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert str(root.graph) == before


@helion.kernel(static_shapes=True, autotune_effort="none")
def _scan_pruned_tensor(x: torch.Tensor):
    out = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for rows in hl.tile(x.size(0), block_size=8):
        total = hl.zeros([rows], dtype=x.dtype)
        for columns in hl.tile(x.size(1), block_size=4):
            total = total + torch.sum(torch.pow(1.0, x[rows, columns]), dim=1)
        out[rows] = total
    return out


def test_scan_combine_refuses_pruned_required_tensor():
    with cpu_render_target():
        bound = cpu_bind(_scan_pruned_tensor, (torch.empty(8, 16),))
        ir = bound.host_function.device_ir
        power = next(
            node
            for info in ir.graphs
            for node in info.graph.nodes
            if node.target is torch.ops.aten.pow.Scalar
        )
        assert power.args == (1.0, None)
        root = next(info for info in ir.graphs if isinstance(info, RootGraphInfo))
        analyzer = GraphAnalyzer(
            root.graph,
            all_graphs=ir.graphs,
            config_block_sizes={
                block.block_id: int(block.block_size_source.value)
                for block in bound.env.block_sizes
                if isinstance(block.block_size_source, FixedBlockSizeSource)
            },
        )
        with pytest.raises(
            helion.exc.AutodiffNotSupported,
            match="required Tensor argument exponent was pruned",
        ):
            analyzer.extract_computation_graph()
