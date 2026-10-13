from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute.chained_loop import discover_chained_loop
from helion._compiler.cute.chained_matmul import _Expression
from helion._compiler.variable_origin import GridOrigin
from helion._compiler.variable_origin import TileBeginOrigin
from helion._compiler.variable_origin import TileIdOrigin
import helion.language as hl
from helion.language import _tracing_ops

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
_ExportArgs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _export_chain(
    left: torch.Tensor,
    right: torch.Tensor,
    initial: torch.Tensor,
    entering: torch.Tensor,
    products: torch.Tensor,
    history: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = hl.specialize(steps)
    for batch, rows in hl.tile([left.size(0), left.size(2)], block_size=[1, 16]):
        state = initial[batch.id, rows, :].float()
        for step in hl.tile(steps, block_size=1):
            entering[batch.id, step.id, rows, :] = state
            a = left[batch.id, step.id, rows, :]
            b = right[batch.id, step.id, :, :]
            product = hl.dot(a, b).to(left.dtype)
            products[batch.id, step.id, rows, :] = product
            state = hl.dot(state.to(left.dtype), b, acc=state * 0.5 + product.float())
            history[batch.id, step.id, rows, :] = state
    return entering, products, history


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_export_chain(
    left: torch.Tensor, right: torch.Tensor, initial: torch.Tensor
) -> torch.Tensor:
    history = torch.empty_like(left, dtype=torch.float32)
    for batch in hl.tile(left.size(0), block_size=1):
        state = initial[batch.id, :, :]
        for step in hl.tile(left.size(1), block_size=1):
            for _repeat in hl.grid(2):
                state = hl.dot(
                    left[batch.id, step.id, :, :],
                    right[batch.id, step.id, :, :],
                    acc=state * 0.5,
                )
            history[batch.id, step.id, :, :] = state
    return history


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _root_indexed_chain(
    left: torch.Tensor, right: torch.Tensor, use_begin: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(left, dtype=torch.float32)
    for batch in hl.tile(left.size(0), block_size=2):
        if use_begin:
            index = batch.begin
        else:
            index = batch.id
        a = left[index, :, :]
        b = right[index, :, :]
        product = hl.dot(a, b).to(left.dtype)
        out[index, :, :] = hl.dot(product, b)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _symbolic_export_chain(
    left: torch.Tensor,
    right: torch.Tensor,
    initial: torch.Tensor,
    entering: torch.Tensor,
    products: torch.Tensor,
    history: torch.Tensor,
    steps: int,
    use_begin: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = hl.specialize(steps)
    # A non-unit block makes tile ID and tile begin observably different.
    for batch in hl.tile(left.size(0), block_size=2):
        if use_begin:
            index = batch.begin
        else:
            index = batch.id
        state = initial[index, :, :].float()
        for step in hl.grid(steps):
            entering[index, step, :, :] = state
            a = left[index, step, :, :]
            b = right[index, step, :, :]
            product = hl.dot(a, b).to(left.dtype)
            products[index, step, :, :] = product
            state = hl.dot(state.to(left.dtype), b, acc=state * 0.5 + product.float())
            history[index, step, :, :] = state
    return entering, products, history


def _args(steps: int = 3, *, device: str = "cpu", batches: int = 2) -> _ExportArgs:
    # One extra untouched slot makes missing, late, and out-of-domain writes
    # visible, including the zero-trip case.
    shape = (batches, steps + 1, 16, 16)
    return (
        torch.empty(shape, dtype=torch.bfloat16, device=device),
        torch.empty(shape, dtype=torch.bfloat16, device=device),
        torch.empty((batches, 16, 16), dtype=torch.float32, device=device),
        torch.full(shape, 7.0, dtype=torch.float32, device=device),
        torch.full(shape, 11.0, dtype=torch.bfloat16, device=device),
        torch.full(shape, 13.0, dtype=torch.float32, device=device),
        steps,
    )


def _config(schedule: str) -> helion.Config:
    if schedule == "preparation":
        return helion.Config(
            cute_chained_mma_schedule="tcgen05_tmem",
            num_warps=8,
            cute_chained_preparation_pipeline=True,
            cute_chained_warp_mma_rows=32,
        )
    return helion.Config(
        cute_chained_mma_schedule=schedule,
        num_warps=4,
    )


def _source(bound: BoundKernel, schedule: str) -> str:
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        loop = discover_chained_loop(bound.host_function.device_ir.graphs)
    assert loop is not None
    assert len(loop.region.stores) == 3
    assert not loop.final_stores
    assert bound.config_spec.cute_chained_loop_search_enabled
    source = bound.to_code(_config(schedule))
    assert "chain_final_store_" not in source
    assert "ptp_thread" not in source
    assert "chain_loop_carry_" in source
    tree = ast.parse(source)
    outputs = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"entering", "products", "history"}
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id in outputs
    ]
    assert len(stores) == 3
    return source


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem", "preparation"])
@pytest.mark.parametrize("steps", [0, 3])
def test_loop_body_exports_need_no_final_root_store(schedule: str, steps: int) -> None:
    args = _args(steps)
    with _cpu_codegen():
        bound = _export_chain._bind_isolated(args)
        source = _source(bound, schedule)
    assert ("MmaF16BF16Op" if schedule == "coalesced" else "tcgen05") in source
    if schedule == "preparation":
        assert "chain_prep_thread" in source
        assert "chain_recurrence_thread" in source


@pytest.mark.parametrize("alias", ["input", "outputs", "overlapping"])
def test_body_only_exports_keep_external_storage_guards(alias: str) -> None:
    args = list(_args())
    if alias == "input":
        args[4] = args[0]
    elif alias == "outputs":
        args[5] = args[3]
    else:
        assert isinstance(args[5], torch.Tensor)
        args[5] = args[5].as_strided(args[5].shape, (*args[5].stride()[:-1], 0))
    with _cpu_codegen():
        bound = _export_chain._bind_isolated(tuple(args))
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(_config("tcgen05_tmem"))


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem", "preparation"])
def test_body_exports_admit_dense_permuted_destinations(schedule: str) -> None:
    args = list(_args())
    for index in (3, 4, 5):
        assert isinstance(args[index], torch.Tensor)
        args[index] = args[index].transpose(-1, -2)
    with _cpu_codegen():
        bound = _export_chain._bind_isolated(tuple(args))
        _source(bound, schedule)


def test_body_only_exports_do_not_admit_nested_control_flow() -> None:
    with _cpu_codegen():
        bound = _nested_export_chain._bind_isolated(_args()[:3])
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert discover_chained_loop(bound.host_function.device_ir.graphs) is None
        assert not bound.config_spec.cute_chained_loop_search_enabled
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(_config("tcgen05_tmem"))


@pytest.mark.parametrize("use_begin", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_symbolic_grid_indices_do_not_consume_matrix_coordinates(
    use_begin: bool, schedule: str
) -> None:
    args = (*_args(batches=4), use_begin)
    with _cpu_codegen():
        bound = _symbolic_export_chain._bind_isolated(args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            origins = {
                type(origin)
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if (origin := _Expression._grid_origin(node)) is not None
            }
        assert GridOrigin in origins
        assert (TileBeginOrigin if use_begin else TileIdOrigin) in origins
        _source(bound, schedule)


def test_symbolic_tile_extent_remains_a_vector_coordinate() -> None:
    args = _args()
    with _cpu_codegen():
        bound = _export_chain._bind_isolated(args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            extents = [
                node
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if node.target is _tracing_ops._get_symnode
                and _Expression._grid_origin(node) is None
                and bound.env.resolve_block_id(node.meta.get("val")) is not None
            ]
        assert extents
        _source(bound, "coalesced")


@pytest.mark.parametrize("use_begin", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_root_indexed_chain_admits_tile_id_and_begin(
    use_begin: bool, schedule: str
) -> None:
    args = (
        torch.empty((4, 128, 128), dtype=torch.bfloat16),
        torch.empty((4, 128, 128), dtype=torch.bfloat16),
        use_begin,
    )
    with _cpu_codegen():
        bound = _root_indexed_chain._bind_isolated(args)
        assert bound.config_spec.cute_chained_matmul_search_enabled
        source = bound.to_code(_config(schedule))
    assert ("MmaF16BF16Op" if schedule == "coalesced" else "tcgen05") in source


@requires_cuda
@pytest.mark.parametrize("use_begin", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_root_indexed_chain_preserves_selected_batches_gpu(
    use_begin: bool, schedule: str
) -> None:
    if schedule != "coalesced" and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires compute capability 10.0 or newer")
    torch.manual_seed(1007)
    # Dyadic inputs make the FP32 sums exact: this exercises tile indexing
    # independently of accumulator-order differences at BF16 rounding ties.
    left = torch.randint(-4, 5, (4, 128, 128), device="cuda").to(torch.bfloat16) * 0.125
    right = torch.randint(-4, 5, left.shape, device="cuda").to(left.dtype) * 0.125
    args = (left, right, use_begin)
    active = slice(None, None, 2) if use_begin else slice(0, 2)
    # hl.dot produces an FP32 accumulator before the source casts to BF16.
    product = (left[active].double() @ right[active].double()).float().to(left.dtype)
    expected = (product.double() @ right[active].double()).float()
    bound = _root_indexed_chain._bind_isolated(args)
    actual = bound.compile_config(_config(schedule))(*args)
    torch.testing.assert_close(actual[active], expected, atol=2e-4, rtol=2e-3)


@requires_cuda
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem", "preparation"])
@pytest.mark.parametrize("steps", [0, 3])
@pytest.mark.parametrize("transposed", [False, True])
def test_body_only_exports_preserve_every_iteration_gpu(
    schedule: str, steps: int, transposed: bool
) -> None:
    if schedule != "coalesced" and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires compute capability 10.0 or newer")
    torch.manual_seed(1005)
    args = _args(steps, device="cuda")
    if transposed:
        args = (
            *args[:3],
            *(value.transpose(-1, -2) for value in args[3:6]),
            args[6],
        )
    left, right, initial, entering, products, history, _ = args
    for value in (left, right, initial):
        value.normal_(std=0.125)
    originals = [value.clone() for value in (left, right, initial)]
    expected_entering = entering.clone()
    expected_products = products.clone()
    expected_history = history.clone()
    state = initial.clone()
    for step in range(steps):
        expected_entering[:, step] = state
        product = (left[:, step].double() @ right[:, step].double()).to(left.dtype)
        expected_products[:, step] = product
        state = (
            state.to(left.dtype).double() @ right[:, step].double()
            + (state * 0.5 + product.float()).double()
        ).float()
        expected_history[:, step] = state
    bound = _export_chain._bind_isolated(args)
    _source(bound, schedule)
    compiled = bound.compile_config(_config(schedule))
    for _replay in range(2):
        entering.fill_(7)
        products.fill_(11)
        history.fill_(13)
        actual = compiled(*args)
        for observed, expected in zip(
            actual,
            (expected_entering, expected_products, expected_history),
            strict=True,
        ):
            torch.testing.assert_close(observed, expected, atol=2e-4, rtol=2e-3)
            torch.testing.assert_close(
                observed[:, steps:], expected[:, steps:], atol=0, rtol=0
            )
        for value, original in zip((left, right, initial), originals, strict=True):
            torch.testing.assert_close(value, original, atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("use_begin", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_symbolic_grid_id_and_begin_exports_gpu(use_begin: bool, schedule: str) -> None:
    if schedule != "coalesced" and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires compute capability 10.0 or newer")
    torch.manual_seed(1006)
    args = _args(device="cuda", batches=4)
    left, right, initial, entering, products, history, steps = args
    for value in (left, right, initial):
        value.normal_(std=0.125)
    active = slice(None, None, 2) if use_begin else slice(0, left.size(0) // 2)
    expected_entering, expected_products, expected_history = (
        value.clone() for value in (entering, products, history)
    )
    state = initial[active].clone()
    original = initial.clone()
    for step in range(steps):
        expected_entering[active, step] = state
        product = (left[active, step].double() @ right[active, step].double()).to(
            left.dtype
        )
        expected_products[active, step] = product
        state = (
            state.to(left.dtype).double() @ right[active, step].double()
            + (state * 0.5 + product.float()).double()
        ).float()
        expected_history[active, step] = state
    bound_args = (*args, use_begin)
    bound = _symbolic_export_chain._bind_isolated(bound_args)
    _source(bound, schedule)
    actual = bound.compile_config(_config(schedule))(*bound_args)
    for observed, expected in zip(
        actual, (expected_entering, expected_products, expected_history), strict=True
    ):
        torch.testing.assert_close(observed, expected, atol=2e-4, rtol=2e-3)
        torch.testing.assert_close(
            observed[:, steps:], expected[:, steps:], atol=0, rtol=0
        )
    torch.testing.assert_close(initial, original, atol=0, rtol=0)
