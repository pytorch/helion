"""Complete Helion rows using generic register permutations, without top-k IR."""

from __future__ import annotations

import ast
import importlib.util
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_register_program import _run

import helion
from helion._compiler.cute.register_region import plan_register_region
from helion._compiler.cute.selection_coarse import coarse_rank_full_keys
from helion._compiler.cute.selection_coarse import coarse_rank_guard
from helion._compiler.cute.selection_coarse import coarse_rank_keys
from helion._compiler.cute.selection_coarse import coarse_rank_recover
from helion._compiler.cute.selection_network import selection_network
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any

    from helion._compiler.tile_dispatch import TileStrategyDispatch


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _permutation(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        value = x[row, :, :]
        groups = hl.arange(x.size(1))[:, None]
        positions = hl.arange(x.size(2))[None, :]
        peer = torch.gather(value, 0, (groups ^ 1).expand(-1, x.size(2)))
        local = torch.gather(peer, 1, (positions ^ 1).expand(x.size(1), -1))
        out[row, :, :] = torch.maximum(local, value) + torch.minimum(local, value)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _selection(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.size(0), 4, 2), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        value = x[row, :, :].reshape(4, 8)
        out[row, :, :] = selection_network(value, 8, "compact_pruned", "balanced")
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _with_output(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for row in hl.grid(x.size(0)):
        value = x[row, :, :]
        groups = hl.arange(x.size(1))[:, None]
        out[row, :, :] = torch.gather(value, 0, (groups ^ 1).expand(-1, x.size(2)))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _dynamic_gather(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        out[row, :, :] = torch.gather(x[row, :, :], 1, indices[row, :, :])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bounded_gather(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        out[row, :, :] = torch.gather(x[row, :, :], 1, indices[row, :, :] & 7)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _conditional_permutation(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        value = x[row, :, :]
        groups = hl.arange(x.size(1))[:, None]
        if (value > 0).reshape(-1).any():
            result = torch.gather(value, 0, (groups ^ 1).expand(-1, x.size(2)))
        else:
            result = value - 3
        out[row, :, :] = result
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _conditional_preserved_values(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        value = x[row, :, :]
        left = value + 1
        right = value - 1
        if (value > 0).reshape(-1).any():
            left = value * 2
        else:
            right = value * 3
        out[row, :, :] = left + right
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _conditional_effect(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        value = x[row, :, :]
        if (value > 0).reshape(-1).any():
            out[row, :, :] = value + 2
        else:
            out[row, :, :] = value - 3
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _coarse_selection(x: torch.Tensor, recovery: str) -> torch.Tensor:
    out = torch.empty((x.size(0), 32, 1), dtype=torch.int64, device=x.device)
    for row in hl.grid(x.size(0)):
        keys = x[row, :, :].reshape(32, 4)
        coarse = coarse_rank_keys(keys, 4, 1, 4)
        selected = selection_network(
            coarse, 4, "compact", "balanced", groups_per_result=4
        )
        fallback = coarse_rank_guard(keys, selected, 4, 4, 4)
        if fallback:
            full = coarse_rank_full_keys(keys, 4, 1, 4)
            result = selection_network(
                full, 4, "compact", "balanced", groups_per_result=4
            )
        else:
            recovered = coarse_rank_recover(keys, selected, 4, 4, 1, 4, recovery)
            result = selection_network(
                recovered, 4, "compact", "balanced", groups_per_result=4
            )
        out[row, :, :] = result
    return out


def _plan(kernel, args, threads=0):
    bound = _cpu_bind(kernel, args)
    config = bound.config_spec.default_config()
    config.config["num_threads"] = [threads]
    tile_strategy = SimpleNamespace(
        strategies=[SimpleNamespace(fn=SimpleNamespace(config=config))]
    )
    with bound.env, bound.host_function:
        plan = plan_register_region(
            bound.host_function.device_ir.graphs,
            cast("TileStrategyDispatch", tile_strategy),
        )
    return bound, config, plan


def test_generic_permutation_register_codegen():
    x = torch.arange(7 * 4 * 8, dtype=torch.int64).reshape(7, 4, 8)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config, plan = _plan(_permutation, (x,))
        assert plan is not None
        assert plan.groups == 4
        assert plan.threads == 128
        code = bound.to_code(config)
    assert "register_input" in code
    assert "_cute_execute_register_plan" in code
    assert "SmemAllocator" not in code
    assert "sync_threads" not in code
    assert "local_topk" not in code
    assert "block=(128, 1, 1)" in code
    assert "((7 + 32 - 1) // 32,)" in code
    outputs, _source = _run(plan.module, x[0])
    groups = torch.arange(4) ^ 1
    positions = torch.arange(8) ^ 1
    expected = x[0][groups][:, positions] + x[0]
    torch.testing.assert_close(outputs[0], expected)


@pytest.mark.parametrize("threads", [1, 32, 128])
def test_register_region_does_not_ignore_inner_axis_thread_requests(threads):
    x = torch.empty((7, 4, 8), dtype=torch.int64)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, _, plan = _plan(_permutation, (x,), threads)
        assert 0 not in bound.config_spec.num_threads.valid_block_ids()
        assert plan is None


def test_selection_helper_is_an_ordinary_register_graph():
    x = torch.randperm(3 * 4 * 8).reshape(3, 4, 8)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config, plan = _plan(_selection, (x,))
        assert plan is not None
        assert all("topk" not in str(node.target) for node in plan.module.graph.nodes)
        code = bound.to_code(config)
    assert "register_input" in code
    assert "_cute_execute_register_plan" in code
    assert "SmemAllocator" not in code
    assert "sync_threads" not in code
    outputs, _source = _run(plan.module, x[0])
    expected = x[0].flatten().topk(8).values.reshape(2, 4).T
    torch.testing.assert_close(outputs[0], expected)


@pytest.mark.parametrize("alias", ["same", "view", "dlpack", "overlap"])
def test_register_region_rejects_output_alias_or_internal_overlap(alias):
    x = torch.zeros((3, 4, 8), dtype=torch.int64)
    out = {
        "same": lambda: x,
        "view": lambda: x.view_as(x),
        "dlpack": lambda: torch.from_dlpack(x),
        "overlap": lambda: torch.empty((1, 4, 8), dtype=x.dtype).expand(3, -1, -1),
    }[alias]()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        _, _, plan = _plan(_with_output, (x, out))
    assert plan is None


def test_register_region_rejects_data_dependent_gather():
    x = torch.zeros((3, 4, 8), dtype=torch.int64)
    indices = torch.zeros_like(x)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        _, _, plan = _plan(_dynamic_gather, (x, indices))
    assert plan is None


def test_register_region_admits_proven_bounded_runtime_gather():
    x = torch.arange(3 * 4 * 8, dtype=torch.int64).reshape(3, 4, 8)
    indices = torch.randint(-100, 100, x.shape)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        _, _, plan = _plan(_bounded_gather, (x, indices))
    assert plan is not None
    outputs, _source = _run(plan.module, x[0], indices[0])
    torch.testing.assert_close(outputs[0], torch.gather(x[0], 1, indices[0] & 7))


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("preserved", [False, True])
def test_native_helion_if_lowers_to_uniform_register_cond(positive, preserved):
    x = torch.full((3, 32, 4), 2 if positive else -2, dtype=torch.int32)
    kernel = _conditional_preserved_values if preserved else _conditional_permutation
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config, plan = _plan(kernel, (x,))
        assert plan is not None
        code = bound.to_code(config)
    assert any(
        node.target is torch.ops.higher_order.cond for node in plan.module.graph.nodes
    )
    assert "register_conditional" in code
    assert "SmemAllocator" not in code
    outputs, _source = _run(plan.module, x[0])
    if preserved:
        expected = x[0] * 3 - 1 if positive else x[0] * 4 + 1
    else:
        expected = x[0][torch.arange(32) ^ 1] if positive else x[0] - 3
    torch.testing.assert_close(outputs[0], expected)


def test_register_conditional_rejects_partial_warp_or_memory_effects():
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        _, _, partial = _plan(
            _conditional_permutation, (torch.ones((3, 4, 8), dtype=torch.int32),)
        )
        _, _, effects = _plan(
            _conditional_effect, (torch.ones((3, 32, 4), dtype=torch.int32),)
        )
    assert partial is None
    assert effects is None


@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("case", ["ordinary", "tied", "infinity"])
def test_full_coarse_algorithm_is_a_native_helion_register_program(recovery, case):
    generator = torch.Generator().manual_seed(39)
    x = (torch.rand((3, 32, 4), generator=generator) + 1).view(torch.int32)
    if case == "tied":
        x[0].fill_(0x3F800001)
    elif case == "infinity":
        x[0, 0, 0] = 0x7F800000
    coarse = coarse_rank_keys(x[0], 4, 1, 4)
    selected = selection_network(coarse, 4, "compact", "balanced", groups_per_result=4)
    assert bool(coarse_rank_guard(x[0], selected, 4, 4, 4)) == (case != "ordinary")
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config, plan = _plan(_coarse_selection, (x, recovery))
        assert plan is not None
        code = bound.to_code(config)
    assert "register_conditional" in code and "shuffle_sync" in code
    assert "SmemAllocator" not in code and "sync_threads" not in code
    assert "coarse_rank_topk" not in code and "distributed_topk" not in code
    outputs, _source = _run(plan.module, x[0])
    columns = torch.arange(4)[None, :] * 4 + torch.arange(32)[:, None] % 4
    exact = (x[0].to(torch.int64) << 4) | (15 - columns)
    expected = exact.reshape(8, 16).topk(4).values.reshape(32, 1)
    torch.testing.assert_close(outputs[0], expected)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("recovery", ["direct", "packed"])
def test_generic_register_coarse_sdk(tmp_path, recovery):
    import cutlass
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    cuda_initialized = torch.cuda.is_initialized()
    x = torch.ones((7, 32, 4), dtype=torch.int32)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config, plan = _plan(_coarse_selection, (x, recovery))
        assert plan is not None
        source = bound.to_code(config)
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    kernel.name = "staged_register_region"
    kernel.decorator_list = [ast.parse("cute.jit", mode="eval").body]
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                *[
                    node
                    for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign))
                ],
                kernel,
            ],
            type_ignores=[],
        )
    )
    path = tmp_path / "register_region_sdk.py"
    path.write_text(ast.unparse(module) + "\n")
    spec = importlib.util.spec_from_file_location("register_region_sdk", path)
    assert spec is not None and spec.loader is not None
    sdk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk)
    mlir = cast("Any", ir)
    with mlir.Context(), mlir.Location.unknown():
        emitted = mlir.Module.create()
        with mlir.InsertionPoint(emitted.body):
            entry = func.FuncOp("entry", ([], []))
            block = entry.add_entry_block()
            with mlir.InsertionPoint(block):
                tensors = [
                    cute.make_tensor(
                        cute.make_ptr(
                            dtype, 0, cute.AddressSpace.gmem, assumed_align=16
                        ),
                        cute.make_layout(shape, stride=stride),
                    )
                    for dtype, shape, stride in [
                        (cutlass.Int32, (7, 32, 4), (128, 4, 1)),
                        (cutlass.Int64, (7, 32, 1), (32, 1, 1)),
                    ]
                ]
                sdk.staged_register_region(*tensors)
                func.ReturnOp([])
        assert emitted.operation.verify()
        text = str(emitted)
        (tmp_path / "register_region.mlir").write_text(text)
        assert "nvvm.shfl.sync" in text and "scf.if" in text
    assert torch.cuda.is_initialized() == cuda_initialized


def test_register_region_uses_bound_runtime_disjointness():
    x = torch.zeros((3, 4, 8), dtype=torch.int64)
    out = torch.empty_like(x)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        _, _, plan = _plan(_with_output, (x, out))
        assert plan is not None
        with patch(
            "helion._compiler.cute.register_region.runtime_tensors_are_proven_disjoint",
            return_value=False,
        ):
            _, _, rejected = _plan(_with_output, (x, out))
        assert rejected is None


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("selection", [False, True])
def test_generic_permutation_register_native(selection):
    x = torch.arange(7 * 4 * 8, dtype=torch.int64, device="cuda").reshape(7, 4, 8)
    if selection:
        expected = x.flatten(1).topk(8).values.reshape(7, 2, 4).transpose(1, 2)
        actual = _selection(x)
    else:
        expected = (
            x[:, torch.arange(4, device="cuda") ^ 1][
                :, :, torch.arange(8, device="cuda") ^ 1
            ]
            + x
        )
        actual = _permutation(x)
    torch.testing.assert_close(actual, expected)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("recovery", ["direct", "packed"])
def test_coarse_native_if_register_native(recovery):
    generator = torch.Generator().manual_seed(39)
    x = (torch.rand((7, 32, 4), generator=generator) + 1).to("cuda").view(torch.int32)
    x[1].fill_(0x3F800001)
    x[2, 0, 0] = 0x7F800000
    columns = (
        torch.arange(4, device="cuda")[None, :] * 4
        + torch.arange(32, device="cuda")[:, None] % 4
    )
    exact = (x.to(torch.int64) << 4) | (15 - columns)
    expected = exact.reshape(7, 8, 16).topk(4).values.reshape(7, 32, 1)
    torch.testing.assert_close(_coarse_selection(x, recovery), expected)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _float32_extrema(left: torch.Tensor, right: torch.Tensor):
    minimum = torch.empty_like(left)
    maximum = torch.empty_like(left)
    for row in hl.grid(left.size(0)):
        lhs = left[row, :, :]
        rhs = right[row, :, :]
        groups = hl.arange(left.size(1))[:, None]
        peer = torch.gather(rhs, 0, (groups ^ 1).expand(-1, left.size(2)))
        minimum[row, :, :] = torch.minimum(lhs, peer)
        maximum[row, :, :] = torch.maximum(lhs, peer)
    return minimum, maximum


def _float32_extrema_input_bits():
    # Cross every sign/zero/subnormal/infinity/NaN boundary in both operand
    # positions, including signaling NaNs. The remainder is finite random bits.
    edge = torch.tensor(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x00000002,
            0x80000002,
            0x007FFFFF,
            0x807FFFFF,
            0x00800000,
            0x80800000,
            0x00800001,
            0x80800001,
            0x3F800000,
            0xBF800000,
            0x3F000000,
            0xBF000000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00000,
            0x7FC12345,
            0xFFC12345,
            0x7F800001,
            0xFF800001,
            0x7FBFFFFF,
            0xFFBFFFFF,
            0x3F800001,
            0xBF800001,
            0x3F7FFFFF,
            0xBF7FFFFF,
        ],
        dtype=torch.int64,
    )
    generator = torch.Generator().manual_seed(617)
    random = torch.randint(0, 2**32, (2, 3072), generator=generator, dtype=torch.int64)
    random = torch.where(
        (random & 0x7F800000) == 0x7F800000, random ^ 0x00800000, random
    )
    return (
        torch.cat((edge.repeat_interleave(32), random[0])).reshape(128, 4, 8),
        torch.cat((edge.repeat(32), random[1])).reshape(128, 4, 8),
    )


def test_float32_extrema_use_generic_register_region():
    left, right = (
        bits.to(torch.int32).view(torch.float32)
        for bits in _float32_extrema_input_bits()
    )
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config, plan = _plan(_float32_extrema, (left, right))
        assert plan is not None
        source = bound.to_code(config)
    assert "_cute_execute_register_plan" in source
    assert "SmemAllocator" not in source
    assert all("topk" not in str(node.target) for node in plan.module.graph.nodes)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_float32_extrema_register_native_ieee_values():
    left_bits, right_bits = _float32_extrema_input_bits()
    left = left_bits.to(torch.int32).view(torch.float32).to("cuda")
    right = right_bits.to(torch.int32).view(torch.float32).to("cuda")
    minimum, maximum = _float32_extrema(left, right)
    peer_bits = right_bits[:, torch.arange(4) ^ 1, :]

    def is_nan(bits):
        return ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)

    # Integer total ordering independently specifies finite extrema, including
    # min(-0,+0)=-0 and max(-0,+0)=+0. NaN payloads may be canonicalized.
    left_order = torch.where(
        (left_bits & 0x80000000) != 0, left_bits ^ 0xFFFFFFFF, left_bits ^ 0x80000000
    )
    right_order = torch.where(
        (peer_bits & 0x80000000) != 0, peer_bits ^ 0xFFFFFFFF, peer_bits ^ 0x80000000
    )
    nan = is_nan(left_bits) | is_nan(peer_bits)
    for actual, choose_left in (
        (minimum, left_order <= right_order),
        (maximum, left_order >= right_order),
    ):
        expected = torch.where(choose_left, left_bits, peer_bits)
        bits = actual.cpu().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        assert torch.equal(is_nan(bits), nan)
        assert torch.equal(bits[~nan], expected[~nan])
    for tensor, original in ((left, left_bits), (right, right_bits)):
        assert torch.equal(
            tensor.cpu().view(torch.int32).to(torch.int64) & 0xFFFFFFFF, original
        )
