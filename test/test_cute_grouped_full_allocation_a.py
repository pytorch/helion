from __future__ import annotations

import ast
import itertools
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _forbid_cuda_work
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
from .test_cute_grouped_fp32_output import _grouped_fp32_output
from .test_cute_grouped_gemm_split_sizes import _device_offsets_kernel
from .test_cute_grouped_gemm_split_sizes import _require_runtime_cuda13_sm100
from .test_cute_grouped_gemm_split_sizes import _selected_config
from .test_cute_grouped_grid_multiplier import _config
import helion
from helion import exc
from helion._testing import skipUnlessBackends
from helion.runtime.cute.launcher import _append_cute_wrapper_plan


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_global_weights_host_descriptor_matches_device_coordinates(
    offset_dtype: torch.dtype,
) -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 256, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=offset_dtype),
    )
    config = dict(_config(1).config)
    config["tcgen05_grouped_full_allocation_a"] = True
    with _target(), _mock_cuda_unavailable():
        code = _cpu_bind(_grouped_fp32_output, args).to_code(
            helion.Config.from_dict(config)
        )
    parsed = ast.parse(code)
    assignment = next(
        node
        for node in ast.walk(parsed)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )
    plans = ast.literal_eval(assignment.value)
    ab = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab["full_allocation_a"] is True
    assert ab["dynamic_ab_tensormaps"] is True
    body: list[str] = []
    arguments: list[str] = []
    _append_cute_wrapper_plan(body, arguments, ab, num_sm=148)
    source = ab["lhs_idx"]
    assert f"arg{source}_shape0 * arg{source}_shape1" in "\n".join(body)
    assert "tcgen05_grouped_group_idx * cutlass.Int32(2)" in code
    assert "cute.domain_offset((tcgen05_grouped_global_m_start" in code
    assert code.count(".update_tensormap(") == 1
    descriptor_arguments = [
        keyword.value.id
        for node in ast.walk(parsed)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "tma_desc_ptr" and isinstance(keyword.value, ast.Name)
    ]
    assert descriptor_arguments
    assert all("_d_tensormap_desc_ptr" in name for name in descriptor_arguments)


@pytest.mark.parametrize("cluster_m", [1, 2])
def test_global_weights_keeps_feature_tail_on_dynamic_path(cluster_m: int) -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 192, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    with _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        config = dict((_config(1) if cluster_m == 1 else _selected_config(128)).config)
        bound.to_code(helion.Config.from_dict(config))
        config["tcgen05_grouped_full_allocation_a"] = True
        with pytest.raises(exc.BackendUnsupported, match="whole output feature"):
            bound.to_code(helion.Config.from_dict(config))


def _two_config(source_m_tile: int, *, immutable: bool = True) -> helion.Config:
    config = _selected_config(128, source_m_tile=source_m_tile)
    config.config["tcgen05_grouped_full_allocation_a"] = immutable
    return config


def _assignment_call(tree: ast.AST, name: str) -> ast.Call:
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    return assignment.value


def _plans(tree: ast.AST) -> list[dict[str, object]]:
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )
    return ast.literal_eval(assignment.value)


def _expression_value(node: ast.expr, values: dict[str, int]) -> object:
    return eval(
        compile(ast.Expression(node), "<generated grouped coordinate>", "eval"),
        {"cutlass": SimpleNamespace(Int32=int, Int64=int)},
        values,
    )


@pytest.mark.parametrize("source_m_tile", [128, 256])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_two_immutable_descriptors_address_each_expert_and_live_interval(
    source_m_tile: int, offset_dtype: torch.dtype
) -> None:
    groups, rows, columns, reduction = 4, 640, 512, 128
    args = (
        torch.empty((rows, reduction), dtype=torch.bfloat16),
        torch.empty((groups, columns, reduction), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, rows], dtype=offset_dtype),
    )
    with _target(), _mock_cuda_unavailable(), _forbid_cuda_work():
        source = _cpu_bind(_grouped_fp32_output, args).to_code(
            _two_config(source_m_tile)
        )
    tree = ast.parse(source)
    plans = _plans(tree)
    grouped = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped["device_split_sizes"] is True
    assert grouped["device_layout_kind"] == "offsets"
    assert grouped["source_m_tile"] == source_m_tile
    assert grouped["cluster_m"] == 2 and grouped["cluster_n"] == 1
    assert grouped["dynamic_d_tensormap"] is True
    ab = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab["full_allocation_a"] is True
    assert ab["bm"] == 256 and ab["bn"] == source_m_tile
    body: list[str] = []
    arguments: list[str] = []
    _append_cute_wrapper_plan(body, arguments, ab, num_sm=148)
    wrapper = ast.parse(dedent("\n".join(body)))
    for operand, tensor, expected_shape in (
        ("a", args[1], (groups * columns, reduction)),
        ("b", args[0], (rows, reduction)),
    ):
        index = ab["lhs_idx" if operand == "a" else "rhs_idx"]
        values = {
            f"arg{index}_shape{axis}": value for axis, value in enumerate(tensor.shape)
        } | {
            f"arg{index}_stride{axis}": value
            for axis, value in enumerate(tensor.stride())
        }
        make_tensor = _assignment_call(
            wrapper, f"tma_atom_{operand}_{'lhs' if operand == 'a' else 'rhs'}_tma"
        )
        assert ast.unparse(make_tensor.args[0]) == f"arg{index}.iterator"
        layout = next(
            keyword.value for keyword in make_tensor.keywords if keyword.arg == "layout"
        )
        assert isinstance(layout, ast.Call)
        assert _expression_value(layout.args[0], values) == expected_shape
        stride = next(
            keyword.value for keyword in layout.keywords if keyword.arg == "stride"
        )
        assert _expression_value(stride, values) == (reduction, 1)

    # Execute the emitted tile-coordinate expressions against the original
    # contiguous tensor address contract, including both halves of a TWO tile.
    a_tile = _assignment_call(tree, "gA_tma")
    assert ast.unparse(a_tile.args[0]) == "tma_tensor_a"
    assert isinstance(a_tile.args[1], ast.Tuple)
    assert isinstance(a_tile.args[2], ast.Tuple)
    tile_columns = _expression_value(a_tile.args[1].elts[0], {})
    assert tile_columns == 256
    for expert in range(groups):
        for column_tile in range(columns // 256):
            coordinate = _expression_value(
                a_tile.args[2].elts[0],
                {
                    "tcgen05_grouped_group_idx": expert,
                    "tcgen05_grouped_cta_tile_idx_n": column_tile,
                },
            )
            assert isinstance(coordinate, int)
            for column_in_tile in (0, 127, 128, 255):
                descriptor_row = coordinate * 256 + column_in_tile
                assert (
                    descriptor_row
                    == expert * columns + column_tile * 256 + column_in_tile
                )
                assert descriptor_row < groups * columns

    b_tile = _assignment_call(tree, "gB_tma")
    assert isinstance(b_tile.args[0], ast.Call)
    offset = b_tile.args[0]
    assert ast.unparse(offset.func) == "cute.domain_offset"
    assert ast.unparse(offset.args[1]) == "tma_tensor_b"
    assert isinstance(offset.args[0], ast.Tuple)
    assert isinstance(b_tile.args[1], ast.Tuple)
    assert _expression_value(b_tile.args[1], {}) == (source_m_tile, reduction)
    for clipped_start in (0, 1, 129, rows):
        for peer in (0, 1):
            assert _expression_value(
                offset.args[0],
                {
                    "tcgen05_grouped_global_m_start": clipped_start,
                    "tcgen05_cta_id_in_cluster": peer,
                },
            ) == (clipped_start, 0)

    # The output descriptor retains per-expert tail masking; neither input
    # copy may consume a rewritten descriptor or its acquire fence.
    copies = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.copy"
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in ("tma_atom_a", "tma_atom_b")
    ]
    assert {node.args[0].id for node in copies} == {"tma_atom_a", "tma_atom_b"}
    assert all(
        not any(keyword.arg == "tma_desc_ptr" for keyword in node.keywords)
        for node in copies
    )
    assert "init_tensormap_from_atom(tma_atom_a" not in source
    assert "init_tensormap_from_atom(tma_atom_b" not in source
    assert "fence_tensormap_update(tcgen05_grouped_tensormap_a_ptr)" not in source
    assert "fence_tensormap_update(tcgen05_grouped_tensormap_b_ptr)" not in source
    assert source.count(".update_tensormap(") == 1
    assert "tcgen05_grouped_d_tensormap_manager.update_tensormap" in source


@pytest.mark.parametrize("source_m_tile", [128, 256])
def test_two_ordinary_inputs_keep_the_original_dynamic_descriptors(
    source_m_tile: int,
) -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 512, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    with _target(), _mock_cuda_unavailable(), _forbid_cuda_work():
        source = _cpu_bind(_grouped_fp32_output, args).to_code(
            _two_config(source_m_tile, immutable=False)
        )
    ab = next(
        plan for plan in _plans(ast.parse(source)) if plan["kind"] == "tcgen05_ab_tma"
    )
    assert not ab.get("full_allocation_a", False)
    for operand in ("a", "b"):
        assert f"init_tensormap_from_atom(tma_atom_{operand}" in source
        assert f"tma_desc_ptr=tcgen05_grouped_tensormap_{operand}_desc_ptr" in source


@pytest.mark.parametrize("source_m_tile", [128, 256])
@pytest.mark.parametrize(
    "proof",
    ["_tcgen05_full_allocation_b_index_domain", "_shared_rhs_region_is_exclusive"],
)
def test_two_immutable_inputs_require_index_and_ownership_proofs(
    source_m_tile: int, proof: str
) -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 512, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    with _target(), _mock_cuda_unavailable(), _forbid_cuda_work():
        bound = _cpu_bind(_grouped_fp32_output, args)
        with (
            patch(f"helion._compiler.cute.cute_mma.{proof}", return_value=False),
            pytest.raises(exc.BackendUnsupported),
        ):
            bound.to_code(_two_config(source_m_tile))


@pytest.mark.parametrize("source_m_tile", [128, 256])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@skipUnlessBackends(["cute"])
@pytest.mark.cuda
def test_two_immutable_inputs_live_experts_and_cuda_graph_replay(
    source_m_tile: int, offset_dtype: torch.dtype, output_dtype: torch.dtype
) -> None:
    _require_runtime_cuda13_sm100()
    groups, rows, columns, reduction = 8, 640, 512, 128
    generator = torch.Generator().manual_seed(731)

    def values() -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.randint(0, 9, (rows, reduction), generator=generator).float() / 8
        right = (
            torch.randint(
                -4, 5, (groups, columns, reduction), generator=generator
            ).float()
            / 8
        )
        right += torch.arange(groups).view(groups, 1, 1) / 2
        return left.bfloat16(), right.bfloat16()

    left, right = values()
    a, b = left.cuda(), right.cuda()
    original = [0, 0, 1, 1, 128, 256, 256, 385, rows]
    offsets = torch.tensor(original, dtype=offset_dtype, device="cuda")
    args = (a, b, offsets)
    function = (
        _grouped_fp32_output.fn
        if output_dtype == torch.float32
        else _device_offsets_kernel.fn
    )
    kernel = helion.kernel(
        function, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated(args)
    config = _two_config(source_m_tile)
    source = bound.to_code(config)
    ab = next(
        plan for plan in _plans(ast.parse(source)) if plan["kind"] == "tcgen05_ab_tma"
    )
    assert ab["full_allocation_a"] is True and ab["cluster_m"] == 2
    bound.set_config(config)

    def expected(routing: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        result = torch.full((rows, columns), float("nan"), dtype=torch.float64)
        covered = torch.zeros(rows, dtype=torch.bool)
        logical_rows = torch.arange(rows, dtype=offset_dtype)
        for expert, (raw_start, raw_end) in enumerate(itertools.pairwise(routing)):
            start = torch.tensor(raw_start, dtype=offset_dtype)
            extent = torch.tensor(raw_end, dtype=offset_dtype) - start
            row_indices = start + logical_rows
            valid = (logical_rows < extent) & (row_indices >= 0) & (row_indices < rows)
            indices = row_indices[valid].long()
            if indices.numel():
                result[indices] = left[indices].double() @ right[expert].double().T
                covered[indices] = True
        return result.to(output_dtype), covered

    # Dyadic inputs keep every product and FP32 sum exact for this K, so
    # mismatches identify routing/descriptor errors rather than rounding noise.
    initial = bound(*args)
    reference, covered = expected(original)
    assert covered.all()
    torch.testing.assert_close(initial.cpu(), reference, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    routing_cases = [
        ([0, 80, 160, 240, 320, 400, 480, 560, rows], 0),
        ([0, 320, 374, 428, 481, 534, 587, rows, rows], 0),
        ([0, 255, 384, 384, 512, 639, 639, rows, rows], 0),
        ([0, 0, 0, 0, 0, rows, rows, rows, rows], 0),
        ([-17, 1, 128, 128, 257, 512, rows, rows + 37, rows + 37], 0),
    ]
    if offset_dtype == torch.int64:
        # The frontend visits only `rows` logical rows per expert. These wide
        # negative intervals therefore leave the prefix unwritten. Their low
        # Int32 bits are positive, exposing any narrowing before clipping.
        limit = (1 << 35) - 17
        routing_cases.append(
            ([-limit, -limit + 128, 256, 256, 320, 400, 480, rows, rows], 256)
        )
    routing_cases.append((original, 0))
    for routing, covered_start in routing_cases:
        left, right = values()
        a.copy_(left)
        b.copy_(right)
        offsets.copy_(offsets.new_tensor(routing))
        captured.fill_(float("nan"))
        graph.replay()
        reference, covered = expected(routing)
        assert torch.equal(covered, torch.arange(rows) >= covered_start)
        observed = captured.cpu()
        torch.testing.assert_close(
            observed[covered], reference[covered], atol=0, rtol=0
        )
        assert torch.isnan(observed[~covered]).all()
        torch.testing.assert_close(a.cpu(), left, atol=0, rtol=0)
        torch.testing.assert_close(b.cpu(), right, atol=0, rtol=0)
        assert offsets.cpu().tolist() == routing
