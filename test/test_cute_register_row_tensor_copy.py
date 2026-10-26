from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
from .test_cute_rolled_online_epilogue import row_logsumexp
from .test_cute_rolled_online_epilogue import row_shifted_other
import helion
from helion import exc
from helion._compiler.cute import cluster_online_pair as pair
from helion._compiler.cute import register_row_reduction
from helion._compiler.cute.register_row_tensor_copy import TensorCopyContext
from helion._compiler.cute.register_row_tensor_copy import _original_vector_element
from helion._compiler.cute.register_row_tensor_copy import prove_tensor_copy
from helion._compiler.cute.rolled_cluster_online_pair import _prove
from helion._compiler.cute.rolled_cluster_online_pair import _wrap_sites


def _config(
    threads: int = 256, tensor_copy: bool | None = True, vector_width: int = 8
) -> helion.Config:
    values = {
        "block_sizes": [1],
        "num_threads": [1, threads],
        "reduction_loops": [16384],
        "cute_vector_widths": [vector_width, 1],
        "cute_lane_layouts": ["strided", "blocked"],
        "cute_reduction_reloads": ["register"],
        "cute_cluster_n": 2,
        "cute_cluster_warp_online": True,
        "cute_scalar_epilogue": True,
        "cute_register_row": True,
        "cute_warp_fp32_redux": True,
    }
    if tensor_copy is not None:
        values["cute_register_row_tensor_copy"] = tensor_copy
    return helion.Config.from_dict(values)


def _generate(
    *,
    threads: int = 256,
    tensor_copy: bool | None = True,
    kernel=row_logsumexp,
    dtype: torch.dtype = torch.bfloat16,
    vector_width: int = 8,
) -> str:
    x = torch.empty((3, 32000), dtype=dtype)
    args = (x, torch.empty_like(x)) if kernel is row_shifted_other else (x,)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        assert not bound.settings.fast_math
        return bound.to_code(_config(threads, tensor_copy, vector_width))


@pytest.fixture(scope="module")
def graph() -> SimpleNamespace:
    return _graph(torch.bfloat16, 8)


@pytest.fixture(scope="module")
def fp32_graph() -> SimpleNamespace:
    return _graph(torch.float32, 4)


def _graph(dtype: torch.dtype, vector_width: int) -> SimpleNamespace:
    captured = {}
    original = register_row_reduction.lower_register_row_reduction

    def intercept(body, constants, **kwargs):
        captured.update(
            body=ast.parse(ast.unparse(ast.Module(body=body, type_ignores=[]))).body,
            constants=dict(constants or {}),
            context=kwargs["tensor_copy_context"],
        )
        return original(body, constants, **kwargs)

    with patch.object(
        register_row_reduction, "lower_register_row_reduction", intercept
    ):
        code = _generate(dtype=dtype, vector_width=vector_width)
    assert "_cute_complete_row_tensor_copy(" in code
    return SimpleNamespace(**captured, code=code)


def _plan(body, constants, context):
    working, _owners = _wrap_sites(body)
    sites = pair._find_sites(working)
    if sites is None or len(sites) < 2:
        return None
    first, second = sites[:2]
    if _prove(working, first, second) is None:
        return None
    caches = {
        pair._assigned_name(statement)
        for statement in working
        if pair._is_name_assign(statement)
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value.func) == "cute.make_rmem_tensor"
    }
    reduced = pair._reduced_values(working, first, str, caches)
    if reduced is None:
        return None
    _elements, chain, values = reduced
    fold = next(
        node
        for node in ast.walk(first.for_node)
        if pair._is_name_assign(node)
        and pair._assigned_name(node) == cast("ast.Name", first.call.args[0]).id
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.arch.fmax"
    )
    trips = [
        register_row_reduction._trip_count(loop, constants)
        for loop, _holder in chain
        if loop is not first.for_node
    ]
    if any(trip is None for trip in trips):
        return None
    return prove_tensor_copy(
        working,
        first,
        second,
        fold,
        caches,
        chain,
        values,
        [trip for trip in trips if trip is not None],
        constants,
        context,
    )


@pytest.mark.parametrize("threads", [128, 256, 512])
def test_complete_tensor_copy_uses_proved_thread_geometry(threads: int) -> None:
    code = _generate(threads=threads)
    assert code.count("_cute_complete_row_tensor_copy(") == 1
    assert f"columns=16384, threads={threads}, vector_width=8" in code
    assert "_register_row = cute.make_rmem_tensor" not in code
    assert code.count(".reduce(cute.ReductionOp.MAX") == 1
    assert code.count(".reduce(cute.ReductionOp.ADD") == 1
    assert code.count("_cute_grouped_reduce_cluster_warp_online_pair(") == 1
    tree = ast.parse(code)
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(decorator) == "cute.kernel" for decorator in node.decorator_list
        )
    )
    call = next(
        node
        for node in ast.walk(device)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_cute_grouped_reduce_warp"
    )
    lane = cast("ast.Name", call.args[3]).id
    assignment = next(
        node
        for node in ast.walk(device)
        if pair._is_name_assign(node) and pair._assigned_name(node) == lane
    )
    assert assignment.lineno < call.lineno


@pytest.mark.parametrize("threads", [128, 256, 512])
@pytest.mark.parametrize("vector_width", [4, 8])
def test_fp32_tensor_copy_keeps_original_precision_and_geometry(
    threads: int, vector_width: int
) -> None:
    code = _generate(threads=threads, dtype=torch.float32, vector_width=vector_width)
    assert code.count("_cute_complete_row_tensor_copy(") == 1
    assert f"columns=16384, threads={threads}, vector_width={vector_width}" in code
    assert "cutlass.BFloat16" not in code
    assert "cutlass.Float16" not in code
    assert code.count(".reduce(cute.ReductionOp.MAX") == 1
    assert code.count(".reduce(cute.ReductionOp.ADD") == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp32_extension_keeps_default_loader_unselected(dtype: torch.dtype) -> None:
    missing = _generate(dtype=dtype, tensor_copy=None)
    disabled = _generate(dtype=dtype, tensor_copy=False)
    assert missing == disabled
    assert "_cute_complete_row_tensor_copy(" not in missing


@pytest.mark.parametrize("position", range(4))
def test_fp32_source_requires_uint32_identity_bitcast(position: int) -> None:
    packet = "cute.arch.load(pointer, ir.VectorType.get([4], cutlass.Uint32.mlir_type))"
    raw = f"cutlass.Uint32({packet}[{position}])"
    identity = f"{raw}.bitcast(cutlass.Float32)"
    for expression in (identity, f"cutlass.Float32({identity})"):
        assert (
            _original_vector_element(
                ast.parse(expression, mode="eval").body, "cutlass.Float32", position
            )
            is not None
        )
    for expression in (
        f"cutlass.Float32({raw})",
        f"cutlass.Float32(cutlass.Float16({identity}))",
        f"{identity} * 2",
        f"cutlass.Uint16({packet}[{position}]).bitcast(cutlass.Float32)",
    ):
        assert (
            _original_vector_element(
                ast.parse(expression, mode="eval").body, "cutlass.Float32", position
            )
            is None
        )


def test_fp32_carrier_numeric_conversion_execution_witness() -> None:
    words = torch.tensor(
        [0x3F800000, 0x40000001, -1073741824, -2147483648, 0x7FC01234, 1],
        dtype=torch.int32,
    )
    original = words.view(torch.float32)
    copied = original.clone()
    assert torch.equal(copied.view(torch.int32), words)
    assert not torch.equal(words.float().view(torch.int32), words)


def test_fp32_cache_storage_must_keep_all_uint32_bits(fp32_graph) -> None:
    body = ast.parse(
        ast.unparse(ast.Module(body=fp32_graph.body, type_ignores=[]))
    ).body
    allocation = next(
        node
        for node in body
        if pair._is_name_assign(node)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
    )
    assert ast.unparse(cast("ast.Call", allocation.value).args[1]) == "cutlass.Uint32"
    cast("ast.Call", allocation.value).args[1] = ast.parse(
        "cutlass.Uint16", mode="eval"
    ).body
    assert _plan(body, fp32_graph.constants, fp32_graph.context) is None


@pytest.mark.parametrize("alignment", [0, 2, 4, 8])
def test_fp32_packet_alignment_uses_four_byte_elements(fp32_graph, alignment) -> None:
    metadata = fp32_graph.context.inputs["x"]
    context = TensorCopyContext(
        {"x": replace(metadata, base_alignment=alignment)},
        fp32_graph.context.threads,
    )
    assert _plan(fp32_graph.body, fp32_graph.constants, context) is None


def test_default_and_false_keep_original_loader() -> None:
    missing = _generate(tensor_copy=None)
    disabled = _generate(tensor_copy=False)
    assert missing == disabled
    assert "_cute_complete_row_tensor_copy(" not in missing
    assert "_register_row = cute.make_rmem_tensor" in missing


def test_tensor_copy_requires_register_row() -> None:
    x = torch.empty((3, 32000), dtype=torch.bfloat16)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(row_logsumexp, (x,))
        config = _config()
        config.config["cute_register_row"] = False
        with pytest.raises(exc.InvalidConfig, match="requires cute_register_row"):
            bound.config_spec.normalized_config(config)
        repaired = helion.Config.from_dict(config.config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired["cute_register_row_tensor_copy"] is False


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_tensor_copy_requires_boolean(value) -> None:
    x = torch.empty((3, 32000), dtype=torch.bfloat16)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(row_logsumexp, (x,))
        config = _config()
        config.config["cute_register_row_tensor_copy"] = value
        with pytest.raises(exc.InvalidConfig, match="must be a boolean"):
            bound.config_spec.normalized_config(config)


def test_other_input_reduction_retains_original_path() -> None:
    code = _generate(kernel=row_shifted_other)
    assert "_cute_complete_row_tensor_copy(" not in code


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "cutlass.Float64"},
        {"strides": (32000, 2)},
        {"strides": (32001, 1)},
        {"shape": (3, 32004)},
        {"shape": (0, 32000)},
        {"base_alignment": 8},
        {"base_alignment": 0},
    ],
)
def test_tensor_copy_rejects_unproved_metadata(graph, change) -> None:
    metadata = graph.context.inputs["x"]
    context = TensorCopyContext(
        {"x": replace(metadata, **change)}, graph.context.threads
    )
    assert _plan(graph.body, graph.constants, context) is None


def test_multirow_thread_mapping_declines(graph) -> None:
    context = replace(graph.context, threads=(128, 2, 1))
    assert _plan(graph.body, graph.constants, context) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "sum_mask",
        "row_bias",
        "thread_row",
        "narrowing",
        "arithmetic",
        "input_write_inside",
        "input_write_crossed",
        "loaded_row_alias",
        "impure_row_alias",
        "row_rebound",
        "row_branch_rebound",
        "row_loop_rebound",
        "bound_rebound",
        "cache_narrowing",
        "cache_extent",
        "escaped_load",
        "initial_max",
    ],
)
def test_tensor_copy_rejects_changed_values_masks_or_effects(graph, mutation) -> None:
    body = ast.parse(ast.unparse(ast.Module(body=graph.body, type_ignores=[]))).body
    root = ast.Module(body=body, type_ignores=[])
    loops = [node for node in body if isinstance(node, ast.For)]
    first, second = loops[:2]
    vector = next(
        node
        for node in ast.walk(first)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.arch.load"
    )
    if mutation == "sum_mask":
        mask = next(
            node
            for node in ast.walk(second)
            if pair._is_name_assign(node)
            and pair._assigned_name(node).startswith("mask_")
        )
        cast("ast.Compare", mask.value).comparators[0] = ast.Constant(value=31992)
    elif mutation in ("row_bias", "thread_row", "loaded_row_alias", "impure_row_alias"):
        from helion._compiler.cute.pipeline_state_loads import _indices_by_stride
        from helion._compiler.cute.pipeline_state_loads import _pointer_info

        pointer_selector = cast("ast.IfExp", vector.args[0])
        pointer = pointer_selector.body
        access = _pointer_info(pointer)
        assert access is not None
        indices = _indices_by_stride(access.offset, "x", 2)
        assert indices is not None
        row = indices[0]
        replacement = (
            ast.BinOp(
                left=ast.parse(ast.unparse(row), mode="eval").body,
                op=ast.Add(),
                right=ast.Constant(value=1),
            )
            if mutation == "row_bias"
            else ast.parse(
                "cutlass.Int32(cute.arch.thread_idx()[0])"
                if mutation == "thread_row"
                else "copy_row",
                mode="eval",
            ).body
        )

        class Row(ast.NodeTransformer):
            def visit_Name(self, node):
                return replacement if node.id == "indices_0" else node

        pointer_selector.body = cast("ast.expr", Row().visit(pointer))
        if mutation.endswith("alias"):
            expression = (
                "(x.iterator + 0).load()"
                if mutation == "loaded_row_alias"
                else "changing_row()"
            )
            body.insert(
                body.index(first), ast.parse(f"copy_row = {expression}").body[0]
            )
    elif mutation in ("narrowing", "arithmetic"):
        for node in list(ast.walk(root)):
            if (
                pair._is_name_assign(node)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "cutlass.Float32"
                and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id.startswith("load")
            ):
                if mutation == "narrowing":
                    node.value.args[0] = ast.Call(
                        func=ast.parse("cutlass.Float16", mode="eval").body,
                        args=[node.value.args[0]],
                        keywords=[],
                    )
                else:
                    node.value = ast.BinOp(
                        left=node.value, op=ast.Mult(), right=ast.Constant(value=2)
                    )
    elif mutation in ("input_write_inside", "input_write_crossed"):
        statement = ast.parse("(x.iterator + 0).store(cutlass.BFloat16(1))").body[0]
        if mutation.endswith("inside"):
            first.body.insert(0, statement)
        else:
            body.insert(body.index(second), statement)
    elif mutation in ("row_rebound", "row_branch_rebound", "row_loop_rebound"):
        text = {
            "row_rebound": "indices_0 = 1",
            "row_branch_rebound": "if active:\n    indices_0 = 1",
            "row_loop_rebound": "for i in range(1):\n    indices_0 = i",
        }[mutation]
        body.insert(body.index(first), ast.parse(text).body[0])
    elif mutation == "bound_rebound":
        for loop in loops:
            for node in ast.walk(loop):
                if (
                    isinstance(node, ast.Compare)
                    and isinstance(node.comparators[0], ast.Constant)
                    and node.comparators[0].value == 32000
                ):
                    node.comparators[0] = ast.Name(id="bound", ctx=ast.Load())
        body[body.index(first) : body.index(first)] = ast.parse(
            "bound = 32000\nif active:\n    bound = runtime_bound"
        ).body
    elif mutation.startswith("cache_"):
        allocation = next(
            node
            for node in body
            if pair._is_name_assign(node)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
        )
        if mutation == "cache_narrowing":
            cast("ast.Call", allocation.value).args[1] = ast.parse(
                "cutlass.Uint8", mode="eval"
            ).body
        else:
            cast("ast.Call", allocation.value).args[0] = ast.Constant(value=32)
    elif mutation == "escaped_load":
        name = next(
            pair._assigned_name(node)
            for node in ast.walk(first)
            if pair._is_name_assign(node)
            and pair._assigned_name(node).startswith("load")
        )
        body.append(ast.parse(f"observed = {name}").body[0])
    elif mutation == "initial_max":
        node = next(
            node
            for node in body
            if pair._is_name_assign(node) and pair._assigned_name(node) == "maximum_acc"
        )
        node.value = ast.parse("cutlass.Float32(1)", mode="eval").body
    assert _plan(body, graph.constants, graph.context) is None, mutation


def test_narrowed_cache_execution_witness(graph) -> None:
    raw = torch.tensor([1.0, -2.0, 128.0], dtype=torch.bfloat16).view(torch.int16)
    narrowed = (raw.to(torch.int32) & 255).to(torch.int16).view(torch.bfloat16)
    assert not torch.equal(narrowed.float(), raw.view(torch.bfloat16).float())


@pytest.mark.parametrize("columns", [2048, 32000])
@pytest.mark.parametrize("threads", [128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_complete_packet_mapping_and_empty_slice_numerics(
    columns, threads, dtype
) -> None:
    width, capacity = 8, 16384
    generator = torch.Generator().manual_seed(0)
    row = torch.randn(columns, generator=generator).to(dtype)
    results = []
    for cluster in range(2):
        indices = (
            cluster * capacity
            + torch.arange(threads)[:, None, None] * width
            + torch.arange(capacity // (threads * width))[None, :, None]
            * (threads * width)
            + torch.arange(width)[None, None, :]
        )
        valid = indices[:, :, 0] < columns
        assert torch.equal(valid[:, :, None].expand_as(indices), indices < columns)
        values = torch.full(indices.shape, -float("inf"), dtype=torch.float32)
        values[indices < columns] = row[indices[indices < columns]].float()
        # Copy original bits, widening Half once and preserving FP32 identity.
        raw = values.to(dtype)
        torch.testing.assert_close(raw.float(), values, rtol=0, atol=0)
        maximum = values.reshape(threads, -1).amax(1).reshape(-1, 32).amax(1)
        frame = torch.where(maximum == -float("inf"), 0.0, maximum)
        exponent = torch.exp(
            values.reshape(-1, 32, values.numel() // threads) - frame[:, None, None]
        )
        total = exponent.sum((1, 2))
        results.append((maximum, total))
        selected = indices[indices < columns].sort().values
        expected = torch.arange(
            min(cluster * capacity, columns), min((cluster + 1) * capacity, columns)
        )
        assert torch.equal(selected, expected)
    maxima = torch.cat([value[0] for value in results])
    sums = torch.cat([value[1] for value in results])
    maximum = maxima.amax()
    rescale = torch.where(sums == 0, 0.0, torch.exp(maxima - maximum))
    output = maximum + torch.log((sums * rescale).sum())
    reference = torch.logsumexp(row.double(), 0)
    torch.testing.assert_close(output.double(), reference, atol=2e-6, rtol=2e-6)
