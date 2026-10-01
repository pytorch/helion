from __future__ import annotations

import ast
from functools import cache
import importlib
import operator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_vector_expression as expression_tests
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_prepared_vector import _inputs
from .test_cute_chained_prepared_vector import _prepared_vector_loop
from .test_cute_chained_prepared_vector import _vector_config
from helion._compiler.cute import chained_prepared_values
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts
from helion._compiler.cute.chained_tcgen05 import _layout


@cache
def _capture(dtype, fp32, total_warps, consumer_warps, layout_mode="xor"):
    original = chained_prepared_values.emit_prepared_value
    records = []

    def observe(cg, plan, buffer, boundaries, execution, **kwargs):
        lines = original(cg, plan, buffer, boundaries, execution, **kwargs)
        if buffer.name == "chain_prepared_1":
            ordinary = original(cg, plan, buffer, boundaries, execution)
            records.append((buffer, execution, lines, ordinary))
        return lines

    config = _vector_config(consumer_warps)
    config.config["num_warps"] = total_warps
    config.config["cute_chained_scratch_layout"] = layout_mode
    with patch.object(chained_prepared_values, "emit_prepared_value", observe):
        _source(_prepared_vector_loop, _inputs("cpu", dtype, fp32), config)
    assert len(records) == 1
    return records[0]


def _copy_body(lines, execution, *, vector_store=True):
    tree = ast.parse("\n".join(lines))
    sync = ast.unparse(tree.body[-1])
    assert sync == execution.sync
    assert (
        sum(
            isinstance(node, ast.Expr) and ast.unparse(node) == sync
            for node in ast.walk(tree)
        )
        == 1
    )
    active = (
        1 << (execution.threads.bit_length() - 1) if vector_store else execution.threads
    )
    if active != execution.threads:
        assert len(tree.body) == 2 and isinstance(tree.body[0], ast.If)
        assert ast.unparse(tree.body[0].test) == f"{execution.thread} < {active}"
        assert not tree.body[0].orelse
        body = tree.body[0].body
    else:
        body = tree.body[:-1]
    assert all(execution.sync not in ast.unparse(node) for node in body)
    return body, active


@pytest.mark.parametrize(
    "total_warps,consumer_warps", [(8, 4), (16, 8), (16, 4), (32, 4)]
)
@pytest.mark.parametrize(
    "dtype,fp32",
    [(torch.bfloat16, False), (torch.float16, False), (torch.bfloat16, True)],
)
def test_frontier_dynamic_thread_partitions_actual_layout_and_full_role_sync_cpu(
    dtype, fp32, total_warps, consumer_warps
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    buffer, execution, lines, _ordinary = _capture(
        dtype, fp32, total_warps, consumer_warps, "row_major" if fp32 else "xor"
    )
    body, active = _copy_body(lines, execution)
    assert active in (128, 256, 512)
    tag = f"{buffer.name}_vector"
    native_dtype = {
        torch.float32: cutlass.Float32,
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
    }[buffer.dtype]
    dtype_name = f"cutlass.{native_dtype.__name__}"
    modes = ("row_major",) if fp32 else ("row_major", "native_b")
    ir = importlib.import_module("cutlass._mlir.ir")
    before = torch.cuda.is_initialized()
    for mode in modes:
        with (
            patch(
                "torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")
            ),
            ir.Context(),
            ir.Location.unknown(),
        ):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                pointer = cute.make_ptr(
                    native_dtype, 128, cute.AddressSpace.smem, assumed_align=128
                )
                namespace: dict[str, Any] = {
                    "cute": cute,
                    "cutlass": cutlass,
                    "tcgen05": tcgen05,
                    # A real dynamic MLIR value, not an enumerated integer.
                    execution.thread: cutlass.Int32(cute.arch.thread_idx()[0]),
                }
                if mode == "native_b":
                    namespace[f"{buffer.name}_ptr"] = pointer
                    exec(
                        "\n".join(_layout(buffer.name, buffer.shape, 1, dtype_name)),
                        namespace,
                    )
                else:
                    layout = chained_prepared_values.buffer_layout(
                        buffer, ScratchLayouts(mode)
                    )
                    namespace[buffer.name] = cute.make_tensor(
                        pointer, eval(layout, namespace)
                    )
                setup = ast.Module(body=body[:4], type_ignores=[])
                exec(compile(setup, "<dynamic-frontier-copy-setup>", "exec"), namespace)
                values = namespace[f"{tag}_values"]
                assert (
                    values.element_type == native_dtype and int(cute.size(values)) == 8
                )
                loop = body[4]
                assert isinstance(loop, ast.For)
                guard = loop.body[2]
                assert isinstance(guard, ast.If)
                copy = guard.body[-1]
                assert isinstance(copy, ast.Expr) and ast.unparse(copy).startswith(
                    "cute.copy("
                )
                # Execute the actual emitted dynamic target slice and copy;
                # static-thread layout enumeration cannot expose this failure.
                namespace[f"{tag}_step"] = cutlass.Int32(cute.arch.block_idx()[0])
                exec(
                    compile(
                        ast.Module(body=[copy], type_ignores=[]),
                        "<dynamic-frontier-copy>",
                        "exec",
                    ),
                    namespace,
                )
            assert module.operation.verify()
    assert torch.cuda.is_initialized() is before


@pytest.mark.parametrize(
    "dtype,fp32",
    [(torch.bfloat16, False), (torch.float16, False), (torch.bfloat16, True)],
)
def test_frontier_masked_pointwise_value_matches_original_scalar_fallback_cpu(
    dtype, fp32
):
    buffer, execution, lines, ordinary = _capture(dtype, fp32, 16, 4)
    body, _active = _copy_body(lines, execution, vector_store=not fp32)
    tree = ast.Module(body=body, type_ignores=[])
    tag = f"{buffer.name}_vector"
    fallback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == f"not {tag}_leaf_0_vectorized"
    )
    pointwise = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.target) == f"{tag}_element"
    )
    namespace: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            Int32=int,
            Float32=lambda value: torch.tensor(value, dtype=torch.float32).item(),
            Float16=lambda value: torch.tensor(value, dtype=torch.float16).item(),
            BFloat16=lambda value: torch.tensor(value, dtype=torch.bfloat16).item(),
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            math=SimpleNamespace(
                exp2=lambda value, **kwargs: 2.0**value,
                rcp=lambda value, **kwargs: 1 / value,
            )
        ),
        "operator": operator,
        "chain_loop_index": 0,
        "chain_origin_0": 0,
        "chain_origin_1": 0,
        f"{tag}_row": 25,
        f"{tag}_base": 0,
        f"{tag}_leaf_0_vectorized": False,
        f"{tag}_leaf_0_values": [float("nan")] * 8,
        f"{tag}_values": [float("nan")] * 8,
        buffer.name: {},
    }
    # No host pointer is supplied: row25 is masked out, but sigmoid(0) is .5.
    exec(
        compile(
            ast.Module(body=[fallback, pointwise], type_ignores=[]),
            "<frontier-mask>",
            "exec",
        ),
        namespace,
    )
    assert namespace[f"{tag}_leaf_0_values"] == [0] * 8
    if fp32:
        assert [namespace[buffer.name][25, column] for column in range(8)] == [0.5] * 8
    else:
        assert namespace[f"{tag}_values"] == [0.5] * 8
    scalar_tree = ast.parse("\n".join(ordinary))
    scalar_guard = next(
        node
        for node in ast.walk(scalar_tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test).startswith(f"{buffer.name}_offset <")
    )
    for column in range(8):
        namespace[f"{buffer.name}_offset"] = 25 * 128 + column
        exec(
            compile(
                ast.Module(body=[scalar_guard], type_ignores=[]),
                "<original-frontier-mask>",
                "exec",
            ),
            namespace,
        )
    assert [namespace[buffer.name][25, column] for column in range(8)] == [0.5] * 8


@cache
def _logical_store(dtype, sigmoid):
    original = expression_tests.emit_vector_expression

    def emit(*args, **kwargs):
        return original(*args, **kwargs, vector_store=False)

    with patch.object(expression_tests, "emit_vector_expression", emit):
        return expression_tests._frontier_capture(
            dtype,
            sigmoid=sigmoid,
            shape=(19, 16),
            execution=ChainedExecution(384, thread="prep_thread"),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("sigmoid", [False, True])
def test_logical_store_retains_host_vectors_masks_row_offset_and_padding_cpu(
    dtype, sigmoid
):
    lines, captures, _shape = _logical_store(dtype, sigmoid)
    assert lines is not None
    source = "\n".join(lines)
    assert "frontier_copy" not in source and "frontier_target" not in source
    assert "frontier_values" not in source
    assert "frontier_leaf_0_values" in source and "_last_pointer" in source
    assert (
        "prepared_image[frontier_row + 32, frontier_base + frontier_element]" in source
    )
    tree = ast.parse(source)
    fallback = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not frontier_leaf_0_vectorized"
    )
    pointwise = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.target) == "frontier_element"
    )
    namespace: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            Int32=int,
            Float32=lambda value: torch.tensor(value, dtype=torch.float32).item(),
            Float16=lambda value: torch.tensor(value, dtype=torch.float16).item(),
            BFloat16=lambda value: torch.tensor(value, dtype=torch.bfloat16).item(),
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            math=SimpleNamespace(
                exp2=lambda value, **kwargs: 2.0**value,
                rcp=lambda value, **kwargs: 1 / value,
            )
        ),
        "operator": operator,
        "frontier_row": 5,
        "frontier_base": 0,
        "chain_origin_0": 16,
        "chain_loop_index": 0,
        **dict.fromkeys(captures.values(), 16),
        "frontier_leaf_0_vectorized": False,
        "frontier_leaf_0_values": [float("nan")] * 8,
        "prepared_image": {},
    }
    exec(
        compile(
            ast.Module(body=[fallback, pointwise], type_ignores=[]),
            "<logical-store-mask>",
            "exec",
        ),
        namespace,
    )
    expected = 0.5 if sigmoid else 1.0
    assert namespace["frontier_leaf_0_values"] == [0] * 8
    assert namespace["prepared_image"] == {
        (37, column): expected for column in range(8)
    }
    # Distinct vector values must retain their logical element order, including
    # a nonzero vector base and the destination's independent row offset.
    values = [0.0, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25, 3.0]
    namespace["frontier_row"] = 3
    namespace["frontier_base"] = 8
    namespace["frontier_leaf_0_values"] = values
    namespace["prepared_image"] = {}
    exec(
        compile(
            ast.Module(body=[pointwise], type_ignores=[]),
            "<logical-store-element-order>",
            "exec",
        ),
        namespace,
    )
    rounded = (torch.tensor(values) * 0.125).half().float()
    expected_values = (
        (rounded.sigmoid() if sigmoid else rounded.exp()).to(dtype).tolist()
    )
    assert namespace["prepared_image"] == pytest.approx(
        {(35, 8 + element): value for element, value in enumerate(expected_values)},
        abs=1e-6,
    )
    # Allocation padding outside the original node's logical rows is different
    # from a masked input: it must remain zero, without evaluating exp(inf).
    namespace["frontier_row"] = 16
    namespace["frontier_base"] = 0
    namespace["frontier_leaf_0_values"] = [float("inf")] * 8
    namespace["prepared_image"] = {}
    exec(
        compile(
            ast.Module(body=[pointwise], type_ignores=[]),
            "<logical-store-padding>",
            "exec",
        ),
        namespace,
    )
    assert namespace["prepared_image"] == {(48, column): 0 for column in range(8)}


@pytest.mark.parametrize("mode", ["row_major", "xor"])
@pytest.mark.parametrize("total_warps,consumer_warps", [(16, 4), (32, 4)])
def test_fp32_xor_uses_logical_dynamic_scalar_stores_with_full_role_ownership_cpu(
    mode, total_warps, consumer_warps
):
    import cutlass
    import cutlass.cute as cute

    buffer, execution, lines, _ordinary = _capture(
        torch.bfloat16, True, total_warps, consumer_warps
    )
    body, active = _copy_body(lines, execution, vector_store=False)
    assert active == execution.threads
    tag = f"{buffer.name}_vector"
    source = "\n".join(lines)
    assert f"{tag}_target" not in source and f"{tag}_values" not in source
    assert f"{tag}_leaf_0_values" in source
    pointwise = next(
        node
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.For) and ast.unparse(node.target) == f"{tag}_element"
    )
    guard = pointwise.body[0]
    assert isinstance(guard, ast.If)
    store = guard.body[-1]
    assert isinstance(store, ast.Assign)
    assert (
        ast.unparse(store.targets[0])
        == f"{buffer.name}[{tag}_row + 0, {tag}_base + {tag}_element]"
    )
    ir = importlib.import_module("cutlass._mlir.ir")
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            namespace: dict[str, Any] = {"cute": cute, "cutlass": cutlass}
            layout = eval(
                chained_prepared_values.buffer_layout(buffer, ScratchLayouts(mode)),
                namespace,
            )
            pointer = cute.make_ptr(
                cutlass.Float32, 128, cute.AddressSpace.smem, assumed_align=128
            )
            namespace[buffer.name] = cute.make_tensor(pointer, layout)
            thread = cutlass.Int32(cute.arch.thread_idx()[0])
            namespace[f"{tag}_row"] = thread // 16
            namespace[f"{tag}_base"] = thread % 16 * 8
            namespace[f"{tag}_element"] = cutlass.Int32(cute.arch.block_idx()[0]) % 8
            for node in ast.walk(store.value):
                if isinstance(node, ast.Name) and node.id.startswith("chain_value_"):
                    namespace[node.id] = cutlass.Float32(0.5)
            exec(
                compile(
                    ast.Module(body=[store], type_ignores=[]),
                    "<dynamic-logical-store>",
                    "exec",
                ),
                namespace,
            )
            # Execute the emitted ownership arithmetic for every participant;
            # the storage mapping is the actual CuTe layout, including XOR.
            outer = body[0]
            assert isinstance(outer, ast.For)
            assert isinstance(outer.iter, ast.Call)
            trips = outer.iter.args[0]
            assert isinstance(trips, ast.Constant) and isinstance(trips.value, int)
            ownership = compile(
                ast.Module(body=outer.body[:2], type_ignores=[]),
                "<frontier-ownership>",
                "exec",
            )
            seen = set()
            for step in range(trips.value):
                for owner in range(execution.threads):
                    names = {execution.thread: owner, f"{tag}_step": step}
                    exec(ownership, names)
                    row, base = names[f"{tag}_row"], names[f"{tag}_base"]
                    if row >= buffer.shape[0]:
                        continue
                    for element in range(8):
                        address = int(layout((row, base + element)))
                        assert address not in seen
                        seen.add(address)
            assert seen == set(range(32 * 128))
        assert module.operation.verify()
