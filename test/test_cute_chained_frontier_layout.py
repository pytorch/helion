from __future__ import annotations

import ast
from dataclasses import replace
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_async_groups import _config
from .test_cute_chained_async_groups import _source
from .test_cute_chained_group_guards import _call
from .test_cute_chained_late_widening import _masked_widening
from .test_cute_chained_m64_loop import _args
from .test_cute_chained_prepared_groups import _lowered_store_addresses
import helion
from helion import exc
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute import chained_prepared_values
from helion._compiler.cute.chained_frontier_layout import PacketXorVectorSink
from helion._compiler.cute.chained_frontier_layout import packet_xor_sink
from helion._compiler.cute.chained_frontier_layout import packet_xor_swizzle
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_vector_leaf import DenseVectorSink
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
from helion.language import memory_ops


@pytest.mark.parametrize("rows", [2, 3, 17, 64])
@pytest.mark.parametrize("columns", [16, 32, 64, 128, 256])
def test_packet_permutation_stays_within_every_original_row(
    rows: int, columns: int
) -> None:
    parameters = packet_xor_swizzle((rows, columns))
    assert parameters is not None
    bits, base, shift = parameters
    assert base == 3 and bits <= shift
    offsets = []
    for row in range(rows):
        for start in range(0, columns, 8):
            packet = []
            for column in range(start, start + 8):
                index = row * columns + column
                physical = index ^ (
                    ((index >> (base + shift)) & ((1 << bits) - 1)) << base
                )
                assert physical // columns == row
                packet.append(physical)
            assert packet[0] % 8 == 0
            assert packet == list(range(packet[0], packet[0] + 8))
            offsets.extend(packet)
    assert sorted(offsets) == list(range(rows * columns))


@pytest.mark.parametrize("shape", [(1, 64), (0, 64), (64, 8), (64, 24), (2, 3, 64)])
def test_unsupported_or_identity_permutations_are_not_selected(
    shape: tuple[int, ...],
) -> None:
    assert packet_xor_swizzle(shape) is None


@pytest.mark.parametrize("dtype_name", ["Float16", "BFloat16"])
@pytest.mark.parametrize("shape", [(3, 16), (17, 32), (33, 64), (16, 128), (2, 256)])
def test_actual_cute_packet_and_scalar_addresses_match_at_each_frame_phase(
    dtype_name: str, shape: tuple[int, int]
) -> None:
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = vars(cutlass)[dtype_name]
    parameters = packet_xor_swizzle(shape)
    ownership = plan_vector_ownership(shape, 128)
    assert parameters is not None and ownership is not None
    sink = PacketXorVectorSink("image", shape, f"cutlass.{dtype_name}", parameters)
    assert not isinstance(sink, DenseVectorSink)
    namespace = {"cute": cute, "cutlass": cutlass}
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        expected = []
        with ir.InsertionPoint(module.body):
            layout = eval(sink.layout(), namespace)
            tiled = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128
                ),
                cute.make_layout(
                    (ownership.thread_rows, ownership.thread_columns),
                    stride=(ownership.thread_columns, 1),
                ),
                cute.make_layout((1, 8)),
            )
            identity = cute.make_identity_tensor(shape)
            visited = []
            for tid in range(128):
                target = tiled.get_slice(tid).partition_D(identity)
                for step in range(ownership.trips):
                    coordinates = [
                        tuple(int(x) for x in target[i, step, 0]) for i in range(8)
                    ]
                    if coordinates[0][0] >= shape[0]:
                        continue
                    physical = [int(layout(coord)) for coord in coordinates]
                    assert physical[0] % 8 == 0
                    assert physical == list(range(physical[0], physical[0] + 8))
                    visited.extend(coordinates)
            assert sorted(visited) == [
                (i, j) for i in range(shape[0]) for j in range(shape[1])
            ]
            for phase in range(8):
                address = 4096 + phase * 16
                image = cute.make_tensor(
                    cute.make_ptr(
                        dtype, address, cute.AddressSpace.smem, assumed_align=16
                    ),
                    layout,
                )
                for row in (0, shape[0] - 1):
                    for column in (0, shape[1] - 8):
                        pointer = eval(
                            sink.pointer("row", "column"),
                            {"image": image, "row": row, "column": column},
                        )
                        packet = cute.make_tensor(
                            eval(sink.aligned_pointer("pointer"), {"pointer": pointer}),
                            cute.make_layout((8,)),
                        )
                        for element in range(8):
                            physical_address = address + 2 * int(
                                layout((row, column + element))
                            )
                            image[row, column + element] = dtype(1)
                            packet[element] = dtype(1)
                            expected.extend([physical_address] * 2)
        assert module.operation.verify()
        assert _lowered_store_addresses(module) == expected
        copy_module = ir.Module.create()
        with ir.InsertionPoint(copy_module.body):
            image = cute.make_tensor(
                cute.make_ptr(dtype, 4096, cute.AddressSpace.smem, assumed_align=16),
                eval(sink.layout(), namespace),
            )
            pointer = eval(sink.pointer("1", "8"), {"image": image})
            packet = cute.make_tensor(
                eval(sink.aligned_pointer("pointer"), {"pointer": pointer}),
                cute.make_layout((8,)),
            )
            source = cute.make_tensor(
                cute.make_ptr(
                    dtype, 0x100000, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout((8,)),
            )
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(), dtype, num_bits_per_copy=128
                ),
                source,
                packet,
            )
        assert copy_module.operation.verify()


@pytest.mark.parametrize(
    "fault",
    [
        "kind",
        "target",
        "dtype",
        "node_dtype",
        "domain",
        "region_name",
        "offset",
        "size",
    ],
)
def test_only_original_full_raw_images_enter_the_plain_binding(fault: str) -> None:
    graph = Graph()
    raw = _call(graph, memory_ops.load, (), (16, 64), torch.bfloat16)
    buffer = PreparationBuffer("raw", "frontier", raw, torch.bfloat16, (16, 64))
    region = SharedBufferRegion("raw", 128, 2048, 1, 8)
    with patch("helion._compiler.cute.chained_matmul._shape", return_value=(16, 64)):
        assert packet_xor_sink(buffer, region) is not None
        if fault == "kind":
            buffer = replace(buffer, kind="cache")
        elif fault == "target":
            raw.target = torch.ops.prims.convert_element_type.default
        elif fault == "dtype":
            buffer = replace(buffer, dtype=torch.float32)
        elif fault == "node_dtype":
            raw.meta["val"] = torch.empty((16, 64), dtype=torch.float16)
        elif fault == "domain":
            buffer = replace(buffer, shape=(8, 64))
        elif fault == "region_name":
            region = replace(region, name="other")
        elif fault == "offset":
            region = replace(region, byte_offset=130)
        else:
            region = replace(region, byte_size=2046)
        assert packet_xor_sink(buffer, region) is None


def _layout_config(*, asynchronous: bool = True, limit: int = 1) -> helion.Config:
    config = _config(limit, cute_chained_scratch_layout="xor")
    if not asynchronous:
        config.config["cute_chained_async_vector_store"] = False
    return config


@pytest.mark.parametrize("asynchronous,limit", [(False, 1), (True, 1), (True, 8)])
def test_original_binder_receipts_and_only_layout_addresses_change(
    asynchronous: bool, limit: int
) -> None:
    config = _layout_config(asynchronous=asynchronous, limit=limit)
    sinks = []
    allocations = []
    finalize = chained_pipeline_storage.finalize_pipeline_storage

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = finalize(*args, **kwargs)
        assert result is not None
        allocations.append(result.allocations)
        return result

    def bind(*args: Any, **kwargs: Any) -> PacketXorVectorSink | None:
        result = packet_xor_sink(*args, **kwargs)
        if result is not None:
            sinks.append(result)
        return result

    with patch.object(chained_pipeline_storage, "finalize_pipeline_storage", observe):
        with patch.object(
            chained_prepared_values, "packet_xor_sink", return_value=None
        ):
            before = _source(config)
        with patch.object(chained_prepared_values, "packet_xor_sink", bind):
            after = _source(config)
    assert sinks and allocations[0] == allocations[1]
    names = {item.target for item in sinks}
    names.update(f"{item.target}_vector_leaf_0_shared_pointer" for item in sinks)
    original, changed = ast.parse(before), ast.parse(after)
    old_assigns = [
        node
        for node in ast.walk(original)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    ]
    new_assigns = [
        node
        for node in ast.walk(changed)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    ]
    assert len(old_assigns) == len(new_assigns)
    for old, new in zip(old_assigns, new_assigns, strict=True):
        assert ast.dump(old.targets[0]) == ast.dump(new.targets[0])
        new.value = old.value
    assert ast.dump(original) == ast.dump(changed)
    assert "make_swizzle(3, 3, 3)" in after


def test_default_source_and_existing_fp32_layouts_remain_exact() -> None:
    with patch.object(chained_prepared_values, "packet_xor_sink", return_value=None):
        before = _source(_config(1))
    assert before == _source(_config(1))


@pytest.mark.parametrize("fault", ["missing", "capacity"])
def test_half_layout_cannot_bypass_final_storage(fault: str) -> None:
    finalize = chained_pipeline_storage.finalize_pipeline_storage

    def broken(*args: Any, **kwargs: Any) -> Any:
        if fault == "missing":
            return None
        kwargs["capacity_bytes"] = 1
        return finalize(*args, **kwargs)

    with (
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", broken),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _source(_layout_config())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("misaligned,limit", [(False, 8), (True, 1)])
def test_changing_masked_half_images_and_state_gpu(
    dtype: torch.dtype, misaligned: bool, limit: int
) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = list(_args("cuda", dtype=dtype, steps=3))
    if misaligned:
        storage = torch.empty(args[2].numel() + 1, dtype=dtype, device="cuda")
        args[2] = storage[1:].view(args[2].shape)
    generator = torch.Generator(device="cuda").manual_seed(3105)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, generator=generator, device="cuda")
            * 0.015625
        )
    args[5].copy_(
        torch.randint(1, 4, args[5].shape, generator=generator, device="cuda") * 0.25
    )
    left, right, value, transform, update, gate, initial, output, final = args
    unchanged = [tensor.clone() for tensor in args[:7]]
    expected, state = output.clone(), initial.double()
    rows = torch.arange(value.size(2), device="cuda")
    for step in range(3):
        raw = torch.where(
            (rows < step * 17 + 3)[None, :, None], value[:, step].double(), 0.0
        )
        entering = state.to(dtype).double()
        corrected = (raw - right[:, step].double() @ entering).to(dtype).double()
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ corrected + cross).to(dtype)
        state = (
            update[:, step].double() @ corrected
            + state * gate[:, step, -1, None, None].double()
        )
    results = []
    inputs = (*args, False)
    for layout in ("row_major", "xor"):
        output.fill_(19.0)
        final.fill_(float("nan"))
        config = _config(limit, cute_chained_scratch_layout=layout)
        result = _masked_widening._bind_isolated(inputs).compile_config(config)(*inputs)
        results.append(tuple(tensor.clone() for tensor in result))
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
    for before, after in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    for before, after in zip(unchanged, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
