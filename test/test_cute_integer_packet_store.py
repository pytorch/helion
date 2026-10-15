"""GPU-free coverage for signed and unsigned integer packet stores."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.cute.persistent_branch_vec import (
    vectorize_branch_local_persistent_fragments,
)
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _integer_rows(
    x: torch.Tensor, dtype: hl.constexpr, limit: hl.constexpr
) -> torch.Tensor:
    output = torch.empty(x.shape, dtype=dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        columns = hl.arange(x.size(1))
        data = x[row, columns]
        maximum = data.amax(-1)
        encoded = (data + maximum[:, None]).to(dtype)
        if limit >= x.size(1):
            output[row, columns] = encoded
        else:
            hl.store(
                output, [row, columns], encoded, extra_mask=columns[None, :] < limit
            )
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _integer_rows_into(x: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    for row in hl.tile(x.size(0)):
        columns = hl.arange(x.size(1))
        data = x[row, columns]
        maximum = data.amax(-1)
        destination[row, columns] = (data + maximum[:, None]).to(destination.dtype)
    return destination


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _integer_rows_readback(
    x: torch.Tensor, dtype: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(x.shape, dtype=dtype, device=x.device)
    observed = torch.empty_like(output)
    for row in hl.tile(x.size(0)):
        columns = hl.arange(x.size(1))
        data = x[row, columns]
        maximum = data.amax(-1)
        output[row, columns] = (data + maximum[:, None]).to(dtype)
        observed[row, columns] = output[row, columns]
    return output, observed


def _config(bound: BoundKernel, width: int) -> helion.Config:
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    spec = bound.config_spec
    return helion.Config(
        block_sizes=[16],
        num_threads=[
            32 // width if block == reduction else 8
            for block in spec.num_threads.valid_block_ids()
        ],
        cute_vector_widths=[
            width if block == reduction else 1
            for block in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_lane_layouts=[
            "strided" if block == reduction else "blocked"
            for block in spec.cute_lane_layouts.valid_block_ids()
        ],
    )


def _generate(kernel: Kernel, args: tuple[object, ...], width: int) -> str:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        return bound.to_code(_config(bound, width))


def _flushes(code: str, tensor: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_cute_store_u16_vec", "_cute_store_u32_vec"}
        and any(
            isinstance(part, ast.Attribute)
            and part.attr == "iterator"
            and isinstance(part.value, ast.Name)
            and part.value.id == tensor
            for part in ast.walk(node.args[0])
        )
    ]


@pytest.mark.parametrize(
    ("dtype", "width"),
    [(dtype, width) for dtype in (torch.int16, torch.uint16) for width in (2, 4, 8)]
    + [(dtype, width) for dtype in (torch.int32, torch.uint32) for width in (2, 4)],
)
def test_integer_reduction_store_emits_one_packet(
    dtype: torch.dtype, width: int
) -> None:
    code = _generate(
        _integer_rows, (torch.empty((37, 32), dtype=torch.int32), dtype, 32), width
    )
    (flush,) = _flushes(code, "output")
    helper = "_cute_store_u16_vec" if dtype.itemsize == 2 else "_cute_store_u32_vec"
    carrier = "cutlass.Uint16" if dtype.itemsize == 2 else "cutlass.Uint32"
    assert ast.unparse(flush.func) == helper, code
    assert f".bitcast({carrier})" in code, code
    assert f"as {helper}" in code, code
    assert "_helion_persistent_branch_vec_store" not in code, code


def test_signed_int16_input_uses_the_same_integer_load_carrier() -> None:
    code = _generate(
        _integer_rows, (torch.empty((37, 32), dtype=torch.int16), torch.int16, 32), 8
    )
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in code, code
    assert _flushes(code, "output"), code


@pytest.mark.parametrize(
    "dtype", [torch.int16, torch.uint16, torch.int32, torch.uint32]
)
def test_straddling_lane_mask_keeps_scalar_integer_stores(dtype: torch.dtype) -> None:
    width = 8 if dtype.itemsize == 2 else 4
    code = _generate(
        _integer_rows, (torch.empty((37, 32), dtype=torch.int32), dtype, 17), width
    )
    assert not _flushes(code, "output"), code
    assert ".store(cutlass." in code, code


@pytest.mark.parametrize("dtype", [torch.int16, torch.uint16])
@pytest.mark.parametrize("fault", ["offset", "row_stride"])
def test_unaligned_integer_destination_keeps_scalar_stores(
    dtype: torch.dtype, fault: str
) -> None:
    shape = (37, 32)
    if fault == "offset":
        storage = torch.empty(shape[0] * shape[1] + 1, dtype=dtype)
        destination = storage[1:].reshape(shape)
    else:
        destination = torch.empty((shape[0], shape[1] + 1), dtype=dtype)[:, : shape[1]]
    code = _generate(
        _integer_rows_into, (torch.empty(shape, dtype=torch.int32), destination), 8
    )
    assert not _flushes(code, "destination"), code


@pytest.mark.parametrize("dtype", [torch.int16, torch.uint16])
def test_a_later_load_of_the_integer_output_prevents_store_sinking(
    dtype: torch.dtype,
) -> None:
    # Its required cross-thread barrier cannot be distributed around the
    # reduction passes, so the existing ordering proof rejects this kernel.
    with pytest.raises(exc.BackendUnsupported, match="between per-lane statements"):
        _generate(
            _integer_rows_readback, (torch.empty((37, 32), dtype=torch.int32), dtype), 8
        )


@pytest.mark.parametrize("dtype_name", ["Int16", "Uint16"])
def test_late_integer_marker_store_stays_before_its_aliasing_load(
    dtype_name: str,
) -> None:
    source = f"""for synthetic_lane_7 in cutlass.range_constexpr(8):
    value = cutlass.{dtype_name}(synthetic_lane_7)
    _helion_persistent_branch_vec_store(7, 8, 'cutlass.{dtype_name}', output.iterator + synthetic_lane_7, value, None)
    observed = (output.iterator + synthetic_lane_7).load()
"""
    lowered = vectorize_branch_local_persistent_fragments(ast.parse(source).body)
    code = ast.unparse(ast.Module(body=lowered, type_ignores=[]))
    assert not _flushes(code, "output"), code
    assert ".store(" in code, code


@pytest.mark.parametrize(
    ("dtype", "width", "input_dtype"),
    [
        (torch.int16, 16, torch.int16),
        (torch.uint16, 16, torch.int16),
        (torch.int32, 8, torch.int32),
        (torch.uint32, 8, torch.int32),
    ],
)
def test_new_integer_flushes_stay_within_the_16_byte_alias_envelope(
    dtype: torch.dtype, width: int, input_dtype: torch.dtype
) -> None:
    code = _generate(
        _integer_rows, (torch.empty((37, 32), dtype=input_dtype), dtype, 32), width
    )
    assert not _flushes(code, "output"), code


@pytest.mark.parametrize("dtype_name", ["Int16", "Uint16", "Int32", "Uint32"])
def test_integer_marker_flush_preserves_signed_and_unsigned_bits(
    dtype_name: str,
) -> None:
    bits = 16 if dtype_name.endswith("16") else 32
    mask = (1 << bits) - 1

    class Word(int):
        def bitcast(self, carrier: type[int]) -> int:
            return carrier(int(self) & mask)

    class Pointer:
        def __init__(self, output: list[int], index: int = 0) -> None:
            self.output = output
            self.index = index

        def __add__(self, offset: int) -> Pointer:
            return Pointer(self.output, self.index + int(offset))

        def store(self, value: int) -> None:
            self.output[self.index] = int(value) & mask

    source = f"""for synthetic_lane_7 in cutlass.range_constexpr(4):
    value = cutlass.{dtype_name}(inputs[synthetic_lane_7])
    _helion_persistent_branch_vec_store(7, 4, 'cutlass.{dtype_name}', out.iterator + synthetic_lane_7, value, None)
"""
    lowered = vectorize_branch_local_persistent_fragments(ast.parse(source).body)
    module = ast.fix_missing_locations(ast.Module(body=lowered, type_ignores=[]))
    inputs = [-(1 << (bits - 1)), -1, 1 << (bits - 1), mask]
    output = [0] * 4
    calls = []

    def store(pointer: Pointer, values: list[int]) -> None:
        calls.append(len(values))
        for offset, value in enumerate(values):
            (pointer + offset).store(value)

    namespace = {
        "cutlass": SimpleNamespace(
            range_constexpr=range,
            Int16=Word,
            Uint16=Word,
            Int32=Word,
            Uint32=Word,
        ),
        "inputs": inputs,
        "out": SimpleNamespace(iterator=Pointer(output)),
        "_cute_store_u16_vec": store,
        "_cute_store_u32_vec": store,
    }
    exec(compile(module, "<integer-packet-store>", "exec"), namespace)
    assert calls == [4]
    assert output == [value & mask for value in inputs]


def test_integer_store_helper_builds_one_signless_16_bit_vector() -> None:
    cutlass = pytest.importorskip("cutlass")
    from unittest.mock import patch

    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func

    from helion._compiler.cute.vec_utils import store_u16_vec

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "store_integer_packet",
                ir.FunctionType.get([cutlass.Int16.mlir_type] * 8, []),
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                values = [
                    cutlass.Int16(argument).bitcast(cutlass.Uint16)
                    for argument in block.arguments
                ]
                captured = []
                with patch(
                    "cutlass.cute.arch.store",
                    side_effect=lambda pointer, packet: captured.append(packet),
                ):
                    store_u16_vec(None, values)
                func.ReturnOp([])
        assert module.operation.verify()
        (packet,) = captured
        assert str(packet.type) == "vector<8xi16>"
