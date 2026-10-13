from __future__ import annotations

import ast
import importlib
import re
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_async_groups import _inputs
from .test_cute_chained_native_async_mapping import _BINARY
from .test_cute_chained_native_async_mapping import _PIPELINE
from .test_cute_chained_native_async_mapping import _parse_addresses
from .test_cute_chained_rectangular_frontier import _capture
from helion._compiler.cute.chained_frontier_layout import PacketXorVectorSink
from helion._compiler.cute.chained_native_vector_sink import NativeVectorSink
from helion._compiler.cute.chained_prepared_groups import _group_layout

if TYPE_CHECKING:
    from helion._compiler.cute.chained_rectangular_frontier import RectangularFrontier


def _lower(item: RectangularFrontier, lines: list[str]) -> str:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    func = importlib.import_module("cutlass._mlir.dialects.func")
    pm = importlib.import_module("cutlass._mlir.passmanager")
    dtype = vars(cutlass)[item.proof.dtype.split(".")[1]]
    rows, columns = item.proof.tile_shape
    guard = ast.parse("\n".join(lines)).body[1]
    assert isinstance(guard, ast.If)
    address_guard = guard.body[-1]
    assert isinstance(address_guard, ast.If)
    # Use the four original compiler statements, rather than another copy
    # template. Guards have separate typed-index and GPU fallback coverage.
    copy_code = ast.unparse(ast.Module(body=address_guard.body[:-1], type_ignores=[]))
    assert len(address_guard.body[:-1]) == 4
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "rectangle",
                (
                    [ir.IntegerType.get_signless(32)] * 2
                    + [ir.IntegerType.get_signless(64)],
                    [],
                ),
            )
        with ir.InsertionPoint(function.add_entry_block()):
            tid = cutlass.Int32(function.arguments[0])
            base = cutlass.Int64(function.arguments[2])
            pointer = cute.make_ptr(
                dtype, base, cute.AddressSpace.smem, assumed_align=128
            )
            namespace: dict[str, Any] = {
                "cute": cute,
                "cutlass": cutlass,
                "tcgen05": tcgen05,
            }
            sink = item.sink
            if isinstance(sink, NativeVectorSink):
                namespace[sink.target + "_ptr"] = pointer
                exec(
                    "\n".join(
                        _group_layout(
                            sink.target,
                            sink.shape,
                            sink.dtype,
                            multi_atom=columns == 128,
                        )
                    ),
                    namespace,
                )
                image = namespace[sink.target]
            elif isinstance(sink, PacketXorVectorSink):
                image = cute.make_tensor(pointer, eval(sink.layout(), namespace))
            else:
                image = cute.make_tensor(
                    pointer, cute.make_layout((rows, columns), stride=(columns, 1))
                )
            namespace.update(
                {
                    sink.target: image,
                    item.execution.thread: tid,
                    item.prefix + "_pointer": cute.make_ptr(
                        dtype, 1 << 20, cute.AddressSpace.gmem, assumed_align=16
                    ),
                }
            )
            exec(copy_code, namespace)
            thread = namespace[item.prefix + "_thread"]
            identity = thread.partition_D(cute.make_identity_tensor((rows, columns)))
            for index in range(cute.size(identity)):
                row, column = identity[index]
                image[row, column] = dtype(0)
            func.ReturnOp([])
        pm.PassManager.parse(_PIPELINE).run(module.operation)
        assert module.operation.verify()
        return str(module)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scratch", ["row_major", "xor"])
def test_actual_full_copy_maps_dynamic_native_and_plain_images(
    dtype: torch.dtype, scratch: str
) -> None:
    rendered: dict[str, list[str]] = {}
    _, captured = _capture(
        args=_inputs(dtype=dtype),
        rendered=rendered,
        cute_chained_scratch_layout=scratch,
    )
    assert any(isinstance(item.sink, NativeVectorSink) for item in captured)
    assert any(not isinstance(item.sink, NativeVectorSink) for item in captured)
    initialized = torch.cuda.is_initialized()
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        for item in captured:
            program = parse(_lower(item, rendered[item.prefix]))
            rows, columns = item.proof.tile_shape
            # Nonzero native member offsets and every swizzle base phase.
            for phase in range(16):
                base = 4096 + phase * 128
                visited, sources = [], []
                for tid in range(item.execution.threads):
                    copied, stored = addresses(program, tid, base)
                    assert [pair[0] for pair in copied] == stored
                    visited.extend(stored)
                    sources.extend(pair[1] for pair in copied)
                assert sorted(visited) == list(
                    range(base, base + rows * columns * 2, 2)
                )
                expected = [
                    2 * (row * item.proof.pitch + column) + (1 << 20)
                    for row in range(rows)
                    for column in range(columns)
                ]
                assert sorted(sources) == expected
    assert torch.cuda.is_initialized() == initialized


def addresses(program, tid, base):
    values = {"%arg0": tid, "%arg1": 0, "%arg2": base}
    copies, stores = ([], [])
    for lhs, op in program:
        kind = op[0]
        if kind == "constant":
            value = op[1]
        elif kind == "binary":
            value = _BINARY[op[1]](values[op[2]], values[op[3]])
        elif kind == "cast":
            value = values[op[1]]
        elif kind == "offset":
            value = values[op[1]] + 2 * (
                values[op[2]] if op[2].startswith("%") else int(op[2])
            )
        elif kind == "align":
            assert values[op[1]] % values[op[2]] == 0
            continue
        elif kind == "store":
            stores.append(values[op[1]])
            continue
        elif kind == "copy":
            target, source, count = (values[x] for x in op[1:])
            assert count == 16 and target % 16 == source % 16 == 0
            copies.extend((target + 2 * i, source + 2 * i) for i in range(8))
            continue
        else:
            raise AssertionError(kind)
        assert lhs is not None
        values[lhs] = value
    return (copies, stores)


def parse(text):
    """Expand only constant bounded SCF loops from the actual lowered copy."""
    lines = [line.strip() for line in text.splitlines() if not line.startswith("#")]
    constants = {}
    expanded = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if match := re.fullmatch(
            "(%\\w+) = arith.constant (-?\\d+) : i(?:32|64)", line
        ):
            constants[match[1]] = int(match[2])
        if match := re.fullmatch(
            "scf.for (%\\w+) = (%\\w+) to (%\\w+) step (%\\w+)  : i32 \\{", line
        ):
            end = next(
                i
                for i in range(index + 1, len(lines))
                if lines[i].startswith("} {loop_annotation")
            )
            body = lines[index + 1 : end]
            assert not any("scf." in child for child in body)
            for iteration in range(
                constants[match[2]], constants[match[3]], constants[match[4]]
            ):
                expanded.append(f"{match[1]} = arith.constant {iteration} : i32")
                expanded.extend(body)
            index = end + 1
        else:
            expanded.append(line)
            index += 1
    return _parse_addresses("\n".join(expanded))
