from __future__ import annotations

import importlib
import operator
import re
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from .test_cute_chained_prepared_groups import _layout_expression
from helion._compiler.cute.chained_native_vector_sink import NativeVectorSink
from helion._compiler.cute.chained_prepared_groups import _group_layout
from helion.runtime import default_cute_launcher

_CASES = ((16, False), (32, False), (64, False), (128, False), (128, True))
_PIPELINE = "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,convert-cute-to-core,canonicalize)"
_BINARY = {
    "addi": operator.add,
    "muli": operator.mul,
    "divsi": operator.floordiv,
    "remsi": operator.mod,
    "andi": operator.and_,
    "shrui": operator.rshift,
    "xori": operator.xor,
}


def _parse_addresses(text: str) -> list[tuple[str | None, tuple[Any, ...]]]:
    """Evaluate only the actual lowered address/16-byte cp.async instruction."""
    program = []
    for raw in text.splitlines():
        line = raw.strip()
        if line in ("module {", "}", "return") or line.startswith("func.func "):
            continue
        lhs, rhs = line.split(" = ", 1) if line.startswith("%") else (None, line)
        regs = re.findall(r"%[\w-]+", rhs)
        if rhs.startswith("arith.constant "):
            token = rhs.split()[1]
            value = 1 if token == "true" else 0 if "e" in token else int(token)
            op = ("constant", value)
        elif match := re.fullmatch(
            r"arith.(\w+) %[\w-]+, %[\w-]+(?: overflow<nsw>)? : i(?:32|64)",
            rhs,
        ):
            assert match[1] in _BINARY
            op = ("binary", match[1], *regs)
        elif rhs.startswith(("llvm.inttoptr", "llvm.ptrtoint", "arith.extsi")):
            op = ("cast", regs[0])
        elif match := re.fullmatch(
            r"llvm.getelementptr (%\w+)\[([%\w]+)\] : .+, (?:bf16|f16)", rhs
        ):
            op = ("offset", match[1], match[2])
        elif rhs.startswith("llvm.intr.assume "):
            assert '"align"' in rhs
            op = ("align", regs[1], regs[2])
        elif rhs.startswith("llvm.store "):
            assert rhs.endswith((": bf16, !llvm.ptr<3>", ": f16, !llvm.ptr<3>"))
            op = ("store", regs[1])
        elif rhs.startswith("nvvm.cp.async.shared.global "):
            assert re.search(r", 16, cache =\s+ca,", rhs)
            assert rhs.endswith(": !llvm.ptr<3>, !llvm.ptr<1>, i32")
            op = ("copy", *regs)
        else:
            raise AssertionError(line)
        program.append((lhs, op))
    return program


def _addresses(
    program: list[tuple[str | None, tuple[Any, ...]]], row: int, column: int, frame: int
) -> tuple[list[int], list[int]]:
    values = {"%arg0": row, "%arg1": column, "%arg2": frame}
    copied, stored = [], []
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
            stored.append(values[op[1]])
            continue
        elif kind == "copy":
            assert values[op[2]] == 1 << 20 and values[op[3]] == 16
            start = values[op[1]]
            assert start % 16 == 0
            copied.extend(range(start, start + 16, 2))
            continue
        else:
            raise AssertionError(kind)
        assert lhs is not None
        values[lhs] = value
    return copied, stored


def _lower(k: int, multi_atom: bool, dtype_name: str, *, repaired: bool) -> str:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    func = importlib.import_module("cutlass._mlir.dialects.func")
    pm = importlib.import_module("cutlass._mlir.passmanager")
    dtype = vars(cutlass)[dtype_name]
    sink = NativeVectorSink("image", (32, k), f"cutlass.{dtype_name}")
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "packet",
                (
                    [ir.IntegerType.get_signless(32)] * 2
                    + [ir.IntegerType.get_signless(64)],
                    [],
                ),
            )
        with ir.InsertionPoint(function.add_entry_block()):
            row = cutlass.Int32(function.arguments[0])
            column = cutlass.Int32(function.arguments[1])
            layout = eval(
                _layout_expression(
                    _group_layout("image", (32, k), sink.dtype, multi_atom=multi_atom)[
                        0
                    ]
                ),
                {"cute": cute, "cutlass": cutlass, "tcgen05": tcgen05},
            )
            image = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        dtype,
                        cutlass.Int64(function.arguments[2]),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    layout.inner,
                    dtype=dtype,
                ),
                layout.outer,
            )
            pointer = eval(
                sink.pointer("row", "column"),
                {"image": image, "row": row, "column": column},
            )
            # Unknown row/column deliberately reduce inferred alignment. Static
            # coordinates would miss align's pointer-rebuilding branch.
            assert pointer.alignment < 16
            aligned = eval(
                sink.aligned_pointer("pointer") if repaired else "pointer.align(16)",
                {
                    "cute": cute,
                    "cutlass": cutlass,
                    "pointer": pointer,
                    "image_layout": layout,
                },
            )
            target = cute.make_tensor(aligned, cute.make_layout((8,)))
            source = cute.make_tensor(
                cute.make_ptr(dtype, 1 << 20, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout((8,)),
            )
            atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), dtype, num_bits_per_copy=128
            )
            cute.copy(atom, source, target)
            for element in range(8):
                image[row, column + element] = dtype(0)
            func.ReturnOp([])
        pm.PassManager.parse(_PIPELINE).run(module.operation)
        assert module.operation.verify()
        return str(module)


@pytest.mark.parametrize("dtype_name", ("Float16", "BFloat16"))
@pytest.mark.parametrize("k,multi_atom", _CASES)
def test_dynamic_native_async_addresses_match_scalar_every_packet(
    k: int, multi_atom: bool, dtype_name: str
) -> None:
    initialized = torch.cuda.is_initialized()
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        correct = _parse_addresses(_lower(k, multi_atom, dtype_name, repaired=True))
        old = _parse_addresses(_lower(k, multi_atom, dtype_name, repaired=False))
        misses = 0
        for slot in range(3):
            for phase in range(8):
                frame = 4096 + slot * 74112 + phase * 128
                addresses = set()
                for row in range(32):
                    for column in range(0, k, 8):
                        copied, scalar = _addresses(correct, row, column, frame)
                        assert copied == scalar and len(copied) == 8
                        assert addresses.isdisjoint(copied)
                        addresses.update(copied)
                        old_copy, old_scalar = _addresses(old, row, column, frame)
                        assert old_scalar == scalar
                        misses += old_copy != scalar
                assert addresses == set(range(frame, frame + 32 * k * 2, 2))
        assert misses > 0  # The previous static proof missed a real regression.
    assert torch.cuda.is_initialized() == initialized


def _source(k: int, multi_atom: bool, dtype_name: str, *, asynchronous: bool) -> str:
    rows, threads = 32, 128
    sink = NativeVectorSink("image", (rows, k), f"cutlass.{dtype_name}")
    lines = [
        "import cutlass",
        "import cutlass.cute as cute",
        "from cutlass.cute.nvgpu import tcgen05",
        "@cute.kernel",
        "def packet_test(source, output):",
        "    thread = cutlass.Int32(cute.arch.thread_idx()[0])",
        "    block = cutlass.Int32(cute.arch.block_idx()[0])",
        f"    allocation = cute.arch.alloc_smem(cutlass.Uint8, {3 * rows * k * 2 + 1024}, alignment=128)",
        f"    frame = allocation + (block % 8) * 128 + (block // 8) * {rows * k * 2}",
        "    image_ptr = cute.recast_ptr(frame, dtype=" + sink.dtype + ")",
        *(
            "    " + line
            for line in _group_layout(
                "image", sink.shape, sink.dtype, multi_atom=multi_atom
            )
        ),
        f"    for step in cutlass.range({max(1, rows * k // (threads * 8))}, unroll=1):",
        f"        row = thread // {k // 8} + step * {threads // (k // 8)}",
        f"        column = thread % {k // 8} * 8",
        f"        if row < {rows}:",
    ]
    if asynchronous:
        lines.extend(
            [
                "            if row < block % 31 + 1:",
                "                pointer = " + sink.pointer("row", "column"),
                "                sink = cute.make_tensor("
                + sink.aligned_pointer("pointer")
                + ", cute.make_layout((8,)))",
                f"                origin = source.iterator + (block % 3 * {rows} + row) * {k} + column",
                "                inputs = cute.make_tensor(origin.align(16), cute.make_layout((8,)))",
                "                atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), "
                + sink.dtype
                + ", num_bits_per_copy=128)",
                "                cute.copy(atom, inputs, sink)",
                "            else:",
                "                for element in cutlass.range_constexpr(8):",
                f"                    image[row, column + element] = {sink.dtype}(0)",
                "    cute.arch.cp_async_commit_group()",
                "    cute.arch.cp_async_wait_group(0)",
            ]
        )
    else:
        lines.extend(
            [
                "            for element in cutlass.range_constexpr(8):",
                f"                image[row, column + element] = source[block % 3, row, column + element] if row < block % 31 + 1 else {sink.dtype}(0)",
            ]
        )
    lines.extend(
        [
            "    cute.arch.sync_threads()",
            f"    for index in cutlass.range(thread, {rows * k}, {threads}):",
            f"        output[block, index // {k}, index % {k}] = image[index // {k}, index % {k}]",
        ]
    )
    return "\n".join(lines) + "\n"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("k,multi_atom", _CASES)
def test_native_async_packets_preserve_bits_masks_and_frame_phases_gpu(
    k: int, multi_atom: bool, dtype: torch.dtype
) -> None:
    name = "Float16" if dtype == torch.float16 else "BFloat16"
    # Includes negative zero, both infinities and NaN payloads. This is a copy
    # test, so compare bits rather than arithmetic tolerances or NaN equality.
    bits = (torch.arange(3 * 32 * k, device="cuda", dtype=torch.int32) * 37).to(
        torch.int16
    )
    source = bits.view(dtype).reshape(3, 32, k)
    source.view(torch.int16)[0, 0, :8] = torch.tensor(
        [0, -32768, 31744, -1024, 31745, 32641, 1, -1], device="cuda", dtype=torch.int16
    )
    outputs = []
    for asynchronous in (False, True):
        generated = _source(k, multi_atom, name, asynchronous=asynchronous)
        module = PyCodeCache.load(generated)
        output = torch.empty((24, 32, k), device="cuda", dtype=dtype)
        default_cute_launcher(
            module.packet_test, (24,), source, output, block=(128, 1, 1)
        )
        outputs.append(output.view(torch.int16))
    expected = torch.stack(
        [
            torch.where(
                torch.arange(32, device="cuda")[:, None] < block % 31 + 1,
                source.view(torch.int16)[block % 3],
                0,
            )
            for block in range(24)
        ]
    )
    torch.testing.assert_close(outputs[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(outputs[1], expected, atol=0, rtol=0)
