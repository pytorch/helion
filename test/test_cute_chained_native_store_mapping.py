from __future__ import annotations

from functools import lru_cache
import importlib
import itertools
import operator
import re
from typing import Any
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.chained_native_stores import plan_native_stmatrix_store
from helion._compiler.cute.chained_prepared_groups import _member_layout
from helion._compiler.cute.chained_tcgen05 import _layout
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership

_INPUT = 1 << 48
_PIPELINE = "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,convert-cute-to-core,canonicalize)"
# Full owner, member rows, member origin, already selected producer threads.
_CASES = ((160, 128, 32, 128), (80, 32, 16, 64), (96, 64, 16, 256))
_OPERATIONS = {
    "addi": operator.add,
    "muli": operator.mul,
    "divsi": operator.floordiv,
    "remsi": operator.mod,
    "andi": operator.and_,
    "shrui": operator.rshift,
    "xori": operator.xor,
}


def _parse(text):
    result = []
    for raw in text.splitlines():
        line = raw.strip()
        if line in ("module {", "}", "return") or line.startswith("func.func "):
            continue
        lhs, rhs = line.split(" = ", 1) if line.startswith("%") else (None, line)
        regs = re.findall(r"%[\w]+(?:#[0-3])?", rhs)
        if rhs.startswith("arith.constant "):
            token = rhs.split()[1]
            assert token.isdigit() or token in ("true", "0.000000e+00")
            op = ("const", int(token) if token.isdigit() else int(token == "true"))
        elif match := re.fullmatch(
            r"arith.(\w+) %\w+, %\w+(?: overflow<nsw>)? : i(32|64)", rhs
        ):
            assert match[1] in (
                "addi",
                "muli",
                "divsi",
                "remsi",
                "andi",
                "shrui",
                "xori",
            )
            op = ("binary", match[1], *regs, int(match[2]))
        elif rhs.startswith(("llvm.inttoptr", "llvm.ptrtoint")):
            op = ("cast", regs[0])
        elif match := re.fullmatch(
            r"llvm.getelementptr (%\w+)\[([%\w]+)\] : .+, (i8|bf16|f16)", rhs
        ):
            op = ("gep", match[1], match[2], 1 if match[3] == "i8" else 2)
        elif rhs.startswith("llvm.intr.assume "):
            assert '"align"' in rhs
            op = ("align", regs[1], regs[2])
        elif rhs.startswith("llvm.alloca "):
            assert "x i16 {alignment = 32" in rhs
            op = ("alloca",)
        elif rhs.startswith("llvm.load "):
            assert rhs.endswith((" -> bf16", " -> f16", " -> vector<4xi32>"))
            op = ("load", regs[0], 16 if rhs.endswith(" -> vector<4xi32>") else 2)
        elif rhs.startswith("llvm.store "):
            assert ": bf16, !llvm.ptr" in rhs or ": f16, !llvm.ptr" in rhs
            op = ("store", regs[0], regs[1], "!llvm.ptr<3>" in rhs)
        elif rhs.startswith("vector.to_elements "):
            assert lhs is not None
            assert lhs.endswith(":4") and rhs.endswith(": vector<4xi32>")
            lhs = lhs[:-2]
            op = ("unpack", regs[0])
        elif rhs.startswith("nvvm.stmatrix "):
            assert "layout = #nvvm.mma_layout<col>" in rhs and len(regs) == 5
            op = ("matrix", *regs)
        else:
            raise AssertionError(line)
        result.append((lhs, op))
    return result


def _run(code, thread, step, frame):
    env = {"%arg0": thread, "%arg1": step, "%arg2": frame}
    memory, stores, matrices = {}, [], []
    for lhs, op in code:
        kind = op[0]
        if kind == "const":
            value = op[1]
        elif kind == "binary":
            x, y = env[op[2]], env[op[3]]
            assert x >= 0 and y >= 0
            value = _OPERATIONS[op[1]](x, y)
            assert 0 <= value < 1 << (op[4] - 1)
        elif kind == "cast":
            value = env[op[1]]
        elif kind == "gep":
            value = env[op[1]] + op[3] * (
                env[op[2]] if op[2].startswith("%") else int(op[2])
            )
        elif kind == "align":
            assert env[op[1]] % env[op[2]] == 0
            continue
        elif kind == "alloca":
            value = 1 << 40
        elif kind == "load":
            address = env[op[1]]
            if op[2] == 2:
                assert _INPUT <= address < _INPUT + 16384 and address % 2 == 0
                value = (address - _INPUT) // 2
            else:
                assert address % 16 == 0
                value = [
                    (memory[address + i * 4], memory[address + i * 4 + 2])
                    for i in range(4)
                ]
        elif kind == "store":
            address = env[op[2]]
            if op[3]:
                stores.append(address)
            else:
                assert address not in memory
                memory[address] = env[op[1]]
            continue
        elif kind == "unpack":
            for index, pair in enumerate(env[op[1]]):
                env[f"{lhs}#{index}"] = pair
            continue
        elif kind == "matrix":
            address = env[op[1]]
            assert address % 16 == 0
            matrices.append((address, [env[register] for register in op[2:]]))
            continue
        else:
            raise AssertionError(op)
        env[lhs] = value
    return stores, matrices


@lru_cache(None)
def _lower(case, dtype):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    passmanager = importlib.import_module("cutlass._mlir.passmanager")
    func = importlib.import_module("cutlass._mlir.dialects.func")
    owner, member, offset, threads = case
    ownership = plan_vector_ownership((32, member), threads, tile_columns=32)
    assert ownership is not None
    store = plan_native_stmatrix_store(
        (owner, 32), (32, member), offset, dtype, ownership
    )
    assert store is not None
    name = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    element_type = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    initialized = torch.cuda.is_initialized()
    programs = []
    for matrix in (True, False):
        with (
            patch(
                "torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")
            ),
            ir.Context(),
            ir.Location.unknown(),
        ):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp(
                    "transport",
                    (
                        [ir.IntegerType.get_signless(32)] * 2
                        + [ir.IntegerType.get_signless(64)],
                        [],
                    ),
                )
            with ir.InsertionPoint(function.add_entry_block()):
                scope: dict[str, Any] = {
                    "cute": cute,
                    "cutlass": cutlass,
                    "tcgen05": tcgen05,
                    "thread": cutlass.Int32(function.arguments[0]),
                    "step": cutlass.Int32(function.arguments[1]),
                    "frame": cute.make_ptr(
                        cutlass.Uint8,
                        cutlass.Int64(function.arguments[2]),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                }
                lines = [
                    f"full_ptr = cute.recast_ptr(frame, dtype={name})",
                    *_layout("full", (owner, 32), 1, name),
                    f"target_ptr = cute.recast_ptr(frame + {offset * 64}, dtype={name})",
                    *_member_layout("target", (member, 32), name, (1, 0)),
                ]
                exec("\n".join(lines), scope)
                if matrix:
                    emission = store.emit("target", "store", "thread", "step")
                    exec("\n".join(emission.setup), scope)
                    source = cute.make_tensor(
                        cute.make_ptr(
                            element_type,
                            _INPUT,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(8192),
                    )
                    for element in range(8):
                        scope[emission.values][element] = source[
                            scope["thread"] * 8 + element
                        ]
                    exec(emission.copy, scope)
                else:
                    scope["target"][scope["thread"], scope["step"]] = element_type(0)
                    scope["full"][scope["step"] + offset, scope["thread"]] = (
                        element_type(0)
                    )
                func.ReturnOp([])
            assert module.operation.verify()
            passmanager.PassManager.parse(_PIPELINE).run(module.operation)
            assert module.operation.verify()
            programs.append(_parse(str(module)))
    assert torch.cuda.is_initialized() == initialized
    return store, programs


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("case", _CASES)
def test_actual_dynamic_stmatrix_matches_scalar_full_owner_all_slot_phases(case, dtype):
    store, (matrix, scalar) = _lower(case, dtype)
    owner, member, offset, threads = case
    ownership = store.ownership
    for phase, slot in itertools.product(range(8), range(4)):
        frame = (1 << 20) + 128 * phase + 49920 * slot
        complete = set()
        for step, warp in itertools.product(
            range(ownership.trips), range(threads // 32)
        ):
            row_origin = (
                warp * 8 + step // ownership.column_tiles * ownership.thread_rows
            )
            if row_origin >= 32:
                continue
            assert row_origin + 8 <= 32
            lanes, expected = [], {}
            for lane in range(32):
                thread = warp * 32 + lane
                stores, instructions = _run(matrix, thread, step, frame)
                assert not stores and len(instructions) == 1
                lanes.append(instructions[0])
                for element in range(8):
                    row = (
                        thread // 4
                        + step // ownership.column_tiles * ownership.thread_rows
                    )
                    column = (
                        thread % 4 * 8 + step % ownership.column_tiles * 32 + element
                    )
                    addresses, instructions = _run(scalar, row, column, frame)
                    raw = frame + 2 * ((offset + column) * 32 + row)
                    assert (
                        not instructions and addresses == [raw ^ ((raw & 384) >> 3)] * 2
                    )
                    assert addresses[0] not in expected
                    expected[addresses[0]] = thread * 8 + element
            actual = {}
            for number, row, column in itertools.product(range(4), range(8), range(8)):
                payload = lanes[column * 4 + row // 2][1][number][row % 2]
                address = lanes[number * 8 + row][0] + 2 * column
                assert address not in actual
                actual[address] = payload
            assert actual == expected and not complete.intersection(actual)
            complete.update(actual)
            # Unique symbolic halves prove all raw bit patterns. Also replay
            # representative half encodings, never converting them numerically.
            bits = (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7C00, 0xFC00, 0x7E01)
            assert {key: bits[value % len(bits)] for key, value in actual.items()} == {
                key: bits[value % len(bits)] for key, value in expected.items()
            }
        assert complete == set(
            range(frame + offset * 64, frame + (offset + member) * 64, 2)
        )
        assert min(complete) >= frame and max(complete) < frame + owner * 64
