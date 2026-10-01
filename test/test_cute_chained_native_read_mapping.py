from __future__ import annotations

import ast
from collections import Counter
from functools import lru_cache
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from helion._compiler.cute.chained_native_reads import plan_native_vector_read
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership
from helion.runtime import default_cute_launcher

_PIPELINE = (
    "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,"
    "convert-cute-to-core,canonicalize)"
)
_OUTPUT = 1 << 48
# Full owner, logical image, original row offset, producer threads, tile width.
_CASES = (
    ((32, 128), (32, 128), 0, 128, 32),
    ((64, 128), (32, 128), 32, 128, 32),
    ((64, 128), (32, 128), 0, 128, 0),
    ((8, 64), (7, 64), 1, 32, 8),
    ((48, 64), (33, 64), 7, 64, 32),
    ((96, 256), (65, 256), 17, 256, 64),
    ((128, 256), (97, 256), 7, 512, 32),
    ((16, 256), (9, 256), 3, 1024, 128),
    ((8, 64), (1, 64), 7, 1024, 64),
    ((24, 192), (17, 192), 5, 256, 64),
)


def _read(case, dtype):
    full, shape, offset, threads, columns = case
    ownership = plan_vector_ownership(shape, threads, tile_columns=columns)
    assert ownership is not None
    read = plan_native_vector_read(full, shape, offset, dtype, ownership)
    assert read is not None, f"native read unexpectedly rejected {case}, {dtype}"
    return read


def _native_setup(read):
    dtype = "cutlass.BFloat16" if read.dtype == torch.bfloat16 else "cutlass.Float16"
    return (
        f"native_layout = cute.tile_to_shape(tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, {dtype}), {read.full_shape}, order=(0, 1))",
        f"full = cute.make_tensor(cute.recast_ptr(frame, native_layout.inner, dtype={dtype}), native_layout.outer)",
        f"alias = cute.domain_offset(({read.row_offset}, 0), full)",
    )


def _execute(lines, scope):
    exec(compile("\n".join(lines), "<actual-native-read-emission>", "exec"), scope)


def _program(function):
    """Compile actual lowered operations into a small fail-closed interpreter."""
    block = function.regions[0].blocks[0]
    assert [str(arg.type) for arg in block.arguments] == ["i32", "i32", "i64"]
    ids = {arg: index for index, arg in enumerate(block.arguments)}
    program, counts = [], Counter()
    for view in block.operations:
        op = view.operation
        counts[op.name] += 1
        args = tuple(ids[arg] for arg in op.operands)
        results = []
        for result in op.results:
            ids[result] = len(ids)
            results.append(ids[result])
        attrs: dict[str, Any] = {
            "types": tuple(str(result.type) for result in op.results)
        }
        if op.name == "arith.constant":
            attrs["value"] = op.attributes["value"].value
            assert str(op.results[0].type).startswith("i")
        elif op.name == "llvm.getelementptr":
            attrs["size"] = {"i8": 1, "bf16": 2, "f16": 2, "i16": 2}[
                str(op.attributes["elem_type"].value)
            ]
            indices = list(op.attributes["rawConstantIndices"])
            assert len(indices) == 1 and len(args) in (1, 2)
            attrs["index"] = indices[0] if len(args) == 1 else None
        elif op.name == "llvm.alloca":
            assert str(op.attributes["elem_type"].value) == "i16"
        elif op.name == "llvm.load":
            attrs["shared"] = str(op.operands[0].type) == "!llvm.ptr<3>"
            value_type = str(op.results[0].type)
            attrs["vector"] = value_type in ("vector<8xbf16>", "vector<8xf16>")
            assert attrs["vector"] or value_type in ("bf16", "f16")
        elif op.name == "llvm.store":
            attrs["output"] = str(op.operands[1].type) == "!llvm.ptr<1>"
            assert str(op.operands[1].type) in ("!llvm.ptr<1>", "!llvm.ptr")
            value_type = str(op.operands[0].type)
            attrs["vector"] = value_type in ("vector<8xbf16>", "vector<8xf16>")
            assert attrs["vector"] or value_type in ("bf16", "f16")
        program.append((op.name, args, tuple(results), attrs))
    return program, len(ids), dict(counts)


@lru_cache(None)
def _lower_copy(case, dtype):
    """No device: actual get_slice(dynamic thread), dynamic step and frame base."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    read = _read(case, dtype)
    ir = importlib.import_module("cutlass._mlir.ir")
    func = importlib.import_module("cutlass._mlir.dialects.func")
    manager = importlib.import_module("cutlass._mlir.passmanager")
    initialized = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "native_copy",
                (
                    [ir.IntegerType.get_signless(32)] * 2
                    + [ir.IntegerType.get_signless(64)],
                    [],
                ),
            )
        with ir.InsertionPoint(function.add_entry_block()):
            scope: dict[str, Any] = {
                "cutlass": cutlass,
                "cute": cute,
                "tcgen05": tcgen05,
            }
            scope["thread"] = cutlass.Int32(function.arguments[0])
            scope["step"] = cutlass.Int32(function.arguments[1])
            scope["frame"] = cute.make_ptr(
                cutlass.Uint8,
                cutlass.Int64(function.arguments[2]),
                cute.AddressSpace.smem,
                assumed_align=128,
            )
            _execute(_native_setup(read), scope)
            emission = read.emit("alias", "read", "thread", "step")
            _execute((*emission.setup, emission.copy), scope)
            _execute(
                (
                    f"row = {read.ownership.row_expression('thread', 'step')}",
                    f"base = {read.ownership.base_expression('thread', 'step')}",
                ),
                scope,
            )
            element_type = (
                cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
            )
            output = cute.make_tensor(
                cute.make_ptr(
                    element_type, _OUTPUT, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout(24),
            )
            for element in range(8):
                output[element] = scope[emission.values][element]
                output[8 + element] = scope["alias"][
                    scope["row"], scope["base"] + element
                ]
                output[16 + element] = scope["full"][
                    scope["row"] + read.row_offset, scope["base"] + element
                ]
            func.ReturnOp([])
        assert module.operation.verify()
        before = str(module)
        manager.PassManager.parse(_PIPELINE).run(module.operation)
        assert module.operation.verify()
        lowered = str(module)
        compiled = _program(next(iter(module.body.operations)).operation)
    assert torch.cuda.is_initialized() == initialized
    return compiled, before, lowered


def _evaluate(compiled, thread, step, origin):
    program, count, _ = compiled
    values = [None] * count
    values[:3] = [thread, step, origin]
    private, output, vectors = {}, {}, []
    for name, args, results, attrs in program:
        operands = [values[arg] for arg in args]
        value = None
        if name == "arith.constant":
            value = attrs["value"]
        elif name == "arith.addi":
            value = operands[0] + operands[1]
        elif name == "arith.muli":
            value = operands[0] * operands[1]
        elif name in ("arith.divsi", "arith.floordivsi"):
            assert operands[0] >= 0 and operands[1] > 0
            value = operands[0] // operands[1]
        elif name == "arith.remsi":
            assert operands[0] >= 0 and operands[1] > 0
            value = operands[0] % operands[1]
        elif name == "arith.andi":
            value = operands[0] & operands[1]
        elif name == "arith.shrui":
            value = operands[0] >> operands[1]
        elif name == "arith.xori":
            value = operands[0] ^ operands[1]
        elif name in ("llvm.inttoptr", "llvm.ptrtoint"):
            value = operands[0]
        elif name == "llvm.getelementptr":
            index = operands[1] if attrs["index"] is None else attrs["index"]
            value = operands[0] + attrs["size"] * index
        elif name == "llvm.alloca":
            assert operands == [8]
            value = 1 << 40
        elif name == "llvm.load":
            addresses = tuple(
                operands[0] + 2 * index for index in range(8 if attrs["vector"] else 1)
            )
            if attrs["shared"]:
                # Tokens denote arbitrary independent 16-bit payloads. No
                # arithmetic or BF16/FP16 conversion is accepted by this VM.
                payloads = addresses
                if attrs["vector"]:
                    assert operands[0] % 16 == 0
                    vectors.append(addresses)
            else:
                payloads = tuple(private[address] for address in addresses)
            value = payloads if attrs["vector"] else payloads[0]
        elif name == "llvm.store":
            payload, pointer = operands
            payloads = payload if attrs["vector"] else (payload,)
            for index, item in enumerate(payloads):
                address = pointer + 2 * index
                if attrs["output"]:
                    assert address not in output
                    output[address] = item
                else:
                    private[address] = item
        elif name in ("llvm.intr.assume", "func.return"):
            pass
        else:
            raise AssertionError(f"Unsupported actual IR operation: {name}")
        if results:
            assert len(results) == 1
            if attrs["types"] == ("i32",):
                assert isinstance(value, int)
                assert -(1 << 31) <= value < 1 << 31
            values[results[0]] = value
    assert set(output) == {_OUTPUT + 2 * index for index in range(24)}
    assert len(vectors) == 1
    return tuple(output[_OUTPUT + 2 * index] for index in range(24)), vectors[0]


def _native_address(origin, height, row, column):
    # The pointer's swizzle acts on BYTES, not on the composed element index.
    outer = row * 64 + column % 64 + (column // 64) * height * 64
    address = origin + 2 * outer
    return address ^ ((address & 896) >> 3)


def _coordinates(ownership, thread, step):
    row_tile, column_tile = divmod(step, ownership.column_tiles)
    return (
        thread // ownership.thread_columns + row_tile * ownership.thread_rows,
        thread % ownership.thread_columns * 8 + column_tile * ownership.tile_columns,
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("case", _CASES)
def test_dynamic_actual_copy_preserves_all_native_payloads_cpu(case, dtype):
    read = _read(case, dtype)
    compiled, before, lowered = _lower_copy(case, dtype)
    assert "!cute" in before and "llvm.load" in lowered
    vector_type = "vector<8xbf16>" if dtype == torch.bfloat16 else "vector<8xf16>"
    assert vector_type in lowered
    ownership = read.ownership
    stride = 2 * read.full_shape[0] * read.full_shape[1]
    for phase in range(0, 1024, 128):
        slot_addresses = []
        for slot in range(3):
            origin = 2048 + phase + slot * stride
            cells, addresses = set(), set()
            for thread in range(ownership.threads):
                for step in range(ownership.trips):
                    row, base = _coordinates(ownership, thread, step)
                    # The component preserves the caller's original row guard.
                    if row >= read.shape[0]:
                        continue
                    payloads, loaded = _evaluate(compiled, thread, step, origin)
                    vector, scalar, full = payloads[:8], payloads[8:16], payloads[16:]
                    assert vector == scalar == full
                    assert vector != tuple(reversed(scalar))
                    expected = tuple(
                        _native_address(
                            origin, read.full_shape[0], row + read.row_offset, base + i
                        )
                        for i in range(8)
                    )
                    assert vector == loaded == expected
                    assert all(
                        origin <= address < origin + stride for address in loaded
                    )
                    for element, address in enumerate(loaded):
                        cell = (row, base + element)
                        assert cell not in cells and address not in addresses
                        cells.add(cell)
                        addresses.add(address)
            assert cells == {
                (row, col)
                for row in range(read.shape[0])
                for col in range(read.shape[1])
            }
            assert all(not addresses & other for other in slot_addresses)
            slot_addresses.append(addresses)


def test_full_owner_second_panel_and_payload_order_negative_controls_cpu():
    case = ((64, 128), (32, 128), 32, 128, 32)
    compiled, _, _ = _lower_copy(case, torch.bfloat16)
    dense_mismatches = []
    for step, column in ((0, 0), (2, 64)):
        copied, _ = _evaluate(compiled, 0, step, 2176)
        expected = copied[16:]
        dense = tuple(
            _native_address(2176 + 32 * 128 * 2, 32, 0, column + i) for i in range(8)
        )
        doubled_offset = tuple(
            _native_address(2176, 64, 64, column + i) for i in range(8)
        )
        assert copied[:8] == expected and expected != doubled_offset
        assert expected != tuple(reversed(expected))
        dense_mismatches.append(expected != dense)
    # A wrong dense rebase happens to coincide in the second panel. Sampling
    # only that panel would not establish the original full-owner alias.
    assert dense_mismatches == [True, False]


def _kernel_source(case, dtype):
    read = _read(case, dtype)
    ownership = read.ownership
    full_height, width = read.full_shape
    height = read.shape[0]
    stride = 2 * full_height * width
    emission = read.emit("alias", "read", "thread", "step")
    lines = [
        "from __future__ import annotations",
        "import cutlass",
        "import cutlass.cute as cute",
        "from cutlass.cute.nvgpu import tcgen05",
        "@cute.kernel",
        "def native_read_test(source, output):",
        "    thread = cutlass.Int32(cute.arch.thread_idx()[0])",
        "    block = cutlass.Int32(cute.arch.block_idx()[0])",
        f"    storage = cute.arch.alloc_smem(cutlass.Uint8, {3 * stride + 1024}, alignment=128)",
        f"    address = storage + (block % 8) * 128 + (block // 8) * {stride}",
        "    frame = cute.make_ptr(cutlass.Uint8, address.toint(), cute.AddressSpace.smem, assumed_align=128)",
        *("    " + line for line in _native_setup(read)),
        f"    for index in cutlass.range(thread, {full_height * width}, {ownership.threads}):",
        f"        full[index // {width}, index % {width}] = source[index // {width}, index % {width}]",
        "    cute.arch.sync_threads()",
        *("    " + line for line in emission.setup),
        f"    for step in cutlass.range({ownership.trips}, unroll=1):",
        f"        row = {ownership.row_expression('thread', 'step')}",
        f"        base = {ownership.base_expression('thread', 'step')}",
        f"        if row < {height}:",
        "            " + emission.copy,
        "            for element in cutlass.range_constexpr(8):",
        f"                output[block, row, base + element] = {emission.values}[element]",
    ]
    return "\n".join(lines) + "\n"


def test_native_read_wrapper_retains_row_guard_and_original_alias_cpu():
    source = _kernel_source(_CASES[4], torch.float16)
    tree = ast.parse(source)
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
    assert len(guards) == 1 and ast.unparse(guards[0].test) == "row < 33"
    assert ast.unparse(guards[0].body[0]).startswith("cute.copy(read_copy,")
    assert "alias = cute.domain_offset((7, 0), full)" in source
    assert "read_thread.partition_S(alias)" in source
    assert "sync_threads()" in source and "num_bits_per_copy=128" in source


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(
    "case", (_CASES[0], _CASES[1], _CASES[4], _CASES[5], _CASES[7])
)
def test_native_read_gpu_raw_bits_tails_slots_replay_and_input_immutability(
    case, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    read = _read(case, dtype)
    module = PyCodeCache.load(_kernel_source(case, dtype))
    count = read.full_shape[0] * read.full_shape[1]
    # Include both signs, zero, subnormal, infinity and NaN payload patterns;
    # comparisons operate on int16 views, never floating-point equality.
    bits = ((torch.arange(count, device="cuda", dtype=torch.int32) * 40503) & 65535).to(
        torch.int16
    )
    specials = torch.tensor(
        [0, -32768, 1, -32767, 31744, -1024, 32257, 32640, -128, 32705],
        dtype=torch.int16,
        device="cuda",
    )
    first = read.row_offset * read.shape[1]
    bits[first : first + specials.numel()] = specials
    source = bits.view(dtype).reshape(read.full_shape)
    before = source.view(torch.int16).clone()
    output = torch.empty(
        (24, read.shape[0] + 1, read.shape[1]), dtype=dtype, device="cuda"
    )
    expected = torch.full_like(output.view(torch.int16), 23130)
    expected[:, : read.shape[0]] = before[
        read.row_offset : read.row_offset + read.shape[0]
    ]

    def run():
        output.view(torch.int16).fill_(23130)
        default_cute_launcher(
            module.native_read_test,
            (24,),
            source,
            output,
            block=(read.ownership.threads, 1, 1),
        )

    for _ in range(3):
        run()
        assert torch.equal(output.view(torch.int16), expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        output.view(torch.int16).zero_()
        graph.replay()
        assert torch.equal(output.view(torch.int16), expected)
    assert torch.equal(source.view(torch.int16), before)
