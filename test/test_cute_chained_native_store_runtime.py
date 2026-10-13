from __future__ import annotations

import ast
from functools import partial
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from .test_cute_chained_frontier_ownership import _native_frame
from helion._compiler.cute.chained_frontier_groups import FrontierStores
from helion._compiler.cute.chained_native_stores import NativeStMatrixStore
from helion._compiler.cute.chained_native_stores import plan_native_stmatrix_store
from helion._compiler.cute.chained_prepared_groups import _member_layout
from helion._compiler.cute.chained_tcgen05 import _layout
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership
from helion.runtime import default_cute_launcher

if TYPE_CHECKING:
    from types import ModuleType

# Full owner rows, member rows, member origin, producer threads. The first
# geometry is the larger transposed member which motivated K_SW128 support;
# the others vary row guards, frame extent and member alignment independently.
_CASES = ((192, 128, 64, 128), (80, 32, 16, 64), (96, 64, 16, 256))
_K = 64
_BLOCKS = 24  # Eight 128-byte swizzle phases in each of three frame slots.


def _launch(
    module: ModuleType,
    source: torch.Tensor,
    guard: torch.Tensor,
    output: torch.Tensor,
    raw: torch.Tensor,
    threads: int,
) -> None:
    output.view(torch.int16).fill_(23130)
    default_cute_launcher(
        module.native_store_test,
        (_BLOCKS,),
        source,
        guard,
        output,
        raw,
        block=(threads, 1, 1),
    )


def _store(case: tuple[int, int, int, int], dtype: torch.dtype) -> NativeStMatrixStore:
    owner, member, offset, threads = case
    ownership = plan_vector_ownership((_K, member), threads, tile_columns=32)
    assert ownership is not None
    store = plan_native_stmatrix_store(
        (owner, _K), (_K, member), offset, dtype, ownership
    )
    assert store is not None
    return store


def _kernel_source(
    case: tuple[int, int, int, int], dtype: torch.dtype, *, matrix: bool, masked: bool
) -> str:
    store = _store(case, dtype)
    ownership = store.ownership
    owner, member, offset, threads = case
    frame_bytes = owner * _K * 2
    allocation_halves = (3 * frame_bytes + 1024) // 2
    dtype_name = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    emission = store.emit("target", "store", "thread", "step")
    lines = [
        "from __future__ import annotations",
        "import cutlass",
        "import cutlass.cute as cute",
        "from cutlass.cute.nvgpu import tcgen05",
        "@cute.kernel",
        "def native_store_test(source, guard, output, raw):",
        "    thread = cutlass.Int32(cute.arch.thread_idx()[0])",
        "    block = cutlass.Int32(cute.arch.block_idx()[0])",
        f"    storage = cute.arch.alloc_smem(cutlass.Uint8, {allocation_halves * 2}, alignment=128)",
        f"    raw_storage = cute.make_tensor(cute.recast_ptr(storage, dtype={dtype_name}), cute.make_layout({allocation_halves}))",
        f"    address = storage + (block % 8) * 128 + (block // 8) * {frame_bytes}",
        "    frame = cute.make_ptr(cutlass.Uint8, address.toint(), cute.AddressSpace.smem, assumed_align=128)",
        f"    full_ptr = cute.recast_ptr(frame, dtype={dtype_name})",
        *("    " + line for line in _layout("full", (owner, _K), 1, dtype_name)),
        f"    target_ptr = cute.recast_ptr(frame + {offset * _K * 2}, dtype={dtype_name})",
        *(
            "    " + line
            for line in _member_layout("target", (member, _K), dtype_name, (1, 0))
        ),
        f"    for index in cutlass.range(thread, {allocation_halves}, {threads}):",
        "        raw_storage[index] = guard[0]",
        "    cute.arch.sync_threads()",
    ]
    if matrix:
        lines.extend("    " + line for line in emission.setup)
    lines.extend(
        [
            f"    for step in cutlass.range({ownership.trips}, unroll=1):",
            f"        row = {ownership.row_expression('thread', 'step')}",
            f"        base = {ownership.base_expression('thread', 'step')}",
            f"        if row < {_K}:",
            "            for element in cutlass.range_constexpr(8):",
            "                value = source[row, base + element]",
        ]
    )
    if masked:
        # Masks choose the original scalar zero; every participating warp still
        # executes the complete StMatrix instruction with eight values per lane.
        lines.extend(
            [
                f"                if row >= {_K - 7} or base + element >= {member - 3}:",
                f"                    value = {dtype_name}(0)",
            ]
        )
    lines.append(
        f"                {emission.values}[element] = value"
        if matrix
        else f"                full[base + element + {offset}, row] = value"
    )
    if matrix:
        lines.append("            " + emission.copy)
    lines.extend(
        [
            "    cute.arch.sync_threads()",
            f"    for index in cutlass.range(thread, {owner * _K}, {threads}):",
            f"        output[block, index // {_K}, index % {_K}] = full[index // {_K}, index % {_K}]",
            f"    for index in cutlass.range(thread, {allocation_halves}, {threads}):",
            "        raw[block, index] = raw_storage[index]",
        ]
    )
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_k64_complete_frontier_frame_selects_proved_native_transport_cpu(
    dtype: torch.dtype,
) -> None:
    frame, group, _operands, bindings = _native_frame(dtype, 64)
    ownership = plan_vector_ownership((64, 64), 128, tile_columns=32)
    assert ownership is not None
    before = (repr(frame), repr(group), repr(bindings))
    initialized = torch.cuda.is_initialized()
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        policy = FrontierStores(True)
        selected = policy.select(frame, group, bindings, ownership)
        assert len(selected) == 1
        store = next(iter(selected.values()))
        assert store.full_shape == (128, 64) and store.shape == (64, 64)
        assert store.dtype == dtype and store.matches()
        assert (
            "partition_D(original_member)"
            in store.emit("original_member", "store", "thread", "step").setup[1]
        )
    assert not policy.activated  # A selection alone never claims emission.
    assert before == (repr(frame), repr(group), repr(bindings))
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_k64_runtime_wrapper_uses_actual_emission_and_uniform_instruction_guard_cpu(
    case: tuple[int, int, int, int], dtype: torch.dtype
) -> None:
    source = _kernel_source(case, dtype, matrix=True, masked=True)
    tree = ast.parse(source)
    emission = _store(case, dtype).emit("target", "store", "thread", "step")
    for line in (*emission.setup, emission.copy):
        assert line in source
    copies = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Expr)
            and ast.unparse(statement).startswith("cute.copy(store_copy,")
            for statement in node.body
        )
    ]
    assert len(copies) == 1 and ast.unparse(copies[0].test) == "row < 64"
    assert source.count("sync_threads()") == 2
    assert "K_SW128" in source
    assert f"frame + {case[2] * _K * 2}" in source
    initialized = torch.cuda.is_initialized()
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        module = PyCodeCache.load(source)
        assert callable(module.native_store_test)
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("masked", (False, True))
def test_k64_native_store_gpu_bits_masks_slots_guards_replay_and_immutable_input(
    case: tuple[int, int, int, int], dtype: torch.dtype, masked: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("StMatrix requires SM90 or newer")
    owner, member, offset, threads = case
    count = _K * member
    allocation_halves = (3 * owner * _K * 2 + 1024) // 2
    bits = ((torch.arange(count, device="cuda", dtype=torch.int32) * 40503) & 65535).to(
        torch.int16
    )
    # Both signs of zero, subnormals, infinities, and different NaN payloads.
    # All assertions use raw integer views, including signaling NaN payloads.
    specials = torch.tensor(
        [0, -32768, 1, -32767, 31744, -1024, 32257, 32640, -128, 32705],
        dtype=torch.int16,
        device="cuda",
    )
    bits[: specials.numel()] = specials
    source = bits.view(dtype).reshape(_K, member)
    before = source.view(torch.int16).clone()
    guard = torch.tensor([23130], dtype=torch.int16, device="cuda").view(dtype)
    expected = torch.full(
        (_BLOCKS, owner + 1, _K), 23130, dtype=torch.int16, device="cuda"
    )
    values = before.clone()
    if masked:
        values[_K - 7 :] = 0
        values[:, member - 3 :] = 0
    expected[:, offset : offset + member] = values.T
    outputs = []
    for matrix in (False, True):
        module = PyCodeCache.load(
            _kernel_source(case, dtype, matrix=matrix, masked=masked)
        )
        output = torch.empty(expected.shape, dtype=dtype, device="cuda")
        raw = torch.empty((_BLOCKS, allocation_halves), dtype=dtype, device="cuda")

        run = partial(_launch, module, source, guard, output, raw, threads)
        for _ in range(3):
            run()
            assert torch.equal(output.view(torch.int16), expected)
        raw_before = raw.view(torch.int16).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(3):
            output.view(torch.int16).zero_()
            raw.view(torch.int16).zero_()
            graph.replay()
            assert torch.equal(output.view(torch.int16), expected)
            assert torch.equal(raw.view(torch.int16), raw_before)
        outputs.append(raw_before)
        assert torch.equal(source.view(torch.int16), before)
        assert guard.view(torch.int16).item() == 23130
    # Compare every allocated half, including before/after each rebased owner.
    # A stale frame, double-applied offset or stray collective store is visible.
    assert torch.equal(outputs[0], outputs[1])
