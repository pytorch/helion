from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import importlib
import linecache
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.chained_register_fragments import RegisterElement
from helion._compiler.cute.chained_register_fragments import plan_warp_fragment_map
from helion._compiler.cute.chained_register_transport import emit_fragment_transport

_CASES = [
    (role, dtype, transpose)
    for role, dtypes in (
        ("a", (torch.float16, torch.bfloat16)),
        ("b", (torch.float16, torch.bfloat16)),
        ("c", (torch.float32,)),
    )
    for dtype in dtypes
    for transpose in (False, True)
]


def _code(mapping):
    return emit_fragment_transport(
        mapping, "source", "destination", prefix="transport", lane="lane"
    )


@dataclass(frozen=True)
class _Word:
    index: int


class _RawView:
    def __init__(self, output, pairs):
        self.output = output
        self.pairs = pairs

    def __getitem__(self, index):
        return _Word(index)

    def __setitem__(self, index, value):
        if not self.pairs:
            self.output[index] = value
        else:
            low, high = self.pairs[index]
            self.output[low] = value & 0xFFFF
            self.output[high] = value >> 16


def _simulate(mapping, values):
    """Execute emitted Python with independent PTX m8n8 and shuffle semantics."""
    primitives = importlib.import_module(
        "helion._compiler.cute.affine_recurrence_primitives"
    )
    lines = _code(mapping)
    assert lines is not None
    result = []
    for lane in range(32):
        source = values[lane]
        destination = [None] * 8

        def recast(tensor, dtype, source=source):
            return _RawView(
                tensor,
                mapping.source_word_pairs
                if tensor is source
                else mapping.destination_word_pairs,
            )

        def movmatrix(word, lane=lane):
            matrix = [[0] * 8 for _ in range(8)]
            pair = mapping.source_word_pairs[word.index]
            for source_lane in range(32):
                for half in range(2):
                    matrix[source_lane // 4][2 * (source_lane % 4) + half] = values[
                        source_lane
                    ][pair[half]]
            return sum(
                matrix[2 * (lane % 4) + half][lane // 4] << (16 * half)
                for half in range(2)
            )

        def shuffle(word, source_lane):
            return values[source_lane][word.index]

        namespace = {
            "source": source,
            "destination": destination,
            "lane": lane,
            "cutlass": SimpleNamespace(Int32=int),
            "cute": SimpleNamespace(
                recast_tensor=recast, arch=SimpleNamespace(shuffle_sync=shuffle)
            ),
        }
        with patch.object(primitives, "movmatrix_b16", movmatrix):
            exec("\n".join(lines), namespace)
        result.append(destination)
    return result


@pytest.mark.parametrize("role,dtype,transpose", _CASES)
def test_all_actual_maps_emit_bounded_exact_raw_permutations(role, dtype, transpose):
    mapping = plan_warp_fragment_map(role, dtype, transpose=transpose)
    assert mapping is not None
    lines = _code(mapping)
    assert lines is not None
    source = "\n".join(lines)
    values = [[32 * slot + lane for slot in range(8)] for lane in range(32)]
    expected = [
        [values[item.lane][item.slot] for item in row] for row in mapping.sources
    ]
    assert _simulate(mapping, values) == expected
    if mapping.same_lane:
        assert len(lines) == 8
        assert "recast" not in source and "shuffle" not in source
    elif dtype is torch.float32:
        assert source.count("shuffle_sync(") == 16
        assert source.count("recast_tensor(") == 2
        assert "movmatrix" not in source
    else:
        assert source.count("movmatrix_b16(") == 4
        assert "shuffle" not in source
    assert "Float32(" not in source and "Float16(" not in source
    assert "barrier" not in source and "sync_threads" not in source


@pytest.mark.parametrize("role,dtype,transpose", _CASES)
def test_emitted_transport_preserves_all_special_bit_patterns(role, dtype, transpose):
    mapping = plan_warp_fragment_map(role, dtype, transpose=transpose)
    assert mapping is not None
    patterns = {
        torch.float16: (
            0,
            0x8000,
            1,
            0x8001,
            0x3FF,
            0x400,
            0x7BFF,
            0x7C00,
            0xFC00,
            0x7C01,
            0xFE15,
        ),
        torch.bfloat16: (
            0,
            0x8000,
            1,
            0x8001,
            0x7F,
            0x80,
            0x7F7F,
            0x7F80,
            0xFF80,
            0x7F81,
            0xFFC5,
        ),
        torch.float32: (
            0,
            0x80000000,
            1,
            0x80000001,
            0x7FFFFF,
            0x800000,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0xFFC00005,
        ),
    }[dtype]
    values = [
        [patterns[(lane * 8 + slot) % len(patterns)] for slot in range(8)]
        for lane in range(32)
    ]
    assert _simulate(mapping, values) == [
        [values[item.lane][item.slot] for item in row] for row in mapping.sources
    ]


def test_word_selection_is_inferred_from_records_not_role_or_transpose():
    mapping = plan_warp_fragment_map("b", torch.bfloat16)
    assert mapping is not None
    # Same semantic values, different actual source-word storage order.
    mapping = replace(
        mapping, source_word_pairs=tuple(reversed(mapping.source_word_pairs))
    )
    lines = _code(mapping)
    assert lines is not None
    code = "\n".join(lines)
    assert (
        "destination_words[0] = transport_primitives.movmatrix_b16(transport_source_words[3])"
        in code
    )
    values = [[lane * 8 + slot for slot in range(8)] for lane in range(32)]
    assert _simulate(mapping, values) == [
        [values[item.lane][item.slot] for item in row] for row in mapping.sources
    ]


def _remap(mapping, operation):
    sources = tuple(
        tuple(operation(lane, slot, item) for slot, item in enumerate(row))
        for lane, row in enumerate(mapping.sources)
    )
    return replace(
        mapping,
        sources=sources,
        destination_coordinates=tuple(
            tuple(mapping.source_coordinates[item.lane][item.slot] for item in row)
            for row in sources
        ),
    )


@pytest.mark.parametrize(
    "case",
    (
        "rows",
        "slots",
        "source_bounds",
        "boolean_lane",
        "coordinate",
        "word_pairs",
        "dtype",
        "orientation",
        "alias",
    ),
)
def test_invalid_map_or_same_storage_is_rejected(case):
    mapping = plan_warp_fragment_map("b", torch.bfloat16)
    assert mapping is not None
    if case == "rows":
        mapping = replace(mapping, sources=mapping.sources[:-1])
    elif case == "slots":
        mapping = replace(
            mapping, sources=(mapping.sources[0][:-1], *mapping.sources[1:])
        )
    elif case in ("source_bounds", "boolean_lane"):
        first = RegisterElement(32 if case == "source_bounds" else True, 0)
        mapping = replace(
            mapping, sources=((first, *mapping.sources[0][1:]), *mapping.sources[1:])
        )
    elif case == "coordinate":
        mapping = replace(mapping, destination_coordinates=mapping.source_coordinates)
    elif case == "word_pairs":
        mapping = replace(mapping, source_word_pairs=((0, 0), (2, 3), (4, 5), (6, 7)))
    elif case == "dtype":
        mapping = replace(mapping, dtype=torch.float32)
    elif case == "orientation":
        mapping = replace(mapping, transpose=1)  # pyrefly: ignore [bad-argument-type]
    if case == "alias":
        assert (
            emit_fragment_transport(mapping, "same", "same", prefix="p", lane="lane")
            is None
        )
    else:
        assert _code(mapping) is None


@pytest.mark.parametrize(
    "case",
    ("lane_switch", "non_bit_lane", "many_slots", "non_bit_selector", "b16_shuffle"),
)
def test_valid_but_unbounded_ownership_has_no_giant_switch_fallback(case):
    mapping = plan_warp_fragment_map("c", torch.float32)
    assert mapping is not None
    if case == "lane_switch":
        mapping = _remap(
            mapping, lambda lane, slot, item: RegisterElement(lane, (slot + lane) % 8)
        )
    elif case == "non_bit_lane":
        mapping = _remap(
            mapping, lambda lane, slot, item: RegisterElement((lane + 1) % 32, slot)
        )
    elif case == "many_slots":
        mapping = _remap(
            mapping,
            lambda lane, slot, item: RegisterElement(lane ^ 1, (slot + lane) % 8),
        )
    elif case == "non_bit_selector":
        mapping = _remap(
            mapping,
            lambda lane, slot, item: RegisterElement(
                lane ^ 1, slot ^ int(lane % 3 == 0)
            ),
        )
    else:
        mapping = plan_warp_fragment_map("a", torch.float16)
        assert mapping is not None
        mapping = _remap(
            mapping, lambda lane, slot, item: RegisterElement(lane ^ 1, slot)
        )
    assert _code(mapping) is None


@pytest.mark.parametrize("role,dtype,transpose", _CASES)
def test_actual_cute_emission_verifies_raw_ir_without_cuda(role, dtype, transpose):
    cutlass = importlib.import_module("cutlass")
    cute = importlib.import_module("cutlass.cute")
    ir = importlib.import_module("cutlass._mlir.ir")
    mapping = plan_warp_fragment_map(role, dtype, transpose=transpose)
    assert mapping is not None
    value_type = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[dtype]
    initialized = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            before = str(module)
            lines = _code(mapping)
            assert lines is not None and str(module) == before
            atom = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(
                        cutlass.Float16 if dtype is torch.float32 else value_type,
                        cutlass.Float32,
                        (16, 8, 16),
                    )
                ),
                atom_layout_mnk=(1, 1, 1),
            )
            identity = cute.make_identity_tensor((16, 16))
            for lane in (0, 4, 17, 31):
                thread = atom.get_slice(lane)
                source = (
                    atom.make_fragment_C(thread.partition_C(identity).shape)
                    if role == "c"
                    else cute.make_rmem_tensor((8,), value_type)
                )
                destination = (
                    atom.make_fragment_A(thread.partition_A(identity).shape)
                    if role == "a"
                    else atom.make_fragment_B(thread.partition_B(identity).shape)
                    if role == "b"
                    else cute.make_rmem_tensor((8,), value_type)
                )
                source.fill(value_type(0))
                exec(
                    "\n".join(lines),
                    {
                        "source": source,
                        "destination": destination,
                        "lane": lane,
                        "cute": cute,
                        "cutlass": cutlass,
                    },
                )
        assert module.operation.verify()
        text = str(module)
        if not mapping.same_lane:
            assert (
                "movmatrix.sync.aligned.m8n8.trans.b16"
                if dtype is not torch.float32
                else "shfl.sync"
            ) in text
        assert "arith.truncf" not in text and "arith.extf" not in text
    assert torch.cuda.is_initialized() is initialized


@pytest.mark.parametrize("role,dtype,transpose", _CASES)
def test_dynamic_lane_full_warp_cpu_ptx_compile(role, dtype, transpose):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            role,
            str(dtype).split(".")[1],
            str(int(transpose)),
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "CUTE_DSL_ARCH": "sm_103a",
            "CUTE_DSL_KEEP_PTX": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _cpu_compile(role, dtype, transpose):
    import cutlass
    import cutlass.cute as cute

    mapping = plan_warp_fragment_map(role, dtype, transpose=transpose)
    assert mapping is not None
    lines = _code(mapping)
    assert lines is not None
    value_type = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[dtype]
    shape = "thread.partition_C(identity).shape"
    source_tensor = (
        f"atom.make_fragment_C({shape})"
        if role == "c"
        else "cute.make_rmem_tensor((8,), value_type)"
    )
    destination_tensor = (
        "cute.make_rmem_tensor((8,), value_type)"
        if role == "c"
        else f"atom.make_fragment_{role.upper()}(thread.partition_{role.upper()}(identity).shape)"
    )
    code = f"""import cutlass
import cutlass.cute as cute
@cute.kernel
def kernel(inputs: cute.Tensor, outputs: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    atom = cute.make_tiled_mma(cute.make_mma_atom(cute.nvgpu.warp.MmaF16BF16Op(input_type, cutlass.Float32, (16, 8, 16))), atom_layout_mnk=(1, 1, 1))
    thread = atom.get_slice(lane)
    identity = cute.make_identity_tensor((16, 16))
    source = {source_tensor}
    destination = {destination_tensor}
    for slot in cutlass.range_constexpr(8):
        source[slot] = inputs[lane, slot]
{textwrap.indent(chr(10).join(lines), "    ")}
    for slot in cutlass.range_constexpr(8):
        outputs[lane, slot] = destination[slot]
@cute.jit
def launch(inputs: cute.Tensor, outputs: cute.Tensor):
    kernel(inputs, outputs).launch(grid=(1, 1, 1), block=(32, 1, 1))
"""
    name = "_register_transport_cpu"
    path = f"<{name}>"
    module = types.ModuleType(name)
    module.__file__ = path
    module.__dict__.update(
        value_type=value_type, input_type=cutlass.Float16 if role == "c" else value_type
    )
    sys.modules[name] = module
    linecache.cache[path] = (len(code), None, code.splitlines(keepends=True), path)
    exec(compile(code, path, "exec"), module.__dict__)
    fake = cute.runtime.make_fake_tensor(value_type, (32, 8), (8, 1), assumed_align=16)
    with tempfile.TemporaryDirectory(prefix="helion_register_transport_") as directory:
        ptx = cute.compile(
            module.launch, fake, fake, options=f"--dump-dir {directory}"
        ).__ptx__
    if not mapping.same_lane:
        assert (
            "movmatrix.sync.aligned.m8n8.trans.b16"
            if dtype is not torch.float32
            else "shfl.sync.bfly.b32"
            if "shfl.sync.bfly.b32" in ptx
            else "shfl.sync.idx.b32"
        ) in ptx
    assert "cvt." not in ptx
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not torch.cuda.is_initialized()
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        _cpu_compile(
            sys.argv[1],
            {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[sys.argv[2]],
            bool(int(sys.argv[3])),
        )
