from __future__ import annotations

import ast
from functools import lru_cache
import operator
import re
import struct
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from examples.low_mem_dropout import low_mem_dropout
from examples.low_mem_dropout import low_mem_dropout_bwd
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import CutePointwiseVecHeuristic
from helion._compiler.cute import affine_vector_io
from helion._compiler.rng_utils import philox_rand_ref
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def _cpu_only():
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


def _config(width: int) -> helion.Config:
    return helion.Config(
        block_sizes=[128 * width],
        num_threads=[128],
        cute_vector_widths=[width],
        cute_lane_layouts=["blocked"],
        cute_cluster_n=1,
    )


@lru_cache(None)
def _capture(width: int):
    original = affine_vector_io.vectorize_affine_tile_lanes
    captured = []

    def record(body, function, constants):
        before = ast.parse(ast.unparse(ast.Module(body, []))).body
        result = original(body, function, constants)
        after = ast.parse(ast.unparse(ast.Module(result, []))).body
        captured.append((before, after))
        return result

    kernel = helion.kernel(
        low_mem_dropout.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_rng_stream="word0",
    )
    bound = _cpu_bind(kernel, (0.25, torch.empty(513), 123))
    assert bound.config_spec.pointwise_facts
    with patch.object(affine_vector_io, "vectorize_affine_tile_lanes", record):
        source = bound.to_code(_config(width))
    assert "_helion_affine_load" in source
    assert "_cute_store_u32_vec" in source
    assert len(captured) == 1
    # Device-body optimizations can emit shared pure helpers. Execute those
    # definitions as well as the captured body, preserving the full program.
    helpers = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(decorator) == "cute.jit" for decorator in node.decorator_list
        )
    ]
    for helper in helpers:
        helper.decorator_list = []
    return *captured[0], helpers


class _F32(float):
    def __new__(cls, value):
        return float.__new__(cls, struct.unpack("f", struct.pack("f", value))[0])

    def __mul__(self, other):
        return _F32(float(self) * float(_F32(other)))

    def bitcast(self, target):
        assert target is _U32
        return _U32(struct.unpack("I", struct.pack("f", self))[0])


class _U32(int):
    mlir_type = "u32"

    def __new__(cls, value):
        return int.__new__(cls, int(value) & 0xFFFFFFFF)

    def bitcast(self, target):
        assert target is _F32
        return _F32(struct.unpack("f", struct.pack("I", self))[0])


class _Pointer:
    def __init__(self, values, accesses, offset=0):
        self.values = values
        self.accesses = accesses
        self.offset = offset

    def __add__(self, offset):
        return _Pointer(self.values, self.accesses, self.offset + int(offset))

    def load(self):
        assert 0 <= self.offset < len(self.values)
        self.accesses.append(self.offset)
        return _F32(self.values[self.offset])

    def store(self, value):
        assert 0 <= self.offset < len(self.values)
        self.accesses.append(self.offset)
        self.values[self.offset] = _F32(value)


def _execute(body, width, data, seed, probability, helpers=()):
    input_values = data.tolist()
    output_values = [float("nan")] * len(input_values)
    reads, writes = [], []
    current = [0, 0]

    def vector_load(pointer, shape):
        return [(pointer + index).load().bitcast(_U32) for index in range(shape[0][0])]

    def vector_store(pointer, values):
        for index, value in enumerate(values):
            (pointer + index).store(value.bitcast(_F32))

    def asm(args, *, asm, constraints, dtype, is_pure):
        # Execute the actual pre-fusion scalar multiplication instructions.
        # The packet pass must keep each original instruction and argument.
        assert is_pure and constraints == "=l,l" and dtype is int
        match = re.search(r"mul\.(lo|hi)\.u32 product, (\d+), operand1;", asm)
        assert match is not None
        value = (int(args[0]) & 0xFFFFFFFF) * int(match[2])
        return (value if match[1] == "lo" else value >> 32) & 0xFFFFFFFF

    namespace: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            Int32=int,
            Int64=int,
            Uint32=_U32,
            Float32=_F32,
            range_constexpr=range,
            const_expr=bool,
        ),
        "cute": SimpleNamespace(
            is_static=lambda values: all(type(value) is int for value in values),
            size=lambda tensor, mode: tensor.layout.shape[mode[0]],
            arch=SimpleNamespace(
                thread_idx=lambda: (current[1], 0, 0),
                block_idx=lambda: (current[0], 0, 0),
                load=vector_load,
            ),
        ),
        "operator": operator,
        "ir": SimpleNamespace(VectorType=SimpleNamespace(get=lambda *args: args)),
        "_cute_inline_asm_elementwise": asm,
        "_cute_store_u32_vec": vector_store,
        "_BLOCK_SIZE_0": 128 * width,
    }
    function = ast.parse("def kernel(x_flat, out_flat, n, seed, p, scale): pass").body[
        0
    ]
    assert isinstance(function, ast.FunctionDef)
    function.body = body
    module = ast.fix_missing_locations(ast.Module([*helpers, function], []))
    exec(compile(module, "<actual-dropout-lane-program>", "exec"), namespace)
    inputs = SimpleNamespace(
        iterator=_Pointer(input_values, reads),
        layout=SimpleNamespace(shape=(len(input_values),), stride=(1,)),
    )
    outputs = SimpleNamespace(
        iterator=_Pointer(output_values, writes),
        layout=SimpleNamespace(shape=(len(output_values),), stride=(1,)),
    )
    for block in range((len(data) + 128 * width - 1) // (128 * width)):
        current[0] = block
        for thread in range(128):
            current[1] = thread
            namespace["kernel"](
                inputs, outputs, len(data), seed, probability, 1 / (1 - probability)
            )
    assert sorted(reads) == sorted(writes) == list(range(len(data)))
    return torch.tensor(output_values, dtype=torch.float32)


@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize("length", (7, 511, 513))
@pytest.mark.parametrize("seed", (123, -1, 0x1234567812345678))
def test_actual_generated_dropout_preserves_seeded_stream(
    width: int, length: int, seed: int
):
    before, after, helpers = _capture(width)
    data = torch.arange(length, dtype=torch.float32) / 37 - 2
    probability = 0.25
    expected = torch.where(
        philox_rand_ref(seed, torch.arange(length, dtype=torch.int64)) > probability,
        data * (1 / (1 - probability)),
        0.0,
    )
    original = _execute(before, width, data, seed, probability, helpers)
    actual = _execute(after, width, data, seed, probability, helpers)
    assert torch.equal(original.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(actual.view(torch.int32), original.view(torch.int32))


@pytest.mark.parametrize("width", (4, 8))
def test_packet_keeps_where_semantics_for_nan_infinity_and_signed_zero(width: int):
    before, after, helpers = _capture(width)
    data = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), -1.0] * 3
    )
    for probability in (0.0, 0.25, 0.6):
        expected = _execute(before, width, data, -1, probability, helpers)
        actual = _execute(after, width, data, -1, probability, helpers)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("function", (low_mem_dropout, low_mem_dropout_bwd))
def test_packet_seeds_survive_config_generation(function):
    kernel = helion.kernel(
        function.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_rng_stream="word0",
    )
    bound = _cpu_bind(kernel, (0.25, torch.empty(33554432), 123))
    configs = []
    with bound.env:
        seeds = CutePointwiseVecHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert seeds is not None
        generation = ConfigGeneration(bound.config_spec)
        for width in (4, 8):
            selected = next(
                seed
                for seed in seeds
                if seed.block_sizes == [128 * width]
                and seed.num_threads == [128]
                and seed.get("cute_vector_widths") == [width]
            )
            flat, recovered = generation.canonicalize_flat(generation.flatten(selected))
            assert recovered["cute_vector_widths"] == [width]
            assert recovered["num_threads"] == [128]
            configs.append(recovered)
    for config in configs:
        assert "_helion_affine_load" in bound.to_code(config)


@pytest.mark.parametrize("backend", ("cute", "triton"))
@pytest.mark.parametrize(
    "pure,assembly",
    (
        (True, "add.f32 $0, $1, $1;"),
        (False, "add.f32 $0, $1, $1;"),
        (True, "ld.global.f32 $0, [$1];"),
    ),
)
def test_only_proved_scalar_assembly_receives_pointwise_facts(
    pure: bool, assembly: str, backend: str
):
    @helion.kernel(backend=backend, autotune_effort="none")
    def kernel(x: torch.Tensor, assembly: hl.constexpr, pure: hl.constexpr):
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = hl.inline_asm_elementwise(  # pyrefly: ignore[no-matching-overload]
                assembly, "=f,f", [x[tile]], dtype=torch.float32, is_pure=pure, pack=1
            )
        return out

    bound = _cpu_bind(kernel, (torch.empty(1024), assembly, pure))
    assert bool(bound.config_spec.pointwise_facts) is (
        backend == "cute" and pure and assembly.startswith("add.")
    )
