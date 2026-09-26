from __future__ import annotations

import ast
from copy import deepcopy
import itertools
import re
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test import test_cute_philox_policy

import helion
from helion._compiler.cute.philox_packets import lower_philox_packets
from helion._compiler.cute.philox_stream import PACKET_ASM
from helion._compiler.cute.philox_stream import SCALAR_ASM
from helion._compiler.cute.philox_stream import SCALAR_CONSTRAINTS
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute.rng import stage_explicit_seed

cpu_only = test_cute_philox_policy.cpu_only

SEEDS = (
    0,
    1,
    -1,
    123,
    2**31 - 1,
    2**31,
    2**32 - 1,
    2**32,
    2**32 + 123,
    -(2**32),
    -(2**63),
    2**63 - 1,
)


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _oracle(seed, offset):
    # Independent scalar implementation retained from the prior rand4x oracle;
    # it neither imports generated helper constants nor interprets its program.
    counter = (offset // 4) & ((1 << 64) - 1)
    c0, c1, c2, c3 = counter & 0xFFFFFFFF, counter >> 32, 0, 0
    k0, k1 = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    for _ in range(10):
        a, b = 0xCD9E8D57 * c2, 0xD2511F53 * c0
        c0, c1, c2, c3 = (
            ((a >> 32) ^ c1 ^ k0) & 0xFFFFFFFF,
            a & 0xFFFFFFFF,
            ((b >> 32) ^ c3 ^ k1) & 0xFFFFFFFF,
            b & 0xFFFFFFFF,
        )
        k0, k1 = (k0 + 0x9E3779B9) & 0xFFFFFFFF, (k1 + 0xBB67AE85) & 0xFFFFFFFF
    signed = _signed((c0, c1, c2, c3)[offset % 4], 32)
    return _f32(
        _f32(float(-signed - 1 if signed < 0 else signed)) * _f32(4.6566127342e-10)
    )


def _program(asm):
    return [
        line.strip()
        for line in asm.splitlines()
        if line.strip() not in ("{", "}") and not line.startswith(".reg")
    ]


def _interpret(program, seed, offset, outputs):
    regs = {
        f"${outputs}": seed & ((1 << 64) - 1),
        f"${outputs + 1}": offset & ((1 << 64) - 1),
    }

    def val(name):
        if name in regs:
            return regs[name]
        if name.startswith("0f"):
            return struct.unpack(">f", bytes.fromhex(name[2:]))[0]
        return int(name, 0)

    for line in program:
        if line.startswith("mov.b64 {"):
            match = re.fullmatch(r"mov.b64 \{(\w+), (\w+)\}, ([^;]+);", line)
            a, b, c = match.groups()
            value = val(c)
            regs[a] = value & 0xFFFFFFFF
            regs[b] = (value >> 32) & 0xFFFFFFFF
            continue
        op, operands = line[:-1].split(" ", 1)
        dest, *args = [v.strip() for v in operands.split(",")]
        values = [val(a) for a in args]
        if op in ("mov.b32", "cvt.u32.u64"):
            result = values[0] & 0xFFFFFFFF
        elif op == "shr.s64":
            result = _signed(values[0], 64) >> values[1]
        elif op == "shr.s32":
            result = _signed(values[0], 32) >> values[1]
        elif op.startswith("mul.hi.u32"):
            result = ((values[0] & 0xFFFFFFFF) * (values[1] & 0xFFFFFFFF)) >> 32
        elif op == "mul.lo.u32":
            result = (values[0] * values[1]) & 0xFFFFFFFF
        elif op == "xor.b32":
            result = (values[0] ^ values[1]) & 0xFFFFFFFF
        elif op == "add.u32":
            result = (values[0] + values[1]) & 0xFFFFFFFF
        elif op == "and.b64":
            result = values[0] & values[1]
        elif op == "setp.eq.u32":
            result = values[0] == values[1]
        elif op == "selp.b32":
            result = values[0] if values[2] else values[1]
        elif op == "cvt.rn.f32.s32":
            result = _f32(float(_signed(values[0], 32)))
        elif op == "mul.rn.f32":
            result = _f32(values[0] * values[1])
        else:
            raise AssertionError(op)
        regs[dest] = result
    return tuple(regs[f"${i}"] for i in range(outputs))


@pytest.mark.parametrize("seed", SEEDS)
def test_actual_scalar_and_packet_assembly_against_independent_oracle(seed):
    scalar, packet = _program(SCALAR_ASM), _program(PACKET_ASM)
    offsets = [
        *range(257),
        *range(-9, 0),
        -(2**63),
        -(2**32) - 1,
        2**31 - 1,
        2**32 - 1,
        2**40 + 3,
        2**63 - 1,
    ]
    for offset in offsets:
        assert _interpret(scalar, seed, offset, 1)[0] == _oracle(seed, offset)
        base = offset // 4 * 4
        assert _interpret(packet, seed, base, 4)[offset % 4] == _oracle(seed, offset)


def test_distribution_and_repeat_determinism():
    program = _program(PACKET_ASM)
    values = [
        v
        for offset in range(0, 16384, 4)
        for v in _interpret(program, -(2**63) + 123, offset, 4)
    ]
    assert values[:256] == [_oracle(-(2**63) + 123, i) for i in range(256)]
    assert 0 <= min(values) < max(values) < 1
    assert abs(sum(values) / len(values) - 0.5) < 0.015
    histogram = [sum(int(v * 16) == i for v in values) for i in range(16)]
    assert all(abs(count - 1024) < 150 for count in histogram)
    assert values[:4] == list(_interpret(program, -(2**63) + 123, 0, 4))


def _packet_source(width=4):
    return f"""def kernel(start, length, seed_tensor, out):
    for lane in cutlass.range_constexpr({width}):
        index = cutlass.Int32(start) + cutlass.Int32(lane)
        offset = cutlass.Int64(index)
        value = _cute_inline_asm_elementwise((seed_tensor[0], offset), asm={SCALAR_ASM!r}, constraints={SCALAR_CONSTRAINTS!r}, dtype=cutlass.Float32, is_pure=True)
        if index < length:
            out.append((index, value))
"""


def _lower(source, seed_names=frozenset({"seed_tensor"})):
    module = ast.parse(source)
    ids = itertools.count()
    module.body[0].body = lower_philox_packets(
        module.body[0].body,
        seed_names=set(seed_names),
        integer_names={"start", "length"},
        new_name=lambda prefix: f"{prefix}_{next(ids)}",
    )
    return ast.fix_missing_locations(module)


def _run(module, start, length, seed):
    stats = {"scalar": 0, "packet": 0}
    out = []

    def asm(args, **kwargs):
        packet = kwargs["asm"] == PACKET_ASM
        stats["packet" if packet else "scalar"] += 1
        result = _interpret(_program(kwargs["asm"]), *args, 4 if packet else 1)
        return result if packet else result[0]

    namespace = {
        "cutlass": SimpleNamespace(
            Int32=lambda v: _signed(v, 32),
            Int64=lambda v: _signed(v, 64),
            Float32=_f32,
            range_constexpr=range,
        ),
        "_cute_inline_asm_elementwise": asm,
    }
    exec(compile(module, "<packet-proof>", "exec"), namespace)
    namespace["kernel"](start, length, [seed], out)
    return out, stats


@pytest.mark.parametrize("width", (4, 8))
def test_packet_transform_preserves_tails_wrap_alignment_and_seed(width):
    source = _packet_source(width)
    original = ast.parse(source)
    transformed = _lower(source)
    assert "philox_packet_" in ast.unparse(transformed)
    for start in (0, 1, 4, -8, -1, 2**31 - 4, 2**31 - 1):
        for length in (0, 1, 3, 7, 2**31):
            for seed in (123, -1, -(2**63), 2**63 - 1):
                expected, _ = _run(original, start, length, seed)
                actual, stats = _run(transformed, start, length, seed)
                assert actual == expected
                assert stats["packet"] == (width // 4 if start % 4 == 0 else 0)
                assert stats["scalar"] == (0 if start % 4 == 0 else width)


@pytest.mark.parametrize(
    "replacement",
    (
        "cutlass.Int64(index * scale)",
        "cutlass.Int64(index // scale)",
        "cutlass.Int64(source[index])",
        "cutlass.Int64(index >> scale)",
    ),
)
def test_unproved_offset_declines_to_same_new_scalar_stream(replacement):
    source = _packet_source().replace("cutlass.Int64(index)", replacement)
    assert ast.dump(_lower(source)) == ast.dump(ast.parse(source))


def test_nonprivate_seed_and_small_lane_loop_decline():
    source = _packet_source()
    assert ast.dump(_lower(source, set())) == ast.dump(ast.parse(source))
    source = _packet_source(2)
    assert ast.dump(_lower(source)) == ast.dump(ast.parse(source))


def test_fresh_seed_host_lifetime_and_signed_bits():
    x = torch.empty(3)
    outputs = [stage_explicit_seed(seed, x) for seed in SEEDS]
    assert len({value.data_ptr() for value in outputs}) == len(SEEDS)
    assert [value.item() for value in outputs] == list(SEEDS)
    assert all(value.dtype == torch.int64 and value.shape == (1,) for value in outputs)


def _config(width, packet):
    return helion.Config(
        block_sizes=[128 * width],
        num_warps=4,
        indexing="pointer",
        num_threads=128,
        cute_vector_widths=[width],
        cute_lane_layouts=["blocked"],
        **({"cute_rng_packet": packet} if packet is not None else {}),
    )


@skipUnlessBackends(["cute"])
def test_original_dynamic_reference_forward_backward_and_stream_choice(cpu_only):
    from examples.low_mem_dropout import low_mem_dropout
    from examples.low_mem_dropout import low_mem_dropout_bwd

    from helion._compiler.rng_utils import philox_rand_ref
    from helion.runtime.ref_mode import RefMode

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        values = torch.linspace(-2, 2, 33, dtype=dtype)
        for fn in (low_mem_dropout.fn, low_mem_dropout_bwd.fn):
            new = helion.kernel(
                fn,
                backend="cute",
                static_shapes=False,
                cute_rng_stream="philox4",
                ref_mode=RefMode.EAGER,
            )._bind_isolated((0.25, values, 123))
            old = helion.kernel(
                fn,
                backend="cute",
                static_shapes=False,
                cute_rng_stream="word0",
                ref_mode=RefMode.EAGER,
            )._bind_isolated((0.25, values, 123))
            for seed in (-(2**63), -1, 123, 2**32 + 123):
                for p in (
                    0.25,
                    0.0,
                    -0.0,
                    -0.5,
                    1.5,
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                ):
                    randoms = torch.tensor(
                        [_oracle(seed, i) for i in range(values.numel())]
                    )
                    expected = torch.where(
                        randoms > p, values.float() * (1.0 / (1.0 - p)), 0.0
                    ).to(dtype)
                    assert torch.equal(new.run_ref(p, values, seed), expected)
                old_expected = torch.where(
                    philox_rand_ref(seed, torch.arange(values.numel())) > 0.25,
                    values.float() * (4 / 3),
                    0.0,
                ).to(dtype)
                assert torch.equal(old.run_ref(0.25, values, seed), old_expected)
            with (
                patch(
                    "torch.empty_like",
                    side_effect=AssertionError("allocation before p=1"),
                ),
                pytest.raises(ZeroDivisionError),
            ):
                new.run_ref(1.0, values, 123)


@skipUnlessBackends(["cute"])
def test_original_emission_scalar_packet_normalizer_and_semantic_cache(cpu_only):
    from examples.low_mem_dropout import low_mem_dropout

    args = (0.25, torch.empty(1025), -(2**63))
    bound = helion.kernel(
        low_mem_dropout.fn,
        backend="cute",
        static_shapes=False,
        cute_rng_stream="philox4",
    )._bind_isolated(args)
    for width in (1, 4, 8):
        source = bound.to_triton_code(_config(width, True))
        assert "seed, x_flat)" in source and "_cute_stage_explicit_seed" in source
        assert ("philox_packet_" in source) == (width in (4, 8))
        assert SCALAR_ASM in [
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
    scalar = bound.to_triton_code(_config(4, False))
    assert "philox_packet_" not in scalar
    original = helion.kernel(
        low_mem_dropout.fn, backend="cute", static_shapes=False, cute_rng_stream="word0"
    )._bind_isolated(args)
    with pytest.raises(helion.exc.InvalidConfig):
        original.to_triton_code(_config(4, True))
    assert bound.settings.to_dict()["cute_rng_stream"] == "philox4"
    assert original.settings.to_dict()["cute_rng_stream"] == "word0"


def _explicit_offsets(seed: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(offsets, dtype=torch.float32)
    for tile in hl.tile(offsets.numel()):
        out[tile] = hl.rand([], seed=hl.load(seed, [0]), offsets=offsets[tile])
    return out


def _unsupported_rand4(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = hl.rand4x(123, hl.tile_index(tile).to(torch.int64))[0]
    return out


@skipUnlessBackends(["cute"])
def test_explicit_tensor_seed_offsets_reference_and_scalar_fallback(cpu_only):
    from helion.runtime.ref_mode import RefMode

    offsets = torch.tensor([-9, -1, 0, 1, 4, 7, 2**32 + 3], dtype=torch.int64)
    seed = torch.tensor([-(2**63)], dtype=torch.int64)
    kernel = helion.kernel(
        _explicit_offsets,
        backend="cute",
        static_shapes=False,
        cute_rng_stream="philox4",
        ref_mode=RefMode.EAGER,
    )
    bound = kernel._bind_isolated((seed, offsets))
    actual = bound.run_ref(seed, offsets)
    assert torch.equal(
        actual, torch.tensor([_oracle(seed.item(), i.item()) for i in offsets])
    )
    bound = helion.kernel(
        _explicit_offsets,
        backend="cute",
        static_shapes=False,
        cute_rng_stream="philox4",
    )._bind_isolated((seed, offsets))
    source = bound.to_triton_code(_config(4, True))
    assert "philox_packet_" not in source
    assert "_cute_stage_explicit_seed" not in source
    assert "cute_rng_stream='philox4'" in bound.format_kernel_decorator(
        _config(4, True), bound.settings
    )


@skipUnlessBackends(["cute"])
def test_unsupported_opt_in_rng_rejects_instead_of_old_stream(cpu_only):
    kernel = helion.kernel(
        _unsupported_rand4,
        backend="cute",
        static_shapes=False,
        cute_rng_stream="philox4",
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="explicit uniform"):
        kernel._bind_isolated((torch.empty(17),))


def test_environment_default_and_invalid_policy(cpu_only, monkeypatch):
    from examples.low_mem_dropout import low_mem_dropout

    monkeypatch.delenv("HELION_CUTE_RNG_STREAM", raising=False)
    assert helion.Settings(backend="cute").cute_rng_stream == "auto"
    assert helion.Settings(backend="triton").cute_rng_stream == "word0"
    monkeypatch.setenv("HELION_CUTE_RNG_STREAM", "philox4")
    assert helion.Settings().cute_rng_stream == "philox4"
    for setting in ("unknown", True):
        with pytest.raises(ValueError, match="cute_rng_stream"):
            helion.kernel(
                low_mem_dropout.fn, backend="cute", cute_rng_stream=setting
            )._bind_isolated((0.25, torch.empty(17), 123))


def _multiple_random(x: torch.Tensor, first: int, second: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        a = hl.rand([tile], seed=first)
        b = hl.rand(
            [], seed=second, offsets=hl.tile_index(tile).to(torch.int64) * 3 + 1
        )
        out[tile] = a - b
    return out


def _implicit_uniform(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = torch.rand([tile])
    return out


def _implicit_normal(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = torch.randn([tile])
    return out


@skipUnlessBackends(["cute"])
def test_multiple_distinct_rng_sites_keep_their_own_seed_and_offset(cpu_only):
    from helion.runtime.ref_mode import RefMode

    values = torch.empty(33)
    bound = helion.kernel(
        _multiple_random,
        backend="cute",
        static_shapes=False,
        cute_rng_stream="philox4",
        ref_mode=RefMode.EAGER,
    )._bind_isolated((values, -1, 2**32 + 123))
    expected = torch.tensor([_oracle(-1, i) for i in range(33)]) - torch.tensor(
        [_oracle(2**32 + 123, 3 * i + 1) for i in range(33)]
    )
    assert torch.equal(bound.run_ref(values, -1, 2**32 + 123), expected)
    bound = helion.kernel(
        _multiple_random, backend="cute", static_shapes=False, cute_rng_stream="philox4"
    )._bind_isolated((values, -1, 2**32 + 123))
    source = bound.to_triton_code(_config(4, True))
    assert "_cute_stage_explicit_seed(first, x)" in source
    assert "_cute_stage_explicit_seed(second, x)" in source
    assert "philox_packet_" not in source


@pytest.mark.parametrize(
    "function", (_implicit_uniform, _implicit_normal, _unsupported_rand4)
)
@skipUnlessBackends(["cute"])
def test_unsupported_rng_both_compiled_and_reference(function, cpu_only):
    from examples.low_mem_dropout import low_mem_dropout

    from helion.runtime.ref_mode import RefMode
    from helion.runtime.ref_mode import RefModeContext

    args = (torch.empty(17),)
    with pytest.raises(helion.exc.BackendUnsupported, match="explicit uniform"):
        helion.kernel(
            function, backend="cute", static_shapes=False, cute_rng_stream="philox4"
        )._bind_isolated(args)
    # Use an accepted pointwise environment so the refused op is reached in
    # reference execution without first invoking the compiled refusal.
    bound = helion.kernel(
        low_mem_dropout.fn,
        backend="cute",
        static_shapes=False,
        cute_rng_stream="philox4",
        ref_mode=RefMode.EAGER,
    )._bind_isolated((0.25, args[0], 123))
    with (
        RefModeContext(bound.env, _config(4, False)),
        pytest.raises(helion.exc.BackendUnsupported, match="explicit uniform"),
    ):
        function(*args)


def test_control_flow_write_invalidates_integer_packet_fact():
    source = _packet_source().replace(
        "    for lane", "    if length == 0:\n        start = 0.5\n    for lane"
    )
    assert ast.dump(_lower(source)) == ast.dump(ast.parse(source))


def _grid_seed(x: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        for tile in hl.tile(x.size(1)):
            out[row, tile] = hl.rand([tile], seed=seed + row)
    return out


@skipUnlessBackends(["cute"])
def test_device_derived_seed_keeps_scalar_new_stream(cpu_only):
    args = (torch.empty((2, 17)), 123)
    bound = helion.kernel(
        _grid_seed, backend="cute", static_shapes=False, cute_rng_stream="philox4"
    )._bind_isolated(args)
    source = bound.to_triton_code(_config(4, True))
    assert "_cute_stage_explicit_seed" not in source
    assert "philox_packet_" not in source
    assert "shr.s64 counter" in source


@skipUnlessBackends(["cute"])
def test_config_mutation_cannot_change_the_stream_policy(cpu_only):
    from examples.low_mem_dropout import low_mem_dropout

    args = (0.25, torch.empty(1025), 123)
    bound = helion.kernel(
        low_mem_dropout.fn,
        backend="cute",
        static_shapes=False,
        cute_rng_stream="philox4",
    )._bind_isolated(args)
    config = _config(4, False)
    scalar = bound.to_code(config)
    config.config["cute_rng_packet"] = True
    packet = bound.to_code(config)
    config.config["cute_rng_packet"] = False
    assert bound.to_code(config) == scalar
    assert (
        scalar != packet and "shr.s64 counter" in scalar and "shr.s64 counter" in packet
    )
    old = helion.kernel(
        low_mem_dropout.fn, backend="cute", static_shapes=False, cute_rng_stream="word0"
    )._bind_isolated(args)
    old_source = old.to_code(_config(4, None))
    assert old_source != scalar and "shr.s64 counter" not in old_source
    assert (
        bound.settings.cute_rng_stream == "philox4"
        and old.settings.cute_rng_stream == "word0"
    )


def test_augmented_write_invalidates_integer_packet_fact():
    source = _packet_source().replace("    for lane", "    start += 0.5\n    for lane")
    assert ast.dump(_lower(source)) == ast.dump(ast.parse(source))


def test_packet_loop_write_does_not_leave_stale_fact_for_next_loop():
    module = ast.parse(_packet_source())
    first = deepcopy(module.body[0].body[0])
    # The first loop can still packetize: it does not use or modify start until
    # after its RNG. The second loop's offset is no longer proved integral.
    first.body.append(ast.parse("length = 0.5").body[0])
    second = (
        ast.parse(
            _packet_source().replace("cutlass.Int32(start)", "cutlass.Int32(length)")
        )
        .body[0]
        .body[0]
    )
    module.body[0].body = [first, second]
    source = ast.unparse(ast.fix_missing_locations(module))
    lowered = _lower(source)
    packet_calls = [
        node
        for node in ast.walk(lowered)
        if isinstance(node, ast.Call)
        and any(
            k.arg == "asm"
            and isinstance(k.value, ast.Constant)
            and k.value.value == PACKET_ASM
            for k in node.keywords
        )
    ]
    assert len(packet_calls) == 1
