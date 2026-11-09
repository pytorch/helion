from __future__ import annotations

import ast
from copy import deepcopy
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest
import torch

from test.test_cute_affine_vector_io import _Float32
from test.test_cute_affine_vector_io import _Pointer
from test.test_cute_affine_vector_io import _Tensor
from test.test_cute_affine_vector_io import _Uint32
from test.test_cute_affine_vector_io import _vector_load
from test.test_cute_affine_vector_io import _vector_store
from test.test_cute_philox_stream import _oracle

import helion
from helion._compiler.cute.pure_lane_packets import _PacketVectorizer
from helion._compiler.cute.pure_lane_packets import _pure_packet_call
from helion._compiler.cute.pure_lane_packets import is_pure_inline_asm_call
from helion._compiler.cute.pure_lane_packets import is_pure_scalar_assembly
from helion._compiler.rng_utils import philox_rand_ref
from helion._testing import import_example
import helion.language as hl

_examples_low_mem_dropout = import_example("low_mem_dropout")
low_mem_dropout = _examples_low_mem_dropout.low_mem_dropout
low_mem_dropout_bwd = _examples_low_mem_dropout.low_mem_dropout_bwd

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.cpu_only

_SOURCE = """
def kernel(src, out, start: cutlass.Int32, length: cutlass.Int32):
    for tile_offset_0 in range(0, length, 8 * WIDTH):
        for lane_0 in range(8):
            lane_base_0 = cutlass.Int32(tile_offset_0) + cutlass.Int32(lane_0) * WIDTH
            for vec_lane_0 in cutlass.range_constexpr(WIDTH):
                index = lane_base_0 + cutlass.Int32(vec_lane_0)
                valid = index < length
                address = cutlass.Int32(start) + index
                value = (src.iterator + cutlass.Int32(address)).load() if valid else cutlass.Float32(0)
                bits = _cute_inline_asm_elementwise((cutlass.Int64(index),), asm='mul.lo.u32 $0, $1, 3;', constraints='=l,l', dtype=cutlass.Int64, is_pure=True)
                selected = operator.gt(bits & 7, 2)
                result = value * 1.25 if selected else cutlass.Float32(0)
                if valid:
                    (out.iterator + cutlass.Int32(address)).store(cutlass.Float32(result))
"""


def _rewrite(
    source: str, width: int = 4, *, disjoint: bool = True, alignment: int = 16
):
    module = ast.parse(source.replace("WIDTH", str(width)))
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    vectorizer = _PacketVectorizer(
        function.body,
        strides={("src", 0): 1, ("out", 0): 1},
        alignments={"src": alignment, "out": alignment},
        dtypes={"src": "cutlass.Float32", "out": "cutlass.Float32"},
        disjoint={frozenset(("src", "out"))} if disjoint else set(),
        constexpr={},
    )
    # The fixture's scalar annotations supply the same integer scope facts
    # that production traversal learns for its generated induction variables.
    integer_names = frozenset(
        argument.arg
        for argument in function.args.args
        if argument.annotation is not None
        and ast.unparse(argument.annotation) == "cutlass.Int32"
    )
    function.body = vectorizer.body(function.body, int32_names=integer_names)
    return ast.fix_missing_locations(module), vectorizer.changed


def _run(module: ast.Module, start: float, length: int, *, int32=int, int64=int):
    stats = dict.fromkeys(
        ("scalar_loads", "scalar_stores", "vector_loads", "vector_stores"), 0
    )
    values = [float(index - 20) for index in range(128)]
    output = [999.0] * 128
    src = _Tensor(iterator=_Pointer(values, stats), layout=SimpleNamespace(stride=(1,)))
    out = _Tensor(iterator=_Pointer(output, stats), layout=SimpleNamespace(stride=(1,)))
    namespace: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            Int32=int32,
            Int64=int64,
            Uint32=_Uint32,
            Float32=_Float32,
            range_constexpr=range,
        ),
        "operator": operator,
        "cute": SimpleNamespace(arch=SimpleNamespace(load=_vector_load)),
        "ir": SimpleNamespace(
            VectorType=SimpleNamespace(get=lambda shape, dtype: (shape, dtype))
        ),
        "_cute_store_u32_vec": _vector_store,
        "_cute_inline_asm_elementwise": lambda args, **kwargs: (
            (args[0] * 3) & 0xFFFFFFFF
        ),
    }
    _Uint32.mlir_type = "i32"
    exec(compile(module, "<pure-lane-packets>", "exec"), namespace)
    namespace["kernel"](src, out, start, length)
    return output, stats


@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize("length", (0, 1, 3, 4, 7, 8, 9, 17, 32, 65))
@pytest.mark.parametrize("start", (0, 1, 4, 8))
def test_packet_values_addresses_and_scalar_tails(width: int, length: int, start: int):
    original = ast.parse(_SOURCE.replace("WIDTH", str(width)))
    actual, changed = _rewrite(_SOURCE, width)
    assert changed == 1
    expected, _ = _run(original, start, length)
    output, statistics = _run(actual, start, length)
    assert output == expected
    vector_elements = (length // width) * width if start % width == 0 else 0
    assert (
        statistics["vector_loads"]
        == statistics["vector_stores"]
        == vector_elements // 4
    )
    assert (
        statistics["scalar_loads"]
        == statistics["scalar_stores"]
        == length - vector_elements
    )


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("length", [0, 3, 17, 32])
def test_snapshot_gate_preserves_packet_induction_names(width: int, length: int):
    source = _SOURCE.replace("vec_lane_0", "packet_element")
    original = ast.parse(source.replace("WIDTH", str(width)))
    transformed, changed = _rewrite(source, width)
    # The packet prefilter accepts this name, but its delegated affine emitter
    # does not. Traversal eligibility must not broaden the emitter's contract.
    assert changed == 0
    assert ast.dump(transformed) == ast.dump(original)
    expected, _ = _run(original, 0, length)
    actual, _ = _run(transformed, 0, length)
    assert actual == expected


@pytest.mark.parametrize("width", (4, 8))
def test_packet_memory_surrounds_unchanged_lane_arithmetic(width: int):
    module, changed = _rewrite(_SOURCE, width)
    assert changed == 1
    branch = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.If) and len(node.body) > 1
    )
    loop_position = next(
        i for i, node in enumerate(branch.body) if isinstance(node, ast.For)
    )
    before = ast.unparse(ast.Module(branch.body[:loop_position], []))
    after = ast.unparse(ast.Module(branch.body[loop_position + 1 :], []))
    assert before.count("cute.arch.load") == width // 4
    assert after.count("_cute_store_u32_vec") == width // 4
    original = ast.parse(_SOURCE.replace("WIDTH", str(width)))
    before_calls = [
        node
        for node in ast.walk(original)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_cute_inline_asm_elementwise"
    ]
    after_calls = [
        node
        for node in ast.walk(branch.body[loop_position])
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_cute_inline_asm_elementwise"
    ]
    assert [ast.dump(node) for node in after_calls] == [
        ast.dump(node) for node in before_calls
    ]


@pytest.mark.parametrize(
    "mask", ("valid and (index % 4 != 1)", "valid and (index % 8 != 6)")
)
def test_mask_holes_retain_exact_scalar_predicates(mask: str):
    source = (
        _SOURCE.replace(
            "valid = index < length",
            f"valid = index < length\n                valid_hole = {mask}",
        )
        .replace("if valid else", "if valid_hole else")
        .replace("if valid:", "if valid_hole:")
    )
    module, changed = _rewrite(source, 8)
    assert changed == 1
    expected, _ = _run(ast.parse(source.replace("WIDTH", "8")), 0, 65)
    actual, stats = _run(module, 0, 65)
    assert actual == expected
    assert stats["vector_loads"] == stats["vector_stores"] == 0


@pytest.mark.parametrize("value", ("2.5", "-0.0"))
def test_numeric_mask_keeps_its_original_value(value: str):
    source = _SOURCE.replace(
        "valid = index < length",
        f"valid = (index < length) and cutlass.Float32({value})",
    ).replace("result = value * 1.25", "result = value * valid")
    transformed, changed = _rewrite(source)
    assert changed == 1
    expected, _ = _run(ast.parse(source.replace("WIDTH", "4")), 0, 65)
    actual, _ = _run(transformed, 0, 65)
    assert actual == expected


def test_captured_float_comparison_needs_no_float_arithmetic_replay():
    source = _SOURCE.replace(
        "    for tile_offset_0",
        "    captured = cutlass.Float32(0.5)\n    for tile_offset_0",
    ).replace("valid = index < length", "valid = (index < length) and captured > 0")
    transformed, changed = _rewrite(source)
    assert changed == 1
    assert (
        _run(transformed, 0, 65)[0]
        == _run(ast.parse(source.replace("WIDTH", "4")), 0, 65)[0]
    )


@pytest.mark.parametrize(
    "predicate",
    (
        "(index < length) and cutlass.Float32(index) * 1.25 > 1.0",
        "(index < length) and cutlass.Int32(cutlass.Float32(index) * 1.25) > 1",
    ),
)
def test_float_predicate_arithmetic_is_not_rematerialized(predicate: str):
    source = _SOURCE.replace("valid = index < length", f"valid = {predicate}")
    transformed, changed = _rewrite(source)
    assert changed == 0
    assert ast.dump(transformed) == ast.dump(ast.parse(source.replace("WIDTH", "4")))


@pytest.mark.parametrize("width", (4, 8))
def test_mixed_width_mask_wrap_checks_every_lane(width: int):
    source = _SOURCE.replace(
        "valid = index < length",
        "valid = (index < length) and cutlass.Int64(2) + cutlass.Int32(cutlass.Int32(2147483646) + cutlass.Int32(vec_lane_0)) < cutlass.Int64(2147483648)",
    )
    transformed, changed = _rewrite(source, width)
    assert changed == 1
    with np.errstate(over="ignore"):
        expected, original_stats = _run(
            ast.parse(source.replace("WIDTH", str(width))),
            0,
            32,
            int32=np.int32,
            int64=np.int64,
        )
        actual, actual_stats = _run(transformed, 0, 32, int32=np.int32, int64=np.int64)
    assert actual == expected
    assert actual_stats == original_stats
    assert actual_stats["scalar_stores"] == (32 // width) * (width - 2)


@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize("store", (False, True))
@pytest.mark.parametrize("operation", ("//", "%", "<<", ">>"))
def test_inactive_pointer_operations_keep_original_evaluation(
    width: int, store: bool, operation: str
):
    predicate = "start != 0" if operation in ("//", "%") else "start >= 0"
    invalid_start = 0 if operation in ("//", "%") else -1
    pointer = (
        f"cutlass.Int32(cutlass.Int32(8) {operation} cutlass.Int32(start) + index)"
        if operation in ("//", "%")
        else f"cutlass.Int32((cutlass.Int32(8) {operation} cutlass.Int32(start)) + index)"
    )
    if store:
        # Loads remain live while only the destination pointer is masked.
        source = (
            _SOURCE.replace("if valid:", f"if valid and ({predicate}):")
            .replace("address = cutlass.Int32(start) + index", "address = index")
            .replace(
                "(out.iterator + cutlass.Int32(address))",
                f"(out.iterator + {pointer})",
            )
        )
    else:
        source = _SOURCE.replace(
            "valid = index < length", f"valid = (index < length) and ({predicate})"
        ).replace(
            "(src.iterator + cutlass.Int32(address))",
            f"(src.iterator + {pointer})",
        )
    original = ast.parse(source.replace("WIDTH", str(width)))
    transformed, changed = _rewrite(source, width)
    assert changed == 0
    assert ast.dump(transformed) == ast.dump(original)
    for start in (invalid_start, 1):
        expected, before = _run(original, start, 32)
        actual, after = _run(transformed, start, 32)
        assert actual == expected
        assert after == before
        if start == invalid_start:
            assert after["scalar_stores"] == 0
            if not store:
                assert after["scalar_loads"] == 0
        else:
            assert after["scalar_loads"] == after["scalar_stores"] == 32


@pytest.mark.parametrize("width", (4, 8))
def test_masked_float_to_integer_pointer_cast_keeps_original_evaluation(width: int):
    source = (
        _SOURCE.replace("start: cutlass.Int32", "start: cutlass.Float32")
        .replace("address = cutlass.Int32(start) + index", "address = index")
        .replace(
            "valid = index < length", "valid = (index < length) and (start == start)"
        )
        .replace(
            "(src.iterator + cutlass.Int32(address))",
            "(src.iterator + cutlass.Int32(cutlass.Int32(start) + index))",
        )
    )
    original = ast.parse(source.replace("WIDTH", str(width)))
    transformed, changed = _rewrite(source, width)
    assert changed == 0
    assert ast.dump(transformed) == ast.dump(original)
    for start in (float("nan"), 1.0):
        expected, before = _run(original, start, 32)
        actual, after = _run(transformed, start, 32)
        assert actual == expected
        assert after == before
        if start != start:
            assert after["scalar_loads"] == after["scalar_stores"] == 0
        else:
            assert after["scalar_loads"] == after["scalar_stores"] == 32


@pytest.mark.parametrize(
    "asm",
    (
        "ld.global.u32 $0, [$1];",
        "st.global.u32 [$0], $1;",
        "bar.sync 0;",
        "mov.u32 $0, %tid.x;",
        "label: add.u32 $0, $1, 1;",
        "unknown.u32 $0, $1;",
    ),
)
def test_declared_pure_assembly_with_unmodelled_behavior_is_rejected(asm: str):
    call = ast.parse(
        f"_cute_inline_asm_elementwise((index,), asm={asm!r}, constraints='=l,l', dtype=cutlass.Int64, is_pure=True)",
        mode="eval",
    ).body
    assert isinstance(call, ast.Call)
    assert not _pure_packet_call(call)
    assert not is_pure_inline_asm_call(call)


_F32X2_MULTIPLY_ASM = """
{
    .reg .b64 ra, rb, rc;
    mov.b64 ra, { $2, $3 };
    mov.b64 rb, { $4, $5 };
    mul.f32x2 rc, ra, rb;
    mov.b64 { $0, $1 }, rc;
}
"""


def test_brace_grouped_register_operands_are_scalar_arithmetic():
    # PTX braces only group: the nested scope around register declarations
    # and the ``{ $2, $3 }`` vector operand that packs two 32-bit registers
    # into a 64-bit one (``mul.f32x2``); the opcode allowlist still gates
    # every instruction inside.
    assert is_pure_scalar_assembly(_F32X2_MULTIPLY_ASM, "=r,=r,r,r,r,r", True)
    assert not is_pure_scalar_assembly(
        _F32X2_MULTIPLY_ASM.replace("mul.f32x2", "ld.global.v2.f32"),
        "=r,=r,r,r,r,r",
        True,
    )
    assert not is_pure_scalar_assembly(_F32X2_MULTIPLY_ASM, "=r,=r,r,r,r,r", False)


def test_packed_pure_asm_is_pure_but_not_a_packet_call():
    # ``pack=2`` keeps the asm pure (the lowering replays it once per
    # element), so the lane passes may relocate it; the packet pass still
    # declines the packed call shape it does not model.
    call = ast.parse(
        "_cute_inline_asm_elementwise((a, b), asm="
        f"{_F32X2_MULTIPLY_ASM!r}, constraints='=r,=r,r,r,r,r', "
        "dtype=cutlass.Float32, is_pure=True, pack=2)",
        mode="eval",
    ).body
    assert isinstance(call, ast.Call)
    assert is_pure_inline_asm_call(call)
    assert not _pure_packet_call(call)
    call.keywords = [keyword for keyword in call.keywords if keyword.arg != "pack"]
    assert is_pure_inline_asm_call(call)
    assert _pure_packet_call(call)


@pytest.mark.parametrize(
    "change",
    (
        lambda source: source.replace("is_pure=True", "is_pure=False"),
        lambda source: source.replace("value * 1.25", "unknown(value)"),
        lambda source: source.replace("result = value", "result = result + value"),
        lambda source: source + "    live_out = result\n",
        lambda source: source.replace(
            "for lane_0 in", "seen = result\n        for lane_0 in"
        ),
        lambda source: source.replace(
            "address = cutlass.Int32(start) + index",
            "address = cutlass.Int32(cutlass.Float32(start) + index)",
        ),
        lambda source: source.replace("value * 1.25", "value * src[0]"),
        lambda source: source.replace(
            "src.iterator +", "(src.iterator if index > 0 else out.iterator) +"
        ),
        lambda source: source.replace(
            "    for tile_offset_0", "    operator = replacement\n    for tile_offset_0"
        ),
        lambda source: source.replace(
            "for lane_0 in range(8):", "for operator in range(8):"
        ),
    ),
)
def test_unsafe_packets_preserve_original_ast(change):
    source = change(_SOURCE)
    before = ast.parse(source.replace("WIDTH", "4"))
    after, changed = _rewrite(source)
    assert changed == 0
    assert ast.dump(after) == ast.dump(before)


@pytest.mark.parametrize("options", ({"disjoint": False}, {"alignment": 1}))
def test_unproven_storage_keeps_scalar_code(options):
    after, changed = _rewrite(_SOURCE, **options)
    assert changed == 0
    assert ast.dump(after) == ast.dump(ast.parse(_SOURCE.replace("WIDTH", "4")))


def test_long_unrelated_arithmetic_dag_does_not_expand_for_address_proof():
    statements = ["hash_0 = cutlass.Int32(index)"]
    for index in range(80):
        statements.append(f"hash_{index + 1} = hash_{index} ^ (hash_{index} << 1)")
    source = _SOURCE.replace(
        "selected = operator.gt(bits & 7, 2)",
        "\n                ".join(statements)
        + "\n                selected = operator.gt(hash_80 & 7, 2)",
    )
    original = ast.parse(source.replace("WIDTH", "4"))
    transformed, changed = _rewrite(source)
    assert changed == 1
    assert _run(transformed, 0, 65)[0] == _run(original, 0, 65)[0]


def test_oversized_address_dag_declines_entire_proof():
    statements = ["address_0 = cutlass.Int32(index)"]
    for index in range(24):
        statements.append(f"address_{index + 1} = address_{index} + address_{index}")
    source = _SOURCE.replace(
        "address = cutlass.Int32(start) + index",
        "\n                ".join(statements)
        + "\n                address = address_24",
    )
    original = ast.parse(source.replace("WIDTH", "4"))
    transformed, changed = _rewrite(source)
    assert changed == 0
    assert ast.dump(transformed) == ast.dump(deepcopy(original))


CUDA_DEVICE = "cuda"


def _config(width: int, layout: str = "blocked") -> helion.Config:
    return helion.Config(
        block_sizes=[256 * width],
        num_threads=[128],
        cute_vector_widths=[width],
        cute_lane_layouts=[layout],
        cute_cluster_n=1,
    )


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    integer = torch.int32 if actual.element_size() == 4 else torch.int16
    assert torch.equal(actual[~nan].view(integer), expected[~nan].view(integer))


def _mutating_graph(
    invoke: Callable[[], torch.Tensor],
    change_inputs: Callable[[int], object],
    reference: Callable[[], torch.Tensor],
) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = invoke()
    for iteration in range(3):
        change_inputs(iteration)
        expected = reference()
        graph.replay()
        torch.cuda.synchronize()
        _assert_exact(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float16, torch.bfloat16), ids=("f32", "f16", "bf16")
)
@pytest.mark.parametrize("backward", (False, True))
def test_dropout_packets_exact(width: int, dtype: torch.dtype, backward: bool) -> None:
    function = low_mem_dropout_bwd if backward else low_mem_dropout
    kernel = helion.kernel(
        function.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_rng_stream="word0",
    )
    config = _config(width)
    for length in (8192, 8221):
        _dropout_case(kernel, config, length, dtype)


def _dropout_case(
    kernel: helion.Kernel, config: helion.Config, length: int, dtype: torch.dtype
) -> None:
    data = torch.linspace(-2, 2, length, device=CUDA_DEVICE, dtype=dtype)
    data[:5] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
        device=CUDA_DEVICE,
        dtype=dtype,
    )
    bound = kernel._bind_isolated((0.25, data, 123))
    source = bound.to_code(config)
    assert "_helion_affine_load" in source
    assert "_helion_affine_store" in source
    bound.set_config(config)

    def expected(seed: int, probability: float) -> torch.Tensor:
        random = philox_rand_ref(seed, torch.arange(length, dtype=torch.int64)).to(
            device=CUDA_DEVICE
        )
        return torch.where(
            random > probability,
            data.float() * (1 / (1 - probability)),
            0.0,
        ).to(dtype)

    for seed, probability in ((123, 0.25), (-1, 0.6), (0x1234567812345678, 0.0)):
        _assert_exact(bound(probability, data, seed), expected(seed, probability))
    replacement = data.clone()
    _assert_exact(bound(0.25, replacement, 123), expected(123, 0.25))

    def mutate(iteration: int) -> None:
        data.copy_(
            torch.linspace(
                -1 - iteration, 3 + iteration, length, device=CUDA_DEVICE, dtype=dtype
            )
        )

    _mutating_graph(lambda: bound(0.25, data, 123), mutate, lambda: expected(123, 0.25))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("policy", (None, "philox4"), ids=("default", "philox4"))
@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_philox4_masks_config_independent_and_poisoned_replay(
    policy: str | None, backward: bool, dtype: torch.dtype
) -> None:
    function = low_mem_dropout_bwd if backward else low_mem_dropout
    if policy is None:
        kernel = helion.kernel(
            function.fn, backend="cute", static_shapes=False, autotune_effort="none"
        )
    else:
        kernel = helion.kernel(
            function.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
            cute_rng_stream=policy,
        )
    data = torch.linspace(-2, 2, 517, device=CUDA_DEVICE, dtype=dtype)

    def special_values() -> None:
        data[:5] = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
            device=data.device,
            dtype=dtype,
        )

    special_values()
    # Independent integer Philox oracle, shared by all layouts and packet policies.
    randoms = {
        seed: torch.tensor(
            [_oracle(seed, i) for i in range(data.numel())], device=CUDA_DEVICE
        )
        for seed in (123, -(2**63))
    }

    def expected(seed: int, probability: float) -> torch.Tensor:
        return torch.where(
            randoms[seed] > probability,
            data.float() * (1 / (1 - probability)),
            0.0,
        ).to(dtype)

    for width, layout, packet in (
        (4, "blocked", False),
        (8, "blocked", True),
        (8, "strided", True),
    ):
        config = _config(width, layout)
        config.config["cute_rng_packet"] = packet
        bound = kernel._bind_isolated((0.25, data, 123))
        assert bound.settings.cute_rng_stream == ("auto" if policy is None else policy)
        bound.set_config(config)
        for seed in randoms:
            for probability in (
                0.0,
                0.25,
                0.6,
                float("nan"),
                float("inf"),
                -float("inf"),
            ):
                _assert_exact(
                    bound(probability, data, seed), expected(seed, probability)
                )
        with pytest.raises(ZeroDivisionError):
            bound(1.0, data, 123)
        first = bound(0.25, data, 123)
        second = bound(0.25, data, 123)
        assert first.data_ptr() != second.data_ptr()
        first.fill_(float("nan"))
        _assert_exact(second, expected(123, 0.25))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                bound(0.25, data, 123)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = bound(0.25, data, 123)
        for iteration in range(3):
            data.fill_(iteration + 1.0)
            special_values()
            result.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            _assert_exact(result, expected(123, 0.25))


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _twice_assembly(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.numel()):
        value = hl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $1;",
            "=f,f",
            [x[tile]],
            dtype=torch.float32,
            is_pure=True,
            pack=1,
        )
        out[tile] = value
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize("layout", ("blocked", "strided"))
def test_generic_packet_assembly_and_fallback(width: int, layout: str) -> None:
    config = _config(width, layout)
    for mode in ("aligned", "offset", "strided", "alias"):
        _generic_case(config, mode)


def _generic_case(config: helion.Config, mode: str) -> None:
    length = 4093
    storage = torch.linspace(-2, 2, 2 * length + 2, device=CUDA_DEVICE)
    data = (
        storage[1 : length + 1]
        if mode == "offset"
        else storage[::2][:length]
        if mode == "strided"
        else storage[:length]
    )
    output = data if mode == "alias" else torch.empty(length, device=CUDA_DEVICE)
    bound = _twice_assembly._bind_isolated((data, output))
    source = bound.to_code(config)
    assert ("_helion_affine_load" in source) is (mode == "aligned")
    bound.set_config(config)
    expected = data * 2
    _assert_exact(bound(data, output), expected)
    # Refresh before each call, including aliases, so the exact reference
    # always observes the input before this invocation writes its result.
    for iteration in range(2):
        data.fill_(float(iteration) - 0.75)
        expected = data * 2
        _assert_exact(bound(data, output), expected)
    _mutating_graph(
        lambda: bound(data, output),
        lambda iteration: data.fill_(float(iteration) + 0.5),
        lambda: data * 2,
    )


def _arbitrary_offset_uniform(offsets: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.empty_like(offsets, dtype=torch.float32)
    for tile in hl.tile(offsets.numel()):
        out[tile] = hl.rand([], seed=seed, offsets=offsets[tile])
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width,layout", ((4, "blocked"), (8, "strided")))
def test_default_philox4_arbitrary_offsets(width: int, layout: str) -> None:
    # Include every counter word, noncontiguous offsets, negative offsets,
    # signed boundaries and high counter bits. The input has a vector tail.
    values = [3 * index + 1 for index in range(257)]
    values[:13] = [
        -(2**63),
        -(2**32) - 1,
        -9,
        -1,
        0,
        1,
        2,
        3,
        4,
        2**31 - 1,
        2**32 + 3,
        2**40 + 1,
        2**63 - 1,
    ]
    patterns = [values, list(reversed(values))]
    offsets = torch.tensor(values, device=CUDA_DEVICE, dtype=torch.int64)
    kernel = helion.kernel(
        _arbitrary_offset_uniform,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((offsets, -1))
    assert bound.settings.cute_rng_stream == "auto"
    bound.set_config(_config(width, layout))
    for pattern in patterns:
        offsets.copy_(torch.tensor(pattern, device=CUDA_DEVICE, dtype=torch.int64))
        for seed in (-1, 2**32 + 123):
            expected = torch.tensor(
                [_oracle(seed, offset) for offset in pattern], device=CUDA_DEVICE
            )
            first = bound(offsets, seed)
            second = bound(offsets, seed)
            _assert_exact(first, expected)
            _assert_exact(second, expected)
            assert first.data_ptr() != second.data_ptr()
            first.fill_(float("nan"))
            _assert_exact(second, expected)

    current = [patterns[-1]]

    def mutate(iteration: int) -> None:
        current[0] = patterns[iteration % len(patterns)]
        offsets.copy_(torch.tensor(current[0], device=CUDA_DEVICE, dtype=torch.int64))

    _mutating_graph(
        lambda: bound(offsets, -1),
        mutate,
        lambda: torch.tensor(
            [_oracle(-1, offset) for offset in current[0]], device=CUDA_DEVICE
        ),
    )
