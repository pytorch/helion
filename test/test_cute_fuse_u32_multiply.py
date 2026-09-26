"""CPU correctness tests for native lowering of the two-limb multiply identity."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.fuse_u32_multiply import fuse_u32_multiply
from helion.language.inline_asm_ops import inline_asm_elementwise
from helion.language.random_ops import _mulhi_lo_u32
from helion.language.random_ops import _philox_uint32x4

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from torch.fx.node import Argument
    from torch.fx.node import Target

_MASK32 = (1 << 32) - 1


def _trace(fn: Callable[..., object], *args: object) -> torch.fx.GraphModule:
    # The portable RNG helper consults the backend only to choose how constants
    # are represented on Pallas. All tracing and execution here stay on CPU.
    env = SimpleNamespace(backend=SimpleNamespace(name="cute"))
    with patch.object(CompileEnvironment, "current", return_value=env):
        return make_fx(fn)(*args)


class _NativeMultiplyInterpreter(torch.fx.Interpreter):
    """Execute the emitted integer PTX subset with CPU tensors.

    This checks operand numbering, constant reconstruction, truncation, and
    output extension as well as the matched arithmetic. GPU integration tests
    still need to exercise the actual inline assembly compiler.
    """

    def call_function(
        self,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if target is not inline_asm_elementwise:
            return super().call_function(target, args, kwargs)
        asm, constraints, operands, dtype, is_pure, pack = args
        assert isinstance(asm, str)
        assert isinstance(operands, list)
        operands = cast("list[torch.Tensor]", operands)
        assert constraints == "=l" + ",l" * len(operands)
        assert dtype is torch.int64 and is_pure is True and pack == 1
        registers: dict[str, int | torch.Tensor] = {
            f"${index + 1}": operand for index, operand in enumerate(operands)
        }

        def read(name: str) -> int | torch.Tensor:
            return registers[name] if name in registers else int(name)

        for instruction in asm.strip("{}\n").split(";"):
            instruction = instruction.strip()
            if not instruction or instruction.startswith(".reg "):
                continue
            opcode, operands_text = instruction.split(" ", 1)
            destination, *sources = (part.strip() for part in operands_text.split(","))
            values = [read(source) for source in sources]
            if opcode in ("cvt.u32.u64", "cvt.u64.u32"):
                result = values[0] & _MASK32
            elif opcode in ("mul.lo.u32", "mul.hi.u32"):
                product = values[0] * values[1]
                result = (
                    product & _MASK32
                    if opcode == "mul.lo.u32"
                    else (product >> 32) & _MASK32
                )
            else:
                raise AssertionError(
                    f"unexpected native multiply instruction: {opcode}"
                )
            registers[destination] = result
        return registers["$0"]


def _values() -> torch.Tensor:
    edges = torch.tensor(
        [
            -(1 << 63),
            -(1 << 32) - 1,
            -(1 << 32),
            -(1 << 31),
            -1,
            0,
            1,
            (1 << 16) - 1,
            1 << 16,
            (1 << 16) + 1,
            (1 << 31) - 1,
            1 << 31,
            (1 << 32) - 2,
            (1 << 32) - 1,
            1 << 32,
            (1 << 32) + 1,
            (1 << 63) - 1,
        ],
        dtype=torch.int64,
    )
    generator = torch.Generator().manual_seed(32816)
    random = torch.randint(
        -(1 << 63), (1 << 63) - 1, (257,), generator=generator, dtype=torch.int64
    )
    return torch.cat((edges, random))


def _exact_product(
    a: int | torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = torch.broadcast_tensors(torch.as_tensor(a, dtype=torch.int64), b)
    # Python integers give an independent non-overflowing 64-bit product oracle.
    products = [
        (int(x) & _MASK32) * (int(y) & _MASK32)
        for x, y in zip(left.flatten(), right.flatten(), strict=True)
    ]
    high = torch.tensor([product >> 32 for product in products], dtype=torch.int64)
    low = torch.tensor([product & _MASK32 for product in products], dtype=torch.int64)
    return high.reshape(left.shape), low.reshape(left.shape)


@pytest.mark.parametrize(
    "multiplier",
    [0, 1, 65535, 65536, 3528531795, 3449720151, _MASK32, (1 << 32) + 17, -1],
)
def test_constant_multiplier_exact_words(multiplier: int) -> None:
    values = _values()
    graph_module = _trace(_mulhi_lo_u32, multiplier, values)
    expected = _exact_product(multiplier, values)
    torch.testing.assert_close(
        graph_module(multiplier, values), expected, rtol=0, atol=0
    )
    assert fuse_u32_multiply(graph_module.graph) == 2
    actual = _NativeMultiplyInterpreter(graph_module).run(multiplier, values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tensor_operands_broadcast_and_wrap_high_bits() -> None:
    values = _values()
    a, b = values[:31, None], values[None, :29]
    graph_module = _trace(_mulhi_lo_u32, a, b)
    assert fuse_u32_multiply(graph_module.graph) == 2
    actual = _NativeMultiplyInterpreter(graph_module).run(a, b)
    torch.testing.assert_close(actual, _exact_product(a, b), rtol=0, atol=0)


def test_commuted_operations_and_alternate_shift_targets() -> None:
    a, b = _values(), _values().flip(0)
    graph_module = _trace(_mulhi_lo_u32, a, b)
    for node in graph_module.graph.nodes:
        if node.target in (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.bitwise_or.Tensor,
        ):
            node.args = (node.args[1], node.args[0])
        elif node.target is torch.ops.aten.__rshift__.Scalar:
            node.target = torch.ops.aten.bitwise_right_shift.Tensor_Scalar
        elif node.target is torch.ops.aten.__lshift__.Scalar:
            node.target = torch.ops.aten.bitwise_left_shift.Tensor_Scalar
    graph_module.recompile()
    expected = graph_module(a, b)
    assert fuse_u32_multiply(graph_module.graph) == 2
    actual = _NativeMultiplyInterpreter(graph_module).run(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("word", [0, 1])
def test_single_live_word(word: int) -> None:
    def select_word(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _mulhi_lo_u32(a, b)[word]

    a, b = _values(), _values().flip(0)
    graph_module = _trace(select_word, a, b)
    graph_module.graph.eliminate_dead_code()
    assert fuse_u32_multiply(graph_module.graph) == 1
    actual = _NativeMultiplyInterpreter(graph_module).run(a, b)
    torch.testing.assert_close(actual, _exact_product(a, b)[word], rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 1, -1, -(1 << 63), (1 << 63) - 1])
def test_full_philox_rounds_preserve_every_output_bit(seed: int) -> None:
    seed_tensor = torch.tensor(seed, dtype=torch.int64)
    offsets = _values()
    graph_module = _trace(_philox_uint32x4, seed_tensor, offsets)
    expected = graph_module(seed_tensor, offsets)
    assert fuse_u32_multiply(graph_module.graph) == 40
    actual = _NativeMultiplyInterpreter(graph_module).run(seed_tensor, offsets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fuse_u32_multiply(graph_module.graph) == 0


def test_philox_known_zero_counter_vector() -> None:
    seed = torch.tensor(0, dtype=torch.int64)
    offset = torch.tensor([0], dtype=torch.int64)
    graph_module = _trace(_philox_uint32x4, seed, offset)
    fuse_u32_multiply(graph_module.graph)
    actual = _NativeMultiplyInterpreter(graph_module).run(seed, offset)
    assert [value.item() for value in actual] == [
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    ]


def test_live_intermediate_carry_is_preserved() -> None:
    a, b = _values(), _values().flip(0)
    graph_module = _trace(_mulhi_lo_u32, a, b)
    carry = next(
        node
        for node in graph_module.graph.nodes
        if node.target is torch.ops.aten.add.Tensor
    )
    output = next(node for node in graph_module.graph.nodes if node.op == "output")
    output.args = ((*cast("tuple[torch.fx.Node, ...]", output.args[0]), carry),)
    graph_module.recompile()
    expected = graph_module(a, b)
    assert fuse_u32_multiply(graph_module.graph) == 2
    actual = _NativeMultiplyInterpreter(graph_module).run(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_incorrect_high_product_does_not_block_valid_low_word() -> None:
    a, b = _values(), _values().flip(0)
    graph_module = _trace(_mulhi_lo_u32, a, b)
    products = [
        node
        for node in graph_module.graph.nodes
        if node.target is torch.ops.aten.mul.Tensor
    ]
    products[-1].args = (products[-1].args[0], products[0].args[1])
    graph_module.recompile()
    expected = graph_module(a, b)
    assert fuse_u32_multiply(graph_module.graph) == 1
    actual = _NativeMultiplyInterpreter(graph_module).run(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["mask", "shift", "dtype", "add_alpha"])
def test_unproved_arithmetic_is_not_rewritten(mutation: str) -> None:
    a, b = _values(), _values().flip(0)
    graph_module = _trace(_mulhi_lo_u32, a, b)
    nodes = list(graph_module.graph.nodes)
    if mutation == "mask":
        target = next(
            node for node in nodes if node.target is torch.ops.aten.bitwise_and.Scalar
        )
        target.args = (target.args[0], 65534)
    elif mutation == "shift":
        target = next(
            node for node in nodes if node.target is torch.ops.aten.__rshift__.Scalar
        )
        target.args = (target.args[0], 15)
    elif mutation == "dtype":
        target = next(
            node for node in nodes if node.target is torch.ops.aten.mul.Tensor
        )
        target.meta["val"] = target.meta["val"].to(torch.int32)
    else:
        target = next(
            node for node in nodes if node.target is torch.ops.aten.add.Tensor
        )
        target.kwargs = {"alpha": 2}
    before = str(graph_module.graph)
    assert fuse_u32_multiply(graph_module.graph) == 0
    assert str(graph_module.graph) == before
