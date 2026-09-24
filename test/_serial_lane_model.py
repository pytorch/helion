"""CPU interpretation of the actual generated device AST, never compilation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from helion.runtime.cute import serial_lane_guard


@dataclass
class DType:
    dtype: torch.dtype

    def __call__(self, value: object) -> torch.Tensor:
        return torch.as_tensor(value).to(self.dtype)


class Registers:
    def __init__(self, shape: tuple[int, ...], dtype: DType) -> None:
        assert len(shape) == 1 and shape[0] in (2, 4)
        self.size = shape[0]
        self.dtype = dtype
        self.values: dict[int, torch.Tensor] = {}

    def __getitem__(self, lane: int) -> torch.Tensor:
        return self.values[lane]

    def __setitem__(self, lane: int, value: torch.Tensor) -> None:
        assert 0 <= lane < self.size
        self.values[lane] = self.dtype(value)


class Memory:
    def __init__(self, tensor: torch.Tensor, address: int) -> None:
        self.layout = SimpleNamespace(stride=tensor.stride())
        self.address = address
        self.dtype = tensor.dtype
        self.itemsize = tensor.element_size()
        self.values = {}
        for index in product(*(range(size) for size in tensor.shape)):
            offset = sum(i * s for i, s in zip(index, tensor.stride(), strict=True))
            assert offset not in self.values
            self.values[offset] = tensor[index].clone()
        self.writes: dict[int, int] = {}
        self.reads = 0
        self.iterator = Pointer(self, 0)


@dataclass
class Pointer:
    owner: Memory
    offset: int

    def __add__(self, value: int) -> Pointer:
        return Pointer(self.owner, self.offset + int(value))

    def toint(self) -> int:
        return self.owner.address + self.offset * self.owner.itemsize

    def align(self, alignment: int) -> Pointer:
        assert self.toint() % alignment == 0
        return self

    def load(self) -> torch.Tensor:
        assert self.offset in self.owner.values
        self.owner.reads += 1
        return self.owner.values[self.offset]

    def store(self, value: torch.Tensor) -> None:
        assert self.offset in self.owner.values
        self.owner.values[self.offset] = value.to(self.owner.dtype)
        self.owner.writes[self.offset] = self.owner.writes.get(self.offset, 0) + 1


@dataclass
class Vector:
    pointer: Pointer
    size: int


def run(
    source: str, args: tuple[torch.Tensor, ...], *, reordered: bool
) -> tuple[torch.Tensor, dict[str, int]]:
    from .test_cute_serial_lane import _module

    calls = []
    module = _module(source, calls)
    with patch.object(serial_lane_guard, "_require_cuda"):
        module._recurrence(*args)
    selected, grid, device_args, kwargs = calls[0]
    name = selected.__name__.removesuffix("_serial_lane")
    fn = getattr(module, name + ("_serial_lane" if reordered else ""))
    memories = {id(t): Memory(t, (i + 1) * 0x10000000) for i, t in enumerate(args)}
    before = {
        id(t): {k: v.clone() for k, v in memories[id(t)].values.items()}
        for t in args[:2]
    }
    counts = {"vector_load": 0, "vector_store": 0}

    def copy(
        atom: tuple[torch.dtype, int], src: Vector | Registers, dst: Vector | Registers
    ) -> None:
        if isinstance(src, Vector):
            assert isinstance(dst, Registers) and atom == (torch.float32, src.size * 32)
            counts["vector_load"] += 1
            for lane in range(src.size):
                dst[lane] = (src.pointer + lane).load()
        else:
            assert isinstance(dst, Vector)
            assert atom == (
                dst.pointer.owner.dtype,
                dst.size * dst.pointer.owner.itemsize * 8,
            )
            counts["vector_store"] += 1
            for lane in range(dst.size):
                (dst.pointer + lane).store(src[lane])

    def make_tensor(pointer: Pointer, layout: tuple) -> Vector:
        shape, stride = layout
        assert len(shape) == 1 and stride == (1,)
        return Vector(pointer, shape[0])

    current: dict[str, tuple[int, ...]] = {"block": (0, 0, 0), "thread": (0, 0, 0)}
    module.__dict__.update(
        cute=SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: current["block"], thread_idx=lambda: current["thread"]
            ),
            math=SimpleNamespace(exp2=torch.exp2),
            make_rmem_tensor=Registers,
            make_copy_atom=lambda op, dtype, num_bits_per_copy: (
                dtype.dtype,
                num_bits_per_copy,
            ),
            nvgpu=SimpleNamespace(CopyUniversalOp=object),
            make_layout=lambda shape, stride: (shape, stride),
            make_tensor=make_tensor,
            copy=copy,
        ),
        cutlass=SimpleNamespace(
            Int32=int,
            Int64=int,
            Float32=DType(torch.float32),
            Float16=DType(torch.float16),
            BFloat16=DType(torch.bfloat16),
            range_constexpr=range,
        ),
    )
    wrapped: list[Any] = [memories[id(v)] for v in device_args]
    for block in product(*(range(size) for size in grid)):
        current["block"] = block
        for thread in product(*(range(size) for size in kwargs["block"])):
            current["thread"] = thread
            fn(*wrapped)
    for tensor in args[:2]:
        memory = memories[id(tensor)]
        assert not memory.writes
        assert all(
            torch.equal(v, before[id(tensor)][key]) for key, v in memory.values.items()
        )
    output = memories[id(args[2])]
    assert set(output.writes) == set(output.values)
    assert set(output.writes.values()) == {1}
    result = torch.empty_like(args[2])
    for index in product(*(range(size) for size in result.shape)):
        key = sum(i * s for i, s in zip(index, args[2].stride(), strict=True))
        result[index] = output.values[key]
    return result, {
        **counts,
        "state_reads": memories[id(args[0])].reads,
        "coefficient_reads": memories[id(args[1])].reads,
    }
