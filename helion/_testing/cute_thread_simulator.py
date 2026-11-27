"""Execute scalar CuTe thread programs on CPU for tests."""

from __future__ import annotations

import ast
import itertools
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import torch

import helion

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


def execute_pointwise_thread_program(
    source: str,
    inputs: Sequence[object],
    *,
    int32: Callable[[int], object] | None = None,
    int64: Callable[[int], object] = int,
    expected_writes: int | None = None,
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    """Execute the actual scalar program for every launched CTA/thread on CPU."""
    writes: dict[tuple[int, int], int] = {}

    def scalar(tensor: torch.Tensor) -> object:
        # Keep narrow signed loads typed: CuTe uses their source width when
        # it zero-extends into a wider unsigned integer.
        if tensor.dtype in (torch.int8, torch.int16, torch.int32):
            return tensor.numpy()[()]
        return tensor.item()

    def unsigned(value: int, bits: int) -> int:
        if isinstance(value, np.signedinteger):
            bits = min(bits, value.dtype.itemsize * 8)
        return int(value) % (1 << bits)

    def device_range(start: int, *args: int) -> object:
        values = range(start, *args)
        if isinstance(start, np.integer):
            return map(type(start), values)
        return values

    class Pointer:
        def __init__(self, tensor: torch.Tensor, offset: int = 0) -> None:
            self.tensor, self.offset = tensor, int(offset)

        def __add__(self, offset: int) -> Pointer:
            return Pointer(self.tensor, self.offset + int(offset))

        def load(self) -> object:
            storage = self.tensor.as_strided(
                (self.tensor.untyped_storage().nbytes() // self.tensor.element_size(),),
                (1,),
                storage_offset=0,
            )
            offset = int(self.tensor.storage_offset()) + self.offset
            assert 0 <= offset < storage.numel()
            return scalar(storage[offset])

        def store(self, value: float | bool) -> None:
            storage = self.tensor.as_strided(
                (self.tensor.untyped_storage().nbytes() // self.tensor.element_size(),),
                (1,),
                storage_offset=0,
            )
            offset = int(self.tensor.storage_offset()) + self.offset
            assert 0 <= offset < storage.numel()
            key = (self.tensor.untyped_storage().data_ptr(), offset)
            writes[key] = writes.get(key, 0) + 1
            storage[offset] = value

    class TensorRef:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor
            self.iterator = Pointer(tensor)
            self.layout = SimpleNamespace(stride=tensor.stride())

        def __getitem__(self, key: int | tuple[int, ...]) -> object:
            return scalar(self.tensor[key])

    current: dict[str, tuple[int, ...]] = {"block": (0, 0, 0), "thread": (0, 0, 0)}
    launches: list[tuple[int, ...]] = []

    def launcher(
        function: Callable[..., None],
        grid: tuple[int, ...],
        *arguments: object,
        block: tuple[int, ...],
    ) -> None:
        launches.append(block)
        args = [
            TensorRef(arg) if isinstance(arg, torch.Tensor) else arg
            for arg in arguments
        ]
        for cta in itertools.product(*(range(size) for size in grid)):
            current["block"] = (*cta, *((0,) * (3 - len(cta))))
            for thread in itertools.product(*(range(size) for size in block)):
                current["thread"] = thread
                function(*args)

    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        body.append(node)
    namespace: dict[str, Any] = {
        "helion": helion,
        "torch": torch,
        "operator": operator,
        "range": device_range,
        "_cute_python_mod": operator.mod,
        "cutlass": SimpleNamespace(
            Int8=lambda v: np.asarray(v).astype(np.int8)[()],
            Int16=lambda v: np.asarray(v).astype(np.int16)[()],
            Int32=int32 or (lambda v: np.asarray(v).astype(np.int32)[()]),
            Int64=int64,
            Uint32=lambda v: np.uint32(unsigned(v, 32)),
            Uint64=lambda v: np.uint64(unsigned(v, 64)),
            Float32=float,
            Float64=float,
            Boolean=bool,
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: current["block"], thread_idx=lambda: current["thread"]
            )
        ),
        "_default_cute_launcher": launcher,
        "_next_power_of_2": lambda value: 1 << (value - 1).bit_length(),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "<pointwise-thread-model>",
            "exec",
        ),
        namespace,
    )
    wrapper = next(
        node.name for node in reversed(body) if isinstance(node, ast.FunctionDef)
    )
    result = namespace[wrapper](*inputs)
    assert len(writes) == (
        result.numel() if expected_writes is None else expected_writes
    )
    assert set(writes.values()) <= {1}, "duplicate or missing output ownership"
    return result, launches
