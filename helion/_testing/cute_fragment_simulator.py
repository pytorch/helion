"""CPU replay of generated CuTe fragments with independent element loops."""

from __future__ import annotations

import ast
import math
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping


def simulate_independent_fragment(
    source: str,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
    blocks: int,
    *,
    int32: Callable[..., object] = int,
    int64: Callable[..., object] = int,
    inline_asm: Callable[..., object] | None = None,
    gather_registers: Callable[..., object] | None = None,
) -> dict[str, int]:
    """Execute generated fragment phases serially, preserving every coordinate.

    Each fragment element loop visits all coordinates instead of one thread's
    residue class. The barriers therefore delimit complete CPU phases. This
    validates generated indexing/math, not native CUDA synchronization.
    """
    packets = {"vector": 0, "scalar": 0}

    class Pointer:
        def __init__(self, tensor: torch.Tensor, offset: int = 0) -> None:
            self.tensor = tensor
            self.offset = offset

        def __add__(self, offset: int) -> Pointer:
            return Pointer(self.tensor, self.offset + int(offset))

        def storage(self) -> torch.Tensor:
            return self.tensor.as_strided(
                (self.tensor.untyped_storage().nbytes() // self.tensor.element_size(),),
                (1,),
                storage_offset=0,
            )

        def align(self, alignment: int) -> Pointer:
            assert (
                self.tensor.data_ptr() + self.offset * self.tensor.element_size()
            ) % alignment == 0
            return self

        def load(self) -> object:
            packets["scalar"] += 1
            storage = self.storage()
            offset = self.tensor.storage_offset() + self.offset
            assert 0 <= offset < storage.numel()
            return storage[offset].item()

        def store(self, value: float | bool | np.generic) -> None:
            storage = self.storage()
            offset = self.tensor.storage_offset() + self.offset
            assert 0 <= offset < storage.numel()
            storage[offset] = value.item() if isinstance(value, np.generic) else value

    class Shared:
        def __init__(self, dtype: type, shape: tuple[int, ...]) -> None:
            self.values = np.empty(math.prod(shape), dtype=dtype)
            self.written = np.zeros(math.prod(shape), dtype=bool)

        def __getitem__(self, index: int) -> np.generic:
            assert self.written[int(index)], f"read before shared write: {index}"
            return self.values[int(index)]

        def __setitem__(self, index: int, value: object) -> None:
            self.values[int(index)] = value
            self.written[int(index)] = True

    class Allocator:
        def allocate_tensor(
            self, dtype: type, layout: tuple[int, ...], byte_alignment: int
        ) -> Shared:
            return Shared(dtype, layout)

    def copy_packet(atom: int, source: SimpleNamespace, destination: Shared) -> None:
        assert atom == 128
        assert destination.values.size == 4
        assert (
            source.iterator.tensor.data_ptr() + source.iterator.offset * 4
        ) % 16 == 0
        packets["vector"] += 1
        for lane in range(4):
            destination[lane] = (source.iterator + lane).load()

    class SerialElements(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.Call:
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast":
                # NumPy scalars expose view(), unlike the generated scalar's bitcast().
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_bitcast", ctx=ast.Load()),
                        args=[node.func.value, *node.args],
                        keywords=node.keywords,
                    ),
                    node,
                )
            return node

        def visit_For(self, node: ast.For) -> ast.For:
            self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id.startswith(
                ("fragment_index", "fragment_packet_index")
            ):
                assert isinstance(node.iter, ast.Call) and len(node.iter.args) == 3
                node.iter.args[0] = ast.Constant(0)
                node.iter.args[2] = ast.Constant(1)
            return node

    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    function.decorator_list = []
    SerialElements().visit(function)
    constants = [node for node in tree.body if isinstance(node, ast.Assign)]
    for block in range(blocks):
        environment: dict[str, object] = {
            "operator": operator,
            "_cute_python_mod": operator.mod,
            "_bitcast": lambda value, dtype: np.asarray(value).view(dtype)[()],
            "cutlass": SimpleNamespace(
                Float16=np.float16,
                Float32=np.float32,
                Float64=np.float64,
                Int16=np.int16,
                Int32=int32,
                Int64=int64,
                Uint32=int,
                Uint64=int,
                Boolean=bool,
                utils=SimpleNamespace(SmemAllocator=Allocator),
            ),
            "cute": SimpleNamespace(
                math=SimpleNamespace(
                    min=lambda a, b, propagate_nan=False: (
                        np.minimum(a, b) if propagate_nan else np.fmin(a, b)
                    ),
                    max=lambda a, b, propagate_nan=False: (
                        np.maximum(a, b) if propagate_nan else np.fmax(a, b)
                    ),
                    tanh=np.tanh,
                ),
                make_layout=lambda shape: shape,
                make_rmem_tensor=lambda layout, dtype: Shared(dtype, layout),
                make_tensor=lambda iterator, layout: SimpleNamespace(
                    iterator=iterator, layout=layout
                ),
                make_copy_atom=lambda op, dtype, num_bits_per_copy: num_bits_per_copy,
                nvgpu=SimpleNamespace(CopyUniversalOp=lambda: None),
                copy=copy_packet,
                arch=SimpleNamespace(
                    thread_idx=lambda: (0, 0, 0),
                    block_idx=lambda block=block: (block, 0, 0),
                    sync_threads=lambda: None,
                    fmax=lambda a, b, nan=False: (
                        np.maximum(a, b) if nan else np.fmax(a, b)
                    ),
                    fmin=lambda a, b, nan=False: (
                        np.minimum(a, b) if nan else np.fmin(a, b)
                    ),
                ),
            ),
        }
        if gather_registers is not None:
            environment["_cute_gather_registers"] = gather_registers
        if inline_asm is not None:
            environment["_cute_inline_asm_elementwise"] = inline_asm
        for name, tensor in {**inputs, **outputs}.items():
            if isinstance(tensor, torch.Tensor):
                environment[name] = SimpleNamespace(
                    iterator=Pointer(tensor),
                    layout=SimpleNamespace(stride=tensor.stride()),
                )
            else:
                environment[name] = tensor
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[*constants, function], type_ignores=[])
                ),
                "<generated-fragment-cpu>",
                "exec",
            ),
            environment,
        )
        kernel = environment[function.name]
        assert callable(kernel)
        kernel(*(environment[arg.arg] for arg in function.args.args))
    return packets
