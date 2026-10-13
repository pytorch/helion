"""Native shared destinations for the original guarded root operand copies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_root_native import RootNativeStage


@lru_cache(maxsize=64)
def _native_layout(
    shape: tuple[int, int], dtype: torch.dtype, threads: int
) -> tuple[int, tuple[int, ...], bool]:
    """Evaluate the actual native swizzle and the original 128-bit copy TV map."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    kind = cutlass.BFloat16 if dtype is torch.bfloat16 else cutlass.Float16
    atom_bytes = min(128, (shape[1] * 2) & -(shape[1] * 2))
    atoms = {
        32: tcgen05.SmemLayoutAtomKind.K_SW32,
        64: tcgen05.SmemLayoutAtomKind.K_SW64,
        128: tcgen05.SmemLayoutAtomKind.K_SW128,
    }
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            layout = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(atoms[atom_bytes], kind),
                shape,
                order=(0, 1),
            )
            cells = tuple(
                int(layout((row, column)))
                for row in range(shape[0])
                for column in range(shape[1])
            )
            capacity = int(cute.cosize(layout))
            if len(set(cells)) != math.prod(shape) or not all(
                0 <= x < capacity for x in cells
            ):
                raise ValueError("native shared destination is not injective")
            columns = min(threads, shape[1] // 8)
            complete = bool(
                columns
                and not columns & (columns - 1)
                and shape[0] % (threads // columns) == 0
            )
            if complete:
                copy = cute.make_tiled_copy_tv(
                    cute.make_copy_atom(
                        cute.nvgpu.cpasync.CopyG2SOp(), kind, num_bits_per_copy=128
                    ),
                    cute.make_layout(
                        (threads // columns, columns), stride=(columns, 1)
                    ),
                    cute.make_layout((1, 8)),
                )
                identity = cute.make_identity_tensor(shape)
                seen = []
                for thread in range(threads):
                    fragment = copy.get_slice(thread).partition_D(identity)
                    source = copy.get_slice(thread).partition_S(identity)
                    count = int(cute.size(fragment))
                    if count % 8 or int(cute.size(source)) != count:
                        raise ValueError("native copy packet shape changed")
                    for first in range(0, count, 8):
                        packet = []
                        for slot in range(first, first + 8):
                            point = tuple(int(x) for x in fragment[slot])
                            if point != tuple(int(x) for x in source[slot]):
                                raise ValueError(
                                    "native copy changed source coordinates"
                                )
                            address = int(layout(point))
                            packet.append(address)
                            seen.append(point)
                        if packet[0] % 8 or packet != list(
                            range(packet[0], packet[0] + 8)
                        ):
                            raise ValueError(
                                "native copy packet is not contiguous and aligned"
                            )
                if len(seen) != math.prod(shape) or len(set(seen)) != len(seen):
                    raise ValueError("native copy does not cover the exact destination")
        if not module.operation.verify():
            raise ValueError("invalid native destination layout proof")
    return capacity, cells, complete


@dataclass(frozen=True)
class NativeDestination:
    codegen: GenerateAST
    plan: ChainedMatmulPlan
    stage: RootNativeStage
    role: str
    reference_stride: tuple[int, int]
    layout: tuple[int, tuple[int, ...], bool]

    @property
    def shape(self) -> tuple[int, int]:
        m, n, k = self.stage.geometry.physical
        return (m if self.role == "a" else n, k)

    @property
    def name(self) -> str:
        return f"chain_{self.stage.index}_{self.role}"

    def check(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        operand: Node,
        prefix: str,
        role: str,
        shape: tuple[int, int],
        stride: tuple[int, int],
        inner: int,
        dtype: str,
    ) -> None:
        from ..compile_environment import CompileEnvironment
        from .chained_matmul import _UnsupportedChain

        native = plan.root_native
        if native is None:
            raise _UnsupportedChain("native destination has no root schedule")
        native.check(plan)
        if (
            cg is not self.codegen
            or plan is not self.plan
            or cg.device_function.cute_state.chained_matmul_plan is not plan
            or not any(item is self.stage for item in native.stages)
            or role != self.role
            or role not in ("a", "b")
            or operand is not self.stage.operands[0 if role == "a" else 1]
            or prefix != f"chain_{self.stage.index}"
            or shape != self.shape
            or stride != self.reference_stride
            or inner != 1
            or dtype != CompileEnvironment.current().backend.dtype_str(self.stage.dtype)
            or self.layout != _native_layout(shape, self.stage.dtype, plan.threads)
            or self.layout[0]
            > native.original_workspace_elements[0 if role == "a" else 1]
        ):
            raise _UnsupportedChain("native shared destination binding changed")

    def facts(self) -> object:
        return (
            id(self.codegen),
            id(self.plan),
            id(self.stage),
            self.role,
            self.reference_stride,
            self.shape,
            self.name,
            self.layout,
        )

    def declaration(self) -> list[str]:
        from ..compile_environment import CompileEnvironment
        from .chained_tcgen05 import _layout

        dtype = CompileEnvironment.current().backend.dtype_str(self.stage.dtype)
        return _layout(self.name, self.shape, 1, dtype)


def native_destinations(
    cg: GenerateAST, plan: ChainedMatmulPlan, stage: RootNativeStage, padding: int
) -> tuple[NativeDestination, NativeDestination]:
    result = []
    for role in ("a", "b"):
        m, n, k = stage.geometry.physical
        shape = (m if role == "a" else n, k)
        result.append(
            NativeDestination(
                cg,
                plan,
                stage,
                role,
                (k + padding, 1),
                _native_layout(shape, stage.dtype, plan.threads),
            )
        )
    return result[0], result[1]
