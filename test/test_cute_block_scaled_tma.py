"""CPU compilation checks for the packed scale descriptor's coordinate rank."""

from __future__ import annotations

import ast
import re

import pytest

from helion._compiler.cute.block_scaled_config import block_scaled_workspace_shape

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
cpasync = pytest.importorskip("cutlass.cute.nvgpu.cpasync")
tcgen05 = pytest.importorskip("cutlass.cute.nvgpu.tcgen05")
runtime = pytest.importorskip("cutlass.cute.runtime")
_COPY_RANKS: list[tuple[int, int]] = []


def _record_copy_ranks(atom: cute.CopyAtom, coordinates: cute.Tensor) -> None:
    # The public CopyAtom has no accessor for the TMA global basis. Inspect its
    # MLIR type, whose outer shape modes become the descriptor's dimensions.
    match = re.search(r'tma_gbasis = <"([^:]+):', str(atom.type))
    assert match is not None
    shape = ast.literal_eval(match[1])
    descriptor_rank = len(shape) if isinstance(shape, tuple) else 1
    _COPY_RANKS.append((descriptor_rank, cute.rank(coordinates.iterator)))


@cute.jit
def _inspect_padded_scale_map(
    pointer: cute.Pointer,
    n: cutlass.Constexpr,
    k: cutlass.Constexpr,
    bn: cutlass.Constexpr,
    bk: cutlass.Constexpr,
) -> None:
    _mp, np, kp = block_scaled_workspace_shape(73, n, k, bn, bk)
    scale = cute.make_tensor(pointer, cute.make_layout((256, kp // 64, np // 128)))
    window = cute.make_layout((256, bk // 64, bn // 128))
    atom, coordinates = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        scale,
        window,
        window.shape,
    )
    _record_copy_ranks(atom, coordinates)


@pytest.mark.parametrize("n", [1, 128, 129, 257])
@pytest.mark.parametrize("k", [16, 80, 272])
@pytest.mark.parametrize("bn,bk", [(128, 64), (256, 64), (256, 128), (256, 256)])
def test_scale_descriptor_and_device_coordinate_ranks_agree(
    n: int, k: int, bn: int, bk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    # Only compile a host function, with a null symbolic pointer. No CUDA
    # allocation, device context, JIT engine or device code execution occurs.
    pointer = runtime.make_ptr(
        cutlass.Int16, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    _COPY_RANKS.clear()
    cute.compile(_inspect_padded_scale_map, pointer, n, k, bn, bk, no_jit_engine=True)
    assert len(_COPY_RANKS) == 1
    descriptor_rank, coordinate_rank = _COPY_RANKS[0]
    assert descriptor_rank == coordinate_rank
