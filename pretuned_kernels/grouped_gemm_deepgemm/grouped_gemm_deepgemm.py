"""Pretuned B200 BF16 grouped GEMM versus pinned DeepGEMM.

This dashboard kernel uses the eight reviewed ``(groups, expected M/group, N,
K)`` shapes and their deterministic seed-0 stream of actual per-group M sizes.
Expected M/group is benchmark-generation data, not a kernel argument.
Both implementations consume the
same logical A and grouped B values, with A repacked to each implementation's
required physical alignment and logically equivalent group metadata. They
write separate outputs, pass the same FP32-oracle check, and are timed only by
replaying pre-captured CUDA graphs with L2 cleared before every replay; packing
and compilation are excluded.
DeepGEMM uses ``ensure_zero_padding=False``, so its undefined aligned padding
is excluded from correctness; Helion is additionally required to zero its own
aligned padding. Both implementations validate the same logical rows.

Set ``HELION_DEEPGEMM_ROOT`` to a clean checkout of DeepGEMM commit
``559d79fb6994a58b8a15b4b93bf13ccc16edf247`` with its extension built in
place. The compact support module verifies the checkout and dependency
commits, package version, module origin, native ABI, and effective
contiguous-layout alignment, and records the native artifact hash. Point it at
a freshly built checkout when producing benchmark evidence.
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING
from typing import cast

from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles as _REVIEWED
from pretuned_kernels.grouped_gemm_deepgemm import reviewed_runtime as _RUNTIME
import torch
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.utils.weak import WeakIdKeyDictionary

import helion
from helion._compiler.cute.grouped_worklist import (
    tcgen05_grouped_worklist_compatible_source_m_tiles,
)
from helion._compiler.cute.grouped_worklist import (
    tcgen05_grouped_worklist_source_m_tiles_by_preference,
)
import helion.language as hl
from helion.runtime.cute.launcher import _tcgen05_grouped_tensor_mutation_key

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


_WORKLIST_FACTS_CACHE: WeakIdKeyDictionary = WeakIdKeyDictionary()
_WORKLIST_FACTS_CACHE_LOCK = Lock()


def _worklist_rows(
    worklist: torch.Tensor,
    mutation_key: tuple[object, ...],
) -> tuple[tuple[int, ...], ...]:
    if mutation_key[0] == "values":
        flat_values = cast("tuple[int, ...]", mutation_key[1])
    else:
        flat_values = tuple(
            int(value) for value in worklist.detach().reshape(-1).cpu().tolist()
        )
    return tuple(
        flat_values[offset : offset + 4] for offset in range(0, len(flat_values), 4)
    )


def _worklist_dispatch_facts(
    worklist: torch.Tensor,
    *,
    groups: int,
    packed_m: int,
) -> tuple[int, str]:
    """Cache the source tile and exact normalized rows from a valid worklist."""
    if isinstance(worklist, FakeTensor):
        concrete_worklist = worklist.constant
        if not isinstance(concrete_worklist, torch.Tensor):
            raise ValueError(
                "reviewed AOT dispatch requires concrete packed-worklist values"
            )
        worklist = concrete_worklist
    with unset_fake_temporarily():
        mutation_key = _tcgen05_grouped_tensor_mutation_key(worklist)
        with _WORKLIST_FACTS_CACHE_LOCK:
            cached = _WORKLIST_FACTS_CACHE.get(worklist)
            if cached is not None and cached[:3] == (mutation_key, groups, packed_m):
                return cached[3], cached[4]
            rows = _worklist_rows(worklist, mutation_key)
            compatible = tcgen05_grouped_worklist_compatible_source_m_tiles(
                rows,
                group_count=groups,
                packed_m=packed_m,
            )
            if not compatible:
                raise ValueError(
                    "worklist does not describe a valid supported packed source-M "
                    "layout"
                )
            source_m_tile = tcgen05_grouped_worklist_source_m_tiles_by_preference(
                compatible
            )[0]
            normalized_worklist = _REVIEWED.worklist_signature(rows)
            _WORKLIST_FACTS_CACHE[worklist] = (
                mutation_key,
                groups,
                packed_m,
                source_m_tile,
                normalized_worklist,
            )
    return source_m_tile, normalized_worklist


def _selected_key(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
) -> tuple[int, int, int, str, str, str, str, int, int, str]:
    """Select by logical shape, physical layout, and validated work envelope."""
    if b_grouped.ndim != 3:
        raise ValueError("b_grouped must have shape [groups, n, k]")
    if worklist.ndim != 2 or int(worklist.size(1)) != 4:
        raise ValueError("worklist must have shape [rows, 4]")
    if a_packed.ndim != 2 or int(a_packed.size(0)) <= 0:
        raise ValueError("a_packed must have a positive [packed_m, k] shape")
    groups, n, k = (int(value) for value in b_grouped.shape)
    b_strides = tuple(int(value) for value in b_grouped.stride())
    if b_strides[2] == 1:
        b_major = "k"
    elif b_strides[1] == 1:
        b_major = "n"
    else:
        raise ValueError(
            "b_grouped must use contiguous K-major or N-major grouped storage"
        )
    packed_m = int(a_packed.size(0))
    source_m_tile, normalized_worklist = _worklist_dispatch_facts(
        worklist,
        groups=groups,
        packed_m=packed_m,
    )
    a_layout = _REVIEWED.tensor_layout_signature(
        tuple(int(value) for value in a_packed.shape),
        tuple(int(value) for value in a_packed.stride()),
        int(a_packed.storage_offset()),
        str(a_packed.dtype),
    )
    b_layout = _REVIEWED.tensor_layout_signature(
        tuple(int(value) for value in b_grouped.shape),
        b_strides,
        int(b_grouped.storage_offset()),
        str(b_grouped.dtype),
    )
    worklist_layout = _REVIEWED.tensor_layout_signature(
        tuple(int(value) for value in worklist.shape),
        tuple(int(value) for value in worklist.stride()),
        int(worklist.storage_offset()),
        str(worklist.dtype),
    )
    return (
        groups,
        n,
        k,
        b_major,
        a_layout,
        b_layout,
        worklist_layout,
        source_m_tile,
        packed_m,
        normalized_worklist,
    )


def _reference(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
) -> torch.Tensor:
    """FP32 oracle for valid rows and zero-filled aligned padding."""
    output = torch.zeros(
        (a_packed.size(0), b_grouped.size(1)),
        device=a_packed.device,
        dtype=torch.float32,
    )
    rows = cast("list[list[int]]", worklist.cpu().tolist())
    for group, start, valid_m, _store_m in rows:
        output[start : start + valid_m] = (
            a_packed[start : start + valid_m].float() @ b_grouped[group].float().T
        )
    return output


def _check_aot_output(actual: object, expected: object) -> None:
    """Require a BF16 candidate within the selected path's FP32-oracle bound."""
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise AssertionError("grouped GEMM AOT validation requires tensor outputs")
    if actual.dtype is not torch.bfloat16 or expected.dtype is not torch.float32:
        raise AssertionError(
            "grouped GEMM AOT validation requires BF16 output and FP32 oracle"
        )
    difference = _RUNTIME.normalized_difference(actual, expected)
    if not difference <= 1e-5:
        raise AssertionError(
            f"grouped GEMM AOT FP32-oracle difference {difference} exceeds 1e-5"
        )


def grouped_gemm_deepgemm(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
) -> torch.Tensor:
    """BF16 ``A[sum(align(Mg)),K] @ B[G,N,K].T`` using an N,M worklist."""
    m_total_aligned, k = a_packed.shape
    _groups, n, k2 = b_grouped.shape
    assert k == k2, "K dimension mismatch between A and B"
    assert worklist.size(1) == 4

    block_m = hl.register_block_size(_REVIEWED.BLOCK_M)
    block_n = hl.register_block_size(_REVIEWED.BLOCK_N)
    block_k = hl.register_block_size(
        _REVIEWED.DEFAULT_BLOCK_K,
        _REVIEWED.LARGE_BLOCK_K,
    )
    out = torch.empty(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )

    for work_tile, tile_m, tile_n in hl.tile(
        [worklist.size(0), _REVIEWED.BLOCK_M, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = worklist[work_id, 0]
        global_m_start = worklist[work_id, 1]
        valid_m = worklist[work_id, 2]
        store_m = worklist[work_id, 3]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = local_m < store_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_block = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_block,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


_GROUPED_GEMM_DEEPGEMM_BODY = grouped_gemm_deepgemm


def create_grouped_gemm_deepgemm_kernel() -> Kernel[torch.Tensor]:
    """Create an independent deployed kernel using the reviewed AOT profiles."""
    return helion.aot_kernel(
        _GROUPED_GEMM_DEEPGEMM_BODY,
        backend="cute",
        key=_selected_key,
        static_shapes=True,
        standalone=False,
        autotune_baseline_fn=_reference,
        autotune_baseline_accuracy_check_fn=_check_aot_output,
    )


grouped_gemm_deepgemm = cast(
    "Kernel[torch.Tensor]",
    create_grouped_gemm_deepgemm_kernel(),
)


def use_cudagraph() -> bool:
    """The benchmark helper replays pre-captured implementation graphs."""
    return True


def main(verbose: bool = True) -> dict[str, object]:
    """Benchmark reviewed Helion profiles against DeepGEMM's public API."""
    from pretuned_kernels.grouped_gemm_deepgemm import _deepgemm_public_api

    return _deepgemm_public_api.main(
        create_grouped_gemm_deepgemm_kernel,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
