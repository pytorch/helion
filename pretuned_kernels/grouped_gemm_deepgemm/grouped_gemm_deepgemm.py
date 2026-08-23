"""Pretuned B200 BF16 grouped GEMM versus pinned DeepGEMM.

This dashboard kernel uses the eight reviewed ``(groups, expected M/group, N,
K)`` shapes and their deterministic seed-0 stream of actual per-group M sizes.
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

from typing import TYPE_CHECKING
from typing import cast

from benchmarks.cute import grouped_gemm_deepgemm_support as _SUPPORT
from pretuned_kernels.grouped_gemm_deepgemm import reviewed_profiles as _REVIEWED
import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


def _selected_key(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
    expected_m_per_group: int | None = None,
) -> tuple[int, int, int, int, str, int, int]:
    """Select by logical shape, physical layout, and packed work volume."""
    if expected_m_per_group is not None and (
        type(expected_m_per_group) is not int or expected_m_per_group <= 0
    ):
        raise ValueError("expected_m_per_group must be None or a positive integer")
    if b_grouped.ndim != 3:
        raise ValueError("b_grouped must have shape [groups, n, k]")
    if worklist.ndim != 2 or int(worklist.size(1)) != 4:
        raise ValueError("worklist must have shape [rows, 4]")
    if a_packed.ndim != 2 or int(a_packed.size(0)) <= 0:
        raise ValueError("a_packed must have a positive [packed_m, k] shape")
    groups, n, k = (int(value) for value in b_grouped.shape)
    if int(b_grouped.stride(2)) == 1:
        b_major = "k"
    elif int(b_grouped.stride(1)) == 1:
        b_major = "n"
    else:
        raise ValueError(
            "b_grouped must use contiguous K-major or N-major grouped storage"
        )
    profile = _REVIEWED.reviewed_worklist_profile(
        groups,
        expected_m_per_group,
        n,
        k,
    )
    # AOT key flattening omits None values. Use zero as an internal sentinel so
    # legacy calls retain a stable expected-M key field. The physical source
    # tile is derived from the reviewed profile rather than a user annotation.
    return (
        groups,
        expected_m_per_group or 0,
        n,
        k,
        b_major,
        profile.source_m_tile,
        int(a_packed.size(0)),
    )


def _reference(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
    expected_m_per_group: int | None = None,
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
    difference = _SUPPORT.normalized_difference(actual, expected)
    if not difference <= 1e-5:
        raise AssertionError(
            f"grouped GEMM AOT FP32-oracle difference {difference} exceeds 1e-5"
        )


def grouped_gemm_deepgemm(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
    expected_m_per_group: int | None = None,
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
