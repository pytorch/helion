"""Generated direct-pointer Blackwell grouped GEMM benchmark harness."""

# ruff: noqa: E402

from __future__ import annotations

from collections import OrderedDict
from itertools import starmap
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import helion
import helion.language as hl

BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES = (
    (128, 128, 128, 1),
    (512, 128, 128, 1),
    (128, 256, 128, 1),
)
BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES = (
    (8192, 1280, 32, 1),
    (16, 384, 1536, 1),
    (640, 1280, 16, 1),
    (640, 160, 16, 1),
)
_BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE_MAX = 16
# Reserved-SM caps are scoped to the generated direct Blackwell grouped-GEMM
# path. They are B200-measured tail-wave launch caps; the selector returns 0
# outside those partial-wave bands.
_BLACKWELL_GROUPED_GENERATED_MTAIL_RESERVED_SMS = 3
_BLACKWELL_GROUPED_GENERATED_LARGE_MTAIL_RESERVED_SMS = 2


class _BlackwellGeneratedTensorGuard(NamedTuple):
    tensor_id: int
    data_ptr: int
    device_type: str
    device_index: int | None
    dtype: torch.dtype
    ndim: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


class _BlackwellGeneratedStableLaunchGuard(NamedTuple):
    mma_tiler_mn: tuple[int, ...]
    cluster_shape_mn: tuple[int, ...]
    use_2cta_instrs: bool
    a_guards: tuple[_BlackwellGeneratedTensorGuard, ...]
    b_guards: tuple[_BlackwellGeneratedTensorGuard, ...]
    out_guards: tuple[_BlackwellGeneratedTensorGuard, ...]


class _BlackwellGeneratedStableLaunch(NamedTuple):
    cache_key: tuple[object, ...]
    guard: _BlackwellGeneratedStableLaunchGuard
    kernel_args: tuple[torch.Tensor, ...]
    bound: Callable[..., object]
    fast_call: Callable[[], object] | None
    cuda_graph: torch.cuda.CUDAGraph | None
    runtime_cache_entry: Any | None


class _BlackwellGeneratedStableLaunchCache(
    OrderedDict[tuple[object, ...], _BlackwellGeneratedStableLaunch]
):
    def _drop_last_if(
        self,
        launch: object,
    ) -> None:
        global _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH

        if _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH is launch:
            _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH = None

    def __setitem__(
        self,
        key: tuple[object, ...],
        value: _BlackwellGeneratedStableLaunch,
    ) -> None:
        self._drop_last_if(self.get(key))
        super().__setitem__(key, value)

    def __delitem__(self, key: tuple[object, ...]) -> None:
        launch = self[key]
        super().__delitem__(key)
        self._drop_last_if(launch)

    def clear(self) -> None:
        global _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH

        super().clear()
        _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH = None

    def discard(
        self,
        key: tuple[object, ...],
    ) -> None:
        try:
            launch = self[key]
        except KeyError:
            return
        super().__delitem__(key)
        self._drop_last_if(launch)

    def popitem(
        self,
        last: bool = True,
    ) -> tuple[tuple[object, ...], _BlackwellGeneratedStableLaunch]:
        key, launch = super().popitem(last=last)
        self._drop_last_if(launch)
        return key, launch


_BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE: _BlackwellGeneratedStableLaunchCache = (
    _BlackwellGeneratedStableLaunchCache()
)
_BLACKWELL_GENERATED_LAST_STABLE_LAUNCH: _BlackwellGeneratedStableLaunch | None = None


# %%
class _BlackwellGeneratedGemmProblem(NamedTuple):
    m: int
    n: int
    k: int


class _BlackwellGeneratedGemmMetadata(NamedTuple):
    device: torch.device
    problems: tuple[_BlackwellGeneratedGemmProblem, ...]
    a_mode0_major: bool
    b_mode0_major: bool
    c_mode0_major: bool


def _blackwell_grouped_gemm_elements_per_16b(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 8
    if dtype == torch.float32:
        return 4
    raise ValueError("dtype must be torch.float16, torch.bfloat16, or torch.float32")


def make_blackwell_grouped_gemm_nt_args(
    problem_sizes_mnkl: Sequence[
        tuple[int, int, int, int]
    ] = BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
    *,
    dtype: torch.dtype = torch.float16,
    out_dtype: torch.dtype | None = None,
    device: torch.device | str = "cuda",
) -> tuple[
    tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    tuple[torch.Tensor, ...],
]:
    """
    Build inputs for the generated Blackwell grouped GEMM direct benchmark.

    ``problem_sizes_mnkl`` follows the NVIDIA example convention: one
    ``(M, N, K, L)`` tuple per group, with ``L == 1``. This helper creates
    row-major contiguous tensors, so each ``K`` must be a multiple of the
    A/B dtype's 16-byte element count and each ``N`` must be a multiple of the
    output dtype's 16-byte element count. The default device is CUDA to match
    the benchmark target and the CUDA-oriented Helion example default without
    importing testing helpers into benchmark code; pass ``device`` explicitly
    for CPU-only validation.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("dtype must be torch.float16 or torch.bfloat16")
    output_dtype = dtype if out_dtype is None else out_dtype
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "out_dtype must be torch.float16, torch.bfloat16, or torch.float32"
        )
    if not problem_sizes_mnkl:
        raise ValueError("problem_sizes_mnkl must contain at least one group")

    ab_elements_per_16b = _blackwell_grouped_gemm_elements_per_16b(dtype)
    out_elements_per_16b = _blackwell_grouped_gemm_elements_per_16b(output_dtype)
    group_A: list[torch.Tensor] = []
    group_B: list[torch.Tensor] = []
    for group, (m, n, k, l_mode) in enumerate(problem_sizes_mnkl):
        if l_mode != 1:
            raise ValueError("Blackwell grouped GEMM benchmark requires L == 1")
        if m <= 0 or n <= 0 or k <= 0:
            raise ValueError(f"group {group} problem dimensions must be positive")
        if k % ab_elements_per_16b != 0:
            raise ValueError(
                f"group {group} K must be a multiple of {ab_elements_per_16b} "
                "elements for 16-byte contiguous-dimension alignment"
            )
        if n % out_elements_per_16b != 0:
            raise ValueError(
                f"group {group} N must be a multiple of {out_elements_per_16b} "
                "elements for 16-byte contiguous-dimension alignment"
            )
        group_A.append(torch.randn(m, k, device=device, dtype=dtype).contiguous())
        group_B.append(torch.randn(n, k, device=device, dtype=dtype).contiguous())

    expected = _reference_blackwell_grouped_gemm_nt(
        group_A,
        group_B,
        out_dtype=output_dtype,
    )
    return (tuple(group_A), tuple(group_B)), expected


def _reference_blackwell_grouped_gemm_nt(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    *,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, ...]:
    if len(group_A) != len(group_B):
        raise ValueError("group_A and group_B must have the same number of groups")
    if not group_A:
        raise ValueError("group_A must contain at least one group")
    output_dtype = group_A[0].dtype if out_dtype is None else out_dtype
    return tuple(
        (a.float() @ b.float().T).to(output_dtype)
        for a, b in zip(group_A, group_B, strict=True)
    )


def _blackwell_grouped_gemm_nt_generated_major_mode(
    tensor: torch.Tensor,
    *,
    mode0: str,
    mode1: str,
    tensor_name: str,
    group: int,
) -> str:
    if int(tensor.stride(1)) == 1:
        return mode1
    if int(tensor.stride(0)) == 1:
        return mode0
    raise ValueError(
        f"{tensor_name}[{group}] must have a contiguous {mode0} or {mode1} mode"
    )


def _blackwell_grouped_gemm_nt_generated_check_16b_alignment(
    tensor: torch.Tensor,
    *,
    major: str,
    mode0: str,
    tensor_name: str,
    group: int,
) -> None:
    if int(tensor.data_ptr()) % 16 != 0:
        raise ValueError(f"{tensor_name}[{group}] data pointer must be 16-byte aligned")
    elements_per_16b = 16 // tensor.element_size()
    major_dim = 0 if major == mode0 else 1
    if int(tensor.size(major_dim)) % elements_per_16b != 0:
        raise ValueError(
            f"{tensor_name}[{group}] contiguous dimension must be 16-byte aligned"
        )
    leading_stride_bytes = int(tensor.stride(1 - major_dim)) * tensor.element_size()
    if leading_stride_bytes % 16 != 0:
        raise ValueError(
            f"{tensor_name}[{group}] leading stride must be 16-byte aligned"
        )


def _blackwell_grouped_gemm_nt_generated_check_strides_fit_int32(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    group: int,
) -> None:
    for stride in tensor.stride():
        if stride < 0 or stride > torch.iinfo(torch.int32).max:
            raise ValueError(
                f"{tensor_name}[{group}] strides must be non-negative int32 values"
            )


def _blackwell_grouped_gemm_nt_generated_metadata(
    a_groups: Sequence[torch.Tensor],
    b_groups: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
) -> _BlackwellGeneratedGemmMetadata:
    if not a_groups:
        raise ValueError("grouped GEMM requires at least one group")
    if len(a_groups) != len(b_groups) or len(a_groups) != len(out_groups):
        raise ValueError("A, B, and output group counts must match")

    first_a = a_groups[0]
    first_b = b_groups[0]
    first_out = out_groups[0]
    if first_a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "generated grouped GEMM A/B dtype must be torch.float16 or torch.bfloat16"
        )
    if first_out.dtype != first_a.dtype:
        raise ValueError("generated grouped GEMM output dtype must match A/B dtype")
    if first_a.device.type != "cuda":
        raise ValueError("generated grouped GEMM tensors must be on CUDA")
    if first_b.device != first_a.device or first_out.device != first_a.device:
        raise ValueError("all generated grouped GEMM tensors must share one device")

    a_major = _blackwell_grouped_gemm_nt_generated_major_mode(
        first_a,
        mode0="m",
        mode1="k",
        tensor_name="A",
        group=0,
    )
    b_major = _blackwell_grouped_gemm_nt_generated_major_mode(
        first_b,
        mode0="n",
        mode1="k",
        tensor_name="B",
        group=0,
    )
    c_major = _blackwell_grouped_gemm_nt_generated_major_mode(
        first_out,
        mode0="m",
        mode1="n",
        tensor_name="out",
        group=0,
    )

    problems: list[_BlackwellGeneratedGemmProblem] = []
    for group, (a, b, out) in enumerate(
        zip(a_groups, b_groups, out_groups, strict=True)
    ):
        if a.ndim != 2:
            raise ValueError(f"A[{group}] must have shape [M, K]")
        if b.ndim != 2:
            raise ValueError(f"B[{group}] must have shape [N, K]")
        if out.ndim != 2:
            raise ValueError(f"out[{group}] must have shape [M, N]")
        if a.dtype != first_a.dtype or b.dtype != first_a.dtype:
            raise ValueError("all generated grouped GEMM A/B tensors must share dtype")
        if out.dtype != first_out.dtype:
            raise ValueError("all generated grouped GEMM outputs must share dtype")
        if a.device != first_a.device or b.device != first_a.device:
            raise ValueError("all generated grouped GEMM A/B tensors must share device")
        if out.device != first_a.device:
            raise ValueError("all generated grouped GEMM outputs must share device")

        m = int(a.size(0))
        k = int(a.size(1))
        n = int(b.size(0))
        if m <= 0 or n <= 0 or k <= 0:
            raise ValueError("generated grouped GEMM dimensions must be positive")
        if int(b.size(1)) != k:
            raise ValueError(f"K dimension mismatch for group {group}")
        if tuple(int(size) for size in out.shape) != (m, n):
            raise ValueError(f"out[{group}] must have shape [{m}, {n}]")

        group_a_major = _blackwell_grouped_gemm_nt_generated_major_mode(
            a,
            mode0="m",
            mode1="k",
            tensor_name="A",
            group=group,
        )
        group_b_major = _blackwell_grouped_gemm_nt_generated_major_mode(
            b,
            mode0="n",
            mode1="k",
            tensor_name="B",
            group=group,
        )
        group_c_major = _blackwell_grouped_gemm_nt_generated_major_mode(
            out,
            mode0="m",
            mode1="n",
            tensor_name="out",
            group=group,
        )
        if group_a_major != a_major:
            raise ValueError("all generated grouped GEMM A tensors must share layout")
        if group_b_major != b_major:
            raise ValueError("all generated grouped GEMM B tensors must share layout")
        if group_c_major != c_major:
            raise ValueError("all generated grouped GEMM outputs must share layout")

        _blackwell_grouped_gemm_nt_generated_check_16b_alignment(
            a,
            major=group_a_major,
            mode0="m",
            tensor_name="A",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_16b_alignment(
            b,
            major=group_b_major,
            mode0="n",
            tensor_name="B",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_16b_alignment(
            out,
            major=group_c_major,
            mode0="m",
            tensor_name="out",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_strides_fit_int32(
            a,
            tensor_name="A",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_strides_fit_int32(
            b,
            tensor_name="B",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_strides_fit_int32(
            out,
            tensor_name="out",
            group=group,
        )
        problems.append(_BlackwellGeneratedGemmProblem(m, n, k))

    return _BlackwellGeneratedGemmMetadata(
        device=first_a.device,
        problems=tuple(problems),
        a_mode0_major=a_major == "m",
        b_mode0_major=b_major == "n",
        c_mode0_major=c_major == "m",
    )


@helion.kernel(backend="cute")
def _blackwell_grouped_gemm_nt_generated_kernel(
    a_placeholder: torch.Tensor,
    b_placeholder: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    out_placeholder: torch.Tensor,
    direct_pointers: torch.Tensor,
    direct_strides: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a_placeholder.size()
    _g, max_n, _k = b_placeholder.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m]
        valid_rows = row_group_ids == safe_group_id
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a_placeholder[tile_m, tile_k]
            b_tile = b_placeholder[safe_group_id, tile_n, tile_k]
            valid_k_mask = valid_k[None, :]  # pyrefly: ignore[bad-index]
            masked_a = torch.where(valid_k_mask, a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k_mask, b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)
        out_placeholder[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out_placeholder.dtype),
            out_placeholder[tile_m, tile_n],
        )
    return out_placeholder


def _blackwell_grouped_gemm_nt_generated_config(
    *, reserved_sms: int = 0
) -> helion.Config:
    config_kwargs: dict[str, Any] = {}
    if reserved_sms:
        config_kwargs["tcgen05_grouped_static_reserved_sms"] = int(reserved_sms)
    return helion.Config(
        block_sizes=[128, 64, 64],
        l2_groupings=[1],
        loop_orders=[[0, 1]],
        num_stages=2,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_static_persistent=True,
        tcgen05_grouped_dynamic_ab_tensormaps=True,
        tcgen05_grouped_direct_pointer_metadata=True,
        tcgen05_grouped_external_direct_pointers="direct_pointers",
        tcgen05_grouped_external_direct_strides="direct_strides",
        **config_kwargs,
    )


def _blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
    problems: Sequence[_BlackwellGeneratedGemmProblem],
    *,
    num_sm: int,
    block_m: int,
    block_n: int,
) -> int:
    if num_sm <= 0 or block_m <= 0 or block_n <= 0:
        return 0
    total_ctas = 0
    m_tail_ctas = 0
    for problem in problems:
        m_tiles = (problem.m + block_m - 1) // block_m
        n_tiles = (problem.n + block_n - 1) // block_n
        total_ctas += m_tiles * n_tiles
        if problem.m % block_m != 0:
            m_tail_ctas += n_tiles
    if m_tail_ctas == 0:
        return 0
    residual_ctas = total_ctas % num_sm
    # The active-cluster cap is only useful for M-tail work landing in a
    # partially full final scheduler wave. Small-tail waves below this band and
    # no-tail shapes retain the default full-SM launch.
    if residual_ctas * 16 > num_sm * 7 and residual_ctas * 2 < num_sm:
        return _BLACKWELL_GROUPED_GENERATED_MTAIL_RESERVED_SMS
    if (
        residual_ctas * 8 >= num_sm * 3
        and residual_ctas * 16 <= num_sm * 7
        and residual_ctas * 2 < num_sm
        and m_tail_ctas * 8 >= num_sm
    ):
        return _BLACKWELL_GROUPED_GENERATED_LARGE_MTAIL_RESERVED_SMS
    return 0


def _blackwell_grouped_gemm_nt_generated_config_block_mn(
    config: helion.Config,
) -> tuple[int, int]:
    block_sizes = config.config["block_sizes"]
    assert isinstance(block_sizes, list)
    return int(block_sizes[0]), int(block_sizes[1])


def _blackwell_grouped_gemm_nt_generated_reserved_sms(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    *,
    block_m: int,
    block_n: int,
) -> int:
    if not group_A:
        return 0
    device = group_A[0].device
    if device.type != "cuda":
        return 0
    num_sm = int(torch.cuda.get_device_properties(device).multi_processor_count)
    problems = tuple(
        _BlackwellGeneratedGemmProblem(
            int(a.size(0)),
            int(b.size(0)),
            int(a.size(1)),
        )
        for a, b in zip(group_A, group_B, strict=True)
    )
    return _blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
        problems,
        num_sm=num_sm,
        block_m=block_m,
        block_n=block_n,
    )


def _blackwell_grouped_gemm_nt_generated_capture_active() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _blackwell_grouped_gemm_nt_placeholder(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if len(shape) == 2:
        base = torch.empty(max(1, shape[1]), device=device, dtype=dtype)
        return torch.as_strided(base, shape, (0, 1))
    if len(shape) == 3:
        base = torch.empty(max(1, shape[2]), device=device, dtype=dtype)
        return torch.as_strided(base, shape, (0, 0, 1))
    raise AssertionError(f"unexpected placeholder shape {shape!r}")


def _blackwell_grouped_gemm_nt_generated_tensor_cache_key(
    tensor: torch.Tensor,
) -> tuple[object, ...]:
    return (
        id(tensor),
        int(tensor.data_ptr()),
        tensor.device.type,
        tensor.device.index,
        str(tensor.dtype),
        tensor.ndim,
        tuple(int(size) for size in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        int(tensor.storage_offset()),
    )


def _blackwell_grouped_gemm_nt_generated_tensor_guard(
    tensor: torch.Tensor,
) -> _BlackwellGeneratedTensorGuard:
    return _BlackwellGeneratedTensorGuard(
        tensor_id=id(tensor),
        data_ptr=int(tensor.data_ptr()),
        device_type=tensor.device.type,
        device_index=tensor.device.index,
        dtype=tensor.dtype,
        ndim=tensor.ndim,
        shape=tuple(int(size) for size in tensor.shape),
        stride=tuple(int(stride) for stride in tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
    )


def _blackwell_grouped_gemm_nt_generated_stable_launch_guard(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
) -> _BlackwellGeneratedStableLaunchGuard:
    return _BlackwellGeneratedStableLaunchGuard(
        mma_tiler_mn=tuple(int(dim) for dim in mma_tiler_mn),
        cluster_shape_mn=tuple(int(dim) for dim in cluster_shape_mn),
        use_2cta_instrs=bool(use_2cta_instrs),
        a_guards=tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_guard(a) for a in group_A
        ),
        b_guards=tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_guard(b) for b in group_B
        ),
        out_guards=tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_guard(out) for out in out_groups
        ),
    )


def _blackwell_grouped_gemm_nt_generated_int_sequence_matches(
    actual: Sequence[int],
    expected: Sequence[int],
) -> bool:
    return len(actual) == len(expected) and all(
        int(actual_value) == expected_value
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )


def _blackwell_grouped_gemm_nt_generated_tensor_guard_matches(
    guard: _BlackwellGeneratedTensorGuard,
    tensor: torch.Tensor,
) -> bool:
    return (
        guard.tensor_id == id(tensor)
        and guard.data_ptr == int(tensor.data_ptr())
        and guard.device_type == tensor.device.type
        and guard.device_index == tensor.device.index
        and guard.dtype == tensor.dtype
        and guard.ndim == tensor.ndim
        and guard.shape == tuple(int(size) for size in tensor.shape)
        and guard.stride == tuple(int(stride) for stride in tensor.stride())
        and guard.storage_offset == int(tensor.storage_offset())
    )


def _blackwell_grouped_gemm_nt_generated_tensor_guards_match(
    guards: Sequence[_BlackwellGeneratedTensorGuard],
    tensors: Sequence[torch.Tensor],
) -> bool:
    return len(guards) == len(tensors) and all(
        starmap(
            _blackwell_grouped_gemm_nt_generated_tensor_guard_matches,
            zip(guards, tensors, strict=True),
        )
    )


def _blackwell_grouped_gemm_nt_generated_stable_launch_guard_matches(
    guard: _BlackwellGeneratedStableLaunchGuard,
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
) -> bool:
    return (
        _blackwell_grouped_gemm_nt_generated_int_sequence_matches(
            mma_tiler_mn,
            guard.mma_tiler_mn,
        )
        and _blackwell_grouped_gemm_nt_generated_int_sequence_matches(
            cluster_shape_mn,
            guard.cluster_shape_mn,
        )
        and guard.use_2cta_instrs == bool(use_2cta_instrs)
        and _blackwell_grouped_gemm_nt_generated_tensor_guards_match(
            guard.a_guards,
            group_A,
        )
        and _blackwell_grouped_gemm_nt_generated_tensor_guards_match(
            guard.b_guards,
            group_B,
        )
        and _blackwell_grouped_gemm_nt_generated_tensor_guards_match(
            guard.out_guards,
            out_groups,
        )
    )


def _blackwell_grouped_gemm_nt_generated_stable_launch_key(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
) -> tuple[object, ...]:
    return (
        tuple(int(dim) for dim in mma_tiler_mn),
        tuple(int(dim) for dim in cluster_shape_mn),
        bool(use_2cta_instrs),
        tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_cache_key(a) for a in group_A
        ),
        tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_cache_key(b) for b in group_B
        ),
        tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_cache_key(out)
            for out in out_groups
        ),
    )


def _blackwell_grouped_gemm_nt_generated_fast_call(
    bound: Callable[..., object],
) -> tuple[Callable[[], object], Any] | None:
    run = getattr(bound, "_run", None)
    run_globals = getattr(run, "__globals__", None)
    if not isinstance(run_globals, dict):
        return None

    try:
        from helion.runtime import _cute_current_stream
    except ImportError:
        return None

    for value in run_globals.values():
        entry = getattr(value, "_helion_cute_last_launch_cache", None)
        compiled = getattr(entry, "compiled", None)
        launch_args = getattr(entry, "launch_args", None)
        if callable(compiled) and isinstance(launch_args, tuple):

            def fast_call(
                compiled: Callable[..., object] = compiled,
                launch_args: tuple[object, ...] = launch_args,
            ) -> object:
                return compiled(*launch_args, _cute_current_stream())

            return fast_call, entry
    return None


def _blackwell_grouped_gemm_nt_generated_cuda_graph(
    fast_call: Callable[[], object],
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fast_call()
    return graph


def _blackwell_grouped_gemm_nt_generated_call_stable_launch(
    launch: _BlackwellGeneratedStableLaunch,
) -> None:
    if (
        launch.cuda_graph is not None
        and not _blackwell_grouped_gemm_nt_generated_capture_active()
    ):
        launch.cuda_graph.replay()
        return
    if launch.fast_call is not None:
        launch.fast_call()
    else:
        launch.bound(*launch.kernel_args)


def _blackwell_grouped_gemm_nt_generated_last_stable_launch(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
) -> _BlackwellGeneratedStableLaunch | None:
    last_launch = _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH
    if last_launch is None:
        return None
    if not _blackwell_grouped_gemm_nt_generated_stable_launch_guard_matches(
        last_launch.guard,
        group_A,
        group_B,
        out_groups,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    ):
        return None
    return last_launch


def _blackwell_grouped_gemm_nt_generated_args(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int] = (128, 64),
    cluster_shape_mn: Sequence[int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None:
    if tuple(int(dim) for dim in mma_tiler_mn) != (128, 64):
        return None
    if tuple(int(dim) for dim in cluster_shape_mn) != (1, 1):
        return None
    if use_2cta_instrs:
        return None
    if _blackwell_grouped_gemm_nt_generated_capture_active():
        return None

    a_tuple = tuple(group_A)
    b_tuple = tuple(group_B)
    out_tuple = tuple(out_groups)
    try:
        metadata = _blackwell_grouped_gemm_nt_generated_metadata(
            a_tuple,
            b_tuple,
            out_tuple,
        )
    except ValueError:
        return None
    if metadata.a_mode0_major or metadata.b_mode0_major or metadata.c_mode0_major:
        return None
    if any(problem.k % 16 != 0 for problem in metadata.problems):
        return None

    block_m = 128
    aligned_m = tuple(
        ((problem.m + block_m - 1) // block_m) * block_m
        for problem in metadata.problems
    )
    padded_m = sum(aligned_m)
    max_n = max(problem.n for problem in metadata.problems)
    max_k = max(problem.k for problem in metadata.problems)
    if max_n % 64 != 0 or max_k % 64 != 0:
        return None
    device = metadata.device

    layout = torch.empty(padded_m, device=device, dtype=torch.int32)
    cursor = 0
    for group, (problem, group_aligned_m) in enumerate(
        zip(metadata.problems, aligned_m, strict=True)
    ):
        layout[cursor : cursor + problem.m].fill_(group)
        if problem.m != group_aligned_m:
            layout[cursor + problem.m : cursor + group_aligned_m].fill_(-1)
        cursor += group_aligned_m

    n_sizes = torch.tensor(
        [problem.n for problem in metadata.problems],
        device=device,
        dtype=torch.int32,
    )
    k_sizes = torch.tensor(
        [problem.k for problem in metadata.problems],
        device=device,
        dtype=torch.int32,
    )
    direct_pointers = torch.tensor(
        [
            (int(a.data_ptr()), int(b.data_ptr()), int(out.data_ptr()))
            for a, b, out in zip(a_tuple, b_tuple, out_tuple, strict=True)
        ],
        device=device,
        dtype=torch.int64,
    )
    direct_strides = torch.tensor(
        [
            (
                (int(a.stride(0)), int(a.stride(1))),
                (int(b.stride(0)), int(b.stride(1))),
                (int(out.stride(0)), int(out.stride(1))),
            )
            for a, b, out in zip(a_tuple, b_tuple, out_tuple, strict=True)
        ],
        device=device,
        dtype=torch.int32,
    )

    a_placeholder = _blackwell_grouped_gemm_nt_placeholder(
        (padded_m, max_k),
        dtype=a_tuple[0].dtype,
        device=device,
    )
    b_placeholder = _blackwell_grouped_gemm_nt_placeholder(
        (len(metadata.problems), max_n, max_k),
        dtype=a_tuple[0].dtype,
        device=device,
    )
    out_placeholder = _blackwell_grouped_gemm_nt_placeholder(
        (padded_m, max_n),
        dtype=out_tuple[0].dtype,
        device=device,
    )
    return (
        (
            a_placeholder,
            b_placeholder,
            layout,
            n_sizes,
            k_sizes,
            out_placeholder,
            direct_pointers,
            direct_strides,
        ),
        out_tuple,
    )


def _blackwell_grouped_gemm_nt_generated(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
    cache_stable_launch: bool = False,
) -> tuple[torch.Tensor, ...] | None:
    global _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH

    stable_launch_key: tuple[object, ...] | None = None
    if cache_stable_launch:
        cached_launch = _blackwell_grouped_gemm_nt_generated_last_stable_launch(
            group_A,
            group_B,
            out_groups,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
        )
        if cached_launch is not None:
            _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.move_to_end(
                cached_launch.cache_key
            )
            _blackwell_grouped_gemm_nt_generated_call_stable_launch(cached_launch)
            return tuple(out_groups)
        stable_launch_key = _blackwell_grouped_gemm_nt_generated_stable_launch_key(
            group_A,
            group_B,
            out_groups,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
        )
        cached_launch = _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.get(stable_launch_key)
        if cached_launch is not None:
            if _blackwell_grouped_gemm_nt_generated_stable_launch_guard_matches(
                cached_launch.guard,
                group_A,
                group_B,
                out_groups,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                use_2cta_instrs=use_2cta_instrs,
            ):
                _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.move_to_end(stable_launch_key)
                _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH = cached_launch
                _blackwell_grouped_gemm_nt_generated_call_stable_launch(cached_launch)
                return tuple(out_groups)
            _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.discard(stable_launch_key)
        if _blackwell_grouped_gemm_nt_generated_capture_active():
            return None

    prepared = _blackwell_grouped_gemm_nt_generated_args(
        group_A,
        group_B,
        out_groups,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    if prepared is None:
        return None

    kernel_args, out_tuple = prepared
    config = _blackwell_grouped_gemm_nt_generated_config()
    block_m, block_n = _blackwell_grouped_gemm_nt_generated_config_block_mn(config)
    block_sizes = config.config["block_sizes"]
    assert isinstance(block_sizes, list)
    block_k = int(block_sizes[2])
    use_deep_ab_stages = max(int(a.size(1)) for a in group_A) >= 3 * block_k
    reserved_sms = _blackwell_grouped_gemm_nt_generated_reserved_sms(
        group_A,
        group_B,
        block_m=block_m,
        block_n=block_n,
    )
    if reserved_sms:
        config = _blackwell_grouped_gemm_nt_generated_config(reserved_sms=reserved_sms)
    if use_deep_ab_stages:
        config.config["tcgen05_ab_stages"] = 4
    bound = _blackwell_grouped_gemm_nt_generated_kernel.bind(kernel_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(config)
    bound(*kernel_args)
    if stable_launch_key is not None:
        stable_fast_call = _blackwell_grouped_gemm_nt_generated_fast_call(bound)
        stable_cuda_graph = (
            _blackwell_grouped_gemm_nt_generated_cuda_graph(stable_fast_call[0])
            if stable_fast_call is not None
            else None
        )
        stable_launch = _BlackwellGeneratedStableLaunch(
            cache_key=stable_launch_key,
            guard=_blackwell_grouped_gemm_nt_generated_stable_launch_guard(
                group_A,
                group_B,
                out_groups,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                use_2cta_instrs=use_2cta_instrs,
            ),
            kernel_args=kernel_args,
            bound=bound,
            fast_call=stable_fast_call[0] if stable_fast_call is not None else None,
            cuda_graph=stable_cuda_graph,
            runtime_cache_entry=(
                stable_fast_call[1] if stable_fast_call is not None else None
            ),
        )
        _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE[stable_launch_key] = stable_launch
        _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.move_to_end(stable_launch_key)
        _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH = stable_launch
        while (
            len(_BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE)
            > _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE_MAX
        ):
            _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.popitem(last=False)
    return out_tuple


def blackwell_grouped_gemm_nt_direct(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    *,
    out_groups: Sequence[torch.Tensor] | None = None,
    out_dtype: torch.dtype | None = None,
    mma_tiler_mn: Sequence[int] = (128, 64),
    cluster_shape_mn: Sequence[int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> tuple[torch.Tensor, ...]:
    """
    Run the explicit generated direct-pointer Blackwell grouped GEMM route.

    This opt-in path is for callers that intentionally want the generated
    direct-pointer kernel. Unsupported shapes or configs raise ``ValueError``
    instead of silently falling back to the general grouped GEMM harness.
    """
    a_tuple = tuple(group_A)
    b_tuple = tuple(group_B)
    explicit_out_groups = out_groups is not None
    if out_groups is None:
        if not a_tuple:
            raise ValueError("grouped GEMM requires at least one group")
        if len(a_tuple) != len(b_tuple):
            raise ValueError("A and B group counts must match")
        output_dtype = a_tuple[0].dtype if out_dtype is None else out_dtype
        out_tuple = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=a.device,
                dtype=output_dtype,
            )
            for a, b in zip(a_tuple, b_tuple, strict=True)
        )
    else:
        if out_dtype is not None:
            raise ValueError("out_dtype cannot be provided with explicit out_groups")
        out_tuple = tuple(out_groups)

    generated = _blackwell_grouped_gemm_nt_generated(
        a_tuple,
        b_tuple,
        out_tuple,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
        cache_stable_launch=explicit_out_groups,
    )
    if generated is None:
        raise ValueError(
            "blackwell_grouped_gemm_nt_direct does not support this shape or config"
        )
    return generated
