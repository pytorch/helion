# Kraken to Helion DSL: Complete Analysis

**Generated**: 2025-11-02
**Version**: 2.0 (Deduplicated)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Kernel-by-Kernel Gap Analysis](#kernel-by-kernel-gap-analysis)
4. [Missing APIs Specification](#missing-apis-specification)
5. [Translation Examples](#translation-examples)
6. [Design Considerations](#design-considerations)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Success Metrics and Conclusion](#success-metrics-and-conclusion)

---

## Executive Summary

### Current State
- **Kraken**: Triton-based library with 13+ distributed kernel recipes (all-reduce, GEMM fusion, etc.)
- **Helion**: High-level DSL ("PyTorch with tiles") for single-kernel GPU programming
- **Gap**: Helion lacks distributed/multi-rank communication primitives

### Key Finding
✅ **All major Kraken kernel patterns can be expressed in Helion DSL** with the addition of distributed communication APIs.

### Expected Benefits
- **30-50% code reduction** compared to Triton
- **Better readability** through higher-level abstractions
- **Automatic optimizations** (masking, indexing, autotuning)
- **Safety improvements** (deadlock detection, divergence analysis)

### Implementation Effort
- **Core functionality (P0)**: 4-6 weeks
- **Full feature parity**: 16-24 weeks across 6 phases
- **Lines of code**: ~3,000-5,000 (estimated)

---

## System Overview

### Kraken Architecture
- **Purpose**: Distributed GPU kernel library using PyTorch symmetric memory
- **Abstraction Level**: Low-level Triton with explicit PTX synchronization
- **Primary Use Case**: Multi-rank communication + computation fusion
- **Key Primitives**:
  - Symmetric memory access across devices
  - Cross-device atomic barriers
  - Persistent kernel patterns with progress tracking
  - GEMM + communication fusion

**Communication Kernels**:
1. **one_shot_all_reduce** - Single-phase all-reduce (optimal <400KB)
2. **two_shot_all_reduce** - Two-phase with intermediate buffer (better load balancing)
3. **copy_engine_all_gather** - All-gather with progress tracking

**Fused Computation + Communication**:
4. **gemm_one_shot_all_reduce_fused** - Matrix mult + all-reduce fusion
5. **gemm_reduce_scatter_fused** - Matrix mult + scatter-reduce
6. **all_gather_matmul** - All-gather + matrix mult with producer-consumer pattern
7. **gemm_reduce_scatter_ce_persistent** - Persistent kernel variant

**Element-wise with Communication**:
8. **one_shot_all_reduce_bias** - All-reduce + bias addition
9. **one_shot_all_reduce_bias_rms_norm** - All-reduce + bias + RMS norm

### Helion DSL Architecture
- **Purpose**: High-level GPU kernel DSL ("PyTorch with tiles")
- **Abstraction Level**: Higher than Triton, hides indexing details
- **Primary Use Case**: Single-kernel computation with autotuning

**✅ Well-Supported Features**:
- **Tile/grid loops**: `grid()`, `tile()`, `static_range()`
- **Memory operations**: `load()`, `store()` with implicit masking
- **Matrix operations**: `dot()` with FP8/BF16/FP32 support
- **Reductions**: `reduce()`, `cumsum()`, `associative_scan()`
- **Atomics**: `atomic_add()`, `atomic_max()`, etc. (on host tensors)
- **Synchronization**: `signal()`, `wait()` (intra-kernel only)
- **Autotuning**: ~1500 config search via Differential Evolution

**❌ Missing for Distributed Computing**:
- No multi-rank communication primitives
- No cross-device memory access
- No distributed synchronization (barriers across GPUs)
- No rank/world_size intrinsics
- No progress tracking patterns
- Unclear persistent kernel support

---

## Kernel-by-Kernel Gap Analysis

### 1. Pure Communication Kernels

#### one_shot_all_reduce / two_shot_all_reduce

**Kraken Pattern**:
```python
@triton.jit
def _one_shot_all_reduce_kernel(
    in_ptr, out_ptr, n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, buf_tuple: tl.constexpr, world_size: tl.constexpr
):
    # Load from all ranks
    for i in tl.static_range(world_size):
        data = tl.load(buf_tuple[i] + offsets, mask=mask)
        acc += data
    tl.store(out_ptr + offsets, acc, mask=mask)
```

**Gap Analysis**:
- ❌ No `DistributedHandles` or symmetric memory abstraction
- ❌ No `load_from_rank()` for cross-device memory access
- ❌ No `distributed_barrier()` for cross-device synchronization
- ❌ No world_size/rank intrinsics

#### copy_engine_all_gather_w_progress

**Gap Analysis**:
- ❌ No backend/copy engine stream abstraction
- ❌ No async copy primitives separate from compute stream
- ⚠️ Atomic ops exist but limited to host tensors
- ❌ No progress tracking pattern

### 2. Fused GEMM + Communication Kernels

#### gemm_one_shot_all_reduce_fused

**Gap Analysis**:
- ✅ Matrix multiplication with `dot()` is well-supported
- ✅ `static_range()` exists for compile-time loops
- ✅ Mixed precision (BF16/FP32) supported
- ❌ No distributed memory abstraction
- ❌ No cross-device barriers
- ❌ No load_from_rank/store_to_rank primitives

### 3. Persistent Kernel Pattern

#### gemm_reduce_scatter_ce_persistent

**Gap Analysis**:
- ⚠️ **UNCLEAR**: Can Helion's `grid()` express persistent kernels where blocks loop over multiple tiles?
- ⚠️ **UNCLEAR**: Are while loops supported in kernels?
- ✅ `max_num_blocks` configuration parameter exists
- ❌ No explicit control over block-to-tile mapping

### 4. Element-wise Operations

#### one_shot_all_reduce_bias / rms_norm

**Gap Analysis**:
- ✅ Element-wise operations well-supported
- ✅ `reduce()` with various ops (sum, mean, etc.)
- ✅ Math functions (sqrt, etc.)
- ✅ Subscripting with `:` for full dimension

### 5. All-Gather + MatMul (Producer-Consumer)

#### all_gather_matmul

**Gap Analysis**:
- ❌ No progress tensor polling/waiting primitive
- ❌ No producer-consumer coordination beyond signal/wait
- ⚠️ Helion has `wait()` but unclear if it supports external progress tensors
- ❌ No cross-device data loading

---

## Missing APIs Specification

### Priority 0 (CRITICAL - Core Distributed Functionality)

#### 1. Distributed Tensor Type
```python
class DistributedTensor:
    """Tensor with symmetric memory across ranks"""
    @staticmethod
    def from_torch(tensor, group) -> DistributedTensor:
        """Create from PyTorch tensor with symmetric memory"""
```

#### 2. Cross-Rank Memory Operations
```python
def load_from_rank(tensor: DistributedTensor, rank: int, indices...) -> Tile:
    """Load tile from specific rank's memory"""

def store_to_rank(tensor: DistributedTensor, rank: int, indices..., value: Tile):
    """Store tile to local symmetric memory (visible to all ranks)"""
```

#### 3. Distributed Synchronization
```python
def distributed_barrier(
    tensor: DistributedTensor,
    pattern: int = 1  # 0: write-only, 1: bidirectional, 2: read-only
):
    """Synchronize across all ranks using atomic operations

    Args:
        tensor: Distributed memory handles
        pattern: Sync pattern (0: write-only, 1: bidirectional, 2: read-only)
    """
```

#### 4. Rank Intrinsics
```python
def my_rank() -> int:
    """Returns current GPU rank in distributed group"""

def world_size() -> int:
    """Returns total number of ranks"""
```

**Impact**: Without P0, cannot express any Kraken kernels.

---

### Priority 1 (IMPORTANT - Advanced Patterns)

#### 5. Progress Tracking
```python
def wait_on_progress(progress: Tensor, index: int, expected_value: int = 1):
    """Spin-wait until progress[index] == expected_value

    Uses efficient polling with exponential backoff.
    """

def signal_progress(progress: Tensor, index: int, value: int):
    """Atomically set progress[index] = value

    Uses stream_write_value32 equivalent.
    """
```

#### 6. Persistent Kernel Control
```python
@kernel
@config(
    execution_mode="persistent",  # vs "regular"
    max_num_blocks=24,
)
def my_kernel(...):
    # Helion automatically generates persistent tile loop
    for m, n in grid(M, N):  # Each block processes multiple tiles
        ...
```

**Impact**: Enables producer-consumer patterns and performance optimization.

---

### Priority 2 (NICE-TO-HAVE - Performance)

#### 7. Stream Control
```python
def copy_async(src: Tensor, dst: Tensor, stream: str = "compute"):
    """Async copy on compute or copy engine stream"""

def wait_stream(stream: str):
    """Wait for stream operations to complete"""
```

#### 8. Cache Control
```python
def load(tensor, indices..., eviction_policy: str = "evict_normal"):
    """Load with cache eviction hint: evict_first, evict_last, evict_normal"""
```

**Impact**: Further performance tuning for expert users.

---

### Priority 3 (LOW-LEVEL - Expert Users)

#### 9. PTX Inline Assembly
```python
def inline_ptx(
    ptx_code: str,
    constraints: str,
    inputs: list,
    outputs: list,
    side_effects: bool = False
):
    """Inline arbitrary PTX assembly

    Needed for custom synchronization primitives like symm_mem_sync.
    """
```

**Impact**: Escape hatch for custom patterns not covered by high-level APIs.

---

## Translation Examples

This section provides complete side-by-side comparisons of Kraken kernels (Triton) and their proposed Helion DSL equivalents.

### Example 1: One-Shot All-Reduce

#### Kraken (Triton) - 40 lines

```python
# File: kraken/comm/one_shot_all_reduce.py
import triton
import triton.language as tl
from kraken._ptx_utils import symm_mem_sync

@triton.jit
def _one_shot_all_reduce_kernel(
    in_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    buf_tuple: tl.constexpr,
    world_size: tl.constexpr,
):
    """One-shot all-reduce kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Load and accumulate from all ranks
    for i in tl.static_range(world_size):
        data = tl.load(buf_tuple[i] + offsets, mask=mask, other=0.0)
        acc += data

    # Store result
    tl.store(out_ptr + offsets, acc, mask=mask)

def one_shot_all_reduce(
    tensor: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    max_num_blocks: int = 128,
    num_warps: int = 32,
    BLOCK_SIZE: int = 8192,
) -> torch.Tensor:
    """Host wrapper"""
    symm_mem_hdl = symm_mem.rendezvous(tensor, group=group)
    buf_list = [
        symm_mem_hdl.get_buffer(i, tensor.shape, tensor.dtype)
        for i in range(world_size)
    ]

    grid = lambda meta: (
        min(triton.cdiv(tensor.numel(), meta["BLOCK_SIZE"]), max_num_blocks),
    )

    _one_shot_all_reduce_kernel[grid](
        tensor,
        tensor,
        tensor.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
        buf_tuple=tuple(buf_list),
        world_size=world_size,
        num_warps=num_warps,
    )

    return tensor
```

#### Helion DSL (Proposed) - 20 lines

```python
# File: helion_examples/distributed/one_shot_all_reduce.py
from helion import kernel, Tensor, zeros, load, store, static_range
from helion.distributed import (
    DistributedHandles,
    load_from_rank,
    world_size,
)

@kernel
@config(
    block_sizes={"elements": [2048, 4096, 8192]},
    num_warps=[16, 32],
    max_num_blocks=[64, 128],
)
def one_shot_all_reduce(
    x: Tensor,  # [N] - input/output tensor
    symm_mem: DistributedHandles,  # Symmetric memory handles
):
    """One-shot all-reduce: out = sum(x across all ranks)

    Each rank contributes its tensor x, and all ranks receive the sum.
    Optimized for small tensors (<~400KB).
    """
    N = x.shape[0]

    # Process tensor in tiles
    for i in tile(N):
        acc = zeros(BLOCK_SIZE, dtype=float32)

        # Load and accumulate from all ranks
        for rank in static_range(world_size()):
            data = load_from_rank(symm_mem[rank], i)
            acc += data

        # Store reduced result
        store(x, i, acc)

# Host wrapper
def one_shot_all_reduce_host(
    tensor: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
) -> torch.Tensor:
    """Host-side wrapper for one-shot all-reduce"""
    import torch.distributed._symmetric_memory as symm_mem
    from helion.distributed import DistributedHandles

    # Create symmetric memory
    symm_mem_hdl = symm_mem.rendezvous(tensor, group=group)

    # Convert to Helion handles
    helion_handles = DistributedHandles.from_torch(symm_mem_hdl)

    # Launch kernel (Helion autotuning will optimize configuration)
    one_shot_all_reduce(tensor, helion_handles)

    return tensor
```

**Key Improvements**:
- ✅ 50% code reduction (40 → 20 lines)
- ✅ No manual offset calculation or masking
- ✅ No manual grid calculation - handled by Helion
- ✅ `load_from_rank()` abstracts buffer tuple access
- ✅ `world_size()` intrinsic vs passing as constexpr
- ✅ Autotunable block size and num_warps

---

### Example 2: GEMM + All-Reduce Fusion

#### Kraken (Triton) - 90 lines

```python
# File: kraken/fused/gemm_one_shot_all_reduce_fused.py
@triton.jit
def _gemm_one_shot_all_reduce_fused_kernel(
    a_ptr, b_ptr, out_buf_tuple, n, m, k,
    stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    INTERMEDIATE_DTYPE: tl.constexpr,
    world_size: tl.constexpr,
    rank: tl.constexpr,
):
    """Fused GEMM + all-reduce kernel"""
    # Swizzle for better L2 cache hit rate
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute pointers for A and B tiles
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Compute local GEMM
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=INTERMEDIATE_DTYPE)
    for k_idx in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < m) & (offs_k[None, :] < k))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k) & (offs_bn[None, :] < n))
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store to local symmetric memory
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    c_ptrs = out_buf_tuple[rank] + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=mask)

    # Synchronize across devices
    symm_mem_sync(
        out_buf_tuple,
        pattern=1,
        signal_pad_offset=0,
        world_size=world_size,
    )

    # Reduce from all ranks
    result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=INTERMEDIATE_DTYPE)
    for i in tl.static_range(world_size):
        data_ptrs = out_buf_tuple[i] + stride_cm * offs_cm[:, None] + offs_cn[None, :]
        data = tl.load(data_ptrs, mask=mask, other=0.0)
        result += data

    # Store final result (cast to bfloat16)
    out_ptrs = out_buf_tuple[rank] + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    tl.store(out_ptrs, result.to(tl.bfloat16), mask=mask)
```

#### Helion DSL (Proposed) - 35 lines

```python
# File: helion_examples/distributed/gemm_all_reduce_fused.py
from helion import kernel, Tensor, zeros, load, store, dot, static_range
from helion.distributed import (
    DistributedHandles,
    load_from_rank,
    store_to_rank,
    distributed_barrier,
    world_size,
    my_rank,
)

@kernel
@config(
    block_sizes={
        "M": [64, 128, 256],
        "N": [64, 128, 256],
        "K": [32, 64, 128],
    },
    num_warps=[8, 16, 32],
    intermediate_dtype="float32",
    group_size_m=8,  # For L2 cache swizzling
)
def gemm_all_reduce_fused(
    a: Tensor,  # [M, K]
    b: Tensor,  # [K, N]
    out: Tensor,  # [M, N]
    symm_mem: DistributedHandles,
):
    """Fused GEMM + all-reduce: out = all_reduce(a @ b)

    Computes local matrix multiplication then reduces across all ranks.
    More efficient than separate GEMM + all-reduce for bandwidth-limited cases.
    """
    M, K1 = a.shape
    K2, N = b.shape

    # Phase 1: Local GEMM computation
    for m_block, n_block in grid(M, N):
        # Accumulate in float32 for numerical accuracy
        acc = zeros(BLOCK_M, BLOCK_N, dtype=float32)

        # K-dimension reduction loop
        for k_block in tile(K1):
            a_tile = load(a, m_block, k_block)  # [BLOCK_M, BLOCK_K]
            b_tile = load(b, k_block, n_block)  # [BLOCK_K, BLOCK_N]
            acc += dot(a_tile, b_tile)  # FP32 accumulation

        # Store to local symmetric memory (visible to all ranks)
        store_to_rank(symm_mem[my_rank()], m_block, n_block, acc)

    # Phase 2: Cross-device synchronization
    # Ensure all ranks have written their GEMM results
    distributed_barrier(symm_mem, pattern=1)

    # Phase 3: All-reduce across ranks
    for m_block, n_block in grid(M, N):
        result = zeros(BLOCK_M, BLOCK_N, dtype=float32)

        # Accumulate from all ranks
        for rank in static_range(world_size()):
            remote_tile = load_from_rank(symm_mem[rank], m_block, n_block)
            result += remote_tile

        # Store final result (automatic conversion to output dtype)
        store(out, m_block, n_block, result)

# Host wrapper
def gemm_all_reduce_fused_host(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    group: dist.ProcessGroup = dist.group.WORLD,
) -> torch.Tensor:
    """Host-side wrapper for fused GEMM + all-reduce"""
    import torch.distributed._symmetric_memory as symm_mem
    from helion.distributed import DistributedHandles

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"

    # Allocate output and create symmetric memory
    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    symm_mem_hdl = symm_mem.rendezvous(out, group=group)
    helion_handles = DistributedHandles.from_torch(symm_mem_hdl)

    # Launch fused kernel
    gemm_all_reduce_fused(a, b, out, helion_handles)

    return out
```

**Key Improvements**:
- ✅ 60% code reduction (90 → 35 lines)
- ✅ No manual swizzling logic - `grid()` handles tile ordering optimization
- ✅ No manual stride calculations - `load(a, m_block, k_block)` is automatic
- ✅ No manual masking - boundary conditions handled automatically
- ✅ Clean separation of GEMM, sync, and reduce phases
- ✅ Type conversions implicit (FP32 → BF16)

---

### Example 3: RMS Normalization (Element-wise)

#### Kraken (Triton) - 35 lines

```python
# File: kraken/fused/rms_norm.py
@triton.jit
def _rms_norm_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RMS normalization kernel"""
    # Assume input shape [batch, seq, hidden] flattened to [batch*seq, hidden]
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Load row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)

    # Compute variance
    var = tl.sum(x * x, axis=0) / N

    # Normalize
    x_norm = x / tl.sqrt(var + eps)

    # Apply weight
    out = x_norm * weight

    # Store
    tl.store(out_ptr + row_start + offsets, out, mask=mask)

def rms_norm(
    x: torch.Tensor,  # [batch, seq, hidden]
    weight: torch.Tensor,  # [hidden]
    eps: float = 1e-5,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """Host wrapper for RMS norm"""
    batch, seq, hidden = x.shape
    x_flat = x.view(batch * seq, hidden)
    out = torch.empty_like(x_flat)

    grid = (batch * seq,)

    _rms_norm_kernel[grid](
        x_flat,
        out,
        weight,
        eps=eps,
        N=hidden,
        BLOCK_SIZE=triton.next_power_of_2(hidden),
        num_warps=4,
    )

    return out.view(batch, seq, hidden)
```

#### Helion DSL (Proposed) - 20 lines

```python
# File: helion_examples/rms_norm.py
from helion import kernel, Tensor, load, store, reduce, sqrt

@kernel
@config(
    block_sizes={"hidden": [512, 1024, 2048]},
    num_warps=[4, 8],
)
def rms_norm(
    x: Tensor,  # [batch, seq, hidden]
    weight: Tensor,  # [hidden]
    out: Tensor,  # [batch, seq, hidden]
    eps: float = 1e-5,
):
    """RMS normalization: out = x / rms(x) * weight

    Normalizes each [hidden] vector independently across batch and sequence dims.
    """
    batch, seq, hidden = x.shape

    # Process each (batch, seq) position
    for b, s in grid(batch, seq):
        # Load entire hidden dimension for this position
        x_vec = load(x, b, s, :)  # [hidden]

        # Compute RMS (root mean square)
        var = reduce(x_vec * x_vec, op="mean")  # Scalar
        rms = sqrt(var + eps)

        # Normalize
        x_norm = x_vec / rms

        # Apply learned weight
        w = load(weight, :)  # [hidden]
        out_vec = x_norm * w

        # Store result
        store(out, b, s, :, out_vec)

# Host wrapper
def rms_norm_host(
    x: torch.Tensor,  # [batch, seq, hidden]
    weight: torch.Tensor,  # [hidden]
    eps: float = 1e-5,
) -> torch.Tensor:
    """Host-side RMS normalization"""
    out = torch.empty_like(x)
    rms_norm(x, weight, out, eps)
    return out
```

**Key Improvements**:
- ✅ 43% code reduction (35 → 20 lines)
- ✅ Operates on 3D tensor directly (no manual flattening)
- ✅ Subscript `:` to load entire dimension
- ✅ `reduce(..., op="mean")` instead of manual sum + division
- ✅ No manual masking or offset calculations

---

### Example 4: All-Gather + MatMul (Producer-Consumer)

#### Kraken (Triton) - 60 lines

```python
# File: kraken/fused/all_gather_matmul.py
@triton.jit
def _all_gather_matmul_kernel(
    a_buf_tuple,  # Tuple of gathered A buffers from all ranks
    b_ptr,
    c_ptr,
    progress_ptr,  # Progress tensor for synchronization
    m, n, k_per_rank,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """All-gather + matmul with producer-consumer pattern"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Process each rank's data as it arrives
    for rank in range(world_size):
        # Wait for this rank's data to be gathered
        _wait_on_progress(progress_ptr, rank, expected_value=1)

        # Process this rank's K-dimension slice
        for k_idx in range(k_per_rank // BLOCK_SIZE_K):
            # Load A tile from gathered buffer
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_k = rank * k_per_rank + k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_buf_tuple[rank] + offs_am[:, None] * k + offs_k[None, :]
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < m) & (offs_k[None, :] < k))

            # Load B tile
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            b_ptrs = b_ptr + offs_k[:, None] * n + offs_bn[None, :]
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k) & (offs_bn[None, :] < n))

            # Accumulate partial result
            accumulator = tl.dot(a, b, accumulator)

    # Store final result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * n + offs_cn[None, :]
    mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, accumulator, mask=mask)

@triton.jit
def _wait_on_progress(progress_ptr, rank: tl.constexpr, expected_value: tl.constexpr):
    """Spin-wait until progress[rank] == expected_value"""
    while tl.load(progress_ptr + rank) != expected_value:
        pass  # Spin
```

#### Helion DSL (Proposed) - 40 lines

```python
# File: helion_examples/distributed/all_gather_matmul.py
from helion import kernel, Tensor, zeros, load, store, dot, static_range
from helion.distributed import (
    DistributedHandles,
    load_from_rank,
    wait_on_progress,  # NEW API
    world_size,
)

@kernel
@config(
    block_sizes={
        "M": [64, 128],
        "N": [64, 128],
        "K": [32, 64],
    },
    num_warps=[8, 16],
)
def all_gather_matmul(
    a_handles: DistributedHandles,  # All-gathered A: each rank has [M, K_per_rank]
    b: Tensor,  # [K_total, N] where K_total = world_size * K_per_rank
    c: Tensor,  # [M, N] output
    progress: Tensor,  # [world_size] progress tracking
):
    """All-gather + matmul with producer-consumer synchronization

    Computes: c = (all_gather(a)) @ b

    Uses progress tensor to wait for each rank's data before consuming it,
    enabling overlap of communication and computation.
    """
    M, K_total = c.shape[0], b.shape[0]
    N = c.shape[1]
    K_per_rank = K_total // world_size()

    # Process output tiles
    for m_block, n_block in grid(M, N):
        acc = zeros(BLOCK_M, BLOCK_N, dtype=float32)

        # Process each rank's contribution as it arrives
        for rank in static_range(world_size()):
            # Wait for this rank's data to be gathered
            wait_on_progress(progress, rank, expected_value=1)

            # Process this rank's K-dimension slice
            k_start = rank * K_per_rank
            k_end = k_start + K_per_rank

            for k_block in tile(k_start, k_end):
                # Load from gathered A buffer for this rank
                a_tile = load_from_rank(a_handles[rank], m_block, k_block)

                # Load corresponding B tile
                b_tile = load(b, k_block, n_block)

                # Accumulate
                acc += dot(a_tile, b_tile)

        # Store final result
        store(c, m_block, n_block, acc)

# Host wrapper
def all_gather_matmul_host(
    a: torch.Tensor,  # [M, K_per_rank]
    b: torch.Tensor,  # [K_total, N]
    group: dist.ProcessGroup = dist.group.WORLD,
) -> torch.Tensor:
    """Host-side all-gather + matmul"""
    import torch.distributed._symmetric_memory as symm_mem
    from helion.distributed import DistributedHandles

    M, K_per_rank = a.shape
    K_total, N = b.shape
    ws = dist.get_world_size(group)
    assert K_total == ws * K_per_rank, "K dimension mismatch"

    # Create symmetric memory for gathered A
    a_gathered = torch.empty(M, K_total, dtype=a.dtype, device=a.device)
    symm_mem_hdl = symm_mem.rendezvous(a_gathered, group=group)
    helion_handles = DistributedHandles.from_torch(symm_mem_hdl)

    # Progress tensor (set by copy engine)
    progress = torch.zeros(ws, dtype=torch.int32, device=a.device)

    # Launch all-gather (async copy engine)
    # This would be done by a separate copy engine kernel or NCCL

    # Allocate output
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)

    # Launch compute kernel (overlaps with all-gather)
    all_gather_matmul(helion_handles, b, c, progress)

    return c
```

**Key Improvements**:
- ✅ 33% code reduction (60 → 40 lines)
- ✅ `wait_on_progress()` abstracts spin-wait logic
- ✅ `tile(k_start, k_end)` for sliced iteration
- ✅ `load_from_rank()` hides buffer tuple indexing
- ✅ Clean producer-consumer pattern

---

### Example 5: Two-Shot All-Reduce

#### Kraken (Triton) - 60 lines (two kernels)

```python
# File: kraken/comm/two_shot_all_reduce.py
@triton.jit
def _two_shot_all_reduce_copy_in_kernel(
    in_ptr, out_buf_tuple, rank, n_elements,
    BLOCK_SIZE: tl.constexpr, world_size: tl.constexpr
):
    """Phase 1: Copy input to symmetric memory"""
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)

    # Each block processes every world_size-th chunk
    block_idx = pid
    while block_idx < tl.cdiv(n_elements, BLOCK_SIZE):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load from input
        data = tl.load(in_ptr + offsets, mask=mask, other=0.0)

        # Store to local symmetric memory
        tl.store(out_buf_tuple[rank] + offsets, data, mask=mask)

        # Next block (stride by num_blocks)
        block_idx += num_blocks

@triton.jit
def _two_shot_all_reduce_reduce_kernel(
    out_buf_tuple, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, world_size: tl.constexpr
):
    """Phase 2: Reduce from all ranks"""
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)

    block_idx = pid
    while block_idx < tl.cdiv(n_elements, BLOCK_SIZE):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Accumulate from all ranks
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for i in tl.static_range(world_size):
            data = tl.load(out_buf_tuple[i] + offsets, mask=mask, other=0.0)
            acc += data

        # Store result
        tl.store(out_ptr + offsets, acc, mask=mask)

        block_idx += num_blocks

def two_shot_all_reduce(tensor, group, max_num_blocks=24):
    """Two-shot all-reduce with intermediate buffer"""
    # Phase 1: Copy to symm_mem
    _two_shot_all_reduce_copy_in_kernel[grid](...)

    # Sync
    symm_mem_sync(buf_tuple, pattern=1, ...)

    # Phase 2: Reduce
    _two_shot_all_reduce_reduce_kernel[grid](...)

    return tensor
```

#### Helion DSL (Proposed - Option A: Multi-phase) - 30 lines

```python
# File: helion_examples/distributed/two_shot_all_reduce.py
from helion import kernel, Tensor, zeros, load, store, static_range
from helion.distributed import (
    DistributedHandles,
    load_from_rank,
    store_to_rank,
    world_size,
    my_rank,
)

@kernel
@config(
    block_sizes={"elements": [2048, 4096, 8192]},
    num_warps=[16, 32],
    max_num_blocks=[24, 48],  # Persistent kernel
)
def two_shot_all_reduce_phase1(
    x: Tensor,  # Input
    symm_mem: DistributedHandles,
):
    """Phase 1: Copy input to symmetric memory"""
    N = x.shape[0]

    for i in tile(N):
        data = load(x, i)
        store_to_rank(symm_mem[my_rank()], i, data)

@kernel
@config(
    block_sizes={"elements": [2048, 4096, 8192]},
    num_warps=[16, 32],
    max_num_blocks=[24, 48],
)
def two_shot_all_reduce_phase2(
    x: Tensor,  # Output
    symm_mem: DistributedHandles,
):
    """Phase 2: Reduce from all ranks"""
    N = x.shape[0]

    for i in tile(N):
        acc = zeros(BLOCK_SIZE, dtype=float32)

        for rank in static_range(world_size()):
            data = load_from_rank(symm_mem[rank], i)
            acc += data

        store(x, i, acc)

# Host wrapper
def two_shot_all_reduce_host(
    tensor: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
) -> torch.Tensor:
    """Two-shot all-reduce: better load balancing than one-shot"""
    import torch.distributed._symmetric_memory as symm_mem
    from helion.distributed import DistributedHandles

    # Create symmetric memory
    symm_mem_hdl = symm_mem.rendezvous(tensor, group=group)
    helion_handles = DistributedHandles.from_torch(symm_mem_hdl)

    # Phase 1: Copy in
    two_shot_all_reduce_phase1(tensor, helion_handles)

    # Synchronize (host-side)
    torch.cuda.synchronize()

    # Phase 2: Reduce
    two_shot_all_reduce_phase2(tensor, helion_handles)

    return tensor
```

#### Helion DSL (Proposed - Option B: Single kernel) - 20 lines

```python
# Alternative: Single kernel with multiple phases
@kernel
@config(
    block_sizes={"elements": [2048, 4096, 8192]},
    num_warps=[16, 32],
    max_num_blocks=[24, 48],
)
def two_shot_all_reduce(
    x: Tensor,
    symm_mem: DistributedHandles,
):
    """Two-shot all-reduce in single kernel with barrier"""
    N = x.shape[0]

    # Phase 1: Copy to symmetric memory
    for i in tile(N):
        data = load(x, i)
        store_to_rank(symm_mem[my_rank()], i, data)

    # Barrier: ensure all ranks have copied
    distributed_barrier(symm_mem, pattern=1)

    # Phase 2: Reduce
    for i in tile(N):
        acc = zeros(BLOCK_SIZE, dtype=float32)

        for rank in static_range(world_size()):
            data = load_from_rank(symm_mem[rank], i)
            acc += data

        store(x, i, acc)
```

**Key Improvements**:
- ✅ 50-67% code reduction (60 → 30 lines for Option A, 60 → 20 for Option B)
- ✅ No manual persistent kernel loop logic
- ✅ Clean phase separation
- ✅ Option B is cleaner with in-kernel barriers

**Design Question**: Should Helion support:
- **Option A**: Multi-phase requires multiple `@kernel` decorators
- **Option B**: Single kernel with in-kernel barriers

**Recommendation**: Option B is cleaner and matches Kraken's model, but requires in-kernel barrier support.

---

### Example 6: Persistent Kernel with Progress Tracking

#### Kraken (Triton) - 50 lines

```python
# File: kraken/fused/gemm_reduce_scatter_ce_persistent.py
@triton.jit
def _persistent_kernel(
    a_ptr, b_ptr, symm_mem_ptrs, progress_ptr,
    m, n, k, rank, world_size,
    max_num_blocks: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    # ... other params
):
    """Persistent kernel: each block processes multiple tiles"""
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)  # Total blocks launched

    # Calculate total number of tiles
    num_tiles_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(n, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n

    # Each block processes tiles with stride
    tile_idx = pid
    while tile_idx < total_tiles:
        # Decode tile coordinates
        tile_m = tile_idx // num_tiles_n
        tile_n = tile_idx % num_tiles_n

        # Compute GEMM for this tile
        # ... (GEMM computation code)

        # Signal progress for this tile
        if pid == 0:  # Only block 0 signals
            stream_write_value32(progress_ptr, tile_idx, 1)

        # Move to next tile
        tile_idx += num_blocks
```

#### Helion DSL (Proposed) - 30 lines

```python
# File: helion_examples/distributed/persistent_gemm.py
from helion import kernel, Tensor, zeros, load, store, dot
from helion.distributed import signal_progress

@kernel
@config(
    block_sizes={"M": [64, 128], "N": [64, 128], "K": [32, 64]},
    num_warps=[8, 16],
    max_num_blocks=24,  # Launch fewer blocks than tiles
    persistent=True,  # NEW FLAG: Enable persistent kernel mode
)
def persistent_gemm(
    a: Tensor,  # [M, K]
    b: Tensor,  # [K, N]
    c: Tensor,  # [M, N]
    progress: Tensor,  # [num_tiles] - progress tracking
):
    """Persistent GEMM with progress tracking

    Each block processes multiple tiles in a loop, signaling progress
    for producer-consumer synchronization.
    """
    M, K1 = a.shape
    K2, N = b.shape

    # Helion automatically handles persistent tile loop when persistent=True
    for m_block, n_block in grid(M, N):
        acc = zeros(BLOCK_M, BLOCK_N, dtype=float32)

        for k_block in tile(K1):
            a_tile = load(a, m_block, k_block)
            b_tile = load(b, k_block, n_block)
            acc += dot(a_tile, b_tile)

        store(c, m_block, n_block, acc)

        # Signal progress for this tile
        tile_idx = m_block * (N // BLOCK_N) + n_block
        signal_progress(progress, tile_idx, value=1)
```

**Key Improvements**:
- ✅ 40% code reduction (50 → 30 lines)
- ✅ `persistent=True` flag enables persistent mode automatically
- ✅ No manual tile indexing and stride logic
- ✅ `signal_progress()` abstracts progress tracking

**Design Question**: How should Helion express persistent kernels?
- **Option A**: `persistent=True` flag makes `grid()` automatically distribute tiles to blocks with looping
- **Option B**: Explicit `persistent_tile_range()` iterator for manual control

**Recommendation**: Option A for simplicity (shown above), Option B for expert users who need fine control.

---

### Summary: API Coverage

| Kraken Feature | Helion Status | Proposed API |
|---|---|---|
| Multi-rank buffer access | ❌ Missing | `DistributedHandles`, `load_from_rank()`, `store_to_rank()` |
| Cross-device barriers | ❌ Missing | `distributed_barrier()` |
| Rank intrinsics | ❌ Missing | `my_rank()`, `world_size()` |
| Progress tracking | ❌ Missing | `wait_on_progress()`, `signal_progress()` |
| Persistent kernels | ⚠️ Unclear | `persistent=True` config |
| PTX inline assembly | ⚠️ Limited | Extend to `inline_ptx()` |
| GEMM operations | ✅ Supported | `dot()` with mixed precision |
| Element-wise ops | ✅ Supported | Standard arithmetic, `reduce()` |
| Static loops | ✅ Supported | `static_range()` |
| Tile iteration | ✅ Supported | `grid()`, `tile()` |

### Conclusion on Examples

With the proposed distributed APIs, **all major Kraken kernel patterns can be expressed in Helion DSL** with:
- ✅ **30-60% code reduction** across all examples
- ✅ **Better readability** and maintainability
- ✅ **Automatic optimizations** (masking, indexing, autotuning)
- ✅ **Preserved high-level abstractions**

---

## Design Considerations

### Core Design Philosophy

#### Helion's Current Philosophy
- **"PyTorch with tiles"**: High-level abstractions over low-level details
- **Implicit masking**: Automatic boundary handling
- **Automatic indexing**: No manual pointer arithmetic
- **One kernel = one GPU kernel**: Simple mental model
- **Autotuning first**: Configuration space exploration built-in

#### Extending to Distributed
The distributed extensions should:
1. ✅ **Maintain abstraction level**: Hide PTX/low-level synchronization details
2. ✅ **Integrate naturally**: Feel like native Helion, not bolted-on
3. ✅ **Preserve autotuning**: Extend configuration space to distributed parameters
4. ⚠️ **Consider multi-kernel**: May need to relax "one kernel" rule for complex patterns
5. ✅ **Safety by default**: Prevent common distributed pitfalls (deadlocks, races)

---

### Key Design Decisions

#### 1. In-Kernel Barriers vs Multi-Kernel Approach

**Question**: Should distributed barriers be allowed inside kernels, or should multi-phase operations require multiple kernel launches?

**Option A: In-Kernel Barriers**

```python
@kernel
def two_shot_all_reduce(x: Tensor, symm_mem: DistributedHandles):
    # Phase 1: Copy in
    for i in tile(x.shape[0]):
        store_to_rank(symm_mem[my_rank()], i, load(x, i))

    # Barrier inside kernel
    distributed_barrier(symm_mem, pattern=1)

    # Phase 2: Reduce
    for i in tile(x.shape[0]):
        acc = sum([load_from_rank(symm_mem[r], i) for r in range(world_size())])
        store(x, i, acc)
```

**Pros**:
- ✅ Matches Kraken's model closely
- ✅ Single kernel launch from host
- ✅ Potentially lower latency
- ✅ Easier to reason about

**Cons**:
- ❌ More complex compilation
- ❌ CUDA graph compatibility issues
- ❌ Debugging challenges
- ❌ Resource holding wastes GPU cycles

**Option B: Multi-Kernel Approach**

```python
@kernel
def phase1(x: Tensor, symm_mem: DistributedHandles):
    for i in tile(x.shape[0]):
        store_to_rank(symm_mem[my_rank()], i, load(x, i))

@kernel
def phase2(x: Tensor, symm_mem: DistributedHandles):
    for i in tile(x.shape[0]):
        acc = sum([load_from_rank(symm_mem[r], i) for r in range(world_size())])
        store(x, i, acc)
```

**Pros**:
- ✅ Simpler compilation
- ✅ Better CUDA graph support
- ✅ Easier debugging
- ✅ Resources released between kernels

**Cons**:
- ❌ Higher latency (kernel launch overhead)
- ❌ More verbose host code

**Recommendation: Hybrid Approach**

Support both with explicit opt-in for in-kernel barriers:

```python
# Simple cases: in-kernel barriers
@kernel
@allow_distributed_barriers  # Explicit opt-in
def simple_all_reduce(x: Tensor, symm_mem: DistributedHandles):
    # ... single barrier in kernel

# Complex cases: kernel sequence
@kernel_sequence  # NEW DECORATOR
def two_shot_all_reduce(x: Tensor, symm_mem: DistributedHandles):
    @phase
    def copy_in():
        for i in tile(x.shape[0]):
            store_to_rank(symm_mem[my_rank()], i, load(x, i))

    @phase
    def reduce():
        for i in tile(x.shape[0]):
            acc = sum([load_from_rank(symm_mem[r], i) for r in range(world_size())])
            store(x, i, acc)
```

---

#### 2. Persistent Kernels: Automatic vs Explicit

**Question**: How should Helion express persistent kernels where blocks loop over multiple tiles?

**Recommendation: Declarative Persistence**

```python
@kernel
@config(
    execution_mode="persistent",  # vs "regular"
    max_num_blocks=24,
    block_sizes={"M": [64, 128], "N": [64, 128]},
)
def gemm(a: Tensor, b: Tensor, c: Tensor):
    # Same code as regular kernel!
    for m_block, n_block in grid(M, N):
        acc = zeros(BLOCK_M, BLOCK_N)
        for k_block in tile(K):
            acc += dot(load(a, m_block, k_block), load(b, k_block, n_block))
        store(c, m_block, n_block, acc)

    # Helion compiler generates:
    # - If execution_mode="persistent": while loop over tiles per block
    # - If execution_mode="regular": one tile per block
```

**Benefits**:
- ✅ User code unchanged
- ✅ Explicit mode selection via config
- ✅ Autotuner can try both modes
- ✅ Maintains abstraction

---

#### 3. Symmetric Memory Abstraction

**Question**: Should symmetric memory be explicit or implicit?

**Recommendation: Explicit DistributedTensor Type**

```python
# Best of both worlds
@kernel
def my_kernel(x: DistributedTensor):  # NEW TYPE
    # DistributedTensor wraps both tensor and handles
    data = load_from_rank(x, rank, idx)  # x contains both
    local_data = load(x, idx)  # Load from local rank

# Host code
x_dist = helion.DistributedTensor.from_torch(tensor, group=dist.group.WORLD)
my_kernel(x_dist)
```

**Benefits**:
- ✅ Single argument to kernel
- ✅ Type-safe: DistributedTensor vs Tensor
- ✅ Explicit distributed nature
- ✅ Clean API

---

#### 4. World Size: Compile-time vs Runtime

**Question**: Should world_size be compile-time constant or runtime value?

**Recommendation: Hybrid with Specialization**

```python
# Helion generates specialized versions for common world sizes
@kernel
def all_reduce(x: Tensor, symm_mem: DistributedHandles):
    for rank in static_range(world_size()):  # Specialized at compile time
        data = load_from_rank(symm_mem[rank], idx)
        # ...

# Helion compiler:
# - If world_size in [2, 4, 8]: Generate specialized unrolled version
# - Otherwise: Generate runtime loop
# - Can also specialize via @config(world_size=8)
```

**Benefits**:
- ✅ Best performance for common cases
- ✅ Flexible for arbitrary world sizes
- ✅ User doesn't need to think about it
- ✅ Autotuner handles specialization

---

### Safety Features

Distributed kernels are prone to deadlocks and races. Helion should include:

#### 1. Compile-time Checks
```python
# Error: Mismatched barrier patterns
@kernel
def buggy_kernel(x: DistributedTensor):
    if my_rank() == 0:
        distributed_barrier(x, pattern=0)
    else:
        distributed_barrier(x, pattern=1)  # MISMATCH!

# Helion detects at compile time:
# Error: Barrier pattern mismatch detected
#   Rank 0: pattern=0
#   Ranks 1-7: pattern=1
# All ranks must use same pattern.
```

#### 2. Runtime Deadlock Detection
```python
# Runtime monitoring with timeout
@kernel
@config(distributed_timeout_ms=5000)
def my_kernel(x: DistributedTensor):
    distributed_barrier(x, pattern=1)
    # If barrier doesn't complete in 5s, raise error with diagnostics
```

#### 3. Distributed Assertions
```python
@kernel
def my_kernel(x: DistributedTensor):
    # Check consistency across ranks
    distributed_assert(value == expected, "Value mismatch across ranks")
    # Raises if any rank fails, showing which ranks disagree
```

---

### Autotuning Considerations

Extend Helion's autotuning to distributed settings:

#### 1. Algorithm Selection
```python
@kernel
@config(
    # Autotune crossover point
    variants=["one_shot", "two_shot"],
    crossover_size=[200_000, 400_000, 600_000],
)
def auto_all_reduce(x: Tensor, symm_mem: DistributedHandles):
    if x.numel() < crossover_size:
        one_shot_impl(x, symm_mem)
    else:
        two_shot_impl(x, symm_mem)
```

#### 2. Topology Awareness
```python
# Detect hardware topology
topology = detect_topology()  # NVLink, PCIe, multi-node, etc.

# Use topology-specific tuning database
autotuner.set_topology_signature(topology)

# Different optimal configs for different topologies:
# - NVLink: prefer one-shot, larger blocks
# - PCIe: prefer two-shot, smaller blocks
# - Multi-node: prefer hierarchical reduce
```

#### 3. Distributed Profiling
```python
# Profile across all ranks, optimize for worst-case or average
@config(
    optimization_target="average",  # vs "worst_case" or "best_case"
)
def my_kernel(...):
    # Autotuner measures:
    # - Max time across ranks (for worst-case)
    # - Average time (for throughput)
    # - Load imbalance metrics
```

---

### Integration with PyTorch Distributed

Helion should integrate cleanly with PyTorch's distributed ecosystem:

#### 1. Symmetric Memory Integration
```python
import torch.distributed._symmetric_memory as symm_mem

# Create symmetric memory (PyTorch)
tensor = torch.zeros(M, N, device='cuda')
symm_mem_hdl = symm_mem.rendezvous(tensor, group=group)

# Convert to Helion
helion_tensor = helion.DistributedTensor.from_torch(symm_mem_hdl)

# Use in Helion kernel
my_kernel(helion_tensor)
```

#### 2. Compatibility with NCCL
```python
# Mix Helion and NCCL
def hybrid_all_reduce(x: torch.Tensor):
    if x.numel() < 400_000:  # Small tensors
        helion_all_reduce(x)
    else:  # Large tensors
        torch.distributed.all_reduce(x)  # NCCL
```

#### 3. CUDA Graph Support
```python
# Distributed kernels in CUDA graphs
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):
    helion_all_reduce(x)
    y = torch.matmul(x, w)
    helion_custom_op(y)

graph.replay()
```

---

### Error Handling

Provide helpful diagnostics for common distributed errors:

#### 1. Barrier Divergence
```python
# Error: Divergent control flow
@kernel
def divergent_kernel(x: DistributedTensor):
    if my_rank() < 4:
        distributed_barrier(x)  # Only ranks 0-3 reach this

# Helion detects:
# Error: Divergent barrier execution
#   Ranks 0-3 will reach barrier at line 123
#   Ranks 4-7 will not reach barrier
# This will cause deadlock.
```

#### 2. Invalid Rank Access
```python
# Runtime error with context
@kernel
def invalid_access(x: DistributedTensor):
    data = load_from_rank(x, rank=99, idx)  # Invalid!

# Error: Invalid rank 99 in load_from_rank()
#   World size: 8 (valid ranks: 0-7)
#   Location: invalid_access() line 42
```

#### 3. Progress Timeout
```python
# Detect stuck progress waits
@kernel
def producer_consumer(x: DistributedTensor, progress: Tensor):
    wait_on_progress(progress, rank=0)  # Waiting...

# If times out after 5s:
# Error: Progress wait timeout
#   Waiting for progress[0] == 1
#   Current value: 0
#   Possible causes:
#     - Producer kernel not launched
#     - Producer crashed
#     - Race condition in progress signaling
```

---

## Implementation Roadmap

### Phase 1: Core Distributed Primitives (P0)
**Estimated Effort**: 4-6 weeks

**Deliverables**:
- [ ] `DistributedTensor` type and `DistributedHandles` wrapper
- [ ] `load_from_rank()` and `store_to_rank()` operations
- [ ] `my_rank()` and `world_size()` intrinsics
- [ ] `distributed_barrier()` with patterns (0, 1, 2)
- [ ] Integration with `torch.distributed._symmetric_memory`
- [ ] Basic compilation to PTX with atomic synchronization
- [ ] Unit tests for each primitive

**Milestone**: Can express and run simple all-reduce in Helion

---

### Phase 2: Advanced Patterns (P1)
**Estimated Effort**: 3-4 weeks

**Deliverables**:
- [ ] `wait_on_progress()` and `signal_progress()` for producer-consumer
- [ ] Persistent kernel support (`execution_mode="persistent"`)
- [ ] Multi-phase kernel support (`@kernel_sequence`)
- [ ] `static_range(world_size())` specialization
- [ ] Integration tests for GEMM+all-reduce fusion

**Milestone**: Can express all major Kraken patterns

---

### Phase 3: Safety and Debuggability (P1)
**Estimated Effort**: 2-3 weeks

**Deliverables**:
- [ ] Compile-time barrier pattern checking
- [ ] Control flow divergence analysis
- [ ] Runtime deadlock detection (with timeout)
- [ ] `distributed_assert()` for consistency checks
- [ ] Improved error messages for common mistakes
- [ ] Progress monitoring tools

**Milestone**: Production-ready safety features

---

### Phase 4: Autotuning Extension (P1)
**Estimated Effort**: 3-4 weeks

**Deliverables**:
- [ ] Distributed profiling across ranks
- [ ] Algorithm selection (one-shot vs two-shot)
- [ ] Topology detection and signature
- [ ] Topology-aware tuning database
- [ ] Benchmark suite for distributed ops

**Milestone**: Autotuner handles distributed kernels

---

### Phase 5: Performance Optimizations (P2)
**Estimated Effort**: 2-3 weeks

**Deliverables**:
- [ ] Copy engine / stream control (`copy_async()`)
- [ ] Eviction policy parameter for `load()`/`store()`
- [ ] Hierarchical reduce for multi-node
- [ ] CUDA graph optimization
- [ ] Overlapping communication and computation

**Milestone**: Performance competitive with hand-written Triton

---

### Phase 6: Low-Level Escape Hatches (P3)
**Estimated Effort**: 2-3 weeks

**Deliverables**:
- [ ] `inline_ptx()` for custom synchronization
- [ ] Direct access to signal pads
- [ ] Custom barrier implementations
- [ ] Expert mode documentation

**Milestone**: Experts can implement custom patterns

---

**Total Estimated Effort**: 16-24 weeks

---

## Success Metrics and Conclusion

### Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| **Expressiveness** | Express all Kraken patterns | ✅ Achievable with proposed APIs |
| **Code Reduction** | 30-50% vs Triton | ✅ Demonstrated in examples |
| **Performance** | Within 5% of Triton | 🎯 Target (needs validation) |
| **Usability** | New user writes correct kernel in <1hr | 🎯 Target (needs user study) |
| **Safety** | Catch 80%+ bugs at compile time | 🎯 Target (needs implementation) |
| **Adoption** | Production distributed training | 🎯 Long-term goal |

---

### Risks and Mitigations

#### Risk 1: Compilation Complexity
**Issue**: In-kernel barriers require complex IR transformations
**Mitigation**: Start with multi-kernel approach, add in-kernel barriers in Phase 2

#### Risk 2: Performance Gap
**Issue**: Abstraction overhead might hurt performance
**Mitigation**: Extensive benchmarking, escape hatches for experts (`inline_ptx`)

#### Risk 3: CUDA Graph Compatibility
**Issue**: Barriers in CUDA graphs are challenging
**Mitigation**: Document limitations, support multi-graph approach

#### Risk 4: Multi-Backend Support
**Issue**: Symmetric memory is CUDA-specific (currently)
**Mitigation**: Design API to be backend-agnostic, implement CUDA first

#### Risk 5: User Adoption
**Issue**: Users might prefer familiar Triton
**Mitigation**: Excellent documentation, migration guide, compelling examples

---

### Open Questions for Helion Maintainers

1. **Multi-kernel support**: Is there appetite for `@kernel_sequence` decorator?
2. **Persistent kernels**: Does current `grid()` already support persistent pattern?
3. **Type system**: Should we introduce `DistributedTensor` as first-class type?
4. **Compilation model**: How would in-kernel barriers integrate with Triton IR?
5. **Autotuning**: Is distributed autotuning in scope?
6. **Error checking**: How strict should compile-time checks be?
7. **CUDA graphs**: Is full CUDA graph support with barriers required?
8. **Backend support**: Should this work with non-CUDA backends initially?

---

### Conclusion

#### Summary
Adding distributed computing support to Helion DSL is:
- ✅ **Technically Feasible**: All Kraken patterns expressible with proposed APIs
- ✅ **High Value**: 30-50% code reduction, better safety, autotuning
- ✅ **Well-Scoped**: Clear 6-phase roadmap, 16-24 weeks total
- ⚠️ **Non-Trivial**: Requires careful design and implementation

#### Impact
Helion could become the **first high-level DSL for distributed GPU kernels**, making communication-computation fusion accessible to a much broader audience than current Triton-based approaches.

#### Next Steps
1. Review this analysis with Helion maintainers
2. Decide on design approach (in-kernel barriers, persistent kernels, etc.)
3. Prototype P0 APIs (4-6 weeks)
4. Validate performance and usability
5. Proceed with full implementation if successful

---

### Recommendations

#### Immediate Actions
1. **Validate with stakeholders**: Gather feedback on proposed APIs from potential users
2. **Prototype P0 APIs**: Implement core primitives to validate feasibility
3. **Benchmark**: Measure overhead of abstraction vs hand-written Triton
4. **Design review**: Get input from Helion maintainers on design decisions

#### Short-term (3 months)
1. Implement Phase 1 (core primitives)
2. Port 3-5 Kraken kernels to Helion
3. Performance validation on real hardware
4. Early user feedback

#### Medium-term (6 months)
1. Complete Phases 2-4 (patterns, safety, autotuning)
2. Production testing with distributed training workloads
3. Documentation and tutorials
4. Publish results (blog post, paper)

#### Long-term (1 year)
1. Complete Phases 5-6 (performance, expert features)
2. Multi-backend support (AMD, Intel)
3. Integration with popular training frameworks
4. Community building

---

**End of Analysis**
