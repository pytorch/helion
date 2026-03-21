# Helion Matmul: Standard vs Persistent — Same Kernel, Different Config

Triton tutorials [03-matrix-multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
and [09-persistent-matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
map to the **same Helion kernel**. The only difference is the `pid_type` config parameter
(plus `l2_groupings` and `range_flattens`). The autotuner searches over these automatically.

## Helion Kernel (identical for both)

```python
@helion.kernel(config=config)
def matmul(x: Tensor, y: Tensor) -> Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out
```

---

## Config A: Standard Matmul (Tutorial 03)

```python
config = helion.Config(
    block_sizes=[128, 128, 64],
    num_warps=8,
    pid_type="flat",
)
```

### Generated Triton Code

Grid: `(M_tiles * N_tiles,)` — one thread block per output tile.

```python
_BLOCK_SIZE_0 = tl.constexpr(128)
_BLOCK_SIZE_1 = tl.constexpr(128)
_BLOCK_SIZE_2 = tl.constexpr(64)

@triton.jit
def _helion_matmul_standard(x, y, out):
    num_blocks_0 = tl.cdiv(1024, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in tl.range(0, 1024, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        acc_copy = acc
        acc_copy_0 = acc_copy
        load = tl.load(x + (indices_0[:, None] * 1024 + indices_2[None, :] * 1), None)
        load_1 = tl.load(y + (indices_2[:, None] * 1024 + indices_1[None, :] * 1), None)
        acc = tl.dot(tl.cast(load, tl.float16), tl.cast(load_1, tl.float16),
                     acc=acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
    v_0 = tl.cast(acc, tl.float16)
    tl.store(out + (indices_0[:, None] * 1024 + indices_1[None, :] * 1), v_0, None)

# Host launcher:
_launcher(_helion_matmul_standard,
          (triton.cdiv(1024, 128) * triton.cdiv(1024, 128),),  # grid = (64,)
          x, y, out, num_warps=8, num_stages=1)
```

---

## Config B: Persistent Matmul (Tutorial 09)

```python
config = helion.Config(
    block_sizes=[128, 128, 64],
    num_warps=8,
    pid_type="persistent_interleaved",
    l2_groupings=[8],
    range_flattens=[True],
)
```

### Generated Triton Code

Grid: `(_NUM_SM,)` — one thread block per SM, each loops over multiple tiles.

```python
_BLOCK_SIZE_0 = tl.constexpr(128)
_BLOCK_SIZE_1 = tl.constexpr(128)
_BLOCK_SIZE_2 = tl.constexpr(64)

@triton.jit
def _helion_matmul_persistent(x, y, out, _NUM_SM: tl.constexpr):
    total_pids = tl.cdiv(1024, _BLOCK_SIZE_0) * tl.cdiv(1024, _BLOCK_SIZE_1)
    for virtual_pid in tl.range(tl.program_id(0), total_pids, _NUM_SM, flatten=True):
        num_pid_m = tl.cdiv(1024, _BLOCK_SIZE_0)
        num_pid_n = tl.cdiv(1024, _BLOCK_SIZE_1)
        inner_2d_pid = virtual_pid
        num_pid_in_group = 8 * num_pid_n
        group_id = inner_2d_pid // num_pid_in_group
        first_pid_m = group_id * 8
        group_size_m = min(num_pid_m - first_pid_m, 8)
        pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
        pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
        offset_0 = pid_0 * _BLOCK_SIZE_0
        indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
        offset_1 = pid_1 * _BLOCK_SIZE_1
        indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
        acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
        for offset_2 in tl.range(0, 1024, _BLOCK_SIZE_2):
            indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            acc_copy = acc
            acc_copy_0 = acc_copy
            load = tl.load(x + (indices_0[:, None] * 1024 + indices_2[None, :] * 1), None)
            load_1 = tl.load(y + (indices_2[:, None] * 1024 + indices_1[None, :] * 1), None)
            acc = tl.dot(tl.cast(load, tl.float16), tl.cast(load_1, tl.float16),
                         acc=acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
        v_0 = tl.cast(acc, tl.float16)
        tl.store(out + (indices_0[:, None] * 1024 + indices_1[None, :] * 1), v_0, None)

# Host launcher:
_NUM_SM = helion.runtime.get_num_sm(x.device)
_launcher(_helion_matmul_persistent,
          (_NUM_SM,),  # grid = (num_SMs,)
          x, y, out, _NUM_SM, num_warps=8, num_stages=1)
```

---

## Helion `persistent_matmul.py` Example (Manual Persistent Pattern)

`examples/persistent_matmul.py` takes a different approach — it manually implements the persistent
pattern using low-level Helion APIs (`hl.grid`, `hl.register_block_size`, `hl.arange`, `hl.load`,
`hl.store`, `hl.inline_triton`) to closely match the Triton tutorial's exact code patterns.

### Helion Kernel

```python
config = helion.Config(
    block_sizes=[128, 128, 64],
    num_warps=8,
    range_flattens=[None, True, None],
)

@helion.kernel(static_shapes=False, config=config)
def persistent_matmul(x: Tensor, y: Tensor) -> Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2

    num_workers = torch.cuda.get_device_properties(x.device.index).multi_processor_count
    BLOCK_M = hl.register_block_size(32, 128)
    BLOCK_N = hl.register_block_size(32, 128)

    out = torch.zeros([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    num_m_tiles = (m + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (n + BLOCK_N - 1) // BLOCK_N
    num_tiles = num_m_tiles * num_n_tiles
    GROUP_SIZE_M = 8
    num_pid_in_group = GROUP_SIZE_M * num_n_tiles

    for worker_id in hl.grid(num_workers):
        for tile_idx in hl.grid(worker_id, num_tiles, step=num_workers):
            group_id = tile_idx // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
            m_tile_idx = first_pid_m + ((tile_idx % num_pid_in_group) % group_size_m)
            n_tile_idx = (tile_idx % num_pid_in_group) // group_size_m

            base_row = m_tile_idx * BLOCK_M
            base_col = n_tile_idx * BLOCK_N
            row_idx = base_row + hl.arange(BLOCK_M)
            col_idx = base_col + hl.arange(BLOCK_N)

            rows_valid = row_idx < m
            cols_valid = col_idx < n
            row_idx = torch.where(rows_valid, row_idx, 0)
            col_idx = torch.where(cols_valid, col_idx, 0)

            row_idx = hl.inline_triton(
                "tl.max_contiguous(tl.multiple_of({0}, {1}), {1})",
                args=[row_idx, BLOCK_M], output_like=row_idx,
            )
            col_idx = hl.inline_triton(
                "tl.max_contiguous(tl.multiple_of({0}, {1}), {1})",
                args=[col_idx, BLOCK_N], output_like=col_idx,
            )

            acc = hl.zeros([BLOCK_M, BLOCK_N], dtype=torch.float32)
            for k_tile in hl.tile(k):
                k_idx = k_tile.index
                a_blk = hl.load(x, [row_idx, k_idx])
                b_blk = hl.load(y, [k_idx, col_idx])
                acc = torch.addmm(acc, a_blk, b_blk)

            valid_2d = rows_valid[:, None] & cols_valid[None, :]
            hl.store(out, [row_idx, col_idx], acc.to(out.dtype), extra_mask=valid_2d)
    return out
```

### Generated Triton Code

Grid: `(num_workers,)` — one thread block per SM, with manual swizzle and index clamping.

```python
_BLOCK_SIZE_1 = tl.constexpr(128)  # BLOCK_N
_BLOCK_SIZE_0 = tl.constexpr(128)  # BLOCK_M
_BLOCK_SIZE_4 = tl.constexpr(64)   # BLOCK_K

@triton.jit
def _helion_persistent_matmul(x, y, out,
        out_stride_0, out_stride_1,
        x_stride_0, x_stride_1, y_stride_0, y_stride_1,
        num_tiles, n, m, k,
        _BLOCK_SIZE_3: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_2 = pid_0
    for offset_3 in tl.range(tl.cast(offset_2, tl.int32),
                             tl.cast(num_tiles, tl.int32),
                             _BLOCK_SIZE_3, flatten=True):
        # L2 swizzle (GROUP_SIZE_M = 8) — expanded inline by Helion's compiler
        # (uses triton_helpers.div_floor_integer / remainder_integer)
        floordiv = triton_helpers.div_floor_integer(
            offset_3, 8 * triton_helpers.div_floor_integer(
                -1 + _BLOCK_SIZE_1 + n, _BLOCK_SIZE_1))
        # ... swizzle math computing mul_1 (base_row) and mul_2 (base_col) ...

        # Row/col index generation
        iota = tl.arange(0, _BLOCK_SIZE_0)
        v_1 = iota + tl.cast(mul_1, tl.int32)        # row_idx
        iota_1 = tl.arange(0, _BLOCK_SIZE_1)
        v_3 = iota_1 + tl.cast(mul_2, tl.int32)      # col_idx

        # Bounds checking + index clamping
        v_5 = v_1 < tl.cast(m, tl.int32)              # rows_valid
        v_7 = v_3 < tl.cast(n, tl.int32)              # cols_valid
        v_9 = tl.where(v_5, v_1, 0)                   # clamp OOB rows to 0
        v_11 = tl.where(v_7, v_3, 0)                  # clamp OOB cols to 0

        # Contiguity/alignment hints
        inline_triton_result = tl.max_contiguous(
            tl.multiple_of(v_9, _BLOCK_SIZE_0), _BLOCK_SIZE_0)
        inline_triton_result_1 = tl.max_contiguous(
            tl.multiple_of(v_11, _BLOCK_SIZE_1), _BLOCK_SIZE_1)

        # Accumulator init
        acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)

        # K-loop with masks (dynamic shapes)
        for offset_4 in tl.range(0, tl.cast(k, tl.int32), _BLOCK_SIZE_4):
            indices_4 = offset_4 + tl.arange(0, _BLOCK_SIZE_4).to(tl.int32)
            mask_4 = indices_4 < k
            a_blk = tl.load(
                x + (inline_triton_result[:, None] * x_stride_0
                     + indices_4[None, :] * x_stride_1),
                mask_4[None, :], other=0)
            b_blk = tl.load(
                y + (indices_4[:, None] * y_stride_0
                     + inline_triton_result_1[None, :] * y_stride_1),
                mask_4[:, None], other=0)
            acc = tl.dot(tl.cast(a_blk, tl.float16), tl.cast(b_blk, tl.float16),
                         acc=acc, input_precision='tf32', out_dtype=tl.float32)

        # Masked store
        v_12 = v_5[:, None] & v_7[None, :]
        v_13 = tl.cast(acc, tl.float16)
        tl.store(out + (inline_triton_result[:, None] * out_stride_0
                        + inline_triton_result_1[None, :] * out_stride_1),
                 v_13, v_12)

# Host launcher:
_launcher(_helion_persistent_matmul, (num_workers,),
          x, y, out, out.stride(0), out.stride(1),
          x.stride(0), x.stride(1), y.stride(0), y.stride(1),
          num_tiles, n, m, k, num_workers,
          num_warps=8, num_stages=1)
```

---

## Triton Tutorial 09 Kernel (Original)

Reference: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html

```python
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def triton_persistent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)

# Host launcher:
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
grid = (min(NUM_SMS, triton.cdiv(M, 128) * triton.cdiv(N, 128)),)
triton_persistent_matmul_kernel[grid](
    a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
    GROUP_SIZE_M=8, NUM_SMS=NUM_SMS,
)
```

---

## Structural Comparison

| Aspect | Helion Standard (Config A) | Helion Persistent (Config B) | Helion `persistent_matmul.py` | Triton Tutorial 09 |
|--------|---------------------------|-----------------------------|-----------------------------|-------------------|
| Helion kernel | `hl.tile([m, n])` | `hl.tile([m, n])` | `hl.grid` + `hl.arange` + `hl.load`/`hl.store` | N/A (hand-written Triton) |
| Grid size | `M_tiles * N_tiles` | `_NUM_SM` | `num_workers` (= `_NUM_SM`) | `min(NUM_SMS, M_tiles * N_tiles)` |
| Outer loop | None | `for virtual_pid in tl.range(pid, total, _NUM_SM, flatten=True)` | `for offset_3 in tl.range(pid, num_tiles, num_workers, flatten=True)` | `for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True)` |
| PID decomposition | `program_id(0) % / //` | `virtual_pid` with L2 swizzle | Inline swizzle via `triton_helpers` | `_compute_pid()` with L2 swizzle |
| L2 grouping | None | `GROUP_SIZE = 8` via `l2_groupings=[8]` | `GROUP_SIZE_M = 8` (manual) | `GROUP_SIZE_M = 8` |
| Index clamping | None (static shapes) | None (static shapes) | `tl.where(v < m, v, 0)` | `tl.where(offs < M, offs, 0)` |
| Contiguity hints | None | None | `tl.max_contiguous(tl.multiple_of(...))` | `tl.max_contiguous(tl.multiple_of(...))` |
| `tile_id_c` workaround | None | None | None | Yes (Blackwell pipelining bug) |
| K-loop mask | None (static shapes) | None (static shapes) | `indices_4 < k` (dynamic shapes) | `offs_k < K - ki * BLOCK_SIZE_K` |
| Strides | Hardcoded (static shapes) | Hardcoded (static shapes) | Runtime args (`x_stride_0`, ...) | Runtime args (`stride_am`, ...) |
| Store mask | None (static shapes) | None (static shapes) | `rows_valid[:, None] & cols_valid[None, :]` | `(offs_cm[:, None] < M) & (offs_cn[None, :] < N)` |

## Key Takeaway

With Helion, you write **one kernel** and autotune through different PID strategies. The autotuner
searches over `pid_type ∈ {flat, xyz, persistent_blocked, persistent_interleaved}` automatically,
discovering whether persistent scheduling outperforms standard scheduling for a given problem size
and hardware — no kernel code changes needed.

The `examples/persistent_matmul.py` example demonstrates an alternative approach using low-level
Helion APIs to match the Triton tutorial's exact patterns (index clamping, contiguity hints,
dynamic shapes). This gives full control but requires more code. For most use cases, Config B
(same `hl.tile` kernel + `pid_type="persistent_interleaved"`) is sufficient.
