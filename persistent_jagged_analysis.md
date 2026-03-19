# Persistent Jagged Mode Analysis

## Context

PR #1651 identifies three iteration patterns for jagged kernels:

- **Pattern 1** (Masked Batched): Multiple entities batched, inner loop to
  max(lengths), per-lane mask.
- **Pattern 2** (Maskless Per-Entity): One entity at a time, loop to exact length,
  no outer mask. Pattern 1 degrades to Pattern 2 when outer `block_size=1`.
- **Pattern 3** (Persistent Dynamic): Fixed `num_workers` (= num SMs) stride through
  all tiles with dynamic tile-to-coordinate decoding.

A reviewer asked: *"Helion can already generate persistent kernels during autotuning
when the grid-level loop is expressed using `hl.tile`. If that's the case, is
Pattern 3 not already captured by combining Pattern 1 with persistent kernel
autotuning?"*

## What We Tried

### Attempt 1: `grouped_gemm_jagged_tile` + `pid_type="persistent_interleaved"`

The `grouped_gemm_jagged_tile` kernel uses:
```python
for tile_g in hl.tile(G):                    # grid dimension
    ...
    for tile_m in hl.jagged_tile(M_g):       # inner loop (data-dependent)
        for tile_n in hl.tile(N):            # inner loop
            for tile_k in hl.tile(K):        # reduction loop
```

With `pid_type="persistent_interleaved"`, Helion generates:
```
total_pids = G                              ← only G pids
for virtual_pid in stride(program_id, G, _NUM_SM):
    group = virtual_pid
    for m_tile in jagged_range(M_g):        ← serial on this worker
        for n_tile in range(N):             ← serial on this worker
            for k_tile in range(K):
                dot(...)
```

**Problem:** Only `G` pids to distribute. Workers stride through groups, but each
worker handles ALL M×N tiles for its assigned groups. With G=4 groups on 132 SMs,
most SMs are idle.

### Attempt 2: Move `hl.tile(N)` to grid level

```python
for tile_g, tile_n in hl.tile([G, N]):       # both are grid dimensions
    ...
    for tile_m in hl.jagged_tile(M_g):       # inner loop (data-dependent)
        for tile_k in hl.tile(K):            # reduction loop
```

With `pid_type="persistent_interleaved"`, this generates:
```
total_pids = G × ceil(N/BLOCK_N)            ← G×N_tiles pids
for virtual_pid in stride(program_id, total_pids, _NUM_SM):
    (group, n_tile) = decode(virtual_pid)
    for m_tile in jagged_range(M_g):        ← still serial on this worker
        for k_tile in range(K):
            dot(...)
```

**Improvement:** Now `total_pids = G × ceil(N/BLOCK_N)` (e.g., 4×4 = 16). Workers
stride through (group, n_tile) pairs. This is better, but M tiles within each group
are still serial on one worker.

### The Manual Pattern 3 Kernel (`grouped_gemm_jagged_persistent`)

The manually-written persistent kernel does:
```python
for worker_id in hl.grid(num_workers):       # _NUM_SM programs
    for g in hl.grid(G):                     # every worker visits every group
        num_group_tiles = num_m_tiles × num_n_tiles
        for local_tile in hl.grid(num_group_tiles):
            tile_in_group = local_tile * num_workers + worker_id
            if tile_in_group < num_group_tiles:
                (m_tile, n_tile) = decode(tile_in_group)
                # compute and store
```

**Key insight:** ALL workers collaborate on EACH group. A group with 8 M-tiles × 4
N-tiles = 32 tiles gets those 32 tiles distributed across all 132 SMs.

## Why the Gap Exists

### Comparison

| | Attempt 2 (jagged_tile + persistent) | Manual Pattern 3 |
|---|---|---|
| Grid | `_NUM_SM` workers | `num_workers` (= `_NUM_SM`) |
| Distributed tiles | `G × N_tiles` (e.g., 16) | `Σ_g (M_tiles_g × N_tiles)` (e.g., 80) |
| M tiles | Serial per worker | Shared across all workers |
| Load balance | Good if groups × N_tiles >> num_SMs | Good even with very unbalanced groups |

### Root Cause: `jagged_tile` Cannot Be a Grid Dimension

`hl.jagged_tile(M_g)` is always an inner loop because:

1. **Data-dependent bound:** M_g varies per group. You can't assign a program_id to
   an M tile without knowing which group it belongs to and how many M tiles that
   group has.

2. **No compile-time total:** The total number of M tiles across all groups
   (`Σ_g ceil(M_g/BLOCK_M)`) is only known at runtime. The grid launch size must be
   known at kernel launch time.

3. **Tile-to-coordinate decoding requires a prefix sum:** To map a linear tile index
   to `(group, m_tile_within_group)`, you need the cumulative sum of per-group tile
   counts. The manual kernel computes this dynamically via the inner loops.

`pid_type="persistent_*"` only makes grid dimensions persistent. Since `jagged_tile`
is never a grid dimension, persistent mode cannot distribute jagged tiles across SMs.

### How `total_pids` Is Computed (Code Details)

The persistent loop's `total_pids` is the **product** of per-grid-dimension tile
counts, computed by `ProgramIDs.total_pids_expr()` in `program_id.py:80-83`:

```python
def total_pids_expr(self, *, is_device: bool) -> str:
    return " * ".join(
        f"({pid.num_pids_expr(is_device=is_device)})" for pid in self.pid_info
    )
```

Each `PIDInfo` contributes `cdiv(numel, block_size)`. Only grid-level dimensions
produce `PIDInfo` entries — inner loops (including `jagged_tile`) do not participate.

The persistent loop is then wrapped around the entire body by
`PersistentProgramIDs.setup_persistent_kernel()` (`program_id.py:660-683`), which
calls `_setup_persistent_kernel_and_wrap_body()` to emit:

```python
for virtual_pid in tl.range(tl.program_id(0), total_pids, _NUM_SM):
    # decompose virtual_pid into per-dimension PIDs
    # ... original body with inner loops intact ...
```

The inner loops (jagged_tile, reduction tiles) remain nested inside and execute
serially on whichever SM owns that virtual_pid.

## Bridging the Gap

### Approach 1: Precomputed Work-Item Tensors (No Compiler Changes)

The simplest approach requires **zero compiler changes** — the user precomputes a
work-item table on the host and passes it to a standard Helion kernel:

```python
# Host side: enumerate all (group, m_offset, n_offset) tuples
tiles = []
for g in range(G):
    M_g = offsets[g+1] - offsets[g]
    for m in range(cdiv(M_g, BLOCK_M)):
        for n in range(cdiv(N, BLOCK_N)):
            tiles.append((g, m * BLOCK_M, n * BLOCK_N))
tile_groups = torch.tensor([t[0] for t in tiles], device=device)
tile_m_offsets = torch.tensor([t[1] for t in tiles], device=device)
tile_n_offsets = torch.tensor([t[2] for t in tiles], device=device)
total_tiles = len(tiles)
```

```python
@helion.kernel(static_shapes=False)
def grouped_gemm_precomputed(
    A_packed, B, group_offsets,
    tile_groups, tile_m_offsets, tile_n_offsets,
    total_tiles: int,
):
    num_workers = torch.cuda.get_device_properties(...).multi_processor_count
    BLOCK_M = hl.register_block_size(32, 128)
    BLOCK_N = hl.register_block_size(32, 128)
    ...
    out = torch.zeros(total_M, N, ...)

    for worker_id in hl.grid(num_workers):
        for tile_idx in hl.grid(total_tiles):
            linear_idx = tile_idx * num_workers + worker_id
            if linear_idx < total_tiles:
                g = tile_groups[linear_idx]
                m_off = tile_m_offsets[linear_idx]
                n_off = tile_n_offsets[linear_idx]
                group_start = group_offsets[g]
                row_idx = group_start + m_off + hl.arange(BLOCK_M)
                col_idx = n_off + hl.arange(BLOCK_N)
                # ... standard hl.load + dot + hl.store with masks
```

**Pros:** Works today, no framework changes, conceptually simple.
**Cons:** Requires O(total_tiles) host memory and host-side Python loop. The kernel
code is verbose — similar to the existing `grouped_gemm_jagged_persistent`.

### Approach 2: Prefix-Sum Based (Minimal Host Memory)

Instead of enumerating all tiles, pass only a prefix sum (size G+1):

```python
# Host side:
tiles_per_group = []
for g in range(G):
    M_g = offsets[g+1] - offsets[g]
    tiles_per_group.append(cdiv(M_g, BLOCK_M) * cdiv(N, BLOCK_N))
group_tile_prefix = torch.tensor([0] + list(accumulate(tiles_per_group)), device=device)
total_tiles = int(group_tile_prefix[-1].item())
```

The kernel uses a linear scan over the prefix sum to find each tile's group:

```python
for worker_id in hl.grid(num_workers):
    for tile_idx in hl.grid(total_tiles):
        linear_idx = tile_idx * num_workers + worker_id
        if linear_idx < total_tiles:
            # Linear scan to find group
            for g in hl.grid(G):
                if group_tile_prefix[g + 1] > linear_idx:
                    local_tile = linear_idx - group_tile_prefix[g]
                    num_m_tiles_g = ...  # recompute from group_offsets
                    m_tile_idx = local_tile % num_m_tiles_g
                    n_tile_idx = local_tile // num_m_tiles_g
                    # ... body
                    break
```

**Pros:** O(G) memory instead of O(total_tiles). Closer to manual Pattern 3.
**Cons:** Still verbose. The linear scan is O(G) per tile (a binary search would be
O(log G) but Helion doesn't support `tl.where`-based binary search easily).

### Approach 3: Compiler-Level `pid_type="persistent_jagged"` (Cleanest UX)

The cleanest approach: the user writes the **same simple DSL code** and a new
`pid_type` handles the flattening automatically.

**User code (unchanged from Pattern 1/2):**
```python
for tile_g in hl.tile(G):
    starts = group_offsets[tile_g]
    ends = group_offsets[tile_g.index + 1]
    M_g = ends - starts
    for tile_m in hl.jagged_tile(M_g):
        for tile_n in hl.tile(N):
            acc = hl.zeros([tile_g, tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(K):
                # ... matmul body
```

**Config:** `pid_type="persistent_jagged"`

**What the compiler generates:**

Host side (injected into the launcher):
```python
# Compute prefix sum of per-group tile counts
_group_tile_prefix = torch.empty(G + 1, dtype=torch.int32, device=device)
_group_tile_prefix[0] = 0
for g in range(G):
    M_g = group_offsets[g+1] - group_offsets[g]
    _group_tile_prefix[g+1] = _group_tile_prefix[g] + cdiv(M_g, _BLOCK_SIZE_M) * cdiv(N, _BLOCK_SIZE_N)
_total_tiles = int(_group_tile_prefix[G].item())
```

Device side:
```python
total_tiles = tl.load(total_tiles_ptr)
for virtual_pid in tl.range(tl.program_id(0), total_tiles, _NUM_SM):
    # Linear scan to find group (O(G) per tile)
    group = 0
    for g in range(G):
        next_prefix = tl.load(group_tile_prefix_ptr + g + 1)
        if next_prefix > virtual_pid:
            group = g
            break
    group_start_tiles = tl.load(group_tile_prefix_ptr + group)
    local_tile = virtual_pid - group_start_tiles
    group_offset_start = tl.load(group_offsets + group)
    group_offset_end = tl.load(group_offsets + group + 1)
    M_g = group_offset_end - group_offset_start
    num_m_tiles = cdiv(M_g, _BLOCK_SIZE_M)
    m_tile_idx = local_tile % num_m_tiles
    n_tile_idx = local_tile // num_m_tiles

    m_offset = m_tile_idx * _BLOCK_SIZE_M
    n_offset = n_tile_idx * _BLOCK_SIZE_N

    # ... body using (group, m_offset, n_offset) coordinates
```

### Compiler Implementation Plan for Approach 3

**Files that need changes:**

| File | Change |
|------|--------|
| `helion/runtime/config.py` | Add `"persistent_jagged"` to `PidTypeLiteral` |
| `helion/_compiler/program_id.py` | New `PersistentJaggedProgramIDs` class |
| `helion/_compiler/tile_strategy.py` | Detect `tile → jagged_tile → tile` pattern; inject host-side prefix sum; select new PID strategy |
| `helion/_compiler/compile_environment.py` | Track which block_ids form a "jagged group" (parent grid dim + jagged dim + sibling tile dims) |
| `helion/_compiler/host_function.py` | Support injecting host-side tensor computation (prefix sum) before kernel launch |

**Key implementation steps:**

1. **Pattern detection** (in `tile_strategy.py`):
   When `pid_type="persistent_jagged"` is selected, verify the loop structure is
   `tile(G) → jagged_tile(M_g) → tile(N)` (with optional additional inner tiles for
   reduction). Record which block_ids are the "group" dim, "jagged" dim, and
   "regular" dim.

2. **Host-side prefix sum injection** (in `host_function.py`):
   Before the kernel launch call, emit Python code that:
   - Reads `group_offsets` to compute per-group tile counts
   - Builds a prefix-sum tensor `_group_tile_prefix`
   - Computes `_total_tiles`
   - Passes both as extra kernel arguments

   This is novel — Helion currently doesn't inject host-side tensor computation
   that depends on runtime tensor values. The closest existing mechanism is
   `get_num_sm()`, which reads device properties.

3. **`PersistentJaggedProgramIDs` class** (in `program_id.py`):
   - `total_pids_expr()` returns `_total_tiles` (a kernel argument, not a product
     of grid dimensions)
   - `codegen()` emits the linear-scan decode loop instead of the standard
     `virtual_pid % num_blocks` decomposition
   - `codegen_grid()` returns `(_NUM_SM,)` as usual for persistent kernels
   - `setup_persistent_kernel()` wraps the body in the persistent loop with decode

4. **Body transformation** (in `generate_ast.py` or `tile_strategy.py`):
   The nested `tile(G) → jagged_tile(M_g) → tile(N)` loops must be replaced by a
   single flat loop. The offset/index variables (`offset_0`, `indices_0`, etc.)
   that the body references need to be wired to the decoded coordinates instead of
   loop induction variables.

**Estimated complexity:**

| Component | Difficulty | Notes |
|-----------|-----------|-------|
| Config + strategy selection | Easy | Add string literal, add `if` branch |
| Host-side prefix sum | Medium | New capability: injecting runtime tensor computation. Need to handle the dependency on `_BLOCK_SIZE_*` constexprs |
| PID class + decode codegen | Medium | The linear-scan decode is straightforward to emit as Triton AST |
| Nested loop → flat loop transform | Hard | Must replace 3 nested loops with 1 flat loop while preserving all variable bindings that the body uses (`starts`, `ends`, `M_g`, `row_indices`, etc.) |
| Autotuning interaction | Low | Prefix sum recomputed per config automatically since block sizes change |

The hardest part is the **loop flattening transformation**: the body of the original
3-level loop nest references variables (`starts`, `tile_m.index`, `tile_n.index`)
that are set up by each loop level's codegen. The flat loop must reconstruct all
these values from the decoded (group, m_tile, n_tile) coordinates.

## Conclusion

**The original analysis is correct:** the gap exists because `jagged_tile` cannot
participate in `total_pids`, so persistent mode only distributes the outer grid
dimensions. The manual Pattern 3 kernel achieves finer-grained distribution by
computing tile assignments dynamically.

**Three approaches can bridge the gap:**

1. **Precomputed work-item tensors** — works today, no compiler changes, but verbose
   kernel code and O(total_tiles) host memory.

2. **Prefix-sum based** — O(G) memory, closer to manual Pattern 3, but still
   requires verbose kernel code.

3. **Compiler-level `pid_type="persistent_jagged"`** — cleanest UX (user writes
   the same simple code), but requires significant compiler work, especially for
   host-side prefix sum injection and nested-to-flat loop transformation.

**Recommendation:** Start with Approach 1 or 2 as a user-space pattern to validate
the performance benefit. If the benefit is confirmed, invest in Approach 3 for the
clean compiler integration. The manual `grouped_gemm_jagged_persistent` already
serves as a proof-of-concept for Approach 2's runtime behavior.
