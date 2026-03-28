# Tests That Would Trigger CUDA IMA Without Safeguards

The safeguard logic lives in `helion/_compiler/tile_strategy.py::TileStrategy.get_tl_range_kwargs`.
It has three protection paths. Below are all tests that exercise them, plus the GitHub
issues/PRs that motivated each safeguard.

---

## Path 1: `tensor_descriptor` + pipeline/unroll

**Safeguard:** Unconditionally kills `range_num_stages` and kills unroll when `num_stages > 1`.

**Triton issue:** Tensor descriptor + multi-stage pipelines cause "misaligned address" or
"unspecified launch failure".

**History:**
- Issue [#755](https://github.com/pytorch/helion/issues/755): Misaligned address in JSD kernel
- PR [#792](https://github.com/pytorch/helion/pull/792): Set range_num_stages <= 1 if using tensor_descriptor
- PR [#994](https://github.com/pytorch/helion/pull/994): Remove unrolling with tma + pipelining (extended to also kill unroll)

### `test_tensor_descriptor.py::TestTensorDescriptor::test_multistage_range_tensor_descriptor` (line 199)

```
indexing="tensor_descriptor", num_stages=4, range_num_stages=[0, 4], range_unroll_factors=[0, 0]
```

JSD forward kernel. Without safeguard: `range_num_stages=4` on the inner loop would pass
through to Triton, causing misaligned address errors with tensor descriptor indexing.
Test asserts that `num_stages` does NOT appear in any `tl.range` call (clamped to 0).

### `test_tensor_descriptor.py::TestTensorDescriptor::test_tiny_matmul_tile_fallback` (line 314)

```
indexing="tensor_descriptor", num_stages=4, range_num_stages=[0, 1], range_unroll_factors=[0, 4]
```

64x64 matmul with tiny tile (block_sizes=[1, 16, 16]). Without safeguard:
`range_unroll_factors=4` combined with `num_stages=4` on tensor_descriptor would cause
misaligned address. Safeguard zeros the unroll factor because `num_stages > 1`.

---

## Path 2: `unroll > 1` + short loop → cap unroll factor and pipeline depth

**Safeguard:** Halves unroll factor until it divides trip count; caps pipeline depth when
`unrolled_iters <= pipeline_depth`.

**Triton issue:** Unrolled loop body reads beyond the loop bound when trip count is short
relative to unroll * pipeline depth, causing OOB memory access.

**History:**
- Issue [#904](https://github.com/pytorch/helion/issues/904): Benchmark CI misaligned address on gemm kernel
  (256x256 fp16 matmul, autotuner picked block_ptr + unroll + pipeline config)
- PR [#920](https://github.com/pytorch/helion/pull/920): Fix CUDA IMA from combination of unrolling + pipelining
  (original safeguard, only capped pipeline depth based on remainder)
- Commit ddd49e3b: Auto-cap pipeline depth and unroll factor (current version, also caps
  unroll factor and handles multi-dimensional loops)

### `test_loops.py::TestLoops::test_high_pipeline_depth_with_short_loops` (line 1395)

```
num_stages=7
range_unroll_factors=[0, 0, 0, 2, 0, 3, 4]
range_num_stages=[0, 3, 1, 0, 2, 3, 1]
block_sizes=[64, 32, 64, 128, 64, 32]
indexing=["pointer"] * 7
```

Kernel: `transposed_matmul_bwd` with short reduction dims (K=48, M=128, N=192).
Without safeguard: e.g. inner loop with trip_count=2, unroll=4, pipeline=3 → Triton
generates reads 4x past the loop bound with 3-deep pipeline prefetch → CUDA IMA.

---

## Path 3: `block_ptr` + unroll + pipeline → kill pipeline

**Safeguard:** Forces `range_num_stages=1` when `block_ptr` indexing is used with
`unroll > 1` and pipeline > 1.

**Triton issue:** `block_ptr` + unroll + multi-stage pipeline generates misaligned
address accesses (known Triton bug, independent of loop length).

**History:**
- Issue [#904](https://github.com/pytorch/helion/issues/904): gemm benchmark with block_ptr
- Issue [#908](https://github.com/pytorch/helion/issues/908): grouped_gemm benchmark, same class of error
- PR [#920](https://github.com/pytorch/helion/pull/920): Original fix (capped pipeline depth)
- Commit ddd49e3b: Added explicit block_ptr + unroll + pipeline → force num_stages=1

### `test_loops.py::TestLoops::test_unroll_with_pipelining` (line 1353)

```
indexing="block_ptr"
range_num_stages=[4, 2]
range_unroll_factors=[4, 4]
pid_type="persistent_blocked"
```

Kernel: 256x256 bf16 matmul, reduction dim K=256 with block_size=16 → trip_count=16.
Without safeguard: block_ptr + unroll=4 + pipeline=2 → misaligned address.
Test explicitly asserts `num_stages=1` in generated code.

---

## Other IMA-related test (different mechanism)

### `test_indexing.py::TestIndexing::test_large_tensor` (line 750)

```
@skipIfRefEager("Test checks for no IMA")
autotune_effort="none"
```

Tests that large tensor (>2GB) indexing doesn't produce OOB accesses. This is a
32-bit pointer arithmetic overflow IMA (not pipeline/unroll), but still exercises IMA
prevention. Related to issues [#686](https://github.com/pytorch/helion/issues/686)
and [#451](https://github.com/pytorch/helion/issues/451).

---

## Autotuner IMA defense (separate mechanism)

The autotuner in `helion/autotuner/logger.py:473` also catches `"illegal memory access"`
and `"misaligned address"` at runtime and skips those configs. This is a reactive defense
(catch-and-skip), while `get_tl_range_kwargs` is a proactive defense (prevent before
compilation). Both are needed: the proactive defense avoids wasting autotuning time on
configs that are known to be bad.

---

## Reproduction commands

```bash
# All IMA safeguard tests (proactive defense):
pytest test/test_loops.py::TestLoops::test_unroll_with_pipelining -x -vv -s
pytest test/test_loops.py::TestLoops::test_high_pipeline_depth_with_short_loops -x -vv -s
pytest test/test_tensor_descriptor.py::TestTensorDescriptor::test_multistage_range_tensor_descriptor -x -vv -s
pytest test/test_tensor_descriptor.py::TestTensorDescriptor::test_tiny_matmul_tile_fallback -x -vv -s
pytest test/test_indexing.py::TestIndexing::test_large_tensor -x -vv -s

# Run all at once:
pytest \
  test/test_loops.py::TestLoops::test_unroll_with_pipelining \
  test/test_loops.py::TestLoops::test_high_pipeline_depth_with_short_loops \
  test/test_tensor_descriptor.py::TestTensorDescriptor::test_multistage_range_tensor_descriptor \
  test/test_tensor_descriptor.py::TestTensorDescriptor::test_tiny_matmul_tile_fallback \
  test/test_indexing.py::TestIndexing::test_large_tensor \
  -x -vv -s
```

## GitHub issues timeline

| Issue | Title | Root cause | Fix |
|-------|-------|------------|-----|
| [#451](https://github.com/pytorch/helion/issues/451) | IMA on add kernel (large inputs) | 32-bit pointer overflow | Separate fix (not in get_tl_range_kwargs) |
| [#686](https://github.com/pytorch/helion/issues/686) | IMA on default config of add kernel | 32-bit pointer overflow on 51200x51200 | Same as #451 |
| [#755](https://github.com/pytorch/helion/issues/755) | Misaligned address in JSD kernel | tensor_descriptor + range_num_stages | PR #792 |
| [#904](https://github.com/pytorch/helion/issues/904) | Misaligned address on gemm benchmark | block_ptr + unroll + pipeline | PR #920 |
| [#908](https://github.com/pytorch/helion/issues/908) | Misaligned address on grouped_gemm | Same class as #904 | PR #920 |
