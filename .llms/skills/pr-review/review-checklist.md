# Helion PR Review Checklist

This checklist covers areas that reviewers should evaluate when reviewing Helion PRs. CI handles linting (ruff) and type checking (pyrefly), so focus on what automated tools cannot catch.

## Code Quality

### Python Style

**Note:** Formatting and lint style (line length, quotes, import sorting, single imports per line, `from __future__ import annotations`) are enforced by ruff in CI — see `pyproject.toml` for configuration.

- [ ] No local scope imports unless necessary (ruff does not enforce this)
- [ ] Type hints on function signatures in test files (`ANN` rules are disabled for `test/*`)
- [ ] snake_case for modules/files; `test_*.py` for tests

### Helion Import Conventions
- [ ] Uses `import helion` and `import helion.language as hl`
- [ ] Does NOT use `import helion as hl` (this is incorrect)
- [ ] Kernel code uses `hl.tile`, `hl.zeros`, etc. not fully qualified names

### Code Organization
- [ ] Functions and classes have clear, single responsibilities
- [ ] No unnecessary error handling (`hasattr`, `getattr`, bare `except`)
- [ ] No `del ...` or `_ = ...` on unused function args
- [ ] No `print()` statements inside kernels (use logging or host-side code)

## Kernel Correctness

### Tile Indexing
- [ ] `hl.tile()` usage preserves tensor dimensions correctly
- [ ] Tile indices used in slicing maintain correct shape (`x[tile_m, tile_k]`)
- [ ] Begin/end bounds used correctly when needed (`tile.begin`, `tile.end`)
- [ ] Block size accessed via `tile.block_size` when constant size needed

### Dtype Handling
- [ ] Accumulator dtypes are explicit (e.g., `dtype=torch.float32`)
- [ ] Type promotion handled correctly for mixed-precision operations
- [ ] Casts are explicit when needed (`.to(dtype)`)
- [ ] Autocast context handled correctly (not leaking into type propagation)

### Masking
- [ ] Implicit masking sufficient for the use case
- [ ] No out-of-bounds memory access on edge tiles
- [ ] Reduction operations handle partial tiles correctly

### Reductions
- [ ] `reduction_loops` config considered for large reductions
- [ ] Persistent reductions (`None`) vs looped reductions (integer block size)
- [ ] Atomic operations (`hl.atomic_add`) used correctly for split-k patterns

### Memory Operations
- [ ] `hl.load` / `hl.store` used correctly when explicit control needed
- [ ] Eviction policies specified when beneficial
- [ ] Memory layout considerations for coalesced access

## Helion DSL Patterns

### Core APIs
- [ ] `hl.tile()` - correct dimension specification, block_size parameter
- [ ] `hl.zeros()` - correct shape and dtype
- [ ] `hl.full()` - correct shape, fill value, dtype
- [ ] `hl.dot()` - accumulator pattern correct, matrix dimensions valid
- [ ] `hl.register_block_size()` - used for kernel-wide block size constants
- [ ] `hl.grid()` - correct for explicit grid control

### Torch Operations in Kernels
- [ ] `torch.addmm` - accumulator is first argument
- [ ] `torch.matmul` - shapes compatible
- [ ] Pointwise ops - broadcasting handled correctly
- [ ] Views - supported and handled correctly

### Function Traceability
- [ ] Functions called within kernels are traceable with `make_fx`
- [ ] No operations that break tracing (side effects, control flow on tensor values)

## Configuration

### Block Sizes
- [ ] `block_sizes` list length matches number of `hl.tile` dimensions
- [ ] Block sizes are powers of 2 (typically)
- [ ] Block sizes reasonable for hardware (16, 32, 64, 128, 256)

### Indexing Strategies
- [ ] `indexing` parameter valid ("pointer", "block_ptr", "tensor_descriptor")
- [ ] `tensor_descriptor` only used with Hopper+ GPUs
- [ ] Per-load/store indexing specified correctly if using list form

### PID Types
- [ ] `pid_type` valid ("flat", "xyz", "persistent_blocked", "persistent_interleaved")
- [ ] Persistent kernels used appropriately for SM utilization

### Loop Configuration
- [ ] `loop_orders` permutation valid for tile dimensions
- [ ] `range_unroll_factors` reasonable (0-8 typically)
- [ ] `range_num_stages` appropriate for memory-bound vs compute-bound
- [ ] `range_warp_specializes` only on supported architectures (Blackwell+)

### L2 Cache
- [ ] `l2_groupings` reasonable (1 = disabled, higher = grouped PIDs)

## Type Propagation

- [ ] Accumulator dtypes preserved through operations
- [ ] Mixed precision handled correctly (bf16 inputs, fp32 accumulator)
- [ ] Autocast context not leaking into kernel type inference
- [ ] Output dtype matches expected result

## Testing

### Test Structure
- [ ] Tests in `test/` directory, named `test_<feature>.py`
- [ ] Uses `TestCase` from `helion._testing`
- [ ] Uses `code_and_output()` to verify kernel output AND generated code
- [ ] Uses `torch.testing.assert_close()` for numerical comparison

### Test Coverage
- [ ] Tests cover typical input sizes
- [ ] Tests cover edge cases (small sizes, non-power-of-2, zero-size)
- [ ] Tests cover different dtypes if applicable
- [ ] Tests use appropriate tolerances (`atol`, `rtol`)

### Test Decorators
- [ ] `@skipIfRefEager` for tests incompatible with eager mode
- [ ] `@skipIfTileIR` for tests incompatible with TileIR backend
- [ ] `@skipUnlessTensorDescriptor` for TMA-dependent tests
- [ ] `@onlyBackends(["triton"])` for Triton-specific tests

### Golden Files
- [ ] `*.expected` files updated if output format changed
- [ ] `*.expected_tileir` files for TileIR-specific expected output

### Environment
- [ ] Tests run fast (<30s each)
- [ ] CUDA requirements documented or skipped appropriately
- [ ] Uses `DEVICE` from `helion._testing` (not hardcoded "cuda")

## Examples

### Structure
- [ ] Each example file defines a `main()` function (pre-commit enforced)
- [ ] Examples are runnable standalone
- [ ] Examples follow Helion import conventions

### Documentation
- [ ] Docstrings explain what the example demonstrates
- [ ] Comments explain non-obvious patterns

### Testing
- [ ] Example has corresponding test in `test/test_examples.py`
- [ ] Test uses `check_example()` or equivalent

## Security

- [ ] No hardcoded credentials or sensitive data
- [ ] Input validation for user-provided data
- [ ] No arbitrary code execution vulnerabilities
- [ ] Safe handling of file paths and external data

## Performance

### Memory Access
- [ ] Coalesced memory access patterns
- [ ] Appropriate tile sizes for memory bandwidth
- [ ] Consideration of L2 cache reuse

### Compute
- [ ] Reasonable num_warps for compute intensity
- [ ] Appropriate num_stages for pipelining
- [ ] Warp specialization considered where beneficial

### Autotuning
- [ ] Search space not unnecessarily large
- [ ] Default config reasonable for testing
- [ ] `autotune_effort` parameter used appropriately

### Benchmarking
- [ ] Performance claims backed by benchmarks
- [ ] Comparison against baseline (Triton, PyTorch eager)
- [ ] Hardware and configuration documented

## Backward Compatibility

### API Changes
- [ ] Changes to `@helion.kernel` decorator documented
- [ ] Changes to `helion.Config` options documented
- [ ] Changes to `helion.language` (hl) API documented
- [ ] Deprecation warnings added for removed features

### Behavior Changes
- [ ] Default behavior changes documented
- [ ] Autotuning behavior changes documented
- [ ] Generated code changes that affect semantics documented
