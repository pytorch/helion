# NPU Performance Benchmarking

This directory contains performance comparison tests between Triton-Ascend and PyTorch's native implementations on NPU hardware.

## Files

- `triton_add.py` - Pure Triton implementation of element-wise addition
- `compare_add.py` - Performance benchmarking script using `triton.testing.do_bench_npu`
- `README.md` - This file

## Prerequisites

- PyTorch with NPU support (torch + torch_npu)
- Triton with NPU backend support

## Usage

### Quick Test

Run the comparison with default settings:

```bash
cd /workspace/helion/npu_ops_test/benchmark
python compare_add.py
```

### Run Tests with Different Configurations

The `compare_add.py` script automatically tests multiple tensor sizes and block configurations:

- Tensor sizes: (1024, 1024), (2048, 2048), (512, 2048)
- Block sizes: 128x128, 64x64, 256x64, 64x256

### Understanding the Results

The script will output:

1. **Correctness Verification**: Ensures Triton implementation matches PyTorch
2. **Performance Benchmarking**: Timing results using `do_bench_npu`
3. **Speedup Comparison**: Relative performance vs PyTorch baseline
4. **Best Configuration**: Identifies the optimal Triton block size

Example output:

```
========================================================================
Performance Summary
========================================================================

Baseline (PyTorch torch.add): 0.0466 ms

Implementation                 Time (ms)    Speedup      Status
--------------------------------------------------------------------------------
PyTorch torch.add              0.0466       1.00x        ✓
Triton (block=128x128)         0.0500       0.93x        ✗ (SLOWER)
Triton (block=64x64)           0.0480       0.97x        ✗ (SLOWER)

Best Triton configuration: Triton (block=64x64)
  Time: 0.0480 ms
  Speedup vs PyTorch: 0.97x
```

## Why Use do_bench_npu?

`triton.testing.do_bench_npu` is specifically optimized for NPU (Ascend) hardware. Note that it has a different API than standard `do_bench`:

- Accepts a list of functions as the first argument (`funcs`), not a single function
- Uses `active` parameter instead of `rep` for number of iterations
- Returns a list of results, one per function

1. **Accurate Timing**: Uses `torch.npu.synchronize()` for proper NPU synchronization
2. **Cache Management**: Clears NPU caches between measurements
3. **Repeat Estimation**: Dynamically adjusts iteration count based on runtime
4. **Statistical Summary**: Returns median and quantiles for robust results

## Troubleshooting

### Import Error: triton.testing.do_bench_npu

Ensure you have Triton with NPU backend installed. The `do_bench_npu` function should be available in `triton.testing`.

### Performance Issues

If Triton is significantly slower than PyTorch:

1. Check if Triton backend is properly configured for NPU
2. Try different block sizes (modify `compare_add.py`)
3. Verify NPU driver and firmware are up to date
4. Check Triton compiler optimization flags

### Correctness Failures

If the correctness check fails:

1. Verify data types match
2. Check for NaN/Inf values in input tensors
3. Compare with different random seeds
4. Review Triton kernel implementation for bugs

## Modifying Tests

To test different operations:

1. Add new kernel to `triton_add.py`
2. Add benchmark case to `compare_add.py`
3. Run with `python compare_add.py`

To test different tensor sizes or data types, modify the `test_cases` list in `compare_add.py`.
