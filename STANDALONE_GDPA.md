# Standalone GDPA Helion Test

This is a simplified, self-contained version of the GDPA (Generalized Dot Product Attention) helion test, extracted from the original repository with **absolutely no Buck dependencies**.

## Files

- `standalone_gdpa.py` - All-in-one Python script containing:
  - Data generation
  - Helion GDPA forward kernel
  - Helion GDPA backward kernels (dK/dV and dQ)
  - PyTorch reference implementation
  - Test harness

- `run_standalone_gdpa.sh` - Convenience script to run with all required environment variables

## Usage

### Quick Start

```bash
cd /data/users/willfeng/helion
./run_standalone_gdpa.sh
```

This will:
1. Run the GDPA test with helion compilation
2. Print the generated output code (due to `HELION_PRINT_OUTPUT_CODE=1`)
3. Save all output to `/tmp/helion_standalone.log`

### Manual Execution

```bash
HELION_PRINT_OUTPUT_CODE=1 \
TORCH_COMPILE_FORCE_DISABLE_CACHES=1 \
TRITON_LOCAL_BUILD=1 \
HELION_AUTOTUNE_LOG_LEVEL=DEBUG \
HELION_SKIP_CACHE=1 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python standalone_gdpa.py
```

## What It Does

The script:

1. **Generates jagged test data** - Creates random Q, K, V tensors with variable sequence lengths
2. **Runs reference PyTorch implementation** - Simple attention computation for validation
3. **Runs Helion GDPA kernel** - Optimized helion implementation with:
   - Forward pass: Q @ K^T @ V with fast_gelu activation
   - Backward pass: Gradients for Q, K, and V
4. **Compares results** - Validates that helion output matches reference (within tolerance)

## Environment Variables

- `HELION_PRINT_OUTPUT_CODE=1` - Print generated helion code
- `TORCH_COMPILE_FORCE_DISABLE_CACHES=1` - Disable torch compilation caching
- `TRITON_LOCAL_BUILD=1` - Use local triton build
- `HELION_AUTOTUNE_LOG_LEVEL=DEBUG` - Enable detailed autotuning logs
- `HELION_SKIP_CACHE=1` - Skip helion caching
- `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` - Disable inductor caching

## Test Configuration

Default test parameters (can be modified in `generate_jagged_data()`):

- Batch size: 5
- Max sequence length: 5
- Dimension per head: 12
- Number of heads: 3
- Sparsity: 0.5
- Data type: bfloat16
- Activation: fast_gelu

## Output

The test will print:
- Test data generation status
- Reference and helion outputs
- Comparison results (✓ for pass, ✗ for fail)
- All generated code when `HELION_PRINT_OUTPUT_CODE=1` is set

## Notes

- No fbgemm dependencies - Uses simple offset generation instead
- No Buck build system required
- All functionality in a single file
- Matches the logic from the original `mkl/ops/helion:test_gdpa` target
