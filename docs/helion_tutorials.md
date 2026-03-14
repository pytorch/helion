# Helion Tutorials

Programming for accelerators such as GPUs is critical for modern AI systems. This often means programming directly in proprietary low-level languages such as CUDA. Helion is a Python-embedded domain-specific language (DSL) for authoring machine learning kernels, designed to compile down to Triton, a performant backend for programming GPUs and other devices.

Helion aims to raise the level of abstraction compared to Triton, making it easier to write correct and efficient kernels while enabling more automation in the autotuning process.

This set of tutorials is meant to teach you how to use Helion from first principles in an interactive fashion. You will start with trivial examples and build your way up to real algorithms like Flash Attention and Quantized neural networks.

## Setup

First, let's install the necessary dependencies. Helion requires a recent version of PyTorch and a development version of Triton.

```python
import logging

import helion
import helion.language as hl
import torch
from torch import Tensor

# If you set this to info you will see the output Triton Code
logging.getLogger().setLevel(logging.WARNING)
```

Let's also create a simple testing function to verify our implementations.

```python
from triton.testing import do_bench
def test_kernel(kernel_fn, spec_fn, *args, rtol=None, atol=None):
    """Test a Helion kernel against a reference implementation."""
    # Run our implementation
    result = kernel_fn(*args)
    # Run reference implementation
    expected = spec_fn(*args)

    # Check if results match
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
    print("✅ Results Match ✅")

def benchmark_kernel(kernel_fn, *args, **kwargs):
    """Benchmark a Helion kernel."""
    no_args = lambda: kernel_fn(*args, **kwargs)
    time_in_ms = do_bench(no_args)
    print(f"⏱ Time: {time_in_ms} ms")

def compare_implementations(kernel_fn, spec_fn, *args, **kwargs):
    """Benchmark a Helion kernel and its reference implementation."""
    kernel_no_args = lambda: kernel_fn(*args, **kwargs)
    spec_no_args = lambda: spec_fn(*args, **kwargs)
    kernel_time = do_bench(kernel_no_args)
    spec_time = do_bench(spec_no_args)
    print(f"⏱ Helion Kernel Time: {kernel_time:.3f} ms, PyTorch Reference Time: {spec_time:.3f} ms, Speedup: {spec_time/kernel_time:.3f}x")
```

## Basic Structure of a Helion Kernel

Helion allows you to write GPU kernels using familiar PyTorch syntax.

A Helion kernel has three main sections:

1. **Host Section** (CPU)
   This is standard PyTorch code executed on the CPU. Memory allocation, and shape computations are done here. Like with `Triton` and `Cuda` you need to setup your output buffers on the host before launching your kernel.

2. **Device Loop** (GPU Grid)
   `for tile in hl.tile(sizes)` - defines parallel execution across GPU thread blocks

3. **Device Operations** (GPU Kernel)
   PyTorch operations inside the loop - automatically compiled and fused

Example:

```python
@helion.kernel(config=helion.Config(block_sizes = [128, 128]))  # The @helion.kernel decorator marks this function for compilation
def example_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Host code: Standard PyTorch operations
    m, n = x.size()
    out = torch.empty_like(x)  # Allocate output tensor

    # The hl.tile loop defines the parallel execution structure
    for tile_m, tile_n in hl.tile([m, n]):
        # Device code: Everything inside the hl.tile loop runs on GPU
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n] # Simple element-wise addition expressed w/ pytorch ops

    return out  # Return the result back to the host

# Create some sample data
x = torch.randn(10, 10, device="cuda")
y = torch.randn(10, 10, device="cuda")

# Run the kernel
result = example_add(x, y)

# Verify result
expected = x + y
torch.testing.assert_close(result, expected)
print("✅ Results Match ✅")
benchmark_kernel(example_add, x, y)
compare_implementations(example_add, torch.add, x, y)
```

## Autotuning in Helion

In the previous example, we explicitly specified a configuration using `config=helion.Config(block_sizes=[128, 128])`. This bypasses Helion's autotuning mechanism and uses our predefined settings. While this is quick to run, manually choosing optimal parameters can be challenging and hardware-dependent.

### What is Autotuning?

Autotuning is Helion's process of automatically finding the best configuration parameters for your specific:

- Hardware (GPU model)
- Problem size
- Operation patterns

When you omit the `config` parameter, Helion will automatically search for the optimal configuration:

```python
@helion.kernel()  # No config = automatic tuning
def autotuned_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
   m, n = x.size()
   out = torch.empty_like(x)
   for tile_m, tile_n in hl.tile([m, n]):
       out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
```

Feel free to run the above code to see how much more performant it is than the original, although be warned it might take some time 😃

### Autotune Cache (Ahead-of-Time Friendly)

Helion caches autotuned configs on disk so autotuning only runs once per
hardware/code combination.  The default `LocalAutotuneCache` writes one
file per kernel.  For production deployments—where sharing configs across
machines or eliminating cold-start latency matters—use `FileAutotuneCache`:

```bash
# Enable the shared-file cache (tune on first run, hit on all subsequent runs)
HELION_AUTOTUNE_CACHE=FileAutotuneCache \
HELION_AUTOTUNE_CACHE_PATH=~/.helion/project_cache.json \
python my_model.py
```

The cache key includes a version stamp (Helion + Triton + CUDA), so entries
are automatically invalidated when library versions change.  You can also
customize which fields form the key:

```bash
# Pin to kernel identity only, ignoring shape changes
HELION_AUTOTUNE_CACHE_KEY="{version_stamp}-{kernel_name}-{hardware}-{backend}"
```

To force re-autotuning, set `HELION_SKIP_CACHE=1` or delete the cache file.


Now let's move on to our tutorials!

## Problem 1: Constant Add

Add a constant to a vector.

```python
def add_spec(x: Tensor) -> Tensor:
    """This is the spec that you should implement."""
    return x + 10.

@helion.kernel(config = helion.Config(block_sizes = [32,]))
def add_kernel(x: torch.Tensor) -> torch.Tensor:
    TILE_RANGE = x.size()
    out = torch.empty_like(x)

    for tile_n in hl.tile(TILE_RANGE):
        x_tile = x[tile_n]
        out[tile_n] = x_tile + 10.0

    return out

# Test the kernel
x = torch.randn(8192, device="cuda")
test_kernel(add_kernel, add_spec, x)
benchmark_kernel(add_kernel, x)
compare_implementations(add_kernel, add_spec, x)
```

## Problem 2: Outer Vector Add

Add two vectors using an outer product pattern.

```python
def broadcast_add_spec(x: Tensor, y: Tensor) -> Tensor:
    return x[None, :] + y[:, None]

@helion.kernel(config = helion.Config(block_sizes = [32, 32]))
def broadcast_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n0 = x.size(0)
    n1 = y.size(0)
    out = x.new_empty(n1, n0)

    # Use Helion to tile the computation
    for tile_i, tile_j in hl.tile([n1, n0]):
        # Get tiles from x and y
        y_tile = y[tile_i]
        x_tile = x[tile_j]
        # Compute outer sum
        out[tile_i, tile_j] = y_tile[:, None] + x_tile[None, :]

    return out

# Test the kernel
x = torch.randn(1142, device="cuda")
y = torch.randn(512, device="cuda")
test_kernel(broadcast_add_kernel, broadcast_add_spec, x, y)
benchmark_kernel(broadcast_add_kernel, x, y)
compare_implementations(broadcast_add_kernel, broadcast_add_spec, x, y)
```

## Problem 3: Fused Outer Multiplication

Multiply a row vector to a column vector and take a relu.

```python
def mul_relu_block_spec(x: Tensor, y: Tensor) -> Tensor:
    return torch.relu(x[None, :] * y[:, None])

@helion.kernel(config = helion.Config(block_sizes = [32, 32]))
def mul_relu_block_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    n0 = x.size(0)
    n1 = y.size(0)
    # Create output tensor
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)

    # Use Helion to tile the computation
    for tile_i, tile_j in hl.tile([n1, n0]):
        # Get tiles from x and y
        y_tile = y[tile_i]
        x_tile = x[tile_j]
        # Compute outer product followed by ReLU
        out[tile_i, tile_j] = torch.relu(y_tile[:, None] * x_tile[None, :])

    return out

# Test the kernel
x = torch.randn(512, device="cuda")
y = torch.randn(512, device="cuda")
test_kernel(mul_relu_block_kernel, mul_relu_block_spec, x, y)
compare_implementations(mul_relu_block_kernel, mul_relu_block_spec, x, y)
```

## Problem 4: Fused Outer Multiplication - Backwards

While PyTorch and torch.compile automatically generates the backwards pass for your Tensor Operations, Helion does not. So lets practice by writing the backwards function for a fused mul_relu kernel

```python
def mul_relu_block_back_spec(x: Tensor, y: Tensor, dz: Tensor) -> tuple[Tensor, Tensor]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    grad_x, grad_y = torch.autograd.grad(z, [x, y], dz, retain_graph=True)
    return grad_x, grad_y

@helion.kernel(config=helion.Config(block_sizes=[[32, 32], [32, 32]]))
def mul_relu_block_back_kernel(
    x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n0 = x.size(1)
    n1 = x.size(0)
    dx = torch.empty_like(x)
    dy = torch.empty([n1], dtype=x.dtype, device=x.device)

    # First loop block: compute dx
    # For z = relu(x * y[:, None]), dx = dz * relu_mask * y[:, None]
    for tile_i, tile_j in hl.tile([n1, n0]):
        x_tile = x[tile_i, tile_j]
        y_tile = y[tile_i]
        dz_tile = dz[tile_i, tile_j]
        relu_mask = (x_tile * y_tile[:, None]) > 0
        dx[tile_i, tile_j] = dz_tile * relu_mask * y_tile[:, None]

    # Second loop block: compute dy (reduce over columns)
    # dy[i] = sum_j (dz[i,j] * relu_mask[i,j] * x[i,j])
    for tile_i2 in hl.tile(n1):
        acc_dy = hl.zeros([tile_i2], dtype=torch.float32)
        for tile_j2 in hl.tile(n0):
            x2 = x[tile_i2, tile_j2]
            y2 = y[tile_i2]
            dz2 = dz[tile_i2, tile_j2]
            mask2 = (x2 * y2[:, None]) > 0
            acc_dy += torch.sum(dz2 * mask2 * x2, dim=1)
        dy[tile_i2] = acc_dy

    return dx, dy

# Test the kernel
x = torch.randn(512, 1024, device="cuda")
y = torch.randn(512, device="cuda")
dz = torch.randn(512, 1024, device="cuda")
test_kernel(mul_relu_block_back_kernel, mul_relu_block_back_spec, x, y, dz)
```

## Problem 5: Long Sum

Sum of a batch of numbers.

```python
def sum_spec(x: Tensor) -> Tensor:
    return x.sum(1)

@helion.kernel(config=helion.Config(block_sizes=[4, 64]))
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    batch, seq_len = x.size()
    # Create output tensor
    out = torch.empty(batch, dtype=x.dtype, device=x.device)

    # Use Helion to tile the batch dimension
    for tile_batch in hl.tile(batch):
        # Initialize accumulator for each batch element
        acc = hl.zeros([tile_batch], dtype=torch.float32)

        # Process the sequence in chunks
        for tile_seq in hl.tile(seq_len):
            # Get the current chunk
            chunk = x[tile_batch, tile_seq]
            # Accumulate sum
            acc += torch.sum(chunk, dim=1)

        # Store result
        out[tile_batch] = acc

    return out

# Test the kernel
x = torch.randn(4, 200, device="cuda")
test_kernel(sum_kernel, sum_spec, x)
```

## Problem 6: Long Softmax

Softmax of a batch of logits.

```python
def softmax_spec(x: Tensor) -> Tensor:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)

@helion.kernel(config=helion.Config(block_sizes=[4, 64, 64, 64]))
def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    batch, seq_len = x.size()
    # Create output tensor
    out = torch.empty_like(x)

    # Use Helion to tile the batch dimension
    for tile_batch in hl.tile(batch):
        # First pass: find max value for each sequence
        max_i = hl.full([tile_batch], float("-inf"), dtype=torch.float32)
        for tile_seq in hl.tile(seq_len):
            x_tile = x[tile_batch, tile_seq]
            max_i = torch.maximum(max_i, torch.amax(x_tile, dim=1))

        # Second pass: compute sum of exp(x - max)
        denom = hl.zeros([tile_batch], dtype=torch.float32)
        for tile_seq in hl.tile(seq_len):
            x_tile = x[tile_batch, tile_seq]
            denom += torch.exp(x_tile - max_i[:, None]).sum(dim=1)

        # Third pass: compute softmax
        for tile_seq in hl.tile(seq_len):
            x_tile = x[tile_batch, tile_seq]
            out[tile_batch, tile_seq] = (
                torch.exp(x_tile - max_i[:, None]) / denom[:, None]
            )

    return out

# Test the kernel
x = torch.randn(4, 200, device="cuda")
test_kernel(softmax_kernel, softmax_spec, x)
```

## Problem 7: Simple FlashAttention

A scalar version of FlashAttention using online softmax for numerical stability.

```python
def flashatt_spec(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)

@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def flashatt_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Get tensor size
    seq_len = q.size(0)
    # Create output tensor
    out = torch.empty_like(q)

    # Process each query position
    for tile_q in hl.tile(seq_len):
        q_tile = q[tile_q]

        # Initialize tracking variables for stable softmax
        max_val = hl.full([tile_q], float('-inf'), dtype=torch.float32)
        sum_exp = hl.zeros([tile_q], dtype=torch.float32)
        weighted_sum = hl.zeros([tile_q], dtype=torch.float32)

        # Process in tiles for better cache efficiency
        for tile_kv in hl.tile(seq_len):
            k_tile = k[tile_kv]
            v_tile = v[tile_kv]

            # Compute attention scores
            scores = q_tile[:, None] * k_tile[None, :]

            # Find max for numerical stability
            batch_max = torch.amax(scores, dim=1)
            new_max = torch.maximum(max_val, batch_max)

            # Scale old accumulations
            scale_factor = torch.exp(max_val - new_max)
            sum_exp = sum_exp * scale_factor
            weighted_sum = weighted_sum * scale_factor

            # Update with new values
            exp_scores = torch.exp(scores - new_max[:, None])
            sum_exp = sum_exp + torch.sum(exp_scores, dim=1)
            weighted_sum = weighted_sum + torch.sum(exp_scores * v_tile[None, :], dim=1)

            # Update max_val
            max_val = new_max

        # Compute final output
        out[tile_q] = weighted_sum / sum_exp

    return out

# Test the kernel
q = torch.randn(200, device="cuda")
k = torch.randn(200, device="cuda")
v = torch.randn(200, device="cuda")
test_kernel(flashatt_kernel, flashatt_spec, q, k, v)
```

## Problem 8: Two Dimensional Convolution

A batched 2D convolution.

```python
def conv2d_spec(x: Tensor, k: Tensor) -> Tensor:
    z = torch.zeros(4, 8, 8, device=x.device)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k * x[:, i: i+4, j: j + 4]).sum(1).sum(1)
    return z

@helion.kernel(config=helion.Config(block_sizes=[4]), ignore_warnings=[helion.exc.TensorOperationInWrapper])
def conv2d_kernel(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    batch, h, w = x.size()
    kh, kw = k.size()[1:]

    # Create output tensor
    out = torch.empty_like(x)

    # Pad the input
    x_padded = torch.nn.functional.pad(x, (0, kw, 0, kh, 0, 0), value=0.0)

    # Use Helion to tile the computation
    for tile_batch in hl.tile(batch):
        # Process each output position
        for i in range(h):
            for j in range(w):
                # Extract the patch
                patch = x_padded[tile_batch, i:i+kh, j:j+kw]
                # Apply the kernel (chain reductions since Helion requires single-dim reduction)
                out[tile_batch, i, j] = (k[tile_batch,:,:] * patch).sum(1).sum(1)

    return out

# Test the kernel
x = torch.randn(4, 8, 8, device="cuda")
k = torch.randn(4, 4, 4, device="cuda")
test_kernel(conv2d_kernel, conv2d_spec, x, k)
```

## Problem 9: Matrix Multiplication

A blocked matrix multiplication.

```python
def dot_spec(x: Tensor, y: Tensor) -> Tensor:
    return x @ y

@helion.kernel(config=helion.Config(block_sizes=[4, [32, 32], 32]))
def dot_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    batch, m, k = x.size()
    _, k, n = y.size()

    # Create output tensor
    out = torch.empty([batch, m, n], dtype=x.dtype, device=x.device)

    # Use Helion to tile the computation
    for tile_batch in hl.tile(batch):
        for tile_m, tile_n in hl.tile([m, n]):
            # Initialize accumulator (3D to match batched output)
            acc = hl.zeros([tile_batch, tile_m, tile_n], dtype=torch.float32)

            # Process the reduction dimension in tiles
            for tile_k in hl.tile(k):
                # Accumulate batched matrix multiplication
                acc = torch.baddbmm(acc, x[tile_batch, tile_m, tile_k], y[tile_batch, tile_k, tile_n])

            # Store result
            out[tile_batch, tile_m, tile_n] = acc

    return out

# Test the kernel
x = torch.randn(4, 32, 32, device="cuda", dtype=torch.float16)
y = torch.randn(4, 32, 32, device="cuda", dtype=torch.float16)
test_kernel(dot_kernel, dot_spec, x, y)
```

## Problem 10: Quantized Matrix Multiplication

When doing matrix multiplication with quantized neural networks, a common strategy is to store the weight matrix in lower precision, with per-group scale factors and zero-point offsets. Each int32 packs 8 x 4-bit values. The kernel iterates over groups, unpacks the nibbles using bitwise operations, applies per-group dequantization, and accumulates the matrix multiplication.

```python
FPINT = 32 // 4
GROUP = 8

def extract_4bit(x: torch.Tensor) -> torch.Tensor:
    """Extract 8 x 4-bit values from packed int32 tensor.
    Each int32 contains 8 nibbles at bit positions 0, 4, 8, ..., 28."""
    over = torch.arange(8, device=x.device) * 4
    mask = 2**4 - 1
    return (x[..., None] >> over) & mask

def quant_dot_spec(scale: Tensor, offset: Tensor,
                   weight: Tensor, activation: Tensor) -> Tensor:
    offset = offset.view(32, 1)
    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = extract_4bit(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    return (scale * (extract_4bit(weight).view(-1, 64) - offset)) @ activation

@helion.kernel(
    config=helion.Config(block_sizes=[32, 32]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def quant_dot_kernel(scale: torch.Tensor, offset: torch.Tensor,
                     weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    n_out, n_groups = scale.size()
    mid, n_in = activation.size()
    out = torch.empty([n_out, n_in], dtype=scale.dtype, device=scale.device)

    # Precompute shift amounts for 4-bit extraction
    shifts = torch.arange(FPINT, device=scale.device) * 4  # [0, 4, 8, ..., 28]
    mask = 2**4 - 1

    for tile_out, tile_in in hl.tile([n_out, n_in]):
        acc = hl.zeros([tile_out, tile_in], dtype=torch.float32)

        # Process each group of packed weights
        for group_idx in range(n_groups):
            # Get scale for this group
            scale_group = scale[tile_out, group_idx]  # [block_out]

            # Get packed weight for this group and extract 8 nibbles
            w_packed = weight[tile_out, group_idx]  # [block_out] int32
            w_nibbles = (w_packed[:, None] >> shifts[None, :]) & mask
            w_nibbles = w_nibbles.to(scale.dtype)  # [block_out, 8]

            # Extract offset nibble for this group
            o_packed = offset[tile_out]  # [block_out] int32
            o_nibble = ((o_packed >> (group_idx * 4)) & mask).to(scale.dtype)
            # → [block_out]

            # Dequantize: scale * (weight - offset)
            dequant = scale_group[:, None] * (w_nibbles - o_nibble[:, None])
            # → [block_out, 8]

            # Get activations for this group (8 rows per group)
            act_group = activation[group_idx * 8 : (group_idx + 1) * 8, tile_in]
            # → [8, block_in]

            # Accumulate
            acc = acc + torch.matmul(dequant, act_group)

        out[tile_out, tile_in] = acc

    return out

# Test the kernel
scale = torch.randn(32, 8, device="cuda")
offset = torch.randint(-2**31, 2**31, (32,), device="cuda", dtype=torch.int32)
weight = torch.randint(-2**31, 2**31, (32, 8), device="cuda", dtype=torch.int32)
activation = torch.randn(64, 32, device="cuda")
test_kernel(quant_dot_kernel, quant_dot_spec, scale, offset, weight, activation, rtol=1e-2, atol=1e-1)
```

## Autotuning in Helion

One of the major advantages of Helion is its sophisticated autotuning capability. Let's see how we can leverage this for our matrix multiplication kernel:

```python
import torch
import helion
import helion.language as hl
import time

# Define a matrix multiplication kernel
@helion.kernel()  # No config means autotuning will be used
def matmul_autotune(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out

# Create larger tensors for better autotuning results
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# First run will trigger autotuning
print("Running with autotuning (this might take a while)...")
start = time.time()
result = matmul_autotune(x, y)
end = time.time()
print(f"First run time (including autotuning): {end - start:.2f}s")

# Second run will use the tuned configuration
start = time.time()
result = matmul_autotune(x, y)
end = time.time()
print(f"Second run time (using tuned config): {end - start:.2f}s")

# Verify correctness
expected = x @ y
print(f"Result is correct: {torch.allclose(result, expected, rtol=1e-2, atol=1e-2)}")
```

## Hardcoding Configurations

After autotuning, you might want to hardcode the best configuration:

```python
# Example of hardcoding a configuration after autotuning
@helion.kernel(config=helion.Config(
    block_sizes=[[64, 128], [16]],
    loop_orders=[[1, 0]],
    num_warps=4,
    num_stages=3,
    indexing='block_ptr',
    l2_grouping=32
))
def matmul_fixed_config(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out

# Run with fixed configuration (no autotuning)
start = time.time()
result = matmul_fixed_config(x, y)
end = time.time()
print(f"Run time with fixed config: {end - start:.2f}s")

# Verify correctness
expected = x @ y
print(f"Result is correct: {torch.allclose(result, expected, rtol=1e-2, atol=1e-2)}")
```

## Conclusion

In this notebook, we've explored how to use Helion to write efficient GPU kernels using a high-level, PyTorch-like syntax. The key advantages of Helion include:

1. **Higher-level abstraction** than raw Triton, making it easier to write correct kernels
2. **Automatic tiling and memory management**, eliminating a common source of bugs
3. **Powerful autotuning** that can explore a wide range of implementations automatically
4. **Familiar PyTorch syntax** that builds on existing knowledge

These tutorials should give you a good foundation for writing your own Helion kernels for a variety of applications.
