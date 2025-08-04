#!/usr/bin/env -S grimaldi --kernel bento_kernel_helion
# fmt: off

""":md
# Helion Tutorial

(Acknowledgment:
This notebook is adapted from Driss's https://www.internalfb.com/intern/anp/view/?id=7338830)

Programming for accelerators such as GPUs is critical for modern AI systems. This often means programming directly in proprietary low-level languages such as CUDA. Helion is a Python-embedded domain-specific language (DSL) for authoring machine learning kernels, currently designed to compile down to Triton, a performant backend for programming GPUs and other devices.

Helion aims to raise the level of abstraction compared to Triton, making it easier to write correct and efficient kernels while enabling more automation in the autotuning process.

This tutorial is meant to teach you how to use Helion from first principles. You will start with trivial examples and build your way up to real algorithms that are useful in the modern transformer.

**Instructions:**
- TODO

# Make a copy of this notebook and run on On-Demand GPU / your devgpu (Use "Connect" button at top-right corner)
"""

""":md
## Setup

The kernel used here `Helion` is based off of the fbcode version of Helion. This is synced per-commit w/ OSS: https://github.com/pytorch-labs/helion. It will have all the dependencies to prototype and execute Helion kernels. Be sure to run on a machine w/ access to GPUs.
"""

""":py"""
%load_ext autoreload
%autoreload 2

import logging

from typing import Tuple
import helion
import helion.language as hl
import torch
from torch import Tensor

# If you set this to info you will see the output Triton Code
logging.getLogger().setLevel(logging.WARNING)

""":md
Let's also create a simple testing function to verify our implementations.
"""

""":py"""
from triton.testing import do_bench


def test_kernel(kernel_fn, spec_fn, *args, **kwargs):
    """Test a Helion kernel against a reference implementation."""
    # Run our implementation
    result = kernel_fn(*args, **kwargs)
    # Run reference implementation
    expected = spec_fn(*args, **kwargs)

    # Check if results match
    torch.testing.assert_close(result, expected)
    print("✅ Results Match ✅")


def benchmark_kernel(kernel_fn, *args, **kwargs):
    """Benchmark a Helion kernel."""
    no_args = lambda: kernel_fn(*args, **kwargs)
    time_in_ms = do_bench(no_args)
    return time_in_ms
    print(f"⏱ Time: {time_in_ms} ms")


def compare_implementations(kernel_fn, spec_fn, *args, **kwargs):
    """Benchmark a Helion kernel and its reference implementation."""
    kernel_no_args = lambda: kernel_fn(*args, **kwargs)
    spec_no_args = lambda: spec_fn(*args, **kwargs)
    kernel_time = do_bench(kernel_no_args)
    spec_time = do_bench(spec_no_args)
    print(
        f"⏱ Helion Kernel Time: {kernel_time:.3f} ms, PyTorch Reference Time: {spec_time:.3f} ms, Speedup: {spec_time/kernel_time:.3f}x"
    )

""":md
## Basic Structure of a Helion Kernel
Helion allows you to write GPU kernels using familiar PyTorch syntax. 

A Helion kernel has three main sections:

### 1. **Host Section** (CPU)
This is standard PyTorch code executed on the CPU. memory allocation, and shape computations are done here. Like with `Triton` and `Cuda` you need to setup your output buffers on the host before launching your kenrel.

### 2. **Device Loop** (GPU Grid)  
`for tile in hl.tile(sizes)` - defines parallel execution across GPU thread blocks

### 3. **Device Operations** (GPU Kernel)
PyTorch operations inside the loop - automatically compiled and fused

```python
@helion.kernel()
def add_kernel(x, y):
   out = torch.empty_like(x)           # Host: setup
   for tile in hl.tile(x.shape):       # Device: parallelization  
       out[tile] = x[tile] + y[tile]   # Device: computation
   return out

```

Unlike raw Triton, by default Helion handles tiling, indexing, and other low-level details automatically. This allows you to focus on the algorithm rather than the implementation details.

**Let's examine a simple example**:
"""

""":py"""
# The @helion.kernel decorator marks this function for Helion compilation
# `use_default_config=True` will skip autotuning and is handy for quick development iteration
@helion.kernel(use_default_config=True)
def example_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Host code: Standard PyTorch operations
    m, n = x.size()
    out = torch.empty_like(x)  # Allocate output tensor

    # The hl.tile loop (device loop) defines the parallel execution structure
    for tile_m, tile_n in hl.tile([m, n]):
        # Device code: Everything inside the hl.tile loop runs on GPU
        out[tile_m, tile_n] = (
            x[tile_m, tile_n] + y[tile_m, tile_n]
        )  # Simple element-wise addition expressed w/ pytorch ops

    return out  # Return the result back to the host

# Alternatively, you can use @helion.kernel(config=helion.Config(block_sizes = [32, 32])) to manually specify block size and other configs

# Create some sample data
x = torch.randn(512, 512, device="cuda")
y = torch.randn(512, 512, device="cuda")

# Run the kernel
result = example_add(x, y)

# Verify result
expected = x + y
torch.testing.assert_close(result, expected)
print("✅ Results Match ✅")
benchmark_kernel(example_add, x, y)
compare_implementations(example_add, torch.add, x, y)

""":md
## Autotuning in Helion

In the previous example, we explicitly specified a configuration using `config=helion.Config(block_sizes=[32, 32])`. This bypasses Helion's autotuning mechanism and uses our predefined settings. While this is quick to run, manually choosing optimal parameters can be challenging and hardware-dependent.

### What is Autotuning?

Autotuning is Helion's process of automatically finding the best configuration parameters for your specific:
- Hardware (GPU model)
- Problem size

When you omit the `config` parameter, Helion will automatically search for the optimal configuration:

```python
@helion.kernel()  # No config = automatic tuning
def autotuned_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
   m, n = x.size()
   out = torch.empty_like(x)
   for tile_m, tile_n in hl.tile([m, n]):
       out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
   return out
```

Feel free to run the above code to see how much more performant it is than the original, although be warned it might take some time 😃

For kernel development, it's recommended to first use `@helion.kernel(use_default_config=True)` to get the kernel logic correct and passing accuracy test, then use autotuning to find most performant config.
"""

""":md
Now let's check out more examples!
"""

""":md
## TODO Example 1: Constant Add

Implement a kernel that adds 10.0 to every element of the input tensor.
"""

""":py"""
def add_spec(x: Tensor) -> Tensor:
    """This is the spec that you should implement."""
    return x + 10.0


# ---- ✨ Is this the best block size? ----
@helion.kernel(
    config=helion.Config(
        block_sizes=[
            256,
        ]
    )
)
def add_kernel(x: torch.Tensor) -> torch.Tensor:
    # ---- ✨ Your Code Here ✨----
    # TODO: Set up the output buffer which you will return
    out = torch.empty_like(x)
    # TODO define the proper tile range
    TILE_RANGE = x.shape
    # ---- End of Code ----

    # Use Helion to tile the computation
    for tile_n in hl.tile(TILE_RANGE):
        # ---- ✨ Your Code Here ✨----
        # TODO: Load input tile and add constant 10.0
        out[tile_n] = x[tile_n] + 10.0

    return out


# Test the kernel
x = torch.randn(8192, device="cuda")
test_kernel(add_kernel, add_spec, x)
benchmark_kernel(add_kernel, x)
compare_implementations(add_kernel, add_spec, x)

""":md
## TODO Example 2: Outer Vector Add

Add two vectors using an outer product pattern.

**Your Task:** Implement broadcasting addition where `x[None, :]` is added to `y[:, None]`.

<details>
<summary>Click to expand hints</summary>

**Hints:**
- Output shape should be `[len(y), len(x)]`
- Tile over both dimensions: `for tile_i, tile_j in hl.tile([n1, n0])`
- Broadcasting works the same in Helion as it does in PyTorch
</details>

"""

""":py"""
def broadcast_add_spec(x: Tensor, y: Tensor) -> Tensor:
    return x[None, :] + y[:, None]

 # ---- ✨ Is this the best block size? ----
@helion.kernel(config = helion.Config(block_sizes = [32, 32]))
def broadcast_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
     # ---- ✨ Your Code Here ✨----
    # TODO: Get sizes of x and y, create output tensor
    # Fill out N1, no
    n0 = x.size(0)
    n1 = y.size(0)
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)

    # Use Helion to tile the computation
    for tile_i, tile_j in hl.tile([n1, n0]):
        # ---- ✨ Your Code Here ✨----
        # TODO: Get tiles from x and y, compute outer sum
        out[tile_i, tile_j] = x[None, tile_j] + y[tile_i, None]

    return out

# Test the kernel
x = torch.randn(1142, device="cuda")
y = torch.randn(512, device="cuda")
test_kernel(broadcast_add_kernel, broadcast_add_spec, x, y)
benchmark_kernel(broadcast_add_kernel, x, y)
compare_implementations(broadcast_add_kernel, broadcast_add_spec, x, y)

""":md
## TODO Example 3: Fused Outer Multiplication

Multiply a row vector to a column vector and take a relu.

**Your Task:** Implement `torch.relu(x[None, :] * y[:, None])`.

<details>
<summary>Click to expand hints</summary>

**Hints:**
- Similar structure to Puzzle 2, but multiply instead of add
- Apply `torch.relu()` to the result of the multiplication
- Output shape is `[len(y), len(x)]`
</details>

"""

""":py"""
def mul_relu_block_spec(x: Tensor, y: Tensor) -> Tensor:
    return torch.relu(x[None, :] * y[:, None])


 # ---- ✨ Is this the best block size? ----
@helion.kernel(config = helion.Config(block_sizes = [32, 32]))
def mul_relu_block_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ---- ✨ Your Code Here ✨----
    # Training wheels off for this one
    # Get tensor sizes
    n0 = x.size(0)
    n1 = y.size(0)
    
    # Allocate output tensor
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)
    
    # Use Helion to tile the computation
    for tile_i, tile_j in hl.tile([n1, n0]):
        # Multiply with broadcasting and apply ReLU
        out[tile_i, tile_j] = torch.relu(x[None, tile_j] * y[tile_i, None])
    
    return out

# Test the kernel
x = torch.randn(512, device="cuda")
y = torch.randn(512, device="cuda")
test_kernel(mul_relu_block_kernel, mul_relu_block_spec, x, y)
compare_implementations(mul_relu_block_kernel, mul_relu_block_spec, x, y)

""":md
## Example 4: Using next_power_of_2() for Efficient Tiling

Sometimes we need to work with data sizes that aren't powers of 2, but GPU hardware often performs better with power-of-2 sizes.
The `next_power_of_2()` function helps us compute efficient tile sizes.

**Your Task:** Implement a kernel that processes data in power-of-2 sized chunks for better memory alignment.

<details>
<summary>Click to expand hints</summary>

**Hints:**
- Use `helion.next_power_of_2()` to determine an efficient block size
- Use `hl.register_block_size()` to make the block size available inside loops
- Process data in chunks of the power-of-2 size
</details>
"""

""":py"""
def chunked_sum_spec(x: Tensor) -> Tensor:
    """Compute the sum of elements in a 2D tensor along the last dimension."""
    return x.sum(-1)

@helion.kernel(config=helion.Config(block_sizes=[1, 128]))
def chunked_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Sum using power-of-2 chunk processing for efficiency."""
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    
    # Calculate an efficient chunk size based on the input dimension
    # This demonstrates using next_power_of_2 for optimization hints
    chunk_size = helion.next_power_of_2(min(n, 128))
    
    # Register the block size so it's available in the device loop
    block_size_n = hl.register_block_size(chunk_size)
    
    for tile_m in hl.tile(m):
        # Initialize accumulator with the registered block size
        acc = hl.zeros([tile_m, block_size_n], dtype=x.dtype)
        
        # Process in chunks of power-of-2 size
        for tile_n in hl.tile(n, block_size=block_size_n):
            acc += x[tile_m, tile_n]
        
        # Reduce the accumulator
        out[tile_m] = acc.sum(-1)
    
    return out

# Test the kernel
x = torch.randn(256, 1234, device="cuda")  # Non-power-of-2 width
# Note: Using relaxed tolerance due to different accumulation order
torch.testing.assert_close(chunked_sum_kernel(x), chunked_sum_spec(x), rtol=1e-4, atol=1e-4)
print("✅ Results Match ✅")
compare_implementations(chunked_sum_kernel, chunked_sum_spec, x)
"""

""":md
## TODO Example 5: Long Sum

Lets take a look at our first kernel involving reductions. There are number of ways to do this in Helion, by default letting Helion handle more of the kernel configuration is better since that gives it more opportunity to autotune.

**Your Task:** Implement a sum reduction along dimension 1 of a 3D tensor.

<details>
<summary>Click to expand hints</summary>

**Hints:**
- Input shape: `[batch, seq_len, hidden]`, Output shape: `[batch, hidden]`
- Tile over batch and hidden dimensions: `hl.tile([batch, hidden])`
- Sum over the middle dimension: Do think about tiling the reduction and call PyTorch 
</details>
"""

""":py"""
def sum_spec(x: torch.Tensor) -> torch.Tensor:
    return x.sum(1)

@helion.kernel(config=helion.Config(block_sizes=[1, 16]))
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    # Get tensor sizes
    batch, seq_len, hidden = x.size()
    # TODO Allocate
    out = torch.empty([batch, hidden], dtype=x.dtype, device=x.device)

    # Use Helion to tile the batch and hidden dimensions
    for tile_batch, tile_hidden in hl.tile([batch, hidden]):
        # ---- ✨ Your Code Here ✨----
        # TODO: Sum over the sequence length dimension (dim=1)
        # Get the slice for current batch and hidden tiles, all sequence positions
        # x[tile_batch, :, tile_hidden] has shape [batch_tile, seq_len, hidden_tile]
        # We want to sum over the seq_len dimension (dim=1)
        out[tile_batch, tile_hidden] = x[tile_batch, :, tile_hidden].sum(1)
    return out

x = torch.randn(128, 129, 128, device="cuda")
compare_implementations(sum_kernel, sum_spec, x)
test_kernel(sum_kernel, sum_spec, x)

""":md
Above is one way of tiling this problem, try being more explicit w/ how you break up the problem into subproblems. See if that has an impact on overall performance.
"""

""":md
## Understanding tile objects
Tiles returned by `hl.tile()` serve dual purposes: direct indexing with PyTorch operations and explicit
coordinate access for advanced operations.

#### Direct Tile Indexing
```python
for tile_m, tile_n in hl.tile([m, n]):
    # Tiles can be used directly as indices with PyTorch tensors
    data = input_tensor[tile_m, tile_n]
    output[tile_m, tile_n] = data * 2
```

Explicit Coordinate Access
```Py
for tile0 in hl.tile(num_rows):
    # Access the tile's coordinate with .index
    starts = x_offsets[tile0.index]      # Current tile position
    ends = x_offsets[tile0.index + 1]    # Next tile position

    # Use coordinates for manual indexing and bounds checking
    mask = tile1.index[None, :] < nnz[:, None]
    indices = starts[:, None] + tile1.index[None, :]
```

Programming Boundary: Use tiles directly with PyTorch operations when possible. Use `.index` when you need
explicit control over memory access patterns or complex indexing logic that PyTorch cant express directly.


### Understanding `hl.zeros()` and `hl.load()` with Masking

Before working with jagged tensors, lets learn about safe memory operations:

#### `hl.zeros(shape, dtype=...)`
Creates zero-initialized tensors within device loops. We ultimately want to allow users to directly create tile local tensors using `torch` operators but for now we use an explicit construct.

**Example:**
```python
for tile_m, tile_n in hl.tile([m, n]):
    accumulator = hl.zeros([tile_m, tile_n], dtype=torch.float32)
    # Use accumulator for reductions...
```

#### `hl.load(tensor, indices, extra_mask=...)`
Safe memory loading with bounds checking

**Example:**
```python
# Load with dynamic bounds
data = hl.load(
    source_tensor, 
    [start_idx + offset], 
    extra_mask=offset < actual_length  # Prevents out-of-bounds access
)
```
"""

""":md
# TODO Example 6: Jagged Tensor Addition

In many applications, we work with sparse data where each row has a different number of non-zero elements. A common storage format for this is the **jagged-row, prefix-sparse layout**.

## Understanding Jagged Tensors

A jagged tensor stores only the non-zero elements of each row in a compressed format:
- `x_data`: 1D tensor containing all non-zero elements concatenated row-by-row
- `x_offsets`: Index array of length `num_rows + 1` that provides random access to rows

For row `i`, the slice `x_data[x_offsets[i] : x_offsets[i+1]]` contains exactly the first `K_i` non-zero entries of that row.

**Example:**
```
Original sparse matrix:
[[1.0, 2.0, 0.0, 0.0],
 [3.0, 0.0, 0.0, 0.0], 
 [4.0, 5.0, 6.0, 0.0]]

Jagged representation:
x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
x_offsets = [0, 2, 3, 6]
```

## Your Task

Implement a kernel that adds the mean of a jagged sparse tensor to a dense matrix and returns the dense result.

**Key Steps:**
1. For each row, find start/end indices in x_data
2. Compute mean of non-zero elements for that row
3. Add mean to corresponding positions in dense matrix y
4. Handle dynamic loop bounds safely with `hl.load()` and masking

<details>
<summary>Click to expand hints</summary>  The hl.tile() function returns tile objects that represent chunks of work. Each tile has an .index attribute
   for accessing its position:

**Hints:**
- Use `hl.zeros()` for accumulators
- Use `hl.load()` with `extra_mask` for safe memory access
- Handle variable-length rows with dynamic bound
- In order to calculate NNZ in the kernel form the offset's tensor you will need to use tile.index to get the next tile number
- Calculate the Mean in first
- Use torch.where when writing output value
</details>
"""

""":py"""
def jagged_dense_add_spec(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation in pure PyTorch."""
    num_rows = x_offsets.numel() - 1
    assert y.shape[0] == num_rows
    out = y.clone()
    for i in range(num_rows):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        out[i, 0 : end - start] += torch.mean(x_data[start:end])
    return out

@helion.kernel(config=helion.Config(block_sizes=[1, 512, 512, 512], num_warps=8, num_stages=6))
def jagged_dense_add_kernel(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Add the mean of a jagged sparse tensor to a dense matrix.

    Args:
        x_data: 1D tensor with all non-zero elements row-by-row
        x_offsets: (num_rows + 1) tensor with row start/end indices
        y: (num_rows, N) dense tensor where N >= max(K_i)

    Returns:
        Dense tensor of shape (num_rows, N)
    """
    num_rows = y.size(0)
    out = torch.zeros_like(y)

    # ---- ✨ Your Code Here ✨----
    # TODO: Implement jagged tensor processing
    #
    # Key steps:
    # 1. Tile over jagged rows
    # 2. Use start and end indices to determine row boundaries,
    # 3. Initialize accumulator
    # 4. Loop over max_nnz with hl.load() and extra_mask
    # 5. Compute mean and add to output with proper masking
    # Hint use an outer and Two inner for loop over tiles
    
    # Tile over rows
    for tile0 in hl.tile(num_rows):
        # Get start and end indices for the current rows
        starts = x_offsets[tile0]
        ends = x_offsets[tile0.index + 1]
        nnz = ends - starts  # Number of non-zero elements per row
        max_nnz = nnz.amax()  # Maximum nnz in this tile
        
        # First, compute the mean for each row
        # Initialize accumulator for sum
        row_sum = hl.zeros([tile0], dtype=x_data.dtype)
        
        # Sum all elements in each row
        for tile1 in hl.tile(0, max_nnz):
            # Load data with bounds checking
            x_slice = hl.load(
                x_data,
                [starts[:, None] + tile1.index[None, :]],
                extra_mask=tile1.index[None, :] < nnz[:, None]
            )
            # Accumulate the sum
            row_sum += x_slice.sum(axis=1)
        
        # Compute mean by dividing sum by nnz
        # Handle case where nnz might be 0 to avoid division by zero
        row_mean = torch.where(nnz > 0, row_sum / nnz.float(), torch.zeros_like(row_sum))
        
        # Now add the mean to each element in the output
        # Process columns up to max_nnz (where we have jagged data)
        for tile1 in hl.tile(0, max_nnz):
            mask = tile1.index[None, :] < nnz[:, None]
            # Add mean to y where mask is true, otherwise just copy y
            out[tile0, tile1] = torch.where(
                mask, 
                y[tile0, tile1] + row_mean[:, None], 
                y[tile0, tile1]
            )
        
        # Copy remaining columns from y (beyond max_nnz)
        for tile1 in hl.tile(max_nnz, y.size(1)):
            out[tile0, tile1] = y[tile0, tile1]

    return out

def random_jagged_2d(
    num_rows: int,
    max_cols: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random jagged tensor data for testing."""
    # Random prefix lengths for each row
    lengths = torch.randint(1, max_cols + 1, (num_rows,), device=device)
    # Compute offsets via prefix sum
    x_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(lengths, dim=0)
    ])
    # Generate random non-zero data
    nnz = int(x_offsets[-1])
    x_data = torch.randn(nnz, dtype=dtype, device=device)
    return x_data, x_offsets

# Test the kernel
rows, cols = 64, 1024
x_data, x_offsets = random_jagged_2d(rows, cols, device="cuda")
y = torch.randn(rows, cols, device="cuda")

test_kernel(jagged_dense_add_kernel, jagged_dense_add_spec, x_data, x_offsets, y)
compare_implementations(jagged_dense_add_kernel, jagged_dense_add_spec, x_data, x_offsets, y)

""":md
## Conclusion

In this notebook, we've explored how to use Helion to write efficient GPU kernels using a high-level, PyTorch-like syntax. The key advantages of Helion include:

1. **Higher-level abstraction** than raw Triton, making it easier to write correct kernels
2. **Automatic tiling and memory management**, eliminating a common source of bugs
3. **Powerful autotuning** that can explore a wide range of implementations automatically
4. **Familiar PyTorch syntax** that builds on existing knowledge

(For Helion autotuning tutorial, please see https://www.internalfb.com/intern/anp/view/?id=7338830 "Autotuning in Helion" section.)

TODO: common kernel pattern mappings to examples in examples/ folder. Other APIs worth mentioning?

These example kernels should give you a good foundation for writing your own Helion kernels for a variety of applications.
"""

""":md
**If you are here for Helion Day (08/07/2025), please go to TODO spreadsheet link to choose a kernel to implement. Happy hacking!**
"""

""":py"""

