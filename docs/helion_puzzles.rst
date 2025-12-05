Helion Puzzles
==============

Programming for accelerators such as GPUs is critical for modern AI systems. This often means programming directly in proprietary low-level languages such as CUDA. Helion is a Python-embedded domain-specific language (DSL) for authoring machine learning kernels, designed to compile down to Triton, a performant backend for programming GPUs and other devices.

Helion aims to raise the level of abstraction compared to Triton, making it easier to write correct and efficient kernels while enabling more automation in the autotuning process.

This set of puzzles is meant to teach you how to use Helion from first principles in an interactive fashion. You will start with trivial examples and build your way up to real algorithms like Flash Attention and Quantized neural networks.

Setup
-----

First, let's install the necessary dependencies. Helion requires a recent version of PyTorch and a development version of Triton.

.. code-block:: python

    import logging

    import helion
    import helion.language as hl
    import torch
    from torch import Tensor

    # If you set this to info you will see the output Triton Code
    logging.getLogger().setLevel(logging.WARNING)

Let's also create a simple testing function to verify our implementations.

.. code-block:: python

    from triton.testing import do_bench
    def test_kernel(kernel_fn, spec_fn, *args):
        """Test a Helion kernel against a reference implementation."""
        # Run our implementation
        result = kernel_fn(*args)
        # Run reference implementation
        expected = spec_fn(*args)

        # Check if results match
        torch.testing.assert_close(result, expected)
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

Basic Structure of a Helion Kernel
---------------------------------

Helion allows you to write GPU kernels using familiar PyTorch syntax.

A Helion kernel has three main sections:

1. **Host Section** (CPU)
   This is standard PyTorch code executed on the CPU. Memory allocation, and shape computations are done here. Like with `Triton` and `Cuda` you need to setup your output buffers on the host before launching your kernel.

2. **Device Loop** (GPU Grid)
   `for tile in hl.tile(sizes)` - defines parallel execution across GPU thread blocks

3. **Device Operations** (GPU Kernel)
   PyTorch operations inside the loop - automatically compiled and fused

Example:

.. code-block:: python

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

Autotuning in Helion
--------------------

In the previous example, we explicitly specified a configuration using `config=helion.Config(block_sizes=[128, 128])`. This bypasses Helion's autotuning mechanism and uses our predefined settings. While this is quick to run, manually choosing optimal parameters can be challenging and hardware-dependent.

### What is Autotuning?

Autotuning is Helion's process of automatically finding the best configuration parameters for your specific:

- Hardware (GPU model)
- Problem size
- Operation patterns

When you omit the `config` parameter, Helion will automatically search for the optimal configuration:

.. code-block:: python

    @helion.kernel()  # No config = automatic tuning
    def autotuned_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
       m, n = x.size()
       out = torch.empty_like(x)
       for tile_m, tile_n in hl.tile([m, n]):
           out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]

Feel free to run the above code to see how much more performant it is than the original, although be warned it might take some time 😃

Now let's move on to our puzzles!

Puzzle 1: Constant Add
----------------------

Add a constant to a vector.

.. code-block:: python

    def add_spec(x: Tensor) -> Tensor:
        """This is the spec that you should implement in the helion kernel below."""
        return x + 10.

    # ---- ✨ Is this the best block size? ----
    @helion.kernel(config = helion.Config(block_sizes = [1,]))
    def add_kernel(x: torch.Tensor) -> torch.Tensor:
        # ---- ✨ Your Code Here ✨----
        # Set up the output buffer which you will return

        # Use Helion to tile the computation
        for tile_n in hl.tile(TILE_RANGE):
             # ---- ✨ Your Code Here ✨----

        return out

    # Test the kernel
    x = torch.randn(8192, device="cuda")
    test_kernel(add_kernel, add_spec, x)
    benchmark_kernel(add_kernel, x)
    compare_implementations(add_kernel, add_spec, x)

.. code-block:: python

    def add_spec(x: Tensor) -> Tensor:
        """This is the spec that you should implement."""
        return x + 10.

    # ---- ✨ Is this the best block size? ----
    @helion.kernel(config = helion.Config(block_sizes = [32,]))
    def add_kernel(x: torch.Tensor) -> torch.Tensor:
        # ---- ✨ Your Code Here ✨----
        # Set up the output buffer which you will return
        TILE_RANGE = x.size()
        out = torch.empty_like(x)
        # ---- End of Code ----

        # Use Helion to tile the computation
        for tile_n in hl.tile(TILE_RANGE):
             # ---- ✨ Your Code Here ✨----
            x_tile = x[tile_n]
            out[tile_n] = x_tile + 10.0

        return out

    # Test the kernel
    x = torch.randn(8192, device="cuda")
    test_kernel(add_kernel, add_spec, x)
    benchmark_kernel(add_kernel, x)
    compare_implementations(add_kernel, add_spec, x)

Puzzle 2: Outer Vector Add
--------------------------

Add two vectors using an outer product pattern.

.. code-block:: python

    def broadcast_add_spec(x: Tensor, y: Tensor) -> Tensor:
        return x[None, :] + y[:, None]

    # ---- ✨ Is this the best block size? ----
    @helion.kernel(config = helion.Config(block_sizes = [32, 32]))
    def broadcast_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Get tensor sizes
         # ---- ✨ Your Code Here ✨----
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

Puzzle 3: Fused Outer Multiplication
-----------------------------------

Multiply a row vector to a column vector and take a relu.

.. code-block:: python

    def mul_relu_block_spec(x: Tensor, y: Tensor) -> Tensor:
        return torch.relu(x[None, :] * y[:, None])

    # ---- ✨ Is this the best block size? ----
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

Puzzle 4: Fused Outer Multiplication - Backwards
------------------------------------------------

While PyTorch and torch.compile automatically generates the backwards pass for your Tensor Operations, Helion does not. So lets practice by writing the backwards function for a fused mul_relu kernel

.. code-block:: python

    def mul_relu_block_back_spec(x: Tensor, y: Tensor, dz: Tensor) -> Tensor:
        x = x.clone()
        y = y.clone()
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        z = torch.relu(x * y[:, None])
        grad_x, grad_y = torch.autograd.grad(z, [x, y], dz, retain_graph=True)
        return grad_x

    @helion.kernel(config=helion.Config(block_sizes=[32, 32]))
    def mul_relu_block_back_kernel(
        x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor
    ) -> torch.Tensor:
        # Get tensor sizes
        n0 = x.size(1)
        n1 = x.size(0)
        # Create output tensor for gradients
        dx = torch.empty_like(x)
        dy = torch.empty_like(y)

        # Use Helion to tile the computation
        for tile_i, tile_j in hl.tile([n1, n0]):
            # Get input tiles
            x_tile = x[tile_i, tile_j]
            y_tile = y[tile_i]
            dz_tile = dz[tile_i, tile_j]

            # Compute gradients for ReLU * multiplication backward
            # For ReLU, gradient is 1 where input > 0, 0 otherwise
            relu_mask = (x_tile * y_tile[:, None]) > 0
            # Chain rule: dx = dz * relu_grad * y
            dx[tile_i, tile_j] = dz_tile * relu_mask * y_tile[:, None]

        return dx, dy

    # Test the kernel
    x = torch.randn(512, 1024, device="cuda")
    y = torch.randn(512, device="cuda")
    dz = torch.randn(512, 1024, device="cuda")
    test_kernel(mul_relu_block_back_kernel, mul_relu_block_back_spec, x, y, dz)

Puzzle 7: Long Sum
-----------------

Sum of a batch of numbers.

.. code-block:: python

    def sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
        return x.sum(1)

    @helion.kernel()
    def sum_kernel(x: torch.Tensor) -> torch.Tensor:
        # Get tensor sizes
        batch, seq_len = x.size()
        # Create output tensor
        out = torch.empty(batch, dtype=x.dtype, device=x.device)

        # Use Helion to tile the batch dimension
        for tile_batch in hl.tile(batch):
            # Initialize accumulator for each batch element
            acc = torch.zeros(tile_batch, dtype=torch.float32, device=x.device)

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

Puzzle 8: Long Softmax
---------------------

Softmax of a batch of logits.

.. code-block:: python

    def softmax_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4 200"]:
        x_max = x.max(1, keepdim=True)[0]
        x = x - x_max
        x_exp = x.exp()
        return x_exp / x_exp.sum(1, keepdim=True)

    @helion.kernel()
    def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
        # Get tensor sizes
        batch, seq_len = x.size()
        # Create output tensor
        out = torch.empty_like(x)

        # Use Helion to tile the batch dimension
        for tile_batch in hl.tile(batch):
            # First pass: find max value for each sequence
            max_vals = torch.full_like(tile_batch, float('-inf'), dtype=torch.float32)

            for tile_seq in hl.tile(seq_len):
                chunk = x[tile_batch, tile_seq]
                max_vals = torch.maximum(max_vals, torch.max(chunk, dim=1)[0])

            # Second pass: compute sum of exp(x - max)
            sum_exp = torch.zeros_like(tile_batch, dtype=torch.float32)

            for tile_seq in hl.tile(seq_len):
                chunk = x[tile_batch, tile_seq]
                exp_vals = torch.exp(chunk - max_vals[:, None])
                sum_exp += torch.sum(exp_vals, dim=1)

            # Third pass: compute softmax
            for tile_seq in hl.tile(seq_len):
                chunk = x[tile_batch, tile_seq]
                exp_vals = torch.exp(chunk - max_vals[:, None])
                out[tile_batch, tile_seq] = exp_vals / sum_exp[:, None]

        return out

    # Test the kernel
    x = torch.randn(4, 200, device="cuda")
    test_kernel(softmax_kernel, softmax_spec, x)

Puzzle 9: Simple FlashAttention
-------------------------------

A scalar version of FlashAttention.

.. code-block:: python

    def flashatt_spec(q: Float32[Tensor, "200"], k: Float32[Tensor, "200"], v: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
        x = q[:, None] * k[None, :]
        x_max = x.max(1, keepdim=True)[0]
        x = x - x_max
        x_exp = x.exp()
        soft = x_exp / x_exp.sum(1, keepdim=True)
        return (v[None, :] * soft).sum(1)

    @helion.kernel()
    def flashatt_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Get tensor size
        seq_len = q.size(0)
        # Create output tensor
        out = torch.empty_like(q)

        # Process each query position
        for tile_q in hl.tile(seq_len):
            q_tile = q[tile_q]

            # Initialize tracking variables for stable softmax
            max_val = torch.full_like(q_tile, float('-inf'))
            sum_exp = torch.zeros_like(q_tile)
            weighted_sum = torch.zeros_like(q_tile)

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

Puzzle 10: Two Dimensional Convolution
--------------------------------------

A batched 2D convolution.

.. code-block:: python

    def conv2d_spec(x: Float32[Tensor, "4 8 8"], k: Float32[Tensor, "4 4"]) -> Float32[Tensor, "4 8 8"]:
        z = torch.zeros(4, 8, 8, device=x.device)
        x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
        for i in range(8):
            for j in range(8):
                z[:, i, j] = (k[None, :, :] * x[:, i: i+4, j: j + 4]).sum(1).sum(1)
        return z

    @helion.kernel()
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
                    # Apply the kernel
                    out[tile_batch, i, j] = (k[tile_batch,:,:] * patch).sum([1, 2])

        return out

    # Test the kernel
    x = torch.randn(4, 8, 8, device="cuda")
    k = torch.randn(4, 4, 4, device="cuda")
    test_kernel(conv2d_kernel, conv2d_spec, x, k)

Puzzle 11: Matrix Multiplication
-------------------------------

A blocked matrix multiplication.

.. code-block:: python

    def dot_spec(x: Float32[Tensor, "4 32 32"], y: Float32[Tensor, "4 32 32"]) -> Float32[Tensor, "4 32 32"]:
        return x @ y

    @helion.kernel()
    def dot_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Get tensor sizes
        batch, m, k = x.size()
        _, k, n = y.size()

        # Create output tensor
        out = torch.empty([batch, m, n], dtype=x.dtype, device=x.device)

        # Use Helion to tile the computation
        for tile_batch in hl.tile(batch):
            for tile_m, tile_n in hl.tile([m, n]):
                # Initialize accumulator
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

                # Process the reduction dimension in tiles
                for tile_k in hl.tile(k):
                    # Get tiles
                    x_tile = x[tile_batch, tile_m, tile_k]
                    y_tile = y[tile_batch, tile_k, tile_n]

                    # Accumulate matrix multiplication
                    acc = acc + torch.matmul(x_tile, y_tile)

                # Store result
                out[tile_batch, tile_m, tile_n] = acc

        return out

    # Test the kernel
    x = torch.randn(4, 32, 32, device="cuda")
    y = torch.randn(4, 32, 32, device="cuda")
    test_kernel(dot_kernel, dot_spec, x, y)

Puzzle 12: Quantized Matrix Multiplication
------------------------------------------

When doing matrix multiplication with quantized neural networks, a common strategy is to store the weight matrix in lower precision, with a shift and scale term.

.. code-block:: python

    FPINT = 32 // 4
    GROUP = 8

    def quant_dot_spec(scale: Float32[Tensor, "32 8"],
                       offset: Int32[Tensor, "32"],
                       weight: Int32[Tensor, "32 8"],
                       activation: Float32[Tensor, "64 32"]) -> Float32[Tensor, "32 32"]:
        offset = offset.view(32, 1)
        def extract(x):
            over = torch.arange(8, device=x.device) * 4
            mask = 2**4 - 1
            return (x[..., None] >> over) & mask
        scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
        offset = extract(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
        return (scale * (extract(weight).view(-1, 64) - offset)) @ activation

    @helion.kernel()
    def quant_dot_kernel(scale: torch.Tensor, offset: torch.Tensor, weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        # Get tensor sizes
        n_out, n_groups = scale.size()
        mid, n_in = activation.size()

        # Create output tensor
        out = torch.empty([n_out, n_in], dtype=scale.dtype, device=scale.device)

        # Helper function to extract 4-bit values
        def extract_4bit(x, bit_positions):
            mask = 2**4 - 1
            shifted = x[..., None] >> (bit_positions * 4)
            return shifted & mask

        # Bit positions for extraction
        bit_positions = torch.arange(8, device=scale.device)

        # Use Helion to tile the computation
        for tile_out in hl.tile(n_out):
            for tile_in in hl.tile(n_in):
                # Initialize accumulator
                acc = hl.zeros([tile_out, tile_in], dtype=torch.float32)

                # Get the offset values for this tile
                offset_tile = offset[tile_out]
                # Extract 4-bit values from offsets
                offset_extracted = extract_4bit(offset_tile, bit_positions)

                # Process in chunks across the middle dimension
                for group_idx in range(n_groups):
                    # Get scale for this group
                    scale_group = scale[tile_out, group_idx]

                    # Get weights for this group
                    weight_group = weight[tile_out, group_idx]

                    # Extract 4-bit values from weights
                    weight_extracted = extract_4bit(weight_group, bit_positions)

                    # Compute dequantized weights: scale * (weight - offset)
                    offset_group = offset_extracted[:, group_idx:group_idx+1]  # Shape: [tile_out, 1, 8]
                    dequant_weights = scale_group[:, None, None] * (weight_extracted - offset_group)

                    # Reshape dequantized weights for matrix multiplication
                    dequant_weights = dequant_weights.reshape(tile_out.size(0), 8)

                    # Get activations for this group
                    acts_idx = group_idx * 8 + torch.arange(8, device=scale.device)
                    act_group = activation[acts_idx][:, tile_in]

                    # Accumulate to result
                    acc = acc + torch.matmul(dequant_weights, act_group)

                # Store result
                out[tile_out, tile_in] = acc

        return out

    # Test the kernel with smaller inputs for quicker testing
    scale = torch.randn(32, 8, device="cuda")
    offset = torch.randint(-10, 10, (32,), device="cuda")
    weight = torch.randint(0, 16, (32, 8), device="cuda", dtype=torch.int32)
    activation = torch.randn(64, 32, device="cuda")
    test_kernel(quant_dot_kernel, quant_dot_spec, scale, offset, weight, activation)

Autotuning in Helion
--------------------

One of the major advantages of Helion is its sophisticated autotuning capability. Let's see how we can leverage this for our matrix multiplication kernel:

.. code-block:: python

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
                acc = acc + torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc

        return out

    # Create larger tensors for better autotuning results
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")

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

Hardcoding Configurations
-------------------------

After autotuning, you might want to hardcode the best configuration:

.. code-block:: python

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
                acc = acc + torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
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

Conclusion
----------

In this notebook, we've explored how to use Helion to write efficient GPU kernels using a high-level, PyTorch-like syntax. The key advantages of Helion include:

1. **Higher-level abstraction** than raw Triton, making it easier to write correct kernels
2. **Automatic tiling and memory management**, eliminating a common source of bugs
3. **Powerful autotuning** that can explore a wide range of implementations automatically
4. **Familiar PyTorch syntax** that builds on existing knowledge

These puzzles should give you a good foundation for writing your own Helion kernels for a variety of applications.
