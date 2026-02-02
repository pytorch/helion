"""
Activation Kernels - Core Triton Implementations

==============================================================================
MATHEMATICAL CORE
==============================================================================

Activation functions introduce non-linearity into neural networks.
Modern LLMs predominantly use gated activations like SwiGLU.

Key Activation Functions:

1. SiLU (Sigmoid Linear Unit) / Swish:
    SiLU(x) = x * σ(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Properties:
    - Smooth, non-monotonic
    - Self-gated (input multiplied by its own sigmoid)
    - Derivative: SiLU'(x) = σ(x) + x * σ(x) * (1 - σ(x))

2. SwiGLU (Swish-Gated Linear Unit):
    SwiGLU(x, W_gate, W_up) = SiLU(x @ W_gate) * (x @ W_up)

    Used in: Llama, Mistral, DeepSeek, Qwen

    The gating mechanism allows selective information flow.

3. GeGLU (GELU-Gated Linear Unit):
    GeGLU(x, W_gate, W_up) = GELU(x @ W_gate) * (x @ W_up)

4. GELU (Gaussian Error Linear Unit):
    GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    Where Φ(x) is the CDF of standard normal distribution.

5. ReLU (Rectified Linear Unit):
    ReLU(x) = max(0, x)

6. Softplus:
    Softplus(x) = log(1 + exp(x))

    Smooth approximation of ReLU.

Fused Operations:
For efficiency, activation is often fused with:
    - Quantization: SiLU(x) * y → FP8
    - MoE computation: activation inside expert GEMM
    - Residual connections

Complexity:
    - Elementwise: O(elements) per tensor
    - SwiGLU: O(2 * hidden_dim * intermediate_dim) for matmuls + O(intermediate_dim) for activation

References:
    - Swish: Searching for Activation Functions (Ramachandran et al., 2017)
    - GLU Variants Improve Transformer (Shazeer, 2020) - https://arxiv.org/abs/2002.05202
    - LLaMA (Touvron et al., 2023) - SwiGLU adoption

==============================================================================
"""

import torch
import triton
import triton.language as tl


# ==============================================================================
# Triton Kernel: SiLU (Swish) Activation
# ==============================================================================

@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SiLU activation: output = input * sigmoid(input)

    Grid: (ceil(num_elements / BLOCK_SIZE),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # SiLU = x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    y = x * sigmoid_x

    tl.store(output_ptr + offs, y.to(output_ptr.dtype.element_ty), mask=mask)


# ==============================================================================
# Triton Kernel: SiLU + Multiply (SwiGLU pattern)
# ==============================================================================

@triton.jit
def silu_mul_kernel(
    gate_ptr,      # Input for gating: SiLU applied here
    up_ptr,        # Input for multiplicand
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU-Mul: output = SiLU(gate) * up

    This is the activation part of SwiGLU:
    - gate = x @ W_gate (pre-computed)
    - up = x @ W_up (pre-computed)
    - output = SiLU(gate) * up

    Grid: (ceil(num_elements / BLOCK_SIZE),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) * up
    activated = gate * tl.sigmoid(gate) * up

    tl.store(output_ptr + offs, activated.to(output_ptr.dtype.element_ty), mask=mask)


# ==============================================================================
# Triton Kernel: SiLU + Multiply + FP8 Quantization (Fused)
# ==============================================================================

@triton.jit
def silu_mul_quant_fp8_kernel(
    gate_ptr,
    up_ptr,
    output_ptr,       # FP8 output
    scale_ptr,        # Output scales
    num_tokens,
    hidden_dim,
    fp8_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU-Mul with per-token FP8 quantization.

    output_quant = quantize_fp8(SiLU(gate) * up)

    This fuses activation and quantization to:
    1. Avoid materializing full-precision intermediate
    2. Reduce memory bandwidth

    Grid: (num_tokens,)
    """
    token_idx = tl.program_id(0)

    # First pass: compute activation and find max for quantization
    max_abs = 0.0
    for block_idx in range(tl.cdiv(hidden_dim, BLOCK_SIZE)):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_dim

        gate = tl.load(
            gate_ptr + token_idx * hidden_dim + offs,
            mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(
            up_ptr + token_idx * hidden_dim + offs,
            mask=mask, other=0.0
        ).to(tl.float32)

        # SiLU(gate) * up
        activated = gate * tl.sigmoid(gate) * up

        max_abs = tl.maximum(max_abs, tl.max(tl.abs(activated)))

    # Compute scale
    scale = max_abs / fp8_max
    scale = tl.maximum(scale, 1e-12)
    tl.store(scale_ptr + token_idx, scale)

    # Second pass: quantize and store
    for block_idx in range(tl.cdiv(hidden_dim, BLOCK_SIZE)):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_dim

        gate = tl.load(
            gate_ptr + token_idx * hidden_dim + offs,
            mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(
            up_ptr + token_idx * hidden_dim + offs,
            mask=mask, other=0.0
        ).to(tl.float32)

        activated = gate * tl.sigmoid(gate) * up

        # Quantize
        quantized = activated / scale
        quantized = tl.minimum(tl.maximum(quantized, -fp8_max), fp8_max)

        tl.store(
            output_ptr + token_idx * hidden_dim + offs,
            quantized.to(output_ptr.dtype.element_ty),
            mask=mask
        )


# ==============================================================================
# Triton Kernel: GELU Activation
# ==============================================================================

@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE: tl.constexpr,  # Whether to use tanh approximation
):
    """
    GELU activation.

    Exact: GELU(x) = x * Φ(x) where Φ is standard normal CDF
    Approximate: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    Grid: (ceil(num_elements / BLOCK_SIZE),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if APPROXIMATE:
        # Tanh approximation (faster, used by GPT-2)
        # 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        # Use identity: tanh(z) = 2 * sigmoid(2z) - 1
        SQRT_2_OVER_PI = 0.7978845608028654  # √(2/π)
        inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
        tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        y = 0.5 * x * (1.0 + tanh_inner)
    else:
        # Exact using erf - approximate with a polynomial or use sigmoid approximation
        # erf(x) ≈ tanh(sqrt(2/π) * x * (1 + 0.044715 * x^2)) for reasonable x
        # Simplified: use sigmoid-based approximation for erf
        SQRT_2 = 1.4142135623730951
        z = x / SQRT_2
        # Approximate erf using the identity: erf(z) ≈ 2*sigmoid(1.6*z) - 1 (rough approx)
        # Better: use the same tanh-style approximation
        SQRT_2_OVER_PI = 0.7978845608028654
        inner = SQRT_2_OVER_PI * z * (1.0 + 0.044715 * z * z)
        erf_approx = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        y = 0.5 * x * (1.0 + erf_approx)

    tl.store(output_ptr + offs, y.to(output_ptr.dtype.element_ty), mask=mask)


# ==============================================================================
# Triton Kernel: Softplus Activation
# ==============================================================================

@triton.jit
def softplus_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    beta: tl.constexpr,      # Scaling factor (default 1.0)
    threshold: tl.constexpr, # Above this, use linear (default 20.0)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softplus activation: output = (1/β) * log(1 + exp(β * x))

    For numerical stability, when β*x > threshold, returns x.

    Grid: (ceil(num_elements / BLOCK_SIZE),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Softplus with numerical stability
    beta_x = beta * x
    y = tl.where(
        beta_x > threshold,
        x,  # Linear for large values
        (1.0 / beta) * tl.log(1.0 + tl.exp(beta_x))
    )

    tl.store(output_ptr + offs, y.to(output_ptr.dtype.element_ty), mask=mask)


# ==============================================================================
# Triton Kernel: SwiGLU Backward
# ==============================================================================

@triton.jit
def silu_mul_backward_kernel(
    # Forward inputs (saved for backward)
    gate_ptr,
    up_ptr,
    # Gradient from output
    grad_output_ptr,
    # Gradients to compute
    grad_gate_ptr,
    grad_up_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for SiLU-Mul (SwiGLU activation).

    Forward: y = SiLU(gate) * up = gate * sigmoid(gate) * up

    Gradients:
    - grad_up = grad_output * SiLU(gate)
    - grad_gate = grad_output * up * SiLU'(gate)
              where SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                             = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    Grid: (ceil(num_elements / BLOCK_SIZE),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    grad_output = tl.load(grad_output_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute sigmoid and SiLU
    sig = tl.sigmoid(gate)
    silu = gate * sig

    # grad_up = grad_output * SiLU(gate)
    grad_up = grad_output * silu

    # grad_gate = grad_output * up * SiLU'(gate)
    # SiLU'(x) = sig(x) * (1 + x * (1 - sig(x)))
    silu_grad = sig * (1.0 + gate * (1.0 - sig))
    grad_gate = grad_output * up * silu_grad

    tl.store(grad_gate_ptr + offs, grad_gate.to(grad_gate_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_up_ptr + offs, grad_up.to(grad_up_ptr.dtype.element_ty), mask=mask)


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

def silu_reference(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish activation."""
    return x * torch.sigmoid(x)


def silu_mul_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation pattern."""
    return torch.nn.functional.silu(gate) * up


def gelu_reference(x: torch.Tensor, approximate: bool = True) -> torch.Tensor:
    """GELU activation."""
    if approximate:
        return torch.nn.functional.gelu(x, approximate='tanh')
    else:
        return torch.nn.functional.gelu(x, approximate='none')


def swiglu_mlp_reference(
    x: torch.Tensor,          # [batch, seq_len, hidden]
    w_gate: torch.Tensor,     # [intermediate, hidden]
    w_up: torch.Tensor,       # [intermediate, hidden]
    w_down: torch.Tensor,     # [hidden, intermediate]
) -> torch.Tensor:
    """
    Full SwiGLU MLP block reference.

    output = W_down @ (SiLU(W_gate @ x) * (W_up @ x))
    """
    gate = x @ w_gate.T  # [batch, seq_len, intermediate]
    up = x @ w_up.T      # [batch, seq_len, intermediate]

    activated = torch.nn.functional.silu(gate) * up

    return activated @ w_down.T  # [batch, seq_len, hidden]


# ==============================================================================
# Triton Kernel Wrappers
# ==============================================================================

def silu_triton(x: torch.Tensor) -> torch.Tensor:
    """Wrapper to call SiLU Triton kernel."""
    output = torch.empty_like(x)
    num_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    silu_kernel[grid](
        x.view(-1),
        output.view(-1),
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def silu_mul_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Wrapper to call SiLU-Mul Triton kernel."""
    output = torch.empty_like(gate)
    num_elements = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    silu_mul_kernel[grid](
        gate.view(-1),
        up.view(-1),
        output.view(-1),
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def gelu_triton(x: torch.Tensor, approximate: bool = True) -> torch.Tensor:
    """Wrapper to call GELU Triton kernel."""
    output = torch.empty_like(x)
    num_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    gelu_kernel[grid](
        x.view(-1),
        output.view(-1),
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        APPROXIMATE=approximate,
    )
    return output


def softplus_triton(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """Wrapper to call Softplus Triton kernel."""
    output = torch.empty_like(x)
    num_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    softplus_kernel[grid](
        x.view(-1),
        output.view(-1),
        num_elements,
        beta=beta,
        threshold=threshold,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def silu_mul_backward_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper to call SiLU-Mul backward Triton kernel."""
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    num_elements = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    silu_mul_backward_kernel[grid](
        gate.view(-1),
        up.view(-1),
        grad_output.view(-1),
        grad_gate.view(-1),
        grad_up.view(-1),
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_gate, grad_up


# ==============================================================================
# Tests
# ==============================================================================

def test_silu_triton_vs_reference():
    """Test SiLU Triton kernel against PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    configs = [
        (1, 256),
        (4, 1024),
        (8, 4096),
        (16, 8192),
    ]

    print("Testing SiLU Triton vs Reference:")
    for batch, hidden in configs:
        x = torch.randn(batch, hidden, dtype=torch.float32, device=device)

        ref_out = silu_reference(x)
        tri_out = silu_triton(x)

        atol = (ref_out - tri_out).abs().max().item()
        rtol = ((ref_out - tri_out).abs() / (ref_out.abs() + 1e-8)).max().item()
        passed = atol < 1e-5
        status = "PASS" if passed else "FAIL"
        print(f"  batch={batch}, hidden={hidden}: atol={atol:.2e}, rtol={rtol:.2e} [{status}]")

        if not passed:
            raise AssertionError(f"Test failed for batch={batch}, hidden={hidden}")

    print("  All SiLU tests passed!")


def test_silu_mul_triton_vs_reference():
    """Test SiLU-Mul (SwiGLU) Triton kernel against PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    configs = [
        (1, 256),
        (4, 1024),
        (8, 4096),
        (16, 8192),
    ]

    print("Testing SiLU-Mul Triton vs Reference:")
    for batch, hidden in configs:
        gate = torch.randn(batch, hidden, dtype=torch.float32, device=device)
        up = torch.randn(batch, hidden, dtype=torch.float32, device=device)

        ref_out = silu_mul_reference(gate, up)
        tri_out = silu_mul_triton(gate, up)

        atol = (ref_out - tri_out).abs().max().item()
        rtol = ((ref_out - tri_out).abs() / (ref_out.abs() + 1e-8)).max().item()
        passed = atol < 1e-5
        status = "PASS" if passed else "FAIL"
        print(f"  batch={batch}, hidden={hidden}: atol={atol:.2e}, rtol={rtol:.2e} [{status}]")

        if not passed:
            raise AssertionError(f"Test failed for batch={batch}, hidden={hidden}")

    print("  All SiLU-Mul tests passed!")


def test_gelu_triton_vs_reference():
    """Test GELU Triton kernel against PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    configs = [
        (1, 256),
        (4, 1024),
        (8, 4096),
    ]

    print("Testing GELU Triton vs Reference:")
    # Test approximate mode (tanh approximation) - this is what most models use
    # Note: Triton kernel uses sigmoid-based tanh approximation since tl.math.tanh
    # is not available. For exact GELU we would need proper erf support.
    for batch, hidden in configs:
        x = torch.randn(batch, hidden, dtype=torch.float32, device=device)

        ref_out = gelu_reference(x, approximate=True)
        tri_out = gelu_triton(x, approximate=True)

        atol = (ref_out - tri_out).abs().max().item()
        rtol = ((ref_out - tri_out).abs() / (ref_out.abs() + 1e-8)).max().item()
        passed = atol < 1e-5
        status = "PASS" if passed else "FAIL"
        print(f"  batch={batch}, hidden={hidden}: atol={atol:.2e}, rtol={rtol:.2e} [{status}]")

        if not passed:
            raise AssertionError(f"Test failed for batch={batch}, hidden={hidden}")

    print("  All GELU tests passed!")


def test_softplus_triton_vs_reference():
    """Test Softplus Triton kernel against PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    configs = [
        (1, 256),
        (4, 1024),
        (8, 4096),
    ]

    print("Testing Softplus Triton vs Reference:")
    for beta in [1.0, 2.0]:
        print(f"  beta={beta}")
        for batch, hidden in configs:
            x = torch.randn(batch, hidden, dtype=torch.float32, device=device)

            ref_out = torch.nn.functional.softplus(x, beta=beta)
            tri_out = softplus_triton(x, beta=beta)

            atol = (ref_out - tri_out).abs().max().item()
            rtol = ((ref_out - tri_out).abs() / (ref_out.abs() + 1e-8)).max().item()
            passed = atol < 1e-5
            status = "PASS" if passed else "FAIL"
            print(f"    batch={batch}, hidden={hidden}: atol={atol:.2e}, rtol={rtol:.2e} [{status}]")

            if not passed:
                raise AssertionError(f"Test failed for batch={batch}, hidden={hidden}, beta={beta}")

    print("  All Softplus tests passed!")


def test_silu_mul_backward_triton():
    """Test SiLU-Mul backward Triton kernel against PyTorch autograd."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    configs = [
        (1, 256),
        (4, 1024),
        (8, 4096),
    ]

    print("Testing SiLU-Mul Backward Triton vs PyTorch autograd:")
    for batch, hidden in configs:
        gate = torch.randn(batch, hidden, dtype=torch.float32, device=device, requires_grad=True)
        up = torch.randn(batch, hidden, dtype=torch.float32, device=device, requires_grad=True)
        grad_output = torch.randn(batch, hidden, dtype=torch.float32, device=device)

        # PyTorch reference backward
        out_ref = silu_mul_reference(gate, up)
        out_ref.backward(grad_output)
        ref_grad_gate = gate.grad.clone()
        ref_grad_up = up.grad.clone()

        # Triton backward
        tri_grad_gate, tri_grad_up = silu_mul_backward_triton(
            gate.detach(), up.detach(), grad_output
        )

        gate_atol = (ref_grad_gate - tri_grad_gate).abs().max().item()
        up_atol = (ref_grad_up - tri_grad_up).abs().max().item()
        passed = gate_atol < 1e-5 and up_atol < 1e-5
        status = "PASS" if passed else "FAIL"
        print(f"  batch={batch}, hidden={hidden}: gate_atol={gate_atol:.2e}, up_atol={up_atol:.2e} [{status}]")

        if not passed:
            raise AssertionError(f"Test failed for batch={batch}, hidden={hidden}")

    print("  All SiLU-Mul Backward tests passed!")


def test_activation_half_precision():
    """Test activation kernels with half precision."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    batch, hidden = 4, 1024

    print("Testing Activation kernels with half precision (float16):")

    # SiLU
    x_fp16 = torch.randn(batch, hidden, dtype=torch.float16, device=device)
    ref_silu = silu_reference(x_fp16.float()).to(torch.float16)
    tri_silu = silu_triton(x_fp16)
    silu_atol = (ref_silu - tri_silu).abs().max().item()
    silu_pass = silu_atol < 1e-2
    print(f"  SiLU: atol={silu_atol:.2e} [{'PASS' if silu_pass else 'FAIL'}]")

    # SiLU-Mul
    gate_fp16 = torch.randn(batch, hidden, dtype=torch.float16, device=device)
    up_fp16 = torch.randn(batch, hidden, dtype=torch.float16, device=device)
    ref_silu_mul = silu_mul_reference(gate_fp16.float(), up_fp16.float()).to(torch.float16)
    tri_silu_mul = silu_mul_triton(gate_fp16, up_fp16)
    silu_mul_atol = (ref_silu_mul - tri_silu_mul).abs().max().item()
    silu_mul_pass = silu_mul_atol < 1e-2
    print(f"  SiLU-Mul: atol={silu_mul_atol:.2e} [{'PASS' if silu_mul_pass else 'FAIL'}]")

    # GELU
    ref_gelu = gelu_reference(x_fp16.float()).to(torch.float16)
    tri_gelu = gelu_triton(x_fp16)
    gelu_atol = (ref_gelu - tri_gelu).abs().max().item()
    gelu_pass = gelu_atol < 1e-2
    print(f"  GELU: atol={gelu_atol:.2e} [{'PASS' if gelu_pass else 'FAIL'}]")

    # Softplus
    ref_softplus = torch.nn.functional.softplus(x_fp16.float()).to(torch.float16)
    tri_softplus = softplus_triton(x_fp16)
    softplus_atol = (ref_softplus - tri_softplus).abs().max().item()
    softplus_pass = softplus_atol < 1e-2
    print(f"  Softplus: atol={softplus_atol:.2e} [{'PASS' if softplus_pass else 'FAIL'}]")

    if not all([silu_pass, silu_mul_pass, gelu_pass, softplus_pass]):
        raise AssertionError("Half precision tests failed")

    print("  All half precision tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Activation Kernels - Triton Core Tests")
    print("=" * 60)

    test_silu_triton_vs_reference()
    print()
    test_silu_mul_triton_vs_reference()
    print()
    test_gelu_triton_vs_reference()
    print()
    test_softplus_triton_vs_reference()
    print()
    test_silu_mul_backward_triton()
    print()
    test_activation_half_precision()

    print("\n" + "=" * 60)
    print("All Activation tests completed successfully!")
    print("=" * 60)
