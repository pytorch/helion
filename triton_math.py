# Triton Math Functions
# Extracted from mkl/ops/triton/math.py

import triton
import triton.language as tl


@triton.jit
def tanh(x):
    """Tanh is just a scaled sigmoid"""
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def tanh_approx_fp32(x):
    """PTX-based fast tanh approximation for FP32"""
    output = tl.inline_asm_elementwise(
        asm="""
            tanh.approx.f32 $0, $1;
            """,
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return output


@triton.jit
def fast_gelu(x):
    """Fast GELU activation using tanh approximation"""
    return x * 0.5 * (1 + tanh_approx_fp32(0.7978845608 * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def fast_gelu_grad(x):
    """Gradient of fast GELU"""
    tanh_out = tanh_approx_fp32(0.7978845608 * x * (1.0 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


@triton.jit
def raw(x):
    """Identity activation"""
    return x


@triton.jit
def raw_grad(x):
    """Gradient of identity activation"""
    return tl.full(x.shape, 1.0, x.dtype)
