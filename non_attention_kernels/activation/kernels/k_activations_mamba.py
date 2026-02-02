# Copyright (c) 2024, Tri Dao, Albert Gu.
# Source: mamba_ssm/ops/triton/k_activations.py
"""
SwiGLU activation kernels from Mamba.

SwiGLU (Swish-Gated Linear Unit) is a gated activation function that combines
SiLU (Swish) with a gating mechanism:
    output = x * sigmoid(x) * y = silu(x) * y

This is used extensively in transformer FFN layers and Mamba architectures.
"""
import torch

import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(
    X,
    Y,
    OUT,
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_out_row,
    ncols,
    BLOCK_N: tl.constexpr,
):
    """
    SwiGLU forward kernel: out = x * sigmoid(x) * y = silu(x) * y

    Grid: (M, cdiv(N, BLOCK_N))
    where M = number of rows, N = number of columns
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    OUT += row * stride_out_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < ncols)


@triton.jit
def _swiglu_bwd_kernel(
    X,
    Y,
    DOUT,
    OUT,
    DX,
    DY,
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_dout_row,
    stride_out_row,
    stride_dx_row,
    stride_dy_row,
    ncols,
    BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    """
    SwiGLU backward kernel.

    Computes gradients for x and y given the output gradient dout:
    - dx = sigmoid(x) * (1 + x * (1 - sigmoid(x))) * y * dout
    - dy = x * sigmoid(x) * dout

    Optionally recomputes the forward output if RECOMPUTE_OUTPUT is True.

    Grid: (M, cdiv(N, BLOCK_N))
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    DOUT += row * stride_dout_row
    if RECOMPUTE_OUTPUT:
        OUT += row * stride_out_row
    DX += row * stride_dx_row
    DY += row * stride_dy_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=cols < ncols, other=0.).to(tl.float32)
    x_sigmoid = tl.sigmoid(x)
    dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
    dy = x * x_sigmoid * dout
    tl.store(DX + cols, dx, mask=cols < ncols)
    tl.store(DY + cols, dy, mask=cols < ncols)
    if RECOMPUTE_OUTPUT:
        out = x * x_sigmoid * y
        tl.store(OUT + cols, out, mask=cols < ncols)


def _swiglu_fwd(xy, out=None):
    """
    SwiGLU forward pass.

    Args:
        xy: Input tensor of shape [..., 2*N] where first half is x, second half is y
        out: Optional output tensor of shape [..., N]

    Returns:
        Output tensor of shape [..., N] where out = silu(x) * y
    """
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)
    if out is None:
        out = torch.empty_like(x)
    else:
        out = out.reshape(-1, out.shape[-1])
        assert out.shape == x.shape
    assert out.stride(-1) == 1
    M, N = x.shape
    # Use fixed block sizes for simplicity (original uses autotune)
    BLOCK_N = min(1024, triton.next_power_of_2(N))
    grid = (M, triton.cdiv(N, BLOCK_N))
    with torch.cuda.device(x.device.index):
        _swiglu_fwd_kernel[grid](x, y, out, x.stride(0), y.stride(0), out.stride(0), N, BLOCK_N=BLOCK_N)
    return out.reshape(*batch_shape, out.shape[-1])


def _swiglu_bwd(xy, dout, dxy=None, recompute_output=False, out=None):
    """
    SwiGLU backward pass.

    Args:
        xy: Input tensor from forward pass
        dout: Gradient of the output
        dxy: Optional output tensor for input gradients
        recompute_output: Whether to recompute forward output
        out: Optional output tensor for recomputed forward output

    Returns:
        dxy: Gradient tensor of shape [..., 2*N]
        out (optional): Recomputed forward output if recompute_output is True
    """
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)
    dout = dout.reshape(-1, dout.shape[-1])
    assert dout.shape == x.shape
    if dxy is None:
        dxy = torch.empty_like(xy)
    else:
        dxy = dxy.reshape(-1, dxy.shape[-1])
        assert dxy.shape == xy.shape
    dx, dy = dxy.chunk(2, dim=-1)
    assert dx.stride(-1) == 1
    assert dy.stride(-1) == 1
    if recompute_output:
        if out is None:
            out = torch.empty_like(x)
        else:
            out = out.reshape(-1, out.shape[-1])
            assert out.shape == x.shape
        assert out.stride(-1) == 1
    M, N = x.shape
    BLOCK_N = min(1024, triton.next_power_of_2(N))
    grid = (M, triton.cdiv(N, BLOCK_N))
    with torch.cuda.device(x.device.index):
        _swiglu_bwd_kernel[grid](
            x, y, dout, out if recompute_output else None, dx, dy,
            x.stride(0), y.stride(0), dout.stride(0),
            out.stride(0) if recompute_output else 0,
            dx.stride(0), dy.stride(0),
            N,
            BLOCK_N=BLOCK_N,
            RECOMPUTE_OUTPUT=recompute_output
        )
    if not recompute_output:
        return dxy.reshape(*batch_shape, dxy.shape[-1])
    else:
        return dxy.reshape(*batch_shape, dxy.shape[-1]), out.reshape(*batch_shape, out.shape[-1])


class SwiGLU(torch.autograd.Function):
    """
    SwiGLU activation function with autograd support.

    Forward:  out = silu(x) * y = x * sigmoid(x) * y
    Backward: dx = d_silu(x) * y * dout, dy = silu(x) * dout
    """

    @staticmethod
    def forward(ctx, xy):
        ctx.save_for_backward(xy)
        return _swiglu_fwd(xy)

    @staticmethod
    def backward(ctx, dout):
        xy, = ctx.saved_tensors
        return _swiglu_bwd(xy, dout)


swiglu = SwiGLU.apply
