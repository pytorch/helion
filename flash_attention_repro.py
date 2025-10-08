from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

@triton.jit
def _helion_attention(q_view, k_view, v_view, out, _BLOCK_SIZE_1: tl.constexpr, _RDIM_SIZE_2: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    num_blocks_0 = 192
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    indices_4 = tl.arange(0, _RDIM_SIZE_2).to(tl.int32)
    m_i = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], float('-inf'), tl.float32)
    l_i = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 1.0, tl.float32)
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1, 128], 0.0, tl.float32)
    q = tl.load(q_view + (indices_0[:, None, None] * 32768 + indices_1[None, :, None] * 128 + indices_4[None, None, :] * 1), None)
    for offset_2 in tl.range(0, 256, _BLOCK_SIZE_3):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
        q_copy = q
        m_i_copy = m_i
        l_i_copy = l_i
        acc_copy = acc
        q_copy_0 = q_copy
        m_i_copy_0 = m_i_copy
        l_i_copy_0 = l_i_copy
        acc_copy_0 = acc_copy
        k = tl.load(k_view + (indices_0[:, None, None] * 32768 + indices_4[None, :, None] * 1 + indices_2[None, None, :] * 128), None)
        qk = tl.reshape(tl.dot(tl.reshape(tl.cast(q_copy_0, tl.bfloat16), [_BLOCK_SIZE_1, 128]), tl.reshape(tl.cast(k, tl.bfloat16), [128, _BLOCK_SIZE_3]), input_precision='tf32', out_dtype=tl.float32), [_BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_3])
        amax = tl.cast(tl.max(qk, 2), tl.bfloat16)
        v_0 = 0.12751743074602467
        v_1 = amax * v_0
        v_2 = tl.cast(v_1, tl.float32)
        v_3 = triton_helpers.maximum(m_i_copy_0, v_2)
        v_4 = 0.12751743074602467
        v_5 = qk * v_4
        subscript = v_3[:, :, None]
        v_6 = tl.cast(v_5, tl.float32)
        v_7 = v_6 - subscript
        v_8 = libdevice.exp2(v_7)
        l_ij = tl.cast(tl.sum(v_8, 2), tl.float32)
        v_9 = m_i_copy_0 - v_3
        v_10 = libdevice.exp2(v_9)
        v_11 = l_i_copy_0 * v_10
        l_i = v_11 + l_ij
        subscript_1 = v_10[:, :, None]
        v_13 = acc_copy_0 * subscript_1
        v = tl.load(v_view + (indices_0[:, None, None] * 32768 + indices_2[None, :, None] * 128 + indices_4[None, None, :] * 1), None)
        v_14 = tl.cast(v_8, tl.bfloat16)
        acc = tl.reshape(tl.dot(tl.reshape(tl.cast(v_14, tl.bfloat16), [_BLOCK_SIZE_1, _BLOCK_SIZE_3]), tl.reshape(tl.cast(v, tl.bfloat16), [_BLOCK_SIZE_3, 128]), acc=tl.reshape(v_13, [_BLOCK_SIZE_1, 128]), input_precision='tf32', out_dtype=tl.float32), [_BLOCK_SIZE_0, _BLOCK_SIZE_1, 128])
        m_i = v_3
    subscript_2 = l_i[:, :, None]
    v_15 = acc / subscript_2
    v_16 = tl.cast(v_15, tl.bfloat16)
    tl.store(out + (indices_0[:, None, None] * 32768 + indices_1[None, :, None] * 128 + indices_4[None, None, :] * 1), v_16, None)

def attention(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, *, _launcher=_default_launcher):
    """
    Computes scaled dot-product attention.

    Implements the attention mechanism: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        q_in: Query tensor of shape [..., seq_len_q, head_dim]
        k_in: Key tensor of shape [..., seq_len_k, head_dim]
        v_in: Value tensor of shape [..., seq_len_k, head_dim]

    Returns:
        Output tensor of shape [..., seq_len_q, head_dim]
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 128
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    _BLOCK_SIZE_1 = 256
    _RDIM_SIZE_2 = 128
    _BLOCK_SIZE_3 = 32
    _launcher(_helion_attention, (192 * triton.cdiv(256, _BLOCK_SIZE_1),), q_view, k_view, v_view, out, _BLOCK_SIZE_1, _RDIM_SIZE_2, 1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
    return out.view(q_in.size())

if __name__ == "__main__":
    tensor_specs = (
        {
            "device": "cuda:0",
            "dtype": "torch.bfloat16",
            "shape": (4, 48, 256, 128),
            "stride": (1572864, 32768, 128, 1),
        },
        {
            "device": "cuda:0",
            "dtype": "torch.bfloat16",
            "shape": (4, 48, 256, 128),
            "stride": (1572864, 32768, 128, 1),
        },
        {
            "device": "cuda:0",
            "dtype": "torch.bfloat16",
            "shape": (4, 48, 256, 128),
            "stride": (1572864, 32768, 128, 1),
        },
    )

    if not torch.cuda.is_available():
        raise RuntimeError("flash_attention_repro requires CUDA to reproduce the kernel launch.")

    def _make_tensor(spec: dict[str, object]) -> torch.Tensor:
        dtype_str = spec["dtype"]
        if not isinstance(dtype_str, str) or not dtype_str.startswith("torch."):
            raise ValueError(f"Unsupported dtype format: {dtype_str!r}")
        dtype_attr = dtype_str.split(".", 1)[1]
        try:
            dtype = getattr(torch, dtype_attr)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype attribute: {dtype_attr}") from exc

        tensor = torch.randn(
            spec["shape"],
            dtype=dtype,
            device=torch.device(spec["device"]),
        )

        # Sanity check to make sure the recorded stride matches what we create.
        expected_stride = spec.get("stride")
        if expected_stride is not None and tuple(tensor.stride()) != tuple(expected_stride):
            raise ValueError(
                f"Unexpected stride {tensor.stride()} (expected {expected_stride})."
            )
        return tensor

    q_tensor, k_tensor, v_tensor = (_make_tensor(spec) for spec in tensor_specs)
    out = attention(q_tensor, k_tensor, v_tensor)
    print({"out.shape": out.shape, "out.dtype": out.dtype, "out.device": out.device})
