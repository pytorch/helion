import torch
import triton

# Standalone imports - use local files
from forward import _attn_fwd
from backward import (
    _attn_bwd_preprocess,
    _attn_bwd_dk_dv,
    _attn_bwd_dq,
)

###################################### Flash Attention Class ######################################


class FlashAttention(torch.autograd.Function):
    """
    Flash Attention 2 for Self-Attention which supports:
    - Forward & Backward pass
    - Global Attention
    - Causal Attention
    - Sliding Window Attention
    - No dropout
    - Float16
    """

    @staticmethod
    def forward(ctx, Q, K, V, WINDOW_SIZE=None, attn_mode="global"):

        # Note: Q, K and V are the matrices after the linear layer in the attention.
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        HEAD_DIM_Q = Q.shape[-1]
        HEAD_DIM_K = K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        softmax_factor = 1 / HEAD_DIM**0.5

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert (
            attn_mode == "causal"
            or attn_mode == "global"
            or attn_mode == "sliding_window"
            or attn_mode == "causal_sliding_window"
        )
        if attn_mode == "sliding_window" or attn_mode == "causal_sliding_window":
            assert WINDOW_SIZE
            # assert WINDOW_SIZE < SEQ_LEN

        # Tensor where we will store the output
        O = torch.empty_like(Q)

        if attn_mode == "global":
            mode = 0
        elif attn_mode == "causal":
            mode = 1
        elif attn_mode == "sliding_window":
            mode = 2
        elif attn_mode == "causal_sliding_window":
            mode = 3

        # First Dim (X): Which group of queries are we going to work with. How many blocks of Q we have.
        # Second Dim (Y): Which head of which batch element we are going to work with
        # Third Dim (Z): We set it to 1 because we don't want to use it
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # Number of parallel programs (kernels): (BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q)

        # L is the logsumexp for the backward pass, one for each query
        L = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        # Self-Attention so the strides should be the same
        assert Q.stride() == K.stride() == V.stride() == O.stride()

        # Strides for the dimension of Q, K, V, O tensors are the same so we only pass them once (because its causal attention)
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_factor=softmax_factor,
            L=L,
            O=O,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM,
            WINDOW_SIZE=WINDOW_SIZE,
            MODE=mode,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.grid = grid
        ctx.softmax_factor = softmax_factor
        ctx.attn_mode = attn_mode
        ctx.WINDOW_SIZE = WINDOW_SIZE

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        # Ensure dO is contiguous (may be non-contiguous from transpose operations)
        if not dO.is_contiguous():
            dO = dO.contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        dQ = torch.empty_like(Q)
        dV = torch.empty_like(V)
        dK = torch.empty_like(K)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        preprocess_grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE"]),
            BATCH_SIZE * NUM_HEADS,
        )

        # (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
        D = torch.empty_like(L)

        # Compute all the D_i elements
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
        )

        # First Dim (X): The number of K, V blocks we will have rounded up
        # Second Dim (Y): Which head of which batch element we are going to work with
        # Third Dim (Z): We set it to 1 because we don't want to use it
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_KV"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        if ctx.attn_mode == "global":
            mode = 0
        elif ctx.attn_mode == "causal":
            mode = 1
        elif ctx.attn_mode == "sliding_window":
            mode = 2
        elif ctx.attn_mode == "causal_sliding_window":
            mode = 3

        # Unlike the paper, we do 2 for loops, one to compute dK, dV and the other to compute dQ
        # Also, this allows us to only write to the HBM once for each dK, dV, dQ (in the paper they write multiple times to the HBM for the each query blocks, which is slow)

        # K, V blocks are fixed, we itterate through all the Q blocks -> compute dK and dV
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_factor=ctx.softmax_factor,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            L=L,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            WINDOW_SIZE=ctx.WINDOW_SIZE,
            MODE=mode,
        )

        # Q block is fixed, we iterate through all the K, V blocks -> compute dQ
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_factor=ctx.softmax_factor,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            L=L,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            WINDOW_SIZE=ctx.WINDOW_SIZE,
            MODE=mode,
        )

        return dQ, dK, dV, None, None
