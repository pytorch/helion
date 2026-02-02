import triton
import triton.language as tl


###################################### Backward Pass Flash Attention 2 Triton kernels ######################################


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE": BLOCK_SIZE},
        )
        for BLOCK_SIZE in [16, 32]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Index of the block in the sequence length to process
    block_index_q = tl.program_id(0)
    # Index representing the combination of batch and head to process. Each program handles one head in one batch.
    index_batch_head = tl.program_id(1)

    # array of indices which represent the offsets for the tokens in Q to process
    q_offsets = block_index_q * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # array of indices which represent the offsets on dimension. We need to load all the dimensions.
    dim_offsets = tl.arange(0, HEAD_DIM)

    # We load the O_block with the correct pointer (could also be done with tl.make_block_ptr()) -> (BLOCK_SIZE_Q, HEAD_DIM)
    # O: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    O_block = tl.load(
        O
        + index_batch_head * SEQ_LEN * HEAD_DIM
        + q_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )

    # We load a single block -> (BLOCK_SIZE_Q, HEAD_DIM)
    dO_block = tl.load(
        dO
        + index_batch_head * SEQ_LEN * HEAD_DIM
        + q_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    ).to(tl.float32)

    # Compute the D block -> (BLOCK_SIZE_Q,)
    D_block = tl.sum(dO_block * O_block, axis=1)
    D_block_ptrs = D + index_batch_head * SEQ_LEN + q_offsets

    tl.store(D_block_ptrs, D_block)


# Setting the config for autotune. Issue when giving different block size for Q and KV  (need to be solved)
configs = [
    triton.Config(
        {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for BLOCK_SIZE_Q in [16]
    for BLOCK_SIZE_KV in [16]
    for num_stages in [3, 4, 7]
    for num_warps in [2, 4]
]


@triton.autotune(configs, key=["SEQ_LEN", "HEAD_DIM"])
@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_factor,
    dO,
    dQ,
    dK,
    dV,
    L,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    MODE: tl.constexpr,
):

    # Index representing the combination of batch and head to process
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + index_head * stride_head).to(
        tl.int64
    )
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # put the pointers are the right place for the current batch and head
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dV += offset_batch_head
    dK += offset_batch_head

    # put the pointers are the right place for the current batch, head and sequence
    L += offset_batch_head_seq
    D += offset_batch_head_seq

    # array of indices which represent the offsets on dimension. We need to load all the dimensions.
    dim_offsets = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)

    # Index of the block in the sequence length to process
    start_kv = index_block_kv * BLOCK_SIZE_KV

    # array of indices which represent the offsets for the tokens in K and V sequence to process
    kv_offsets = start_kv + tl.arange(0, BLOCK_SIZE_KV)

    dV_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)

    # Load K and V in SRAM -> (BLOCK_SIZE_KV, HEAD_DIM)
    V_block = tl.load(
        V + kv_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    K_block = tl.load(
        K + kv_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )

    q_offsets = tl.arange(0, BLOCK_SIZE_Q)

    # We access Q as a transposed matrix -> (HEAD_DIM, BLOCK_SIZE_Q)
    # We point to the first BLOCK_SIZE_Q rows of Q for both q_T and dO pointers, inside the for loop we will move forward by BLOCK_SIZE_Q rows at each iteration
    Q_T_ptrs = Q + q_offsets[None, :] * stride_seq + dim_offsets[:, None] * stride_dim
    dO_ptrs = dO + q_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim

    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_SIZE_Q
    # Iterate over the SEQ_LEN of the Q matrix
    for _ in range(num_steps):

        # Load a block of Q transpose
        Q_T_block = tl.load(Q_T_ptrs)

        # load the logsumexp values for the queries in the current block
        q_offsets = curr_q + tl.arange(0, BLOCK_SIZE_Q)
        L_block = tl.load(L + q_offsets)

        # K(Q^T) = S^T
        S_T_block = softmax_factor * tl.dot(K_block, Q_T_block)
        # softmax with logsumexp trick, we need L^T
        P_T_block = tl.math.exp(S_T_block - L_block[None, :])

        if MODE == 1:  # Causal
            # Mask is TRUE for values that do not need to be masked -> (BLOCK_SIZE_KV, BLOCK_SIZE_Q)
            mask_block = q_offsets[None, :] >= kv_offsets[:, None]
            # Replace the masked values with 0
            # Not needed to mask with -inf before applying softmax as we already computed the normalization factors (stored in L)
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        elif MODE == 2:  # Sliding Window
            half_window = WINDOW_SIZE // 2
            mask_block = (q_offsets[None, :] + half_window >= kv_offsets[:, None]) & (
                q_offsets[None, :] - half_window <= kv_offsets[:, None]
            )
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        elif MODE == 3:  # Causal Sliding Window
            # q attends to k where (q >= k) AND (k >= q - window_size + 1)
            mask_block = (q_offsets[None, :] >= kv_offsets[:, None]) & (
                kv_offsets[:, None] >= q_offsets[None, :] - WINDOW_SIZE + 1
            )
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        # load dO block
        dO_block = tl.load(dO_ptrs)

        # Accumulate the dot product in dV_block (convert P_T_block to match dO_block dtype)
        dV_block = tl.dot(P_T_block.to(dO_block.dtype), dO_block, dV_block)

        # Di that we computed in the preprocessing step
        Di = tl.load(D + q_offsets)

        # We compute dP^T (not dP because we will need dS^T to update dK)
        dP_T_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # Element wise multiplication to get S^T
        dS_T_block = (dP_T_block - Di[None, :]) * P_T_block
        # Convert to match Q_T_block dtype for dot product
        dS_T_block = dS_T_block.to(Q_T_block.dtype)

        # We accumulate the dot product in dK_block
        dK_block += softmax_factor * tl.dot(dS_T_block, tl.trans(Q_T_block))

        # Note: tl.advance is usable only if the pointer was created using tl.make_ptr (not the case here)
        # Update the pointers
        curr_q += BLOCK_SIZE_Q
        Q_T_ptrs += BLOCK_SIZE_Q * stride_seq
        dO_ptrs += BLOCK_SIZE_Q * stride_seq

    # Store the computed dK and dV blocks
    dV_block_ptrs = (
        dV + kv_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = (
        dK + kv_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    tl.store(dK_block_ptrs, dK_block)


@triton.autotune(configs, key=["SEQ_LEN", "HEAD_DIM"])
@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_factor,
    dO,
    dQ,
    dK,
    dV,
    L,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    MODE: tl.constexpr,
):

    # Index representing the combination of batch and head to process
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (index_batch * stride_batch + index_head * stride_head).to(
        tl.int64
    )
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # put the pointers are the right place for the current batch and head
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dQ += offset_batch_head
    dV += offset_batch_head
    dK += offset_batch_head
    dO += offset_batch_head

    # put the pointers are the right place for the current batch, head and sequence
    L += offset_batch_head_seq
    D += offset_batch_head_seq

    # array of indices which represent the offsets on dimension. We need to load all the dimensions
    dim_offsets = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_q = index_block_kv * BLOCK_SIZE_Q
    q_offsets = start_q + tl.arange(0, BLOCK_SIZE_Q)

    Q_block = tl.load(
        Q + q_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    dQ_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + q_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )

    L_block = tl.load(L + q_offsets)
    L_block = L_block[:, None]

    kv_offsets = tl.arange(0, BLOCK_SIZE_KV)

    # We access the K and V as transposed blocks
    K_T_ptrs = K + kv_offsets[None, :] * stride_seq + dim_offsets[:, None] * stride_dim
    V_T_ptrs = V + kv_offsets[None, :] * stride_seq + dim_offsets[:, None] * stride_dim

    Di = tl.load(D + q_offsets)

    # We iterate over the K and V blocks
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_SIZE_KV

    for _ in range(num_steps):
        K_T_block = tl.load(K_T_ptrs)
        V_T_block = tl.load(V_T_ptrs)
        S_block = softmax_factor * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(S_block - L_block)

        if MODE == 1:  # Causal
            kv_offsets = curr_kv + tl.arange(0, BLOCK_SIZE_KV)
            mask_block = q_offsets[:, None] >= kv_offsets[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        elif MODE == 2:  # Sliding Window
            kv_offsets = curr_kv + tl.arange(0, BLOCK_SIZE_KV)
            half_window = WINDOW_SIZE // 2
            mask_block = (q_offsets[:, None] + half_window >= kv_offsets[None, :]) & (
                q_offsets[:, None] - half_window <= kv_offsets[None, :]
            )
            P_block = tl.where(mask_block, P_block, 0.0)

        elif MODE == 3:  # Causal Sliding Window
            kv_offsets = curr_kv + tl.arange(0, BLOCK_SIZE_KV)
            # q attends to k where (q >= k) AND (k >= q - window_size + 1)
            mask_block = (q_offsets[:, None] >= kv_offsets[None, :]) & (
                kv_offsets[None, :] >= q_offsets[:, None] - WINDOW_SIZE + 1
            )
            P_block = tl.where(mask_block, P_block, 0.0)

        # Compute dP and dS
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        # Convert to match K_T_block dtype for dot product
        dS_block = dS_block.to(K_T_block.dtype)

        # Accumulate the dot product in dQ_block
        dQ_block += softmax_factor * tl.dot(dS_block, tl.trans(K_T_block))

        # Update the pointers
        curr_kv += BLOCK_SIZE_KV
        K_T_ptrs += BLOCK_SIZE_KV * stride_seq
        V_T_ptrs += BLOCK_SIZE_KV * stride_seq

    dQ_block_ptrs = (
        dQ + q_offsets[:, None] * stride_seq + dim_offsets[None, :] * stride_dim
    )
    # Single write to the HBM (compared to multiple writes in the loop in the paper)
    tl.store(dQ_block_ptrs, dQ_block)
