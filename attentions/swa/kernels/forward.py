import triton
import triton.language as tl

###################################### Forward Pass Flash Attention 2 Triton kernels ######################################


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [16]
        for BLOCK_SIZE_KV in [16]
        for num_stages in [3, 4, 7]
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    softmax_factor,
    L,
    O,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    MODE: tl.constexpr,
):
    """
    Q, K, V, O --> (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    L --> (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
    """

    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    # Index of the block in the sequence length to process
    block_index_q = tl.program_id(0)
    # Index representing the combination of batch and head to process. Each program handles one head in one batch.
    index_batch_head = tl.program_id(1)
    # Extract the batch index from the combined batch-head index (assuming each batch contains NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # Extract the head index within the current batch
    index_head = index_batch_head % NUM_HEADS

    # Offset to get the (SEQ_LEN, HEAD_DIM) block in Q, K, V with the index batch and head
    qvk_start = (
        index_batch.to(tl.int64) * stride_batch + index_head.to(tl.int64) * stride_head
    )

    # Note: Q, K, V are pointers
    # Pointer to the current Q block: Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_start,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # Note: Each program iterate through all the keys and values so the offsets are (0, 0).

    # Pointer to the current V block: V[index_batch, index_head, :, :]
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_start,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    # Pointer to the current K block: K[index_batch, index_head, :, :]
    K_T_block_ptr = tl.make_block_ptr(
        base=K + qvk_start,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_dim,
            stride_seq,
        ),  # We invert the strides to transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    # Pointer to the current O block: O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_start,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # array of indices which represent the offsets for the tokens in Q to process
    q_offsets = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # array of indices which represent the offsets for the tokens in K and V sequence to process
    kv_offsets = tl.arange(0, BLOCK_SIZE_KV)

    # maximum for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    # sum for each query (we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # accumulator for the output. A group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the Q block in SRAM
    Q_block = tl.load(Q_block_ptr)

    # The reason we don't fuse the for loops to the left of the diagonal with the one exactly on the diagonal
    # for the causal attention is to optimize the pipelining that Triton does (Same for sliding window)

    if MODE == 0:  # Global
        # Stage 1: Runs all the blocks.
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            WINDOW_SIZE,
            q_offsets,
            kv_offsets,
            SEQ_LEN,
            MODE,
            1,
        )

    elif MODE == 1:  # Causal
        # Stage 1: Runs for the blocks on the left of the main diagonal (full attention)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            WINDOW_SIZE,
            q_offsets,
            kv_offsets,
            SEQ_LEN,
            MODE,
            1,
        )
        # Stage 2: Runs for the blocks in the main diagonal (Transition between non-masked and masked keys)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            WINDOW_SIZE,
            q_offsets,
            kv_offsets,
            SEQ_LEN,
            MODE,
            2,
        )

    elif MODE == 2:  # Sliding Window
        # Stage 1: Runs for the blocks in between the two diagonals (full attention)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            WINDOW_SIZE,
            q_offsets,
            kv_offsets,
            SEQ_LEN,
            MODE,
            1,
        )
        # Stage 2: Blocks on the right side of the sliding window (Transition between non-masked and masked keys)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            WINDOW_SIZE,
            q_offsets,
            kv_offsets,
            SEQ_LEN,
            MODE,
            2,
        )
        # Stage 3: Blocks on the left side of the sliding window (Transition between non-masked and masked keys)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            WINDOW_SIZE,
            q_offsets,
            kv_offsets,
            SEQ_LEN,
            MODE,
            3,
        )

    elif MODE == 3:  # Causal Sliding Window
        # Query at position q attends to keys in [max(0, q - window_size + 1), q]
        # Stage 1: Runs for the blocks fully within the window (full attention within causal window)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            WINDOW_SIZE,
            q_offsets,
            kv_offsets,
            SEQ_LEN,
            MODE,
            1,
        )
        # Stage 2: Blocks on the causal diagonal (Transition at the causal boundary)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            WINDOW_SIZE,
            q_offsets,
            kv_offsets,
            SEQ_LEN,
            MODE,
            2,
        )
        # Stage 3: Blocks on the left window boundary (Transition at the sliding window left edge)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_T_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_factor,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            WINDOW_SIZE,
            q_offsets,
            kv_offsets,
            SEQ_LEN,
            MODE,
            3,
        )

    # Broadcasting so it's done element wise just like with a diag matrix
    O_block = O_block / l_i[:, None]

    # Pointer to the correct L block. We need to skip by SEQ_LEN for each head in each batch and by q_offsets to select the correct tokens for the current query block.
    L_block_ptr = L + index_batch_head * SEQ_LEN + q_offsets

    # logsumexp for the backward pass
    tl.store(L_block_ptr, m_i + tl.math.log(l_i))
    # Save O_block to the correct location pointed to by O_block_ptr
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_T_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_factor,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    q_offsets: tl.constexpr,
    kv_offsets: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    MODE: tl.constexpr,
    STAGE: tl.constexpr,
):
    """
    Triton kernel for the inner loop of the forward pass of Flash Attention
    """

    # Various stages for each mode

    if MODE == 0:  # Global
        if STAGE == 1:
            # All the blocks
            lower, higher = 0, SEQ_LEN

    elif MODE == 1:  # Causal
        if STAGE == 1:
            # Blocks on the left of the middle diagonal
            lower, higher = 0, block_index_q * BLOCK_SIZE_Q

        elif STAGE == 2:
            # Blocks in the diagonal, where there is transition between non-masked and masked keys
            lower = block_index_q * BLOCK_SIZE_Q
            higher = (block_index_q + 1) * BLOCK_SIZE_Q
            lower = tl.multiple_of(lower, BLOCK_SIZE_Q)

    elif MODE == 2:  # Sliding Window
        # Compute the number of blocks to attend on each side of the main diagonal
        half_window = WINDOW_SIZE // 2
        window_block_index = tl.cdiv(1 + half_window, BLOCK_SIZE_Q)
        NUM_BLOCKS_Q = tl.cdiv(SEQ_LEN, BLOCK_SIZE_Q)

        if STAGE == 1:
            # Blocks in between the right and left diagonals, where there is full attention
            if (half_window + 1) % BLOCK_SIZE_Q == 0:
                lower = max(0, block_index_q - window_block_index + 1) * BLOCK_SIZE_Q
                higher = (
                    min(NUM_BLOCKS_Q, block_index_q + window_block_index) * BLOCK_SIZE_Q
                )
            else:
                lower = max(0, block_index_q - window_block_index + 2) * BLOCK_SIZE_Q
                higher = (
                    min(NUM_BLOCKS_Q, block_index_q + window_block_index - 1)
                    * BLOCK_SIZE_Q
                )

        elif STAGE == 2:
            # Blocks on the right side of the sliding window, where there is transition between non-masked and masked keys
            # Note: if half_window is not divisible by BLOCK_SIZE_Q then there are 2 diagonals to handle at this stage
            # if half_window is divisible by BLOCK_SIZE_Q then there is only a single diagonal to handle at this stage
            if (half_window + 1) % BLOCK_SIZE_Q == 0:
                lower = (
                    min(NUM_BLOCKS_Q, block_index_q + window_block_index) * BLOCK_SIZE_Q
                )
            else:
                lower = (
                    min(NUM_BLOCKS_Q, block_index_q + window_block_index - 1)
                    * BLOCK_SIZE_Q
                )

            higher = (
                min(NUM_BLOCKS_Q, block_index_q + window_block_index + 1) * BLOCK_SIZE_Q
            )

        elif STAGE == 3:
            # Blocks on the left side of the sliding window, where there is transition between non-masked and masked keys
            # Note: if half_window is not divisible by BLOCK_SIZE_Q then there are 2 diagonals do deal with on the left side
            # if half_window is divisible by BLOCK_SIZE_Q then there is only a single diagonal to handle at this stage
            if (half_window + 1) % BLOCK_SIZE_Q == 0:
                higher = max(0, block_index_q - window_block_index + 1) * BLOCK_SIZE_Q
            else:
                higher = max(0, block_index_q - window_block_index + 2) * BLOCK_SIZE_Q

            lower = (
                max(0, block_index_q - tl.cdiv(half_window, BLOCK_SIZE_Q))
                * BLOCK_SIZE_Q
            )

        lower = tl.multiple_of(lower, BLOCK_SIZE_Q)
        higher = tl.multiple_of(higher, BLOCK_SIZE_Q)

    elif MODE == 3:  # Causal Sliding Window
        # Query at position q attends to keys in [max(0, q - window_size + 1), q]
        # window_size here is the total window size (not half)
        window_block_index = tl.cdiv(WINDOW_SIZE, BLOCK_SIZE_Q)
        NUM_BLOCKS_Q = tl.cdiv(SEQ_LEN, BLOCK_SIZE_Q)

        if STAGE == 1:
            # Blocks fully within the causal sliding window (between left window edge and causal diagonal)
            # Lower bound: max(0, q_block - window_blocks + 2) to ensure full blocks
            # Upper bound: q_block to stay left of causal diagonal
            if WINDOW_SIZE % BLOCK_SIZE_Q == 0:
                lower = max(0, block_index_q - window_block_index + 1) * BLOCK_SIZE_Q
            else:
                lower = max(0, block_index_q - window_block_index + 2) * BLOCK_SIZE_Q
            higher = block_index_q * BLOCK_SIZE_Q

        elif STAGE == 2:
            # Blocks on the causal diagonal (where q >= k condition transitions)
            lower = block_index_q * BLOCK_SIZE_Q
            higher = (block_index_q + 1) * BLOCK_SIZE_Q

        elif STAGE == 3:
            # Blocks on the left window boundary (where k >= q - window_size + 1 transitions)
            if WINDOW_SIZE % BLOCK_SIZE_Q == 0:
                higher = max(0, block_index_q - window_block_index + 1) * BLOCK_SIZE_Q
            else:
                higher = max(0, block_index_q - window_block_index + 2) * BLOCK_SIZE_Q
            lower = max(0, block_index_q - window_block_index) * BLOCK_SIZE_Q

        lower = tl.multiple_of(lower, BLOCK_SIZE_Q)
        higher = tl.multiple_of(higher, BLOCK_SIZE_Q)

    # Adjust the pointer to the correct block
    K_T_block_ptr = tl.advance(
        K_T_block_ptr, (0, lower)
    )  # offset is inverted because K is transposed
    V_block_ptr = tl.advance(V_block_ptr, (lower, 0))

    # LOOP over all the keys and values blocks and store inside the accumulator: O_block

    for start_kv in range(lower, higher, BLOCK_SIZE_KV):
        # Let the compiler know that start_kv is a multiple of BLOCK_SIZE_KV, so the compiler can do optimization
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # load the K block in SRAM and compute the dot product
        K_T_block = tl.load(K_T_block_ptr)
        # (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
        S_block = tl.dot(Q_block, K_T_block)

        if MODE == 1 and STAGE == 2:  # Causal - Stage 2
            # Build the causal mask for the S_block
            mask = q_offsets[:, None] >= (start_kv + kv_offsets[None, :])
            # Apply causal mask by adding -1.0e6 to masked positions (upper triangle)
            S_block = S_block * softmax_factor + tl.where(mask, 0, -1.0e6)
            # max per row for stability (log-sum-exp trick)
            m_ij = tl.maximum(m_i, tl.max(S_block, 1))
            # subtracting the row-wise max for numerical stability
            S_block -= m_ij[:, None]

        elif MODE == 2 and (STAGE == 2 or STAGE == 3):  # Sliding Window - Stage 2, 3
            half_window = WINDOW_SIZE // 2

            if STAGE == 2:  # Right diagonal so the upper triangle part is masked
                # TRUE for the valid tokens
                mask = (
                    q_offsets[:, None] + half_window >= start_kv + kv_offsets[None, :]
                )

            elif STAGE == 3:  # Left diagonal so the lower triangle part is masked
                # TRUE for the valid tokens
                mask = (
                    q_offsets[:, None] - half_window <= start_kv + kv_offsets[None, :]
                )

            # Apply mask by adding -1.0e6 to False positions
            S_block = S_block * softmax_factor + tl.where(mask, 0, -1.0e6)
            # max per row for stability (log-sum-exp trick)
            m_ij = tl.maximum(m_i, tl.max(S_block, 1))
            # subtracting the row-wise max for numerical stability
            S_block -= m_ij[:, None]

        elif MODE == 3 and (STAGE == 2 or STAGE == 3):  # Causal Sliding Window - Stage 2, 3
            # For causal sliding window: q attends to k where (q >= k) AND (k >= q - window_size + 1)
            k_indices = start_kv + kv_offsets[None, :]

            if STAGE == 2:  # Causal diagonal - mask where q < k
                # TRUE for the valid tokens (q >= k)
                mask = q_offsets[:, None] >= k_indices

            elif STAGE == 3:  # Left window boundary - mask where k < q - window_size + 1
                # TRUE for the valid tokens (k >= q - window_size + 1)
                mask = k_indices >= (q_offsets[:, None] - WINDOW_SIZE + 1)

            # Apply mask by adding -1.0e6 to False positions
            S_block = S_block * softmax_factor + tl.where(mask, 0, -1.0e6)
            # max per row for stability (log-sum-exp trick)
            m_ij = tl.maximum(m_i, tl.max(S_block, 1))
            # subtracting the row-wise max for numerical stability
            S_block -= m_ij[:, None]

        else:  # Global - Stage 1 | Causal - Stage 1 | Sliding Window - Stage 1 | Causal Sliding Window - Stage 1
            # compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(S_block, 1) * softmax_factor)
            S_block = S_block * softmax_factor - m_ij[:, None]

        # exp(qk_ij - m_i)
        P_block = tl.math.exp(S_block)

        # Correction factor
        cor_fact = tl.math.exp(m_i - m_ij)

        # Apply correction factor to the previous l_i and add the new l_ij
        l_i = cor_fact * l_i + tl.sum(P_block, 1)

        # load the V block on SRAM
        V_block = tl.load(V_block_ptr)

        # Convert P_block to match V_block's dtype for the dot product
        P_block = P_block.to(V_block.dtype)

        # Compute the new O_block
        O_block = O_block * cor_fact[:, None]
        # O_block is an accumulator. It is more efficient
        O_block = tl.dot(P_block, V_block, O_block)

        # update the maximum tensor
        m_i = m_ij

        # Move the pointers to the next K and V blocks
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_T_block_ptr = tl.advance(K_T_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i
