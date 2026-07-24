"""Helion kernels for SGLang's Kimi Delta Attention prefill path.

The public :func:`chunk_kda` entry point in this module is intended to match
``sglang.kernels.ops.attention.fla.kda.chunk_kda``.  KDA uses 64-token chunks
and keeps the cumulative per-key decay in base-2 logarithm space.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

CHUNK_SIZE = 64
# Rounded identically to flash-linear-attention/SGLang before FP32 multiply.
RCP_LN2 = 1.4426950216293335
L2_NORM_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0


_CHUNK_INDICES_CACHE: list[tuple[torch.Tensor, int, torch.Tensor]] = []
_CHUNK_OFFSETS_CACHE: list[tuple[torch.Tensor, int, torch.Tensor]] = []


_L2_NORM_CONFIG = helion.Config(
    block_sizes=[8],
    num_warps=4,
    num_stages=2,
    indexing="pointer",
)


@helion.kernel(static_shapes=False, config=_L2_NORM_CONFIG)
def _l2norm_qk(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q and K rows with Triton's FP32 accumulation contract."""
    B = q.size(0)
    T = q.size(1)
    H = hl.specialize(q.size(2))
    K = hl.specialize(q.size(3))
    hl.specialize(
        (
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(1),
            k.stride(2),
            k.stride(3),
        )
    )

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_rows = q.view(B * T * H, K)
    k_rows = k.view(B * T * H, K)
    q_out_rows = q_out.view(B * T * H, K)
    k_out_rows = k_out.view(B * T * H, K)
    block_rows = hl.register_block_size(1, 16)

    for tile_rows in hl.tile(B * T * H, block_size=block_rows):
        q_value = q_rows[tile_rows, :].float()
        k_value = k_rows[tile_rows, :].float()
        q_norm = torch.sqrt((q_value * q_value).sum(-1) + L2_NORM_EPS)
        k_norm = torch.sqrt((k_value * k_value).sum(-1) + L2_NORM_EPS)
        q_out_rows[tile_rows, :] = (q_value / q_norm[:, None]).to(q.dtype)
        k_out_rows[tile_rows, :] = (k_value / k_norm[:, None]).to(k.dtype)

    return q_out, k_out


_GATE_FIXED_CONFIG = helion.Config(
    block_sizes=[16],
    loop_orders=[[2, 3, 1, 0]],
    num_warps=1,
    num_stages=1,
    indexing="pointer",
)


_GATE_VARLEN_CONFIG = helion.Config(
    block_sizes=[16],
    loop_orders=[[2, 1, 0]],
    num_warps=1,
    num_stages=1,
    indexing="pointer",
)


def _activate_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    lower_bound: float,
    use_lower_bound: hl.constexpr,
) -> torch.Tensor:
    a = torch.exp(a_log.float())
    if use_lower_bound:
        return lower_bound * torch.sigmoid(a * raw_gate)
    softplus = torch.where(
        raw_gate < SOFTPLUS_THRESHOLD,
        torch.log(1.0 + torch.exp(raw_gate)),
        raw_gate,
    )
    return -a * softplus


@helion.kernel(static_shapes=False, config=_GATE_FIXED_CONFIG)
def _gate_cumsum_fixed(
    g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    activate: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    has_bias: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    use_lower_bound: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Gate activation and chunk-local cumsum for equal-length batches."""
    B = g.size(0)
    T = g.size(1)
    H = hl.specialize(g.size(2))
    K = hl.specialize(g.size(3))
    hl.specialize(
        (
            g.stride(1),
            g.stride(2),
            g.stride(3),
            a_log.stride(0),
            dt_bias.stride(0),
        )
    )

    out = torch.empty_like(g, dtype=torch.float32)
    g_rows = g.view(B * T * H, K)
    out_rows = out.view(B * T * H, K)
    chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    block_k = hl.register_block_size(16, K)

    for tile_b, tile_chunk, tile_h, tile_k in hl.tile(
        [B, chunks, H, K],
        block_size=[1, 1, 1, block_k],
    ):
        time = hl.arange(64)
        token = tile_chunk.id * CHUNK_SIZE + time
        valid = token < T
        row = (tile_b.id * T + token) * H + tile_h.id
        value = hl.load(
            g_rows,
            [row[:, None], tile_k.index[None, :]],
            extra_mask=valid[:, None],
        ).float()
        if activate:
            if has_bias:
                value = value + dt_bias[tile_h.id * K + tile_k.index].float()[None, :]
            value = _activate_gate(
                value,
                a_log[tile_h.id],
                lower_bound,
                use_lower_bound,
            )
        value = torch.cumsum(value, dim=0) * scale
        hl.store(
            out_rows,
            [row[:, None], tile_k.index[None, :]],
            value,
            extra_mask=valid[:, None],
        )

    return out


@helion.kernel(static_shapes=False, config=_GATE_VARLEN_CONFIG)
def _gate_cumsum_varlen(
    g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    scale: float,
    lower_bound: float,
    activate: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    has_bias: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    use_lower_bound: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Gate activation and chunk-local cumsum for packed ragged sequences."""
    T = g.size(1)
    H = hl.specialize(g.size(2))
    K = hl.specialize(g.size(3))
    chunks = chunk_indices.size(0)
    hl.specialize(
        (
            g.stride(1),
            g.stride(2),
            g.stride(3),
            a_log.stride(0),
            dt_bias.stride(0),
            cu_seqlens.stride(0),
            chunk_indices.stride(0),
            chunk_indices.stride(1),
        )
    )

    out = torch.empty_like(g, dtype=torch.float32)
    g_rows = g.view(T * H, K)
    out_rows = out.view(T * H, K)
    block_k = hl.register_block_size(16, K)

    for tile_chunk, tile_h, tile_k in hl.tile(
        [chunks, H, K],
        block_size=[1, 1, block_k],
    ):
        time = hl.arange(64)
        sequence = chunk_indices[tile_chunk.id, 0].long()
        local_chunk = chunk_indices[tile_chunk.id, 1].long()
        begin = cu_seqlens[sequence].long()
        end = cu_seqlens[sequence + 1].long()
        token = begin + local_chunk * CHUNK_SIZE + time
        valid = token < end
        row = token * H + tile_h.id
        value = hl.load(
            g_rows,
            [row[:, None], tile_k.index[None, :]],
            extra_mask=valid[:, None],
        ).float()
        if activate:
            if has_bias:
                value = value + dt_bias[tile_h.id * K + tile_k.index].float()[None, :]
            value = _activate_gate(
                value,
                a_log[tile_h.id],
                lower_bound,
                use_lower_bound,
            )
        value = torch.cumsum(value, dim=0) * scale
        hl.store(
            out_rows,
            [row[:, None], tile_k.index[None, :]],
            value,
            extra_mask=valid[:, None],
        )

    return out


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Build the same ``(sequence, local_chunk)`` map used by SGLang."""
    for index, (cached_cu_seqlens, cached_chunk_size, result) in enumerate(
        _CHUNK_INDICES_CACHE
    ):
        if cu_seqlens is cached_cu_seqlens and chunk_size == cached_chunk_size:
            _CHUNK_INDICES_CACHE.append(_CHUNK_INDICES_CACHE.pop(index))
            return result

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_counts = torch.div(
        lengths + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    )
    local_chunks = torch.cat(
        [
            torch.arange(count, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
            for count in chunk_counts.tolist()
        ]
    )
    result = torch.stack(
        [local_chunks.eq(0).cumsum(0) - 1, local_chunks],
        dim=1,
    )
    _CHUNK_INDICES_CACHE.append((cu_seqlens, chunk_size, result))
    if len(_CHUNK_INDICES_CACHE) > 4:
        _CHUNK_INDICES_CACHE.pop(0)
    return result


def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Return the packed output offset for each ragged sequence."""
    for index, (cached_cu_seqlens, cached_chunk_size, result) in enumerate(
        _CHUNK_OFFSETS_CACHE
    ):
        if cu_seqlens is cached_cu_seqlens and chunk_size == cached_chunk_size:
            _CHUNK_OFFSETS_CACHE.append(_CHUNK_OFFSETS_CACHE.pop(index))
            return result

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_counts = torch.div(
        lengths + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    )
    result = torch.cat([cu_seqlens.new_zeros(1), chunk_counts]).cumsum(0)
    _CHUNK_OFFSETS_CACHE.append((cu_seqlens, chunk_size, result))
    if len(_CHUNK_OFFSETS_CACHE) > 4:
        _CHUNK_OFFSETS_CACHE.pop(0)
    return result


def gate_chunk_cumsum(
    g: torch.Tensor,
    *,
    a_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
    lower_bound: float | None = None,
    scale: float = RCP_LN2,
) -> torch.Tensor:
    """Apply KDA gate preprocessing with SGLang-compatible option semantics."""
    flat_a_log = (
        a_log.reshape(-1)
        if a_log is not None
        else torch.empty(1, device=g.device, dtype=torch.float32)
    )
    flat_bias = (
        dt_bias.reshape(-1)
        if dt_bias is not None
        else torch.empty(1, device=g.device, dtype=torch.float32)
    )
    activate = a_log is not None
    has_bias = dt_bias is not None
    use_lower_bound = lower_bound is not None
    lower_bound_value = 0.0 if lower_bound is None else lower_bound

    if cu_seqlens is None:
        return _gate_cumsum_fixed(
            g,
            flat_a_log,
            flat_bias,
            scale,
            lower_bound_value,
            activate,
            has_bias,
            use_lower_bound,
        )

    if g.size(0) != 1:
        raise ValueError("varlen KDA requires batch size 1")
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens)
    return _gate_cumsum_varlen(
        g,
        flat_a_log,
        flat_bias,
        cu_seqlens,
        chunk_indices,
        scale,
        lower_bound_value,
        activate,
        has_bias,
        use_lower_bound,
    )


_INTRA_MATRIX_CONFIG = helion.Config(
    block_sizes=[32],
    loop_orders=[[2, 1, 0]],
    num_warps=1,
    num_stages=2,
    indexing="pointer",
)


@helion.kernel(static_shapes=False, config=_INTRA_MATRIX_CONFIG)
def _intra_matrices_wide(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    scale: float,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a full 16x64 causal matrix row per CTA."""
    B = q.size(0)
    T = q.size(1)
    H = hl.specialize(q.size(2))
    K = hl.specialize(q.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    hl.specialize(
        (
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            beta.stride(1),
            beta.stride(2),
        )
    )

    aqk = torch.empty([B, T, H, CHUNK_SIZE], dtype=q.dtype, device=q.device)
    akk = torch.empty([B, T, H, CHUNK_SIZE], dtype=torch.float32, device=q.device)
    q_rows = q.view(B * T * H, K)
    k_rows = k.view(B * T * H, K)
    g_rows = g.view(B * T * H, K)
    beta_rows = beta.view(B * T * H)
    aqk_rows = aqk.view(B * T * H, CHUNK_SIZE)
    akk_rows = akk.view(B * T * H, CHUNK_SIZE)
    block_k = hl.register_block_size(32, K)

    for tile_chunk, tile_h, tile_row_block in hl.tile(
        [total_chunks, H, 4],
        block_size=[1, 1, 1],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T

        row_lane = hl.arange(16)
        col_lane = hl.arange(64)
        chunk_begin = begin + local_chunk * CHUNK_SIZE
        row_local = tile_row_block.id * 16 + row_lane
        row_token = chunk_begin + row_local
        col_token = chunk_begin + col_lane
        row_valid = row_token < end
        col_valid = col_token < end
        block_causal = col_lane < (tile_row_block.id + 1) * 16
        row = row_token * H + tile_h.id
        col = col_token * H + tile_h.id
        anchor_token = chunk_begin + tile_row_block.id * 16
        anchor = anchor_token * H + tile_h.id
        aqk_off = hl.zeros([16, 64], dtype=torch.float32)
        akk_off = hl.zeros([16, 64], dtype=torch.float32)
        aqk_diag = hl.zeros([16, 16], dtype=torch.float32)
        akk_diag = hl.zeros([16, 16], dtype=torch.float32)

        for tile_k in hl.tile(K, block_size=block_k):
            q_row = hl.load(
                q_rows,
                [row[:, None], tile_k.index[None, :]],
                extra_mask=row_valid[:, None],
            ).float()
            k_row = hl.load(
                k_rows,
                [row[:, None], tile_k.index[None, :]],
                extra_mask=row_valid[:, None],
            ).float()
            g_row = hl.load(
                g_rows,
                [row[:, None], tile_k.index[None, :]],
                extra_mask=row_valid[:, None],
            ).float()
            k_col = hl.load(
                k_rows,
                [col[:, None], tile_k.index[None, :]],
                extra_mask=col_valid[:, None] & block_causal[:, None],
            ).float()
            g_col = hl.load(
                g_rows,
                [col[:, None], tile_k.index[None, :]],
                extra_mask=col_valid[:, None] & block_causal[:, None],
            ).float()
            g_anchor = hl.load(
                g_rows,
                [anchor, tile_k.index],
                extra_mask=anchor_token < end,
            ).float()
            off_col = col_lane < tile_row_block.id * 16
            off_col_delta = torch.where(
                off_col[:, None],
                g_anchor[None, :] - g_col,
                0.0,
            )
            q_off = (q_row * torch.exp2(g_row - g_anchor[None, :])).to(torch.bfloat16)
            k_off = (k_row * torch.exp2(g_row - g_anchor[None, :])).to(torch.bfloat16)
            k_col_off = (k_col * torch.exp2(off_col_delta)).to(torch.bfloat16)
            aqk_off = hl.dot(
                q_off,
                k_col_off.T,
                acc=aqk_off,
                out_dtype=torch.float32,
            )
            akk_off = hl.dot(
                k_off,
                k_col_off.T,
                acc=akk_off,
                out_dtype=torch.float32,
            )

            diag_delta = torch.clamp(
                g_row - g_anchor[None, :],
                -126.0,
                126.0,
            )
            q_diag = q_row * torch.exp2(diag_delta)
            k_diag_fwd = k_row * torch.exp2(diag_delta)
            k_diag_bwd = k_row * torch.exp2(-diag_delta)
            aqk_diag = hl.dot(
                q_diag,
                k_diag_bwd.T,
                acc=aqk_diag,
                out_dtype=torch.float32,
            )
            akk_diag = hl.dot(
                k_diag_fwd,
                k_diag_bwd.T,
                acc=akk_diag,
                out_dtype=torch.float32,
            )

        causal = row_local[:, None] >= col_lane[None, :]
        strictly_causal = row_local[:, None] > col_lane[None, :]
        row_beta = hl.load(
            beta_rows,
            [row],
            extra_mask=row_valid,
        ).float()
        hl.store(
            aqk_rows,
            [row[:, None], col_lane[None, :]],
            torch.where(causal & col_valid[None, :], aqk_off * scale, 0.0),
            extra_mask=row_valid[:, None],
        )
        hl.store(
            akk_rows,
            [row[:, None], col_lane[None, :]],
            torch.where(
                strictly_causal & col_valid[None, :],
                akk_off * row_beta[:, None],
                0.0,
            ),
            extra_mask=row_valid[:, None],
        )
        diag_col = tile_row_block.id * 16 + row_lane
        diag_causal = row_lane[:, None] >= row_lane[None, :]
        diag_strict = row_lane[:, None] > row_lane[None, :]
        hl.store(
            aqk_rows,
            [row[:, None], diag_col[None, :]],
            torch.where(diag_causal & row_valid[None, :], aqk_diag * scale, 0.0),
            extra_mask=row_valid[:, None],
        )
        hl.store(
            akk_rows,
            [row[:, None], diag_col[None, :]],
            torch.where(
                diag_strict & row_valid[None, :],
                akk_diag * row_beta[:, None],
                0.0,
            ),
            extra_mask=row_valid[:, None],
        )

    return aqk, akk


@helion.kernel(static_shapes=False, config=_INTRA_MATRIX_CONFIG)
def _intra_matrices(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    scale: float,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute causal QK and beta-scaled KK blocks for each KDA chunk."""
    B = q.size(0)
    T = q.size(1)
    H = hl.specialize(q.size(2))
    K = hl.specialize(q.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    hl.specialize(
        (
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            beta.stride(1),
            beta.stride(2),
        )
    )

    aqk = torch.zeros([B, T, H, CHUNK_SIZE], dtype=q.dtype, device=q.device)
    akk = torch.zeros([B, T, H, CHUNK_SIZE], dtype=torch.float32, device=q.device)
    q_rows = q.view(B * T * H, K)
    k_rows = k.view(B * T * H, K)
    g_rows = g.view(B * T * H, K)
    beta_rows = beta.view(B * T * H)
    aqk_rows = aqk.view(B * T * H, CHUNK_SIZE)
    akk_rows = akk.view(B * T * H, CHUNK_SIZE)
    block_k = hl.register_block_size(32, K)

    for tile_chunk, tile_h, tile_row_block, tile_col_block in hl.tile(
        [total_chunks, H, 4, 4],
        block_size=[1, 1, 1, 1],
    ):
        if tile_col_block.id <= tile_row_block.id:
            if is_varlen:
                sequence = chunk_indices[tile_chunk.id, 0].long()
                local_chunk = chunk_indices[tile_chunk.id, 1].long()
                begin = cu_seqlens[sequence].long()
                end = cu_seqlens[sequence + 1].long()
            else:
                sequence = tile_chunk.id // chunks_per_batch
                local_chunk = tile_chunk.id % chunks_per_batch
                begin = sequence * T
                end = begin + T

            lane = hl.arange(16)
            chunk_begin = begin + local_chunk * CHUNK_SIZE
            row_token = chunk_begin + tile_row_block.id * 16 + lane
            col_token = chunk_begin + tile_col_block.id * 16 + lane
            row_valid = row_token < end
            col_valid = col_token < end
            row = row_token * H + tile_h.id
            col = col_token * H + tile_h.id
            anchor_row = chunk_begin + tile_row_block.id * 16
            anchor = anchor_row * H + tile_h.id
            aqk_value = hl.zeros([16, 16], dtype=torch.float32)
            akk_value = hl.zeros([16, 16], dtype=torch.float32)

            for tile_k in hl.tile(K, block_size=block_k):
                q_row = hl.load(
                    q_rows,
                    [row[:, None], tile_k.index[None, :]],
                    extra_mask=row_valid[:, None],
                ).float()
                k_row = hl.load(
                    k_rows,
                    [row[:, None], tile_k.index[None, :]],
                    extra_mask=row_valid[:, None],
                ).float()
                g_row = hl.load(
                    g_rows,
                    [row[:, None], tile_k.index[None, :]],
                    extra_mask=row_valid[:, None],
                ).float()
                k_col = hl.load(
                    k_rows,
                    [col[:, None], tile_k.index[None, :]],
                    extra_mask=col_valid[:, None],
                ).float()
                g_col = hl.load(
                    g_rows,
                    [col[:, None], tile_k.index[None, :]],
                    extra_mask=col_valid[:, None],
                ).float()
                g_anchor = hl.load(
                    g_rows,
                    [anchor, tile_k.index],
                    extra_mask=anchor_row < end,
                ).float()

                if tile_row_block.id == tile_col_block.id:
                    delta_row = torch.clamp(
                        g_row - g_anchor[None, :],
                        -126.0,
                        126.0,
                    )
                    delta_col = torch.clamp(
                        g_anchor[None, :] - g_col,
                        -126.0,
                        126.0,
                    )
                    q_scaled = (q_row * torch.exp2(delta_row)).to(torch.bfloat16)
                    k_scaled = (k_row * torch.exp2(delta_row)).to(torch.bfloat16)
                    k_col_scaled = (k_col * torch.exp2(delta_col)).to(torch.bfloat16)
                else:
                    q_scaled = (q_row * torch.exp2(g_row - g_anchor[None, :])).to(
                        torch.bfloat16
                    )
                    k_scaled = (k_row * torch.exp2(g_row - g_anchor[None, :])).to(
                        torch.bfloat16
                    )
                    k_col_scaled = (k_col * torch.exp2(g_anchor[None, :] - g_col)).to(
                        torch.bfloat16
                    )

                aqk_value = hl.dot(
                    q_scaled,
                    k_col_scaled.T,
                    acc=aqk_value,
                    out_dtype=torch.float32,
                )
                akk_value = hl.dot(
                    k_scaled,
                    k_col_scaled.T,
                    acc=akk_value,
                    out_dtype=torch.float32,
                )

            row_local = tile_row_block.id * 16 + lane
            col_local = tile_col_block.id * 16 + lane
            aqk_mask = row_valid[:, None] & col_valid[None, :]
            akk_mask = row_valid[:, None] & col_valid[None, :]
            aqk_causal = row_local[:, None] >= col_local[None, :]
            akk_causal = row_local[:, None] > col_local[None, :]
            row_beta = hl.load(
                beta_rows,
                [row],
                extra_mask=row_valid,
            ).float()
            hl.store(
                aqk_rows,
                [row[:, None], col_local[None, :]],
                torch.where(aqk_causal, aqk_value * scale, 0.0),
                extra_mask=aqk_mask,
            )
            hl.store(
                akk_rows,
                [row[:, None], col_local[None, :]],
                torch.where(akk_causal, akk_value * row_beta[:, None], 0.0),
                extra_mask=akk_mask,
            )

    return aqk, akk


def _invert_lower_16(matrix: torch.Tensor) -> torch.Tensor:
    lane = hl.arange(16)
    strictly_lower = lane[:, None] > lane[None, :]
    inverse = -torch.where(strictly_lower, matrix, 0.0)
    for row in range(2, 16):
        value = -torch.where((lane == row)[:, None], matrix, 0.0).sum(0)
        value = torch.where(lane < row, value, 0.0)
        value = value + (value[:, None] * inverse).sum(0)
        inverse = torch.where((lane == row)[:, None], value[None, :], inverse)
    return inverse + (lane[:, None] == lane[None, :]).float()


_INTRA_SOLVE_CONFIG = helion.Config(
    loop_orders=[[1, 0]],
    num_warps=2,
    num_stages=2,
    indexing="pointer",
)


@helion.kernel(static_shapes=False, config=_INTRA_SOLVE_CONFIG)
def _intra_solve(
    akk: torch.Tensor,
    output_template: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Invert the 64x64 unit-lower KDA system as four 16x16 blocks."""
    B = akk.size(0)
    T = akk.size(1)
    H = hl.specialize(akk.size(2))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    hl.specialize((akk.stride(1), akk.stride(2), akk.stride(3)))

    inverse = torch.empty(
        [B, T, H, CHUNK_SIZE],
        dtype=output_template.dtype,
        device=akk.device,
    )
    akk_rows = akk.view(B * T * H, CHUNK_SIZE)
    inverse_rows = inverse.view(B * T * H, CHUNK_SIZE)

    for tile_chunk, tile_h in hl.tile(
        [total_chunks, H],
        block_size=[1, 1],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T

        lane = hl.arange(16)
        chunk_begin = begin + local_chunk * CHUNK_SIZE
        row0 = chunk_begin + lane
        row1 = chunk_begin + 16 + lane
        row2 = chunk_begin + 32 + lane
        row3 = chunk_begin + 48 + lane
        valid0 = row0 < end
        valid1 = row1 < end
        valid2 = row2 < end
        valid3 = row3 < end
        flat0 = row0 * H + tile_h.id
        flat1 = row1 * H + tile_h.id
        flat2 = row2 * H + tile_h.id
        flat3 = row3 * H + tile_h.id
        col0 = lane
        col1 = 16 + lane
        col2 = 32 + lane
        col3 = 48 + lane

        m00 = hl.load(
            akk_rows,
            [flat0[:, None], col0[None, :]],
            extra_mask=valid0[:, None] & valid0[None, :],
        ).float()
        m10 = hl.load(
            akk_rows,
            [flat1[:, None], col0[None, :]],
            extra_mask=valid1[:, None] & valid0[None, :],
        ).float()
        m11 = hl.load(
            akk_rows,
            [flat1[:, None], col1[None, :]],
            extra_mask=valid1[:, None] & valid1[None, :],
        ).float()
        m20 = hl.load(
            akk_rows,
            [flat2[:, None], col0[None, :]],
            extra_mask=valid2[:, None] & valid0[None, :],
        ).float()
        m21 = hl.load(
            akk_rows,
            [flat2[:, None], col1[None, :]],
            extra_mask=valid2[:, None] & valid1[None, :],
        ).float()
        m22 = hl.load(
            akk_rows,
            [flat2[:, None], col2[None, :]],
            extra_mask=valid2[:, None] & valid2[None, :],
        ).float()
        m30 = hl.load(
            akk_rows,
            [flat3[:, None], col0[None, :]],
            extra_mask=valid3[:, None] & valid0[None, :],
        ).float()
        m31 = hl.load(
            akk_rows,
            [flat3[:, None], col1[None, :]],
            extra_mask=valid3[:, None] & valid1[None, :],
        ).float()
        m32 = hl.load(
            akk_rows,
            [flat3[:, None], col2[None, :]],
            extra_mask=valid3[:, None] & valid2[None, :],
        ).float()
        m33 = hl.load(
            akk_rows,
            [flat3[:, None], col3[None, :]],
            extra_mask=valid3[:, None] & valid3[None, :],
        ).float()

        i00 = _invert_lower_16(m00)
        i11 = _invert_lower_16(m11)
        i22 = _invert_lower_16(m22)
        i33 = _invert_lower_16(m33)
        i10 = -hl.dot(
            hl.dot(i11, m10, out_dtype=torch.float32),
            i00,
            out_dtype=torch.float32,
        )
        i21 = -hl.dot(
            hl.dot(i22, m21, out_dtype=torch.float32),
            i11,
            out_dtype=torch.float32,
        )
        i32 = -hl.dot(
            hl.dot(i33, m32, out_dtype=torch.float32),
            i22,
            out_dtype=torch.float32,
        )
        i20 = -hl.dot(
            i22,
            hl.dot(m20, i00, out_dtype=torch.float32)
            + hl.dot(m21, i10, out_dtype=torch.float32),
            out_dtype=torch.float32,
        )
        i31 = -hl.dot(
            i33,
            hl.dot(m31, i11, out_dtype=torch.float32)
            + hl.dot(m32, i21, out_dtype=torch.float32),
            out_dtype=torch.float32,
        )
        i30 = -hl.dot(
            i33,
            hl.dot(m30, i00, out_dtype=torch.float32)
            + hl.dot(m31, i10, out_dtype=torch.float32)
            + hl.dot(m32, i20, out_dtype=torch.float32),
            out_dtype=torch.float32,
        )

        hl.store(
            inverse_rows,
            [flat0[:, None], col0[None, :]],
            i00,
            extra_mask=valid0[:, None] & valid0[None, :],
        )
        hl.store(
            inverse_rows,
            [flat1[:, None], col0[None, :]],
            i10,
            extra_mask=valid1[:, None] & valid0[None, :],
        )
        hl.store(
            inverse_rows,
            [flat1[:, None], col1[None, :]],
            i11,
            extra_mask=valid1[:, None] & valid1[None, :],
        )
        hl.store(
            inverse_rows,
            [flat2[:, None], col0[None, :]],
            i20,
            extra_mask=valid2[:, None] & valid0[None, :],
        )
        hl.store(
            inverse_rows,
            [flat2[:, None], col1[None, :]],
            i21,
            extra_mask=valid2[:, None] & valid1[None, :],
        )
        hl.store(
            inverse_rows,
            [flat2[:, None], col2[None, :]],
            i22,
            extra_mask=valid2[:, None] & valid2[None, :],
        )
        hl.store(
            inverse_rows,
            [flat3[:, None], col0[None, :]],
            i30,
            extra_mask=valid3[:, None] & valid0[None, :],
        )
        hl.store(
            inverse_rows,
            [flat3[:, None], col1[None, :]],
            i31,
            extra_mask=valid3[:, None] & valid1[None, :],
        )
        hl.store(
            inverse_rows,
            [flat3[:, None], col2[None, :]],
            i32,
            extra_mask=valid3[:, None] & valid2[None, :],
        )
        hl.store(
            inverse_rows,
            [flat3[:, None], col3[None, :]],
            i33,
            extra_mask=valid3[:, None] & valid3[None, :],
        )

    return inverse


_RECOMPUTE_U_CONFIG = helion.Config(
    block_sizes=[128],
    loop_orders=[[0, 1, 2, 3]],
    static_ranges=[True],
    num_warps=2,
    num_stages=2,
    indexing="pointer",
)


@helion.kernel(static_shapes=False, config=_RECOMPUTE_U_CONFIG)
def _recompute_u(
    v: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    B = v.size(0)
    T = v.size(1)
    H = hl.specialize(v.size(2))
    V = hl.specialize(v.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    u = torch.empty_like(v)
    v_rows = v.view(B * T * H, V)
    beta_rows = beta.view(B * T * H)
    inverse_rows = inverse.view(B * T * H, CHUNK_SIZE)
    u_rows = u.view(B * T * H, V)
    block_v = hl.register_block_size(32, V)

    for tile_chunk, tile_h, tile_row_block, tile_v in hl.tile(
        [total_chunks, H, 4, V],
        block_size=[1, 1, 1, block_v],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T

        lane = hl.arange(16)
        chunk_begin = begin + local_chunk * CHUNK_SIZE
        row_token = chunk_begin + tile_row_block.id * 16 + lane
        row_valid = row_token < end
        row = row_token * H + tile_h.id
        value = hl.zeros([16, tile_v], dtype=torch.float32)
        for source_block in range(4):
            if source_block <= tile_row_block.id:
                source_token = chunk_begin + source_block * 16 + lane
                source_valid = source_token < end
                source = source_token * H + tile_h.id
                inv = hl.load(
                    inverse_rows,
                    [row[:, None], (source_block * 16 + lane)[None, :]],
                    extra_mask=row_valid[:, None] & source_valid[None, :],
                )
                source_v = hl.load(
                    v_rows,
                    [source[:, None], tile_v.index[None, :]],
                    extra_mask=source_valid[:, None],
                )
                source_beta = hl.load(
                    beta_rows,
                    [source],
                    extra_mask=source_valid,
                )
                value = hl.dot(
                    inv,
                    (source_v * source_beta[:, None]).to(v.dtype),
                    acc=value,
                    out_dtype=torch.float32,
                )
        hl.store(
            u_rows,
            [row[:, None], tile_v.index[None, :]],
            value,
            extra_mask=row_valid[:, None],
        )
    return u


_RECOMPUTE_W_CONFIG = helion.Config(
    block_sizes=[128],
    loop_orders=[[1, 2, 0, 3]],
    l2_groupings=[4],
    static_ranges=[True],
    num_warps=4,
    num_stages=2,
    indexing="pointer",
)


@helion.kernel(static_shapes=False, config=_RECOMPUTE_W_CONFIG)
def _recompute_w_kg(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    B = k.size(0)
    T = k.size(1)
    H = hl.specialize(k.size(2))
    K = hl.specialize(k.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    w = torch.empty_like(k)
    kg = torch.empty_like(k)
    k_rows = k.view(B * T * H, K)
    g_rows = g.view(B * T * H, K)
    beta_rows = beta.view(B * T * H)
    inverse_rows = inverse.view(B * T * H, CHUNK_SIZE)
    w_rows = w.view(B * T * H, K)
    kg_rows = kg.view(B * T * H, K)
    block_k = hl.register_block_size(32, K)

    for tile_chunk, tile_h, tile_row_block, tile_k in hl.tile(
        [total_chunks, H, 4, K],
        block_size=[1, 1, 1, block_k],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T

        lane = hl.arange(16)
        chunk_begin = begin + local_chunk * CHUNK_SIZE
        row_token = chunk_begin + tile_row_block.id * 16 + lane
        row_valid = row_token < end
        row = row_token * H + tile_h.id
        value = hl.zeros([16, tile_k], dtype=torch.float32)
        for source_block in range(4):
            if source_block <= tile_row_block.id:
                source_token = chunk_begin + source_block * 16 + lane
                source_valid = source_token < end
                source = source_token * H + tile_h.id
                inv = hl.load(
                    inverse_rows,
                    [row[:, None], (source_block * 16 + lane)[None, :]],
                    extra_mask=row_valid[:, None] & source_valid[None, :],
                )
                source_k = hl.load(
                    k_rows,
                    [source[:, None], tile_k.index[None, :]],
                    extra_mask=source_valid[:, None],
                )
                source_g = hl.load(
                    g_rows,
                    [source[:, None], tile_k.index[None, :]],
                    extra_mask=source_valid[:, None],
                ).float()
                source_beta = hl.load(
                    beta_rows,
                    [source],
                    extra_mask=source_valid,
                ).float()
                weighted_k = (
                    source_k.float() * source_beta[:, None] * torch.exp2(source_g)
                ).to(k.dtype)
                value = hl.dot(
                    inv,
                    weighted_k,
                    acc=value,
                    out_dtype=torch.float32,
                )

        row_k = hl.load(
            k_rows,
            [row[:, None], tile_k.index[None, :]],
            extra_mask=row_valid[:, None],
        ).float()
        row_g = hl.load(
            g_rows,
            [row[:, None], tile_k.index[None, :]],
            extra_mask=row_valid[:, None],
        ).float()
        if is_varlen:
            chunk_end = end.new_full([], CHUNK_SIZE) + chunk_begin
            last_token = torch.minimum(chunk_end, end) - 1
        else:
            last_token = min(chunk_begin + CHUNK_SIZE, end) - 1
        last = last_token * H + tile_h.id
        last_g = g_rows[last, tile_k.index].float()
        kg_value = row_k * torch.exp2(last_g[None, :] - row_g)
        hl.store(
            w_rows,
            [row[:, None], tile_k.index[None, :]],
            value,
            extra_mask=row_valid[:, None],
        )
        hl.store(
            kg_rows,
            [row[:, None], tile_k.index[None, :]],
            kg_value,
            extra_mask=row_valid[:, None],
        )
    return w, kg


_FUSED_SOLVE_RECOMPUTE_CONFIG = helion.Config(
    block_sizes=[64, 32],
    loop_orders=[[0, 1]],
    num_warps=1,
    num_stages=2,
    indexing="pointer",
)


@helion.kernel(static_shapes=False, config=_FUSED_SOLVE_RECOMPUTE_CONFIG)
def _intra_solve_recompute(
    akk: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the 64x64 system and emit W, U, and KG from registers."""
    B = k.size(0)
    T = k.size(1)
    H = hl.specialize(k.size(2))
    K = hl.specialize(k.size(3))
    V = hl.specialize(v.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    akk_rows = akk.view(B * T * H, CHUNK_SIZE)
    k_rows = k.view(B * T * H, K)
    v_rows = v.view(B * T * H, V)
    g_rows = g.view(B * T * H, K)
    beta_rows = beta.view(B * T * H)
    w_rows = w.view(B * T * H, K)
    u_rows = u.view(B * T * H, V)
    kg_rows = kg.view(B * T * H, K)
    block_v = hl.register_block_size(32, V)
    block_k = hl.register_block_size(32, K)

    for tile_chunk, tile_h in hl.tile(
        [total_chunks, H],
        block_size=[1, 1],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T

        lane = hl.arange(16)
        chunk_begin = begin + local_chunk * CHUNK_SIZE
        row0 = chunk_begin + lane
        row1 = chunk_begin + 16 + lane
        row2 = chunk_begin + 32 + lane
        row3 = chunk_begin + 48 + lane
        valid0 = row0 < end
        valid1 = row1 < end
        valid2 = row2 < end
        valid3 = row3 < end
        flat0 = row0 * H + tile_h.id
        flat1 = row1 * H + tile_h.id
        flat2 = row2 * H + tile_h.id
        flat3 = row3 * H + tile_h.id
        col0 = lane
        col1 = 16 + lane
        col2 = 32 + lane
        col3 = 48 + lane

        m00 = hl.load(
            akk_rows,
            [flat0[:, None], col0[None, :]],
            extra_mask=valid0[:, None] & valid0[None, :],
        ).float()
        m10 = hl.load(
            akk_rows,
            [flat1[:, None], col0[None, :]],
            extra_mask=valid1[:, None] & valid0[None, :],
        ).float()
        m11 = hl.load(
            akk_rows,
            [flat1[:, None], col1[None, :]],
            extra_mask=valid1[:, None] & valid1[None, :],
        ).float()
        m20 = hl.load(
            akk_rows,
            [flat2[:, None], col0[None, :]],
            extra_mask=valid2[:, None] & valid0[None, :],
        ).float()
        m21 = hl.load(
            akk_rows,
            [flat2[:, None], col1[None, :]],
            extra_mask=valid2[:, None] & valid1[None, :],
        ).float()
        m22 = hl.load(
            akk_rows,
            [flat2[:, None], col2[None, :]],
            extra_mask=valid2[:, None] & valid2[None, :],
        ).float()
        m30 = hl.load(
            akk_rows,
            [flat3[:, None], col0[None, :]],
            extra_mask=valid3[:, None] & valid0[None, :],
        ).float()
        m31 = hl.load(
            akk_rows,
            [flat3[:, None], col1[None, :]],
            extra_mask=valid3[:, None] & valid1[None, :],
        ).float()
        m32 = hl.load(
            akk_rows,
            [flat3[:, None], col2[None, :]],
            extra_mask=valid3[:, None] & valid2[None, :],
        ).float()
        m33 = hl.load(
            akk_rows,
            [flat3[:, None], col3[None, :]],
            extra_mask=valid3[:, None] & valid3[None, :],
        ).float()

        i00 = _invert_lower_16(m00)
        i11 = _invert_lower_16(m11)
        i22 = _invert_lower_16(m22)
        i33 = _invert_lower_16(m33)
        i10 = -hl.dot(
            hl.dot(i11, m10, out_dtype=torch.float32),
            i00,
            out_dtype=torch.float32,
        )
        i21 = -hl.dot(
            hl.dot(i22, m21, out_dtype=torch.float32),
            i11,
            out_dtype=torch.float32,
        )
        i32 = -hl.dot(
            hl.dot(i33, m32, out_dtype=torch.float32),
            i22,
            out_dtype=torch.float32,
        )
        i20 = -hl.dot(
            i22,
            hl.dot(m20, i00, out_dtype=torch.float32)
            + hl.dot(m21, i10, out_dtype=torch.float32),
            out_dtype=torch.float32,
        )
        i31 = -hl.dot(
            i33,
            hl.dot(m31, i11, out_dtype=torch.float32)
            + hl.dot(m32, i21, out_dtype=torch.float32),
            out_dtype=torch.float32,
        )
        i30 = -hl.dot(
            i33,
            hl.dot(m30, i00, out_dtype=torch.float32)
            + hl.dot(m31, i10, out_dtype=torch.float32)
            + hl.dot(m32, i20, out_dtype=torch.float32),
            out_dtype=torch.float32,
        )
        i00 = i00.to(k.dtype)
        i10 = i10.to(k.dtype)
        i11 = i11.to(k.dtype)
        i20 = i20.to(k.dtype)
        i21 = i21.to(k.dtype)
        i22 = i22.to(k.dtype)
        i30 = i30.to(k.dtype)
        i31 = i31.to(k.dtype)
        i32 = i32.to(k.dtype)
        i33 = i33.to(k.dtype)

        beta0 = hl.load(beta_rows, [flat0], extra_mask=valid0).float()
        beta1 = hl.load(beta_rows, [flat1], extra_mask=valid1).float()
        beta2 = hl.load(beta_rows, [flat2], extra_mask=valid2).float()
        beta3 = hl.load(beta_rows, [flat3], extra_mask=valid3).float()
        for tile_v in hl.tile(V, block_size=block_v):
            v0 = hl.load(
                v_rows,
                [flat0[:, None], tile_v.index[None, :]],
                extra_mask=valid0[:, None],
            )
            v1 = hl.load(
                v_rows,
                [flat1[:, None], tile_v.index[None, :]],
                extra_mask=valid1[:, None],
            )
            v2 = hl.load(
                v_rows,
                [flat2[:, None], tile_v.index[None, :]],
                extra_mask=valid2[:, None],
            )
            v3 = hl.load(
                v_rows,
                [flat3[:, None], tile_v.index[None, :]],
                extra_mask=valid3[:, None],
            )
            vb0 = (v0 * beta0[:, None]).to(v.dtype)
            vb1 = (v1 * beta1[:, None]).to(v.dtype)
            vb2 = (v2 * beta2[:, None]).to(v.dtype)
            vb3 = (v3 * beta3[:, None]).to(v.dtype)
            u0 = hl.dot(i00, vb0, out_dtype=torch.float32)
            u1 = hl.dot(i10, vb0, out_dtype=torch.float32) + hl.dot(
                i11, vb1, out_dtype=torch.float32
            )
            u2 = (
                hl.dot(i20, vb0, out_dtype=torch.float32)
                + hl.dot(i21, vb1, out_dtype=torch.float32)
                + hl.dot(i22, vb2, out_dtype=torch.float32)
            )
            u3 = (
                hl.dot(i30, vb0, out_dtype=torch.float32)
                + hl.dot(i31, vb1, out_dtype=torch.float32)
                + hl.dot(i32, vb2, out_dtype=torch.float32)
                + hl.dot(i33, vb3, out_dtype=torch.float32)
            )
            hl.store(
                u_rows,
                [flat0[:, None], tile_v.index[None, :]],
                u0,
                extra_mask=valid0[:, None],
            )
            hl.store(
                u_rows,
                [flat1[:, None], tile_v.index[None, :]],
                u1,
                extra_mask=valid1[:, None],
            )
            hl.store(
                u_rows,
                [flat2[:, None], tile_v.index[None, :]],
                u2,
                extra_mask=valid2[:, None],
            )
            hl.store(
                u_rows,
                [flat3[:, None], tile_v.index[None, :]],
                u3,
                extra_mask=valid3[:, None],
            )

        if is_varlen:
            chunk_end = end.new_full([], CHUNK_SIZE) + chunk_begin
            last_token = torch.minimum(chunk_end, end) - 1
        else:
            last_token = min(chunk_begin + CHUNK_SIZE, end) - 1
        last = last_token * H + tile_h.id
        for tile_k in hl.tile(K, block_size=block_k):
            k0 = hl.load(
                k_rows,
                [flat0[:, None], tile_k.index[None, :]],
                extra_mask=valid0[:, None],
            )
            k1 = hl.load(
                k_rows,
                [flat1[:, None], tile_k.index[None, :]],
                extra_mask=valid1[:, None],
            )
            k2 = hl.load(
                k_rows,
                [flat2[:, None], tile_k.index[None, :]],
                extra_mask=valid2[:, None],
            )
            k3 = hl.load(
                k_rows,
                [flat3[:, None], tile_k.index[None, :]],
                extra_mask=valid3[:, None],
            )
            g0 = hl.load(
                g_rows,
                [flat0[:, None], tile_k.index[None, :]],
                extra_mask=valid0[:, None],
            ).float()
            g1 = hl.load(
                g_rows,
                [flat1[:, None], tile_k.index[None, :]],
                extra_mask=valid1[:, None],
            ).float()
            g2 = hl.load(
                g_rows,
                [flat2[:, None], tile_k.index[None, :]],
                extra_mask=valid2[:, None],
            ).float()
            g3 = hl.load(
                g_rows,
                [flat3[:, None], tile_k.index[None, :]],
                extra_mask=valid3[:, None],
            ).float()
            wk0 = (k0.float() * beta0[:, None] * torch.exp2(g0)).to(k.dtype)
            wk1 = (k1.float() * beta1[:, None] * torch.exp2(g1)).to(k.dtype)
            wk2 = (k2.float() * beta2[:, None] * torch.exp2(g2)).to(k.dtype)
            wk3 = (k3.float() * beta3[:, None] * torch.exp2(g3)).to(k.dtype)
            w0 = hl.dot(i00, wk0, out_dtype=torch.float32)
            w1 = hl.dot(i10, wk0, out_dtype=torch.float32) + hl.dot(
                i11, wk1, out_dtype=torch.float32
            )
            w2 = (
                hl.dot(i20, wk0, out_dtype=torch.float32)
                + hl.dot(i21, wk1, out_dtype=torch.float32)
                + hl.dot(i22, wk2, out_dtype=torch.float32)
            )
            w3 = (
                hl.dot(i30, wk0, out_dtype=torch.float32)
                + hl.dot(i31, wk1, out_dtype=torch.float32)
                + hl.dot(i32, wk2, out_dtype=torch.float32)
                + hl.dot(i33, wk3, out_dtype=torch.float32)
            )
            last_g = g_rows[last, tile_k.index].float()
            kg0 = k0.float() * torch.exp2(last_g[None, :] - g0)
            kg1 = k1.float() * torch.exp2(last_g[None, :] - g1)
            kg2 = k2.float() * torch.exp2(last_g[None, :] - g2)
            kg3 = k3.float() * torch.exp2(last_g[None, :] - g3)
            hl.store(
                w_rows,
                [flat0[:, None], tile_k.index[None, :]],
                w0,
                extra_mask=valid0[:, None],
            )
            hl.store(
                w_rows,
                [flat1[:, None], tile_k.index[None, :]],
                w1,
                extra_mask=valid1[:, None],
            )
            hl.store(
                w_rows,
                [flat2[:, None], tile_k.index[None, :]],
                w2,
                extra_mask=valid2[:, None],
            )
            hl.store(
                w_rows,
                [flat3[:, None], tile_k.index[None, :]],
                w3,
                extra_mask=valid3[:, None],
            )
            hl.store(
                kg_rows,
                [flat0[:, None], tile_k.index[None, :]],
                kg0,
                extra_mask=valid0[:, None],
            )
            hl.store(
                kg_rows,
                [flat1[:, None], tile_k.index[None, :]],
                kg1,
                extra_mask=valid1[:, None],
            )
            hl.store(
                kg_rows,
                [flat2[:, None], tile_k.index[None, :]],
                kg2,
                extra_mask=valid2[:, None],
            )
            hl.store(
                kg_rows,
                [flat3[:, None], tile_k.index[None, :]],
                kg3,
                extra_mask=valid3[:, None],
            )

    return w, u, kg


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helion equivalent of SGLang's intra-chunk KDA preparation."""
    is_varlen = cu_seqlens is not None
    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens)
        metadata = cu_seqlens
    else:
        metadata = torch.empty(0, device=q.device, dtype=torch.int32)
        chunk_indices = torch.empty(0, 2, device=q.device, dtype=torch.long)

    aqk, akk = _intra_matrices_wide(
        q,
        k,
        g,
        beta,
        metadata,
        chunk_indices,
        scale,
        is_varlen,
    )
    w, u, kg = _intra_solve_recompute(
        akk,
        k,
        v,
        g,
        beta,
        metadata,
        chunk_indices,
        is_varlen,
    )
    return w, u, kg, aqk


_STATE_CONFIG = helion.Config(
    block_sizes=[16],
    num_warps=4,
    num_stages=3,
    indexing="pointer",
)


@helion.kernel(static_shapes=False, config=_STATE_CONFIG)
def _chunk_state(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate KDA state between chunks and update the state pool in place."""
    B = kg.size(0)
    T = kg.size(1)
    H = hl.specialize(kg.size(2))
    K = hl.specialize(kg.size(3))
    V = hl.specialize(u.size(3))
    N = cu_seqlens.size(0) - 1 if is_varlen else B
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else chunks_per_batch
    hl.specialize(
        (
            kg.stride(1),
            kg.stride(2),
            kg.stride(3),
            w.stride(1),
            w.stride(2),
            w.stride(3),
            u.stride(1),
            u.stride(2),
            u.stride(3),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            initial_state.stride(0),
            initial_state.stride(1),
            initial_state.stride(2),
            initial_state.stride(3),
            initial_state_indices.stride(0),
        )
    )

    h = torch.empty(
        [B, total_chunks, H, V, K],
        dtype=kg.dtype,
        device=kg.device,
    )
    v_new = torch.empty_like(u)
    kg_rows = kg.view(B * T * H, K)
    w_rows = w.view(B * T * H, K)
    u_rows = u.view(B * T * H, V)
    g_rows = g.view(B * T * H, K)
    v_new_rows = v_new.view(B * T * H, V)
    h_rows = h.view(B * total_chunks * H, V, K)
    block_v = hl.register_block_size(1, V)

    for tile_sequence, tile_h, tile_v in hl.tile(
        [N, H, V],
        block_size=[1, 1, block_v],
    ):
        if is_varlen:
            begin = cu_seqlens[tile_sequence.id].long()
            end = cu_seqlens[tile_sequence.id + 1].long()
            output_offset = chunk_offsets[tile_sequence.id].long()
        else:
            begin = tile_sequence.id * T
            end = begin + T
            output_offset = tile_sequence.id * chunks_per_batch
        sequence_length = end - begin
        state_index = initial_state_indices[tile_sequence.id].long()
        state = initial_state[
            state_index,
            tile_h.id,
            tile_v.index,
            :,
        ].float()

        for token_tile in hl.tile(sequence_length, block_size=64):
            global_chunk = output_offset + token_tile.id
            h_rows[
                global_chunk * H + tile_h.id,
                tile_v,
                :,
            ] = state.to(h.dtype)
            token = begin + token_tile.index
            valid = token < end
            row = token * H + tile_h.id
            w_value = hl.load(
                w_rows,
                [row[:, None], hl.arange(K)[None, :]],
                extra_mask=valid[:, None],
            )
            residual = -hl.dot(
                w_value,
                state.T.to(w.dtype),
                out_dtype=torch.float32,
            )
            residual = residual + u_rows[row, tile_v].float()
            v_new_rows[row, tile_v] = residual.to(v_new.dtype)
            if is_varlen:
                chunk_end = (
                    end.new_full([], CHUNK_SIZE) + begin + token_tile.id * CHUNK_SIZE
                )
                last_token = torch.minimum(chunk_end, end) - 1
            else:
                last_token = (
                    min(
                        begin + (token_tile.id + 1) * CHUNK_SIZE,
                        end,
                    )
                    - 1
                )
            last_row = last_token * H + tile_h.id
            last_g = g_rows[last_row, :].float()
            state = state * torch.exp2(last_g)[None, :]
            kg_value = hl.load(
                kg_rows,
                [row[:, None], hl.arange(K)[None, :]],
                extra_mask=valid[:, None],
            )
            state = state + hl.dot(
                residual.T.to(kg.dtype),
                kg_value,
                out_dtype=torch.float32,
            )

        initial_state[
            state_index,
            tile_h.id,
            tile_v.index,
            :,
        ] = state.to(initial_state.dtype)

    return h, v_new


_OUTPUT_CONFIG = helion.Config(
    block_sizes=[64],
    loop_orders=[[2, 1, 0]],
    num_warps=8,
    num_stages=2,
    indexing="pointer",
)


@helion.kernel(static_shapes=False, config=_OUTPUT_CONFIG)
def _chunk_output(
    q: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    aqk: torch.Tensor,
    h: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    scale: float,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Compose inter-chunk state output and causal intra-chunk output."""
    B = q.size(0)
    T = q.size(1)
    H = hl.specialize(q.size(2))
    K = hl.specialize(q.size(3))
    V = hl.specialize(v_new.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    h_chunks = h.size(1)
    hl.specialize(
        (
            q.stride(1),
            q.stride(2),
            q.stride(3),
            v_new.stride(1),
            v_new.stride(2),
            v_new.stride(3),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            aqk.stride(1),
            aqk.stride(2),
            aqk.stride(3),
            h.stride(1),
            h.stride(2),
            h.stride(3),
            h.stride(4),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
    )

    q_rows = q.view(B * T * H, K)
    g_rows = g.view(B * T * H, K)
    v_rows = v_new.view(B * T * H, V)
    aqk_rows = aqk.view(B * T * H, CHUNK_SIZE)
    h_rows = h.view(B * h_chunks * H, V, K)
    out_rows = out.view(B * T * H, V)
    block_v = hl.register_block_size(32, V)

    for tile_chunk, tile_h, tile_v in hl.tile(
        [total_chunks, H, V],
        block_size=[1, 1, block_v],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
            h_chunk = tile_chunk.id
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T
            h_chunk = tile_chunk.id

        lane = hl.arange(64)
        token = begin + local_chunk * CHUNK_SIZE + lane
        valid = token < end
        row = token * H + tile_h.id
        q_value = hl.load(
            q_rows,
            [row[:, None], hl.arange(K)[None, :]],
            extra_mask=valid[:, None],
        ).float()
        g_value = hl.load(
            g_rows,
            [row[:, None], hl.arange(K)[None, :]],
            extra_mask=valid[:, None],
        ).float()
        h_value = h_rows[
            h_chunk * H + tile_h.id,
            tile_v,
            :,
        ]
        qg = (q_value * scale * torch.exp2(g_value)).to(q.dtype)
        output = hl.dot(
            qg,
            h_value.T,
            out_dtype=torch.float32,
        )
        a_value = hl.load(
            aqk_rows,
            [row[:, None], lane[None, :]],
            extra_mask=valid[:, None],
        )
        v_value = hl.load(
            v_rows,
            [row, tile_v],
            extra_mask=valid[:, None],
        )
        output = hl.dot(
            a_value.to(v_new.dtype),
            v_value,
            acc=output,
            out_dtype=torch.float32,
        )
        hl.store(
            out_rows,
            [row, tile_v],
            output,
            extra_mask=valid[:, None],
        )

    return out


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_intermediate_states: bool = False,
    **kwargs: object,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Match the public forward contract of SGLang's Triton ``chunk_kda``."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if initial_state is None or initial_state_indices is None:
        raise ValueError("KDA prefill requires an indexed initial-state pool")

    q = q.contiguous()
    k = k.contiguous()
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm_qk(q, k)
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens) if cu_seqlens is not None else None
    )
    g = gate_chunk_cumsum(
        g,
        a_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
    )
    w, u, kg, aqk = chunk_kda_fwd_intra(
        q,
        k,
        v,
        g,
        beta,
        scale,
        cu_seqlens,
        chunk_indices,
    )

    is_varlen = cu_seqlens is not None
    if is_varlen:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens)
        metadata = cu_seqlens
    else:
        metadata = torch.empty(0, device=q.device, dtype=torch.int32)
        chunk_offsets = torch.empty(0, device=q.device, dtype=torch.long)
    h, v_new = _chunk_state(
        kg,
        w,
        u,
        g,
        initial_state,
        initial_state_indices,
        metadata,
        chunk_indices
        if chunk_indices is not None
        else torch.empty(0, 2, device=q.device, dtype=torch.long),
        chunk_offsets,
        is_varlen,
    )
    if chunk_indices is None:
        chunk_indices = torch.empty(0, 2, device=q.device, dtype=torch.long)
    output = _chunk_output(
        q,
        v_new,
        g,
        aqk,
        h,
        v,
        metadata,
        chunk_indices,
        scale,
        is_varlen,
    )
    if output_intermediate_states:
        return output, h
    return output


def main() -> None:
    """Compile the front-end kernels for the production Kimi-Linear shape."""
    device = torch.device("cuda")
    q = torch.randn(1, 512, 16, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    g = torch.randn_like(q)
    a_log = torch.randn(16, device=device)
    dt_bias = torch.randn(16 * 128, device=device)
    _l2norm_qk(q, k)
    gate_chunk_cumsum(
        g,
        a_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=None,
    )


if __name__ == "__main__":
    main()
