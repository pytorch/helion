"""Causal scaled-dot-product attention tuned for TPU v7.

The kernel flattens all batch and head dimensions, divides the sequence into
2K query and K/V panels, and executes only the lower-triangular panel
worklist. Each work item updates online-softmax state in private VMEM scratch.

Sequences up to 2K use one panel and explicitly omit every 256x256 matrix
product above the causal diagonal. At 4K, each work item owns one K/V panel so
there is enough parallel work for both tensor cores. Longer sequences group
two adjacent K/V panels per work item to reduce scratch-state traffic. The
diagonal panel retains the same 256x256 decomposition, avoiding matrix
products that the final elementwise causal mask would otherwise discard.
"""

from __future__ import annotations

import functools
import math

import torch

from .configs import CAUSAL_HELION_CONFIGS
from .configs import causal_helion_config
import helion
import helion.language as hl

SHORT_SEQUENCE_CONFIG = helion.Config.from_dict(CAUSAL_HELION_CONFIGS["default"])

LONG_SEQUENCE_CONFIG = helion.Config.from_dict(
    CAUSAL_HELION_CONFIGS["low_level_scheduler"]
)


@helion.kernel(backend="pallas", static_shapes=True, config=SHORT_SEQUENCE_CONFIG)
def causal_attention_kernel(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    query_block_ids: torch.Tensor,
    kv_group_ids: torch.Tensor,
) -> torch.Tensor:
    sequence_length = hl.specialize(q_in.size(-2))
    head_dim = hl.specialize(q_in.size(-1))
    query_panel_size = hl.specialize(min(sequence_length, 2048))
    kv_compute_block = hl.specialize(min(query_panel_size, 512))
    diagonal_subtile = hl.specialize(256)

    # TPU vector reductions replicate one scalar across 128 hardware lanes.
    state_width = hl.specialize(128)
    score_repeats = hl.specialize(kv_compute_block // state_width)
    panel_score_repeats = hl.specialize(query_panel_size // state_width)
    head_repeats = hl.specialize(max(head_dim // state_width, 1))
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504

    heads = q_in.numel() // (sequence_length * head_dim)
    query_blocks = sequence_length // query_panel_size
    q = q_in.reshape([heads, query_blocks, query_panel_size, head_dim])
    # At 4K, separate K/V panels expose enough work to both tensor cores. For
    # longer sequences, pairing adjacent panels reduces online-softmax state
    # traffic without limiting parallelism.
    kv_group_width = hl.specialize(1 if sequence_length <= 4096 else 2)
    kv_groups = query_blocks // kv_group_width
    k = k_in.reshape([heads, kv_groups, kv_group_width, query_panel_size, head_dim])
    v = v_in.reshape([heads, kv_groups, kv_group_width, query_panel_size, head_dim])
    out = torch.empty_like(q)

    if query_panel_size == 1024:
        diagonal_parts = (0, 1, 2, 3)
        diagonal_offsets = (0, 256, 512, 768)
    else:
        diagonal_parts = (0, 1, 2, 3, 4, 5, 6, 7)
        diagonal_offsets = (0, 256, 512, 768, 1024, 1280, 1536, 1792)

    if sequence_length <= 2048:
        q_by_head = q_in.reshape([heads, sequence_length, head_dim])
        k_by_head = k_in.reshape([heads, sequence_length, head_dim])
        v_by_head = v_in.reshape([heads, sequence_length, head_dim])
        out_by_head = out.view([heads, sequence_length, head_dim])

    # Grid programs execute the worklist in order, so a query panel can carry
    # online-softmax state between consecutive K/V groups in private scratch.
    running_max = torch.empty(
        [query_panel_size, state_width], dtype=torch.float32, device=q.device
    )
    running_sum = torch.empty_like(running_max)
    accumulator = torch.empty(
        [query_panel_size, head_dim], dtype=torch.float32, device=q.device
    )

    for head, work in hl.grid([heads, query_block_ids.size(0)]):
        query_block = query_block_ids[work]
        kv_group = kv_group_ids[work]

        if sequence_length <= 2048:
            # Skip every 256x256 matrix multiplication wholly above the diagonal.
            for tile_q in hl.tile(sequence_length, block_size=sequence_length):
                query = q_by_head[head, tile_q, :] * qk_scale
                row_sum = hl.zeros([sequence_length, 1], dtype=torch.float32)
                acc = hl.zeros([sequence_length, head_dim], dtype=torch.float32)
                for tile_kv in hl.tile(sequence_length, block_size=sequence_length):
                    key_transposed = k_by_head[head, tile_kv, :].transpose(0, 1)
                    scores = torch.cat(
                        tuple(
                            torch.cat(
                                tuple(
                                    hl.full(
                                        [diagonal_subtile, diagonal_subtile],
                                        float("-inf"),
                                        dtype=torch.float32,
                                    )
                                    if key_part > query_part
                                    else torch.addmm(
                                        hl.zeros(
                                            [diagonal_subtile, diagonal_subtile],
                                            dtype=torch.float32,
                                        ),
                                        query[
                                            slice(
                                                diagonal_offsets[query_part],
                                                diagonal_offsets[query_part]
                                                + diagonal_subtile,
                                            ),
                                            :,
                                        ],
                                        key_transposed[
                                            :,
                                            slice(
                                                diagonal_offsets[key_part],
                                                diagonal_offsets[key_part]
                                                + diagonal_subtile,
                                            ),
                                        ],
                                    )
                                    for key_part in diagonal_parts
                                ),
                                dim=-1,
                            )
                            for query_part in diagonal_parts
                        ),
                        dim=0,
                    )
                    scores = torch.where(
                        tile_q.index[:, None] >= tile_kv.index[None, :],
                        scores,
                        float("-inf"),
                    )
                    row_max = torch.amax(scores, dim=-1)[:, None]
                    probabilities = torch.exp2(
                        scores - torch.cat([row_max] * sequence_length, dim=-1)
                    )
                    row_sum = torch.sum(probabilities, dim=-1)[:, None]
                    value = v_by_head[head, tile_kv, :]
                    probability_rows = tuple(
                        probabilities[
                            slice(
                                diagonal_offsets[query_part],
                                diagonal_offsets[query_part] + diagonal_subtile,
                            ),
                            :,
                        ]
                        for query_part in diagonal_parts
                    )
                    probability_parts = tuple(
                        tuple(
                            probability_rows[query_part][
                                :,
                                slice(
                                    diagonal_offsets[key_part],
                                    diagonal_offsets[key_part] + diagonal_subtile,
                                ),
                            ]
                            for key_part in diagonal_parts
                        )
                        for query_part in diagonal_parts
                    )
                    value_parts = tuple(
                        value[
                            slice(
                                diagonal_offsets[key_part],
                                diagonal_offsets[key_part] + diagonal_subtile,
                            ),
                            :,
                        ]
                        for key_part in diagonal_parts
                    )
                    products = tuple(
                        tuple(
                            hl.zeros([diagonal_subtile, head_dim], dtype=torch.float32)
                            if key_part > query_part
                            else torch.addmm(
                                hl.zeros(
                                    [diagonal_subtile, head_dim], dtype=torch.float32
                                ),
                                probability_parts[query_part][key_part].to(value.dtype),
                                value_parts[key_part],
                            )
                            for key_part in diagonal_parts
                        )
                        for query_part in diagonal_parts
                    )
                    acc = torch.cat(
                        tuple(
                            sum(products[query_part]) for query_part in diagonal_parts
                        ),
                        dim=0,
                    )
                denominator = torch.cat([row_sum] * head_dim, dim=-1)
                out_by_head[head, tile_q, :] = (acc / denominator).to(out.dtype)

        if sequence_length > 2048:
            if kv_group == 0:
                running_max[:, :] = hl.full(
                    [query_panel_size, state_width],
                    float("-inf"),
                    dtype=torch.float32,
                )
                running_sum[:, :] = hl.zeros(
                    [query_panel_size, state_width], dtype=torch.float32
                )
                accumulator[:, :] = hl.zeros(
                    [query_panel_size, head_dim], dtype=torch.float32
                )

            if kv_group_width == 2:
                # An odd diagonal panel shares its final group with the
                # preceding off-diagonal panel. Consume that first half before
                # applying the causal diagonal half below.
                if (kv_group == query_block // 2) & (query_block % 2 == 1):
                    for last_offdiag_tile_q in hl.tile(
                        query_panel_size, block_size=query_panel_size
                    ):
                        query = q[head, query_block, last_offdiag_tile_q, :] * qk_scale
                        for last_kv_half in range(1):
                            for tile_kv in hl.tile(
                                query_panel_size, block_size=kv_compute_block
                            ):
                                row_max = running_max[last_offdiag_tile_q, :]
                                row_sum = running_sum[last_offdiag_tile_q, :]
                                acc = accumulator[last_offdiag_tile_q, :]
                                key = k[head, kv_group, last_kv_half, tile_kv, :]
                                scores = torch.bmm(
                                    query.unsqueeze(0),
                                    key.transpose(0, 1).unsqueeze(0),
                                    torch.float32,
                                ).squeeze(0)
                                block_max = torch.amax(scores, dim=-1)[:, None]
                                new_max = torch.maximum(row_max, block_max)
                                probabilities = torch.exp2(
                                    scores
                                    - torch.cat([new_max] * score_repeats, dim=-1)
                                )
                                correction = torch.exp2(row_max - new_max)
                                row_sum = (
                                    row_sum * correction
                                    + torch.sum(probabilities, dim=-1)[:, None]
                                )
                                if head_dim < state_width:
                                    acc_correction = correction[:, hl.arange(head_dim)]
                                else:
                                    acc_correction = torch.cat(
                                        [correction] * head_repeats, dim=-1
                                    )
                                acc = acc * acc_correction
                                value = v[head, kv_group, last_kv_half, tile_kv, :]
                                acc = torch.addmm(
                                    acc, probabilities.to(value.dtype), value
                                )
                                running_max[last_offdiag_tile_q, :] = new_max
                                running_sum[last_offdiag_tile_q, :] = row_sum
                                accumulator[last_offdiag_tile_q, :] = acc
            if kv_group == query_block // kv_group_width:
                # The diagonal panel is decomposed into static 256x256 pieces,
                # so wholly masked QK and probability/V products are omitted.
                diagonal_half = query_block % kv_group_width
                for diagonal_tile_q in hl.tile(
                    query_panel_size, block_size=query_panel_size
                ):
                    diagonal_query = q[head, query_block, diagonal_tile_q, :] * qk_scale
                    for tile_kv in hl.tile(
                        query_panel_size, block_size=query_panel_size
                    ):
                        diagonal_max = running_max[diagonal_tile_q, :]
                        diagonal_sum = running_sum[diagonal_tile_q, :]
                        diagonal_acc = accumulator[diagonal_tile_q, :]
                        diagonal_key = k[head, kv_group, diagonal_half, tile_kv, :]
                        diagonal_key_transposed = diagonal_key.transpose(0, 1)
                        diagonal_scores = torch.cat(
                            tuple(
                                torch.cat(
                                    tuple(
                                        hl.full(
                                            [diagonal_subtile, diagonal_subtile],
                                            float("-inf"),
                                            dtype=torch.float32,
                                        )
                                        if key_part > query_part
                                        else torch.addmm(
                                            hl.zeros(
                                                [
                                                    diagonal_subtile,
                                                    diagonal_subtile,
                                                ],
                                                dtype=torch.float32,
                                            ),
                                            diagonal_query[
                                                slice(
                                                    diagonal_offsets[query_part],
                                                    diagonal_offsets[query_part]
                                                    + diagonal_subtile,
                                                ),
                                                :,
                                            ],
                                            diagonal_key_transposed[
                                                :,
                                                slice(
                                                    diagonal_offsets[key_part],
                                                    diagonal_offsets[key_part]
                                                    + diagonal_subtile,
                                                ),
                                            ],
                                        )
                                        for key_part in diagonal_parts
                                    ),
                                    dim=-1,
                                )
                                for query_part in diagonal_parts
                            ),
                            dim=0,
                        )
                        diagonal_scores = torch.where(
                            hl.arange(query_panel_size)[:, None]
                            >= tile_kv.index[None, :],
                            diagonal_scores,
                            float("-inf"),
                        )
                        diagonal_block_max = torch.amax(diagonal_scores, dim=-1)[
                            :, None
                        ]
                        diagonal_new_max = torch.maximum(
                            diagonal_max, diagonal_block_max
                        )
                        diagonal_probabilities = torch.exp2(
                            diagonal_scores
                            - torch.cat(
                                [diagonal_new_max] * panel_score_repeats,
                                dim=-1,
                            )
                        )
                        diagonal_correction = torch.exp2(
                            diagonal_max - diagonal_new_max
                        )
                        diagonal_sum = (
                            diagonal_sum * diagonal_correction
                            + torch.sum(diagonal_probabilities, dim=-1)[:, None]
                        )
                        if head_dim < state_width:
                            diagonal_acc_correction = diagonal_correction[
                                :, hl.arange(head_dim)
                            ]
                        else:
                            diagonal_acc_correction = torch.cat(
                                [diagonal_correction] * head_repeats, dim=-1
                            )
                        diagonal_value = v[head, kv_group, diagonal_half, tile_kv, :]
                        diagonal_acc_rows = tuple(
                            diagonal_acc[
                                slice(
                                    diagonal_offsets[query_part],
                                    diagonal_offsets[query_part] + diagonal_subtile,
                                ),
                                :,
                            ]
                            for query_part in diagonal_parts
                        )
                        diagonal_correction_rows = tuple(
                            diagonal_acc_correction[
                                slice(
                                    diagonal_offsets[query_part],
                                    diagonal_offsets[query_part] + diagonal_subtile,
                                ),
                                :,
                            ]
                            for query_part in diagonal_parts
                        )
                        diagonal_probability_rows = tuple(
                            diagonal_probabilities[
                                slice(
                                    diagonal_offsets[query_part],
                                    diagonal_offsets[query_part] + diagonal_subtile,
                                ),
                                :,
                            ]
                            for query_part in diagonal_parts
                        )
                        diagonal_probability_parts = tuple(
                            tuple(
                                diagonal_probability_rows[query_part][
                                    :,
                                    slice(
                                        diagonal_offsets[key_part],
                                        diagonal_offsets[key_part] + diagonal_subtile,
                                    ),
                                ]
                                for key_part in diagonal_parts
                            )
                            for query_part in diagonal_parts
                        )
                        diagonal_value_parts = tuple(
                            diagonal_value[
                                slice(
                                    diagonal_offsets[key_part],
                                    diagonal_offsets[key_part] + diagonal_subtile,
                                ),
                                :,
                            ]
                            for key_part in diagonal_parts
                        )
                        diagonal_products = tuple(
                            tuple(
                                hl.zeros(
                                    [diagonal_subtile, head_dim],
                                    dtype=torch.float32,
                                )
                                if key_part > query_part
                                else torch.addmm(
                                    hl.zeros(
                                        [diagonal_subtile, head_dim],
                                        dtype=torch.float32,
                                    ),
                                    diagonal_probability_parts[query_part][key_part].to(
                                        diagonal_value.dtype
                                    ),
                                    diagonal_value_parts[key_part],
                                )
                                for key_part in diagonal_parts
                            )
                            for query_part in diagonal_parts
                        )
                        diagonal_acc = torch.cat(
                            tuple(
                                diagonal_acc_rows[query_part]
                                * diagonal_correction_rows[query_part]
                                + sum(diagonal_products[query_part])
                                for query_part in diagonal_parts
                            ),
                            dim=0,
                        )
                        running_max[diagonal_tile_q, :] = diagonal_new_max
                        running_sum[diagonal_tile_q, :] = diagonal_sum
                        accumulator[diagonal_tile_q, :] = diagonal_acc
                    diagonal_sum = running_sum[diagonal_tile_q, :]
                    diagonal_acc = accumulator[diagonal_tile_q, :]
                    if head_dim < state_width:
                        diagonal_denominator = diagonal_sum[:, hl.arange(head_dim)]
                    else:
                        diagonal_denominator = torch.cat(
                            [diagonal_sum] * head_repeats, dim=-1
                        )
                    out[head, query_block, diagonal_tile_q, :] = (
                        diagonal_acc / diagonal_denominator
                    ).to(out.dtype)

            if kv_group < query_block // kv_group_width:
                # Earlier groups are wholly below the causal diagonal and can
                # use dense 512-row streaming tiles without a mask.
                for offdiag_tile_q in hl.tile(
                    query_panel_size, block_size=query_panel_size
                ):
                    query = q[head, query_block, offdiag_tile_q, :] * qk_scale
                    for kv_half in range(kv_group_width):
                        for tile_kv in hl.tile(
                            query_panel_size, block_size=kv_compute_block
                        ):
                            row_max = running_max[offdiag_tile_q, :]
                            row_sum = running_sum[offdiag_tile_q, :]
                            acc = accumulator[offdiag_tile_q, :]
                            key = k[head, kv_group, kv_half, tile_kv, :]
                            scores = torch.bmm(
                                query.unsqueeze(0),
                                key.transpose(0, 1).unsqueeze(0),
                                torch.float32,
                            ).squeeze(0)
                            block_max = torch.amax(scores, dim=-1)[:, None]
                            new_max = torch.maximum(row_max, block_max)
                            probabilities = torch.exp2(
                                scores - torch.cat([new_max] * score_repeats, dim=-1)
                            )
                            correction = torch.exp2(row_max - new_max)
                            row_sum = (
                                row_sum * correction
                                + torch.sum(probabilities, dim=-1)[:, None]
                            )
                            if head_dim < state_width:
                                acc_correction = correction[:, hl.arange(head_dim)]
                            else:
                                acc_correction = torch.cat(
                                    [correction] * head_repeats, dim=-1
                                )
                            acc = acc * acc_correction
                            value = v[head, kv_group, kv_half, tile_kv, :]
                            acc = torch.addmm(acc, probabilities.to(value.dtype), value)
                            running_max[offdiag_tile_q, :] = new_max
                            running_sum[offdiag_tile_q, :] = row_sum
                            accumulator[offdiag_tile_q, :] = acc

    return out.view(q_in.size())


# Both specializations compile the same Helion source. The low-level scheduler
# starts helping at 8K for D=256, and at 32K for the smaller head dimensions;
# below those crossovers its setup cost is larger than its scheduling benefit.
_causal_attention_long = helion.kernel(
    backend="pallas", static_shapes=True, config=LONG_SEQUENCE_CONFIG
)(causal_attention_kernel.fn)


def configured_causal_attention(sequence_length: int, head_dim: int) -> helion.Kernel:
    """Return the measured-best code-generation config for this shape."""
    config = causal_helion_config(head_dim, sequence_length)
    if config["pallas_use_low_level_scheduler"]:
        return _causal_attention_long
    return causal_attention_kernel


def causal_attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """Run causal attention using the shape's fixed TPU configuration."""
    sequence_length = q_in.size(-2)
    query_blocks, kv_groups = causal_worklist(
        sequence_length, q_in.size(-1), q_in.device
    )
    kernel = configured_causal_attention(sequence_length, q_in.size(-1))
    flat_shape = (-1, sequence_length, q_in.size(-1))
    return kernel(
        q_in.view(flat_shape),
        k_in.view(flat_shape),
        v_in.view(flat_shape),
        query_blocks,
        kv_groups,
    ).view(q_in.size())


@functools.cache
def causal_worklist(
    sequence_length: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    del head_dim
    query_panel_size = min(sequence_length, 2048)
    query_block_ids: list[int] = []
    kv_group_ids: list[int] = []
    query_blocks = sequence_length // query_panel_size
    kv_group_width = 1 if sequence_length <= 4096 else 2
    for query_block in range(query_blocks):
        for kv_group in range(query_block // kv_group_width + 1):
            query_block_ids.append(query_block)
            kv_group_ids.append(kv_group)
    return (
        torch.tensor(query_block_ids, dtype=torch.int32, device=device),
        torch.tensor(kv_group_ids, dtype=torch.int32, device=device),
    )
