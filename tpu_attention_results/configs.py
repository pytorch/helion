"""Measured-best fixed configurations used by the TPU attention benchmark.

The key order is ``(head_dim, sequence_length)``. These are deliberately plain
dictionaries so the result artifact does not depend on Helion or Tokamax just
to inspect its schedules.
"""

from __future__ import annotations

from typing import Any

HEAD_DIMS = (64, 128, 256)
SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


def _unroll(
    batch_heads: int,
    query_rows: int,
    kv_rows: int,
    *,
    loop_order: tuple[int, int] = (0, 1),
    pre_broadcast: bool | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "block_sizes": [batch_heads, query_rows, kv_rows],
        "loop_orders": [list(loop_order)],
        "pallas_load_buffer_count": [1, 1, 1],
        "pallas_loop_type": "unroll",
    }
    if pre_broadcast is not None:
        config["pallas_pre_broadcast"] = pre_broadcast
    return config


def _pipeline(
    batch_heads: int,
    query_rows: int,
    kv_rows: int,
    group_size: int,
    *,
    loop_order: tuple[int, int] = (0, 1),
    low_level_scheduler: bool,
) -> dict[str, Any]:
    return {
        "block_sizes": [batch_heads, query_rows, kv_rows],
        "loop_orders": [list(loop_order)],
        "pallas_emit_pipeline_group_size": group_size,
        "pallas_fold_dot_lhs_cast": True,
        "pallas_loop_type": "emit_pipeline",
        "pallas_pre_broadcast": True,
        "pallas_use_low_level_scheduler": low_level_scheduler,
    }


DENSE_HELION_CONFIGS = {
    (64, 1024): _unroll(8, 1024, 512),
    (64, 2048): _unroll(4, 1024, 1024),
    (64, 4096): _unroll(2, 1024, 2048),
    (64, 8192): _unroll(2, 512, 2048, pre_broadcast=False),
    (64, 16384): _unroll(1, 512, 2048),
    (64, 32768): _pipeline(1, 4096, 256, 16, low_level_scheduler=True),
    (64, 65536): _pipeline(1, 4096, 256, 16, low_level_scheduler=True),
    (128, 1024): _unroll(8, 1024, 1024),
    (128, 2048): _unroll(4, 1024, 1024),
    (128, 4096): _unroll(2, 1024, 1024),
    (128, 8192): _unroll(2, 512, 2048, pre_broadcast=False),
    (128, 16384): _unroll(1, 512, 2048),
    (128, 32768): _pipeline(1, 4096, 256, 16, low_level_scheduler=True),
    (128, 65536): _pipeline(1, 4096, 256, 16, low_level_scheduler=True),
    (256, 1024): _unroll(4, 1024, 1024),
    (256, 2048): _unroll(4, 1024, 1024),
    (256, 4096): _unroll(2, 1024, 2048),
    (256, 8192): _unroll(2, 512, 2048, pre_broadcast=False),
    (256, 16384): _unroll(1, 512, 1024),
    (256, 32768): _pipeline(1, 8192, 256, 8, low_level_scheduler=True),
    (256, 65536): _pipeline(1, 4096, 256, 16, low_level_scheduler=True),
}


def _splash(
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    *,
    scheduler: bool,
    q_layout: str = "HEAD_DIM_MINOR",
    k_layout: str = "HEAD_DIM_MINOR",
    v_layout: str = "HEAD_DIM_MINOR",
    diagonal_skip: bool = False,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "block_q": block_q,
        "block_kv": block_kv,
        "block_kv_compute": block_kv_compute,
        "q_layout": q_layout,
        "k_layout": k_layout,
        "v_layout": v_layout,
        "use_experimental_scheduler": scheduler,
        "use_base2_exp": True,
    }
    if diagonal_skip:
        config.update(
            qk_diag_skip=True,
            qk_diag_grid=4,
            sv_diag_skip=True,
        )
    return config


DENSE_TOKAMAX_CONFIGS = {
    (64, 1024): _splash(
        1024,
        1024,
        512,
        scheduler=True,
        q_layout="SEQ_MINOR",
        v_layout="SEQ_MINOR",
    ),
    (64, 2048): _splash(
        2048,
        2048,
        1024,
        scheduler=False,
        q_layout="SEQ_MINOR",
        k_layout="SEQ_MINOR",
    ),
    (64, 4096): _splash(
        4096,
        4096,
        1024,
        scheduler=False,
        q_layout="SEQ_MINOR",
        v_layout="SEQ_MINOR",
    ),
    (64, 8192): _splash(
        4096,
        4096,
        1024,
        scheduler=False,
        q_layout="SEQ_MINOR",
        v_layout="SEQ_MINOR",
    ),
    (64, 16384): _splash(4096, 4096, 256, scheduler=True),
    (64, 32768): _splash(4096, 4096, 256, scheduler=True),
    (64, 65536): _splash(4096, 4096, 256, scheduler=True),
    (128, 1024): _splash(1024, 1024, 1024, scheduler=True),
    (128, 2048): _splash(2048, 2048, 256, scheduler=True),
    (128, 4096): _splash(2048, 4096, 512, scheduler=True),
    (128, 8192): _splash(4096, 4096, 256, scheduler=True),
    (128, 16384): _splash(4096, 4096, 256, scheduler=True),
    (128, 32768): _splash(4096, 4096, 256, scheduler=True),
    (128, 65536): _splash(8192, 2048, 256, scheduler=True),
    (256, 1024): _splash(1024, 1024, 256, scheduler=True),
    (256, 2048): _splash(2048, 2048, 256, scheduler=True),
    (256, 4096): _splash(2048, 4096, 512, scheduler=True),
    (256, 8192): _splash(
        2048,
        8192,
        512,
        scheduler=True,
        v_layout="SEQ_MINOR",
    ),
    (256, 16384): _splash(8192, 2048, 256, scheduler=True),
    (256, 32768): _splash(4096, 4096, 256, scheduler=True),
    (256, 65536): _splash(8192, 2048, 256, scheduler=True),
}


CAUSAL_HELION_CONFIGS = {
    "default": {
        "pallas_fold_dot_lhs_cast": True,
        "pallas_internal_scratch": True,
        "pallas_loop_type": "unroll",
        "pallas_pre_broadcast": False,
        "pallas_use_low_level_scheduler": False,
    },
    "low_level_scheduler": {
        "pallas_fold_dot_lhs_cast": True,
        "pallas_internal_scratch": True,
        "pallas_loop_type": "unroll",
        "pallas_pre_broadcast": False,
        "pallas_use_low_level_scheduler": True,
    },
}


def causal_helion_config(head_dim: int, sequence_length: int) -> dict[str, Any]:
    """Select the measured-best scheduler for the causal source kernel."""
    use_low_level_scheduler = sequence_length >= 32768 or (
        head_dim == 256 and sequence_length >= 8192
    )
    name = "low_level_scheduler" if use_low_level_scheduler else "default"
    return CAUSAL_HELION_CONFIGS[name]


CAUSAL_TOKAMAX_CONFIGS = {
    **{
        (head_dim, 1024): _splash(
            1024,
            1024,
            1024,
            scheduler=False,
            diagonal_skip=True,
        )
        for head_dim in HEAD_DIMS
    },
    **{
        (head_dim, sequence_length): _splash(
            2048,
            2048,
            2048,
            scheduler=False,
            diagonal_skip=True,
        )
        for head_dim in HEAD_DIMS
        for sequence_length in (2048, 4096)
    },
    (64, 8192): _splash(2048, 2048, 2048, scheduler=False, diagonal_skip=True),
    (64, 16384): _splash(2048, 2048, 2048, scheduler=False, diagonal_skip=True),
    (128, 8192): _splash(2048, 2048, 2048, scheduler=False, diagonal_skip=True),
    (128, 16384): _splash(2048, 2048, 2048, scheduler=False, diagonal_skip=True),
    (256, 8192): _splash(2048, 2048, 512, scheduler=True),
    (256, 16384): _splash(2048, 4096, 512, scheduler=True),
    **{
        (head_dim, sequence_length): _splash(2048, 4096, 512, scheduler=True)
        for head_dim in HEAD_DIMS
        for sequence_length in (32768, 65536)
    },
}


def validate_config_coverage() -> None:
    """Ensure every reported shape has one fixed config for each implementation."""
    expected = {
        (head_dim, sequence_length)
        for head_dim in HEAD_DIMS
        for sequence_length in SEQUENCE_LENGTHS
    }
    assert DENSE_HELION_CONFIGS.keys() == expected
    assert DENSE_TOKAMAX_CONFIGS.keys() == expected
    assert CAUSAL_TOKAMAX_CONFIGS.keys() == expected


validate_config_coverage()
