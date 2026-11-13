"""Source and graph-replay checks for paired probability scheduling choices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cute_optional_lse import bind_paged_attention_length_replay
from .cute_optional_lse import check_optional_lse_replay
from .cute_optional_lse import check_paged_attention_length_replay
from .cute_paged_base2 import base2_attention_fixture
from .cute_paged_reference import reference_attention_fixture
from .cute_paged_v_lag import paired_v_lag_config
import helion

if TYPE_CHECKING:
    import torch


def check_probability_schedule_replay(
    schedule: str,
    form: str,
    query_len: int,
    base2: bool,
    reverse: bool,
    fast_math: bool,
    sleep: bool,
    lag: int,
    return_lse: bool,
    device: torch.device | str,
    *,
    capture_overlap: bool = False,
) -> None:
    fixture = base2_attention_fixture if base2 else reference_attention_fixture
    kernel, initial = fixture(form, query_len, return_lse, device)
    bound, inputs = bind_paged_attention_length_replay(
        kernel,
        (*initial, reverse),
        form,
        fast_math=fast_math,
        reverse_pages=reverse,
    )
    config = paired_v_lag_config(lag)
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_probability_schedule=schedule,
        cute_mma_dag_sleep_waits=sleep,
    )
    if capture_overlap:
        config.config.update(
            cute_mma_dag_query_tma=True,
            cute_pdl=True,
            cute_pdl_capture_overlap=True,
        )
    source = bound.to_code(config)
    assert "paged_attention_paired_body(" in source
    assert f"probability_schedule={schedule!r}" in source
    if capture_overlap:
        assert "query_map=" in source
        assert "capture_pdl_wait_mode=capture_pdl_wait_mode" in source
    check_optional_lse_replay(
        bound,
        config,
        inputs,
        form,
        return_lse,
        reference_scores=True,
        reverse_pages=reverse,
        base2_scores=base2,
    )
    check_paged_attention_length_replay(
        bound,
        config,
        inputs,
        form,
        return_lse,
        reference_scores=True,
        reverse_pages=reverse,
        base2_scores=base2,
    )
