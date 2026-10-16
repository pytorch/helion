from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.device_function import DeviceFunction

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only test")
        ),
    ):
        yield


def _source(
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    *,
    causal: bool = False,
    family: str = "fa4",
    persistent: bool = False,
    transport: str = "ring2",
    chunked: bool = True,
    kv_order: str = "descending",
    loop_split: bool = True,
) -> str:
    config = cute_flash.resolve_flash_config(
        head_dim,
        8,
        {
            "cute_flash_pipeline_family": family,
            "cute_flash_stat_transport": transport,
            "cute_flash_softmax_disc": chunked,
            "cute_flash_rowmax": "software",
            "cute_flash_e2e_schedule": "16/2",
            "cute_flash_exp2_packet": "1x1",
            "cute_flash_rescale_threshold": 8.0,
            "cute_flash_kv_stage": 2,
            "cute_flash_persistent": persistent,
            "cute_flash_persistent_loop": "counted",
            "cute_flash_causal_kv_order": kv_order,
            "cute_flash_causal_loop_split": loop_split,
            "cute_flash_role_map": "helion",
        },
        dtype=dtype,
        num_bh=64,
        is_causal=causal,
        standard_dense_output=not causal,
        standard_causal_output=causal,
    )
    assert config.stat_transport == transport
    assert config.softmax_disc is chunked
    assert config.persistent is persistent
    assert config.pipeline_family == family
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=8,
        sequence_extent=1024,
        num_bh=64,
        total_tiles=128 if config.use_2cta_instrs else 256,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.Float16" if dtype is torch.float16 else "cutlass.BFloat16",
        score_plan=causal_score_plan(head_dim)
        if causal
        else dense_score_plan(head_dim),
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def _terminal_acks(source: str) -> list[tuple[ast.If, ast.Call]]:
    result = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If) or len(node.body) != 1:
            continue
        statement = node.body[0]
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        call = statement.value
        if (
            ast.unparse(call.func) == "_helion_flash_rt.mbar_spin_wait"
            and len(call.args) == 3
            and "flash_s_corr_prod_index ^ 1" in ast.unparse(call.args[0])
        ):
            result.append((node, call))
    return sorted(result, key=lambda item: ast.unparse(item[1].args[0]))


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("kv_order", ("ascending", "descending"))
@pytest.mark.parametrize("loop_split", (False, True))
def test_ring2_final_sum_waits_for_last_alpha_notification(
    head_dim: int,
    dtype: torch.dtype,
    causal: bool,
    kv_order: str,
    loop_split: bool,
) -> None:
    source = _source(
        head_dim,
        dtype,
        causal=causal,
        kv_order=kv_order,
        loop_split=loop_split,
    )
    acks = _terminal_acks(source)
    assert len(acks) == 2
    for stage, (guard, call) in enumerate(acks):
        bound = "flash_num_active_kv" if causal else "_flash_num_kv_tiles"
        assert ast.unparse(guard.test) == f"{bound} > 1"
        assert ast.unparse(call.args[1]) == (
            "flash_s_corr_prod_phase ^ flash_s_corr_prod_index"
        )
        ack = source.index(ast.unparse(call))
        acquire = source.index(
            f"_helion_flash_rt.mbar_spin_wait(flash_s{stage}_corr_empty_ptr + flash_s_corr_prod_index, flash_s_corr_prod_phase, 10000000)",
            ack,
        )
        store = source.index(
            f"flash_scale_t[flash_s_corr_prod_index, {stage}, flash_local_tidx] = flash_row_sum",
            acquire,
        )
        notify = source.index(
            f"_helion_flash_rt.named_barrier_arrive_unaligned({3 + 4 * stage} + warp_idx % 4, 64)",
            store,
        )
        assert ack < acquire < store < notify
        assert "flash_s_corr_prod_index ^=" not in source[ack:store]
        assert "flash_s_corr_prod_phase ^=" not in source[ack:store]


@pytest.mark.parametrize("family", ("fa4", "fa4_2cta", "fa4_deep_1cta"))
@pytest.mark.parametrize("persistent", (False, True))
def test_ring2_terminal_ack_covers_dense_pipeline_and_persistence_choices(
    family: str, persistent: bool
) -> None:
    source = _source(family=family, persistent=persistent)
    assert len(_terminal_acks(source)) == 2


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("transport", ("ring2", "single", "single_final"))
def test_other_transports_preserve_their_own_terminal_protocol(
    dtype: torch.dtype, transport: str
) -> None:
    source = _source(
        head_dim=64 if transport == "single_final" else 128,
        dtype=dtype,
        transport=transport,
        chunked=False,
    )
    assert len(_terminal_acks(source)) == (2 if transport == "ring2" else 0)


def _evaluate(expression: ast.expr, values: dict[str, int]) -> int:
    return eval(
        compile(ast.Expression(expression), "<generated-phase>", "eval"),
        {"__builtins__": {}},
        values,
    )


@pytest.mark.parametrize("causal", (False, True))
def test_generated_terminal_wait_observes_consumption_across_ring_wraps(
    causal: bool,
) -> None:
    # Execute the generated guard/address/phase expressions against a two-slot
    # barrier model. Delay only the last alpha consumer: the old row-sum-slot
    # acquire is ready, but its notification barrier has not yet reset.
    source = _source(causal=causal)
    for stage, (guard, call) in enumerate(_terminal_acks(source)):
        index, phase = 0, 0
        empty_phases = [1, 1]  # Correction's initial empty arrivals completed.
        for trips in (1, 2, 3, 4, 7, 8, 1, 3, 2, 9):
            previous = None
            for iteration in range(trips - 1):
                assert empty_phases[index] != phase
                previous = index
                if iteration < trips - 2:
                    empty_phases[index] ^= 1
                index ^= 1
                if index == 0:
                    phase ^= 1
            values = {
                "flash_s_corr_prod_index": index,
                "flash_s_corr_prod_phase": phase,
                f"flash_s{stage}_corr_empty_ptr": 0,
                "flash_num_active_kv": trips,
                "_flash_num_kv_tiles": trips,
            }
            assert empty_phases[index] != phase  # Old terminal acquire passes.
            if _evaluate(guard.test, values):
                ack_slot = _evaluate(call.args[0], values)
                ack_phase = _evaluate(call.args[1], values)
                assert ack_slot == previous
                assert empty_phases[ack_slot] == ack_phase  # Must wait.
                empty_phases[ack_slot] ^= 1  # Last alpha consumed/reset.
                assert empty_phases[ack_slot] != ack_phase  # Can publish sum.
            else:
                assert previous is None  # Single-iteration item has no alpha.
            # Publish/consume final sum, preserving phase/index into next item.
            empty_phases[index] ^= 1
            index ^= 1
            if index == 0:
                phase ^= 1
