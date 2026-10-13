from __future__ import annotations

import ast
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_aux import _cpu_codegen
from test.test_cute_interchanged_store_dce import _PAIRS
from test.test_cute_interchanged_store_dce import _program

import helion
from helion import exc
from helion._compiler import tile_strategy as lanes
import helion.language as hl

MARKER = "_helion_lane_reduce(partial, 'sum', 0.0, 32, 1, 0, '', 1, 1, 'lane')"


def _loop(body: str, *, output: bool = True) -> ast.For:
    serial = ast.For(
        target=ast.Name(id="step", ctx=ast.Store()),
        iter=ast.parse("range(2)", mode="eval").body,
        body=cast("list[ast.stmt]", ast.parse(body).body),
        orelse=[],
    )
    if output:
        serial.body.extend(ast.parse("output.store(reduced)").body)
    return lanes._create_lane_loop(
        "lane",
        2,
        [
            *ast.parse("state = lane + 1.0").body,
            serial,
            *ast.parse("final.store(state)").body,
        ],
    )


@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("spelling", ["plain", "phi", "conditional", "augmented"])
def test_feedback_carry_rejects_before_marker_or_input_mutation(output, spelling):
    body = f"partial = state\nreduced = {MARKER}\n"
    renames = {}
    if spelling == "plain":
        body += "state = state + reduced"
    elif spelling == "phi":
        body = "saved = state\n" + body.replace("partial = state", "partial = saved")
        body += "future = saved + reduced"
        renames = {"saved": "state", "future": "state"}
    elif spelling == "conditional":
        body += "if flag:\n    state = state + reduced"
    else:
        body += "state += reduced"
    loop = _loop(body, output=output)
    before = ast.dump(loop)
    original_renames = renames.copy()
    with pytest.raises(
        exc.BackendUnsupported, match="reduction-dependent loop-carried"
    ):
        lanes.interchange_lane_outside_serial_reductions([loop], rename_groups=renames)
    assert ast.dump(loop) == before
    assert renames == original_renames


def test_independent_update_that_feeds_later_reduction_also_rejects():
    loop = _loop(f"state = state + 1.0\npartial = state\nreduced = {MARKER}")
    with pytest.raises(
        exc.BackendUnsupported, match="reduction-dependent loop-carried"
    ):
        lanes.interchange_lane_outside_serial_reductions([loop])


def test_reduction_result_feeding_an_otherwise_unread_carry_rejects():
    loop = _loop(f"partial = lane + step\nreduced = {MARKER}\nstate = state + reduced")
    with pytest.raises(
        exc.BackendUnsupported, match="reduction-dependent loop-carried"
    ):
        lanes.interchange_lane_outside_serial_reductions([loop])


def test_each_call_rechecks_current_body_and_current_phi_identity():
    def candidate():
        return _loop(
            f"partial = lane + step\nreduced = {MARKER}\nfuture = state + reduced"
        )

    loop = candidate()
    lanes.interchange_lane_outside_serial_reductions([loop])
    # The same spellings acquire their actual compiler phi identity on a fresh call.
    with pytest.raises(
        exc.BackendUnsupported, match="reduction-dependent loop-carried"
    ):
        lanes.interchange_lane_outside_serial_reductions(
            [candidate()], rename_groups={"future": "state"}
        )
    # A changed statement must not reuse an earlier admission.
    loop = _loop(f"partial = lane + step\nreduced = {MARKER}\nstate = state + 1.0")
    lanes.interchange_lane_outside_serial_reductions([loop])
    loop = _loop(f"partial = lane + step\nreduced = {MARKER}\nstate = state + reduced")
    with pytest.raises(
        exc.BackendUnsupported, match="reduction-dependent loop-carried"
    ):
        lanes.interchange_lane_outside_serial_reductions([loop])


def test_norm_backward_independent_accumulator_keeps_exact_transform():
    original = _program()
    with patch.object(lanes, "_reject_serial_reduction_carry", return_value=None):
        before = lanes.interchange_lane_outside_serial_reductions(
            [original], proven_disjoint_tensor_pairs=_PAIRS
        )
    after = lanes.interchange_lane_outside_serial_reductions(
        [_program()], proven_disjoint_tensor_pairs=_PAIRS
    )
    assert [ast.dump(x) for x in before] == [ast.dump(x) for x in after]


def test_per_token_initialization_has_no_serial_carry():
    loop = _loop(
        f"state = lane + step\npartial = state\nreduced = {MARKER}\nstate = state + reduced"
    )
    assert len(lanes.interchange_lane_outside_serial_reductions([loop])) == 2


@helion.kernel()
def _feedback(initial, output, final):
    rows, columns = initial.shape
    for rr in hl.tile(rows):
        kk = hl.arange(columns)
        state = initial[rr, kk]
        for step in hl.grid(2):
            total = state.sum(-1)
            state = state + total[:, None]
            output[step, rr] = state.sum(-1)
        final[rr, kk] = state


@helion.kernel()
def _weighted_feedback(initial, key, output, final):
    rows, columns = initial.shape
    for rr in hl.tile(rows):
        kk = hl.arange(columns)
        state = initial[rr, kk]
        for step in range(2):
            k = key[step, kk]
            state = state * 0.5
            residual = 1.0 - (state * k[None, :]).sum(-1)
            state = state + residual[:, None] * k[None, :]
            output[step, rr] = state.sum(-1)
        final[rr, kk] = state


@pytest.mark.parametrize("kernel", [_feedback, _weighted_feedback])
@pytest.mark.parametrize("vector", [1, 2])
def test_actual_codegen_passes_live_phi_mapping_and_rejects(kernel, vector):
    initial, output, final = torch.ones(4, 64), torch.empty(2, 4), torch.empty(4, 64)
    args = (
        (initial, output, final)
        if kernel is _feedback
        else (initial, torch.ones(2, 64), output, final)
    )
    with _cpu_codegen():
        kernel.settings.backend = "cute"
        kernel.settings.fast_math = False
        bound = kernel._bind_isolated(args)
        cfg = bound.config_spec.default_config()
        cfg = helion.Config(
            **dict(
                cfg.config,
                block_sizes=[4],
                num_threads=[4, 32],
                cute_vector_widths=[vector, 1, 1],
                cute_reduction_reloads=["register"],
            )
        )
        cfg = bound.config_spec.create_config_generation().strict_config_pair(cfg)[1]
        with pytest.raises(
            exc.BackendUnsupported, match="reduction-dependent loop-carried"
        ):
            bound.to_code(cfg)
