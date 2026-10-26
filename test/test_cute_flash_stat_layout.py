"""Check emitted statistic buffers with real CuTe address and bank mappings."""

from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.device_function import DeviceFunction

pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
ir = pytest.importorskip("cutlass._mlir.ir")


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    initialized = torch.cuda.is_initialized()
    with (
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=True),
        _mock_cuda_unavailable(),
        _forbid_native_compile(),
        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield
    assert torch.cuda.is_initialized() == initialized


_CASES = [
    (head_dim, dtype, causal, family, transport)
    for head_dim in (64, 128)
    for dtype in (torch.float16, torch.bfloat16)
    for causal in (False, True)
    for family in ("fa4", "fa4_2cta", "fa4_2cta_causal")
    for transport in ("ring2", "single")
    if (
        family == "fa4"
        or (not causal and family == "fa4_2cta")
        or (
            causal
            and family == "fa4_2cta_causal"
            and head_dim == 64
            and dtype is torch.float16
        )
    )
    and (transport == "ring2" or not causal)
]


@pytest.mark.parametrize(("head_dim", "dtype", "causal", "family", "transport"), _CASES)
def test_statistic_rows_have_disjoint_slots_and_conflict_free_warps(
    head_dim: int,
    dtype: torch.dtype,
    causal: bool,
    family: str,
    transport: str,
) -> None:
    config = cute_flash.resolve_flash_config(
        head_dim,
        8,
        {
            "cute_flash_pipeline_family": family,
            "cute_flash_stat_transport": transport,
            "cute_flash_softmax_disc": transport == "ring2",
            "cute_flash_exp2_packet": "1x1",
            "cute_flash_kv_stage": 3,
            "cute_flash_s_load_rep": 16,
            "cute_flash_e2e_schedule": "16/4",
            "cute_flash_rowmax": "software",
            "cute_flash_persistent": False,
        },
        dtype=dtype,
        num_bh=64,
        is_causal=causal,
        standard_dense_output=not causal,
        standard_causal_output=causal,
        target_device_capability=(10, 3),
    )
    assert config.stat_transport == transport
    assert config.use_2cta_instrs == (family != "fa4")
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
        target_device_capability=(10, 3),
    )
    assignments = [
        node
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "flash_scale_t"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    get_tensor = assignments[0].value
    assert isinstance(get_tensor, ast.Call)
    layout_call = get_tensor.args[0]
    assert isinstance(layout_call, ast.Call)
    assert ast.unparse(layout_call.func) == "cute.make_layout"
    # Evaluate the actual generated layout, not a parallel model of its strides.
    layout = eval(
        compile(ast.Expression(layout_call), "<emitted statistic layout>", "eval"),
        {"cute": cute},
    )
    slots = 2 if transport == "ring2" else 1
    addresses = []
    for slot in range(slots):
        for query in range(2):
            for warp in range(4):
                coordinates = [
                    (slot, query, warp * 32 + lane)
                    if transport == "ring2"
                    else query * 128 + warp * 32 + lane
                    for lane in range(32)
                ]
                offsets = [int(layout(coordinate)) for coordinate in coordinates]
                addresses.extend(offsets)
                # Statistics are Float32: one word per bank, with 32 banks.
                assert len({offset % 32 for offset in offsets}) == 32
    assert sorted(addresses) == list(range(slots * 2 * 128))
    assert int(cute.cosize(layout)) == len(addresses)
