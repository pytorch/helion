from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.cooperative_row_normalization import NormalizedRow
from helion._compiler.cute.cooperative_row_normalization import _complete_consumer
from helion._compiler.cute.cooperative_row_normalization import _tile_coordinate
from helion._compiler.cute.cooperative_row_normalization import private_scratch
from helion._compiler.cute.cooperative_row_normalization import scratch_words
from helion._compiler.cute.cooperative_row_normalization import sweep_signature
from helion._compiler.device_function import TensorArg
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _column_stats(
    x: torch.Tensor, other: torch.Tensor, weight: torch.Tensor, use_output: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, _, width = x.shape
    published = torch.empty_like(other)
    result = torch.empty_like(other)
    for i, j in hl.tile([rows, width], block_size=[1, None]):
        published[i, j] = x[i, :, j].float().sum(dim=1).to(x.dtype)
    hl.barrier()
    for i in hl.tile(rows, block_size=1):
        selected = published if use_output else other
        v = selected[i, :].float()
        inv = torch.rsqrt((v * v).mean(dim=-1) + 0.125)
        result[i, :] = (v * inv[:, None]).to(x.dtype) * weight[None, :]
    return published, result


def test_typed_consumer_must_read_the_actual_rounded_row() -> None:
    x = torch.empty(1, 3, 384, dtype=torch.bfloat16)
    y = torch.empty(1, 384, dtype=x.dtype)
    w = torch.empty(384, dtype=x.dtype)
    with cpu_render_target(fake_device_sm_count=7):
        yes = cpu_bind(_column_stats, (x, y, w, True))
        no = cpu_bind(_column_stats, (x, y, w, False))
        assert yes.config_spec.cute_normalizable_column is not None
        assert yes.config_spec.cute_normalizable_column[1] == 384
        assert no.config_spec.cute_normalizable_column is None
        # This small standalone producer has no checked multi-CTA staged launch.
        assert not any(
            c.get("cute_cooperative_row_norm")
            for _, c in yes.config_spec.create_config_generation().seed_flat_config_pairs()
        )


def test_partial_storage_consumes_only_owned_warps_and_uses_a_fresh_buffer() -> None:
    assert [scratch_words(64 * n) for n in (1, 32, 33, 44, 64, 65)] == [
        64,
        64,
        96,
        96,
        96,
        128,
    ]
    with pytest.raises(exc.InvalidConfig, match="full pairs"):
        scratch_words(127)
    buf = torch.empty(96, dtype=torch.uint32, device=torch.device("meta"))
    assert private_scratch(
        TensorArg(
            "a", buf, "torch.zeros((96,), device=data.device, dtype=torch.uint32)"
        ),
        96,
    )
    assert not private_scratch(TensorArg("a", buf, "user_buffer"), 96)
    assert not private_scratch(
        TensorArg(
            "a", buf, "torch.empty((96,), device=data.device, dtype=torch.uint32)"
        ),
        96,
    )


def test_partial_index_must_be_the_logical_tile_index() -> None:
    assignments = "pid = virtual\npid = pid - cutlass.Int32(7)\nunit = 1\nowner = pid // unit\norigin = owner * TILE"

    def prove(program: str) -> str | None:
        return _tile_coordinate(
            ast.parse(program).body, "origin", "virtual", 7, "TILE", 128, 3
        )

    assert prove(assignments) == "owner"
    for wrong in (
        assignments.replace("Int32(7)", "Int32(6)"),
        assignments.replace("unit = 1", "unit = 2"),
        assignments.replace("owner * TILE", "owner * 64"),
        assignments + "\nowner = virtual",
        assignments + "\nuntracked()",
    ):
        assert prove(wrong) is None


def test_distributed_consumer_cannot_drop_an_extra_effect_or_live_copy() -> None:

    payload = ast.parse(
        "for j in range(4):\n    copy = inverse\n    out.store(data[j] * copy)"
    ).body
    plan = NormalizedRow(
        2,
        "data",
        "weight",
        "out",
        256,
        "eps",
        "acc",
        "inverse",
        "tid",
        sweep_signature(payload, "inverse"),
    )
    (loop,) = ast.parse(
        "for it in range(cutlass.Int32(0), cutlass.Int32((6-5)//_NUM_SM + "
        "cutlass.Int32(cutlass.Int32(cute.arch.block_idx()[0]) < (6-5)%_NUM_SM))):\n"
        "    v = 5 + (cutlass.Int32(cute.arch.block_idx()[0]) + it * _NUM_SM)\n"
        "    if v>=5 and v<6:\n"
        "        acc = cutlass.Float32(0)\n"
        "        for j in range(4):\n"
        "            out.store(data[j] * inverse)"
    ).body
    assert isinstance(loop, ast.For)
    guard = loop.body[-1]
    assert isinstance(guard, ast.If)
    assert _complete_consumer(loop, plan)
    guard.body[0:0] = ast.parse("unrelated.store(7)").body
    assert not _complete_consumer(loop, plan)
    del guard.body[0]
    guard.body.extend(ast.parse("other.store(inverse)").body)
    assert not _complete_consumer(loop, plan)
    # A copy that escapes the loop or whose source changes is semantically live.
    assert sweep_signature(
        [*payload, *ast.parse("other.store(copy)").body], "inverse"
    ) != (*plan.sweep, ast.dump(ast.parse("other.store(copy)").body[0]))
