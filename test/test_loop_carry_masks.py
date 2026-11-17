"""A loop carry updated from a partial tile's loads (``mask_loop_carries``).

``acc = hl.zeros([tile_m, bn])`` updated with ``acc + torch.exp(x[tile_m,
tile_n])`` in ``for tile_n in hl.tile(n, block_size=bn)`` adds 0 in the
padded lanes of a partial tile, its ``torch.exp`` masked there, so the fill
values those lanes load stay out of the sum after the loop.  Code generation
only; test_masking.py runs the kernels.
"""

from __future__ import annotations

import contextlib
import os
import re
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl
from helion.runtime.settings import _get_backend

pytestmark = [
    pytest.mark.cpu_only,
    skipUnlessBackends(["triton", "cute"]),
    pytest.mark.skipif(
        os.environ.get("HELION_INTERPRET") == "1",
        reason="inspects generated code; ref mode has none",
    ),
]


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_exp_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + torch.exp(x[tile_m, tile_n])
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_plus_one(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for _ in hl.tile(n, block_size=bn):
            acc = acc + 1.0
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_plus_row_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n].sum(dim=1, keepdim=True)
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_plus_branch(x: torch.Tensor, beta: float) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for _ in hl.tile(n, block_size=bn):
            if beta > 0:
                acc = acc + 1.0
            else:
                acc = acc + 2.0
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_exp_or_one(x: torch.Tensor, beta: float) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            if beta > 0:
                acc = acc + torch.exp(x[tile_m, tile_n])
            else:
                acc = acc + 1.0
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_exp_or_one_then_one(x: torch.Tensor, beta: float) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            if beta > 0:
                acc = acc + torch.exp(x[tile_m, tile_n])
            else:
                acc = acc + 1.0
            acc = acc + 1.0
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_plus_one_while(x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for _ in hl.tile(n, block_size=bn):
            limit = steps[0]
            step = torch.zeros([], device=x.device, dtype=torch.int32)
            while step < limit:
                acc = acc + 1.0
                step = step + 1
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_plus_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + b[tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_decay(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc * 0.5 + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_doubled(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = torch.add(x[tile_m, tile_n], acc, alpha=2.0)
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_gated(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc * g[tile_m, tile_n] + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_exp_sum_twice(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for _ in hl.tile(2):
            for tile_n in hl.tile(n, block_size=bn):
                acc = acc + torch.exp(x[tile_m, tile_n])
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_rescaled(x: torch.Tensor) -> torch.Tensor:
    """An online logsumexp over a per-lane accumulator."""
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        m_i = hl.full([tile_m], float("-inf"), dtype=x.dtype)
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            m_new = torch.maximum(m_i, torch.amax(v, dim=1))
            acc = acc * torch.exp(m_i - m_new)[:, None] + torch.exp(v - m_new[:, None])
            m_i = m_new
        out[tile_m] = torch.log(acc.sum(-1)) + m_i
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_halved(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = (acc + torch.exp(x[tile_m, tile_n])) / 2
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_divided(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn], 1.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc / torch.exp(x[tile_m, tile_n])
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_decay_divided(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn], 1.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = (acc * 0.5) / torch.exp(x[tile_m, tile_n])
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_divided_by_shifted(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn], 1.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc / (x[tile_m, tile_n] + 2.0)
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_dividing(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn], 1.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = torch.exp(x[tile_m, tile_n]) / acc
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_exp_sum_in_a_nested_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            for _ in hl.tile(2):
                acc = acc + torch.exp(x[tile_m, tile_n])
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_max_of_ints(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A running max of an int32 ``y`` in a carry of ``x``'s dtype."""
    m, n = y.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn], float("-inf"), dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = torch.maximum(acc, y[tile_m, tile_n])
        out[tile_m] = acc.amax(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_min_of_ints(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A running min of an int32 ``y`` in a carry of ``y``'s dtype."""
    m, n = y.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=y.dtype, device=y.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn], 2**31 - 1, dtype=y.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = torch.minimum(acc, y[tile_m, tile_n])
        out[tile_m] = acc.amin(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_plus_x_plus_one(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n] + 1.0
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_plus_one_plus_exp(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + 1.0 + torch.exp(x[tile_m, tile_n])
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_decay_plus_x_minus_one(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc * 0.5 + x[tile_m, tile_n] - 1.0
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_plus_broadcast(x: torch.Tensor) -> torch.Tensor:
    """A ``[tile_m, bn, bn]`` carry plus a load broadcast along its middle
    dim, also on the loop's block."""
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n][:, None, :]
        out[tile_m] = acc.sum(-1).sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_gated_by_broadcast(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn, bn], 1.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc * (x[tile_m, tile_n][:, None, :] + 1.0)
        out[tile_m] = acc.sum(-1).sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carry_plus_infinite_alpha(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = torch.add(acc, x[tile_m, tile_n], alpha=float("inf"))
        out[tile_m] = acc.sum(-1)
    return out


def _code(
    kernel: helion.Kernel, m: int, n: int, *extra: object, masked: bool = True
) -> str:
    """``kernel`` on an [m, n] input (an [m, n] ``g`` for ``_carry_gated``,
    then ``extra``) with a 16-wide ``bn`` and 4-row ``tile_m``;
    ``masked=False`` renders it without ``mask_loop_carries``."""
    without_pass = patch(
        "helion._compiler.device_ir.mask_loop_carries", lambda graphs: None
    )
    tensors = (torch.empty(m, n),) * (2 if kernel is _carry_gated else 1)
    # The loop around or inside the column loop takes a third block size.
    nested = (_carry_exp_sum_twice, _carry_exp_sum_in_a_nested_loop)
    block_sizes = [16, 4, 1] if kernel in nested else [16, 4]
    with (
        cpu_render_target(reduction_smem_budget_bytes=232448),
        contextlib.nullcontext() if masked else without_pass,
    ):
        bound = cpu_bind(kernel, (*tensors, *extra))
        return bound.to_code(
            bound.config_spec.normalized_config(helion.Config(block_sizes=block_sizes))
        )


def _carry_selects(code: str) -> list[str]:
    """The masks of the selects of a whole update and the carried value
    ``acc``."""
    pattern = (
        r"^\s*\w+ = tl\.where\((.*?), (?:v_|acc)\w*, acc\w*\)$"
        if _get_backend() != "cute"
        else r"^\s*\w+ = .* if (.*?) else (?:cutlass\.\w+\()?acc\w*\)?$"
    )
    return re.findall(pattern, code, re.MULTILINE)


def _sum_selects(code: str) -> list[str]:
    """The values that the selects of a whole update or sum take in the
    masked lanes."""
    pattern = (
        r"^\s*\w+ = tl\.where\(.*, v_\w+, (\w+)\)$"
        if _get_backend() != "cute"
        else r"^\s*\w+ = cutlass\.\w+\(v_\w+\) if .* else cutlass\.\w+\((\w+)\)$"
    )
    return re.findall(pattern, code, re.MULTILINE)


def _operand_selects(code: str) -> list[tuple[str, str]]:
    """The masks and fill values of the selects of an operand and a
    constant, the identity of the op that takes it."""
    pattern = (
        r"^\s*_mask_select\w* = tl\.where\((.*?), \w+, tl\.full\(\[\], (\S+), tl\.\w+\)\)$"
        if _get_backend() != "cute"
        else r"^\s*_mask_select\w* = cutlass\.\w+\(\w+\) if (.*?) else cutlass\.\w+\((\S+)\)$"
    )
    return re.findall(pattern, code, re.MULTILINE)


def test_partial_tile_masks_the_operand_that_reads_it() -> None:
    """500 columns end in a 4-column tile: the update's ``torch.exp`` of the
    tile's loads takes 0 on the tile mask, so the padded lanes add 0 to the
    carried value instead of ``exp(0)``."""
    code = _code(_carry_exp_sum, 64, 500)
    assert [fill for _, fill in _operand_selects(code)] == ["0"], code
    assert not _carry_selects(code), code


def test_updates_that_leave_padded_lanes_render_unchanged() -> None:
    """Without a partial tile the select renders as the plain operand.  An
    update that reads no load (``acc + 1.0``, also in a nested while loop),
    only a reduction over the tile (``acc + x.sum(1, keepdim=True)``) or
    lane-uniform branches is the same in every lane and reaches the padded
    ones too (see ``test_masking.py::test_loop_carry_masking``); ``acc + x``
    and ``acc + b[tile_n]`` (broadcast along the rows only) add their loads'
    0 to them, and a decay or doubling of the carry (``acc * 0.5 + x``,
    ``torch.add(x, acc, alpha=2.0)``) applies to them as it does to the live
    lanes, as does a division by ``torch.exp(x)``, 1 there (the divisor's
    identity), of the carry or of its decay: none is masked, and each renders
    exactly as without the pass."""
    steps = torch.ones(1, dtype=torch.int32)
    cases = [
        (_carry_exp_sum, 512),
        (_carry_plus_x_plus_one, 512),
        (_carry_plus_one, 500),
        (_carry_plus_one_while, 500, steps),
        (_carry_plus_row_sum, 500),
        (_carry_plus_branch, 500, 1.0),
        (_carry_sum, 500),
        (_carry_plus_bias, 500, torch.empty(500)),
        (_carry_decay, 500),
        (_carry_doubled, 500),
        (_carry_divided, 500),
        (_carry_decay_divided, 500),
    ]
    for kernel, n, *extra in cases:
        code = _code(kernel, 64, n, *extra)
        assert not _carry_selects(code) and not _operand_selects(code), code
        assert code == _code(kernel, 64, n, *extra, masked=False)


def test_operands_that_read_the_tile_take_their_ops_identity() -> None:
    """A gate (``acc * g + x``) and a divisor (``acc / (x + 2.0)``) take 1
    in the padded lanes and an ``exp`` 0, also in a loop nested in another
    one; an online logsumexp's rescale, the same in every lane, still reaches
    the padded lanes, and only its ``torch.exp(v - m_new[:, None])`` takes 0
    there."""
    for kernel, fills in (
        (_carry_gated, ["1"]),
        (_carry_divided_by_shifted, ["1"]),
        (_carry_exp_sum_twice, ["0"]),
        (_carry_rescaled, ["0"]),
    ):
        code = _code(kernel, 64, 500)
        assert [fill for _, fill in _operand_selects(code)] == fills, code
        assert not _carry_selects(code), code


def test_an_op_on_lane_uniform_values_passes_the_chain() -> None:
    """``(acc + torch.exp(x)) / 2`` divides by a value the same in every
    lane, so the halving applies to the padded lanes as ``* 0.5`` would, and
    only the ``torch.exp`` takes 0 there."""
    code = _code(_carry_halved, 64, 500)
    assert [fill for _, fill in _operand_selects(code)] == ["0"], code
    assert not _carry_selects(code), code


def test_an_operand_without_identity_selects_the_whole_update() -> None:
    """``torch.exp(x) / acc`` divides the tile's lanes by the carry, and a
    dividend has no identity to mask it to: the whole update is selected,
    keeping the carried value in the padded lanes.  So is a float carry's
    running max of int32 loads, whose identity, -inf, the int32 operand
    cannot hold, while an int32 carry's running min masks its operand to the
    int32 maximum."""
    ints = torch.empty(64, 500, dtype=torch.int32)
    for kernel, extra in ((_carry_dividing, ()), (_carry_max_of_ints, (ints,))):
        code = _code(kernel, 64, 500, *extra)
        assert _carry_selects(code), code
        assert not _operand_selects(code), code
    code = _code(_carry_min_of_ints, 64, 500, ints)
    assert [fill for _, fill in _operand_selects(code)] == ["2147483647"], code
    assert not _carry_selects(code), code


def test_a_carry_updated_in_a_loop_nested_in_its_loop_selects_the_whole_update() -> (
    None
):
    """``acc + torch.exp(x)`` in a loop nested in the column loop reaches the
    column loop's update through the inner loop's result, which reads the
    tile's lanes on every path (it was taken for an ``if`` with a path that
    reads none, and left unmasked): the whole update after the inner loop is
    selected, keeping the carried value in the padded lanes."""
    code = _code(_carry_exp_sum_in_a_nested_loop, 64, 500)
    assert _carry_selects(code), code
    assert not _operand_selects(code), code


def test_branches_are_masked_on_their_own() -> None:
    """Only one branch of a runtime ``if`` runs: in ``if beta > 0: acc = acc
    + torch.exp(x) else: acc = acc + 1.0`` the first branch masks its
    ``torch.exp``, and the second, the same in every lane, reaches the padded
    lanes (see ``test_masking.py``)."""
    code = _code(_carry_exp_or_one, 64, 500, 1.0)
    lines = code.splitlines()
    (select,) = [i for i, line in enumerate(lines) if _operand_selects(line)]
    branch = next(i for i, line in enumerate(lines) if re.match(r"\s*if ", line))
    orelse = next(i for i, line in enumerate(lines) if re.match(r"\s*else:$", line))
    assert branch < select < orelse, code


def test_an_update_after_a_branch_reading_no_lanes_is_left_as_it_is() -> None:
    """``acc = acc + 1.0`` after ``if beta > 0: acc = acc + torch.exp(x)
    else: acc = acc + 1.0`` changes the carry outside the ``if``, so only a
    select of the whole update could mask it, and that would keep the carried
    value in the padded lanes where the second branch and the line after the
    ``if`` add 1.0 to every lane (1000 instead of 1024 per row at 500
    columns): the update renders as without the pass."""
    code = _code(_carry_exp_or_one_then_one, 64, 500, 1.0)
    assert not _carry_selects(code) and not _operand_selects(code), code
    assert code == _code(_carry_exp_or_one_then_one, 64, 500, 1.0, masked=False)


def test_select_masks_only_the_loops_tiles() -> None:
    """62 rows end in a partial 4-row tile, whose padded rows the masking
    after the loop handles: with full column tiles the update renders as
    without the pass, and with a partial one its operand is masked on the
    column mask alone."""
    assert _code(_carry_exp_sum, 62, 512) == _code(
        _carry_exp_sum, 62, 512, masked=False
    )
    ((mask, _),) = _operand_selects(_code(_carry_exp_sum, 62, 500))
    assert len(set(re.findall(r"mask_\d+", mask))) == 1, mask


def test_values_added_one_after_another_count_as_one_sum() -> None:
    """``acc + x + 1.0`` adds ``x`` and then 1.0 to the carry, as ``acc + (x
    + 1.0)`` adds their sum: once one of the values reads the tile's lanes,
    none reaches the padded lanes, however the update groups them.  No mask
    of the 1.0 shows that, so the sum takes the carry there, a select of the
    whole update, as in ``acc + 1.0 + torch.exp(x)``; in ``acc * 0.5 + x -
    1.0`` it takes the decayed carry, so the decay still reaches them."""
    for kernel in (_carry_plus_x_plus_one, _carry_plus_one_plus_exp):
        code = _code(kernel, 64, 500)
        assert _carry_selects(code) and not _operand_selects(code), code
    code = _code(_carry_decay_plus_x_minus_one, 64, 500)
    (decayed,) = _sum_selects(code)
    assert re.search(rf"^\s*{decayed} = acc\w* \* v_\d+$", code, re.MULTILINE), code
    assert not _operand_selects(code), code


def test_an_infinite_alpha_selects_the_sum() -> None:
    """``torch.add(acc, x, alpha=float("inf"))`` scales the 0 that the
    padded lanes load into NaN, so the sum takes the carry there."""
    code = _code(_carry_plus_infinite_alpha, 64, 500)
    assert _carry_selects(code) and not _operand_selects(code), code


@onlyBackends(["triton"])
def test_an_operand_broadcast_along_the_loops_block() -> None:
    """``x[tile_m, tile_n][:, None, :]`` added to a ``[tile_m, bn, bn]`` carry
    is broadcast along its middle dim, also on the loop's block, where no
    select in its own shape reaches: the sum takes the carry on the masks of
    both dims, as the operand select of the load's ``expand`` would.  A gate
    broadcast so selects the whole update.  (CuTe refuses a tile with two
    dims on one block.)"""
    for kernel in (_carry_plus_broadcast, _carry_gated_by_broadcast):
        code = _code(kernel, 64, 500)
        ((mask,),) = [(mask,) for mask in _carry_selects(code)]
        assert "mask_0[None, :, None] & mask_0[None, None, :]" in mask, code
        assert not _operand_selects(code), code


@helion.kernel(static_shapes=True, autotune_effort="none")
def _carries_sharing_an_update(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn], 1.0, dtype=x.dtype)
        acc2 = hl.full([tile_m, bn], 1.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            v = (acc + (x[tile_m, tile_n] + 1.0)) * acc2
            acc = v
            acc2 = v
        out[tile_m] = acc.sum(-1) + acc2.sum(-1)
    return out


def test_carries_sharing_an_update_select_it_in_any_order() -> None:
    """``v = (acc + (x + 1.0)) * acc2`` updates both carries: masking an
    operand of ``v`` for one would change the other's update too, so each
    selects ``v`` on its own, whichever is processed first."""
    from helion._compiler import node_masking

    def statements(code: str) -> list[str]:
        """The code's lines but comments, in any order."""
        return sorted(
            line.strip() for line in code.splitlines() if "# src[" not in line
        )

    code = _code(_carries_sharing_an_update, 64, 500)
    assert len(_carry_selects(code)) == 2 and not _operand_selects(code), code
    in_order = node_masking.loop_carries
    with patch.object(
        node_masking,
        "loop_carries",
        lambda loop, graph: list(reversed(in_order(loop, graph))),
    ):
        reordered = _code(_carries_sharing_an_update, 64, 500)
    assert statements(reordered) == statements(code), reordered
