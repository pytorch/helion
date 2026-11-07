"""CPU render pins of the dense fused flash-attention backward
(``cute_flash_bwd``): the 1-CTA and 2-CTA tcgen05 bodies, plain and under
the CLC persistent tile walk.

The persistent bodies wrap each warp role's per-tile region in a
``while fbwd_work.is_valid_tile:`` loop fed by the cluster-launch-control
scheduler. These tests render ``examples/attention.py::attention_backward``
(d64, d128, grouped queries) and the causal
``benchmarks/cute/attnbwd_kernels.py::attention_bwd_causal`` on the CPU under
every (``cute_flash_bwd_persistent``, ``cute_flash_bwd_two_cta``) pair the
shape admits, check the loop-head and scheduler-warp counts, and pin the
exact text of three renders in the ``.expected_cute`` journal next to this
file so a restructuring of the body templates is checked line by line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import TestCase
from helion._testing import import_example
from helion._testing import onlyBackends
from helion._testing.cute_aux import as_sm100

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from benchmarks.cute.attnbwd_kernels import attention_bwd_causal

attention_backward = import_example("attention").attention_backward

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


PERSISTENT = "cute_flash_bwd_persistent"
TWO_CTA = "cute_flash_bwd_two_cta"
LOOP_HEAD = "fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
SCHEDULER = "fbwd_clc_pipe.producer_acquire(fbwd_clc_pstate)\n"
PLACEHOLDERS = ("{clc_setup}", "{clc_warps}", "{clc_mma_nonleader}", "FBWD_")


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        fn,  # type: ignore[arg-type]
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        autotune_accuracy_check=False,
    )


def _dense_args(
    shape: tuple[int, int, int, int], q_seq: int | None = None
) -> tuple[object, ...]:
    """``attention_backward`` arguments (K pre-scaled by the example's
    caller); ``q_seq`` longer than the KV length is the grouped-query form."""
    b, h, s, d = shape
    q_shape = (b, h, q_seq or s, d)
    q, do, o = (torch.empty(q_shape, dtype=torch.bfloat16) for _ in range(3))
    k, v = (torch.empty(shape, dtype=torch.bfloat16) for _ in range(2))
    lse, delta = (torch.empty(q_shape[:-1], dtype=torch.float32) for _ in range(2))
    return (q, k, v, o, lse, do, delta, 0.125)


def _causal_args(
    groups: int, group: int, m_dim: int, n_dim: int, head_dim: int
) -> tuple[object, ...]:
    """``attention_bwd_causal`` arguments: q / do as ``[G, group, M, D]``."""
    q, do = (
        torch.empty((groups, group, m_dim, head_dim), dtype=torch.bfloat16)
        for _ in range(2)
    )
    k, v = (
        torch.empty((groups, n_dim, head_dim), dtype=torch.bfloat16) for _ in range(2)
    )
    lse, delta = (
        torch.empty((groups, group, m_dim), dtype=torch.float32) for _ in range(2)
    )
    return (q, k, v, lse, do, delta, 0.125)


def _bind(fn: object, args: tuple[object, ...]) -> BoundKernel:
    with as_sm100():
        return _kernel(fn)._bind_isolated(args)


def _render(bound: BoundKernel, **knobs: object) -> str:
    config = helion.Config(block_sizes=[128, 128], **knobs)
    with as_sm100():
        return bound.to_code(config, output_origin_lines=False)


def _check_structure(code: str, *, persistent: bool, two_cta: bool) -> None:
    assert "'kind': 'helion_flash_bwd'" in code
    assert f"'two_cta': {two_cta}" in code
    assert f"'persistent': {persistent}" in code
    for marker in PLACEHOLDERS:
        assert marker not in code, marker
    if not persistent:
        assert LOOP_HEAD not in code
        assert "fbwd_clc_hw" not in code
        return
    assert code.count(SCHEDULER) == 1


@pytest.mark.parametrize(
    ("shape", "q_seq", "two_cta_ok"),
    [
        ((1, 2, 256, 64), None, False),
        ((1, 2, 512, 128), None, True),
        ((1, 2, 256, 64), 512, False),
    ],
    ids=["d64", "d128", "d64_gqa"],
)
@pytest.mark.parametrize(
    "knobs", [{}, {PERSISTENT: 1}, {TWO_CTA: 1}, {TWO_CTA: 1, PERSISTENT: 1}]
)
def test_dense_bodies_render(
    shape: tuple[int, int, int, int],
    q_seq: int | None,
    two_cta_ok: bool,
    knobs: dict[str, int],
) -> None:
    bound = _bind(attention_backward.fn, _dense_args(shape, q_seq))
    code = _render(bound, **knobs)
    _check_structure(
        code,
        persistent=bool(knobs.get(PERSISTENT)),
        two_cta=two_cta_ok and bool(knobs.get(TWO_CTA)),
    )


@pytest.mark.parametrize(
    "dims", [(2, 1, 512, 512, 128), (2, 2, 256, 256, 64)], ids=["d128", "d64_gqa"]
)
@pytest.mark.parametrize("knobs", [{PERSISTENT: 1}, {TWO_CTA: 1, PERSISTENT: 1}])
def test_causal_bodies_render(
    dims: tuple[int, int, int, int, int], knobs: dict[str, int]
) -> None:
    bound = _bind(attention_bwd_causal.fn, _causal_args(*dims))
    code = _render(bound, **knobs)
    # The segmented causal walk derives ``fbwd_seg`` per tile.
    assert "fbwd_seg = " in code
    _check_structure(
        code, persistent=True, two_cta=dims[-1] == 128 and bool(knobs.get(TWO_CTA))
    )


# unittest runners see neither pytest marks nor the conftest path gate.
@onlyBackends(["cute"])
class TestDenseBackwardRenderPin(TestCase):
    """The exact text of the persistent dense renders (the journal next to
    this file): the three body templates whose per-tile regions the CLC
    walk wraps."""

    def test_persistent_d64_render(self) -> None:
        bound = _bind(attention_backward.fn, _dense_args((1, 2, 256, 64)))
        code = _render(bound, **{PERSISTENT: 1})
        _check_structure(code, persistent=True, two_cta=False)
        self.assertExpectedJournal(code)

    def test_persistent_two_cta_d128_render(self) -> None:
        bound = _bind(attention_backward.fn, _dense_args((1, 2, 512, 128)))
        code = _render(bound, **{PERSISTENT: 1, TWO_CTA: 1})
        _check_structure(code, persistent=True, two_cta=True)
        self.assertExpectedJournal(code)

    def test_persistent_causal_gqa_d64_render(self) -> None:
        bound = _bind(attention_bwd_causal.fn, _causal_args(2, 2, 256, 256, 64))
        code = _render(bound, **{PERSISTENT: 1})
        _check_structure(code, persistent=True, two_cta=False)
        self.assertExpectedJournal(code)
