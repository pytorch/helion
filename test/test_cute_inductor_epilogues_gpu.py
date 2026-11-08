"""Runtime numerics of Inductor's pointwise GEMM epilogues (sm_100 GPU).

Each form of ``helion._testing.cute_inductor_epilogues`` (0-d FP32 constants
in either operand order, a repeated operand, boolean masks, ``where`` selects
and steps that read the carrier more than once) runs its default config
against torch on the FP32 accumulator: the
tcgen05 pair at 2048 x 2048, and the masks also on partial tiles. The CPU
side, which pins the rendered chain steps, is ``test_cute_inductor_epilogues``.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_inductor_epilogues import INDUCTOR_EPILOGUES
from helion._testing.cute_inductor_epilogues import InductorEpilogue
from helion._testing.cute_inductor_epilogues import inductor_epilogue_args
import helion.language as hl

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="tcgen05 requires an sm_100 GPU",
    ),
]


# One axis, one case per epilogue form.
@pytest.mark.parametrize(
    "epilogue", INDUCTOR_EPILOGUES, ids=[e.name for e in INDUCTOR_EPILOGUES]
)
def test_inductor_epilogue_matches_torch(epilogue: InductorEpilogue) -> None:
    # 2048 x 2048 holds more 128 x 128 tiles than SMs: the pair default.
    args = inductor_epilogue_args(epilogue, 2048, 2048, 256, device=DEVICE, random=True)
    a, b, *aux, out = args
    bound = epilogue.kernel.bind(args)
    config = bound.config_spec.default_config()
    assert config.block_sizes[:2] == [256, 256]
    bound.set_config(config)
    bound(*args)
    expected = epilogue.reference(a.float() @ b.float(), *aux).to(out.dtype)
    torch.testing.assert_close(out, expected, atol=0.25, rtol=1.6e-2)


_MASK_EPILOGUES = [e for e in INDUCTOR_EPILOGUES if "mask" in e.aux]


# Two axes, both needed: the mask forms, and the two store paths a partial
# tile takes (1000 x 1000: the flat register store; 3000 x 2048: the pair's
# hybrid TMA store with SIMT edge tiles), each loading the mask through its
# own Int8 fragment.
@pytest.mark.parametrize("shape", ((1000, 1000, 256), (3000, 2048, 256)))
@pytest.mark.parametrize(
    "epilogue", _MASK_EPILOGUES, ids=[e.name for e in _MASK_EPILOGUES]
)
def test_boolean_mask_on_partial_tiles_matches_torch(
    epilogue: InductorEpilogue, shape: tuple[int, int, int]
) -> None:
    # One element of a Boolean global tensor reads as its Int8 storage, and the
    # DSL rejects rebinding a Boolean TensorSSA across the full/edge branches.
    args = inductor_epilogue_args(epilogue, *shape, device=DEVICE, random=True)
    a, b, *aux, out = args
    bound = epilogue.kernel.bind(args)
    bound.set_config(bound.config_spec.default_config())
    bound(*args)
    expected = epilogue.reference(a.float() @ b.float(), *aux).to(out.dtype)
    torch.testing.assert_close(out, expected, atol=0.25, rtol=1.6e-2)


def test_boolean_mask_with_a_c_input_warp_matches_torch() -> None:
    """The C-input producer warp's SMEM ring would carry a mask at the one-bit
    width of ``cutlass.Boolean``; the mask stays on the GMEM read."""
    (dropout,) = [e for e in INDUCTOR_EPILOGUES if e.name == "dropout"]
    args = inductor_epilogue_args(dropout, 2048, 2048, 256, device=DEVICE, random=True)
    a, b, mask, out = args
    bound = dropout.kernel.bind(args)
    (config,) = [
        seed
        for seed in bound.config_spec.autotune_seed_configs()
        if seed.config.get("tcgen05_warp_spec_c_input_warps") == 1
    ]
    bound.set_config(config)
    bound(*args)
    expected = dropout.reference(a.float() @ b.float(), mask).to(out.dtype)
    torch.testing.assert_close(out, expected, atol=0.25, rtol=1.6e-2)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _residual_row_mask_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``(x @ w + residual) * mask[col]``: a broadcast boolean row beside an
    exact-shape operand."""
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        keep = mask[cols].to(torch.bool).to(torch.float32)
        out[rows, cols] = ((acc + residual[rows, cols].to(torch.float32)) * keep).to(
            out.dtype
        )


def test_boolean_row_mask_with_the_aux_tma_matches_torch() -> None:
    """The aux-TMA seed stages a broadcast row on partial output tiles; a
    ``cutlass.Boolean`` stage and copy atom are one bit wide against the
    mask's byte storage, so a mask row reads global memory directly."""
    a = torch.randn(3000, 256, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(256, 2048, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn(3000, 2048, device=DEVICE, dtype=torch.bfloat16)
    mask = torch.rand(2048, device=DEVICE) > 0.3
    out = torch.empty(3000, 2048, device=DEVICE, dtype=torch.bfloat16)
    args = (a, b, residual, mask, out)
    bound = _residual_row_mask_mm.bind(args)
    seeds = [
        seed
        for seed in bound.config_spec.autotune_seed_configs()
        if seed.config.get("tcgen05_aux_load_mode") == "tma"
    ]
    assert seeds
    expected = ((a.float() @ b.float() + residual.float()) * mask.float()).to(out.dtype)
    for seed in seeds:
        bound.set_config(seed)
        bound(*args)
        torch.testing.assert_close(out, expected, atol=0.25, rtol=1.6e-2)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_square_bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """``acc * acc`` of a batched GEMM: one step reading the carrier twice."""
    batches, m, k = a.shape
    n = b.shape[2]
    for bi, mi, ni in hl.tile([batches, m, n], block_size=[1, None, None]):
        acc = hl.zeros([bi, mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.baddbmm(acc, a[bi, mi, ki], b[bi, ki, ni])
        out[bi, mi, ni] = (acc * acc).to(out.dtype)


def test_carrier_read_twice_on_a_batched_flat_tile_matches_torch() -> None:
    """A step reading the carrier on both sides is not hoisted as an
    auxiliary operand (it rendered with an empty carrier name)."""
    a = torch.randn(2, 128, 64, device=DEVICE, dtype=torch.bfloat16) * 0.1
    b = torch.randn(2, 64, 128, device=DEVICE, dtype=torch.bfloat16) * 0.1
    out = torch.empty(2, 128, 128, device=DEVICE, dtype=torch.bfloat16)
    bound = _batched_square_bmm.bind((a, b, out))
    bound.set_config(helion.Config(block_sizes=[128, 128, 64], pid_type="flat"))
    bound(a, b, out)
    acc = torch.bmm(a.float(), b.float())
    torch.testing.assert_close(out, (acc * acc).to(out.dtype), atol=1e-3, rtol=1.6e-2)
