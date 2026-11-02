"""Runtime numerics of the ``cute_register_subblocks`` family (sm_80+ GPU).

The engine's anchored gram kernels and a few isolating kernels run under the
family and are checked against float64 references at tf32 tolerance; the
family's knobs (warps per CTA, mask method, column-side accumulation,
prefetch) keep the mma order, so their outputs are bitwise identical; the
varlen forward gram is checked ragged against the ordinary lowering.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute import cute_register_subblocks as family
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
        reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
    ),
]

KEY = family.KEY
WARPS = family.WARPS_KEY
PREFETCH = family.PREFETCH_KEY
MASK = family.MASK_KEY
COLSIDE = family.COLSIDE_KEY
ROLL = family.ROLL_KEY
RCP_LN2 = 1.4426950408889634
# tf32 operands: ~1e-3 relative error against float64
TF32_REL = 3e-3


def _engine():
    from examples.linear import linear_attention_engine

    return linear_attention_engine


def _run(fn, args, *, family_on: bool = True, **knobs: object):
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    spec = bound.config_spec
    config = dict(spec.default_config().config)
    config[KEY] = family.MODE_ON if family_on else family.MODE_OFF
    bk = knobs.pop("bk", None)
    if bk is not None:
        config["block_sizes"] = [bk]
    config.update(knobs)
    with bound.env:
        normalized = spec.normalized_config(helion.Config(**config))
    assert normalized.config.get(KEY) == (
        family.MODE_ON if family_on else family.MODE_OFF
    )
    source = bound.to_code(normalized)
    if family_on:
        assert "_helion_rsb.mma_tf32" in source
        assert "alloc_smem" not in source[source.index("@cute.kernel") :]
    bound.set_config(normalized)
    out = bound(*args)
    torch.cuda.synchronize()
    return out


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.double() - ref).norm() / ref.norm()).item()


def _gram2_inputs(
    bhn: int = 64, c: int = 64, d: int = 128, decay: float = 0.1, seed: int = 0
):
    torch.manual_seed(seed)
    f32 = torch.float32
    q = (torch.rand(bhn, c, d, device=DEVICE, dtype=f32) * 2 - 1).to(torch.bfloat16)
    k = (torch.rand(bhn, c, d, device=DEVICE, dtype=f32) * 2 - 1).to(torch.bfloat16)
    gc = (
        (-torch.rand(bhn, c, d, device=DEVICE, dtype=f32) * decay)
        .cumsum(1)
        .contiguous()
    )
    dAqk = torch.rand(bhn, c, c, device=DEVICE, dtype=f32) * 2 - 1
    dAkk = torch.rand(bhn, c, c, device=DEVICE, dtype=f32) * 2 - 1
    return q, k, gc, dAqk, dAkk


def _gram2_reference(q, k, gc, dAqk, dAkk, scale):
    """The docstring's math of ``chunk_bwd_gram2_kda_helion`` in float64."""
    bhn, c, _ = q.shape
    bc = 16
    q = q.double() * scale
    k = k.double()
    gc = gc.double()
    dAqk = dAqk.double()
    dAkk = dAkk.double()
    dq = torch.zeros_like(gc)
    dk = torch.zeros_like(gc)
    dgc = torch.zeros_like(gc)
    rows = torch.arange(bc, device=q.device)
    incl = (rows[:, None] >= rows[None, :]).double()
    strict = (rows[:, None] > rows[None, :]).double()
    for i in range(c // bc):
        r0 = i * bc
        for j in range(i + 1):
            c0 = j * bc
            diag = i == j
            gn = gc[:, r0 + (bc // 2 if diag else 0), :]
            erow = torch.exp2((gc[:, r0 : r0 + bc] - gn[:, None]) * RCP_LN2)
            ecol = torch.exp2((gn[:, None] - gc[:, c0 : c0 + bc]) * RCP_LN2)
            qg = q[:, r0 : r0 + bc] * erow
            kgr = k[:, r0 : r0 + bc] * erow
            kg = k[:, c0 : c0 + bc] * ecol
            aq = dAqk[:, r0 : r0 + bc, c0 : c0 + bc]
            ak = dAkk[:, r0 : r0 + bc, c0 : c0 + bc]
            if diag:
                aq = aq * incl
                ak = ak * strict
            d_qg = aq @ kg
            d_kgr = ak @ kg
            d_kg = aq.transpose(-2, -1) @ qg + ak.transpose(-2, -1) @ kgr
            dq[:, r0 : r0 + bc] += d_qg * erow
            dk[:, r0 : r0 + bc] += d_kgr * erow
            dgc[:, r0 : r0 + bc] += d_qg * qg + d_kgr * kgr
            dk[:, c0 : c0 + bc] += d_kg * ecol
            dgc[:, c0 : c0 + bc] += -d_kg * kg
    return dq, dk, dgc


def _adiag_reference(q, k, gc, scale):
    bhn, c, _ = q.shape
    bc = 16
    q = q.double()
    k = k.double()
    gc = gc.double()
    a = torch.zeros(bhn, c, c, dtype=torch.float64, device=q.device)
    akk = torch.zeros_like(a)
    rows = torch.arange(bc, device=q.device)
    for t in range(c // bc):
        r0 = t * bc
        ncol = bc * (t + 1)
        cols = torch.arange(ncol, device=q.device)
        gn = gc[:, r0, :]
        e = torch.exp2((gc[:, r0 : r0 + bc] - gn[:, None]) * RCP_LN2)
        kgt = (
            k[:, :ncol] * torch.exp2((gn[:, None] - gc[:, :ncol]) * RCP_LN2)
        ).transpose(-2, -1)
        causal = ((r0 + rows)[:, None] >= cols[None, :]).double()
        strict = ((r0 + rows)[:, None] > cols[None, :]).double()
        a[:, r0 : r0 + bc, :ncol] = ((q[:, r0 : r0 + bc] * e) @ kgt) * scale * causal
        akk[:, r0 : r0 + bc, :ncol] = ((k[:, r0 : r0 + bc] * e) @ kgt) * strict
    return a, akk


@pytest.mark.parametrize("bk", [16, 32])
def test_backward_gram_matches_float64(bk: int) -> None:
    inputs = _gram2_inputs()
    scale = 1.0
    # (the 32-channel tile holds its register accumulation only in rmw mode)
    outs = _run(
        _engine().chunk_bwd_gram2_kda_helion.fn,
        (*inputs, scale),
        bk=bk,
        **{COLSIDE: "rmw"},
    )
    refs = _gram2_reference(*inputs, scale)
    for out, ref in zip(outs, refs, strict=True):
        assert _rel(out, ref) < TF32_REL


_BITWISE_BASE: dict[str, tuple[tuple, tuple]] = {}


def _bitwise_base() -> tuple[tuple, tuple]:
    """The backward gram at the base knobs (compiled once per process)."""
    if "base" not in _BITWISE_BASE:
        args = (*_gram2_inputs(bhn=32), 1.0)
        base = _run(
            _engine().chunk_bwd_gram2_kda_helion.fn,
            args,
            bk=16,
            **{WARPS: 1, PREFETCH: 0, MASK: "select", COLSIDE: "rmw"},
        )
        _BITWISE_BASE["base"] = (args, base)
    return _BITWISE_BASE["base"]


@pytest.mark.parametrize(
    "knobs",
    [
        {WARPS: 2},
        {WARPS: 4},
        {MASK: "predicate"},
        {COLSIDE: "regs"},
        {COLSIDE: "regs", WARPS: 4},
    ],
)
def test_backward_gram_knobs_are_bitwise_identical(knobs: dict[str, object]) -> None:
    args, base = _bitwise_base()
    other = _run(
        _engine().chunk_bwd_gram2_kda_helion.fn, args, bk=16, **{PREFETCH: 0, **knobs}
    )
    for lhs, rhs in zip(base, other, strict=True):
        assert torch.equal(lhs, rhs), knobs


@pytest.mark.parametrize(
    "knobs", [{PREFETCH: 1}, {PREFETCH: 1, COLSIDE: "regs", WARPS: 2}]
)
def test_backward_gram_prefetch_within_one_ulp(knobs: dict[str, object]) -> None:
    # The prefetched order keeps the mma order but lets the compiler contract
    # a different multiply-add pair of the epilogue: one ulp in dgc.
    args, base = _bitwise_base()
    other = _run(_engine().chunk_bwd_gram2_kda_helion.fn, args, bk=16, **knobs)
    for lhs, rhs in zip(base, other, strict=True):
        assert (lhs - rhs).abs().max().item() <= 1e-6, knobs


def test_backward_gram_matches_fragment_route_closely() -> None:
    # Both routes run tf32 ``mma.sync``; only the summation order differs.
    inputs = _gram2_inputs(bhn=32)
    args = (*inputs, 1.0)
    fn = _engine().chunk_bwd_gram2_kda_helion.fn
    ours = _run(fn, args, bk=16)
    theirs = _run(fn, args, family_on=False, bk=16, cute_fp32_dot="tf32_mma")
    for lhs, rhs in zip(ours, theirs, strict=True):
        assert _rel(lhs, rhs.double()) < 1e-5


@pytest.mark.parametrize(
    ("warps", "roll"), [(1, "unrolled"), (4, "unrolled"), (1, "rolled"), (4, "rolled")]
)
def test_forward_gram_matches_float64(warps: int, roll: str) -> None:
    q, k, gc, _, _ = _gram2_inputs()
    scale = 0.0883883476
    a, akk = _run(
        _engine().chunk_fwd_A_diag_anchored_helion.fn,
        (q, k, gc, scale, True),
        **{WARPS: warps, ROLL: roll},
    )
    ref_a, ref_akk = _adiag_reference(q, k, gc, scale)
    # bf16 outputs: the rounding of the stored values dominates
    assert _rel(a, ref_a) < 1e-2 and _rel(akk, ref_akk) < 1e-2
    # the columns past each tier's live width stay zero
    assert torch.all(a[:, :16, 16:] == 0) and torch.all(akk[:, :16, 16:] == 0)


_FORWARD_BASE: dict[str, tuple[tuple, tuple]] = {}


def _forward_base() -> tuple[tuple, tuple]:
    if "base" not in _FORWARD_BASE:
        q, k, gc, _, _ = _gram2_inputs(bhn=32)
        args = (q, k, gc, 0.5, True)
        fn = _engine().chunk_fwd_A_diag_anchored_helion.fn
        _FORWARD_BASE["base"] = (
            args,
            _run(fn, args, **{WARPS: 1, ROLL: "unrolled"}),
        )
    return _FORWARD_BASE["base"]


@pytest.mark.parametrize(
    "knobs",
    [
        {WARPS: 4, ROLL: "unrolled"},
        {WARPS: 1, ROLL: "rolled"},
        {WARPS: 4, ROLL: "rolled"},
    ],
)
def test_forward_gram_warps_are_bitwise_identical(knobs: dict[str, object]) -> None:
    args, one = _forward_base()
    other = _run(_engine().chunk_fwd_A_diag_anchored_helion.fn, args, **knobs)
    for lhs, rhs in zip(one, other, strict=True):
        assert torch.equal(lhs, rhs), knobs


@pytest.mark.parametrize("l2norm", [False, True])
def test_varlen_forward_gram_ragged(l2norm: bool) -> None:
    torch.manual_seed(1)
    chunks, heads, c, d = 6, 2, 64, 128
    tokens = chunks * c
    f32 = torch.float32
    q = (torch.rand(tokens, heads, d, device=DEVICE, dtype=f32) * 2 - 1).to(
        torch.bfloat16
    )
    k = (torch.rand(tokens, heads, d, device=DEVICE, dtype=f32) * 2 - 1).to(
        torch.bfloat16
    )
    gc = (
        (-torch.rand(heads * chunks, c, d, device=DEVICE, dtype=f32) * 0.1)
        .cumsum(1)
        .contiguous()
    )
    token_base = (torch.arange(chunks, device=DEVICE) * c).to(torch.int32)
    valid_len = torch.tensor([64, 17, 40, 64, 5, 63], device=DEVICE, dtype=torch.int32)
    fn = _engine().chunk_fwd_A_diag_anchored_varlen_helion.fn

    def run(family_on: bool, **knobs: object):
        a = torch.zeros(heads * chunks, c, c, dtype=torch.bfloat16, device=DEVICE)
        akk = torch.zeros_like(a)
        _run(
            fn,
            (q, k, gc, token_base, valid_len, a, akk, 0.125, l2norm),
            family_on=family_on,
            **knobs,
        )
        return a, akk

    ours = run(True, **{ROLL: "unrolled"})
    rolled = run(True, **{ROLL: "rolled"})
    theirs = run(False, cute_fp32_dot="tf32_mma")
    for lhs, rhs in zip(ours, rolled, strict=True):
        assert torch.equal(lhs, rhs)
    for lhs, rhs in zip(ours, theirs, strict=True):
        assert _rel(lhs, rhs.double()) < 2e-3
    # rows past a sequence's end are zero
    a, _ = ours
    assert torch.all(a[4, 5:, :] == 0)  # chunk 4 of head 0 has 5 valid rows
    if l2norm:
        # the fused q row norm: against the kernel's math in float64 on the
        # tokens of a ragged chunk (chunk 1 of head 1: 17 valid rows)
        r, j, valid = 1 * chunks + 1, 1, 17
        rows = torch.arange(c, device=DEVICE)
        qf = q[token_base[j] + rows, 1].double() * (rows < valid)[:, None]
        kf = k[token_base[j] + rows, 1].double() * (rows < valid)[:, None]
        qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
        g = gc[r].double()
        ref = torch.zeros(c, c, dtype=torch.float64, device=DEVICE)
        for i in range(c // 16):
            blk, cols = slice(16 * i, 16 * i + 16), slice(0, 16 * (i + 1))
            n = g[16 * i : 16 * i + 1]
            e = torch.exp2((g[blk] - n) * RCP_LN2)
            ec = torch.exp2((n - g[cols]) * RCP_LN2)
            tri = rows[blk, None] >= rows[None, cols]
            ref[blk, cols] = (qf[blk] * e) @ (kf[cols] * ec).T * 0.125 * tri
        assert _rel(a[r], ref) < 2e-3


def _row_norm_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # the fused row L2 norm (``x * x`` is a repeated operand of the lowered
    # IR; the sum is a lane-local partial + butterfly exchange)
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        x = a[tile_n, rows, cols]
        x = x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)
        out[tile_n, rows, tile_k] = hl.dot(x, b[tile_n, rows, tile_k])
    return out


@pytest.mark.parametrize("bk", [16, 64])
def test_row_norm_matches_float64(bk: int) -> None:
    torch.manual_seed(3)
    a = torch.randn(64, 64, 64, device=DEVICE) * 3
    b = torch.randn(64, 64, 128, device=DEVICE)
    out = _run(_row_norm_dot, (a, b), bk=bk)
    x = a[:, :16, :16].double()
    x = x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    ref = x @ b[:, :16, :].double()
    assert _rel(out[:, :16, :], ref) < TF32_REL
    assert torch.all(out[:, 16:, :] == 0)
    theirs = _run(_row_norm_dot, (a, b), family_on=False, cute_fp32_dot="tf32_mma")
    assert _rel(out, theirs.double()) < 2e-3


def _loop_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 64, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        for i in range(1, 4):
            r0 = i * 16
            acc = hl.zeros([tile_n, 16, tile_k], dtype=torch.float32)
            for j in range(i):
                c0 = j * 16
                blk = a[tile_n, r0 : r0 + 16, c0 : c0 + 16]
                acc = acc + hl.dot(blk, b[tile_n, c0 : c0 + 16, tile_k])
                out[tile_n, c0 : c0 + 16, tile_k] += hl.dot(
                    blk.transpose(-2, -1), b[tile_n, r0 : r0 + 16, tile_k]
                )
            out[tile_n, r0 : r0 + 16, tile_k] += acc
    return out


def test_static_loops_and_rmw() -> None:
    torch.manual_seed(2)
    a = torch.randn(32, 64, 64, device=DEVICE)
    b = torch.randn(32, 64, 128, device=DEVICE)
    out = _run(_loop_kernel, (a, b), bk=32)
    a64, b64 = a.double(), b.double()
    ref = torch.zeros(32, 64, 128, dtype=torch.float64, device=DEVICE)
    for i in range(1, 4):
        r0 = i * 16
        acc = torch.zeros(32, 16, 128, dtype=torch.float64, device=DEVICE)
        for j in range(i):
            c0 = j * 16
            blk = a64[:, r0 : r0 + 16, c0 : c0 + 16]
            acc = acc + blk @ b64[:, c0 : c0 + 16]
            ref[:, c0 : c0 + 16] += blk.transpose(-2, -1) @ b64[:, r0 : r0 + 16]
        ref[:, r0 : r0 + 16] += acc
    assert _rel(out, ref) < TF32_REL


def _double_acc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        blk = a[tile_n, rows, cols]
        out[tile_n, rows, tile_k] += hl.dot(blk, b[tile_n, rows, tile_k])
        out[tile_n, rows, tile_k] += hl.dot(blk, b[tile_n, 16 + hl.arange(16), tile_k])
    return out


def _repeat_body(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        for _i in range(2):
            out[tile_n, rows, tile_k] += hl.dot(
                a[tile_n, rows, cols], b[tile_n, rows, tile_k]
            )
    return out


def _literal_slices(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        out[tile_n, 0:16, tile_k] += hl.dot(
            a[tile_n, 0:16, 16:32], b[tile_n, 0:16, tile_k]
        )
    return out


def _wide_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a dot of two m-tiles (M = 32)
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 32, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(32)
        cols = 0 + hl.arange(16)
        out[tile_n, rows, tile_k] += hl.dot(
            a[tile_n, rows, cols], b[tile_n, cols, tile_k]
        )
    return out


@pytest.mark.parametrize("colside", ["rmw", "regs"])
def test_second_write_of_an_item_reads_the_first_back(colside: str) -> None:
    torch.manual_seed(3)
    a = torch.randn(64, 64, 64, device=DEVICE)
    b = torch.randn(64, 64, 128, device=DEVICE)
    out = _run(_double_acc, (a, b), bk=16, **{COLSIDE: colside})
    a64, b64 = a.double(), b.double()
    ref = a64[:, :16, :16] @ b64[:, :16] + a64[:, :16, :16] @ b64[:, 16:32]
    assert _rel(out, ref) < TF32_REL


def test_identical_trips_both_accumulate() -> None:
    torch.manual_seed(4)
    a = torch.randn(64, 64, 64, device=DEVICE)
    b = torch.randn(64, 64, 128, device=DEVICE)
    out = _run(_repeat_body, (a, b), bk=16)
    ref = 2 * (a.double()[:, :16, :16] @ b.double()[:, :16])
    assert _rel(out, ref) < TF32_REL


@pytest.mark.parametrize(("fn", "rows"), [(_literal_slices, 16), (_wide_rows, 32)])
def test_literal_slices_and_two_m_tiles(fn, rows: int) -> None:
    torch.manual_seed(5)
    a = torch.randn(64, 64, 64, device=DEVICE)
    b = torch.randn(64, 64, 128, device=DEVICE)
    out = _run(fn, (a, b), bk=16)
    a64, b64 = a.double(), b.double()
    if fn is _literal_slices:
        ref = a64[:, :16, 16:32] @ b64[:, :16]
    else:
        ref = a64[:, :rows, :16] @ b64[:, :16]
    assert _rel(out, ref) < TF32_REL


def _bf16_chain(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    k = c.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        prod = (a[tile_n, rows, cols] * b[tile_n, rows, cols]).to(torch.float32)
        out[tile_n, rows, tile_k] = hl.dot(prod, c[tile_n, rows, tile_k])
    return out


def test_bf16_pointwise_results_round_like_torch() -> None:
    torch.manual_seed(6)
    a = (torch.rand(64, 64, 64, device=DEVICE) * 2 - 1).to(torch.bfloat16)
    b = (torch.rand(64, 64, 64, device=DEVICE) * 2 - 1).to(torch.bfloat16)
    c = torch.randn(64, 64, 128, device=DEVICE)
    # torch rounds the bf16 product to bf16 before the dot
    rounded = (a[:, :16, :16] * b[:, :16, :16]).double() @ c.double()[:, :16]
    unrounded = (a[:, :16, :16].double() * b[:, :16, :16].double()) @ c.double()[:, :16]
    ours = _run(_bf16_chain, (a, b, c), bk=16)
    theirs = _run(
        _bf16_chain, (a, b, c), family_on=False, bk=16, cute_fp32_dot="tf32_mma"
    )
    assert _rel(ours, rounded) < TF32_REL
    assert _rel(ours, rounded) < _rel(ours, unrounded)
    # the fragment route rounds the same way; only the summation order differs
    assert _rel(ours, theirs.double()) < 1e-5


def _gated_gram(x: torch.Tensor, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    k = x.size(2)
    out = torch.zeros([n, 32, k], dtype=torch.float32, device=x.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rr = hl.arange(16)
        lower = (rr[:, None] >= rr[None, :])[None, :, :]
        for i in range(2):
            r0 = i * 16
            gn = g[tile_n, r0, tile_k]
            e = torch.exp2((g[tile_n, r0 : r0 + 16, tile_k] - gn[:, None, :]) * RCP_LN2)
            sa = torch.where(lower, s[tile_n, r0 : r0 + 16, rr], 0.0)
            out[tile_n, r0 : r0 + 16, tile_k] += (
                hl.dot(sa, x[tile_n, r0 : r0 + 16, tile_k] * e) * e
            )
    return out


@pytest.mark.parametrize(
    "knobs",
    [{COLSIDE: "rmw"}, {MASK: "predicate"}, {WARPS: 2, PREFETCH: 1}],
)
def test_gated_program_matches_float64(knobs: dict[str, object]) -> None:
    torch.manual_seed(7)
    x = torch.randn(64, 32, 128, device=DEVICE)
    g = (-torch.rand(64, 32, 128, device=DEVICE) * 0.1).cumsum(1).contiguous()
    s = torch.randn(64, 32, 16, device=DEVICE)
    out = _run(_gated_gram, (x, g, s), bk=16, **knobs)
    x64, g64, s64 = x.double(), g.double(), s.double()
    ref = torch.zeros(64, 32, 128, dtype=torch.float64, device=DEVICE)
    rows = torch.arange(16, device=DEVICE)
    lower = (rows[:, None] >= rows[None, :]).double()
    for i in range(2):
        r0 = i * 16
        e = torch.exp2((g64[:, r0 : r0 + 16] - g64[:, r0][:, None, :]) * RCP_LN2)
        ref[:, r0 : r0 + 16] += (
            (s64[:, r0 : r0 + 16] * lower) @ (x64[:, r0 : r0 + 16] * e)
        ) * e
    assert _rel(out, ref) < TF32_REL
