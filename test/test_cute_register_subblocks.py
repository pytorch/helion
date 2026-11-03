"""Admission, render and search-surface pins of the ``cute_register_subblocks``
family (register-resident static sub-block programs).

CPU-only (the fake CUDA target of the module's ``cpu_render`` marker): every
test binds a kernel, pins a config and inspects the generated CuTe source, the
facts or the normalized config.  The runtime numerics (sm_80+ GPU) are at the
end of the module.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import torch

from test.cute_population_contracts import tcgen05_mma_support
from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE

import helion
from helion._compiler.cute import cute_register_subblocks as family
from helion._testing import DEVICE
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

# The B200 model with tcgen05 MMA support, CUDA hidden, the inputs traced
# as ``cuda:0`` tensors of a 148-SM device, and the reduction SMEM budget.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(
        mma_support=tcgen05_mma_support(),
        forbid_native=False,
        fake_device_sm_count=148,
        reduction_smem_budget_bytes=232448,
    ),
]

KEY = family.KEY
WARPS = family.WARPS_KEY
PREFETCH = family.PREFETCH_KEY
MASK = family.MASK_KEY
COLSIDE = family.COLSIDE_KEY
ROLL = family.ROLL_KEY
RCP_LN2 = 1.4426950408889634
BHN, C, D = 2048, 64, 128


def _engine():
    from helion.experimental import linear_attention as linear_attention_engine

    return linear_attention_engine


def _bind(fn, args, **settings) -> BoundKernel:
    settings = {"static_shapes": True, **settings}
    return helion.kernel(fn, backend="cute", **settings).bind(args)


def _gram2_args():
    f32, bf16 = torch.float32, torch.bfloat16
    return (
        torch.empty(BHN, C, D, dtype=bf16, device=CPU_DEVICE),
        torch.empty(BHN, C, D, dtype=bf16, device=CPU_DEVICE),
        torch.empty(BHN, C, D, dtype=f32, device=CPU_DEVICE),
        torch.empty(BHN, C, C, dtype=f32, device=CPU_DEVICE),
        torch.empty(BHN, C, C, dtype=f32, device=CPU_DEVICE),
        1.0,
    )


def _bind_gram2(**settings) -> BoundKernel:
    return _bind(_engine().chunk_bwd_gram2_kda_helion.fn, _gram2_args(), **settings)


def _bind_adiag() -> BoundKernel:
    f32, bf16 = torch.float32, torch.bfloat16
    args = (
        torch.empty(BHN, C, D, dtype=bf16, device=CPU_DEVICE),
        torch.empty(BHN, C, D, dtype=bf16, device=CPU_DEVICE),
        torch.empty(BHN, C, D, dtype=f32, device=CPU_DEVICE),
        0.0883883476,
        True,
    )
    return _bind(_engine().chunk_fwd_A_diag_anchored_helion.fn, args)


def _bind_adiag_varlen(l2norm: bool = True) -> BoundKernel:
    # (the engine's varlen prefill fuses the q row L2 norm: the default)
    f32, bf16 = torch.float32, torch.bfloat16
    chunks = 32
    heads = BHN // chunks
    tokens = chunks * C
    args = (
        torch.empty(tokens, heads, D, dtype=bf16, device=CPU_DEVICE),
        torch.empty(tokens, heads, D, dtype=bf16, device=CPU_DEVICE),
        torch.empty(BHN, C, D, dtype=f32, device=CPU_DEVICE),
        torch.empty(chunks, dtype=torch.int32, device=CPU_DEVICE),
        torch.empty(chunks, dtype=torch.int32, device=CPU_DEVICE),
        torch.empty(BHN, C, C, dtype=bf16, device=CPU_DEVICE),
        torch.empty(BHN, C, C, dtype=bf16, device=CPU_DEVICE),
        0.0883883476,
        l2norm,
    )
    return _bind(_engine().chunk_fwd_A_diag_anchored_varlen_helion.fn, args)


def _program(bound: BoundKernel):
    host = bound.host_function
    with bound.env, host:
        return family.match_register_subblocks(host.device_ir.graphs)


def _config(bound: BoundKernel, **overrides: object) -> helion.Config:
    config = dict(bound.config_spec.default_config().config)
    config.update(overrides)
    return helion.Config(**config)


def _source(bound: BoundKernel, **overrides: object) -> str:
    with bound.env:
        normalized = bound.config_spec.normalized_config(_config(bound, **overrides))
    assert normalized.config.get(KEY) == family.MODE_ON, normalized
    return bound.to_code(normalized)


def _kernel_body(source: str) -> str:
    start = source.index("@cute.kernel")
    end = source.index("_helion_cute_wrapper_plans")
    return source[start:end]


def _loop_bodies(body: str) -> list[str]:
    """The text inside each ``for rsb_nt in cutlass.range(...)`` loop."""
    lines = body.splitlines()
    bodies: list[str] = []
    for i, line in enumerate(lines):
        if "for rsb_nt in cutlass.range(" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        inner: list[str] = []
        for later in lines[i + 1 :]:
            if later.strip() and len(later) - len(later.lstrip()) <= indent:
                break
            inner.append(later)
        bodies.append("\n".join(inner))
    return bodies


def _row_reductions(program) -> list[list[family.RowReduce]]:
    """The distinct row reductions under each item's stores."""
    return [
        [
            v
            for v in family._values_under([s.value for s in item.stores])
            if isinstance(v, family.RowReduce)
        ]
        for item in program.items
    ]


# ---------------------------------------------------------------------------
# admission


def test_admits_backward_gram() -> None:
    bound = _bind_gram2()
    facts = bound.config_spec.cute_register_subblocks_facts
    assert facts is not None
    assert facts.tile_choices == (16, 32, 64)
    assert facts.groups == 1
    assert facts.seed_worthy
    assert (
        facts.multi_write_windows and facts.prefetch_subject and facts.has_static_mask
    )
    program, reason = _program(bound)
    assert reason is None and program is not None
    assert program.n_mode == "permuted"
    # 6 off-diagonal pairs and 4 diagonal blocks, 4 dots of 16x16x128 each
    assert program.macs == 10 * 4 * 16 * 16 * 128
    assert program.zero_owned == frozenset({"dq", "dk", "dgc"})


def test_admits_forward_gram_as_four_groups() -> None:
    bound = _bind_adiag()
    facts = bound.config_spec.cute_register_subblocks_facts
    assert facts is not None
    assert facts.tile_choices == ()
    # the four tiers write disjoint rows: independent work items
    assert facts.groups == 4
    assert facts.seed_worthy and facts.roll_subject
    program, reason = _program(bound)
    assert reason is None and program is not None
    assert program.n_mode == "tiled"
    assert program.zero_owned == frozenset({"A", "Akk"})


@pytest.mark.parametrize("l2norm", [False, True])
def test_admits_varlen_forward_gram(l2norm: bool) -> None:
    bound = _bind_adiag_varlen(l2norm)
    facts = bound.config_spec.cute_register_subblocks_facts
    assert facts is not None
    assert facts.groups == 4 and facts.tile_choices == ()
    assert facts.seed_worthy and facts.roll_subject
    program, reason = _program(bound)
    assert reason is None and program is not None
    # the outputs are inputs of the kernel: no plain-store proof
    assert program.zero_owned == frozenset()
    assert program.n_mode == "tiled"
    # the fused q row norm is one row reduction per tier (``q * q`` is
    # ``mul(q, None)`` in the lowered IR); tiers 1-3 still merge into one
    # rolled unit
    assert [len(r) for r in _row_reductions(program)] == [int(l2norm)] * 4
    assert [run for run, _items in program.merged] == [(1, 2, 3)]
    if l2norm:
        (reduction,) = _row_reductions(program)[0]
        assert reduction.op == "sum" and reduction.shape == (16, 1)
        squared = reduction.x
        assert isinstance(squared, family.Pointwise) and squared.op == "mul"
        assert squared.args[0] is squared.args[1]


def _dynamic_window(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        for tile_r in hl.tile(16):
            out[tile_n, tile_r, tile_k] += hl.dot(
                a[tile_n, tile_r, :], b[tile_n, :, tile_k]
            )
    return out


def _half_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        out[tile_n, rows, tile_k] += hl.dot(
            a[tile_n, rows, cols], b[tile_n, rows, tile_k]
        )
    return out


def _plain_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        out[tile_n, rows, tile_k] += hl.dot(
            a[tile_n, rows, cols], b[tile_n, rows, tile_k]
        )
    return out


def _acc_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        acc = hl.zeros([tile_n, 16, tile_k], dtype=torch.float32)
        out[tile_n, rows, tile_k] = hl.dot(
            a[tile_n, rows, cols], b[tile_n, rows, tile_k], acc=acc
        )
    return out


def _int_arg_window(a: torch.Tensor, b: torch.Tensor, r0: int) -> torch.Tensor:
    # ``r0`` is a runtime input (Helion keys int arguments by type): its
    # bind-time value is not a constant of the program.
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 64, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = r0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        out[tile_n, rows, tile_k] += hl.dot(
            a[tile_n, rows, cols], b[tile_n, cols, tile_k]
        )
    return out


def _transposed_self_read(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # the second store reads its own window in the A^T layout: other
    # lanes' elements, written by the runs of the same store
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, _tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        out[tile_n, rows, cols] += hl.dot(a[tile_n, rows, cols], b[tile_n, rows, cols])
        out[tile_n, rows, cols] = (
            out[tile_n, rows, cols].transpose(-2, -1) + a[tile_n, rows, cols]
        )
    return out


def _read_then_write(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ``v`` is read before the ``+=`` and stored after it: a lazily rendered
    # load would observe the write
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    out2 = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        v = out[tile_n, rows, tile_k]
        out[tile_n, rows, tile_k] += hl.dot(
            a[tile_n, rows, cols], b[tile_n, rows, tile_k]
        )
        out2[tile_n, rows, tile_k] = v + a[tile_n, rows, tile_k]
    return out, out2


def _f16_chain(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
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


def _row_norm_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # the fused row L2 norm: ``x * x`` (a repeated operand), a sum over the
    # last dim, rsqrt, and the broadcast rescale feeding the dot
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


def _col_norm_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a reduction over the rows (dim -2): not a row reduction
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        x = a[tile_n, rows, cols]
        x = x * torch.rsqrt((x * x).sum(dim=-2, keepdim=True) + 1e-6)
        out[tile_n, rows, tile_k] = hl.dot(x, b[tile_n, rows, tile_k])
    return out


def _row_norm_nokeep(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # the reduced dim dropped and re-inserted by a view
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        x = a[tile_n, rows, cols]
        s = (x * x).sum(dim=-1)
        x = x * torch.rsqrt(s + 1e-6)[:, :, None]
        out[tile_n, rows, tile_k] = hl.dot(x, b[tile_n, rows, tile_k])
    return out


def _tile_norm_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a reduction over the channel tile: a symbolic extent
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        y = b[tile_n, rows, tile_k]
        s = (y * y).sum(dim=-1, keepdim=True)
        out[tile_n, rows, tile_k] = hl.dot(a[tile_n, rows, cols], y) * torch.rsqrt(
            s + 1e-6
        )
    return out


def _where_repeated(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ``where(m, x, x)``: the lowered IR keeps both operands of a ``where``
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        rr = hl.arange(16)
        lower = (rr[:, None] >= rr[None, :])[None, :, :]
        x = a[tile_n, rows, cols]
        xs = torch.where(lower, x, x)
        out[tile_n, rows, tile_k] = hl.dot(xs, b[tile_n, rows, tile_k])
    return out


def _small_args(dtype: torch.dtype = torch.float32, n: int = 64):
    return (
        torch.empty(n, 64, 64, dtype=dtype, device=CPU_DEVICE),
        torch.empty(n, 64, 128, dtype=dtype, device=CPU_DEVICE),
    )


def _chain_args(dtype: torch.dtype):
    return (
        torch.empty(64, 64, 64, dtype=dtype, device=CPU_DEVICE),
        torch.empty(64, 64, 64, dtype=dtype, device=CPU_DEVICE),
        torch.empty(64, 64, 128, dtype=torch.float32, device=CPU_DEVICE),
    )


@pytest.mark.parametrize(
    ("fn", "args", "settings", "needle"),
    [
        (_dynamic_window, _small_args(), {}, "device loop"),
        (_half_dot, _small_args(torch.bfloat16), {}, "fp32"),
        (_plain_dot, _small_args(), {"dot_precision": "ieee"}, "tf32"),
        (_acc_dot, _small_args(), {}, "accumulator"),
        (_int_arg_window, (*_small_args(), 16), {}, "runtime value"),
        (_plain_dot, _small_args(), {"static_shapes": False}, "must be static"),
        (_transposed_self_read, _small_args(), {}, "another layout"),
        (_read_then_write, _small_args(), {}, "used after it"),
        (_f16_chain, _chain_args(torch.float16), {}, "float16"),
        (_col_norm_dot, _small_args(), {}, "over the last dim"),
        (_row_norm_nokeep, _small_args(), {}, "keepdim=True"),
        (_row_norm_dot, _small_args(torch.bfloat16), {}, "sum must be of fp32"),
        (_tile_norm_dot, _small_args(), {}, "not the channel tile"),
    ],
)
def test_refusals(fn, args, settings, needle) -> None:
    bound = _bind(fn, args, **settings)
    assert bound.config_spec.cute_register_subblocks_facts is None
    _program_, reason = _program(bound)
    assert _program_ is None and reason is not None and needle in reason, reason
    with pytest.raises(InvalidConfig, match="available only for roots"), bound.env:
        bound.config_spec.normalized_config(_config(bound, **{KEY: family.MODE_ON}))


def _cross_layout_read_back(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ``out`` is accumulated in the C layout, then read as a dot operand (the
    # A layout) by a later item: other lanes' stores through memory
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    y = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        for i in range(2):
            c0 = i * 16
            out[tile_n, rows, tile_k] += hl.dot(
                a[tile_n, rows, c0 + cols], b[tile_n, c0 + cols, tile_k]
            )
        y[tile_n, rows, tile_k] = hl.dot(
            out[tile_n, rows, 0:16], b[tile_n, cols, tile_k]
        )
    return y


def test_cross_layout_read_back_fails_every_corner() -> None:
    bound = _bind(_cross_layout_read_back, _small_args())
    program, reason = _program(bound)
    assert program is not None and reason is None
    facts = family.register_subblock_facts(program)
    assert facts.failures and all(
        failure is not None and "read back outside the layout" in failure
        for failure in facts.failures.values()
    )
    # a program no corner renders stays off the search surface
    assert bound.config_spec.cute_register_subblocks_facts is None
    assert bound.config_spec.cute_register_subblocks is None


def _b_side_row_norm_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # each K row of the dot's B operand normalized over a static N: the sum's
    # row coordinate is the B-side K index, which differs across the four
    # lanes of a butterfly group
    n = a.size(0)
    k = b.size(2)
    out = torch.zeros([n, 16, k], dtype=torch.float32, device=a.device)
    block_k = hl.register_block_size(k)
    for tile_n, _tile_k in hl.tile([n, k], block_size=[1, block_k]):
        rows = 0 + hl.arange(16)
        cols = 0 + hl.arange(16)
        x = a[tile_n, rows, cols]
        y = b[tile_n, rows, cols]
        y = y * torch.rsqrt((y * y).sum(dim=-1, keepdim=True) + 1e-6)
        out[tile_n, rows, cols] = hl.dot(x, y)
    return out


def test_b_side_row_norm_fails_every_corner() -> None:
    # a row reduction whose row depends on the lane quarter (a B-operand K
    # row) would exchange partials of four different rows: every corner
    # declines at emit time and the program stays off the search surface
    bound = _bind(_b_side_row_norm_dot, _small_args())
    program, reason = _program(bound)
    assert program is not None and reason is None
    facts = family.register_subblock_facts(program)
    assert facts.failures and all(
        failure is not None and "lane quarter" in failure
        for failure in facts.failures.values()
    )
    assert bound.config_spec.cute_register_subblocks_facts is None
    assert bound.config_spec.cute_register_subblocks is None
    with pytest.raises(InvalidConfig, match="available only for roots"), bound.env:
        bound.config_spec.normalized_config(_config(bound, **{KEY: family.MODE_ON}))


def test_duplicate_loop_inputs_and_affine_windows_fail_closed() -> None:
    from helion.language import _tracing_ops

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    loop = graph.call_function(_tracing_ops._for_loop, (0, [0], [2], [x, x]))
    with pytest.raises(family._NotMine, match="two positions"):
        family._Matcher.loop_output_positions(loop)
    # merged units compare group-affine window bounds: no ordering, overlap
    affine = ("t", (("window", family.Affine(cgp=16), 16),))
    other = ("t", (("window", 16, 16),))
    assert family._windows_overlap(affine, other)
    assert family._windows_overlap(affine, affine)
    assert not family._windows_overlap(("u", (("window", 0, 16),)), other)


def test_small_root_is_admitted_but_not_promoted() -> None:
    bound = _bind(_plain_dot, _small_args())
    facts = bound.config_spec.cute_register_subblocks_facts
    assert facts is not None
    assert not facts.seed_worthy
    promoted = bound.config_spec.compiler_default_config
    assert promoted is None or promoted.config.get(KEY) != family.MODE_ON
    assert KEY in bound.config_spec.autotuner_heuristics


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


def test_literal_slices_are_static_windows() -> None:
    bound = _bind(_literal_slices, _small_args())
    program, reason = _program(bound)
    assert program is not None and reason is None
    (item,) = program.items
    (store,) = item.stores
    assert store.dims[1] == family.IdxWindow(0, 16)
    dot = store.value.args[1]
    assert isinstance(dot, family.Dot)
    assert dot.a.dims[2] == family.IdxWindow(16, 16)
    body = _kernel_body(_source(bound, block_sizes=[16], **{KEY: "on"}))
    assert body.count("_helion_rsb.mma_tf32(") == 4
    assert body.count("_helion_rsb.stg_v4_f32(") == 2


def _gated_gram(x: torch.Tensor, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # a gated program unlike the engine's: a static loop, a triangular
    # ``where`` mask, exp2 gating, ``+=`` into a zero-filled output
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


def _gated_args():
    return (
        torch.empty(64, 32, 128, device=CPU_DEVICE),
        torch.empty(64, 32, 128, device=CPU_DEVICE),
        torch.empty(64, 32, 16, device=CPU_DEVICE),
    )


def test_admits_gated_program() -> None:
    bound = _bind(_gated_gram, _gated_args())
    facts = bound.config_spec.cute_register_subblocks_facts
    assert facts is not None
    assert facts.tile_choices == (16, 32, 64) and facts.has_static_mask
    # the two row blocks write disjoint windows: two independent groups
    assert facts.groups == 2 and not facts.multi_write_windows
    program, reason = _program(bound)
    assert program is not None and reason is None
    assert program.zero_owned == frozenset({"out"})
    body = _kernel_body(_source(bound, block_sizes=[16], **{KEY: "on"}))
    assert body.count("_helion_rsb.mma_tf32(") == 8
    assert "sel_ge_f32" in body and "_helion_rsb.ldg_v4_f32(" not in body


# ---------------------------------------------------------------------------
# render pins


def test_backward_gram_render() -> None:
    bound = _bind_gram2()
    source = _source(
        bound, block_sizes=[16], **{KEY: "on", WARPS: 1, PREFETCH: 0, COLSIDE: "rmw"}
    )
    body = _kernel_body(source)
    assert "sync_threads" not in body and "barrier" not in body
    assert "alloc_smem" not in body
    # 10 blocks x 4 dots x 2 k-steps x 2 n-tiles
    assert body.count("_helion_rsb.mma_tf32(") == 160
    # every output window is written as 16-byte stores: 7 dq + 13 dk + 13
    # dgc block writes, two rows of four fp32 per lane each
    assert body.count("_helion_rsb.stg_v4_f32(") == 66
    # first writes of zero-filled owned windows are plain (folded zeros);
    # the 21 later writes of a window read it back in program order
    assert body.count("_helion_rsb.ldg_v4_f32(") == 42
    assert "ldg_v4_f32_ro" in body and "ldg_v2_b32_ro" in body
    # permuted N: the accumulator layout's lane columns are 2*NT*c (16 B)
    # and the B operand's NT*g (8 B) for fp32 at NT = 2
    assert "16 * rsb_c64" in body and "8 * rsb_g64" in body
    # the diagonal masks select on the lane's (row, col) coordinates
    assert "_helion_rsb.sel_ge_f32(" in body and "_helion_rsb.sel_gt_f32(" in body
    assert "block=(32, 1, 1)" in source
    assert "'total_ctas': 16384" in source  # 2048 rows x 8 channel tiles, one warp each


def test_backward_gram_prefetch_and_warps() -> None:
    bound = _bind_gram2()
    rmw = {KEY: "on", COLSIDE: "rmw", WARPS: 1}
    base = _kernel_body(_source(bound, block_sizes=[16], **rmw, **{PREFETCH: 0}))
    ahead = _kernel_body(_source(bound, block_sizes=[16], **rmw, **{PREFETCH: 1}))
    assert base != ahead
    assert base.count("mma_tf32") == ahead.count("mma_tf32")
    assert base.count("_ro(") == ahead.count("_ro(")

    def loads_before_store(body: str, index: int) -> int:
        position = -1
        for _ in range(index + 1):
            position = body.index("stg_v4_f32", position + 1)
        return body[:position].count("_ro(")

    # The first pair writes its column windows (4 stores); the row-side
    # suffix follows, and the second pair's input loads are prefetched at
    # its start, ahead of its first store.
    assert loads_before_store(ahead, 4) > loads_before_store(base, 4)
    two = _source(bound, block_sizes=[32], **{KEY: "on", WARPS: 2, COLSIDE: "rmw"})
    # (the printed launch tuple is the generic root grid; the launcher takes
    # the plan's work-item count: 2048 rows x 4 tiles / 2 warps)
    assert "block=(64, 1, 1)" in two and "'total_ctas': 4096" in two
    assert "rsb_gw = rsb_bid * 2 + rsb_warp" in two


def test_backward_gram_mask_and_colside_knobs() -> None:
    bound = _bind_gram2()
    rmw = {KEY: "on", COLSIDE: "rmw"}
    select = _kernel_body(_source(bound, block_sizes=[16], **rmw, **{MASK: "select"}))
    predicate = _kernel_body(
        _source(bound, block_sizes=[16], **rmw, **{MASK: "predicate"})
    )
    assert "ldg_f32_pred" in predicate and "ldg_f32_pred" not in select
    assert predicate.count("sel_g") < select.count("sel_g")
    regs = _kernel_body(
        _source(bound, block_sizes=[16], **{KEY: "on", COLSIDE: "regs"})
    )
    # register accumulation flushes each window once: no read-modify-write
    assert "_helion_rsb.ldg_v4_f32(" not in regs
    assert regs.count("_helion_rsb.stg_v4_f32(") == 24  # 12 windows x 2 rows


def test_forward_gram_render() -> None:
    bound = _bind_adiag()
    unrolled = _source(bound, **{KEY: "on", WARPS: 4, ROLL: "unrolled"})
    body = _kernel_body(unrolled)
    assert "alloc_smem" not in body and "sync_threads" not in body
    # four tiers of 2 dots x 16 k-steps x (2, 4, 6, 8) n-tiles, one code
    # path per tier
    assert body.count("_helion_rsb.mma_tf32(") == 640
    assert body.count("if rsb_grp <= 0:") == 1 and body.count("elif rsb_grp <= 2:") == 1
    # bf16 pairs stored as 4-byte words, 32-channel bf16 packets loaded
    assert body.count("_helion_rsb.stg_b32(") == 80
    assert "ldg_v4_b32_ro" in body
    assert "block=(128, 1, 1)" in unrolled and "'total_ctas': 2048" in unrolled
    rolled = _kernel_body(_source(bound, **{KEY: "on", WARPS: 1, ROLL: "rolled"}))
    # The four tiers are one program up to constants affine in the tier
    # (row start 16t, anchor row 16t, 16(t+1) live columns): one code path
    # whose n-tile walk is a rolled loop of 2(t+1) trips.
    assert "rsb_grp <=" not in rolled and "elif" not in rolled
    assert "for rsb_nt in cutlass.range(2 * rsb_grp + 2):" in rolled
    assert rolled.count("_helion_rsb.mma_tf32(") == 32
    assert rolled.count("_helion_rsb.stg_b32(") == 4
    # lane c's 8-channel packets interleave across the row's four lanes
    # (16-byte bf16 packets 64 bytes apart)
    assert "16 * rsb_c64" in rolled and "+ 64)" in rolled
    # the row operands are rendered ahead of the loop, the column operand
    # and the masks inside it
    loop = rolled.index("for rsb_nt in cutlass.range")
    assert rolled[:loop].count("mma_tf32") == 0 and "sel_ge_f32" in rolled[loop:]


def _double_acc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # two ``+=`` into one window inside one item: the second must read the
    # first's result back (or take it over in registers)
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
    # a static loop whose body does not depend on the trip: the two ``+=``
    # are two values of ``out`` (the dot itself is one value, shared)
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


def test_second_write_of_an_item_reads_the_first_back() -> None:
    bound = _bind(_double_acc, _small_args())
    program, reason = _program(bound)
    assert program is not None and reason is None
    assert program.zero_owned == frozenset({"out"})
    (item,) = program.items
    waves = family._store_waves(item, program.stored_names)
    assert [[index for index, _store in wave] for wave in waves] == [[0], [1]]
    rmw = _kernel_body(_source(bound, block_sizes=[16], **{KEY: "on", COLSIDE: "rmw"}))
    # the first write is plain (folded zeros); the second reads the lane's
    # two 16-byte rows back after every run of the first
    assert rmw.count("_helion_rsb.stg_v4_f32(") == 4
    assert rmw.count("_helion_rsb.ldg_v4_f32(") == 2
    assert rmw.index("_helion_rsb.ldg_v4_f32(") > rmw.index("_helion_rsb.stg_v4_f32(")
    regs = _kernel_body(
        _source(bound, block_sizes=[16], **{KEY: "on", COLSIDE: "regs"})
    )
    assert regs.count("_helion_rsb.stg_v4_f32(") == 2
    assert "_helion_rsb.ldg_v4_f32(" not in regs
    assert regs.count("_helion_rsb.mma_tf32(") == 8


def test_identical_trips_are_distinct_writes() -> None:
    bound = _bind(_repeat_body, _small_args())
    program, reason = _program(bound)
    assert program is not None and reason is None
    assert len(program.items) == 2 and program.groups == [[0, 1]]
    first, second = (item.stores[0].value for item in program.items)
    # the ``out + dot`` of each trip is its own value, the dot is shared
    assert first.key != second.key
    assert first.args[1].key == second.args[1].key
    body = _kernel_body(_source(bound, block_sizes=[16], **{KEY: "on"}))
    assert body.count("_helion_rsb.mma_tf32(") == 4
    assert body.count("_helion_rsb.stg_v4_f32(") == 4
    assert body.count("_helion_rsb.ldg_v4_f32(") == 2


def test_repeated_operands() -> None:
    # ``x * x`` is ``mul(x, None)`` in the lowered IR (``strip_unused_inputs``
    # keeps the first occurrence of an input): the matcher restores the one
    # other Node operand
    bound = _bind(_row_norm_dot, _small_args())
    program, reason = _program(bound)
    assert program is not None and reason is None
    ((reduction,),) = _row_reductions(program)
    assert isinstance(reduction.x, family.Pointwise)
    assert reduction.x.args[0] is reduction.x.args[1]
    # a ``where`` of one tensor twice keeps both operands: admitted as is
    bound = _bind(_where_repeated, _small_args())
    program, reason = _program(bound)
    assert program is not None and reason is None
    body = _kernel_body(_source(bound, block_sizes=[16], **{KEY: "on"}))
    assert body.count("_helion_rsb.mma_tf32(") == 4 and "sel_ge_f32" in body
    # the rule itself: the None of a lowered binary op is its other operand;
    # several candidates, or a None on an unlowered node, decline
    graph = torch.fx.Graph()
    m = graph.placeholder("m")
    x = graph.placeholder("x")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (x, None))
    mul.meta["lowering"] = object()
    assert family._Matcher.lowered_args(mul) == (x, x)
    where = graph.call_function(torch.ops.aten.where.self, (m, x, None))
    where.meta["lowering"] = object()
    with pytest.raises(family._NotMine, match="ambiguous repeated operand"):
        family._Matcher.lowered_args(where)
    bare = graph.call_function(torch.ops.aten.mul.Tensor, (x, None))
    with pytest.raises(family._NotMine, match="None operand"):
        family._Matcher.lowered_args(bare)
    plain = graph.call_function(torch.ops.aten.mul.Tensor, (x, m))
    assert family._Matcher.lowered_args(plain) == (x, m)


def test_row_norm_render() -> None:
    bound = _bind(_row_norm_dot, _small_args())
    facts = bound.config_spec.cute_register_subblocks_facts
    assert facts is not None and facts.tile_choices == (16, 32, 64)
    program, reason = _program(bound)
    assert program is not None and reason is None
    ((reduction,),) = _row_reductions(program)
    assert reduction.shape == (1, 16, 1) and reduction.x.shape == (1, 16, 16)
    body = _kernel_body(_source(bound, block_sizes=[16], **{KEY: "on"}))
    # two rows per lane: a lane-local sum of its four values, a butterfly
    # pair across the row's four lanes and an rsqrt for each
    assert body.count("shuffle_sync_bfly(") == 4
    assert body.count("offset=1,") == 2 and body.count("offset=2,") == 2
    assert body.count("cute.math.rsqrt(") == 2 and "fastmath" not in body
    # the norm reads the dot's A-operand registers: ``a`` is loaded once, as
    # 16-byte fp32 packets (two rows of four per lane)
    assert body.count("_helion_rsb.ldg_v4_f32_ro(") == 2
    assert body.count("_helion_rsb.mma_tf32(") == 4
    assert "sync_threads" not in body and "alloc_smem" not in body
    # (``fast_math`` lets rsqrt take the approximate path, as exp2 does)
    bound.env.settings.fast_math = True
    try:
        fast_body = _kernel_body(
            bound.to_code(_config(bound, block_sizes=[16], **{KEY: "on"}))
        )
    finally:
        bound.env.settings.fast_math = False
    assert fast_body.count("cute.math.rsqrt(") == 2
    assert fast_body.count("fastmath=True") == 2


def test_bf16_pointwise_results_are_rounded() -> None:
    bound = _bind(_bf16_chain, _chain_args(torch.bfloat16))
    program, reason = _program(bound)
    assert program is not None and reason is None
    body = _kernel_body(_source(bound, block_sizes=[16], **{KEY: "on"}))
    # 16 x 16 bf16 products per lane's A fragment: 8 values, each rounded
    assert body.count("_helion_rsb.round_bf16_f32(") == 8
    assert family._round_bf16(1.00390625) == 1.0  # ties to even
    assert family._round_bf16(1.01171875) == 1.015625
    assert family._round_bf16(float("inf")) == float("inf")


def test_varlen_forward_gram_render() -> None:
    bound = _bind_adiag_varlen()
    body = _kernel_body(_source(bound, **{KEY: "on", ROLL: "unrolled"}))
    # the token rows are predicated by the window limit (zero-filled)
    assert "ldg_v4_b32_pred" in body
    assert "ldg_v4_b32_ro" not in body
    assert body.count("_helion_rsb.mma_tf32(") == 640
    # the padded columns of the 48-wide tier are neither computed nor stored
    assert body.count("_helion_rsb.stg_b32(") == 80
    assert "rsb_bhn % 32" in body or "(rsb_bhn) % (32)" in body
    rolled = _kernel_body(_source(bound, **{KEY: "on", ROLL: "rolled"}))
    # tier 0 reads its own rows as the column operand (its own program);
    # tiers 1-3 are one program up to the tier and merge
    assert rolled.count("for rsb_nt in cutlass.range(") == 2
    assert "cutlass.range(2 * rsb_grp + 2)" in rolled
    assert rolled.count("_helion_rsb.mma_tf32(") == 64
    assert rolled.count("if rsb_grp <= 0:") == 1 and rolled.count("else:") == 1


def test_varlen_forward_gram_row_norm_render() -> None:
    with_norm = _bind_adiag_varlen(True)
    without = _bind_adiag_varlen(False)
    for roll, units in (("unrolled", 4), ("rolled", 2)):
        knobs = {KEY: "on", WARPS: 4, ROLL: roll}
        norm = _kernel_body(_source(with_norm, **knobs))
        plain = _kernel_body(_source(without, **knobs))
        # per unit, two rows per lane: a butterfly pair (offsets 1 and 2
        # across the row's four lanes) and an rsqrt for each
        assert norm.count("shuffle_sync_bfly(") == 4 * units
        assert norm.count("offset=1,") == 2 * units
        assert norm.count("offset=2,") == 2 * units
        assert norm.count("cute.math.rsqrt(") == 2 * units
        assert "fastmath" not in norm
        # q is loaded once: the norm reads the dot's A-operand registers
        for needle in (
            "_helion_rsb.ldg_v4_b32_pred(",
            "_helion_rsb.mma_tf32(",
            "_helion_rsb.stg_b32(",
        ):
            assert norm.count(needle) == plain.count(needle), needle
        assert "sync_threads" not in norm and "alloc_smem" not in norm
        if roll == "rolled":
            # loop-invariant: hoisted ahead of every rolled n-tile walk
            bodies = _loop_bodies(norm)
            assert len(bodies) == 2
            assert not any(
                "shuffle_sync_bfly" in loop or "rsqrt" in loop for loop in bodies
            )
    # the no-norm program renders as before the norm was admitted
    assert plain.count("_helion_rsb.mma_tf32(") == 64


# ---------------------------------------------------------------------------
# search surface


def test_knob_normalization() -> None:
    bound = _bind_gram2()
    spec = bound.config_spec
    with bound.env:
        on = spec.normalized_config(
            _config(bound, block_sizes=[16], **{KEY: "on", WARPS: 2, PREFETCH: 0})
        )
        assert on.config[WARPS] == 2 and on.config[PREFETCH] == 0
        assert "cute_fp32_dot" not in on.config  # inert fragment knobs are pinned
        # the promoted default carries the family and its first seed
        assert spec.default_config().config.get(KEY) == "on"
        off = spec.normalized_config(
            _config(bound, block_sizes=[16], **{KEY: "off", WARPS: 4, COLSIDE: "regs"})
        )
        assert off.config[KEY] == "off"
        # the knobs are inert without the mode and pinned to their defaults
        assert off.config[WARPS] == 1 and off.config[COLSIDE] == "rmw"
        with pytest.raises(InvalidConfig, match="channel tile"):
            spec.normalized_config(_config(bound, block_sizes=[128], **{KEY: "on"}))
        with pytest.raises(InvalidConfig, match=WARPS):
            spec.normalized_config(
                _config(bound, block_sizes=[16], **{KEY: "on", WARPS: 3})
            )
        repaired = spec.flat_config(lambda fragment: fragment.default())
        assert repaired is not None
    flat = spec.flat_fields() if hasattr(spec, "flat_fields") else None
    if flat is not None:
        assert KEY in flat and WARPS in flat and PREFETCH in flat


def test_seeds_and_promotion() -> None:
    bound = _bind_gram2()
    spec = bound.config_spec
    promoted = spec.compiler_default_config
    assert promoted is not None and promoted.config.get(KEY) == family.MODE_ON
    assert promoted.config["block_sizes"][0] in family.TILE_CHOICES
    assert KEY in spec.autotuner_heuristics
    facts = spec.cute_register_subblocks_facts
    assert facts is not None
    seeds = family.register_subblocks_seed_knobs(facts)
    assert seeds and all(
        family.register_subblocks_config_reason(
            facts,
            bk=s["bk"],
            warps=s[WARPS],
            prefetch=s[PREFETCH],
            mask=s[MASK],
            colside=s[COLSIDE],
        )
        is None
        for s in seeds
    )


def test_forward_gram_knobs() -> None:
    bound = _bind_adiag()
    spec = bound.config_spec
    with bound.env:
        four = spec.normalized_config(_config(bound, **{KEY: "on", WARPS: 4}))
        assert four.config[WARPS] == 4
        # no window is written twice: the register accumulation knob is inert
        assert four.config[COLSIDE] == "rmw"
        with pytest.raises(InvalidConfig):
            spec.normalized_config(_config(bound, **{KEY: "on", COLSIDE: "regs"}))
    promoted = spec.compiler_default_config
    assert promoted is not None and promoted.config.get(KEY) == family.MODE_ON


def test_affine_render() -> None:
    assert family.Affine(cg=1, k=8).render("g", "c") == "(g + 8)"
    assert family.Affine(cc=4, k=1).render("g", "c") == "(4 * c + 1)"
    assert family.Affine().render("g", "c") == "0"
    assert re.fullmatch(r"\(2 \* g \+ 3\)", family.Affine(cg=2, k=3).render("g", "c"))


# Runtime numerics of the ``cute_register_subblocks`` family (sm_80+ GPU).
#
# The engine's anchored gram kernels and a few isolating kernels run under the
# family and are checked against float64 references at tf32 tolerance; the
# family's knobs (warps per CTA, mask method, column-side accumulation,
# prefetch) keep the mma order, so their outputs are bitwise identical; the
# varlen forward gram is checked ragged against the ordinary lowering.


# tf32 operands: ~1e-3 relative error against float64
TF32_REL = 3e-3


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
def test_backward_gram_matches_fragment_route_closely() -> None:
    # Both routes run tf32 ``mma.sync``; only the summation order differs.
    inputs = _gram2_inputs(bhn=32)
    args = (*inputs, 1.0)
    fn = _engine().chunk_bwd_gram2_kda_helion.fn
    ours = _run(fn, args, bk=16)
    theirs = _run(fn, args, family_on=False, bk=16, cute_fp32_dot="tf32_mma")
    for lhs, rhs in zip(ours, theirs, strict=True):
        assert _rel(lhs, rhs.double()) < 1e-5


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
@pytest.mark.parametrize("colside", ["rmw", "regs"])
def test_second_write_of_an_item_reads_the_first_back_on_device(colside: str) -> None:
    torch.manual_seed(3)
    a = torch.randn(64, 64, 64, device=DEVICE)
    b = torch.randn(64, 64, 128, device=DEVICE)
    out = _run(_double_acc, (a, b), bk=16, **{COLSIDE: colside})
    a64, b64 = a.double(), b.double()
    ref = a64[:, :16, :16] @ b64[:, :16] + a64[:, :16, :16] @ b64[:, 16:32]
    assert _rel(out, ref) < TF32_REL


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
def test_identical_trips_both_accumulate() -> None:
    torch.manual_seed(4)
    a = torch.randn(64, 64, 64, device=DEVICE)
    b = torch.randn(64, 64, 128, device=DEVICE)
    out = _run(_repeat_body, (a, b), bk=16)
    ref = 2 * (a.double()[:, :16, :16] @ b.double()[:, :16])
    assert _rel(out, ref) < TF32_REL


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the register sub-block family runs mma.sync tf32 (sm_80+)",
)
@pytest.mark.cuda
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
