"""Admission, render and search-surface pins of the ``cute_register_subblocks``
family (register-resident static sub-block programs).

CPU-only (fake CUDA target), like ``test_cute_warp_mma_gemm.py`` whose fixture
this file reuses: every test binds a kernel, pins a config and inspects the
generated CuTe source, the facts or the normalized config.  Runtime coverage
lives in ``test_cute_register_subblocks_gpu.py``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import torch

from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401

import helion
from helion._compiler.cute import cute_register_subblocks as family
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends(["cute"])

KEY = family.KEY
WARPS = family.WARPS_KEY
PREFETCH = family.PREFETCH_KEY
MASK = family.MASK_KEY
COLSIDE = family.COLSIDE_KEY
ROLL = family.ROLL_KEY
RCP_LN2 = 1.4426950408889634
BHN, C, D = 2048, 64, 128


def _engine():
    from examples.linear import linear_attention_engine

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
