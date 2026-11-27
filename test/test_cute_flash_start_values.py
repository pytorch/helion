"""The fused flash bodies start every row from statistics of their own,
whatever the kernel wrote, so a kernel is fused only when its start gives the
same result.

Forward (``attention_match._attention_loop_starts_fused``): the bodies start
the online softmax from ``(m, l, acc) = (-inf, 0, 0)``. A running max of
``-inf`` reproduces that with any finite constant row sum and accumulator,
since it makes the first KV step wipe both. So does a finite sentinel far
enough below every score (``-1e30``; ``_running_max_sentinel_bound``) on
each row that sees a score above it, and it takes a zero accumulator and a
row sum that is not zero, which a row that sees no key keeps (``KNOWN_GAPS.md``,
cute-flash-sentinel-start-taken-for-minus-inf, has the rows it does not
cover). Backward (``cute_flash_bwd._match_bwd_root_graph``): the body
accumulates dV and dK from zero.

The two kernels here, bound on the CPU, take their starts as arguments:
compile-time constants (``hl.constexpr``) trace to the literal fills a kernel
spells, a plain float to a fill by a runtime scalar. The ``hl.zeros(...) -
float("inf")`` spelling is ``examples/blackwell_attention.py``'s, fused in
test_cute_flattened_attention_match.py and test_cute_flash_render_golden.py;
the FlexAttention envelope's starts are in test_cute_flex_match.py.
"""

from __future__ import annotations

import math

import pytest
import torch

import helion
from helion._compiler.cute.analysis.attention_match import _running_max_sentinel_bound
from helion._compiler.cute.analysis.match_common import constant_fill
from helion._testing.cute_aux import cpu_codegen
from helion._testing.cute_aux import flash_env
import helion.language as hl

pytestmark = pytest.mark.cpu_only

pytest.importorskip("cutlass")


@helion.kernel(backend="cute", static_shapes=True)
def _attention_from(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    m_start: float,
    l_start: float,
    acc_start: float,
) -> torch.Tensor:
    """``examples/attention.py``'s forward (the output alone) with its three
    statistics started at the arguments."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = 1.0 / math.sqrt(head_dim) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], m_start, dtype=torch.float32)
        l_i = hl.full([tile_b, tile_m], l_start, dtype=torch.float32)
        acc = hl.full([tile_b, tile_m, head_dim], acc_start, dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q * qk_scale, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            p = torch.exp2(qk - m_ij[:, :, None])
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            v = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc * alpha[:, :, None], p.to(v.dtype), v)
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _attention_backward_from(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    o_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
    dv_start: float,
    dk_start: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``examples/attention.py``'s ``attention_backward`` with its dV and dK
    accumulators started at the arguments."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_n_rows = k.size(0)
    LN2: float = 0.6931471824645996
    dq = torch.zeros((q.size(0), head_dim), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_n in hl.tile(total_n_rows, block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.full([tile_n, head_dim], dv_start, dtype=torch.float32)
        dk_acc = hl.full([tile_n, head_dim], dk_start, dtype=torch.float32)
        batch_idx = tile_n.begin // n_dim
        start_m = batch_idx * m_dim
        end_m = start_m + m_dim
        for tile_m in hl.tile(start_m, end_m, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            m_i = lse[tile_m]
            di = delta[tile_m]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            p_t = torch.exp2(qk_t - m_i[None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - di[None, :])).to(q.dtype)
            dq_acc = hl.dot(ds_t.T, k_j, out_dtype=torch.float32)
            hl.atomic_add(dq, [tile_m, slice(None)], dq_acc * LN2)
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
        dv[tile_n, :] = dv_acc.to(v.dtype)
        dk[tile_n, :] = (dk_acc * sm_scale).to(k.dtype)
    return dq.reshape(q_in.size()), dk.reshape(k_in.size()), dv.reshape(v_in.size())


def _constants(*starts: float) -> tuple[object, ...]:
    return tuple(hl.constexpr(start) for start in starts)


def _lowering(m_start: object, l_start: object, acc_start: object) -> tuple[bool, bool]:
    """Whether a CPU bind of the forward with these starts opens the flash
    search surface, and whether the fused body's block sizes render that
    body."""
    q, k, v = (torch.randn(1, 2, 256, 64, dtype=torch.bfloat16) for _ in range(3))
    _attention_from.reset()
    with flash_env(), cpu_codegen():
        bound = _attention_from.bind((q, k, v, m_start, l_start, acc_start))
        searched = bound.config_spec.cute_flash_search_enabled
        code = bound.to_code(helion.Config(block_sizes=[1, 128, 128]))
    return searched, "'kind': 'helion_flash'" in code


def _backward_lowering(dv_start: object, dk_start: object) -> tuple[bool, bool]:
    """The same pair for the backward: its search surface and its fused body."""
    q, k, v, o, do = (
        torch.randn(1, 2, 256, 64, dtype=torch.bfloat16) for _ in range(5)
    )
    lse, delta = (torch.randn(1, 2, 256, dtype=torch.float32) for _ in range(2))
    _attention_backward_from.reset()
    with flash_env(), cpu_codegen():
        bound = _attention_backward_from.bind(
            (q, k, v, o, lse, do, delta, 0.125, dv_start, dk_start)
        )
        searched = bound.config_spec.cute_flash_bwd_search_enabled
        code = bound.to_code(helion.Config(block_sizes=[128, 128]))
    return searched, "'kind': 'helion_flash_bwd'" in code


@pytest.mark.parametrize(
    "starts",
    [
        pytest.param((-math.inf, 1.0, 0.0), id="row-sum-one"),
        pytest.param((-math.inf, 0.0, 0.0), id="row-sum-zero"),
        pytest.param((-math.inf, 128.0, 1.0), id="finite-sum-and-accumulator"),
        pytest.param((-1e30, 1.0, 0.0), id="sentinel-max"),
    ],
)
def test_a_running_max_from_minus_infinity_or_a_sentinel_is_fused(
    starts: tuple[float, float, float],
) -> None:
    """``alpha = exp2(start - m)`` of the first KV step that shows a row a
    score is zero, so every finite row sum and accumulator ends as the fused
    body's zeros do."""
    assert _lowering(*_constants(*starts)) == (True, True)


@pytest.mark.parametrize(
    "starts",
    [
        # Source with 128 zero scores and unit values: 128 / 256, fused 1.
        pytest.param((0.0, 128.0, 0.0), id="finite-max"),
        pytest.param((0.0, 0.0, 0.0), id="zero-statistics"),
        # The floats near -1e9 are 64 apart, and ``exp2(-64)`` is not zero.
        pytest.param((-1e9, 1.0, 0.0), id="max-above-the-sentinel-bound"),
        pytest.param((1e30, 1.0, 0.0), id="positive-max"),
    ],
)
def test_a_finite_running_max_that_is_no_sentinel_takes_the_general_path(
    starts: tuple[float, float, float],
) -> None:
    """The search surface stays closed and the fused body's block sizes
    render the general path: the codegen detector declines with it."""
    assert _lowering(*_constants(*starts)) == (False, False)


@pytest.mark.parametrize(
    "starts",
    [
        # ``inf * 0`` and ``nan * 0`` are NaN in the source's first KV step.
        pytest.param((-math.inf, math.inf, 0.0), id="infinite-sum"),
        pytest.param((-math.inf, 1.0, math.nan), id="nan-accumulator"),
        pytest.param((-1e30, math.inf, 0.0), id="sentinel-max-infinite-sum"),
        # A row that sees no key keeps a sentinel's start: ``acc / l`` is 1
        # and ``0 / 0`` here, where the fused body stores 0.
        pytest.param((-1e30, 1.0, 1.0), id="sentinel-max-nonzero-accumulator"),
        pytest.param((-1e30, 0.0, 0.0), id="sentinel-max-zero-sum"),
        # The fp32 row sum holds 0: the start is read as the tile holds it.
        pytest.param((-1e30, 1e-50, 0.0), id="sentinel-max-sum-rounding-to-zero"),
    ],
)
def test_a_row_sum_or_accumulator_its_running_max_cannot_wipe_takes_the_general_path(
    starts: tuple[float, float, float],
) -> None:
    assert _lowering(*_constants(*starts)) == (False, False)


@pytest.mark.parametrize(
    ("dtype", "bound"),
    [(torch.float32, 300 * 2.0**23), (torch.float64, 2150 * 2.0**52)],
)
def test_sentinel_bound_is_where_alpha_vanishes(
    dtype: torch.dtype, bound: float
) -> None:
    """``2 * (underflow + 1) / eps``, with 149 and 1074 the exponents of the
    smallest fp32 and fp64 subnormals. From a start at the bound the next
    float above is far enough for ``alpha = exp2(start - m)`` to be zero;
    from an eighth of the bound it is not."""
    assert _running_max_sentinel_bound(dtype) == bound
    for scale, wiped in ((1.0, True), (0.125, False)):
        start = torch.tensor(-bound * scale, dtype=dtype)
        above = torch.nextafter(start, torch.zeros((), dtype=dtype))
        assert (torch.exp2(start - above) == 0).item() is wiped


def test_a_start_is_read_one_level_deep() -> None:
    """``constant_fill`` reads a fill and a fill of zero plus or minus a
    number. A sum of sums is no constant to it, so a start that doubles
    itself 64 times is declined at once, not walked ``2 ** 64`` times."""
    graph = torch.fx.Graph()
    aten = torch.ops.aten

    def tile(target: object, *args: object) -> torch.fx.Node:
        node = graph.call_function(target, args)
        node.meta["val"] = torch.empty(4)
        return node

    zero = tile(aten.full.default, [4], 0.0)
    one = tile(aten.add.Tensor, zero, 1.0)
    minus_inf = tile(aten.sub.Tensor, zero, math.inf)
    assert [constant_fill(node) for node in (zero, one, minus_inf)] == [
        0.0,
        1.0,
        -math.inf,
    ]
    doubled = one
    for _ in range(64):
        doubled = tile(aten.add.Tensor, doubled, doubled)
    assert constant_fill(doubled) is None


@pytest.mark.parametrize(
    ("dtype", "literal", "held"),
    [
        (torch.float32, 1e-50, 0.0),
        (torch.float16, 1e-10, 0.0),
        (torch.float16, 1.0 + 2.0**-12, 1.0),
        (torch.float32, -1e30, -1.0000000150474662e30),
        (torch.float16, 1e30, None),
    ],
)
def test_a_start_is_read_as_its_dtype_holds_it(
    dtype: torch.dtype, literal: float, held: float | None
) -> None:
    """``constant_fill`` returns the fill's literal rounded to the tile's
    dtype, the value every element holds: a literal at or below half the
    smallest subnormal is the zero the start checks look for. A finite
    literal that rounds to an infinity is not read."""
    graph = torch.fx.Graph()
    aten = torch.ops.aten

    def tile(target: object, *args: object) -> torch.fx.Node:
        node = graph.call_function(target, args)
        node.meta["val"] = torch.empty(4, dtype=dtype)
        return node

    filled = tile(aten.full.default, [4], literal)
    summed = tile(aten.add.Tensor, tile(aten.full.default, [4], 0.0), literal)
    assert constant_fill(filled) == held
    assert constant_fill(summed) == held
    if held is not None:
        assert held == torch.full([4], literal, dtype=dtype)[0].item()


def test_a_start_from_a_kernel_argument_takes_the_general_path() -> None:
    """A row sum filled from a runtime scalar is not a constant, whatever
    value the call passes."""
    m_start, _, acc_start = _constants(-math.inf, 1.0, 0.0)
    assert _lowering(m_start, 1.0, acc_start) == (False, False)


@pytest.mark.parametrize(
    "starts",
    [
        pytest.param(_constants(-math.inf, 1.0, 0.0), id="fused-start"),
        pytest.param(_constants(0.0, 0.0, 0.0), id="finite-max"),
        pytest.param(
            (hl.constexpr(-math.inf), 1.0, hl.constexpr(0.0)), id="runtime-row-sum"
        ),
    ],
)
def test_a_tma_only_output_keeps_the_generic_bounds_whatever_the_start(
    starts: tuple[object, object, object],
) -> None:
    """One KV tile with an output of more than 2**31 elements has no fused
    body (only the pair body stores through TMA), and the search surface
    hands it to the generic search with its 64-row block sizes before it
    reads the starts: a start that is not fused keeps those bounds."""
    q, k, v = (
        torch.empty(
            65537,
            4,
            128,
            64,
            dtype=torch.bfloat16,
            device="meta",  # @ignore-device-lint
        )
        for _ in range(3)
    )
    _attention_from.reset()
    with flash_env(), cpu_codegen():
        spec = _attention_from.bind((q, k, v, *starts)).config_spec
    assert not spec.cute_flash_search_enabled
    assert spec.cute_attention_generic_fallback_enabled
    assert spec.default_config().config["block_sizes"] == [1, 64, 64]


def test_backward_accumulators_from_zero_are_fused() -> None:
    assert _backward_lowering(*_constants(0.0, 0.0)) == (True, True)


def test_backward_accumulators_from_elsewhere_take_the_general_path() -> None:
    """A constant other than zero, and a runtime scalar that is zero in this
    call."""
    (zero, one) = _constants(0.0, 1.0)
    assert _backward_lowering(one, zero) == (False, False)
    assert _backward_lowering(zero, 0.0) == (False, False)
