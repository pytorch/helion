"""The per-row KV lower bound on the single-warpgroup flash body (CPU renders).

One KV tile resolves to the single-warpgroup body (``ws_overlap`` with
``cute_flash_s_stage`` 1), and a config can pin that body at more tiles.  It
runs no KV range prelude and its loops walk every KV tile, the tiles above
the diagonal included: each thread loads its row's bound once, ahead of the
loops, every tile takes the row's exact range mask, and a row the bound
leaves without a key gets a zero normalizer.  The body takes the kernels that
start their running max from ``-inf``; one that starts from a finite sentinel
keeps the generic lowering.  The device runs are
``test_flash_attention_row_kv_begin_single_warpgroup_*`` in
``test/test_cute_backend.py``.
"""

from __future__ import annotations

import ast
import functools
import math

import pytest
import torch

import helion
from helion._testing.cute_aux import cpu_codegen
from helion._testing.cute_aux import flash_env
from helion._testing.cute_flash import render_flash_config
import helion.language as hl

pytestmark = pytest.mark.cpu_only

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _row_bound_attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    kv_begin: torch.Tensor,
    m_start: float,
) -> torch.Tensor:
    """``kv_begin[batch, query] <= key <= query``: the per-row KV lower bound,
    over a running max that starts at ``m_start`` (a compile-time constant,
    ``hl.constexpr``, traces to the literal fill a kernel spells)."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    heads = hl.specialize(q_in.size(1))
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    begin_view = kv_begin.reshape([-1, m_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], m_start, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            begin_batch = tile_b.index // heads
            row_begin = begin_view[begin_batch, tile_m]
            causal = tile_m.index[None, :, None] >= tile_n.index[None, None, :]
            in_range = tile_n.index[None, None, :] >= row_begin[:, :, None]
            qk = torch.where(in_range & causal, qk, float("-inf"))
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


# One KV tile takes the body by default; three tiles pin it, with a two-slot
# K/V ring so that both of its KV loops (the prefetch and the drain phase)
# render.
_CASES: dict[str, tuple[int, dict[str, object]]] = {
    "one_tile": (128, {}),
    "three_tiles": (
        384,
        {
            "cute_flash_pipeline_family": "ws_overlap",
            "cute_flash_s_stage": 1,
            "cute_flash_kv_stage": 2,
        },
    ),
}

_BATCH_ROW = "flash_rkb_bh = flash_bh // _flash_rkb_heads_per_batch"
_ROW_BOUND = (
    "flash_rkb_row_lb = _helion_flash_rt.row_kv_begin_row_bound("
    "_flash_mRowKvBegin, flash_rkb_bh, "
    "flash_m_tile * cutlass.Int32(128) + cutlass.Int32(tLDcS[0][0]))"
)
_MASK = (
    "_helion_flash_rt.row_kv_begin_mask_t2r("
    "tLDrS, tLDcS, flash_rkb_row_lb, flash_m_tile, flash_kv)"
)
_ZERO_SUM_NORMALIZER = (
    "flash_inv_sum = cutlass.Float32(cutlass.select_("
    "flash_row_sum == cutlass.Float32(0.0), cutlass.Float32(0.0), "
    "cutlass.Float32(1.0) / flash_row_sum))"
)


def _args(
    seq: int,
    m_start: float = -math.inf,
    *,
    batch: int = 2,
    heads: int = 2,
    device: str = "cpu",
) -> tuple[object, ...]:
    q, k, v = (
        torch.empty(batch, heads, seq, 64, dtype=torch.bfloat16, device=device)
        for _ in range(3)
    )
    begin = torch.zeros(batch, seq, dtype=torch.int32, device=device)
    return q, k, v, begin, hl.constexpr(m_start)


def _device_function(code: str) -> ast.FunctionDef:
    (function,) = (
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and [ast.unparse(decorator) for decorator in node.decorator_list]
        == ["cute.kernel"]
    )
    return function


def _render(case: str) -> tuple[str, ast.FunctionDef]:
    """The kernel's render at ``case`` and its device function."""
    seq, config = _CASES[case]
    code = render_flash_config(
        _row_bound_attention,
        (seq,),
        functools.partial(_args, seq),
        {"block_sizes": [1, 128, 128], **config},
    )
    # The single-warpgroup body: one score stage, a 128-thread launch.
    assert "'topology': 'ws_overlap'" in code and "'s_stage': 1" in code
    assert "block=(128, 1, 1)" in code
    return code, _device_function(code)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_row_bound_masks_every_kv_tile(case: str) -> None:
    """The score transform used to read the range prelude's interior bound
    and the softmax prelude's row bound, neither of which this body emits:
    the kernel failed to compile at one KV tile."""
    code, function = _render(case)
    statements = [ast.unparse(statement) for statement in function.body]
    kv_loops = [
        statement
        for statement in function.body
        if isinstance(statement, ast.For)
        and ast.unparse(statement.target) in ("flash_kv", "flash_kv_d")
    ]
    # Each thread loads its row's bound once, ahead of the KV loops, and the
    # body reads no other name of the row bound (the range prelude's, which
    # it does not run) and neither of the two ahead of its binding.
    bindings = {"flash_rkb_bh": _BATCH_ROW, "flash_rkb_row_lb": _ROW_BOUND}
    assert statements.count(_BATCH_ROW) == 1
    assert statements.count(_ROW_BOUND) == 1
    assert (
        statements.index(_BATCH_ROW)
        < statements.index(_ROW_BOUND)
        < function.body.index(kv_loops[0])
    )
    assert {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id.startswith("flash_rkb_")
    } == set(bindings)
    for name, binding in bindings.items():
        assert name not in "\n".join(statements[: statements.index(binding)])
    # The loops walk every KV tile, the ones above the diagonal included,
    # and the body has no interior bound to test: each tile takes the mask
    # unguarded (a direct statement of its loop, not of a branch).
    tiles = 0
    for loop in kv_loops:
        assert [ast.unparse(statement) for statement in loop.body].count(_MASK) == 1
        assert isinstance(loop.iter, ast.Call)
        assert ast.unparse(loop.iter.func) == "cutlass.range"
        tiles += ast.literal_eval(loop.iter.args[0])
    assert tiles == _CASES[case][0] // 128
    assert code.count("row_kv_begin_mask_t2r(") == len(kv_loops)
    # A row the bound leaves without a key has a zero row sum: a 0 output
    # row, as on the other bodies.
    assert statements.count(_ZERO_SUM_NORMALIZER) == 1


def test_a_sentinel_start_keeps_the_generic_lowering() -> None:
    """The body takes the row-bound kernels that start their running max
    from ``-inf``. From a finite sentinel a row keeps part of its start
    where the body computes from its own (``KNOWN_GAPS.md``,
    cute-flash-sentinel-start-taken-for-minus-inf), and no such kernel ever
    compiled on this body: one KV tile, which only this body takes, opens no
    flash surface and renders the generic lowering, and a config that pins
    the body at three tiles is declined, where the default config takes the
    body with two score stages. The fa4 pair keeps the kernel, as it did."""
    sentinel = -1e30
    with flash_env(), cpu_codegen():
        one_tile = _row_bound_attention._bind_isolated(_args(128, sentinel))
        assert not one_tile.config_spec.cute_flash_search_enabled
        generic = one_tile.to_code(one_tile.config_spec.default_config())
        assert "'kind': 'helion_flash'" not in generic
        assert "'kind': 'helion_flash'" not in one_tile.to_code(
            helion.Config(block_sizes=[1, 128, 128])
        )
        three_tiles = _row_bound_attention._bind_isolated(_args(384, sentinel))
        assert three_tiles.config_spec.cute_flash_search_enabled
        default = three_tiles.to_code(three_tiles.config_spec.default_config())
        assert "'topology': 'ws_overlap'" in default and "'s_stage': 2" in default
        with pytest.raises(helion.exc.BackendUnsupported, match="late validation"):
            three_tiles.to_code(
                helion.Config(block_sizes=[1, 128, 128], **_CASES["three_tiles"][1])
            )
        pair = _row_bound_attention._bind_isolated(_args(256, sentinel))
        assert pair.config_spec.cute_flash_search_enabled
        code = pair.to_code(helion.Config(block_sizes=[1, 128, 128]))
        # One KV tile with an output of more than 2**31 elements is handed to
        # the generic search with its block-size bounds, whatever the start.
        for m_start in (-math.inf, sentinel):
            large = _row_bound_attention._bind_isolated(
                _args(
                    128,
                    m_start,
                    batch=65537,
                    heads=4,
                    device="meta",  # @ignore-device-lint
                )
            )
            spec = large.config_spec
            assert spec.cute_attention_generic_fallback_enabled
            assert spec.default_config().config["block_sizes"] == [1, 64, 64]
    assert "'kind': 'helion_flash'" in code and "'topology': 'fa4'" in code
