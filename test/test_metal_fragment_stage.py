"""Fragment operands loaded from threadgroup tiles (``metal_matmul=
"fragment"``, ``metal_fragment_stage``, ``fragment_mma.tile_side``): every
thread of the threadgroup copies each device-view operand's tile, raw bits a
vector at a time, between one barrier pair per loop body, and each SIMD
group loads its fragments with ``simdgroup_load``.  Renders on CPU tensors
(``helion._testing.metal.render_msl``), with no Apple GPU;
``test_metal_fragment_mma.py`` runs the kernels."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion import exc
from helion._compiler.metal.search_surface import kernel_facts
from helion._testing import TestCase
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_fragment import fragment_transposed
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import two_groups
from helion._testing.metal_fragment import views_epilogue
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

_attention = import_example("attention")
attention = helion.kernel(
    _attention.attention.fn, backend="metal", static_shapes=True, fast_math=True
)


@helion.kernel(backend="metal", static_shapes=True)
def attention_fp32_pv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Attention over [B, S, D] with P.V in float32: V widened to float."""
    b, m, d = q.size()
    n = k.size(1)
    hl.specialize(d)
    out = torch.empty_like(q)
    for tile_b, tile_m in hl.tile([b, m]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([tile_b, tile_m], dtype=torch.float32)
        acc = hl.zeros([tile_b, tile_m, d], dtype=torch.float32)
        qt = q[tile_b, tile_m, :]
        for tile_n in hl.tile(n):
            qk = torch.bmm(
                qt, k[tile_b, tile_n, :].transpose(1, 2), out_dtype=torch.float32
            )
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            p = torch.exp2(qk - m_ij[:, :, None])
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            vt = v[tile_b, tile_n, :].to(torch.float32)
            acc = torch.baddbmm(acc * alpha[:, :, None], p, vt)
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out


def _qkv(s: int) -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn(2, 2, s, 128, dtype=torch.bfloat16) for _ in range(3))


def _stage(block_sizes: list[int], **knobs: object) -> helion.Config:
    return fragment_config(block_sizes, metal_fragment_stage=True, **knobs)


class TestFragmentStage(TestCase):
    def test_attention(self) -> None:
        """Attention's K and V tiles (16 rows of D = 128, rows padded to 136
        elements) are copied by all 128 threads of 4 SIMD groups, 4 bfloat
        at a time, between the loop body's one barrier pair; the scores'
        run loads K transposed (its rows are the tile's columns), P's run V
        as it lies, both in natural K order; no lane loads K or V itself.
        The search offers the key, a seed takes it, a config without it
        loads elements as before, and a K split stages nothing."""
        args = _qkv(128)
        config = _stage(
            [1, 32, 16], metal_fragment_row_split=4, metal_fragment_hoist=True
        )
        msl = render_msl(attention, args, config)
        self.assertIn("threadgroup bfloat _frag_tile[2176];", msl)
        self.assertIn("threadgroup bfloat _frag_tile_1[2176];", msl)
        self.assertEqual(msl.count("threadgroup_barrier"), 2)
        self.assertEqual(
            msl.count(
                "for (uint _c = _frag_sg * 32 + _frag_lane; _c < 512; _c += 128) {"
            ),
            2,
        )
        self.assertIn(
            "*reinterpret_cast<threadgroup packed_ushort4*>(_frag_tile +", msl
        )
        self.assertIn(
            "simdgroup_load(_frag_b[1], _frag_tile + (8) * 136 + 0, 136, "
            "ulong2(0, 0), true);",
            msl,
        )
        self.assertIn(
            "simdgroup_load(_frag_b[1], _frag_tile_1 + (0) * 136 + 8, 136, "
            "ulong2(0, 0), false);",
            msl,
        )
        self.assertNotRegex(msl, r"[kv]_view\[")
        plain = render_msl(attention, args, fragment_config([1, 32, 16]))
        self.assertNotIn("_frag_tile", plain)
        spec = attention.bind(args).config_spec
        self.assertTrue(kernel_facts(spec).fragment_staging)
        field = spec._flat_fields()["metal_fragment_stage"]
        assert isinstance(field, EnumFragment)
        self.assertEqual(field.choices, (False, True))
        self.assertIn(
            True,
            [
                c.config.get("metal_fragment_stage")
                for c in spec.autotune_seed_configs()
            ],
        )
        self.assertNotIn("metal_fragment_stage", spec.default_config().config)
        with self.assertRaisesRegex(exc.InvalidConfig, "must be a bool"):
            spec.normalize(fragment_config([1, 32, 16], metal_fragment_stage=1))

    def test_tails(self) -> None:
        """S = 72 in K and V tiles of 16 rows: rows past the end copy as 0,
        so the scores past S and their products with V add nothing; x and
        y of 13 x 40 and 40 x 20 half matmuls in tiles of 16, whose
        unit-stride dims end inside a tile, copy one element at a time."""
        msl = render_msl(attention, _qkv(72), _stage([1, 16, 16]))
        self.assertIn("if ((offset_3 + _tr) < 72", msl)
        self.assertIn("= packed_ushort4(0, 0, 0, 0);", msl)
        args = (
            torch.randn(13, 40, dtype=torch.float16),
            torch.randn(40, 20, dtype=torch.float16),
            torch.randn(20, dtype=torch.float16),
        )
        msl = render_msl(views_epilogue, args, _stage([16, 16, 16]))
        self.assertEqual(msl.count("*reinterpret_cast<device const ushort*>("), 2)
        self.assertIn("simdgroup_matrix<half, 8, 8> _frag_a[2];", msl)
        self.assertNotIn("x[", msl)

    def test_groups(self) -> None:
        """Two runs of one K loop body copy their three float tiles, 2
        elements at a time, between one barrier pair."""
        args = (torch.randn(32, 64), torch.randn(64, 32), torch.randn(64, 32))
        msl = render_msl(two_groups, args, _stage([16, 16, 16]))
        self.assertEqual(msl.count("threadgroup_barrier"), 2)
        self.assertEqual(len(re.findall(r"threadgroup float _frag_tile\w*\[", msl)), 4)
        self.assertIn("*reinterpret_cast<threadgroup packed_uint2*>(", msl)

    def test_refusals(self) -> None:
        """A K loop split over SIMD groups stages nothing: a config with both
        fails, and the search's repairs drop the key."""
        args = (torch.randn(32, 64), torch.randn(64, 32), torch.randn(32))
        with self.assertRaisesRegex(exc.BackendUnsupported, "metal_fragment_stage"):
            render_msl(views_epilogue, args, _stage([16, 16, 16], num_warps=2))
        spec = views_epilogue.bind(args).config_spec
        config = _stage([16, 16, 16], num_warps=2)
        spec.normalize(config, _fix_invalid=True)
        self.assertNotIn("metal_fragment_stage", config.config)

    def test_transposed(self) -> None:
        """A transposed group (an nvfp4 few-row matmul, its rhs decoded)
        stages its lhs view, whose fragments are the MMA's B side: each
        loads x's tile (8 rows of M, K 16 along a row) transposed at M row
        8 i and K column 8 s of step s.  A 4-bit affine one, whose
        activation is a few-row view widened at each element
        (``_Operand.few_rows``), stages nothing and offers no staging."""
        args = (
            torch.randn(8, 4096).to(torch.bfloat16),
            torch.randint(-(2**31), 2**31 - 1, (256, 512), dtype=torch.int32),
            torch.randint(0, 0x60, (256, 256), dtype=torch.uint8),
        )
        self.assertTrue(
            kernel_facts(nvfp4_rows.bind(args).config_spec).fragment_staging
        )
        msl = render_msl(nvfp4_rows, args, _stage([16, 16, 16]))
        self.assertTrue(fragment_transposed(msl))
        self.assertIn("threadgroup bfloat _frag_tile[192];", msl)
        for step in (0, 8):
            self.assertIn(
                f"simdgroup_load(_frag_b[0], _frag_tile + (0) * 24 + {step}, 24, "
                "ulong2(0, 0), true);",
                msl,
            )
        args = affine_args(4, 32, 8, 256, 1024)
        self.assertFalse(
            kernel_facts(affine_mm.bind(args).config_spec).fragment_staging
        )
        msl = render_msl(affine_mm, args, _stage([16, 16, 16]))
        self.assertNotIn("_frag_tile", msl)

    def test_widened_view(self) -> None:
        """P.V in float32 (``v.to(float32)``): V's half view is copied into
        its tile as raw bits, like any view, so no lane loads V itself; P's
        run loads each fragment of it as half and widens the lane's two
        elements into its float fragment."""
        q, k, v = (torch.randn(2, s, 128, dtype=torch.float16) for s in (64, 128, 128))
        config = _stage([1, 32, 16], metal_fragment_row_split=4)
        msl = render_msl(attention_fp32_pv, (q, k, v), config)
        self.assertIn("threadgroup half _frag_tile_1[2176];", msl)
        self.assertIn("simdgroup_matrix<half, 8, 8> _frag_w;", msl)
        self.assertIn("simdgroup_load(_frag_w, _frag_tile_1 + ", msl)
        self.assertRegex(
            msl,
            r"_frag_b\[0\]\.thread_elements\(\)\[0\] = _frag_w\.thread_elements\(\)\[0\];",
        )
        self.assertNotRegex(msl, r"= _helion_masked\(v\[")
