"""Fragment operands built from another fragment group's result
(``metal_matmul="fragment"``, ``fragment_mma._chain_refusal``): attention's
P from its scores' fragments, in place, with the consumer taking K in its
natural order, and element values two consumers read computed once into
registers (``fragment_lines.plan_holds``).  Renders on CPU tensors
(``helion._testing.metal.render_msl``), with no Apple GPU;
``test_metal_fragment_mma.py`` runs the kernels."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion import exc
from helion._compiler.metal.search_surface import kernel_facts
from helion._compiler.metal.search_surface import matmul_choices
from helion._testing import TestCase
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import chained_rhs
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_fragment import k_from_16
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

_attention = import_example("attention")


def _metal(kernel: helion.Kernel) -> helion.Kernel:
    return helion.kernel(kernel.fn, backend="metal", static_shapes=True, fast_math=True)


attention = _metal(_attention.attention)


@helion.kernel(backend="metal", static_shapes=True)
def seeded_row_max(x: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The row max of x @ y over N tiles, a running max the N loop carries
    from ``b``'s rows: a line seed that loads its rows before the first
    fragment run."""
    m = x.size(0)
    hl.specialize(x.size(1))
    n = y.size(1)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        m_i = b[tile_m]
        q = x[tile_m, :]
        for tile_n in hl.tile(n):
            m_i = torch.maximum(m_i, torch.amax(torch.mm(q, y[:, tile_n]), 1))
        out[tile_m] = m_i
    return out


@helion.kernel(backend="metal", static_shapes=True)
def int_epilogue(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The accumulator converted to int32, read by two element ops (``q &
    7`` and ``q >> 2``): an int element value computed once into
    registers."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        q = acc.to(torch.int32)
        out[tile_m, tile_n] = ((q & 7) + (q >> 2)).to(torch.float32)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def early_chain(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A consumer of ``exp(acc)`` that reads the K loop's carry before the
    matmul accumulating into it."""
    m, k = x.size()
    n = hl.specialize(y.size(1))
    out = torch.empty([m, w.size(1)], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            out[tile_m, :] = torch.mm(torch.exp(acc), w[:, :])
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
    return out


def _qkv(s: int = 64) -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn(2, 2, s, 128, dtype=torch.bfloat16) for _ in range(3))


class TestFragmentChain(TestCase):
    def test_attention(self) -> None:
        """``examples/attention.py`` lowers on fragments: P's A fragments of
        each K step are its scores' fragments of that step's columns, in
        natural K order, held in bfloat; the exponential, read by the row
        sum and by P, runs once per element; the row max, sum, alpha and
        running values stay in registers; nothing reaches threadgroup
        memory.  The search offers fragments for it.  (``causal_attention``
        reads ``tile.index`` through a load, which no Metal lowering
        takes yet.)"""
        args = _qkv()
        msl = render_msl(attention, args, fragment_config([1, 16, 16]))
        self.assertNotIn("threadgroup_barrier", msl)
        self.assertNotRegex(msl, r"\bthreadgroup \w+ ")
        self.assertIn("int _e_k = 8 * _fs + _fn + (_fi & 1);", msl)
        self.assertIn("simdgroup_matrix<bfloat, 8, 8> _frag_a[2];", msl)
        self.assertRegex(
            msl, r"_held_\d+\[2 \* \(\(_fi >> 1\) \* 2 \+ _fs\) \+ \(_fi & 1\)\]"
        )
        # One exponential per score element (2 x 2 fragments of a 16 x 16
        # tile, 8 per lane) and one per row value (2 rows).
        self.assertEqual(msl.count("metal::fast::exp("), 8 + 2)
        self.assertIn("simd_shuffle_xor(_lr, 8)", msl)
        self.assertIn("fragment", matmul_choices(attention.bind(args).config_spec))

    def test_chained_rhs(self) -> None:
        """A chained rhs, the consumer's K the scores' M: B fragment ``j`` of
        step ``s`` is the scores' fragment at row ``s``, column ``j``; float
        operands keep float fragments."""
        args = (torch.randn(32, 16), torch.randn(16, 24), torch.randn(16, 32))
        msl = render_msl(chained_rhs, args, fragment_config([16, 16]))
        self.assertIn("int _e_k = 8 * _fs + _fm;", msl)
        self.assertRegex(
            msl, r"_mpp_c_0\[\(_fs \* 2 \+ \(_fi >> 1\)\)\]\.thread_elements\(\)"
        )
        self.assertIn("simdgroup_float8x8 _frag_b[2];", msl)
        self.assertNotIn("threadgroup_barrier", msl)
        self.assertIsNone(re.search(r"_held_", msl))

    def test_early_chain_refused(self) -> None:
        """A chain that reads its source's carry before the source's op runs
        has no registers to read: the fragment lowering refuses it.
        ``num_threads=[1, 1]`` keeps the K block off threads, which an
        earlier MPP launch check would refuse first."""
        args = (torch.randn(16, 32), torch.randn(32, 16), torch.randn(16, 32))
        config = fragment_config([16, 16], num_threads=[1, 1])
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "before the source's op runs"
        ):
            render_msl(early_chain, args, config)

    def test_held_dtype(self) -> None:
        """A held element value's registers take its dtype: an int32
        conversion of the accumulator, which the two element ops shift and
        mask, is held in ``int`` registers."""
        args = (torch.randn(16, 32), torch.randn(32, 16))
        msl = render_msl(int_epilogue, args, fragment_config([16, 16, 16]))
        (held,) = re.findall(r"\b(\w+) (_held_\d+)\[8\];", msl)
        self.assertEqual(held[0], "int")
        self.assertRegex(msl, rf"\({held[1]}\[_fi\] >> v_\d+\)")

    def test_hoist(self) -> None:
        """``metal_fragment_hoist`` builds attention's scaled queries, which
        the N loop doesn't change, into fragments once before it: 16 MMA
        steps of 2 row fragments over D = 128, which the scores' run reads;
        no load of q runs in the loop.  The search offers the key, a seed
        takes it, and a config without it keeps building at each run."""
        args = _qkv()
        config = fragment_config([1, 16, 16], metal_fragment_hoist=True)
        msl = render_msl(attention, args, config)
        hoisted = msl.index("simdgroup_float8x8 _frag_h0_0[32];")
        loop = msl.index("for (int offset_3 = 0;")
        self.assertLess(hoisted, loop)
        self.assertNotIn("q_view[", msl[loop:])
        self.assertIn(
            "simdgroup_multiply_accumulate(_mpp_c_0[0], _frag_h0_0[0], _frag_b[0]",
            msl,
        )
        self.assertNotIn(
            "_frag_h", render_msl(attention, args, fragment_config([1, 16, 16]))
        )
        spec = attention.bind(args).config_spec
        self.assertTrue(kernel_facts(spec).fragment_hoists)
        field = spec._flat_fields()["metal_fragment_hoist"]
        assert isinstance(field, EnumFragment)
        self.assertEqual(field.choices, (False, True))
        self.assertIn(
            True,
            [
                c.config.get("metal_fragment_hoist")
                for c in spec.autotune_seed_configs()
            ],
        )
        self.assertNotIn("metal_fragment_hoist", spec.default_config().config)
        with self.assertRaisesRegex(exc.InvalidConfig, "must be a bool"):
            spec.normalize(fragment_config([1, 16, 16], metal_fragment_hoist=1))

    def test_row_split(self) -> None:
        """``metal_fragment_row_split`` runs the tiles on 4 SIMD groups, each
        on its 16 of the 64 rows: the launch stacks them along z, each
        element's and row value's row starts at its SIMD group's share, and
        each thread's position comes from its lane.  The search offers the
        key, a seed takes it, a config without it keeps one SIMD group, and
        it can't split rows and K loops at once."""
        args = _qkv(128)
        config = fragment_config(
            [1, 64, 16], metal_fragment_row_split=4, metal_fragment_hoist=True
        )
        msl = render_msl(attention, args, config)
        self.assertIn("[[required_threads_per_threadgroup(32, 1, 4)]]", msl)
        self.assertIn("uint _frag_sg [[simdgroup_index_in_threadgroup]]", msl)
        self.assertIn("int _e_m = 8 * (_fi >> 1) + _fm + _frag_sg * 16;", msl)
        self.assertIn("int _e_m = 8 * _ri + _fm + _frag_sg * 16;", msl)
        self.assertIn("const uint3 tid = uint3(_frag_lane % 32,", msl)
        self.assertNotIn(
            "_frag_sg", render_msl(attention, args, fragment_config([1, 64, 16]))
        )
        spec = attention.bind(args).config_spec
        self.assertIsNotNone(kernel_facts(spec).fragment_row_block)
        field = spec._flat_fields()["metal_fragment_row_split"]
        assert isinstance(field, EnumFragment)
        self.assertEqual(field.choices, (1, 2, 4, 8))
        self.assertIn(
            4,
            [
                c.config.get("metal_fragment_row_split")
                for c in spec.autotune_seed_configs()
            ],
        )
        self.assertNotIn("metal_fragment_row_split", spec.default_config().config)
        with self.assertRaisesRegex(exc.InvalidConfig, "must be one of"):
            spec.normalize(fragment_config([1, 64, 16], metal_fragment_row_split=3))
        # 16 rows are 2 fragments, which 4 SIMD groups can't share.
        with self.assertRaisesRegex(exc.BackendUnsupported, "whole 8-row fragments"):
            render_msl(
                attention,
                args,
                fragment_config([1, 16, 16], metal_fragment_row_split=4),
            )
        x, y = torch.randn(64, 48), torch.randn(48, 32)
        with self.assertRaisesRegex(exc.BackendUnsupported, "split the K loops"):
            render_msl(
                k_from_16,
                (x, y),
                fragment_config([32, 16, 16], num_warps=2, metal_fragment_row_split=2),
            )

    def test_row_split_seed(self) -> None:
        """With 4 SIMD groups splitting the rows, a line's seed, loaded
        before any fragment accumulator is declared, reads each group's own
        16 rows, as its updates and its store do."""
        args = (torch.randn(128, 32), torch.randn(32, 64), torch.randn(128))
        config = fragment_config([64, 16], metal_fragment_row_split=4)
        msl = render_msl(seeded_row_max, args, config)
        rows = re.findall(r"int _e_m = 8 \* _ri \+ _fm[^;]*;", msl)
        self.assertTrue(rows)
        self.assertEqual(set(rows), {"int _e_m = 8 * _ri + _fm + _frag_sg * 16;"})
        self.assertLess(
            msl.index("auto m_i = b[indices_0];"),
            msl.index("simdgroup_float8x8 _mpp_c_0"),
        )

    def test_row_split_refused(self) -> None:
        """A chained rhs takes the scores' rows as its K, which no SIMD group
        holds alone: the search doesn't offer the key, and a config with it
        fails, saying why."""
        args = (torch.randn(32, 16), torch.randn(16, 24), torch.randn(16, 32))
        spec = chained_rhs.bind(args).config_spec
        self.assertIsNone(kernel_facts(spec).fragment_row_block)
        self.assertNotIn("metal_fragment_row_split", spec._flat_fields())
        with self.assertRaisesRegex(exc.BackendUnsupported, "rows"):
            render_msl(
                chained_rhs, args, fragment_config([16, 16], metal_fragment_row_split=2)
            )
