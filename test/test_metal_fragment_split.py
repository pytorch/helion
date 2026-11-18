"""The K loop of a fragment run split over a threadgroup's SIMD groups
(``metal_matmul="fragment"`` at ``num_warps`` > 1, ``fragment_mma``): the
MSL each config renders and the kernels the split refuses.  Renders on CPU
tensors (``helion._testing.metal.render_msl``), with no Apple GPU."""

from __future__ import annotations

import re
import unittest

import pytest
import torch

import helion
from helion import exc
from helion._testing import TestCase
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import filled
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_fragment import k_from_16
from helion._testing.metal_fragment import nvfp4_cpu_args
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import seeded_rhs
from helion._testing.metal_fragment import two_groups
from helion._testing.metal_fragment import views_epilogue
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32
S = {"backend": "metal", "static_shapes": True}


def _views_args() -> tuple[torch.Tensor, ...]:
    """``views_epilogue``'s arguments: M = 13, K = 40 and N = 20 leave a
    tail in tiles of 16, which masks the views and the stores."""
    return (
        torch.randn(13, 40, dtype=torch.float16),
        torch.randn(40, 20, dtype=torch.float16),
        torch.randn(20),
    )


def _seeded_args() -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(8, 32, dtype=torch.float16),
        torch.randn(32, 16, dtype=torch.float16),
        torch.randn(8, 16),
    )


def _xy() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(16, 32, dtype=torch.float16), torch.randn(
        32, 16, dtype=torch.float16
    )


@helion.kernel(**S)
def scaled_in_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The K loop halves the accumulator before each matmul."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = acc * 0.5
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def counted(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The K loop counts its tiles beside the accumulator."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    count = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        tiles = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            tiles = tiles + 1.0
        out[tile_m, tile_n] = acc
        count[tile_m, tile_n] = tiles
    return out, count


@helion.kernel(**S)
def copies_in_loop(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """The K loop copies each x tile into z."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            a = x[tile_m, tile_k]
            z[tile_m, tile_k] = a
            acc = torch.addmm(acc, a, y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def doubles(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Beside the matmul, z doubled in place: a tensor the kernel reads and
    writes."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        z[tile_m, tile_n] = z[tile_m, tile_n] * 2.0
    return out


@helion.kernel(**S)
def row_sums(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Beside the matmul, the sums of x's rows: a reduction."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    sums = torch.empty([m], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        sums[tile_m] = x[tile_m, :].to(F32).sum(1)
    return out, sums


@helion.kernel(**S)
def dot_beside(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Beside the matmul, a whole-K hl.dot, whose scalar contraction
    reduces across threads."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    dots = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        dots[tile_m, tile_n] = hl.dot(
            x[tile_m, :], y[:, tile_n], out_dtype=torch.float16
        )
    return out, dots


@helion.kernel(**S)
def five_groups(
    x: torch.Tensor, ys: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Five matmuls of x in one K loop, x @ ys[i] for each i."""
    m, k = x.size()
    n = ys[0].size(1)
    outs = [torch.empty([m, n], dtype=F32, device=x.device) for _ in range(5)]
    for tile_m, tile_n in hl.tile([m, n]):
        a0 = hl.zeros([tile_m, tile_n], dtype=F32)
        a1 = hl.zeros([tile_m, tile_n], dtype=F32)
        a2 = hl.zeros([tile_m, tile_n], dtype=F32)
        a3 = hl.zeros([tile_m, tile_n], dtype=F32)
        a4 = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            xt = x[tile_m, tile_k]
            a0 = torch.addmm(a0, xt, ys[0][tile_k, tile_n])
            a1 = torch.addmm(a1, xt, ys[1][tile_k, tile_n])
            a2 = torch.addmm(a2, xt, ys[2][tile_k, tile_n])
            a3 = torch.addmm(a3, xt, ys[3][tile_k, tile_n])
            a4 = torch.addmm(a4, xt, ys[4][tile_k, tile_n])
        outs[0][tile_m, tile_n] = a0
        outs[1][tile_m, tile_n] = a1
        outs[2][tile_m, tile_n] = a2
        outs[3][tile_m, tile_n] = a3
        outs[4][tile_m, tile_n] = a4
    return outs[0], outs[1], outs[2], outs[3], outs[4]


class TestFragmentSplit(TestCase):
    def test_one_simdgroup(self) -> None:
        """At ``num_warps=1`` one SIMD group runs each K loop whole, with no
        SIMD group index and no combine.  The journal pins the MSL of a
        computed rhs (the accumulator transposed) whose decode reads a
        table and whose loads issue a K tile early, in a persistent loop
        over the tiles; of an accumulator seeded from a tile, between
        barriers; and of views masked at the tensors' extents, with a
        stored epilogue that reads a bias."""
        for kernel, args, config in (
            (
                nvfp4_rows,
                nvfp4_cpu_args(),
                fragment_config(
                    [8, 16, 16],
                    num_threads=[1, 2, 16],
                    pid_type="persistent_blocked",
                    metal_tabulate_max_entries=16,
                    range_num_stages=[2],
                ),
            ),
            (seeded_rhs, _seeded_args(), fragment_config([8, 16, 16])),
            (views_epilogue, _views_args(), fragment_config([16, 16, 16])),
        ):
            with self.subTest(kernel=kernel.name):
                self.assertExpectedJournal(render_msl(kernel, args, config))

    def test_split(self) -> None:
        """At ``num_warps=2`` two SIMD groups run the tile's 32-thread layout
        (``tid`` from the lane), stacked along z, each over every other K
        tile from its own; after the loop the second passes its fragments'
        elements through threadgroup memory to the first, which adds them
        and alone stores.  The journal holds the whole MSL."""
        args = _views_args()
        config = fragment_config([16, 16, 16], num_warps=2)
        code = views_epilogue.bind(args).to_code(config)
        self.assertRegex(code, r"split=\('offset_\d+', 2\)")
        self.assertIn("_block_dims=(4, 8, 2)", code)
        msl = render_msl(views_epilogue, args, config)
        self.assertExpectedJournal(msl)
        self.assertIn("[[required_threads_per_threadgroup(4, 8, 2)]]", msl)
        self.assertIn("uint _frag_sg [[simdgroup_index_in_threadgroup]]", msl)
        self.assertNotIn("[[thread_position_in_threadgroup]]", msl)
        self.assertIn(
            "const uint3 tid = uint3(_frag_lane % 4, (_frag_lane / 4) % 8, "
            "_frag_lane / 32);",
            msl,
        )
        self.assertRegex(
            msl,
            r"for \(int (offset_\d+) = _frag_sg \* _BLOCK_SIZE_2; \1 < 40; "
            r"\1 \+= 2 \* _BLOCK_SIZE_2\)",
        )
        self.assertIn("threadgroup float _frag_split[256];", msl)
        self.assertEqual(msl.count("threadgroup_barrier"), 2)
        self.assertIn("for (uint _s = 0; _s < 1; _s++) {", msl)
        stores = re.findall(r"if \((.*)\) \{\n\s*out\[", msl)
        self.assertEqual(len(stores), 8)
        self.assertTrue(all(s.startswith("_frag_sg == 0 && ") for s in stores))

    def test_split_forms(self) -> None:
        """Under a split, a loop that loads a K tile early loads each SIMD
        group's first tile before it and the tile ``num_warps`` ahead in it;
        an accumulator that starts from a constant starts from it in the
        first SIMD group and from 0 in the others; and the combine passes
        every accumulator its K loop updates."""
        config = fragment_config(
            [8, 16, 16],
            num_warps=4,
            num_threads=[1, 2, 16],
            pid_type="persistent_blocked",
            range_num_stages=[2],
        )
        msl = render_msl(nvfp4_rows, nvfp4_cpu_args(), config)
        # SIMD group g's first tile is K 16 * g; its next, four tiles on.
        self.assertRegex(msl, r"auto _early_0_next = x\[.* \+ _frag_sg \* 16 \+ ")
        self.assertRegex(msl, r"if \(offset_\d+ \+ 64 < 256\) \{")
        self.assertRegex(msl, r"  _early_0_next = x\[.* \+ offset_\d+ \+ 64\];")
        x, y = _xy()
        msl = render_msl(filled, (x, y), fragment_config([16, 16, 16], num_warps=4))
        self.assertIn(
            "make_filled_simdgroup_matrix<float, 8, 8>(_frag_sg == 0 ? 0.5f : 0.0f)",
            msl,
        )
        msl = render_msl(
            two_groups,
            (x[:8], y, y.clone()),
            fragment_config([16, 16, 16], num_warps=2),
        )
        self.assertIn("threadgroup float _frag_split[256];", msl)
        for f, acc in enumerate(("a[0]", "a[1]", "b[0]", "b[1]")):
            for e in (0, 1):
                self.assertIn(
                    f"_frag_split[((_frag_sg - 1) * 4 + {f}) * 64 + 2 * _frag_lane + "
                    f"{e}] = {acc}.thread_elements()[{e}];",
                    msl,
                )

    def test_refused(self) -> None:
        """The split refuses, at codegen, kernels whose SIMD groups past the
        first would compute or write something else than one SIMD group
        does (``fragment_mma.split_refusal``), and the search keeps their
        fragment configs on one SIMD group; at ``num_warps=1`` each
        compiles."""
        x, y = _xy()
        config = fragment_config([16, 16, 16], num_threads=[4, 8, 1])
        for kernel, args, reason in (
            (
                seeded_rhs,
                (x, y, torch.randn(16, 16)),
                "an accumulator starts from a tile",
            ),
            (
                scaled_in_loop,
                (x, y),
                "the K loop updates an accumulator other than by its matmul",
            ),
            (
                counted,
                (x, y),
                "the K loop carries a value besides its matmuls' accumulators",
            ),
            (copies_in_loop, (x, y, torch.empty_like(x)), "the K loop writes memory"),
            (
                doubles,
                (x, y, torch.randn(16, 16)),
                "the kernel reads a tensor it writes",
            ),
            (row_sums, (x, y), "the kernel has a reduction"),
            (
                dot_beside,
                (x, y),
                (
                    "the kernel has an hl.dot the fragment runs don't take, whose "
                    "contraction reduces across threads"
                ),
            ),
        ):
            with self.subTest(kernel=kernel.name):
                bound = kernel.bind(args)
                bound.to_code(config)
                split = helion.Config(**{**config.config, "num_warps": 4})
                with self.assertRaisesRegex(
                    exc.BackendUnsupported,
                    re.escape(
                        f"split over 4 SIMD groups (num_warps=4): {reason}; "
                        "use num_warps=1"
                    ),
                ):
                    bound.to_code(split)
                bound.config_spec.normalize(split, _fix_invalid=True)
                self.assertEqual(split["num_warps"], 1)
                self.assertEqual(
                    {
                        c.config["num_warps"]
                        for c in bound.config_spec.autotune_seed_configs()
                    },
                    {1},
                )

    def test_limits(self) -> None:
        """The split refuses a combine past the threadgroup memory (five 8x8
        accumulators over 32 SIMD groups take 31 x 1280 bytes), more SIMD
        groups than a threadgroup holds (one accumulator over 64 fits the
        memory), and, in a loop that loads a tile early, more SIMD groups
        than tiles, whose surplus would load past the loop's end."""
        x = torch.randn(8, 32, dtype=torch.float16)
        ys = [torch.randn(32, 8, dtype=torch.float16) for _ in range(5)]
        bound = five_groups.bind((x, ys))
        bound.to_code(fragment_config([8, 8, 16], num_warps=16, num_threads=[4, 8, 1]))
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"takes 39680 bytes of threadgroup memory"
        ):
            bound.to_code(
                fragment_config([8, 8, 16], num_warps=32, num_threads=[4, 8, 1])
            )
        bound = filled.bind((x, ys[0]))
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"split over 64 SIMD groups .* holds at most 32"
        ):
            bound.to_code(
                fragment_config([8, 8, 16], num_warps=64, num_threads=[4, 8, 1])
            )
        # K from 16 to 64 in tiles of 16: three tiles.
        bound = k_from_16.bind((torch.randn(16, 64), torch.randn(64, 16)))
        bound.to_code(fragment_config([16, 16, 16], num_warps=4))
        bound.to_code(fragment_config([16, 16, 16], num_warps=2, range_num_stages=[2]))
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "a K loop of 3 tiles split over 4 SIMD groups"
        ):
            bound.to_code(
                fragment_config([16, 16, 16], num_warps=4, range_num_stages=[2])
            )


if __name__ == "__main__":
    unittest.main()
