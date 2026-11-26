"""The search surface of K-loop matmuls lowered on simdgroup fragments
(``metal_matmul="fragment"``): the config keys, seeds and normalization,
the thread fit (``search_surface.fit_threads``), the code each config
generates, and the lowering a config without the key falls back to
(``search_surface.MatmulFallback``).  Binds CPU tensors only."""

from __future__ import annotations

import random
import re
import unittest

import pytest
import torch

import helion
from helion import exc
from helion._compiler.metal.fragment_mma import _max_offset
from helion._compiler.metal.search_surface import kernel_facts
from helion._compiler.metal.search_surface import matmul_choices
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.metal import addmm_k_loop
from helion._testing.metal import render_msl
from helion._testing.metal_fragment import batched_k_loop
from helion._testing.metal_fragment import batched_whole_k
from helion._testing.metal_fragment import fp8_rows
from helion._testing.metal_fragment import fragment_config
from helion._testing.metal_fragment import k_from_16
from helion._testing.metal_fragment import nvfp4_cpu_args
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import views_epilogue
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32
REFUSED = "metal_matmul='fragment' can't lower"


@helion.kernel(backend="metal", static_shapes=True)
def k_from_offset(x: torch.Tensor, y: torch.Tensor, k0: int) -> torch.Tensor:
    """A K loop from an offset known only at runtime."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k0, k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def _lowering(msl: str) -> str:
    """The lowering whose code the MSL holds."""
    if "matmul2d_descriptor" in msl:
        return "mpp"
    if "simdgroup_matrix" in msl:
        return "fragment"
    return "scalar"


@helion.kernel(backend="metal", static_shapes=True)
def fixed_m(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """M tiled by a literal 16, which has no ``num_threads``."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[16, None]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def two_roots(
    x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two roots, each a K-loop matmul."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out2 = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, u[tile_m, tile_k], v[tile_k, tile_n])
        out2[tile_m, tile_n] = acc
    return out, out2


@helion.kernel(backend="metal", static_shapes=True)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """No matmul."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def centered(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The rhs centered per column: a reduction over K in its producers."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            b = y[tile_k, tile_n]
            acc = torch.addmm(acc, x[tile_m, tile_k], b - b.mean(0, keepdim=True))
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def carried_scale(x: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """The rhs scaled by a per-column factor the K loop carries."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        sc = g[tile_n].to(F32)
        for tile_k in hl.tile(k):
            sc = sc * 0.5
            acc = torch.addmm(
                acc, x[tile_m, tile_k], (y[tile_k, tile_n] * sc).to(x.dtype)
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def overwritten(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The lhs computed from a load of x, which the K loop then overwrites:
    a lane re-running the load at the op would read the new value."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            a = x[tile_m, tile_k]
            x[tile_m, tile_k] = a * 2
            acc = torch.addmm(acc, torch.relu(a), y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def whole_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One matmul over a ':' K, outside any K loop."""
    m, _ = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        out[tile_m, tile_n] = torch.addmm(acc, x[tile_m, :], y[:, tile_n])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def scalar_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An hl.dot to fp16, which neither MPP nor fragments admit: it lowers
    to Scalar code."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(k):
            acc = acc + hl.dot(
                x[tile_m, tile_k], y[tile_k, tile_n], out_dtype=torch.float16
            )
        out[tile_m, tile_n] = acc
    return out


class TestFragmentConfig(unittest.TestCase):
    def test_wide_offsets(self) -> None:
        """A view's offsets take ``long`` by the largest offset its indices
        reach, not its element count: a strided view passes 2**31 at few
        elements."""
        strided = torch.empty_strided(
            (4097, 16), (2**20, 1), device=torch.device("meta")
        )
        self.assertLess(strided.numel(), 2**31)
        self.assertGreaterEqual(_max_offset(strided), 2**31)
        self.assertEqual(
            _max_offset(torch.empty(3, 4, device=torch.device("meta"))), 11
        )

    def test_config(self) -> None:
        """``metal_matmul`` defaults to ``"mpp"``: a config without it, or
        with ``"mpp"``, generates the same code, and ``"fragment"`` the
        fragment run; another value is invalid.  Bound on CPU tensors, as
        the code is all it reads."""
        bound = nvfp4_rows.bind(nvfp4_cpu_args())
        base = helion.Config(
            block_sizes=[16, 16, 64], num_threads=[1, 1, 512], num_warps=1
        )
        code = bound.to_code(base)
        self.assertEqual(
            code, bound.to_code(helion.Config(**base.config, metal_matmul="mpp"))
        )
        self.assertNotIn("fragment=", code)
        fragment = bound.to_code(helion.Config(**base.config, metal_matmul="fragment"))
        self.assertIn("fragment=", fragment)
        with self.assertRaisesRegex(exc.InvalidConfig, "metal_matmul must be one of"):
            bound.config_spec.normalize(helion.Config(metal_matmul="tensor"))
        with self.assertRaisesRegex(exc.InvalidConfig, "metal_matmul must be one of"):
            bound.config_spec.normalize(helion.Config(metal_matmul=None))
        for cap in (16.0, True, 32, None):
            with self.assertRaisesRegex(
                exc.InvalidConfig, "metal_tabulate_max_entries must be one of"
            ):
                bound.config_spec.normalize(
                    helion.Config(
                        metal_matmul="fragment", metal_tabulate_max_entries=cap
                    )
                )

    def test_search(self) -> None:
        """The search tries ``metal_matmul`` over every lowering that applies,
        from fragment seeds beside the MPP default, which leaves the key out,
        and ``metal_tabulate_max_entries`` at the caps that admit the kernel's
        tables, on one SIMD group and on four splitting the K loop
        (``num_warps``, which the search keeps), and on four splitting its
        rows (``metal_fragment_row_split``; eight don't take whole fragments
        of its 48); a matmul the fragment run
        can't lower searches MPP and the scalar contraction, without tables,
        and a kernel with no matmul has neither field."""
        spec = nvfp4_rows.bind(nvfp4_cpu_args()).config_spec
        field = spec._flat_fields()["metal_matmul"]
        assert isinstance(field, EnumFragment)
        self.assertEqual(field.choices, ("mpp", "fragment", "scalar"))
        self.assertNotIn("metal_matmul", spec.default_config().config)
        tables = spec._flat_fields()["metal_tabulate_max_entries"]
        assert isinstance(tables, EnumFragment)
        # The code's nibble takes 16 entries.  The E4M3 scale's 7-bit field
        # takes none: its shift is an integer of promoted type, and the cast
        # after it reads the whole byte.
        self.assertEqual(tables.choices, (0, 16))
        self.assertEqual(
            [
                (
                    c.config.get("metal_matmul"),
                    c.config.get("metal_tabulate_max_entries"),
                    c.config.get("range_num_stages"),
                    c.config.get("num_warps"),
                    c.config.get("metal_fragment_row_split"),
                )
                for c in spec.autotune_seed_configs()
            ],
            [
                # Each lowering's strongest guess first (a budget cut keeps
                # them): the largest row split that takes whole fragments of
                # N's 48 rows (the group is transposed, so its rows are N's),
                # its activation staged and loaded a tile ahead, MPP's
                # default, then the other seeds.
                ("fragment", None, [2], 1, 4),
                (None, None, None, 4, None),
                *(
                    (*knobs, num_warps, None)
                    for num_warps in (1, 4)
                    for knobs in (
                        ("fragment", None, None),
                        ("fragment", None, [2]),
                        ("fragment", 16, None),
                        ("fragment", 16, [2]),
                    )
                ),
            ],
        )
        # The search keeps a fragment config's SIMD groups, which split its
        # K loop.
        split = helion.Config(
            block_sizes=[16, 16, 64], metal_matmul="fragment", num_warps=4
        )
        spec.normalize(split, _fix_invalid=True)
        self.assertEqual(split["num_warps"], 4)
        # A fragment config without num_warps takes one SIMD group.
        bare = helion.Config(block_sizes=[16, 16, 64], metal_matmul="fragment")
        spec.normalize(bare)
        self.assertEqual(bare["num_warps"], 1)
        # One range_num_stages slot, the K loop's, searched at 0 and 2; 3 is
        # rejected.
        self.assertEqual(len(spec.range_num_stages), 1)
        self.assertEqual(
            spec.range_num_stages[0]._fragment(spec).search_values(), [0, 2]
        )
        with self.assertRaisesRegex(exc.InvalidConfig, r"must be one of \(0, 1, 2\)"):
            spec.normalize(
                helion.Config(
                    block_sizes=[16, 16, 64],
                    metal_matmul="fragment",
                    range_num_stages=[3],
                )
            )
        # Only a K loop that can load ahead has a slot: known bounds it runs
        # between (``loads_ahead``), or a view it can stage, whose tile copies
        # load ahead between any bounds (``pipeline_tile_copies``).  One from
        # 16 has one and loads ahead under it.  One from a runtime offset has
        # one through its views: under 2 stages it pipelines its tile copies
        # when it stages them, and is refused (its element loads need known
        # bounds) when it doesn't.  A dynamic kernel whose operands are
        # computed has none, so the search never asks it for 2 stages.
        from_16 = k_from_16.bind((torch.randn(16, 64), torch.randn(64, 16)))
        self.assertEqual(len(from_16.config_spec.range_num_stages), 1)
        code = from_16.to_code(fragment_config([16, 16, 16], range_num_stages=[2]))
        self.assertRegex(code, r"k_step=\('offset_\d+', 16\)")
        x = torch.randn(8, 64, dtype=torch.float16)
        y = torch.randn(64, 16, dtype=torch.float16)
        from_offset = k_from_offset.bind((x, y, 16))
        self.assertEqual(len(from_offset.config_spec.range_num_stages), 1)
        piped = render_msl(
            k_from_offset,
            (x, y, 16),
            fragment_config(
                [16, 16, 16], range_num_stages=[2], metal_fragment_stage=True
            ),
        )
        self.assertIn("_pipe0_frag_tile", piped)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "range_num_stages=2 on a K loop without known"
        ):
            from_offset.to_code(fragment_config([16, 16, 16], range_num_stages=[2]))
        dynamic = helion.kernel(nvfp4_rows.fn, backend="metal", static_shapes=False)
        bound = dynamic.bind(nvfp4_cpu_args())
        self.assertEqual(len(bound.config_spec.range_num_stages), 0)
        # A user's fragment config asking for stages there is refused; the
        # search's repair drops them.
        staged = fragment_config([16, 16, 16], range_num_stages=[2])
        with self.assertRaisesRegex(
            exc.InvalidConfig, "one entry per Metal fragment K loop"
        ):
            bound.config_spec.normalize(staged)
        bound.config_spec.normalize(staged, _fix_invalid=True)
        self.assertNotIn("range_num_stages", staged.config)
        # An MPP config keeps loading with the range_num_stages a config
        # carried before Metal read the key (a Triton config's), as zeros.
        mpp = helion.Config(block_sizes=[16, 16, 64], range_num_stages=[3, 4])
        spec.normalize(mpp)
        self.assertEqual(mpp["range_num_stages"], [0])
        # A byte's decode takes a 256-entry table.
        fp8_args = (
            torch.randn(8, 224).to(torch.bfloat16),
            torch.randint(-(2**31), 2**31 - 1, (40, 56), dtype=torch.int32),
            torch.randint(120, 130, (40, 7), dtype=torch.uint8),
        )
        spec = fp8_rows.bind(fp8_args).config_spec
        tables = spec._flat_fields()["metal_tabulate_max_entries"]
        assert isinstance(tables, EnumFragment)
        self.assertEqual(tables.choices, (0, 256))
        x = torch.randn(16, 32, dtype=torch.float16)
        y = torch.randn(32, 16, dtype=torch.float16)
        for kernel, kernel_args, choices in (
            (centered, (x, y), ("mpp", "scalar")),
            (add, (x, x), None),
        ):
            with self.subTest(kernel=kernel.name):
                spec = kernel.bind(kernel_args).config_spec
                fields = spec._flat_fields()
                field = fields.get("metal_matmul")
                self.assertEqual(
                    None if field is None else getattr(field, "choices", None),
                    choices,
                )
                self.assertNotIn("metal_tabulate_max_entries", fields)
                self.assertEqual(len(spec.range_num_stages), 0)
                # A config with range_num_stages loads as before: without it.
                old = helion.Config(
                    **spec.default_config().config, range_num_stages=[2]
                )
                spec.normalize(old)
                self.assertNotIn("range_num_stages", old.config)

    def test_fallback(self) -> None:
        """A config without ``metal_matmul`` takes the first lowering that
        compiles it: MPP's, then fragments' on one SIMD group, then the
        scalar contraction's, from the config as given.  One that names a
        lowering keeps it and fails with that lowering's error, as the
        search's drawn configs do."""
        h = torch.float16
        deep = (torch.randn(32, 1024, dtype=h), torch.randn(1024, 32, dtype=h))
        wide = (torch.randn(16, 128, dtype=h), torch.randn(128, 512, dtype=h))
        for args, config, expected in (
            (deep, helion.Config(block_sizes=[16, 16, 64]), "mpp"),
            # N = 512, past an MPP op's 256: fragments take it.
            (wide, helion.Config(block_sizes=[16, 512, 32]), "fragment"),
            # K = 512, past an MPP op's 256 and the 16 steps a fragment run
            # unrolls.
            (deep, helion.Config(block_sizes=[16, 16, 512]), "scalar"),
            # One thread, which neither cooperative lowering runs on.
            (
                deep,
                helion.Config(block_sizes=[16, 16, 64], num_threads=[1, 1, 1]),
                "scalar",
            ),
        ):
            with self.subTest(config=config):
                msl = render_msl(addmm_k_loop, args, config)
                self.assertEqual(_lowering(msl), expected)
        # The scalar fallback keeps the config's threads, not the fragment
        # pass's one SIMD group.
        config = helion.Config(
            block_sizes=[16, 16, 512], num_threads=[16, 16, 1], num_warps=8
        )
        msl = render_msl(addmm_k_loop, deep, config)
        self.assertEqual(_lowering(msl), "scalar")
        self.assertIn("required_threads_per_threadgroup(16, 16, 1)", msl)
        for config, message in (
            (
                helion.Config(block_sizes=[16, 16, 512], metal_matmul="mpp"),
                "one op covers M, N <= 256 and K <= 256",
            ),
            (
                helion.Config(
                    block_sizes=[16, 16, 64], num_threads=[1, 1, 1], metal_matmul="mpp"
                ),
                "it needs whole 32-thread SIMD groups",
            ),
        ):
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(exc.BackendUnsupported, re.escape(message)),
            ):
                render_msl(addmm_k_loop, deep, config)
        # Every lowering refusing: the scalar contraction's error, caused by
        # MPP's.
        int8 = torch.int8
        args = (torch.ones(16, 64, dtype=int8), torch.ones(64, 16, dtype=h))
        with self.assertRaises(exc.BackendUnsupported) as caught:
            addmm_k_loop.bind(args).to_code(helion.Config(block_sizes=[16, 16, 16]))
        self.assertIn("on the scalar contraction", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, exc.BackendUnsupported)

    def test_threads(self) -> None:
        """The search gives a matmul config a threadgroup its matmuls run
        on, by the launch its counts make (``thread_counts.launch_threads``):
        a fragment run's tile over one SIMD group, whatever the SIMD groups
        splitting its K loop (``num_warps``), whole SIMD groups up to 1024
        threads for MPP's.  Counts that launch more (fragment ``[0, 0, 4]``:
        the M and N tiles fill the SIMD group, then 4 K threads each; MPP
        ``[16, 32, 8]``: 2048) or fewer (``[2, 2, 1]``: 4) are fitted to the
        budget, keeping the explicit ones; counts that launch one are kept,
        even past their block; a user's config, a kernel whose matmul
        neither lowering takes, and a kernel without matmuls aren't
        repaired."""
        spec = nvfp4_rows.bind(nvfp4_cpu_args()).config_spec
        for lowering, num_warps, blocks, threads, fitted in (
            ("fragment", 1, [16, 16, 16], [0, 0, 4], [8, 1, 4]),
            ("fragment", 1, [16, 16, 16], [1, 1, 0], [2, 1, 16]),
            ("fragment", 1, [16, 16, 16], [1, 32, 0], [1, 32, 0]),
            ("fragment", 1, [16, 16, 16], [16, 16, 16], [2, 4, 4]),
            ("fragment", 1, [16, 16, 64], [1, 1, 0], [1, 1, 0]),
            ("fragment", 1, [16, 16, 16], [0, 0, 0], [0, 0, 0]),
            ("fragment", 1, [16, 16, 16], [2, 2, 0], [2, 2, 0]),
            ("fragment", 4, [16, 16, 16], [0, 0, 4], [8, 1, 4]),
            ("fragment", 8, [16, 16, 16], [16, 16, 16], [2, 4, 4]),
            ("mpp", 16, [16, 16, 16], [16, 32, 8], [8, 8, 8]),
            ("mpp", 4, [16, 16, 16], [2, 2, 1], [8, 4, 4]),
            ("mpp", 4, [16, 16, 16], [0, 0, 4], [0, 0, 4]),
            ("mpp", 4, [16, 16, 16], [0, 0, 0], [0, 0, 0]),
            ("mpp", 8, [64, 32, 64], [512, 256, 1], [512, 256, 1]),
        ):
            with self.subTest(lowering=lowering, blocks=blocks, threads=threads):
                config = helion.Config(
                    block_sizes=blocks,
                    num_threads=threads,
                    num_warps=num_warps,
                    metal_matmul=lowering,
                )
                spec.normalize(config, _fix_invalid=True)
                self.assertEqual(config.config.get("num_threads", [0] * 3), fitted)
        kept = fragment_config([16, 16, 16], [0, 0, 4])
        spec.normalize(kept)
        self.assertEqual(kept["num_threads"], [0, 0, 4])
        x = torch.randn(16, 32, dtype=torch.float16)
        # Nor are those of a kernel whose matmul neither lowering takes.
        spec = scalar_dot.bind((x, x.T.contiguous())).config_spec
        self.assertEqual(kernel_facts(spec).thread_loops, ())
        small = helion.Config(block_sizes=[16, 16, 16], num_threads=[2, 2, 2])
        spec.normalize(small, _fix_invalid=True)
        self.assertEqual(small["num_threads"], [2, 2, 2])
        spec = add.bind((x, x)).config_spec
        wide = helion.Config(block_sizes=[16, 32], num_threads=[16, 32], num_warps=8)
        spec.normalize(wide, _fix_invalid=True)
        self.assertEqual(wide["num_threads"], [16, 32])

    def test_threads_beside_a_literal_tile(self) -> None:
        """A literal-size tile runs one thread per element, so the fit
        counts it: beside M's literal 16, ``[2, 1]`` launches the fragment
        run's one SIMD group and is kept (counting only the tunable tiles,
        the fit made it ``[8, 4]``, 128 threads, which codegen refused), and
        other counts are fitted to what the 16 threads leave."""
        x = torch.randn(32, 64, dtype=torch.float16)
        y = torch.randn(64, 32, dtype=torch.float16)
        bound = fixed_m.bind((x, y))
        spec = bound.config_spec
        self.assertEqual(kernel_facts(spec).literal_threads, (16, 1))
        for lowering, num_warps, threads, fitted in (
            ("fragment", 1, [2, 1], [2, 1]),
            ("fragment", 1, [0, 4], [1, 2]),
            ("fragment", 1, [16, 16], [1, 2]),
            ("mpp", 4, [16, 16], [2, 4]),
        ):
            with self.subTest(lowering=lowering, threads=threads):
                config = helion.Config(
                    block_sizes=[16, 16],
                    num_threads=threads,
                    num_warps=num_warps,
                    metal_matmul=lowering,
                )
                spec.normalize(config, _fix_invalid=True)
                self.assertEqual(config.config["num_threads"], fitted)
                bound.to_code(config)

    def test_no_fit_for_several_roots(self) -> None:
        """The fit models one nest (the grid and one device loop), so a
        kernel with two roots gets none: counts that launch one SIMD group
        per root (``(8, 4, 1)``) are kept, where treating both roots' tiles
        as one nest made them ``[2, 1, 2, 2, 1, 4]``, which codegen
        refused."""
        x = torch.randn(32, 64, dtype=torch.float16)
        y = torch.randn(64, 32, dtype=torch.float16)
        bound = two_roots.bind((x, y, x, y))
        self.assertEqual(kernel_facts(bound.config_spec).thread_loops, ())
        config = fragment_config(
            [16, 16, 16, 16, 16, 16], num_threads=[8, 1, 4, 8, 1, 4]
        )
        bound.config_spec.normalize(config, _fix_invalid=True)
        self.assertEqual(config["num_threads"], [8, 1, 4, 8, 1, 4])
        bound.to_code(config)

    def test_generated_threads(self) -> None:
        """No config the search generates for a matmul is refused for its
        threadgroup, on fragments or on MPP: the random population and the
        neighbors of its configs of each lowering (each flat value moved to
        each of its pattern neighbors, as the pattern and LFBO searches move
        them).  K tiles past ``MAX_STEPS`` may still be refused, at
        codegen, on purpose.  Before the thread fit, about 40% of
        the fragment configs and 30% of the MPP ones launched too many
        threads or too few."""
        random.seed(0)
        bound = nvfp4_rows.bind(nvfp4_cpu_args())
        gen = ConfigGeneration(bound.config_spec)

        def lowering(config: helion.Config) -> object:
            return config.config.get("metal_matmul", "mpp")

        population = [list(flat) for flat in gen.random_population_flat(40)]
        candidates = list(population)
        for kind in ("fragment", "mpp"):
            bases = [f for f in population if lowering(gen.unflatten(list(f))) == kind]
            for base in bases[:2]:
                for index, field in enumerate(gen.flat_spec):
                    candidates.extend(
                        [*base[:index], value, *base[index + 1 :]]
                        for value in field.pattern_neighbors(base[index])
                    )
        configs = [gen.unflatten(list(flat)) for flat in candidates]
        self.assertGreater(len(configs), 40)
        refused = []
        compiled = {"fragment": 0, "mpp": 0, "scalar": 0}
        for config in configs:
            try:
                bound.to_code(config)
            except exc.BackendUnsupported as e:
                # check_mpp_launch's refusals of an MPP threadgroup and of a
                # fragment tile's threads.
                if re.search("threadgroup of|whose tile takes", str(e)):
                    refused.append(config)
            else:
                compiled[str(lowering(config))] += 1
        self.assertEqual(refused, [])
        self.assertTrue(all(compiled.values()), compiled)

    def test_whole_k(self) -> None:
        """A matmul over a ':' K is one fragment run from zeroed
        accumulators, which no K loop carries, so SIMD groups don't split
        it (``split_refusal``)."""
        x = torch.randn(16, 32, dtype=torch.float16)
        y = torch.randn(32, 16, dtype=torch.float16)
        msl = render_msl(whole_k, (x, y), fragment_config([16, 16]))
        self.assertEqual(_lowering(msl), "fragment")
        bound = whole_k.bind((x, y))
        self.assertFalse(kernel_facts(bound.config_spec).fragment_splits)

    def test_batched(self) -> None:
        """A batched matmul lowers on fragments on its batch tile's one
        index: a K loop of baddbmm and a ':' K bmm in an inner N loop, each
        with a computed lhs, whose elements recompute over the batch block,
        and a view rhs, which reads the batch tile's offset as an outer
        index.  A batch tile over 1 raises."""
        x = torch.randn(3, 16, 32, dtype=torch.float16)
        for kernel, args, config in (
            (
                batched_k_loop,
                (x, torch.randn(3, 32, 24, dtype=torch.float16)),
                fragment_config([1, 16, 16, 16]),
            ),
            (
                batched_whole_k,
                (x, torch.randn(3, 24, 32, dtype=torch.float16)),
                fragment_config([1, 16, 16]),
            ),
        ):
            with self.subTest(kernel=kernel.name):
                msl = render_msl(kernel, args, config)
                self.assertEqual(_lowering(msl), "fragment")
                # The computed lhs's re-run load and the y view's outer
                # index read the batch tile.
                self.assertIn("(indices_0 * 512)", msl)
                self.assertIn("(offset_0 * 768)", msl)
                self.assertIn("fragment", matmul_choices(kernel.bind(args).config_spec))
        args = (x, torch.randn(3, 32, 24, dtype=torch.float16))
        with self.assertRaises(exc.BackendUnsupported):
            render_msl(batched_k_loop, args, fragment_config([2, 16, 16, 16]))

    def test_few_row_seeds(self) -> None:
        """Row splits fit the rows an op over the row block holds: the
        block cut to the loop's extent (``op_dim``), not the block.  At 16
        rows no split of 4 or 8 takes whole fragments, so the featured seed
        is one SIMD group's, staged, and the search's repair lowers a split
        of 8 to the 2 the cut rows take; at 64 rows the seeds split 8 and 4.
        Every seed compiles."""
        h = torch.float16
        y, bias = torch.randn(64, 64, dtype=h), torch.randn(64)
        for m, splits in ((16, [None]), (64, [8, 4])):
            with self.subTest(m=m):
                bound = views_epilogue.bind((torch.randn(m, 64, dtype=h), y, bias))
                spec = bound.config_spec
                self.assertEqual(kernel_facts(spec).fragment_row_extent, m)
                gen = ConfigGeneration(spec)
                seeds = [
                    c
                    for _, c in gen.user_seed_flat_config_pairs(
                        spec.autotune_seed_configs()
                    )
                ]
                featured = [
                    c for c in seeds if c.config.get("metal_fragment_stage") is True
                ]
                self.assertEqual(
                    [c.config.get("metal_fragment_row_split") for c in featured],
                    splits,
                )
                for config in seeds:
                    bound.to_code(config)
                repaired = fragment_config([64, 16, 16], metal_fragment_row_split=8)
                spec.normalize(repaired, _fix_invalid=True)
                self.assertEqual(
                    repaired.config.get("metal_fragment_row_split"),
                    2 if m == 16 else 8,
                )

    def test_runtime_bound_stages(self) -> None:
        """A K loop from a runtime offset loads a tile ahead only through
        its tile copies, under ``metal_fragment_stage``: the search's repair
        drops a config's 2 stages there when it stages nothing, and none of
        its drawn configs is refused for them.  A loop with a side that
        loads per lane at any config (an lhs over one row, which a tile
        can't hold) gets no slot."""
        random.seed(0)
        h = torch.float16
        y = torch.randn(64, 16, dtype=h)
        bound = k_from_offset.bind((torch.randn(8, 64, dtype=h), y, 16))
        spec = bound.config_spec
        self.assertEqual(len(spec.range_num_stages), 1)
        for stage, expected in ((False, [0]), (True, [2])):
            config = fragment_config(
                [16, 16, 16], range_num_stages=[2], metal_fragment_stage=stage
            )
            spec.normalize(config, _fix_invalid=True)
            self.assertEqual(config["range_num_stages"], expected)
            bound.to_code(config)
        gen = ConfigGeneration(spec)
        refused = []
        for flat in gen.random_population_flat(16):
            config = gen.unflatten(list(flat))
            try:
                bound.to_code(config)
            except exc.BackendUnsupported as e:
                if "without known bounds" in str(e):
                    refused.append(config)
        self.assertEqual(refused, [])
        one_row = k_from_offset.bind((torch.randn(1, 64, dtype=h), y, 16))
        self.assertEqual(len(one_row.config_spec.range_num_stages), 0)

    def test_whole_k_extent(self) -> None:
        """A ':' K is one fragment K tile, which no config changes: past the
        unrolled steps' bound the search offers no fragment configs, and an
        explicit one raises at codegen."""
        for k, fits in ((64, True), (1024, False)):
            x, y = torch.randn(16, k), torch.randn(k, 16)
            choices = matmul_choices(whole_k.bind((x, y)).config_spec)
            self.assertEqual("fragment" in choices, fits, choices)
            if fits:
                msl = render_msl(whole_k, (x, y), fragment_config([16, 16]))
                self.assertEqual(_lowering(msl), "fragment")
            else:
                with self.assertRaisesRegex(
                    exc.BackendUnsupported,
                    f"{REFUSED} a matmul over a ':' K: .* 128 MMA steps",
                ):
                    render_msl(whole_k, (x, y), fragment_config([16, 16]))

    def test_rejects(self) -> None:
        """Forms the fragment run can't lower raise at codegen rather than
        miscompile."""
        x = torch.randn(16, 32, dtype=torch.float16)
        y = torch.randn(32, 16, dtype=torch.float16)
        g = torch.randn(16)
        for kernel, args, config, message in (
            (centered, (x, y), fragment_config([16, 16, 16]), f"{REFUSED} the operand"),
            (
                carried_scale,
                (x, y, g),
                fragment_config([16, 16, 16]),
                f"{REFUSED} the operand",
            ),
            (
                overwritten,
                (x.clone(), y),
                fragment_config([16, 16, 16]),
                "a tensor the kernel writes",
            ),
            (
                views_epilogue,
                (x, y, g),
                fragment_config([16, 16, 16], num_threads=[2, 2, 1]),
                "whose tile takes 4 threads; it runs on a 32-thread SIMD group",
            ),
            (
                views_epilogue,
                (
                    torch.randn(16, 256, dtype=torch.float16),
                    torch.randn(256, 16, dtype=torch.float16),
                    g,
                ),
                fragment_config([16, 16, 256]),
                "its 32 MMA steps are unrolled (max 16)",
            ),
        ):
            with (
                self.subTest(kernel=kernel.name, message=message),
                self.assertRaisesRegex(exc.BackendUnsupported, re.escape(message)),
            ):
                kernel.bind(args).to_code(config)

    def test_factored_k(self) -> None:
        """int4_gemm's K held as per-thread parts raises."""
        matmul_bf16_int4 = helion.kernel(
            import_example("int4_gemm").matmul_bf16_int4.fn,
            backend="metal",
            static_shapes=True,
        )
        a = torch.randn(16, 128, dtype=torch.bfloat16)
        b = torch.randint(-128, 127, (64, 64), dtype=torch.int8)
        message = f"{REFUSED} a matmul over a K held as per-thread parts"
        with self.assertRaisesRegex(exc.BackendUnsupported, re.escape(message)):
            matmul_bf16_int4.bind((a, b)).to_code(fragment_config([16, 16, 16]))


if __name__ == "__main__":
    unittest.main()
