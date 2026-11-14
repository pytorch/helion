"""The search surface of K-loop matmuls lowered on simdgroup fragments
(``metal_matmul="fragment"``): the config keys, seeds and normalization,
the thread fit (``search_surface.fit_threads``) and the code each config
generates.  Binds CPU tensors only."""

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
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.metal_fragment import nvfp4_cpu_args
from helion._testing.metal_fragment import nvfp4_rows
from helion._testing.metal_fragment import views_epilogue
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32
FRAG = {"metal_matmul": "fragment", "num_warps": 1}
REFUSED = "metal_matmul='fragment' can't lower"


def _frag(block_sizes: list[int], **knobs: object) -> helion.Config:
    return helion.Config(block_sizes=block_sizes, **knobs, **FRAG)


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
def row_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The result read in the Scalar layout: summed over N."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        out[tile_m] = acc.sum(1)
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
            acc = acc + hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], out_dtype=F32).to(
                torch.float16
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

    def test_search(self) -> None:
        """The search tries ``metal_matmul`` where both lowerings apply, from
        a fragment seed beside the MPP default, which leaves the key out, and
        repairs a fragment config to one SIMD group; a matmul the fragment run
        can't lower, and a kernel with none, have no such field."""
        spec = nvfp4_rows.bind(nvfp4_cpu_args()).config_spec
        field = spec._flat_fields()["metal_matmul"]
        assert isinstance(field, EnumFragment)
        self.assertEqual(field.choices, ("mpp", "fragment"))
        self.assertNotIn("metal_matmul", spec.default_config().config)
        self.assertEqual(
            [c.config.get("metal_matmul") for c in spec.autotune_seed_configs()],
            ["fragment"],
        )
        # The search repairs a fragment config to one SIMD group.
        repaired = helion.Config(
            block_sizes=[16, 16, 64], metal_matmul="fragment", num_warps=4
        )
        spec.normalize(repaired, _fix_invalid=True)
        self.assertEqual(repaired["num_warps"], 1)
        # A fragment config without num_warps takes one SIMD group.
        bare = helion.Config(block_sizes=[16, 16, 64], metal_matmul="fragment")
        spec.normalize(bare)
        self.assertEqual(bare["num_warps"], 1)
        x = torch.randn(16, 32, dtype=torch.float16)
        y = torch.randn(32, 16, dtype=torch.float16)
        for kernel, kernel_args in ((centered, (x, y)), (add, (x, x))):
            with self.subTest(kernel=kernel.name):
                spec = kernel.bind(kernel_args).config_spec
                self.assertNotIn("metal_matmul", spec._flat_fields())

    def test_threads(self) -> None:
        """The search gives a matmul config a threadgroup its matmuls run
        on, by the launch its counts make (``thread_counts.launch_threads``):
        one SIMD group for a fragment run, whole SIMD groups up to 1024
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
        kept = helion.Config(block_sizes=[16, 16, 16], num_threads=[0, 0, 4], **FRAG)
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
        config = _frag([16, 16, 16, 16, 16, 16], num_threads=[8, 1, 4, 8, 1, 4])
        bound.config_spec.normalize(config, _fix_invalid=True)
        self.assertEqual(config["num_threads"], [8, 1, 4, 8, 1, 4])
        bound.to_code(config)

    def test_generated_threads(self) -> None:
        """No config the search generates for a matmul is refused for its
        threadgroup, on fragments or on MPP: the random population and the
        neighbors of its configs of each lowering (each flat value moved to
        each of its pattern neighbors, as the pattern and LFBO searches move
        them).  Tiles past ``MAX_FRAGMENTS`` or ``MAX_STEPS`` may still be
        refused, at codegen, on purpose.  Before the thread fit, about 40% of
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
        compiled = {"fragment": 0, "mpp": 0}
        for config in configs:
            try:
                bound.to_code(config)
            except exc.BackendUnsupported as e:
                if "threadgroup" in str(e):
                    refused.append(config)
            else:
                compiled[str(lowering(config))] += 1
        self.assertEqual(refused, [])
        self.assertTrue(all(compiled.values()), compiled)

    def test_rejects(self) -> None:
        """Forms the fragment run can't lower raise at codegen rather than
        miscompile."""
        x = torch.randn(16, 32, dtype=torch.float16)
        y = torch.randn(32, 16, dtype=torch.float16)
        g = torch.randn(16)
        for kernel, args, config, message in (
            (centered, (x, y), _frag([16, 16, 16]), f"{REFUSED} the operand"),
            (carried_scale, (x, y, g), _frag([16, 16, 16]), f"{REFUSED} the operand"),
            (row_sums, (x, y), _frag([16, 16]), "result is read in the Scalar layout"),
            (whole_k, (x, y), _frag([16, 16]), f"{REFUSED} a matmul over a ':' K"),
            (
                overwritten,
                (x.clone(), y),
                _frag([16, 16, 16]),
                "a tensor the kernel writes",
            ),
            (
                views_epilogue,
                (
                    x,
                    torch.randn(32, 64, dtype=torch.float16),
                    torch.randn(64),
                ),
                _frag([16, 64, 16]),
                "8x8 fragments live",
            ),
            (
                views_epilogue,
                (x, y, g),
                helion.Config(
                    block_sizes=[16, 16, 16], metal_matmul="fragment", num_warps=4
                ),
                "one 32-thread SIMD group",
            ),
            (
                views_epilogue,
                (
                    torch.randn(16, 256, dtype=torch.float16),
                    torch.randn(256, 16, dtype=torch.float16),
                    g,
                ),
                _frag([16, 16, 256]),
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
            matmul_bf16_int4.bind((a, b)).to_code(_frag([16, 16, 16]))


if __name__ == "__main__":
    unittest.main()
