from __future__ import annotations

import random
import unittest

import torch

import helion
from helion._compiler.backend import TritonBackend
from helion._compiler.pallas.backend import PallasBackend
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessPallas
from helion.autotuner.config_fragment import BlockSizeChoicesFragment
from helion.autotuner.config_fragment import BlockSizeFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

LOOP_TYPES = ("default", "fori_loop", "emit_pipeline")


def _loop_kwargs(loop_type: str) -> dict[str, object]:
    return {} if loop_type == "default" else {"pallas_loop_type": loop_type}


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = w.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(
    backend="pallas", static_shapes=True, pallas_non_power_of_two_block_search=True
)
def pallas_matmul_non_power_of_two_search(
    x: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = w.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_row_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc = acc + x[tile_m, tile_n].to(torch.float32).sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_axpy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * 2.0 + y[tile]
    return out


@onlyBackends(["pallas"])
@skipUnlessPallas("JAX/Pallas TPU not available")
class TestPallasNonPowerOfTwoBlockSizes(TestCase):
    def test_matmul_non_power_of_two_tiles(self) -> None:
        # Weight-streaming shape: N=2176 (17 lane tiles), K=5120 in 640 chunks.
        x = torch.randn(16, 5120, device=DEVICE, dtype=torch.float32)
        w = torch.randn(5120, 2176, device=DEVICE, dtype=torch.float32)
        expected = x @ w
        for loop_type in LOOP_TYPES:
            with self.subTest(loop_type=loop_type):
                code, result = code_and_output(
                    pallas_matmul,
                    (x, w),
                    block_sizes=[16, 2176, 640],
                    **_loop_kwargs(loop_type),
                )
                self.assertIn("_BLOCK_SIZE_1 = 2176", code)
                self.assertIn("(1, 0, 640, 0)", code)
                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-3)

    def test_matmul_bf16_non_power_of_two_output_tile(self) -> None:
        x = torch.randn(16, 5120, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(5120, 2176, device=DEVICE, dtype=torch.bfloat16)
        _, result = code_and_output(pallas_matmul, (x, w), block_sizes=[16, 2176, 512])
        expected = (x.float() @ w.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected, rtol=2e-2, atol=1.0)

    def test_reduction_loop_non_power_of_two_tiles(self) -> None:
        # 2560 divides 7680; 1536 does not divide 5120 (masked/padded tail).
        for n, block in ((7680, 2560), (5120, 1536)):
            x = torch.randn(16, n, device=DEVICE, dtype=torch.float32)
            for loop_type in LOOP_TYPES:
                with self.subTest(n=n, block=block, loop_type=loop_type):
                    _, result = code_and_output(
                        pallas_row_sum,
                        (x,),
                        block_sizes=[16, block],
                        **_loop_kwargs(loop_type),
                    )
                    torch.testing.assert_close(result, x.sum(-1), rtol=1e-4, atol=1e-3)

    def test_elementwise_non_power_of_two_tile(self) -> None:
        x = torch.randn(64, 5120, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 5120, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_axpy, (x, y), block_sizes=[8, 2560])
        self.assertIn("_BLOCK_SIZE_1 = 2560", code)
        torch.testing.assert_close(result, x * 2.0 + y)

    def test_full_extent_block_on_non_power_of_two_dim(self) -> None:
        # 1000 is neither a power of two nor a multiple of 128, but a block
        # covering the whole axis is a legal BlockSpec.
        x = torch.randn(16, 1000, device=DEVICE, dtype=torch.float32)
        y = torch.randn(16, 1000, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(pallas_axpy, (x, y), block_sizes=[16, 1000])
        self.assertIn("_BLOCK_SIZE_1 = 1000", code)
        torch.testing.assert_close(result, x * 2.0 + y)

    def test_rejects_non_tile_multiple_lane_block(self) -> None:
        x = torch.randn(64, 5120, device=DEVICE, dtype=torch.float32)
        bound = pallas_axpy.bind((x, x))
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig,
            r"block_sizes\]\[1\] must be a power of two, equal the full dimension "
            r"extent \(5120\) or a multiple of 128 .* got 100",
        ):
            bound.to_code(helion.Config(block_sizes=[8, 100]))

    def test_rejects_non_tile_multiple_sublane_block(self) -> None:
        # bf16 packs 16 rows per native (sublane, lane) tile.
        x = torch.randn(64, 5120, device=DEVICE, dtype=torch.bfloat16)
        bound = pallas_axpy.bind((x, x))
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig, r"multiple of 16 .* got 24"
        ):
            bound.to_code(helion.Config(block_sizes=[24, 2560]))
        bound.to_code(helion.Config(block_sizes=[48, 2560]))

    def test_rejects_partial_full_extent_and_non_integer(self) -> None:
        x = torch.randn(16, 1000, device=DEVICE, dtype=torch.float32)
        bound = pallas_axpy.bind((x, x))
        with self.assertRaisesRegex(helion.exc.InvalidConfig, r"got 500"):
            bound.to_code(helion.Config(block_sizes=[16, 500]))
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig, r"must be a positive integer, got 2\.5"
        ):
            bound.to_code(helion.Config(block_sizes=[16, 2.5]))

    def test_non_power_of_two_search_space_is_opt_in(self) -> None:
        x = torch.randn(16, 5120, device=DEVICE, dtype=torch.float32)
        w = torch.randn(5120, 2176, device=DEVICE, dtype=torch.float32)

        default_spec = pallas_matmul.bind((x, w)).config_spec
        for spec in default_spec.block_sizes:
            fragment = spec._fragment(default_spec)
            self.assertIs(type(fragment), BlockSizeFragment)

        search_spec = pallas_matmul_non_power_of_two_search.bind((x, w)).config_spec
        values = [
            spec._fragment(search_spec).search_values()
            for spec in search_spec.block_sizes
        ]
        self.assertIn(2176, values[1])
        for size in (640, 1280, 2560, 5120):
            self.assertIn(size, values[2])
        self.assertNotIn(1088, values[1])

        generation = ConfigGeneration(search_spec)
        random.seed(0)
        for _ in range(20):
            generation.unflatten(generation.random_flat())


class TestNonPowerOfTwoBlockSizeConfig(TestCase):
    def test_backend_property(self) -> None:
        self.assertTrue(TritonBackend().requires_power_of_two_block_sizes)
        self.assertFalse(PallasBackend().requires_power_of_two_block_sizes)

    def test_triton_rejects_non_power_of_two(self) -> None:
        backend = TritonBackend()
        spec = ConfigSpec(backend=backend)
        spec.block_sizes.append(
            BlockSizeSpec(
                block_id=0,
                size_hint=2176,
                non_power_of_two_multiple=(
                    None if backend.requires_power_of_two_block_sizes else 1
                ),
            )
        )
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig, r"must be a power of two, got 2176"
        ):
            spec.normalize({"block_sizes": [2176]})
        config = {"block_sizes": [2048]}
        spec.normalize(config)
        self.assertEqual(config["block_sizes"], [2048])

    def test_tile_multiple_normalization(self) -> None:
        spec = ConfigSpec(backend=PallasBackend())
        block = BlockSizeSpec(block_id=0, size_hint=5120, non_power_of_two_multiple=128)
        block.full_extent = 5120
        spec.block_sizes.append(block)
        for value in (640, 2560, 5120, 1024, 6400):
            config = {"block_sizes": [value]}
            spec.normalize(config)
            self.assertEqual(config["block_sizes"], [value])
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig, r"extent \(5120\) or a multiple of 128"
        ):
            spec.normalize({"block_sizes": [1000]})

    def test_fragments_accept_non_power_of_two_current(self) -> None:
        fragment = BlockSizeFragment(128, 4096, 512)
        self.assertEqual(fragment.pattern_neighbors(2176), [2048, 4096])
        self.assertEqual(fragment.differential_mutation(2176, 1, 2), 2048)
        self.assertEqual(fragment.differential_mutation(2176, 2, 1), 4096)
        self.assertEqual(fragment.differential_mutation(2176, 1, 1), 2048)
        self.assertEqual(len(fragment.encode(2176)), 1)

        choices = BlockSizeChoicesFragment(128, 8192, 512, (640, 1280, 5120, 9000))
        self.assertEqual(choices.extra_values, (640, 1280, 5120))
        self.assertEqual(choices.pattern_neighbors(640), [512, 1024])
        self.assertEqual(
            choices.pattern_neighbors(1024, radius=2), [512, 640, 1280, 2048]
        )
        self.assertEqual(choices.differential_mutation(512, 2, 1), 640)
        self.assertEqual(choices.differential_mutation(5120, 1, 2), 4096)
        self.assertEqual(choices.pattern_neighbors(1000), [640, 1024])
        self.assertEqual(choices.cardinality(), 10)
        random.seed(0)
        for _ in range(50):
            self.assertIn(choices.random(), choices.search_values())


if __name__ == "__main__":
    unittest.main()
