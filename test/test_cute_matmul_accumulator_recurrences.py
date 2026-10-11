"""K-loop matmuls whose accumulator is not a plain ``carry = dot(acc=carry)``.

The CuTe MMA lowerings seed one accumulator fragment in the first K
iteration and read it out after the loop.  The scalar fallback zeroes a
per-K-lane ``dot_acc`` running sum before the K loop and adds the
accumulator from before the loop once.  Both apply only when the loop carry
is exactly the matmul's result over its own previous value.  A rescale
before the dot (``acc = acc * 0.5``), a change after it (``acc = acc + 1.0``)
or another carry reading the running value takes the fallback's owned
product-sum route, which completes each tile's sum before its consumers, or
keeps the running sum only where the chunk-recurrence pass restructures it
per tile; otherwise it is rejected.  A bare matmul reduces the whole K
extent (``test_bare_batched_matmuls_use_collective_k_reduction``).
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _rescale_before_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = acc * 0.5
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _rescaled_dot_accumulator(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc * 0.5)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _change_after_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.relu(
                torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n]) + 1.0
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _sum_of_running_values(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    before = torch.empty_like(out)
    after = torch.empty_like(out)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        s_before = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        s_after = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            s_before = s_before + acc
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            s_after = s_after + acc
        out[tile_m, tile_n] = acc
        before[tile_m, tile_n] = s_before
        after[tile_m, tile_n] = s_after
    return out, before, after


@helion.kernel(backend="cute", static_shapes=True)
def _two_dots_per_tile(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _max_of_tile_products(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.full([tile_m, tile_n], float("-inf"), dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            product = torch.mm(a[tile_m, tile_k], b[tile_k, tile_n])
            acc = torch.maximum(acc, product.to(torch.float32))
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _plain_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def _tile_products(a: torch.Tensor, b: torch.Tensor, block_k: int) -> list:
    af, bf = a.float(), b.float()
    return [
        af[:, k : k + block_k] @ bf[k : k + block_k]
        for k in range(0, a.size(1), block_k)
    ]


def _inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    b = torch.randn(k, n, device=DEVICE, dtype=torch.float16) * 0.25
    return a, b


_SHAPES = [(256, 256, 512), (200, 136, 96)]
# The default config (where a plain accumulator takes the MMA path), a wider
# K tile, row lanes on the scalar fallback and a collective config that falls
# back.
_CONFIGS = [
    {"block_sizes": [16, 16, 16]},
    {"block_sizes": [16, 16, 32]},
    {"block_sizes": [32, 16, 16], "num_threads": [8, 16, 0]},
    {
        "block_sizes": [64, 64, 32],
        "cute_collective_mma": True,
        "cute_collective_compute": "warp",
        "cute_collective_copy": "async_cached",
        "cute_collective_stages": 2,
    },
]
_VECTOR_K = {"block_sizes": [16, 16, 32], "cute_vector_widths": [1, 1, 4]}


@onlyBackends(["cute"])
@skipIfRefEager("references follow the configured K tiles")
class TestCuteMatmulAccumulatorRecurrences(TestCase):
    def _check(
        self,
        kernel: object,
        expected: object,
        shape: tuple[int, int, int],
        config: dict[str, object],
    ) -> str:
        a, b = _inputs(*shape)
        code, result = code_and_output(kernel, (a, b), **config)
        products = _tile_products(a, b, config["block_sizes"][2])
        reference = expected(products)
        results = result if isinstance(result, tuple) else (result,)
        references = reference if isinstance(reference, tuple) else (reference,)
        for got, want in zip(results, references, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-2, atol=5e-2)
        return code

    def _sweep(self, kernel: object, expected: object) -> None:
        for shape in _SHAPES:
            for config in _CONFIGS:
                with self.subTest(kernel=kernel.name, shape=shape, config=config):
                    self._check(kernel, expected, shape, config)

    def test_rescale_before_the_dot(self) -> None:
        def expected(products: list) -> torch.Tensor:
            acc = torch.zeros_like(products[0])
            for product in products:
                acc = acc * 0.5 + product
            return acc

        self._sweep(_rescale_before_dot, expected)
        self._sweep(_rescaled_dot_accumulator, expected)
        code = self._check(_rescale_before_dot, expected, _SHAPES[0], _CONFIGS[0])
        # The fragment would be seeded once and keep no rescale.
        self.assertNotIn("cute.gemm(", code)

    def test_change_after_the_dot(self) -> None:
        def expected(products: list) -> torch.Tensor:
            acc = torch.zeros_like(products[0])
            for product in products:
                acc = torch.relu(acc + product + 1.0)
            return acc

        self._sweep(_change_after_dot, expected)

    def test_other_carries_read_the_tile_values(self) -> None:
        def expected(products: list) -> tuple[torch.Tensor, ...]:
            acc = torch.zeros_like(products[0])
            before = torch.zeros_like(acc)
            after = torch.zeros_like(acc)
            for product in products:
                before = before + acc
                acc = acc + product
                after = after + acc
            return acc, before, after

        self._sweep(_sum_of_running_values, expected)

    def test_two_dots_per_tile(self) -> None:
        def expected(products: list) -> torch.Tensor:
            return 2 * sum(products)

        self._sweep(_two_dots_per_tile, expected)

    def test_bare_matmul_feeding_another_recurrence(self) -> None:
        def expected(products: list) -> torch.Tensor:
            return torch.stack([product.half().float() for product in products]).amax(0)

        self._sweep(_max_of_tile_products, expected)

    def test_vectorized_k_recurrence_is_rejected(self) -> None:
        a, b = _inputs(256, 256, 512)
        for kernel in (_rescale_before_dot, _change_after_dot, _sum_of_running_values):
            with (
                self.subTest(kernel=kernel.name),
                pytest.raises(exc.BackendUnsupported),
            ):
                code_and_output(kernel, (a, b), **_VECTOR_K)

    def test_plain_accumulator_keeps_its_lowerings(self) -> None:
        a, b = _inputs(256, 256, 512)
        expected = a.float() @ b.float()
        code, out = code_and_output(_plain_matmul, (a, b), block_sizes=[16, 16, 16])
        self.assertIn("cute.gemm(", code)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=5e-2)
        code, out = code_and_output(
            _plain_matmul, (a, b), block_sizes=[32, 16, 16], num_threads=[8, 16, 0]
        )
        self.assertIn("dot_acc", code)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=5e-2)
