from __future__ import annotations

import unittest
from unittest import mock

import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
import helion.language as hl

_BULK = "cp.async.bulk.prefetch.L2.global"
_PER_ELEMENT = "prefetch.global.L2 [$1]"


def _prefetch_lines(code: str) -> list[str]:
    return [line for line in code.splitlines() if "tl.inline_asm_elementwise" in line]


@helion.kernel(autotune_effort="none")
def add_prefetched(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        hl.prefetch(y, [tile_m, tile_n])
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(autotune_effort="none")
def add_plain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(autotune_effort="none")
def row_sum_prefetched(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], device=x.device, dtype=torch.float32)
    for tile_m in hl.tile(x.size(0)):
        hl.prefetch(x, [tile_m, slice(None)])
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
    return out


@helion.kernel(autotune_effort="none")
def scale_prefetched(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.prefetch(x, [tile])
        out[tile] = x[tile] * 2
    return out


@helion.kernel(autotune_effort="none")
def gather_prefetched(x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    out = torch.empty([rows.size(0), x.size(1)], device=x.device, dtype=x.dtype)
    for tile_r, tile_n in hl.tile([rows.size(0), x.size(1)]):
        row = rows[tile_r]
        hl.prefetch(x, [row[:, None], tile_n.index[None, :]])
        hl.prefetch(x, [row, tile_n], extra_mask=row[:, None] > 0)
        out[tile_r, tile_n] = x[row, tile_n]
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def independent_roots(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    tmp = torch.empty_like(x)
    out = torch.empty_like(y)
    for tile in hl.tile(x.size(0), block_size=16):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(y.size(0), block_size=16):
        hl.prefetch(tmp, [tile])
        out[tile] = y[tile] * 2
    return tmp, out


@helion.kernel(autotune_effort="none", static_shapes=True)
def dependent_roots(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    tmp = torch.empty_like(x)
    out = torch.empty_like(y)
    for tile in hl.tile(x.size(0), block_size=16):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(y.size(0), block_size=16):
        hl.prefetch(tmp, [tile])
        out[tile] = y[tile] * 2 + tmp[tile]
    return tmp, out


@onlyBackends(["triton"])
@skipIfNotCUDA()
@skipIfRocm("prefetch lowers to PTX")
@skipIfTileIR("prefetch lowers to PTX")
class TestPrefetch(RefEagerTestDisabled, TestCase):
    @skipIfCudaCapabilityLessThan((9, 0), reason="cp.async.bulk needs sm90")
    def test_tile_bulk_partial_last_tile(self) -> None:
        x = torch.randn(100, 1000, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        code, out = code_and_output(add_prefetched, (x, y), block_sizes=[8, 128])
        torch.testing.assert_close(out, x + y)
        (line,) = _prefetch_lines(code)
        self.assertIn(_BULK, line)
        # 1000 % 128 != 0: the last tile prefetches only its valid bytes.
        self.assertIn("tl.minimum(_BLOCK_SIZE_1, 1000 - offset_1) * 2", line)
        self.assertIn("tl.where(", line)  # rows past the 100th are skipped

    @skipIfCudaCapabilityLessThan((9, 0), reason="cp.async.bulk needs sm90")
    def test_full_row_bulk(self) -> None:
        x = torch.randn(64, 1000, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(row_sum_prefetched, (x,), block_sizes=[4])
        torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-2)
        (line,) = _prefetch_lines(code)
        self.assertIn(_BULK, line)
        self.assertIn("1000 * 2", line)

    @skipIfCudaCapabilityLessThan((9, 0), reason="cp.async.bulk needs sm90")
    def test_rolled_reduction_placement(self) -> None:
        x = torch.randn(64, 1000, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(
            row_sum_prefetched, (x,), block_sizes=[4], reduction_loops=[128]
        )
        torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-2)
        (line,) = _prefetch_lines(code)
        # The bulk prefetch covers the whole row once, ahead of the rolled loop ...
        self.assertIn(_BULK, line)
        self.assertLess(code.index(line), code.index("for roffset_1 in"))

        capability = mock.PropertyMock(return_value=(8, 0))
        with mock.patch.object(CompileEnvironment, "ptx_capability", capability):
            code, out = code_and_output(
                row_sum_prefetched, (x,), block_sizes=[4], reduction_loops=[128]
            )
        torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-2)
        (line,) = _prefetch_lines(code)
        # ... a per-element one covers each rolled chunk inside it.
        self.assertIn(_PER_ELEMENT, line)
        self.assertGreater(code.index(line), code.index("for roffset_1 in"))
        self.assertIn("rindex_1", line)

    @skipIfCudaCapabilityLessThan((9, 0), reason="cp.async.bulk needs sm90")
    def test_scalar_prefetch_issues_from_one_thread(self) -> None:
        x = torch.randn(1000, device=DEVICE)
        code, out = code_and_output(scale_prefetched, (x,), block_sizes=[256])
        torch.testing.assert_close(out, x * 2)
        (line,) = _prefetch_lines(code)
        self.assertIn(_BULK, line)
        self.assertIn("%tid.x", line)

    def test_gather_and_extra_mask_prefetch_per_element(self) -> None:
        x = torch.randn(50, 300, device=DEVICE)
        rows = torch.tensor([3, 7, 49, 0, 5], device=DEVICE)
        code, out = code_and_output(gather_prefetched, (x, rows), block_sizes=[4, 128])
        torch.testing.assert_close(out, x[rows])
        lines = _prefetch_lines(code)
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertIn(_PER_ELEMENT, line)
            self.assertNotIn(_BULK, line)

    def test_pre_sm90_and_non_ptx_targets(self) -> None:
        x = torch.randn(100, 1000, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        capability = mock.PropertyMock(return_value=(8, 0))
        with mock.patch.object(CompileEnvironment, "ptx_capability", capability):
            code, out = code_and_output(add_prefetched, (x, y), block_sizes=[8, 128])
        torch.testing.assert_close(out, x + y)
        (line,) = _prefetch_lines(code)
        self.assertIn(_PER_ELEMENT, line)
        self.assertNotIn(_BULK, line)

        capability = mock.PropertyMock(return_value=None)
        with mock.patch.object(CompileEnvironment, "ptx_capability", capability):
            code, out = code_and_output(add_prefetched, (x, y), block_sizes=[16, 64])
        torch.testing.assert_close(out, x + y)
        self.assertEqual(_prefetch_lines(code), [])

    def test_takes_no_config_slot(self) -> None:
        x = torch.randn(100, 1000, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        with_prefetch = add_prefetched.bind((x, y)).config_spec.default_config()
        without = add_plain.bind((x, y)).config_spec.default_config()
        self.assertEqual(dict(with_prefetch), dict(without))

    def test_prefetch_is_not_a_cross_loop_read(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        # Prefetching what an earlier loop writes is no dependency on it ...
        spec = independent_roots.bind((x, y)).config_spec
        self.assertIsNone(spec.cross_loop_pipeline)
        # ... and a megakernel whose loop also reads it keeps the prefetch.
        code, (tmp, out) = code_and_output(
            dependent_roots,
            (x, y),
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
        )
        torch.testing.assert_close(tmp, x + 1)
        torch.testing.assert_close(out, y * 2 + x + 1)
        self.assertEqual(len(_prefetch_lines(code)), 1)
        self.assertIn("tile_dependency", code)


if __name__ == "__main__":
    unittest.main()
