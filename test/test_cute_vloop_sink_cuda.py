"""GPU numerics for grid vector-loop sinking (``cute_vloop_sink``).

Partial tiles on both axes, the interleaved (rows above columns) thread
layout, static shapes with several unroll depths, a max reduction and an
fp32 input; see ``test_cute_vloop_sink.py`` for the code-shape assertions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

import pytest
import torch

from test._cute_vloop_sink_kernels import FRAGMENT_REDUCE
from test._cute_vloop_sink_kernels import _sink_config
from test._cute_vloop_sink_kernels import col_reduce_max_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_from8_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_gather_then_atomic_static
from test._cute_vloop_sink_kernels import col_reduce_sum_pair_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_rescaled_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_static
from test._cute_vloop_sink_kernels import col_weighted_mean_dynamic

from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends

if TYPE_CHECKING:
    import helion

cutlass = pytest.importorskip("cutlass")


def _col_sum(t: torch.Tensor) -> torch.Tensor:
    return t.float().sum(0).to(t.dtype)


def _col_max(t: torch.Tensor) -> torch.Tensor:
    return t.float().amax(0).to(t.dtype)


@onlyBackends(["cute"])
class TestCuteVloopSink(TestCase):
    def _check(
        self,
        kernel: helion.Kernel,
        ref: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        **config: object,
    ) -> str:
        code, out = code_and_output(kernel, (x,), **config)  # pyrefly: ignore
        torch.testing.assert_close(out, ref(x), atol=2e-2, rtol=2e-2)
        # A second, different input must produce a different (correct) output.
        y = torch.randn_like(x) * 3
        _, out_y = code_and_output(kernel, (y,), **config)  # pyrefly: ignore
        torch.testing.assert_close(out_y, ref(y), atol=5e-2, rtol=2e-2)
        self.assertFalse(torch.allclose(out.float(), out_y.float()))
        self.assertIn("_vsink_vec", code)
        self.assertIn(FRAGMENT_REDUCE, code)
        return code

    def test_partial_tiles_on_both_axes(self) -> None:
        """1000 rows against a 4096-row block and 1000 columns against
        16-column blocks: row and column masks, V=4 and V=8."""
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        self._check(
            col_reduce_sum_dynamic,
            _col_sum,
            x,
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        self._check(
            col_reduce_sum_dynamic,
            _col_sum,
            x,
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[256, 2], vec=[1, 8], unroll=4
            ),
        )

    def test_rows_above_columns_partial_tiles(self) -> None:
        x = torch.randn(1000, 56, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            col_reduce_sum_dynamic,
            _col_sum,
            x,
            **_sink_config(
                block_sizes=[2048, 16],
                num_threads=[16, 8],
                vec=[1, 2],
                layouts=["blocked", "blocked"],
                unroll=4,
            ),
        )
        self.assertIn("pre=8, group_span=128", code)

    def test_static_shapes_and_unroll_depths(self) -> None:
        x = torch.randn(2048, 512, device=DEVICE, dtype=torch.bfloat16)
        for unroll in (1, 2, 16):
            self._check(
                col_reduce_sum_static,
                _col_sum,
                x,
                **_sink_config(
                    block_sizes=[2048, 16],
                    num_threads=[128, 2],
                    vec=[1, 8],
                    unroll=unroll,
                ),
            )

    def test_max_reduction(self) -> None:
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            col_reduce_max_dynamic,
            _col_max,
            x,
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        self.assertIn("'max', cutlass.Float32(float('-inf'))", code)
        # The sunk per-lane combine keeps torch.amax's NaN propagation.
        x[17, 5] = float("nan")
        x[900, 640] = float("nan")
        _, out = code_and_output(
            col_reduce_max_dynamic,
            (x,),
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        torch.testing.assert_close(out, _col_max(x), equal_nan=True)

    def test_float32_input(self) -> None:
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.float32)
        code = self._check(
            col_reduce_sum_dynamic,
            _col_sum,
            x,
            **_sink_config(
                block_sizes=[4096, 8], num_threads=[128, 2], vec=[1, 4], unroll=4
            ),
        )
        self.assertIn("ir.VectorType.get([4], cutlass.Uint32.mlir_type)", code)

    def test_nonzero_grid_origin_declines_and_stays_correct(self) -> None:
        """``hl.tile(8, n)`` keeps per-element column masks: the pass declines
        (no out-of-bounds vector transaction) and the knob-off code runs."""
        x = torch.randn(1000, 1008, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            col_reduce_sum_from8_dynamic,
            (x,),
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        expected = torch.zeros_like(out)
        expected[8:] = _col_sum(x[:, 8:])
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        self.assertNotIn("_vsink", code)
        self.assertNotIn(FRAGMENT_REDUCE, code)

    def test_a_misaligned_input_view_declines_and_stays_correct(self) -> None:
        """A view whose base is four bytes into its allocation cannot take the
        8-byte sunk packet: the pass declines (it shares the tile hoist's base
        proof) and the knob-off code runs instead of faulting on a misaligned
        address."""
        x = torch.randn(4096, 4112, device=DEVICE, dtype=torch.bfloat16)[:, 2:4098]
        self.assertEqual(x.data_ptr() % 8, 4)
        config = _sink_config(
            block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
        )
        code, out = code_and_output(col_reduce_sum_static, (x,), **config)
        code_off, out_off = code_and_output(
            col_reduce_sum_static, (x,), **{**config, "cute_vloop_sink": False}
        )
        torch.testing.assert_close(out, _col_sum(x), atol=2e-2, rtol=2e-2)
        self.assertEqual(code, code_off)
        self.assertTrue(torch.equal(out, out_off))
        self.assertNotIn("_vsink", code)

    def test_row_weighted_mean(self) -> None:
        """A row-only weight load and a row-only sum next to the column sum:
        both become per-lane state; partial tiles on both axes."""
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        w = torch.rand(1000, device=DEVICE, dtype=torch.bfloat16) + 0.5
        code, out = code_and_output(
            col_weighted_mean_dynamic,
            (x, w),
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        expected = (x.float() * w.float()[:, None]).sum(0) / w.float().sum()
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        self.assertIn("_vsink_vec", code)
        self.assertEqual(code.count(FRAGMENT_REDUCE), 2)
        self.assertIn("wsum_frag", code)

    def test_lane_invariant_scalar_redefined_by_an_if(self) -> None:
        """``s = scale[0]`` feeds one store, an ``if flag[0]`` doubles ``s``
        and a second store uses it: with the flag set, every column of both
        outputs must see one ``scale`` (not a value compounded over the V
        lanes), and with it clear the second output equals the first."""
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        scale = torch.full((1,), 1.5, device=DEVICE, dtype=torch.float32)
        column_sum = x.float().sum(0)
        for flag_value, factor in ((1, 2.0), (0, 1.0)):
            flag = torch.full((1,), flag_value, device=DEVICE, dtype=torch.int32)
            code, (out1, out2) = code_and_output(
                col_reduce_sum_rescaled_dynamic,
                (x, scale, flag),
                **_sink_config(
                    block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
                ),
            )
            torch.testing.assert_close(out1, column_sum * 1.5, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(
                out2, column_sum * (1.5 * factor), atol=2e-2, rtol=2e-2
            )
            self.assertIn("_vsink_vec", code)
            self.assertIn(FRAGMENT_REDUCE, code)

    def test_a_gathered_value_is_read_before_the_atomic(self) -> None:
        """Each lane gathers its own element of ``count`` and then bumps it,
        unguarded (static shapes, one thread along the rows), so one thread
        alone touches that element: the gathered value is the initial 2.0,
        never the 3.0 the atomic leaves behind, with the knob on as with it
        off.  Recomputing the gather after the row loop read 3.0."""
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        outputs: dict[bool, torch.Tensor] = {}
        for sink in (False, True):
            count = torch.full((1024,), 2.0, device=DEVICE, dtype=torch.float32)
            code, out = code_and_output(
                col_reduce_sum_gather_then_atomic_static,
                (x, count),
                **_sink_config(
                    block_sizes=[1024, 16],
                    num_threads=[1, 4],
                    vec=[1, 4],
                    unroll=8,
                    sink=sink,
                ),
            )
            self.assertEqual("_vsink_vec" in code, sink)
            torch.testing.assert_close(count, torch.full_like(count, 3.0))
            before = out - x.float().sum(0)
            torch.testing.assert_close(
                before, torch.full_like(before, 2.0), atol=5e-2, rtol=0
            )
            outputs[sink] = out
        torch.testing.assert_close(outputs[True], outputs[False])

    def test_sibling_row_loops(self) -> None:
        x = torch.randn(1000, 1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1000, 1024, device=DEVICE, dtype=torch.bfloat16) * 2
        code, out = code_and_output(
            col_reduce_sum_pair_dynamic,
            (x, y),
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        expected = (x.float().sum(0) + y.float().sum(0)).to(torch.bfloat16)
        torch.testing.assert_close(out, expected, atol=5e-2, rtol=2e-2)
        # Each row loop strength-reduces and vectorizes its own tensor.
        self.assertIn("_vsink_ptr = x.iterator", code)
        self.assertIn("_vsink_ptr_1 = y.iterator", code)
        self.assertEqual(code.count(FRAGMENT_REDUCE), 2)
