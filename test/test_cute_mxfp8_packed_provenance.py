"""CPU proof tests for packed scale extraction and broadcast direction."""

from __future__ import annotations

import unittest

import pytest
import torch

from test.test_cute_mxfp8_mma import _config
from test.test_cute_mxfp8_mma import _cpu_target
from test.test_cute_mxfp8_mma import _wrapper_plan

import helion
from helion._compiler.cute.mxfp8_mma import _scale
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops


def example(
    *,
    groups: int = 12,
    start: int = 0,
    word_divisor: int = 4,
    byte_multiplier: int = 8,
    mask: int = 255,
    shift_shape: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.fx.Node]:
    graph = torch.fx.Graph()
    packed = torch.empty((5, (groups + 3) // 4), dtype=torch.int32)
    host = graph.call_function(_tracing_ops._host_tensor, ("scales",))
    host.meta["val"] = packed
    index = graph.call_function(
        torch.ops.aten.arange.start, (start, groups + start), {"dtype": torch.int64}
    )
    index.meta["val"] = torch.arange(start, groups + start)
    column = graph.call_function(
        torch.ops.aten.div.Tensor_mode,
        (index, word_divisor),
        {"rounding_mode": "floor"},
    )
    column.meta["val"] = torch.empty((groups,), dtype=torch.int64)
    words = graph.call_function(memory_ops.load, (host, ["rows", column], None, None))
    words.meta["val"] = torch.empty((5, groups), dtype=torch.int32)
    byte = graph.call_function(torch.ops.aten.remainder.Scalar, (index, 4))
    byte.meta["val"] = torch.empty((groups,), dtype=torch.int64)
    shift = graph.call_function(torch.ops.aten.mul.Tensor, (byte, byte_multiplier))
    shift.meta["val"] = torch.empty(shift_shape or (groups,), dtype=torch.int64)
    shifted = graph.call_function(
        torch.ops.aten.bitwise_right_shift.Tensor, (words, shift)
    )
    shifted.meta["val"] = torch.empty((5, groups), dtype=torch.int32)
    masked = graph.call_function(torch.ops.aten.bitwise_and.Scalar, (shifted, mask))
    masked.meta["val"] = torch.empty((5, groups), dtype=torch.int32)
    result = graph.call_function(
        torch.ops.prims.convert_element_type.default, (masked, torch.uint8)
    )
    result.meta["val"] = torch.empty((5, groups), dtype=torch.uint8)
    return packed, result


class TestPackedScaleProvenance(unittest.TestCase):
    def test_correct_extraction(self) -> None:
        packed, result = example()
        self.assertIs(_scale(result, "rows", 12), packed)

    def test_broadcasted_extraction(self) -> None:
        packed, result = example(shift_shape=(1, 12))
        self.assertIs(_scale(result, "rows", 12), packed)

    def test_reject_wrong_word_divisor(self) -> None:
        _, result = example(word_divisor=2)
        self.assertIsNone(_scale(result, "rows", 12))

    def test_reject_wrong_byte_shift(self) -> None:
        _, result = example(byte_multiplier=4)
        self.assertIsNone(_scale(result, "rows", 12))

    def test_reject_inexact_mask(self) -> None:
        _, result = example(mask=127)
        self.assertIsNone(_scale(result, "rows", 12))

    def test_reject_shift_along_rows(self) -> None:
        _, result = example(shift_shape=(12, 1))
        self.assertIsNone(_scale(result, "rows", 12))

    def test_reject_displaced_iota(self) -> None:
        _, result = example(start=1)
        self.assertIsNone(_scale(result, "rows", 12))

    def test_reject_wrong_contract_extent(self) -> None:
        _, result = example()
        self.assertIsNone(_scale(result, "rows", 16))

    def test_reject_overflow_cancelled_index(self) -> None:
        _, result = example()
        shifted = result.args[0].args[0]
        words = shifted.args[0]
        column = words.args[1][1]
        graph = result.graph
        with graph.inserting_before(words):
            overflowing = graph.call_function(
                torch.ops.aten.mul.Tensor, (column, 1 << 32)
            )
            overflowing.meta["val"] = torch.empty((12,), dtype=torch.int32)
            restored = graph.call_function(
                torch.ops.aten.div.Tensor_mode,
                (overflowing, 1 << 32),
                {"rounding_mode": "floor"},
            )
            restored.meta["val"] = torch.empty((12,), dtype=torch.int32)
        words.args = (words.args[0], ["rows", restored], None, None)
        self.assertIsNone(_scale(result, "rows", 12))

    def symbolic_shift(
        self, result: torch.fx.Node, *, rows: bool = False
    ) -> torch.SymInt:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        extent = shape_env.create_unbacked_symint()
        with FakeTensorMode(shape_env=shape_env):
            metadata = torch.empty(
                (extent, 1) if rows else (1, extent), dtype=torch.int32
            )
        shifted = result.args[0].args[0]
        shifted.args[1].meta["val"] = metadata
        return extent

    def test_symbolic_full_group_shift(self) -> None:
        packed, result = example()
        self.symbolic_shift(result)
        self.assertIs(_scale(result, "rows", 12), packed)

    def test_reject_symbolic_row_shift_without_guard(self) -> None:
        _, result = example()
        self.symbolic_shift(result, rows=True)
        self.assertIsNone(_scale(result, "rows", 12))

    def test_reject_inexact_extent_with_symbolic_shift(self) -> None:
        _, result = example()
        self.symbolic_shift(result)
        self.assertIsNone(_scale(result, "rows", 16))

    def test_reject_rank_three_shift(self) -> None:
        _, result = example(shift_shape=(1, 1, 12))
        self.assertIsNone(_scale(result, "rows", 12))


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    cute_full_slice_matmul_tiling=True,
    cute_segmented_matmul_tiling=True,
    cute_materialize_transformed_operands=True,
)
def _packed_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(0)
    for row, col in hl.tile([m, n]):
        group = hl.arange(k // 32)
        left_words = hl.load(sa, [row, group // 4])
        left_scale = ((left_words >> ((group % 4) * 8).unsqueeze(0)) & 255).to(
            torch.uint8
        )
        right_words = hl.load(sb, [col, group // 4])
        right_scale = ((right_words >> ((group % 4) * 8).unsqueeze(0)) & 255).to(
            torch.uint8
        )
        out[row, col] = hl.dot_scaled(
            a[row, :], left_scale, "e4m3", b[col, :].T, right_scale, "e4m3"
        ).to(out.dtype)
    return out


@pytest.mark.parametrize(
    "m,n,k", [(1, 129, 32), (7, 257, 160), (16, 41, 288), (73, 273, 512)]
)
@skipUnlessBackends(["cute"])
def test_cpu_packed_source_admits_and_preserves_native_geometry(
    m: int, n: int, k: int
) -> None:
    initialized = torch.cuda.is_initialized()
    values = (
        torch.empty((m, k), dtype=torch.float8_e4m3fn),
        torch.empty((n, k), dtype=torch.float8_e4m3fn),
        torch.empty_strided(
            (m, (k + 127) // 128), (1, (m + 3) // 4 * 4), dtype=torch.int32
        ),
        torch.empty_strided(
            (n, (k + 127) // 128), (1, (n + 3) // 4 * 4), dtype=torch.int32
        ),
        torch.empty((m, n), dtype=torch.bfloat16),
    )
    with _cpu_target():
        bound = _packed_mxfp8._bind_isolated(values)
        assert bound.config_spec.cute_mxfp8_mma_available
        assert bound.config_spec.cute_mxfp8_fused_scales_available
        config = _config(
            bound,
            cute_scaled_fused_scales=True,
            cute_scaled_tile_n=32,
            cute_scaled_tile_m=64,
            cute_scaled_compact_m=True,
            cute_scaled_swap_ab=True,
        )
        with bound.env:
            generation = bound.config_spec.create_config_generation()
            restored = generation.unflatten(generation.flatten(config))
        plan = _wrapper_plan(bound.to_code(restored))
        assert (plan["m"], plan["n"], plan["k"]) == (m, n, k)
        assert plan["fused_scales"] is True and plan["swap_ab"] is True
        assert plan["logical_m_per_cta"] == 64 and plan["compact_m"] is True
    assert torch.cuda.is_initialized() is initialized
