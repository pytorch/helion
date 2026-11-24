"""Pure-AST tests of ``pipeline_inner_loads`` on a tile loop that holds
register statements beside its lane loop: the accumulator a V-fold starts
ahead of the lane loop and the reduction it ends with (``hoist_warp_reduce``),
as the transposed copy and sum over r of better-benchmark's
sum_3759dcabdc6f emits them.  No CUDA."""

from __future__ import annotations

import ast
import os
import textwrap
from unittest.mock import patch

import pytest

from helion._compiler.cute.pipeline_inner_loads import pipeline_inner_loads
from helion._testing import TestCase

pytestmark = pytest.mark.cpu_only

_LOAD = (
    "cute.arch.load(in0.iterator + cutlass.Int32(lane_base_2) * "
    "cutlass.Int32(in0.layout.stride[1]) if lane_base_2 < 128 else in0.iterator, "
    "ir.VectorType.get([16], cutlass.Uint16.mlir_type))"
)


def _tile_loop(before: str = "", after: str = "") -> str:
    return textwrap.dedent(
        f"""
        for tile_offset_2 in range(0, 128, 32):
            _helion_vfold_acc_0 = cutlass.Float32(0)
            {before}
            for lane_2 in range(1):
                lane_base_2 = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[1]) * 16
                _tile_unroll_vec_2_0 = {_LOAD}
                for vec_lane_2 in cutlass.range_constexpr(16):
                    load = cutlass.Uint16(_tile_unroll_vec_2_0[vec_lane_2]).bitcast(cutlass.BFloat16)
                    _helion_vfold_acc_0 = _helion_vfold_acc_0 + cutlass.Float32(load)
            sum_1 = cutlass.Float32(_cute_grouped_reduce_warp(_helion_vfold_acc_0, 'sum', cutlass.Float32(0), cutlass.Int32(cute.arch.thread_idx()[0]), pre=16, group_span=32))
            acc = acc + sum_1
            {after}
        """
    )


def _pipelined(src: str) -> str:
    body = pipeline_inner_loads(ast.parse(src).body)
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


class TestPipelineBesideLaneLoop(TestCase):
    def test_register_statements_beside_the_lane_loop_keep_the_pipeline(
        self,
    ) -> None:
        # The first tile's packet is issued ahead of the loop, each lane
        # takes the prefetched one and issues the next tile's; the V-fold's
        # accumulator and reduction stay where they were.
        out = _pipelined(_tile_loop())
        module = ast.parse(out)
        prologue = [ast.unparse(stmt) for stmt in module.body[:-1]]
        self.assertTrue(prologue[0].startswith("_pipe_lane_base_0 = "))
        self.assertTrue(prologue[1].startswith("_pipe_load_0 = cute.arch.load("))
        (tile_loop,) = [n for n in module.body if isinstance(n, ast.For)]
        texts = [ast.unparse(stmt) for stmt in tile_loop.body]
        self.assertEqual(texts[0], "_helion_vfold_acc_0 = cutlass.Float32(0)")
        self.assertTrue(texts[2].startswith("sum_1 = "))
        self.assertEqual(texts[3], "acc = acc + sum_1")
        lane = [ast.unparse(stmt) for stmt in tile_loop.body[1].body]
        self.assertEqual(lane[0], "lane_base_2 = _pipe_lane_base_0")
        self.assertEqual(lane[1], "_tile_unroll_vec_2_0 = _pipe_load_0")
        self.assertTrue(lane[2].startswith("_pipe_lane_base_0 = "))
        self.assertTrue(lane[3].startswith("_pipe_load_0 = cute.arch.load("))

    def test_the_carried_only_gate_sees_the_carried_write_beside_the_lane_loop(
        self,
    ) -> None:
        # HELION_LOAD_PIPELINE_CARRIED_ONLY pipelines a loop that carries a
        # scalar into the next tile: here ``acc``, written beside the lane
        # loop, not in it.
        src = "acc = cutlass.Float32(0.0)\n" + _tile_loop()
        with patch.dict(os.environ, {"HELION_LOAD_PIPELINE_CARRIED_ONLY": "1"}):
            self.assertIn("_pipe_load", _pipelined(src))

    def test_a_store_beside_the_lane_loop_keeps_the_loads_in_place(self) -> None:
        # The next tile's packet would be read before this statement wrote.
        src = _tile_loop(after="(out1.iterator + tile_offset_2).store(acc)")
        self.assertNotIn("_pipe_load", _pipelined(src))

    def test_a_sibling_that_writes_what_the_load_reads_keeps_it_in_place(
        self,
    ) -> None:
        # The prefetch computes the next tile's address before the
        # statement that rebinds a name of it runs.
        src = _tile_loop(before="in0 = other")
        self.assertNotIn("_pipe_load", _pipelined(src))

    def test_a_sibling_writing_an_alias_of_the_loads_base_keeps_it_in_place(
        self,
    ) -> None:
        # The pass runs before ast_rename: in0_1 becomes in0 there.
        src = _tile_loop(before="in0_1 = other")
        body = pipeline_inner_loads(ast.parse(src).body, rename_groups={"in0_1": "in0"})
        out = ast.unparse(ast.Module(body=body, type_ignores=[]))
        self.assertNotIn("_pipe_load", out)
        self.assertIn("_pipe_load", _pipelined(src))

    def test_a_sibling_writing_memory_or_the_base_by_tuple_keeps_it_in_place(
        self,
    ) -> None:
        # A subscript target writes a (register) tensor's element; a tuple or
        # annotated target binds each of its names.
        for before in ("acc_frag[0] = acc", "in0, other = pair", "in0: int = other"):
            self.assertNotIn("_pipe_load", _pipelined(_tile_loop(before=before)))
        self.assertIn("_pipe_load", _pipelined(_tile_loop(before="a, b = pair")))

    def test_a_sibling_call_with_effects_keeps_the_loads_in_place(self) -> None:
        # Only calls proven free of observable effects stand beside the lane
        # loop: an inline asm of declared purity, or an unknown helper, may
        # write what the next tile's prefetch reads.
        for before in (
            "t = _cute_inline_asm_elementwise('st.global.f32', acc, is_pure=False)",
            "t = some_helper(acc)",
        ):
            self.assertNotIn("_pipe_load", _pipelined(_tile_loop(before=before)))

    def test_a_second_loop_beside_the_lane_loop_keeps_the_loads_in_place(
        self,
    ) -> None:
        src = _tile_loop(after="for i in range(2):\n                acc = acc + i")
        self.assertNotIn("_pipe_load", _pipelined(src))
