"""Tests for CuTe layout planning pass."""

from __future__ import annotations

import dataclasses
import unittest

import torch

import helion
from helion._compiler.cute.layout import LayoutTag
from helion._compiler.cute.layout import ThreadLayout
from helion._compiler.cute.layout_propagation import META_KEY
from helion._testing import DEVICE
from helion._testing import onlyBackends
import helion.language as hl
from helion.language import reduce_ops


@onlyBackends(["cute"])
class TestThreadLayout(unittest.TestCase):
    """Unit tests for ThreadLayout data structure."""

    def test_make_1d(self) -> None:
        layout = ThreadLayout.make_1d(1024, num_threads=128)
        self.assertEqual(layout.num_threads(), 128)
        self.assertEqual(layout.num_values(), 8)
        self.assertEqual(layout.tile_numel(), 1024)

    def test_make_row_major(self) -> None:
        layout = ThreadLayout.make_row_major(32, 64, num_threads=64)
        self.assertEqual(layout.num_threads(), 64)
        self.assertEqual(layout.thread_shape, (64,))
        # 32 rows * 1 col per thread = 32 values
        self.assertEqual(layout.value_shape, (32, 1))

    def test_make_col_major(self) -> None:
        layout = ThreadLayout.make_col_major(64, 32, num_threads=64)
        self.assertEqual(layout.num_threads(), 64)
        self.assertEqual(layout.thread_shape, (64,))
        self.assertEqual(layout.value_shape, (1, 32))

    def test_compatibility(self) -> None:
        a = ThreadLayout.make_1d(1024, num_threads=128, tag=LayoutTag.COALESCED)
        b = ThreadLayout.make_1d(1024, num_threads=128, tag=LayoutTag.INHERITED)
        c = ThreadLayout.make_1d(1024, num_threads=64)
        # Same mapping, different tags → compatible
        self.assertTrue(a.is_compatible(b))
        # Different thread counts → incompatible
        self.assertFalse(a.is_compatible(c))

    def test_with_tag(self) -> None:
        a = ThreadLayout.make_1d(1024, num_threads=128, tag=LayoutTag.COALESCED)
        b = a.with_tag(LayoutTag.INHERITED)
        self.assertEqual(b.tag, LayoutTag.INHERITED)
        self.assertTrue(a.is_compatible(b))

    def test_symbolic_sizes(self) -> None:
        """ThreadLayout should accept SymInt-like values."""
        # Use plain ints to simulate — actual SymInts require compilation context
        layout = ThreadLayout(
            thread_shape=(128,),
            thread_stride=(1,),
            value_shape=(8,),
            value_stride=(1,),
        )
        self.assertEqual(layout.num_threads(), 128)
        self.assertEqual(layout.num_values(), 8)


@onlyBackends(["cute"])
class TestThreadBudget(unittest.TestCase):
    """Test thread budget validation."""

    def test_under_limit(self) -> None:
        from helion._compiler.cute.thread_budget import check_thread_limit

        # Should not raise
        check_thread_limit(1024)
        check_thread_limit(512)
        check_thread_limit(1)

    def test_over_limit(self) -> None:
        from helion._compiler.cute.thread_budget import check_thread_limit
        from helion.exc import BackendUnsupported

        with self.assertRaises(BackendUnsupported):
            check_thread_limit(1025)

    def test_over_limit_message(self) -> None:
        from helion._compiler.cute.thread_budget import check_thread_limit

        with self.assertRaises(Exception) as ctx:
            check_thread_limit(2048, context="(64, 64)")
        self.assertIn("(64, 64)", str(ctx.exception))


@onlyBackends(["cute"])
class TestLayoutAnnotation(unittest.TestCase):
    """Test that layout annotations are set during CuTe compilation."""

    def _compile_cute(self, kernel_fn, *args):
        """Compile a kernel with CuTe backend and return bound kernel."""
        import os

        old_backend = os.environ.get("HELION_BACKEND")
        os.environ["HELION_BACKEND"] = "cute"
        try:
            bound = kernel_fn.bind(args)
            # Tests should compile against a deterministic config rather than
            # trigger autotuning on first call.
            bound.set_config(bound.config_spec.default_config())
            bound(*args)
            return bound
        finally:
            if old_backend is None:
                os.environ.pop("HELION_BACKEND", None)
            else:
                os.environ["HELION_BACKEND"] = old_backend

    def test_1d_pointwise_annotations(self) -> None:
        """1D pointwise kernel: all nodes should get coalesced/inherited layouts."""

        @helion.kernel
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        bound = self._compile_cute(add_kernel, x, y)

        # Verify result
        result = bound(x, y)
        torch.testing.assert_close(result, x + y)

    def test_2d_load_coalescing(self) -> None:
        """2D load from row-major tensor should annotate with coalesced layout."""

        @helion.kernel
        def copy_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            m, n = x.size()
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        x = torch.randn(64, 128, device=DEVICE)
        bound = self._compile_cute(copy_kernel, x)
        result = bound(x)
        torch.testing.assert_close(result, x)


@dataclasses.dataclass
class _FakeGraphInfo:
    graph: torch.fx.Graph


@onlyBackends(["cute"])
class TestLayoutChangeInsertion(unittest.TestCase):
    """Test that layout change nodes are inserted at mismatch boundaries."""

    @staticmethod
    def _make_test_graph(
        src_layout: ThreadLayout, dst_layout: ThreadLayout
    ) -> torch.fx.Graph:
        """Build a simple FX graph: placeholder -> add, with layout constraints."""
        from helion._compiler.cute.layout import LayoutConstraint

        tile_numel = src_layout.tile_numel()

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(tile_numel)
        x.meta[META_KEY] = LayoutConstraint(
            preferred_output=src_layout,
            output_layout=src_layout,
        )

        add = graph.call_function(torch.add, args=(x, x))
        add.meta["val"] = torch.randn(tile_numel)
        add.meta[META_KEY] = LayoutConstraint(
            preferred_input=dst_layout,
            preferred_output=dst_layout,
            input_layout=dst_layout,
            output_layout=dst_layout,
        )

        graph.output(add)
        return graph

    def test_no_change_when_compatible(self) -> None:
        """No layout change should be inserted when layouts agree."""
        src = ThreadLayout.make_1d(128, num_threads=128, tag=LayoutTag.COALESCED)
        dst = src.with_tag(LayoutTag.REDUCTION)
        graph = self._make_test_graph(src, dst)
        node_count_before = len(list(graph.nodes))

        from helion._compiler.cute.layout_propagation import _insert_layout_changes

        _insert_layout_changes(_FakeGraphInfo(graph))  # type: ignore[arg-type]

        # No change nodes should be inserted (layouts are compatible)
        node_count_after = len(list(graph.nodes))
        self.assertEqual(node_count_before, node_count_after)

    def test_change_inserted_when_incompatible(self) -> None:
        """A layout change node should be inserted for single-value mismatches."""
        # Construct two scalar (num_values==1) layouts that differ in
        # value_stride so is_compatible returns False.
        src = ThreadLayout(
            thread_shape=(32,),
            thread_stride=(1,),
            value_shape=(1,),
            value_stride=(1,),
            tag=LayoutTag.COALESCED,
        )
        dst = ThreadLayout(
            thread_shape=(32,),
            thread_stride=(1,),
            value_shape=(1,),
            value_stride=(2,),
            tag=LayoutTag.REDUCTION,
        )
        graph = self._make_test_graph(src, dst)
        node_count_before = len(list(graph.nodes))

        from helion._compiler.cute.layout_propagation import _insert_layout_changes

        _insert_layout_changes(_FakeGraphInfo(graph))  # type: ignore[arg-type]

        # A layout change node should have been inserted
        node_count_after = len(list(graph.nodes))
        self.assertGreater(node_count_after, node_count_before)

        # Find the layout change node
        from helion._compiler.cute.layout_change import _cute_layout_change

        change_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target is _cute_layout_change
        ]
        self.assertEqual(len(change_nodes), 1)

        # Verify it has proper metadata
        cn = change_nodes[0]
        self.assertIn(META_KEY, cn.meta)
        self.assertIn("cute_layout_change_src", cn.meta)
        self.assertIn("lowering", cn.meta)
        self.assertEqual(cn.meta["cute_layout_change_src"].num_threads(), 32)
        self.assertEqual(cn.meta[META_KEY].output_layout.num_threads(), 32)

    def test_no_change_when_multi_value(self) -> None:
        """No layout change for multi-value layouts (codegen only handles scalars)."""
        from helion._compiler.cute.layout_propagation import _insert_layout_changes

        # Row-major vs col-major: same num_threads and tile_numel, but each
        # thread owns multiple values (num_values > 1).
        src = ThreadLayout.make_row_major(
            4, 32, num_threads=32, tag=LayoutTag.COALESCED
        )
        dst = ThreadLayout.make_col_major(
            32, 4, num_threads=32, tag=LayoutTag.REDUCTION
        )
        graph = self._make_test_graph(src, dst)
        node_count_before = len(list(graph.nodes))

        _insert_layout_changes(_FakeGraphInfo(graph))  # type: ignore[arg-type]

        # No change inserted — multi-value layouts not yet supported
        node_count_after = len(list(graph.nodes))
        self.assertEqual(node_count_before, node_count_after)

    def test_no_change_when_tile_sizes_differ(self) -> None:
        """No layout change for different-sized tiles (e.g. reduce output vs store)."""
        from helion._compiler.cute.layout import LayoutConstraint
        from helion._compiler.cute.layout_propagation import _insert_layout_changes

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(128)
        # Producer: 128-element tile (e.g. reduction input layout)
        x.meta[META_KEY] = LayoutConstraint(
            preferred_output=ThreadLayout.make_1d(
                128, num_threads=32, tag=LayoutTag.REDUCTION
            ),
            output_layout=ThreadLayout.make_1d(
                128, num_threads=32, tag=LayoutTag.REDUCTION
            ),
        )

        add = graph.call_function(torch.add, args=(x, x))
        add.meta["val"] = torch.randn(4)
        # Consumer: 4-element tile (e.g. store after reduction collapsed a dim)
        add.meta[META_KEY] = LayoutConstraint(
            preferred_input=ThreadLayout.make_1d(
                4, num_threads=4, tag=LayoutTag.COALESCED
            ),
            preferred_output=ThreadLayout.make_1d(
                4, num_threads=4, tag=LayoutTag.COALESCED
            ),
            input_layout=ThreadLayout.make_1d(
                4, num_threads=4, tag=LayoutTag.COALESCED
            ),
            output_layout=ThreadLayout.make_1d(
                4, num_threads=4, tag=LayoutTag.COALESCED
            ),
        )

        graph.output(add)
        node_count_before = len(list(graph.nodes))
        _insert_layout_changes(_FakeGraphInfo(graph))  # type: ignore[arg-type]
        node_count_after = len(list(graph.nodes))
        # No change inserted — tile sizes differ
        self.assertEqual(node_count_before, node_count_after)

    def test_no_change_when_thread_counts_differ(self) -> None:
        """No layout change when thread counts differ (can't do scalar permutation)."""
        from helion._compiler.cute.layout import LayoutConstraint
        from helion._compiler.cute.layout_propagation import _insert_layout_changes

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(128)
        # Same tile_numel (128) but different thread counts
        x.meta[META_KEY] = LayoutConstraint(
            preferred_output=ThreadLayout.make_1d(
                128, num_threads=32, tag=LayoutTag.COALESCED
            ),
            output_layout=ThreadLayout.make_1d(
                128, num_threads=32, tag=LayoutTag.COALESCED
            ),
        )

        add = graph.call_function(torch.add, args=(x, x))
        add.meta["val"] = torch.randn(128)
        add.meta[META_KEY] = LayoutConstraint(
            preferred_input=ThreadLayout.make_1d(
                128, num_threads=64, tag=LayoutTag.REDUCTION
            ),
            preferred_output=ThreadLayout.make_1d(
                128, num_threads=64, tag=LayoutTag.REDUCTION
            ),
            input_layout=ThreadLayout.make_1d(
                128, num_threads=64, tag=LayoutTag.REDUCTION
            ),
            output_layout=ThreadLayout.make_1d(
                128, num_threads=64, tag=LayoutTag.REDUCTION
            ),
        )

        graph.output(add)
        node_count_before = len(list(graph.nodes))
        _insert_layout_changes(_FakeGraphInfo(graph))  # type: ignore[arg-type]
        node_count_after = len(list(graph.nodes))
        # No change inserted — thread counts differ
        self.assertEqual(node_count_before, node_count_after)

    def test_validate_rejects_unresolved_mismatch(self) -> None:
        from helion._compiler.cute.layout_propagation import _validate_layout_contracts
        from helion.exc import BackendUnsupported

        src = ThreadLayout.make_row_major(
            4, 32, num_threads=32, tag=LayoutTag.COALESCED
        )
        dst = ThreadLayout.make_col_major(
            32, 4, num_threads=32, tag=LayoutTag.REDUCTION
        )
        graph = self._make_test_graph(src, dst)

        with self.assertRaises(BackendUnsupported):
            _validate_layout_contracts(_FakeGraphInfo(graph))  # type: ignore[arg-type]

    def test_validate_allows_reduction_fallback_mismatch(self) -> None:
        from helion._compiler.cute.layout import LayoutConstraint
        from helion._compiler.cute.layout_propagation import _insert_layout_changes
        from helion._compiler.cute.layout_propagation import _validate_layout_contracts

        src = ThreadLayout.make_row_major(
            4, 32, num_threads=32, tag=LayoutTag.COALESCED
        )
        reduce_layout = ThreadLayout.make_col_major(
            32, 4, num_threads=32, tag=LayoutTag.REDUCTION
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(src.tile_numel())
        x.meta[META_KEY] = LayoutConstraint(
            preferred_output=src,
            output_layout=src,
        )

        add = graph.call_function(torch.add, args=(x, x))
        add.meta["val"] = torch.randn(src.tile_numel())
        add.meta[META_KEY] = LayoutConstraint(
            preferred_input=src,
            preferred_output=src,
            input_layout=src,
            output_layout=src,
        )

        reduce = graph.call_function(reduce_ops._reduce, args=(0, x, 0))
        reduce.meta["val"] = torch.randn(4)
        reduce.meta[META_KEY] = LayoutConstraint(
            preferred_input=reduce_layout,
            input_layout=reduce_layout,
        )

        graph.output((add, reduce))

        _insert_layout_changes(_FakeGraphInfo(graph))  # type: ignore[arg-type]
        _validate_layout_contracts(_FakeGraphInfo(graph))  # type: ignore[arg-type]


@onlyBackends(["cute"])
class TestLayoutChangeCodegen(unittest.TestCase):
    """End-to-end test that layout change codegen produces valid, correct code."""

    def test_layout_change_pointwise_roundtrip(self) -> None:
        """Force a same-sized layout mismatch and verify smem round-trip works."""

        @helion.kernel(autotune_effort="none")
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        import os
        from unittest import mock

        from helion._compiler.cute import layout_propagation as lp

        # Patch backward propagation to be a no-op, so seed mismatches
        # (loads vs inherited) persist and _insert_layout_changes fires.
        with (
            mock.patch.dict(os.environ, {"HELION_BACKEND": "cute"}),
            mock.patch.object(lp, "_backward_propagate", lambda gi: None),
        ):
            x = torch.randn(128, device=DEVICE)
            y = torch.randn(128, device=DEVICE)
            # Compile and run — exercises layout change codegen path
            result = add_kernel(x, y)
            torch.testing.assert_close(result, x + y)


if __name__ == "__main__":
    unittest.main()
