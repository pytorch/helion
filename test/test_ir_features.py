from __future__ import annotations

import functools
import json
import types
import unittest
from unittest.mock import patch

import torch

from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfRefEager
from helion.autotuner._metadata import ir_features
from helion.autotuner._metadata.ir_features import extract_ir_graph

# Reuse the extractor's single capability check (networkx>=3.4, i.e.
# node_link_data's `edges=` kwarg) so the skip gate and the producer agree on one
# definition of "usable networkx" instead of duplicating the probe.
_HAS_NETWORKX = ir_features._has_networkx_node_link()
if _HAS_NETWORKX:
    import networkx as nx
else:
    nx = None  # type: ignore[assignment]


def _node_link_graph(data: dict[str, object]) -> nx.DiGraph:
    """Load a node-link record using the supported networkx API."""
    return nx.node_link_graph(data, edges="edges")


def _extract(kernel: object, args: tuple[object, ...]) -> dict[str, object]:
    bound = kernel.bind(args)  # type: ignore[attr-defined]
    return extract_ir_graph(bound.host_function.device_ir)


class TestIrFeatures(TestCase):
    @skipIfRefEager("device IR is not built in ref eager mode")
    @unittest.skipUnless(_HAS_NETWORKX, "networkx>=3.4 required")
    def test_pointwise_dump_is_networkx_loadable(self) -> None:
        """A pointwise kernel yields a node-link DiGraph with data-only edges."""
        from examples.add import add

        g = _extract(
            add,
            (
                torch.randn(128, 128, device=DEVICE),
                torch.randn(128, 128, device=DEVICE),
            ),
        )
        self.assertIs(g["directed"], True)
        self.assertIs(g["multigraph"], False)
        self.assertTrue(g["nodes"])
        self.assertTrue(g["edges"])
        # Node ids are unique across the union of all graphs.
        ids = [n["id"] for n in g["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))
        # A pointwise node carries its lowering class and iteration ranges.
        pointwise = [
            n for n in g["nodes"] if n["lowering_class"] == "PointwiseLowering"
        ]
        self.assertTrue(pointwise)
        self.assertIsNotNone(pointwise[0]["pointwise_ranges"])
        # Reconstructs as a networkx DiGraph with no effort.
        graph = _node_link_graph(json.loads(json.dumps(g)))
        self.assertTrue(graph.is_directed())
        self.assertEqual(graph.number_of_nodes(), len(g["nodes"]))
        self.assertEqual(graph.number_of_edges(), len(g["edges"]))

    @skipIfRefEager("device IR is not built in ref eager mode")
    @unittest.skipUnless(_HAS_NETWORKX, "networkx>=3.4 required")
    def test_reduction_records_rolled_metadata_not_fake_edges(self) -> None:
        """Rolled reductions are captured as typed metadata, never as edges."""
        from examples.softmax import softmax

        g = _extract(softmax, (torch.randn(128, 128, device=DEVICE),))
        rolled = g["graph"]["rolled_reductions"]
        self.assertTrue(rolled, "softmax should record at least one rolled reduction")
        entry = rolled[0]
        self.assertIsInstance(entry["original_graph_id"], int)
        self.assertIsInstance(entry["rolled_block_ids"], list)
        self.assertIsInstance(entry["used_rdim"], bool)
        self.assertIsInstance(entry["can_be_rolled_by_caller"], bool)
        # A reduction node carries its reduction_type.
        reductions = [
            n for n in g["nodes"] if n["lowering_class"] == "ReductionLowering"
        ]
        self.assertTrue(reductions)
        self.assertIsNotNone(reductions[0]["reduction_type"])
        # We do NOT fabricate region edges for rolled reductions (no call node).
        region_edges = [e for e in g["edges"] if e["edge_kind"] == "region"]
        self.assertEqual(region_edges, [])

    @skipIfRefEager("device IR is not built in ref eager mode")
    @unittest.skipUnless(_HAS_NETWORKX, "networkx>=3.4 required")
    def test_control_flow_yields_resolved_region_edges(self) -> None:
        """An explicit _for_loop resolves live args into region edges."""
        from examples.attention import attention

        args = tuple(
            torch.randn(1, 2, 128, 64, device=DEVICE, dtype=torch.float16)
            for _ in range(3)
        )
        g = _extract(attention, args)
        # The body graph is a ForLoopGraphInfo invoked from the root.
        region_kinds = {gg["region_kind"] for gg in g["graph"]["graphs"]}
        self.assertIn("ForLoopGraphInfo", region_kinds)
        # Region edges must exist and resolve to real nodes in the union graph.
        region_edges = [e for e in g["edges"] if e["edge_kind"] == "region"]
        self.assertTrue(region_edges, "attention _for_loop should yield region edges")
        node_ids = {n["id"] for n in g["nodes"]}
        for edge in region_edges:
            self.assertIn(edge["source"], node_ids)
            self.assertIn(edge["target"], node_ids)
            # A region edge always points into a child-graph placeholder.
            target = next(n for n in g["nodes"] if n["id"] == edge["target"])
            self.assertEqual(target["op_kind"], "placeholder")

    @skipIfRefEager("device IR is not built in ref eager mode")
    @unittest.skipUnless(_HAS_NETWORKX, "networkx>=3.4 required")
    def test_symbolic_dims_not_specialized_to_concrete(self) -> None:
        """Symbolic dims yield concrete_shape None (not specialized to a value)."""
        from examples.add import add

        g = _extract(
            add,
            (
                torch.randn(128, 128, device=DEVICE),
                torch.randn(128, 128, device=DEVICE),
            ),
        )
        saw_symbolic = False
        for n in g["nodes"]:
            if n["shape"] is None:
                continue
            for sym, conc in zip(n["shape"], n["concrete_shape"], strict=True):
                if sym.lstrip("-").isdigit():
                    self.assertEqual(conc, int(sym))  # static dim -> its int
                else:
                    saw_symbolic = True
                    self.assertIsNone(conc)  # symbolic dim -> None (no specialize)
        self.assertTrue(saw_symbolic, "add should expose at least one symbolic dim")


class TestIrFeaturesEdgeCases(TestCase):
    """Negative / degenerate cases for the extractor helpers."""

    def test_target_str_is_stable_and_address_free(self) -> None:
        # String targets pass through (placeholder/output/get_attr).
        self.assertEqual(ir_features._target_str("output"), "output")
        # A function uses its qualified name, never an address (determinism).
        text = ir_features._target_str(extract_ir_graph)
        self.assertIn("extract_ir_graph", text)
        self.assertNotIn("0x", text)
        # functools.partial has no __qualname__/__name__; the str() fallback must
        # still be address-free (deterministic across processes).
        part = functools.partial(extract_ir_graph)
        self.assertNotIn("0x", ir_features._target_str(part))
        # An object with neither qualname nor name falls back to str().
        self.assertEqual(ir_features._target_str(5), "5")

    def test_concrete_dim_is_static_int_only(self) -> None:
        """Concrete only for a static int; non-int (SymInt proxy) -> None."""
        self.assertEqual(ir_features._concrete_dim(8), 8)
        self.assertIsNone(ir_features._concrete_dim("s0"))
        self.assertIsNone(ir_features._concrete_dim(None))

    def test_input_edges_preserve_multiplicity_and_nesting(self) -> None:
        """_input_edges walks fx containers and keeps every producer position."""
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        # a appears twice (direct + inside a nested list) -> two positions.
        node = graph.call_function(torch.add, (a, [a, b]))
        positions = ir_features._input_edges(node)
        self.assertEqual(positions[a], [0, 1])
        self.assertEqual(positions[b], [2])

    def test_value_and_source_loc_are_length_capped(self) -> None:
        """Large stringified values are truncated to exactly the cap."""
        long = "x" * (ir_features._MAX_VALUE_LEN + 50)
        capped = ir_features._truncate(long)
        # A truncated result lands exactly at the cap (suffix included).
        self.assertEqual(len(capped), ir_features._MAX_VALUE_LEN)
        self.assertTrue(capped.endswith(ir_features._TRUNCATION_SUFFIX))
        self.assertEqual(ir_features._truncate("short"), "short")
        # A non-tensor scalar value is capped too.
        feats = ir_features._val_features(long)
        self.assertTrue(feats["value"].endswith(ir_features._TRUNCATION_SUFFIX))

    def test_region_specs_if_resolves_both_branches(self) -> None:
        """_if maps (if_graph_id, if_args) and (else_graph_id, else_args)."""
        from helion.language._tracing_ops import _if

        node = types.SimpleNamespace(target=_if, args=("test", 1, 2, ["a"], ["b"]))
        self.assertEqual(ir_features._region_specs(node), [(1, ["a"]), (2, ["b"])])

    def test_region_specs_while_loop_with_and_without_orelse(self) -> None:
        """_while_loop maps body/cond (and orelse when present) to the same args."""
        from helion.language._tracing_ops import _while_loop

        no_orelse = types.SimpleNamespace(target=_while_loop, args=(3, 4, ["c"]))
        self.assertEqual(ir_features._region_specs(no_orelse), [(4, ["c"]), (3, ["c"])])
        with_orelse = types.SimpleNamespace(target=_while_loop, args=(3, 4, ["c"], 7))
        self.assertEqual(
            ir_features._region_specs(with_orelse),
            [(4, ["c"]), (3, ["c"]), (7, ["c"])],
        )

    def test_region_specs_for_loop_step_uses_args_index_3(self) -> None:
        """_for_loop_step(graph_id, begin, end, args, step) -> live args at index 3."""
        from helion.language._tracing_ops import _for_loop_step

        node = types.SimpleNamespace(
            target=_for_loop_step, args=(5, [0], [10], ["x"], [1])
        )
        self.assertEqual(ir_features._region_specs(node), [(5, ["x"])])

    @unittest.skipUnless(_HAS_NETWORKX, "networkx>=3.4 required")
    def test_networkx_version_gate_parses_release_and_prerelease(self) -> None:
        """The >=3.4 gate parses release and pre-release version strings (the
        leading-digit split must tolerate rc/dev/post suffixes)."""
        cases = {
            "3.6.1": True,
            "3.4": True,
            "3.4.2": True,
            "3.4rc1": True,
            "3.4.0.dev0": True,
            "3.4.post1": True,
            "3.3": False,
            "3.3.4": False,
            "2.5.1": False,
        }
        for version, expected in cases.items():
            with patch.object(nx, "__version__", version):
                self.assertEqual(
                    ir_features._has_networkx_node_link(), expected, version
                )


if __name__ == "__main__":
    unittest.main()
