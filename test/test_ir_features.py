from __future__ import annotations

import json
import types
import unittest

import torch

from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfRefEager
from helion.autotuner import ir_features
from helion.autotuner.ir_features import extract_ir_graph

try:
    import networkx as nx

    _HAS_NETWORKX = True
except ImportError:  # networkx is not a declared Helion dependency
    nx = None  # type: ignore[assignment]
    _HAS_NETWORKX = False


def _node_link_graph(data: dict[str, object]) -> nx.DiGraph:
    """Load a node-link record across networkx versions."""
    try:
        return nx.node_link_graph(data, edges="links")  # networkx >= 3.4
    except TypeError:
        return nx.node_link_graph(data, link="links")  # networkx < 3.4


def _extract(kernel: object, args: tuple[object, ...], name: str) -> dict[str, object]:
    bound = kernel.bind(args)  # type: ignore[attr-defined]
    return extract_ir_graph(bound.host_function.device_ir)


class TestIrFeatures(TestCase):
    @skipIfRefEager("device IR is not built in ref eager mode")
    @unittest.skipUnless(_HAS_NETWORKX, "networkx not installed")
    def test_pointwise_dump_is_networkx_loadable(self) -> None:
        """A pointwise kernel yields a node-link DiGraph with data-only edges."""
        from examples.add import add

        g = _extract(
            add,
            (
                torch.randn(128, 128, device=DEVICE),
                torch.randn(128, 128, device=DEVICE),
            ),
            "add",
        )
        self.assertIs(g["directed"], True)
        self.assertIs(g["multigraph"], False)
        self.assertTrue(g["nodes"])
        self.assertTrue(g["links"])
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
        self.assertEqual(graph.number_of_edges(), len(g["links"]))

    @skipIfRefEager("device IR is not built in ref eager mode")
    def test_reduction_records_rolled_metadata_not_fake_edges(self) -> None:
        """Rolled reductions are captured as typed metadata, never as edges."""
        from examples.softmax import softmax

        g = _extract(softmax, (torch.randn(128, 128, device=DEVICE),), "softmax")
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
        region_edges = [e for e in g["links"] if e["edge_kind"] == "region"]
        self.assertEqual(region_edges, [])

    @skipIfRefEager("device IR is not built in ref eager mode")
    def test_control_flow_yields_resolved_region_edges(self) -> None:
        """An explicit _for_loop resolves live args into region edges."""
        from examples.attention import attention

        args = tuple(
            torch.randn(1, 2, 128, 64, device=DEVICE, dtype=torch.float16)
            for _ in range(3)
        )
        g = _extract(attention, args, "attention")
        # The body graph is a ForLoopGraphInfo invoked from the root.
        region_kinds = {gg["region_kind"] for gg in g["graph"]["graphs"]}
        self.assertIn("ForLoopGraphInfo", region_kinds)
        # Region edges must exist and resolve to real nodes in the union graph.
        region_edges = [e for e in g["links"] if e["edge_kind"] == "region"]
        self.assertTrue(region_edges, "attention _for_loop should yield region edges")
        node_ids = {n["id"] for n in g["nodes"]}
        for edge in region_edges:
            self.assertIn(edge["source"], node_ids)
            self.assertIn(edge["target"], node_ids)
            # A region edge always points into a child-graph placeholder.
            target = next(n for n in g["nodes"] if n["id"] == edge["target"])
            self.assertEqual(target["op_kind"], "placeholder")

    def test_unknown_region_kind_warns_once(self) -> None:
        """An undocumented region_kind warns (so upstream IR shifts are caught)."""
        ir_features._warn_unknown_region_kind.cache_clear()
        with self.assertLogs(ir_features.log, level="WARNING") as captured:
            ir_features._warn_unknown_region_kind("BrandNewGraphInfo")
        self.assertTrue(any("BrandNewGraphInfo" in m for m in captured.output))
        # Known kinds never warn.
        ir_features._warn_unknown_region_kind.cache_clear()
        with self.assertNoLogs(ir_features.log, level="WARNING"):
            ir_features._warn_unknown_region_kind("RootGraphInfo")


class TestIrFeaturesEdgeCases(TestCase):
    """Negative / degenerate cases for the extractor helpers."""

    def test_val_features_handles_missing_and_scalar_values(self) -> None:
        # No val -> every tensor field is None (never raises).
        none_feats = ir_features._val_features(None)
        self.assertTrue(all(v is None for v in none_feats.values()))
        # A non-tensor scalar records its type name and stringified value only.
        scalar = ir_features._val_features(7)
        self.assertEqual(scalar["dtype"], "int")
        self.assertEqual(scalar["value"], "7")
        self.assertIsNone(scalar["shape"])
        self.assertIsNone(scalar["concrete_shape"])

    def test_lowering_features_without_lowering_is_all_none(self) -> None:
        node = types.SimpleNamespace(meta={})  # no "lowering" key
        feats = ir_features._lowering_features(node)
        self.assertTrue(all(v is None for v in feats.values()))

    def test_target_str_is_stable_and_address_free(self) -> None:
        # String targets pass through (placeholder/output/get_attr).
        self.assertEqual(ir_features._target_str("output"), "output")
        # A function uses its qualified name, never an address (determinism).
        text = ir_features._target_str(extract_ir_graph)
        self.assertIn("extract_ir_graph", text)
        self.assertNotIn("0x", text)
        # An object with neither qualname nor name falls back to str().
        self.assertEqual(ir_features._target_str(5), "5")

    @unittest.skipUnless(_HAS_NETWORKX, "networkx not installed")
    def test_extract_handles_empty_device_ir(self) -> None:
        """A device IR with no graphs yields a valid, empty node-link record."""
        empty_ir = types.SimpleNamespace(graphs=[], root_ids=[], rolled_reductions=[])
        g = extract_ir_graph(empty_ir)  # type: ignore[arg-type]
        self.assertEqual(g["nodes"], [])
        self.assertEqual(g["links"], [])
        self.assertEqual(g["schema_version"], ir_features.IR_SCHEMA_VERSION)
        self.assertEqual(g["graph"]["num_graphs"], 0)
        # Still a loadable (empty) networkx DiGraph.
        graph = _node_link_graph(json.loads(json.dumps(g)))
        self.assertEqual(graph.number_of_nodes(), 0)
        self.assertTrue(graph.is_directed())

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
        """Large stringified values are truncated so they can't bloat the JSON."""
        long = "x" * (ir_features._MAX_VALUE_LEN + 50)
        capped = ir_features._truncate(long)
        self.assertLess(len(capped), len(long))
        self.assertTrue(capped.endswith("...<truncated>"))
        self.assertEqual(ir_features._truncate("short"), "short")
        # A non-tensor scalar value is capped too.
        feats = ir_features._val_features(long)
        self.assertTrue(feats["value"].endswith("...<truncated>"))

    def test_known_region_kinds_match_device_ir(self) -> None:
        """Drift guard: the known set must match the concrete GraphInfo subclasses.

        Fails loudly in CI if device_ir.py renames/adds a region kind, so the
        extractor (which only warns at runtime) is kept in sync deliberately.
        """
        from helion._compiler.device_ir import GraphInfo

        def concrete_names(cls: type) -> set[str]:
            names: set[str] = set()
            for sub in cls.__subclasses__():
                names |= concrete_names(sub)
                # Only count region kinds defined in device_ir.py itself.
                if sub.__module__ == GraphInfo.__module__:
                    names.add(sub.__name__)
            return names

        # NodeArgsGraphInfo is an abstract base (no `name`), never instantiated.
        actual = concrete_names(GraphInfo) - {"NodeArgsGraphInfo"}
        self.assertEqual(ir_features._KNOWN_REGION_KINDS, frozenset(actual))

    def test_malformed_control_flow_node_warns(self) -> None:
        """A control-flow target with bad args returns [] AND warns (not silent)."""
        from helion.language._tracing_ops import _for_loop

        ir_features._warn_malformed_region_spec.cache_clear()
        node = types.SimpleNamespace(target=_for_loop, args=())  # missing args
        with self.assertLogs(ir_features.log, level="WARNING") as captured:
            self.assertEqual(ir_features._region_specs(node), [])
        self.assertTrue(any("unexpected args" in m for m in captured.output))

    def test_non_control_flow_node_does_not_warn(self) -> None:
        """A normal (non-control-flow) node returns [] quietly, no warning."""

        def _plain() -> None:
            return None

        node = types.SimpleNamespace(target=_plain, args=())
        with self.assertNoLogs(ir_features.log, level="WARNING"):
            self.assertEqual(ir_features._region_specs(node), [])

    @skipIfRefEager("device IR is not built in ref eager mode")
    def test_ir_node_composes_val_and_lowering_keys(self) -> None:
        """The composed IrNode carries both val and lowering field keys."""
        from examples.add import add

        g = _extract(
            add,
            (
                torch.randn(64, 64, device=DEVICE),
                torch.randn(64, 64, device=DEVICE),
            ),
            "add",
        )
        node = g["nodes"][0]
        # _NodeCore + ValFeatures + LoweringFeatures keys are all present.
        for key in ("id", "op_kind", "dtype", "shape", "lowering_class"):
            self.assertIn(key, node)


if __name__ == "__main__":
    unittest.main()
