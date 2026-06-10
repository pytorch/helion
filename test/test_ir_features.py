from __future__ import annotations

import json
import unittest

import networkx as nx
import torch

from helion._testing import DEVICE
from helion._testing import TestCase
from helion.autotuner import ir_features
from helion.autotuner.ir_features import extract_ir_graph


def _extract(kernel: object, args: tuple[object, ...], name: str) -> dict[str, object]:
    bound = kernel.bind(args)  # type: ignore[attr-defined]
    return extract_ir_graph(
        bound.host_function.device_ir,
        run_id="rid",
        kernel_id="kid",
        kernel_name=name,
        input_shapes=str([tuple(a.shape) for a in args if hasattr(a, "shape")]),
    )


class TestIrFeatures(TestCase):
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
        graph = nx.node_link_graph(json.loads(json.dumps(g)), edges="links")
        self.assertTrue(graph.is_directed())
        self.assertEqual(graph.number_of_nodes(), len(g["nodes"]))
        self.assertEqual(graph.number_of_edges(), len(g["links"]))

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
        ir_features._warned_region_kinds.clear()
        with self.assertLogs(ir_features.log, level="WARNING") as captured:
            ir_features._warn_unknown_region_kind("BrandNewGraphInfo")
        self.assertTrue(any("BrandNewGraphInfo" in m for m in captured.output))
        # Known kinds never warn.
        ir_features._warned_region_kinds.clear()
        with self.assertNoLogs(ir_features.log, level="WARNING"):
            ir_features._warn_unknown_region_kind("RootGraphInfo")


if __name__ == "__main__":
    unittest.main()
