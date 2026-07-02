from __future__ import annotations

import functools
import json
import unittest

import torch

from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfRefEager
from helion.autotuner._metadata import ir_features
from helion.autotuner._metadata.ir_features import extract_ir_graph

_HAS_NETWORKX = ir_features._has_networkx_node_link()
if _HAS_NETWORKX:
    import networkx as nx
else:
    nx = None  # type: ignore[assignment]


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
        ids = [n["id"] for n in g["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))
        pointwise = [
            n for n in g["nodes"] if n["lowering_class"] == "PointwiseLowering"
        ]
        self.assertTrue(pointwise)
        self.assertIsNotNone(pointwise[0]["pointwise_ranges"])
        graph = nx.node_link_graph(json.loads(json.dumps(g)), edges="edges")
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
        reductions = [
            n for n in g["nodes"] if n["lowering_class"] == "ReductionLowering"
        ]
        self.assertTrue(reductions)
        self.assertIsNotNone(reductions[0]["reduction_type"])
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
        region_kinds = {gg["region_kind"] for gg in g["graph"]["graphs"]}
        self.assertIn("ForLoopGraphInfo", region_kinds)
        region_edges = [e for e in g["edges"] if e["edge_kind"] == "region"]
        self.assertTrue(region_edges, "attention _for_loop should yield region edges")
        node_ids = {n["id"] for n in g["nodes"]}
        for edge in region_edges:
            self.assertIn(edge["source"], node_ids)
            self.assertIn(edge["target"], node_ids)
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
        self.assertEqual(ir_features._target_str("output"), "output")
        text = ir_features._target_str(extract_ir_graph)
        self.assertIn("extract_ir_graph", text)
        self.assertNotIn("0x", text)
        part = functools.partial(extract_ir_graph)
        self.assertNotIn("0x", ir_features._target_str(part))
        self.assertEqual(ir_features._target_str(5), "5")


if __name__ == "__main__":
    unittest.main()
