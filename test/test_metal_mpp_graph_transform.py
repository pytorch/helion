from __future__ import annotations

import operator
import unittest

import torch
from torch.fx import Graph

from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.metal.mpp_graph_transform import MPPGraphInfo
from helion._compiler.metal.mpp_graph_transform import _mpp_graph
from helion._compiler.metal.mpp_graph_transform import rewrite_mpp_graphs
from helion.language import _tracing_ops
from helion.language import memory_ops


def _target_names(graph: Graph) -> list[str]:
    names: list[str] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        names.append(getattr(target, "__name__", str(target)))
    return names


class TestMetalMPPGraphTransform(unittest.TestCase):
    def _build_matmul_ir(
        self,
        *,
        epilogue: str | None,
        extra_phi_user: bool = False,
        lhs_indices: tuple[int, int] = (0, 1),
        rhs_indices: tuple[int, int] = (1, 2),
        store_indices: tuple[int, int] = (0, 2),
    ) -> DeviceIR:
        body_graph = Graph()
        lhs = body_graph.placeholder("lhs")
        lhs.meta["val"] = torch.empty(4, 64, dtype=torch.float32)
        rhs = body_graph.placeholder("rhs")
        rhs.meta["val"] = torch.empty(64, 4, dtype=torch.float32)
        lhs_load = body_graph.call_function(
            memory_ops.load, args=(lhs, list(lhs_indices), None, None)
        )
        lhs_load.meta["val"] = torch.empty(4, 64, dtype=torch.float32)
        rhs_load = body_graph.call_function(
            memory_ops.load, args=(rhs, list(rhs_indices), None, None)
        )
        rhs_load.meta["val"] = torch.empty(64, 4, dtype=torch.float32)
        acc = body_graph.placeholder("acc")
        acc.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        mma = body_graph.call_function(
            torch.ops.aten.addmm.default, args=(acc, lhs_load, rhs_load)
        )
        mma.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        body_graph.output((mma,))

        root_graph = Graph()
        acc_init = root_graph.call_function(_tracing_ops._new_var, args=())
        for_loop = root_graph.call_function(
            _tracing_ops._for_loop,
            args=(0, [0], [64], [acc_init]),
        )
        getitem = root_graph.call_function(operator.getitem, args=(for_loop, 0))
        phi = root_graph.call_function(_tracing_ops._phi, args=(acc_init, getitem))

        value = phi
        if epilogue == "relu":
            value = root_graph.call_function(torch.ops.aten.relu.default, args=(value,))
        elif epilogue == "cast":
            value = root_graph.call_function(
                torch.ops.prims.convert_element_type.default,
                args=(value, torch.float16),
            )
            value.meta["val"] = torch.empty(4, 4, dtype=torch.float16)
        elif epilogue == "sum":
            value = root_graph.call_function(
                torch.ops.aten.sum.dim_IntList,
                args=(value, [0], False),
            )
            value.meta["val"] = torch.empty(4, dtype=torch.float32)
        elif epilogue == "aux_add":
            aux = root_graph.call_function(_tracing_ops._new_var, args=())
            aux.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
            aux_load = root_graph.call_function(
                memory_ops.load, args=(aux, list(store_indices), None, None)
            )
            aux_load.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
            value = root_graph.call_function(
                torch.ops.aten.add.Tensor, args=(value, aux_load)
            )
            value.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        elif epilogue is not None:
            raise AssertionError(f"unknown epilogue: {epilogue}")

        if extra_phi_user:
            root_graph.call_function(torch.ops.aten.neg.default, args=(phi,))

        out_shape = tuple(value.meta.get("val", torch.empty(4, 4)).shape)
        out_dtype = value.meta.get("val", torch.empty(4, 4)).dtype
        if epilogue == "sum":
            actual_store_indices = [store_indices[0]]
        else:
            actual_store_indices = list(store_indices)

        out = root_graph.call_function(_tracing_ops._new_var, args=())
        out.meta["val"] = torch.empty(*out_shape, dtype=out_dtype)
        root_graph.call_function(
            memory_ops.store, args=(out, actual_store_indices, value, None)
        )
        root_graph.output(())

        device_ir = DeviceIR()
        device_ir.graphs = [
            ForLoopGraphInfo(
                graph_id=0,
                graph=body_graph,
                node_args=[acc_init],
                block_ids=[2],
            ),
            RootGraphInfo(graph_id=1, graph=root_graph, phase_index=0),
        ]
        device_ir.root_ids = [1]
        return device_ir

    def test_direct_store_becomes_fused_mpp_graph(self) -> None:
        device_ir = self._build_matmul_ir(epilogue=None)

        rewrites = rewrite_mpp_graphs(device_ir)

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(rewrites[0].epilogue_node_names, ())
        self.assertIsNotNone(rewrites[0].store_node_name)

        root = device_ir.graphs[1]
        assert isinstance(root, RootGraphInfo)
        root_targets = _target_names(root.graph)
        self.assertIn("_mpp_graph", root_targets)
        self.assertNotIn("_for_loop", root_targets)
        self.assertNotIn("_phi", root_targets)
        self.assertNotIn("store", root_targets)

        mpp_graph = device_ir.graphs[2]
        assert isinstance(mpp_graph, MPPGraphInfo)
        self.assertEqual(mpp_graph.result_name, "addmm_default")
        self.assertEqual(mpp_graph.k_block_id, 2)
        self.assertEqual(mpp_graph.begin, [0])
        self.assertEqual(mpp_graph.end, [64])
        self.assertIsNotNone(mpp_graph.lhs_tensor)
        self.assertIsNotNone(mpp_graph.rhs_tensor)
        self.assertEqual(mpp_graph.acc_dtype, torch.float32)
        self.assertEqual(_target_names(mpp_graph.graph), [])
        self.assertIsNotNone(mpp_graph.out_dtype)

    def test_pointwise_epilogue_moves_into_fused_mpp_graph(self) -> None:
        device_ir = self._build_matmul_ir(epilogue="relu")

        rewrites = rewrite_mpp_graphs(device_ir)

        self.assertEqual(len(rewrites), 1)
        self.assertEqual(len(rewrites[0].epilogue_node_names), 1)

        root = device_ir.graphs[1]
        assert isinstance(root, RootGraphInfo)
        self.assertNotIn("relu.default", _target_names(root.graph))
        self.assertNotIn("store", _target_names(root.graph))

        mpp_graph = device_ir.graphs[2]
        assert isinstance(mpp_graph, MPPGraphInfo)
        self.assertEqual(len(mpp_graph.epilogue_node_names), 1)
        self.assertIn("relu.default", _target_names(mpp_graph.graph))
        self.assertNotIn("store", _target_names(mpp_graph.graph))
        self.assertIsNotNone(mpp_graph.out_dtype)

    def test_nonfusible_epilogue_is_not_rewritten(self) -> None:
        device_ir = self._build_matmul_ir(epilogue="sum")

        rewrites = rewrite_mpp_graphs(device_ir)

        self.assertEqual(rewrites, [])
        root = device_ir.graphs[1]
        assert isinstance(root, RootGraphInfo)
        root_targets = _target_names(root.graph)
        self.assertIn("_for_loop", root_targets)
        self.assertIn("_phi", root_targets)
        self.assertIn("sum.dim_IntList", root_targets)
        self.assertIn("store", root_targets)
        self.assertEqual(len(device_ir.graphs), 2)

    def test_same_shape_nonfusible_epilogue_materializes_mpp_result(self) -> None:
        device_ir = self._build_matmul_ir(epilogue="aux_add")

        rewrites = rewrite_mpp_graphs(device_ir)

        self.assertEqual(len(rewrites), 1)
        root = device_ir.graphs[1]
        assert isinstance(root, RootGraphInfo)
        root_targets = _target_names(root.graph)
        self.assertIn("_mpp_graph", root_targets)
        self.assertIn("load", root_targets)
        self.assertIn("add.Tensor", root_targets)
        self.assertIn("store", root_targets)
        self.assertNotIn("_for_loop", root_targets)
        self.assertNotIn("_phi", root_targets)

        mpp_graph = device_ir.graphs[2]
        assert isinstance(mpp_graph, MPPGraphInfo)
        self.assertEqual(mpp_graph.epilogue_node_names, ())
        self.assertTrue(mpp_graph.needs_store_barrier)

    def test_dtype_cast_epilogue_moves_into_fused_mpp_graph(self) -> None:
        device_ir = self._build_matmul_ir(epilogue="cast")

        rewrites = rewrite_mpp_graphs(device_ir)

        self.assertEqual(len(rewrites), 1)
        mpp_graph = device_ir.graphs[2]
        assert isinstance(mpp_graph, MPPGraphInfo)
        self.assertEqual(
            mpp_graph.epilogue_node_names, ("convert_element_type_default",)
        )
        self.assertIn("convert_element_type.default", _target_names(mpp_graph.graph))

    def test_phi_fanout_is_not_rewritten(self) -> None:
        device_ir = self._build_matmul_ir(epilogue="relu", extra_phi_user=True)

        rewrites = rewrite_mpp_graphs(device_ir)

        self.assertEqual(rewrites, [])
        root = device_ir.graphs[1]
        assert isinstance(root, RootGraphInfo)
        self.assertIn("_for_loop", _target_names(root.graph))
        self.assertNotIn("_mpp_graph", _target_names(root.graph))

    def test_noncanonical_load_store_indices_are_not_rewritten(self) -> None:
        device_ir = self._build_matmul_ir(
            epilogue=None,
            rhs_indices=(2, 1),
            store_indices=(0, 2),
        )

        rewrites = rewrite_mpp_graphs(device_ir)

        self.assertEqual(rewrites, [])
        root = device_ir.graphs[1]
        assert isinstance(root, RootGraphInfo)
        self.assertIn("_for_loop", _target_names(root.graph))
        self.assertNotIn("_mpp_graph", _target_names(root.graph))

    def test_mpp_marker_preserves_loop_args(self) -> None:
        device_ir = self._build_matmul_ir(epilogue="relu")

        rewrite_mpp_graphs(device_ir)

        root = device_ir.graphs[1]
        assert isinstance(root, RootGraphInfo)
        mpp_nodes = [
            node
            for node in root.graph.nodes
            if node.op == "call_function" and node.target is _mpp_graph
        ]
        self.assertEqual(len(mpp_nodes), 1)
        self.assertEqual(mpp_nodes[0].args[0], 2)
        self.assertIsInstance(mpp_nodes[0].args[1], list)


if __name__ == "__main__":
    unittest.main()
