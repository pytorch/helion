from __future__ import annotations

from typing import Any
from typing import cast
import unittest
from unittest.mock import patch

import torch
from torch.fx import Graph

import helion
from helion import exc
from helion._compiler.cute import cute_epilogue
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT,
)
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl


def _make_tcgen05_persistent_config(**overrides: object) -> helion.Config:
    defaults: dict[str, object] = {
        "block_sizes": [128, 256, 16],
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "num_stages": 2,
        "num_warps": 8,
        "pid_type": "flat",
        "tcgen05_cluster_m": 1,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
    }
    defaults.update(overrides)
    return helion.Config(**defaults)  # type: ignore[arg-type]


class TestCuteEpilogue(unittest.TestCase):
    def test_auxiliary_tensor_expr_helpers(self) -> None:
        """Expression helpers preserve structure and explicit leaf bindings."""
        graph = Graph()
        load_a = cute_epilogue._AuxiliaryTensorLoadExpr(
            load_node=graph.placeholder("load_a"),
            broadcast_axis=None,
            template="{aux}",
        )
        load_b = cute_epilogue._AuxiliaryTensorLoadExpr(
            load_node=graph.placeholder("load_b"),
            broadcast_axis=1,
            template="{aux}",
        )
        aux_product = cute_epilogue._BinaryTensorExpr(
            op_name="mul",
            op_template="{lhs} * {rhs}",
            lhs=load_a,
            rhs=load_b,
        )
        expr = cute_epilogue._BinaryTensorExpr(
            op_name="mul",
            op_template="{lhs} * {rhs}",
            lhs=cute_epilogue._CurrentTensorExpr(),
            rhs=aux_product,
        )
        step = cute_epilogue._AuxiliaryTensorExprStep(expr=expr)

        self.assertEqual(
            cute_epilogue._auxiliary_tensor_expr_operands(expr),
            (load_a, load_b),
        )
        self.assertFalse(cute_epilogue._tensor_expr_contains_current(aux_product))
        self.assertTrue(cute_epilogue._tensor_expr_contains_current(expr))
        self.assertIs(step.hoistable_aux_expr, aux_product)

        # Reverse insertion order demonstrates that rendering looks up each
        # exact leaf instead of consuming locals positionally.
        prelude, result = step.render_hoistable_aux_prelude_and_expr(
            {load_b: "local_b", load_a: "local_a"},
            lambda prefix: prefix,
            "    ",
        )
        self.assertEqual(
            prelude,
            "    tcgen05_aux_product = local_a * local_b\n",
        )
        self.assertEqual(result, "tcgen05_aux_product")
        self.assertEqual(
            step.render_with_hoisted_aux("acc", result),
            "acc * tcgen05_aux_product",
        )

        direct_load_step = cute_epilogue._AuxiliaryTensorExprStep(
            expr=cute_epilogue._BinaryTensorExpr(
                op_name="add",
                op_template="{lhs} + {rhs}",
                lhs=cute_epilogue._CurrentTensorExpr(),
                rhs=load_a,
            )
        )
        self.assertIsNone(direct_load_step.hoistable_aux_expr)

    def test_auxiliary_tensor_expr_fx_helpers(self) -> None:
        """The structural precheck and metadata-aware classifier stay distinct."""
        graph = Graph()
        load_a = graph.placeholder("load_a")
        load_a.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
        load_b = graph.placeholder("load_b")
        load_b.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
        product = graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(load_a, load_b),
        )
        product.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
        relu = graph.call_function(torch.ops.aten.relu.default, args=(product,))
        relu.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
        unsupported = graph.call_function(torch.ops.aten.sum.default, args=(load_a,))
        unsupported.meta["val"] = torch.empty((), dtype=torch.float32)

        with patch.object(
            cute_epilogue,
            "_is_helion_load_node",
            side_effect=lambda node: node is load_a or node is load_b,
        ):
            self.assertTrue(cute_epilogue._is_auxiliary_tensor_expr_node(relu))
            self.assertFalse(cute_epilogue._is_auxiliary_tensor_expr_node(unsupported))

        def load_kind(
            node: torch.fx.Node,
            **kwargs: object,
        ) -> tuple[str, int | None] | None:
            if node is load_a:
                return "exact", None
            if node is load_b:
                return "broadcast", 1
            return None

        with patch.object(
            cute_epilogue,
            "aux_tensor_load_kind",
            side_effect=load_kind,
        ):
            classified = cute_epilogue._classify_auxiliary_tensor_expr(
                product,
                carrier_tile_shape=(4, 4),
                carrier_tile_index_nodes=None,
                carrier_global_shape=(4, 4),
            )
            self.assertIsNone(
                cute_epilogue._classify_auxiliary_tensor_expr(
                    unsupported,
                    carrier_tile_shape=(4, 4),
                    carrier_tile_index_nodes=None,
                    carrier_global_shape=(4, 4),
                )
            )

        self.assertIsInstance(classified, cute_epilogue._BinaryTensorExpr)
        assert isinstance(classified, cute_epilogue._BinaryTensorExpr)
        self.assertIsInstance(classified.lhs, cute_epilogue._AuxiliaryTensorLoadExpr)
        self.assertIsInstance(classified.rhs, cute_epilogue._AuxiliaryTensorLoadExpr)
        assert isinstance(classified.lhs, cute_epilogue._AuxiliaryTensorLoadExpr)
        assert isinstance(classified.rhs, cute_epilogue._AuxiliaryTensorLoadExpr)
        self.assertIs(classified.lhs.load_node, load_a)
        self.assertIsNone(classified.lhs.broadcast_axis)
        self.assertIs(classified.rhs.load_node, load_b)
        self.assertEqual(classified.rhs.broadcast_axis, 1)

    def test_auxiliary_tensor_expr_helpers_reject_unknown_variant(self) -> None:
        """Every recursive expression helper fails loudly on a new variant."""
        unknown = cast("Any", object())
        for helper in (
            cute_epilogue._auxiliary_tensor_expr_operands,
            cute_epilogue._tensor_expr_contains_current,
        ):
            with (
                self.subTest(helper=helper.__name__),
                self.assertRaisesRegex(
                    AssertionError, "unexpected tensor expression: object"
                ),
            ):
                helper(unknown)
        with self.assertRaisesRegex(
            AssertionError, "unexpected tensor expression: object"
        ):
            cute_epilogue._render_auxiliary_tensor_expr(
                unknown,
                "acc",
                {},
                lambda prefix: prefix,
                "",
            )

    @skipUnlessBackends(["cute"])
    def test_tcgen05_fused_auxiliary_product_codegen(self) -> None:
        """An explicitly grouped pair of scale loads stays one epilogue step."""

        @helion.kernel(backend="cute", static_shapes=True)
        def scaled_fp8(
            x: torch.Tensor,
            y: torch.Tensor,
            scale_m: torch.Tensor,
            scale_n: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                tile_scale = scale_m[tile_m, tile_n] * scale_n[tile_n]
                out[tile_m, tile_n] = (acc * tile_scale).to(torch.bfloat16)
            return out

        args = (
            torch.empty([256, 128], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([128, 128], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([256, 1], device=DEVICE).expand(256, 128),
            torch.empty([128], device=DEVICE),
        )
        cfg = _make_tcgen05_persistent_config(
            block_sizes=[128, 128, 128],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            indexing=["tensor_descriptor"] * 5,
        )
        with patch_cute_mma_support():
            bound = scaled_fp8.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            code = bound.to_triton_code(cfg)

        loop_pos = code.index("for _tcgen05_subtile in cutlass.range")
        product_pos = code.index(
            "tcgen05_aux_loaded_0 * tcgen05_aux_loaded_1", loop_pos
        )
        acc_pos = code.index("tcgen05_acc_loaded", loop_pos)
        self.assertLess(product_pos, acc_pos)

    @skipUnlessBackends(["cute"])
    def test_tcgen05_fused_reused_auxiliary_load_codegen(self) -> None:
        """Repeated uses of one FX load retain distinct leaf-to-local bindings."""

        @helion.kernel(backend="cute", static_shapes=True)
        def scaled_fp8(
            x: torch.Tensor,
            y: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                tile_scale = scale[tile_m, tile_n]
                acc = acc + tile_scale
                out[tile_m, tile_n] = (acc + tile_scale).to(torch.bfloat16)
            return out

        args = (
            torch.empty([256, 128], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([128, 128], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([256, 1], device=DEVICE).expand(256, 128),
        )
        cfg = _make_tcgen05_persistent_config(
            block_sizes=[128, 128, 128],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            indexing=["tensor_descriptor"] * 4,
        )
        with patch_cute_mma_support():
            bound = scaled_fp8.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            code = bound.to_triton_code(cfg)

        loop_pos = code.index("for _tcgen05_subtile in cutlass.range")
        chain_lines = [
            line
            for line in code[loop_pos:].splitlines()
            if "tcgen05_chain_step" in line and " = " in line
        ]
        self.assertGreaterEqual(len(chain_lines), 2)
        self.assertIn("tcgen05_aux_loaded_0", chain_lines[0])
        self.assertIn("tcgen05_aux_loaded_1", chain_lines[1])

    @skipUnlessBackends(["cute"])
    def test_tcgen05_fused_auxiliary_unary_expr_codegen(self) -> None:
        """Whitelisted unary ops can consume a grouped auxiliary expression."""

        @helion.kernel(backend="cute", static_shapes=True)
        def scaled_fp8(
            x: torch.Tensor,
            y: torch.Tensor,
            scale_m: torch.Tensor,
            scale_n: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                tile_scale = torch.exp(scale_m[tile_m, tile_n] * scale_n[tile_n])
                out[tile_m, tile_n] = (tile_scale - acc).to(torch.bfloat16)
            return out

        args = (
            torch.empty([256, 128], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([128, 128], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([256, 1], device=DEVICE).expand(256, 128),
            torch.empty([128], device=DEVICE),
        )
        cfg = _make_tcgen05_persistent_config(
            block_sizes=[128, 128, 128],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            indexing=["tensor_descriptor"] * 5,
        )
        with patch_cute_mma_support():
            bound = scaled_fp8.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            code = bound.to_triton_code(cfg)

        loop_pos = code.index("for _tcgen05_subtile in cutlass.range")
        product_pos = code.index(
            "tcgen05_aux_loaded_0 * tcgen05_aux_loaded_1", loop_pos
        )
        unary_pos = code.index("cute.math.exp(tcgen05_aux_product", product_pos)
        acc_pos = code.index("tcgen05_acc_loaded", loop_pos)
        carrier_line = next(
            line
            for line in code[acc_pos:].splitlines()
            if "tcgen05_chain_step" in line and " = " in line
        )
        self.assertLess(product_pos, unary_pos)
        self.assertLess(unary_pos, acc_pos)
        self.assertIn("tcgen05_acc_loaded", carrier_line)
        self.assertIn("tcgen05_aux_expr", carrier_line)
        self.assertIn(" - ", carrier_line)
        self.assertLess(
            carrier_line.index("tcgen05_aux_expr"),
            carrier_line.index("tcgen05_acc_loaded"),
        )

        int_args = (
            args[0],
            args[1],
            torch.empty([256, 1], device=DEVICE, dtype=torch.int32).expand(256, 128),
            torch.empty([128], device=DEVICE, dtype=torch.int32),
        )
        with patch_cute_mma_support():
            int_bound = scaled_fp8.bind(int_args)
            int_bound.env.config_spec.cute_tcgen05_search_enabled = True
            with self.assertRaises(exc.BackendUnsupported):
                int_bound.to_triton_code(cfg)

        @helion.kernel(backend="cute", static_shapes=True)
        def integer_scaled(
            x: torch.Tensor,
            y: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = (acc + scale[tile_m, tile_n] * 2).to(
                    torch.bfloat16
                )
            return out

        int_scalar_cfg = _make_tcgen05_persistent_config(
            block_sizes=[128, 128, 128],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            indexing=["tensor_descriptor"] * 4,
        )
        with patch_cute_mma_support():
            int_scalar_bound = integer_scaled.bind((args[0], args[1], int_args[2]))
            int_scalar_bound.env.config_spec.cute_tcgen05_search_enabled = True
            with self.assertRaises(exc.BackendUnsupported):
                int_scalar_bound.to_triton_code(int_scalar_cfg)

    @skipUnlessBackends(["cute"])
    def test_tcgen05_bm256_broadcast_scales_reuse_auxiliary_loads(self) -> None:
        """The four-CTA full-tile epilogue reuses both broadcast scales."""

        @helion.kernel(backend="cute", static_shapes=True)
        def scaled_fp8(
            x: torch.Tensor,
            y: torch.Tensor,
            scale_m: torch.Tensor,
            scale_n: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                tile_scale = scale_m[tile_m, tile_n] * scale_n[tile_n]
                out[tile_m, tile_n] = (acc * tile_scale).to(torch.bfloat16)
            return out

        args = (
            torch.empty([512, 128], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([128, 256], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([512, 1], device=DEVICE).expand(512, 256),
            torch.empty([256], device=DEVICE),
        )
        cfg = _make_tcgen05_persistent_config(
            block_sizes=[256, 128, 128],
            pid_type="persistent_blocked",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=2,
            tcgen05_acc_stages=1,
            tcgen05_aux_load_placement=TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT,
            indexing=["tensor_descriptor"] * 5,
        )
        with patch_cute_mma_support():
            bound = scaled_fp8.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            code = bound.to_triton_code(cfg)

        self.assertIn(
            "tcgen05_aux_rowvec_smem_layout_1 = "
            "cute.make_layout((4, 128), stride=(128, 1))",
            code,
        )
        self.assertIn(
            "tcgen05_aux_rowvec_tiled_copy_1 = cute.make_tiled_copy_tv",
            code,
        )
        self.assertIn(
            "tcgen05_aux_rowvec_thr_copy_1 = "
            "tcgen05_aux_rowvec_tiled_copy_1.get_slice("
            "tcgen05_epi_tidx % cutlass.Int32(32))",
            code,
        )
        self.assertIn(
            "tcgen05_aux_rowvec_smem_1[tcgen05_epi_tidx // "
            "cutlass.Int32(32), None].iterator",
            code,
        )
        scalar_hoist = (
            "tcgen05_colvec_scalar_full_0 = tcgen05_tTR_gAux_grouped_0[0, 0, 0, 0]"
        )
        self.assertIn(scalar_hoist, code)
        scalar_read = "tcgen05_aux_loaded_0 = tcgen05_colvec_scalar_full_0"
        self.assertIn(scalar_read, code)
        self.assertNotIn(
            "tcgen05_aux_loaded_0 = tcgen05_tTR_gAux_grouped_0.load()", code
        )
        loop_pos = code.index("for _tcgen05_subtile in cutlass.range")
        self.assertLess(code.index(scalar_hoist), loop_pos)
        scalar_use_pos = code.index(scalar_read, loop_pos)
        product_pos = code.index("tcgen05_aux_product", scalar_use_pos)
        wait_pos = code.index("tcgen05_acc_pipeline.consumer_wait", loop_pos)
        self.assertLess(scalar_use_pos, product_pos)
        self.assertLess(product_pos, wait_pos)
        self.assertNotIn("tcgen05_aux_rowvec_pred_1", code)
