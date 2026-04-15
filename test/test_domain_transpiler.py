"""Tests for the domain-based CuTe DSL transpiler."""

from __future__ import annotations

import unittest

import torch

from helion._compiler.cute.domain_transpiler import Domain
from helion._compiler.cute.domain_transpiler import NodeKind
from helion._compiler.cute.domain_transpiler import classify_node
from helion._compiler.cute.domain_transpiler import has_mma_nodes
from helion._compiler.cute.domain_transpiler import propagate_domains


class TestNodeClassification(unittest.TestCase):
    """Test that FX nodes are classified into the correct NodeKind."""

    def test_mma_ops(self) -> None:
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64)
        b = graph.placeholder("b")
        b.meta["val"] = torch.randn(64, 64)

        mm = graph.call_function(torch.ops.aten.mm.default, (a, b))
        mm.meta["val"] = torch.randn(64, 64)

        bmm = graph.call_function(torch.ops.aten.bmm.default, (a, b))
        bmm.meta["val"] = torch.randn(64, 64)

        self.assertEqual(classify_node(mm), NodeKind.MMA)
        self.assertEqual(classify_node(bmm), NodeKind.MMA)

    def test_pointwise_ops(self) -> None:
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64)

        add = graph.call_function(torch.ops.aten.add.Tensor, (a, a))
        add.meta["val"] = torch.randn(64, 64)

        exp2 = graph.call_function(torch.ops.aten.exp2.default, (a,))
        exp2.meta["val"] = torch.randn(64, 64)

        neg = graph.call_function(torch.ops.aten.neg.default, (a,))
        neg.meta["val"] = torch.randn(64, 64)

        self.assertEqual(classify_node(add), NodeKind.POINTWISE)
        self.assertEqual(classify_node(exp2), NodeKind.POINTWISE)
        self.assertEqual(classify_node(neg), NodeKind.POINTWISE)

    def test_reduction_ops(self) -> None:
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64)

        amax = graph.call_function(torch.ops.aten.amax.default, (a, [-1]))
        amax.meta["val"] = torch.randn(64)

        s = graph.call_function(torch.ops.aten.sum.dim_IntList, (a, [-1]))
        s.meta["val"] = torch.randn(64)

        self.assertEqual(classify_node(amax), NodeKind.REDUCTION)
        self.assertEqual(classify_node(s), NodeKind.REDUCTION)

    def test_cast_op(self) -> None:
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64, dtype=torch.float32)

        cast = graph.call_function(
            torch.ops.prims.convert_element_type.default, (a, torch.float16)
        )
        cast.meta["val"] = torch.randn(64, 64, dtype=torch.float16)

        self.assertEqual(classify_node(cast), NodeKind.CAST)

    def test_placeholder_and_output(self) -> None:
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64)
        out = graph.output(a)

        self.assertEqual(classify_node(a), NodeKind.PLACEHOLDER)
        self.assertEqual(classify_node(out), NodeKind.OTHER)


class TestDomainPropagation(unittest.TestCase):
    """Test the two-pass domain propagation algorithm."""

    def test_single_mma(self) -> None:
        """Single mm: inputs → SHARED, output → FRAGMENT."""
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64)
        b = graph.placeholder("b")
        b.meta["val"] = torch.randn(64, 64)

        mm = graph.call_function(torch.ops.aten.mm.default, (a, b))
        mm.meta["val"] = torch.randn(64, 64)
        graph.output(mm)

        domains = propagate_domains(graph)

        # Backward pass: a, b feed MMA → SHARED
        self.assertEqual(domains[a].domain, Domain.SHARED)
        self.assertEqual(domains[b].domain, Domain.SHARED)
        # Rule 1: MMA output → FRAGMENT
        self.assertEqual(domains[mm].domain, Domain.FRAGMENT)

    def test_pointwise_on_fragment(self) -> None:
        """Pointwise on FRAGMENT input stays FRAGMENT (Rule 2)."""
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64)
        b = graph.placeholder("b")
        b.meta["val"] = torch.randn(64, 64)

        mm = graph.call_function(torch.ops.aten.mm.default, (a, b))
        mm.meta["val"] = torch.randn(64, 64)

        exp2 = graph.call_function(torch.ops.aten.exp2.default, (mm,))
        exp2.meta["val"] = torch.randn(64, 64)
        graph.output(exp2)

        domains = propagate_domains(graph)

        self.assertEqual(domains[mm].domain, Domain.FRAGMENT)
        # Rule 2: pointwise(FRAGMENT) → FRAGMENT
        self.assertEqual(domains[exp2].domain, Domain.FRAGMENT)

    def test_reduction_on_fragment_produces_scalar(self) -> None:
        """Reduction on FRAGMENT → SCALAR (Rule 5)."""
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64)
        b = graph.placeholder("b")
        b.meta["val"] = torch.randn(64, 64)

        mm = graph.call_function(torch.ops.aten.mm.default, (a, b))
        mm.meta["val"] = torch.randn(64, 64)

        rmax = graph.call_function(torch.ops.aten.amax.default, (mm, [-1]))
        rmax.meta["val"] = torch.randn(64)
        graph.output(rmax)

        domains = propagate_domains(graph)

        self.assertEqual(domains[mm].domain, Domain.FRAGMENT)
        # Rule 5: REDUCTION(FRAGMENT) → SCALAR
        self.assertEqual(domains[rmax].domain, Domain.SCALAR)

    def test_fragment_plus_scalar_broadcast(self) -> None:
        """FRAGMENT + SCALAR → FRAGMENT with broadcast (Rule 3)."""
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64)
        b = graph.placeholder("b")
        b.meta["val"] = torch.randn(64, 64)
        bias = graph.placeholder("bias")
        bias.meta["val"] = torch.randn(64)  # not an MMA input

        mm = graph.call_function(torch.ops.aten.mm.default, (a, b))
        mm.meta["val"] = torch.randn(64, 64)

        # bias is not an MMA operand → SCALAR
        add = graph.call_function(torch.ops.aten.add.Tensor, (mm, bias))
        add.meta["val"] = torch.randn(64, 64)
        graph.output(add)

        domains = propagate_domains(graph)

        self.assertEqual(domains[bias].domain, Domain.SCALAR)
        self.assertEqual(domains[mm].domain, Domain.FRAGMENT)
        # Rule 3: POINTWISE(FRAGMENT + SCALAR) → FRAGMENT
        self.assertEqual(domains[add].domain, Domain.FRAGMENT)

    def test_pure_pointwise_all_scalar(self) -> None:
        """No MMA → everything is SCALAR (Rules 4, 9)."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(64)
        y = graph.placeholder("y")
        y.meta["val"] = torch.randn(64)

        add = graph.call_function(torch.ops.aten.add.Tensor, (x, y))
        add.meta["val"] = torch.randn(64)
        graph.output(add)

        domains = propagate_domains(graph)

        # No MMA → backward pass marks nothing as SHARED
        self.assertEqual(domains[x].domain, Domain.SCALAR)
        self.assertEqual(domains[y].domain, Domain.SCALAR)
        # Rule 4: POINTWISE(all SCALAR) → SCALAR
        self.assertEqual(domains[add].domain, Domain.SCALAR)

    def test_attention_pattern(self) -> None:
        """Full attention: 2 MMAs, reduction, pointwise, SMEM staging."""
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        q.meta["val"] = torch.randn(64, 64)
        k = graph.placeholder("k")
        k.meta["val"] = torch.randn(64, 64)
        v = graph.placeholder("v")
        v.meta["val"] = torch.randn(64, 64)

        # GEMM-I: S = Q @ K
        s = graph.call_function(torch.ops.aten.mm.default, (q, k))
        s.meta["val"] = torch.randn(64, 64)

        # Reduction: row_max = amax(S, dim=-1)
        rmax = graph.call_function(torch.ops.aten.amax.default, (s, [-1]))
        rmax.meta["val"] = torch.randn(64)

        # Pointwise: P = exp2(S - row_max)
        sub = graph.call_function(torch.ops.aten.sub.Tensor, (s, rmax))
        sub.meta["val"] = torch.randn(64, 64)
        p = graph.call_function(torch.ops.aten.exp2.default, (sub,))
        p.meta["val"] = torch.randn(64, 64)

        # Reduction: row_sum = sum(P, dim=-1)
        rsum = graph.call_function(torch.ops.aten.sum.dim_IntList, (p, [-1]))
        rsum.meta["val"] = torch.randn(64)

        # GEMM-II: O = P @ V (p feeds as MMA operand → needs staging)
        o = graph.call_function(torch.ops.aten.mm.default, (p, v))
        o.meta["val"] = torch.randn(64, 64)
        graph.output(o)

        domains = propagate_domains(graph)

        # Backward pass: q, k feed GEMM-I; p, v feed GEMM-II → all SHARED
        self.assertEqual(domains[q].domain, Domain.SHARED)
        self.assertEqual(domains[k].domain, Domain.SHARED)
        self.assertEqual(domains[v].domain, Domain.SHARED)

        # Rule 1: MMA outputs → FRAGMENT
        self.assertEqual(domains[s].domain, Domain.FRAGMENT)
        self.assertEqual(domains[o].domain, Domain.FRAGMENT)

        # Rule 5: REDUCTION(FRAGMENT) → SCALAR
        self.assertEqual(domains[rmax].domain, Domain.SCALAR)
        self.assertEqual(domains[rsum].domain, Domain.SCALAR)

        # Rule 3: POINTWISE(FRAGMENT - SCALAR) → FRAGMENT
        self.assertEqual(domains[sub].domain, Domain.FRAGMENT)
        # Rule 2: POINTWISE(FRAGMENT) → FRAGMENT
        self.assertEqual(domains[p].domain, Domain.FRAGMENT)

        # Rule 7: p feeds GEMM-II as operand → needs SMEM staging
        self.assertTrue(domains[p].needs_staging)
        # q, k, v are placeholders (not FRAGMENT) — no staging needed
        self.assertFalse(domains[q].needs_staging)

    def test_cast_inherits_domain(self) -> None:
        """Cast on FRAGMENT stays FRAGMENT (Rule 12)."""
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64)
        b = graph.placeholder("b")
        b.meta["val"] = torch.randn(64, 64)

        mm = graph.call_function(torch.ops.aten.mm.default, (a, b))
        mm.meta["val"] = torch.randn(64, 64, dtype=torch.float32)

        cast = graph.call_function(
            torch.ops.prims.convert_element_type.default, (mm, torch.float16)
        )
        cast.meta["val"] = torch.randn(64, 64, dtype=torch.float16)
        graph.output(cast)

        domains = propagate_domains(graph)

        self.assertEqual(domains[mm].domain, Domain.FRAGMENT)
        # Rule 12: CAST(FRAGMENT) → FRAGMENT
        self.assertEqual(domains[cast].domain, Domain.FRAGMENT)


class TestEmitters(unittest.TestCase):
    """Test that emitters produce correct CuTe DSL code patterns."""

    def test_fragment_pointwise_binary(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_fragment_pointwise

        lines = emit_fragment_pointwise(
            torch.ops.aten.add.Tensor,
            "result_frag",
            ["frag_a", "frag_b"],
            [Domain.FRAGMENT, Domain.FRAGMENT],
        )
        code = "\n".join(lines)
        self.assertIn("for _i in range(cute.size(frag_a))", code)
        self.assertIn("result_frag[_i] = frag_a[_i] + frag_b[_i]", code)

    def test_fragment_pointwise_broadcast(self) -> None:
        """FRAGMENT + SCALAR broadcast: scalar is not indexed."""
        from helion._compiler.cute.domain_transpiler import emit_fragment_pointwise

        lines = emit_fragment_pointwise(
            torch.ops.aten.mul.Tensor,
            "result_frag",
            ["frag_a", "scalar_val"],
            [Domain.FRAGMENT, Domain.SCALAR],
        )
        code = "\n".join(lines)
        self.assertIn("frag_a[_i] * scalar_val", code)
        # scalar_val should NOT have [_i] indexing
        self.assertNotIn("scalar_val[_i]", code)

    def test_fragment_pointwise_unary(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_fragment_pointwise

        lines = emit_fragment_pointwise(
            torch.ops.aten.exp2.default,
            "result_frag",
            ["frag_a"],
            [Domain.FRAGMENT],
        )
        code = "\n".join(lines)
        self.assertIn("cute.exp2(frag_a[_i])", code)

    def test_scalar_pointwise(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_scalar_pointwise

        lines = emit_scalar_pointwise(
            torch.ops.aten.add.Tensor,
            "result",
            ["a", "b"],
        )
        self.assertEqual(lines, ["result = a + b"])

    def test_fragment_reduction_max(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_fragment_reduction

        lines = emit_fragment_reduction(
            torch.ops.aten.amax.default,
            "s_frag",
            "row_max",
            threads_in_group=64,
        )
        code = "\n".join(lines)
        self.assertIn("float('-inf')", code)
        self.assertIn("max(row_max, s_frag[_i])", code)
        self.assertIn("warp_reduction_max(row_max, threads_in_group=64)", code)

    def test_fragment_reduction_sum(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_fragment_reduction

        lines = emit_fragment_reduction(
            torch.ops.aten.sum.dim_IntList,
            "p_frag",
            "row_sum",
            threads_in_group=64,
        )
        code = "\n".join(lines)
        self.assertIn("Float32(0.0)", code)
        self.assertIn("row_sum + p_frag[_i]", code)
        self.assertIn("warp_reduction_sum(row_sum, threads_in_group=64)", code)

    def test_fragment_to_shared(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_fragment_to_shared

        lines = emit_fragment_to_shared("s_frag", "smem_p", "thr_mma", "tCsP")
        code = "\n".join(lines)
        self.assertIn("tCsP = thr_mma.partition_C(smem_p)", code)
        self.assertIn("tCsP[_i] = s_frag[_i]", code)
        self.assertIn("cute.arch.sync_threads()", code)

    def test_mma_setup_universal(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_mma_setup_universal

        lines = emit_mma_setup_universal(
            "tiled_mma", "thr_mma", "acc_frag",
            "thread_linear", "cutlass.Float32",
            bm=64, bn=64,
        )
        code = "\n".join(lines)
        self.assertIn("cute.make_tiled_mma(", code)
        self.assertIn("MmaUniversalOp", code)
        self.assertIn("atom_layout_mnk=(64, 64, 1)", code)
        self.assertIn("tiled_mma.get_slice(thread_linear)", code)
        self.assertIn("cute.make_fragment(", code)
        self.assertIn("partition_shape_C((64, 64))", code)

    def test_gemm_emit(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_gemm

        lines = emit_gemm(
            "tiled_mma", "acc_frag", "thr_mma",
            "smem_a", "smem_b", "cutlass.Float32",
        )
        code = "\n".join(lines)
        self.assertIn("thr_mma.partition_A(smem_a)", code)
        self.assertIn("thr_mma.partition_B(smem_b)", code)
        self.assertIn("cute.gemm(tiled_mma, acc_frag,", code)

    def test_smem_alloc(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_smem_alloc

        lines = emit_smem_alloc(
            "smem_ptr", "smem_tensor", "cutlass.Float16", 64, 32,
        )
        code = "\n".join(lines)
        self.assertIn("cute.arch.alloc_smem(cutlass.Float16, 2048)", code)
        self.assertIn("cute.make_layout((64, 32), stride=(32, 1))", code)

    def test_fragment_readback(self) -> None:
        from helion._compiler.cute.domain_transpiler import emit_fragment_readback

        lines = emit_fragment_readback(
            "acc_frag", "thr_mma", "smem_c", "result",
            "m_local", "n_local",
        )
        code = "\n".join(lines)
        self.assertIn("partition_C(smem_c)", code)
        self.assertIn("sync_threads()", code)
        self.assertIn("result = smem_c[m_local, n_local]", code)


class TestAttentionCodegen(unittest.TestCase):
    """End-to-end test: walk an attention FX graph through domain propagation
    and emitters, verify the generated CuTe DSL code has the expected structure."""

    def test_attention_generates_expected_cutedsl(self) -> None:
        """Walk a synthetic attention graph and emit CuTe DSL code,
        verifying all domain rules fire correctly."""
        from helion._compiler.cute.domain_transpiler import (
            emit_fragment_pointwise,
            emit_fragment_reduction,
            emit_fragment_to_shared,
            emit_gemm,
            emit_mma_setup_universal,
            emit_smem_alloc,
        )

        # Build attention FX graph
        graph = torch.fx.Graph()
        q = graph.placeholder("q")
        q.meta["val"] = torch.randn(64, 64)
        k = graph.placeholder("k")
        k.meta["val"] = torch.randn(64, 64)
        v = graph.placeholder("v")
        v.meta["val"] = torch.randn(64, 64)

        # GEMM-I: S = Q @ K
        s = graph.call_function(torch.ops.aten.mm.default, (q, k))
        s.meta["val"] = torch.randn(64, 64)
        # Reduction: row_max = amax(S)
        rmax = graph.call_function(torch.ops.aten.amax.default, (s, [-1]))
        rmax.meta["val"] = torch.randn(64)
        # Pointwise: S - row_max (broadcast)
        sub = graph.call_function(torch.ops.aten.sub.Tensor, (s, rmax))
        sub.meta["val"] = torch.randn(64, 64)
        # Pointwise: P = exp2(S - row_max)
        p = graph.call_function(torch.ops.aten.exp2.default, (sub,))
        p.meta["val"] = torch.randn(64, 64)
        # Reduction: row_sum = sum(P)
        rsum = graph.call_function(torch.ops.aten.sum.dim_IntList, (p, [-1]))
        rsum.meta["val"] = torch.randn(64)
        # GEMM-II: O = P @ V
        o = graph.call_function(torch.ops.aten.mm.default, (p, v))
        o.meta["val"] = torch.randn(64, 64)
        graph.output(o)

        # Run domain propagation
        domains = propagate_domains(graph)

        # Now walk the graph and collect CuTe DSL code
        all_code: list[str] = []

        # === Setup (outer_prefix) ===
        bm, bn = 64, 64
        all_code.append("# === MMA Setup ===")
        all_code.extend(emit_mma_setup_universal(
            "tiled_mma", "thr_mma", "o_frag",
            "thread_linear", "cutlass.Float32",
            bm=bm, bn=bn,
        ))
        all_code.append("")
        all_code.append("# === SMEM Allocation ===")
        all_code.extend(emit_smem_alloc("smem_q_ptr", "smem_q", "cutlass.Float16", bm, bn))
        all_code.extend(emit_smem_alloc("smem_k_ptr", "smem_k", "cutlass.Float16", bm, bn))
        all_code.extend(emit_smem_alloc("smem_v_ptr", "smem_v", "cutlass.Float16", bm, bn))
        all_code.extend(emit_smem_alloc("smem_p_ptr", "smem_p", "cutlass.Float32", bm, bn))
        all_code.append("")

        # === Loop body ===
        all_code.append("# === GEMM-I: S = Q @ K^T ===")
        all_code.extend(emit_gemm(
            "tiled_mma", "s_frag", "thr_mma",
            "smem_q", "smem_k", "cutlass.Float32",
        ))
        all_code.append("")

        # Reduction: amax → SCALAR (Rule 5)
        self.assertEqual(domains[rmax].domain, Domain.SCALAR)
        all_code.append("# === REDUCTION: row_max (FRAGMENT → SCALAR, Rule 5) ===")
        all_code.extend(emit_fragment_reduction(
            torch.ops.aten.amax.default,
            "s_frag", "new_max", threads_in_group=bn,
        ))
        all_code.append("")

        # Pointwise: S - row_max → FRAGMENT (Rule 3: broadcast)
        self.assertEqual(domains[sub].domain, Domain.FRAGMENT)
        all_code.append("# === POINTWISE: S - row_max (FRAGMENT - SCALAR, Rule 3) ===")
        all_code.extend(emit_fragment_pointwise(
            torch.ops.aten.sub.Tensor,
            "s_frag", ["s_frag", "new_max"],
            [Domain.FRAGMENT, Domain.SCALAR],
        ))
        all_code.append("")

        # Pointwise: exp2 → FRAGMENT (Rule 2)
        self.assertEqual(domains[p].domain, Domain.FRAGMENT)
        all_code.append("# === POINTWISE: P = exp2(S) (FRAGMENT, Rule 2) ===")
        all_code.extend(emit_fragment_pointwise(
            torch.ops.aten.exp2.default,
            "s_frag", ["s_frag"],
            [Domain.FRAGMENT],
        ))
        all_code.append("")

        # Reduction: sum → SCALAR (Rule 5)
        self.assertEqual(domains[rsum].domain, Domain.SCALAR)
        all_code.append("# === REDUCTION: row_sum (FRAGMENT → SCALAR, Rule 5) ===")
        all_code.extend(emit_fragment_reduction(
            torch.ops.aten.sum.dim_IntList,
            "s_frag", "row_sum", threads_in_group=bn,
        ))
        all_code.append("")

        # Domain transition: P → SHARED (Rule 7)
        self.assertTrue(domains[p].needs_staging)
        all_code.append("# === DOMAIN TRANSITION: FRAGMENT → SHARED (Rule 7) ===")
        all_code.extend(emit_fragment_to_shared(
            "s_frag", "smem_p", "thr_mma", "tCsP",
        ))
        all_code.append("")

        # GEMM-II: O = P @ V
        self.assertEqual(domains[o].domain, Domain.FRAGMENT)
        all_code.append("# === GEMM-II: O = P @ V ===")
        all_code.extend(emit_gemm(
            "tiled_mma", "o_frag", "thr_mma",
            "smem_p", "smem_v", "cutlass.Float32",
        ))

        full_code = "\n".join(all_code)

        # Verify expected structure
        # Two GEMMs
        self.assertEqual(full_code.count("cute.gemm("), 2)
        # Fragment-level pointwise
        self.assertIn("cute.exp2(s_frag[_i])", full_code)
        # FRAGMENT - SCALAR broadcast (no indexing on scalar)
        self.assertIn("s_frag[_i] - new_max", full_code)
        self.assertNotIn("new_max[_i]", full_code)
        # Warp reductions
        self.assertIn("warp_reduction_max(", full_code)
        self.assertIn("warp_reduction_sum(", full_code)
        # SMEM staging (Rule 7)
        self.assertIn("thr_mma.partition_C(smem_p)", full_code)
        self.assertIn("thr_mma.partition_A(smem_p)", full_code)
        # MMA setup
        self.assertIn("MmaUniversalOp", full_code)
        self.assertIn("partition_shape_C((64, 64))", full_code)
        # SMEM allocation
        self.assertIn("cute.arch.alloc_smem(", full_code)


class TestHasMmaNodes(unittest.TestCase):
    """Test the MMA node detection utility."""

    def test_with_mma(self) -> None:
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64, 64)
        b = graph.placeholder("b")
        b.meta["val"] = torch.randn(64, 64)
        mm = graph.call_function(torch.ops.aten.mm.default, (a, b))
        mm.meta["val"] = torch.randn(64, 64)
        graph.output(mm)

        self.assertTrue(has_mma_nodes(graph))

    def test_without_mma(self) -> None:
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        a.meta["val"] = torch.randn(64)
        b = graph.placeholder("b")
        b.meta["val"] = torch.randn(64)
        add = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
        add.meta["val"] = torch.randn(64)
        graph.output(add)

        self.assertFalse(has_mma_nodes(graph))


if __name__ == "__main__":
    unittest.main()
