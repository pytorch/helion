from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel
def matmul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tm, tk], b[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@helion.kernel
def row_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


@helion.kernel
def unroll_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """A single source load statically unrolled into 3 emissions, all feeding the reduction."""
    m, _ = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for i in (0, 1, 2):
            acc = acc + x[tile_m, i]
        out[tile_m] = acc
    return out


@helion.kernel
def reread_kernel(x: torch.Tensor) -> torch.Tensor:
    """One load feeds two reductions on the same axis -> row_reread."""
    m, _ = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        v = x[tile_m, :]
        mx = torch.amax(v, dim=1)
        out[tile_m] = torch.sum(v - mx[:, None], dim=1)
    return out


@skipIfRefEager("config spec inspection is not applicable in ref eager mode")
class TestMemoryOpFacts(RefEagerTestBase, TestCase):
    def test_specs_align_with_indexing_list(self):
        x = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        spec = add_kernel.bind((x, x.clone())).config_spec
        specs = spec.memory_op_facts

        # One spec per indexing slot, in slot order.
        self.assertEqual(len(specs), spec.indexing.length)
        self.assertEqual([s.indexing_index for s in specs], list(range(len(specs))))

        # store_indices and eviction list length are derived from the same specs.
        self.assertEqual(
            spec.store_indices,
            [s.indexing_index for s in specs if s.kind == "store"],
        )
        self.assertEqual(
            spec.load_eviction_policies.length,
            sum(1 for s in specs if s.eviction_index is not None),
        )

    @skipIfNotCUDA()
    @onlyBackends(["triton"])
    def test_reduction_fact_indexing_slot_invariant(self):
        """On triton the facts are COLLAPSED to one entry per distinct ``mem_op_id`` with
        ``indexing_index`` taken from the id->slot map (the single numbering source of
        truth). So the rolled-reduction copy is folded into its original (no separate
        ``graph_id>0`` fact), the ``memory_op_facts[i].indexing_index == i`` invariant is
        restored, and ``num_load`` is recovered from ``static_emission_count`` (==1 here:
        one streamed input row, emitted once in the original graph)."""
        x = torch.randn([256, 512], device=DEVICE, dtype=torch.float32)
        spec = row_sum_kernel.bind((x,)).config_spec
        specs = spec.memory_op_facts

        # Collapse restores the slot alignment / length (one fact per distinct slot).
        self.assertEqual(len(specs), spec.indexing.length)
        self.assertEqual([s.indexing_index for s in specs], list(range(len(specs))))
        self.assertEqual(
            spec.load_eviction_policies.length,
            sum(1 for s in specs if s.eviction_index is not None),
        )

        # Phase 3 built exactly one ReductionFact, derived from the enriched facts.
        self.assertEqual(len(spec.reduction_facts), 1)
        fact = spec.reduction_facts[0]
        self.assertEqual(fact.num_load, 1)

        # The rolled-subgraph copy is COLLAPSED into the original first-occurrence fact:
        # exactly one load fact (no graph_id>0 duplicate), and num_load == that load's
        # static emission count (1).
        load_facts = [s for s in specs if s.kind == "load"]
        self.assertEqual(len(load_facts), 1)
        self.assertEqual(load_facts[0].graph_id, 0)
        self.assertEqual(load_facts[0].static_emission_count, 1)
        self.assertEqual(
            sum(s.static_emission_count for s in load_facts), fact.num_load
        )

    @skipIfNotCUDA()
    @onlyBackends(["triton"])
    def test_static_unroll_num_load_preserved(self):
        """A single source load statically unrolled into 3 emissions collapses to ONE fact
        carrying ``static_emission_count == 3``; ``num_load`` recovers the instruction
        count (3), not the distinct-source count (1)."""
        x = torch.randn([256, 3], device=DEVICE, dtype=torch.float32)
        spec = unroll_sum_kernel.bind((x,)).config_spec
        specs = spec.memory_op_facts

        load_facts = [s for s in specs if s.kind == "load"]
        self.assertEqual(len(load_facts), 1)  # the 3 emissions collapse to one fact
        self.assertEqual(load_facts[0].static_emission_count, 3)
        # indexing collapsed to one slot for the load (+ one for the store).
        self.assertEqual([s.indexing_index for s in specs], list(range(len(specs))))
        if spec.reduction_facts:
            self.assertEqual(spec.reduction_facts[0].num_load, 3)

    @skipIfNotCUDA()
    @onlyBackends(["triton"])
    def test_reread_eviction_index_is_map_slot(self):
        """The re-read load's ``reread_eviction_index`` is the DENSE map eviction slot the
        codegen actually reads (the unify soundness fix) — not a separate facts counter."""
        x = torch.randn([256, 512], device=DEVICE, dtype=torch.float32)
        spec = reread_kernel.bind((x,)).config_spec
        self.assertEqual(len(spec.reduction_facts), 1)
        fact = spec.reduction_facts[0]
        self.assertTrue(fact.row_reread)
        self.assertIsNotNone(fact.reread_eviction_index)
        self.assertIn(
            fact.reread_eviction_index, set(spec.mem_op_slot_map.eviction.values())
        )
        # And it indexes a valid slot of the (dense) eviction list.
        self.assertLess(fact.reread_eviction_index, spec.load_eviction_policies.length)

    def test_load_store_metadata(self):
        x = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        specs = add_kernel.bind((x, x.clone())).config_spec.memory_op_facts

        self.assertEqual([s.kind for s in specs], ["load", "load", "store"])
        self.assertEqual([s.tensor_name for s in specs], ["x", "y", "out"])
        # Loads get sequential eviction slots; stores get none.
        self.assertEqual([s.eviction_index for s in specs], [0, 1, None])
        for s in specs:
            self.assertEqual(s.dtype, torch.float32)
            self.assertEqual(s.ndim, 2)
        # Both loads are consumed downstream; the store has no users.
        loads = [s for s in specs if s.kind == "load"]
        self.assertTrue(all(s.num_reuses >= 1 for s in loads))

    def test_matmul_operand_roles(self):
        a = torch.randn([256, 128], device=DEVICE, dtype=torch.float32)
        b = torch.randn([128, 64], device=DEVICE, dtype=torch.float32)
        specs = matmul_kernel.bind((a, b)).config_spec.memory_op_facts

        operands = {s.tensor_name: s.matmul_operand for s in specs if s.kind == "load"}
        self.assertEqual(operands.get("a"), "lhs")
        self.assertEqual(operands.get("b"), "rhs")
        # The output store is not a matmul operand.
        stores = [s for s in specs if s.kind == "store"]
        self.assertTrue(all(s.matmul_operand is None for s in stores))

    def test_computed_operand_is_not_tagged(self):
        """An operand computed from two loads (a + c) must not be mis-attributed
        to its first input; only the direct rhs load is tagged."""

        @helion.kernel
        def mm_bias(a: torch.Tensor, c: torch.Tensor, b: torch.Tensor):
            m, k = a.shape
            _, n = b.shape
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for tm, tn in hl.tile([m, n]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = hl.dot(a[tm, tk] + c[tm, tk], b[tk, tn], acc=acc)
                out[tm, tn] = acc
            return out

        a = torch.randn([128, 64], device=DEVICE, dtype=torch.float32)
        c = torch.randn([128, 64], device=DEVICE, dtype=torch.float32)
        b = torch.randn([64, 32], device=DEVICE, dtype=torch.float32)
        specs = mm_bias.bind((a, c, b)).config_spec.memory_op_facts

        operands = {s.tensor_name: s.matmul_operand for s in specs if s.kind == "load"}
        self.assertIsNone(operands.get("a"))
        self.assertIsNone(operands.get("c"))
        self.assertEqual(operands.get("b"), "rhs")

    @skipIfNotCUDA()
    @onlyBackends(["triton"])
    def test_dot_scaled_operand_roles(self):
        """dot_scaled's rhs matrix is arg 3 (mat2), not arg 1 (mat1_scale)."""

        @helion.kernel(config=helion.Config(block_sizes=[32, 32]))
        def scaled(x, x_scale, y, y_scale):
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tm, tn in hl.tile([m, n]):
                out[tm, tn] = hl.dot_scaled(
                    x[tm, :], x_scale[tm, :], "e4m3", y[:, tn], y_scale[tn, :], "e4m3"
                )
            return out

        scale_factor = 32
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.float16)
        x_scale = torch.full(
            [64, 64 // scale_factor], 127, device=DEVICE, dtype=torch.uint8
        )
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.float16)
        y_scale = torch.full(
            [64, 64 // scale_factor], 127, device=DEVICE, dtype=torch.uint8
        )
        specs = scaled.bind((x, x_scale, y, y_scale)).config_spec.memory_op_facts

        operands = {s.tensor_name: s.matmul_operand for s in specs if s.kind == "load"}
        self.assertEqual(operands.get("x"), "lhs")
        self.assertEqual(operands.get("y"), "rhs")
        # The scale operands are not matmul matrices.
        self.assertIsNone(operands.get("x_scale"))
        self.assertIsNone(operands.get("y_scale"))


if __name__ == "__main__":
    import unittest

    unittest.main()
