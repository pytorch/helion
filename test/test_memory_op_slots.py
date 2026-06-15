"""Tests for the transform-invariant per-memory-op tunable slots (``mem_op_id`` keying).

Codegen resolves each emitted load/store/atomic's config slot
(``config.indexing`` / ``load_eviction_policies`` / ``load_cache_modifiers`` / ``atomic_indexing``)
through a stable ``node.meta['mem_op_id']`` -> slot map (``config_spec.mem_op_slot_map``) instead of a
config-dependent emission counter. These tests pin the resulting contract: 1:N copies SHARE one slot,
distinct buffers stay DISTINCT (no ``root_rw_name`` collision), and a reduction re-read load gets the
SAME eviction slot under persistent and looped configs (the soundness payoff). All GPU-free
(``device='meta'`` + codegen text), triton backend only.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel
def reread_reduce(x: torch.Tensor) -> torch.Tensor:
    """One load feeds two reductions -> ``row_reread``; rolls via ``reduction_loops``."""
    m, _ = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        v = x[tile_m, :]
        mx = torch.amax(v, dim=1)
        out[tile_m] = torch.sum(v - mx[:, None], dim=1)
    return out


@helion.kernel
def static_unroll(x: torch.Tensor) -> torch.Tensor:
    """A literal-tuple loop unrolls one source load into N emissions sharing one (loc, buffer) id."""
    m, _ = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for i in (0, 1, 2):
            acc = acc + x[tile_m, i]
        out[tile_m] = acc
    return out


@helion.kernel
def tuple_buffers(ws: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Distinct tuple-element buffers at ONE source location -> must get DISTINCT ids.

    ``root_rw_name()`` collapses ``ws[0]/ws[1]/ws[2]`` to the parent ``ws`` (a collision); the id keys
    on ``origin.host_str()`` which keeps the ``[k]`` so the three stay distinct.
    """
    m, _ = ws[0].shape
    out = torch.empty([m], dtype=torch.float32, device=ws[0].device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for w in ws:
            acc = acc + torch.sum(w[tile_m, :], dim=1)
        out[tile_m] = acc
    return out


@helion.kernel
def stack_load_kernel(
    dev_ptrs: torch.Tensor, example_tensor: torch.Tensor
) -> torch.Tensor:
    """A stack load (``tensors[i]``) routes to StackIndexingStrategy — its indexing slot is INERT,
    but under uniform source-based membership it still gets one."""
    m = hl.specialize(dev_ptrs.size(0))
    n = example_tensor.size(0)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev_ptrs.device)
    for i in hl.grid(n):
        ptr_tile = dev_ptrs[:]
        tensors = hl.stacktensor_like(example_tensor, ptr_tile)
        out[:, i] = tensors[i]
    return out


@helion.kernel
def atomic_cas_kernel(
    x: torch.Tensor, y: torch.Tensor, expect: torch.Tensor
) -> torch.Tensor:
    """``atomic_cas`` always uses pointer indexing (it never reads an atomic_indexing strategy) — under
    uniform source-based membership it still gets an (inert) atomic_indexing slot, consistent with a
    tile.index / stack op getting an inert indexing slot."""
    for i in hl.tile(x.size(0)):
        hl.atomic_cas(x, [i], expect[i], y[i])
    return x


def _meta(*shape: int) -> torch.Tensor:
    return torch.empty(list(shape), dtype=torch.float32, device="meta")


@onlyBackends(["triton"])
@skipIfRefEager("slot-map inspection is not applicable in ref eager mode")
class TestMemoryOpSlots(RefEagerTestBase, TestCase):
    def test_slot_map_built(self):
        bk = reread_reduce.bind((_meta(256, 512),))
        slot_map = bk.config_spec.mem_op_slot_map
        self.assertIsNotNone(slot_map)
        # One load (reused) -> one indexing + one eviction slot; one store -> one indexing slot.
        self.assertEqual(set(slot_map.eviction.values()), {0})
        self.assertEqual(sorted(slot_map.indexing.values()), [0, 1])

    def test_static_unroll_shares_one_indexing_slot(self):
        bk = static_unroll.bind((_meta(256, 3),))
        slot_map = bk.config_spec.mem_op_slot_map
        # All three unrolled x-loads share ONE indexing slot (1:N share).
        x_slots = {slot for key, slot in slot_map.indexing.items() if key[1] == "x"}
        self.assertEqual(len(x_slots), 1)
        # And codegen applies one strategy to all three even under a per-slot-heterogeneous config.
        base = bk.config_spec.default_config().config
        idx_types = bk.config_spec.valid_indexing_types()
        if len(idx_types) > 1:
            n = len(base["indexing"])
            cycled = [idx_types[i % len(idx_types)] for i in range(n)]
            code = bk.to_triton_code(
                helion.Config.from_dict({**base, "indexing": cycled})
            )
            # The three x loads differ only in offset (x+0/x+1/x+2) but share one indexing slot, so
            # they must all use the SAME indexing strategy kind (all pointer or all block_ptr) — a
            # per-slot config can't split them the way the old per-emission numbering would.
            kinds = {
                "block_ptr" if "make_block_ptr(x" in ln else "pointer"
                for ln in code.splitlines()
                if "tl.load(" in ln and ("x +" in ln or "make_block_ptr(x" in ln)
            }
            self.assertEqual(len(kinds), 1)

    def test_tuple_buffers_distinct_slots_no_collision(self):
        # Binding runs the always-on injectivity check; a host_str collision would raise here.
        bk = tuple_buffers.bind((tuple(_meta(256, 512) for _ in range(3)),))
        slot_map = bk.config_spec.mem_op_slot_map
        ws_slots = {
            slot
            for key, slot in slot_map.indexing.items()
            if isinstance(key[1], str) and key[1].startswith("ws[")
        }
        self.assertEqual(len(ws_slots), 3)

    def test_reread_eviction_shared_and_transform_invariant(self):
        """A reduction re-read load gets the seed's eviction slot under BOTH persistent and looped,
        and every rolled copy shares it — the C4 soundness payoff."""
        bk = reread_reduce.bind((_meta(256, 512),))
        spec = bk.config_spec
        fact = spec.reduction_facts[0]
        self.assertTrue(fact.row_reread)
        slot = fact.reread_eviction_index
        self.assertIsNotNone(slot)
        # The seed's static reread slot is a real eviction slot the codegen map resolves to.
        self.assertIn(slot, set(spec.mem_op_slot_map.eviction.values()))

        n = spec.load_eviction_policies.length
        policies = ["first"] * n
        policies[slot] = "last"
        base = spec.default_config().config
        rl_len = len(base["reduction_loops"])

        def evict_counts(reduction_loops):
            code = bk.to_triton_code(
                helion.Config.from_dict(
                    {
                        **base,
                        "reduction_loops": reduction_loops,
                        "load_eviction_policies": policies,
                    }
                )
            )
            return code.count("evict_last"), code.count("evict_first")

        # Persistent: the single re-read emission gets 'last'.
        last_p, first_p = evict_counts([None] * rl_len)
        self.assertEqual((last_p, first_p), (1, 0))
        # Looped: rolling re-emits the re-read load as 2 copies; BOTH share slot `slot` -> both 'last'
        # (positional numbering would land 'last' on only one copy and 'first' on the other).
        last_l, first_l = evict_counts([16] * rl_len)
        self.assertEqual((last_l, first_l), (2, 0))

    def test_uniform_membership_stack_load_inert_slot(self):
        """Source-based membership: every load/store gets an indexing slot, including a stack load
        (whose slot is inert). store_strategy_slots holds only real-tensor store slots, so the inert
        stack-load slot is excluded from tensor_descriptor forcing (store_indices)."""
        dev_ptrs = torch.empty(4, dtype=torch.uint64, device="meta")
        example = torch.empty(8, dtype=torch.bfloat16, device="meta")
        bk = stack_load_kernel.bind((dev_ptrs, example))
        spec = bk.config_spec
        facts = spec.memory_op_facts

        # Uniform: no op lacks an indexing slot (the old -1 sentinel is gone).
        self.assertTrue(all(f.indexing_index >= 0 for f in facts))
        # The stack load (host buffer None) is present WITH an indexing slot.
        stack_loads = [f for f in facts if f.kind == "load" and f.tensor_name is None]
        self.assertEqual(len(stack_loads), 1)
        self.assertGreaterEqual(stack_loads[0].indexing_index, 0)
        # store_indices == the real-store strategy slots (no inert slots); the inert stack-load slot
        # is not one of them.
        smap = spec.mem_op_slot_map
        self.assertEqual(spec.store_indices, sorted(smap.store_strategy_slots))
        self.assertNotIn(stack_loads[0].indexing_index, smap.store_strategy_slots)
        # The inert slot does not break codegen.
        self.assertIn("tl.load", bk.to_triton_code(spec.default_config()))

    def test_uniform_membership_atomic_cas_inert_slot(self):
        """Source-based membership extends to atomics: ``atomic_cas`` gets an atomic_indexing slot
        even though its codegen always uses pointer indexing (never reads the slot) — consistent with
        tile.index / stack ops getting an inert slot. (Previously cas was excluded from the namespace.)
        """
        xi = torch.empty(256, dtype=torch.int32, device="meta")
        bk = atomic_cas_kernel.bind((xi, xi.clone(), xi.clone()))
        spec = bk.config_spec
        smap = spec.mem_op_slot_map
        # The cas op is now present in the atomic namespace (contiguous, sized into the config list).
        self.assertEqual(smap.atomic_count, 1)
        self.assertEqual(sorted(smap.atomic.values()), [0])
        self.assertIsNotNone(spec.atomic_indexing)
        self.assertEqual(spec.atomic_indexing.length, 1)
        # The slot is INERT: codegen still emits a pointer-based tl.atomic_cas for every legal value
        # of its atomic_indexing slot (the value is never read).
        codes = {
            bk.to_triton_code(
                helion.Config.from_dict(
                    {**spec.default_config().config, "atomic_indexing": [t]}
                )
            )
            for t in spec.valid_atomic_indexing_types()
        }
        self.assertEqual(len(codes), 1)
        self.assertIn("tl.atomic_cas", next(iter(codes)))


if __name__ == "__main__":
    import unittest

    unittest.main()
