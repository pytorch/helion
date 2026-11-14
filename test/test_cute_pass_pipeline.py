from __future__ import annotations

import unittest

import pytest

from helion._compiler.cute.passes.pipeline import DEVICE_BODY_PASSES
from helion._compiler.cute.passes.pipeline import DEVICE_BODY_PROLOGUE
from helion._compiler.cute.passes.pipeline import KERNEL_DEF_PASSES
from helion._compiler.cute.passes.pipeline import DeviceBodyPass
from helion._compiler.cute.passes.pipeline import PassContext
from helion._compiler.cute.passes.pipeline import _check_pass_order

pytestmark = pytest.mark.cpu_only

# The inline call sequence of ``DeviceFunction.codegen_function_def`` as it was
# when the pipeline moved to its tables.  Later passes recognize the shapes
# earlier passes leave behind and the autotune cache key covers the rendered
# source, so an accidental reorder changes generated code; appending or
# removing an entry is a deliberate, render-checked change that updates this
# list.  The fact derivations of the inline sequence (float scalar names,
# constexpr values, the raw launch shape, tensor dtypes, disjoint pairs, reload
# modes, the block-grid dims) are the ``PassContext`` built ahead of the
# prologue; ``freeze_pass_context`` seals the launch shape the prologue decided.
DEVICE_BODY_PROLOGUE_NAMES = [
    "lower_uniform_comparisons",
    "lower_collective_matmul",
    "lower_simt_gemm",
    "extend_gathered_mma_tma_launch",
    "adopt_owned_block_dims",
    "bounded_cache_prepare",
    "freeze_pass_context",
]

DEVICE_BODY_PASS_NAMES = [
    "finalize_shared_reduce_groups",
    "fuse_two_pass_loads",
    "vectorize_branch_local_persistent_fragments",
    "hoist_warp_reduce_from_vloop",
    "eliminate_duplicate_cluster_maxima",
    "bounded_cache_cache",
    "vectorize_affine_tile_lanes",
    "merge_literal_offset_loads",
    "fuse_fp4_pair_dot",
    "merge_sibling_v_loops",
    "optimize_vector_reductions",
    "hoist_lane_invariant_reductions",
    "hoist_factored_reductions",
    "fuse_fp8_pair_decode",
    "hoist_loop_invariant_recips",
    "fuse_fma",
    "ring_mma_dot",
    "fragment_mma_dot",
    "plan_lane_unroll",
    "unroll_lane_loads",
    "fold_lane_reductions",
    "split_lane_loads",
    "prefetch_reduction_packets",
    "pipeline_inner_loads",
    "pipeline_state_loads",
    "pack_fp32_constexpr_loops",
    "hoist_packed_factored_reductions",
    "pack_bf16_recurrences",
    "simplify_proven_loop_bounds",
    "simplify_proven_bounds",
    "fuse_cluster_online_pair",
    "lower_register_row_reduction",
    "fuse_rolled_cluster_online_pair",
    "split_rolled_vector_sums",
    "apply_exp2_fastmath",
    "validate_cluster_reduce_placement",
    "bounded_cache_finalize",
    "lower_full_tile_bounds",
    "enable_fp32_warp_redux",
    "lower_scalar_epilogue",
    "lower_philox_packets",
    "hoist_exact_invariant_divisions",
]

KERNEL_DEF_PASS_NAMES = [
    "lower_online_chunk_reduction",
    "reassociate_boolean_guards",
    "lower_published_scalars",
    "lower_register_producers",
    "validate_thread_axis_accesses",
    "lower_scalar_policy_loads",
    "lower_packet_store_policy",
    "validate_shared_arenas",
]


class TestCutePassPipeline(unittest.TestCase):
    def test_pass_context_compares_by_identity(self) -> None:
        # One context per render; a frozen dataclass with the default
        # ``eq=True`` would synthesize ``__eq__``/``__hash__`` over ``fn``,
        # ``env`` and the dict fields.
        self.assertIs(PassContext.__eq__, object.__eq__)
        self.assertIs(PassContext.__hash__, object.__hash__)

    def test_pass_order_is_pinned(self) -> None:
        self.assertEqual(
            [entry.name for entry in DEVICE_BODY_PROLOGUE], DEVICE_BODY_PROLOGUE_NAMES
        )
        self.assertEqual(
            [entry.name for entry in DEVICE_BODY_PASSES], DEVICE_BODY_PASS_NAMES
        )
        self.assertEqual(
            [entry.name for entry in KERNEL_DEF_PASSES], KERNEL_DEF_PASS_NAMES
        )

    def test_requires_provides_form_a_dag(self) -> None:
        # The import-time check accepted the tables; a reorder that breaks a
        # declared edge is rejected.  The caller's input set is left as is.
        inputs = {"pass_context", "renamed"}
        _check_pass_order(
            (
                ("prologue", DEVICE_BODY_PROLOGUE),
                ("body", DEVICE_BODY_PASSES),
                ("kernel_def", KERNEL_DEF_PASSES),
            ),
            inputs,
        )
        self.assertEqual(inputs, {"pass_context", "renamed"})
        swapped = list(DEVICE_BODY_PASSES)
        state_loads = next(
            i for i, e in enumerate(swapped) if e.name == "pipeline_state_loads"
        )
        fp32_packing = next(
            i for i, e in enumerate(swapped) if e.name == "pack_fp32_constexpr_loops"
        )
        swapped[state_loads], swapped[fp32_packing] = (
            swapped[fp32_packing],
            swapped[state_loads],
        )
        with self.assertRaisesRegex(AssertionError, "state_loads_pipelined"):
            _check_pass_order(
                (("prologue", DEVICE_BODY_PROLOGUE), ("body", tuple(swapped))),
                {"pass_context"},
            )
        with self.assertRaisesRegex(AssertionError, "twice"):
            _check_pass_order(
                (
                    (
                        "body",
                        (
                            DeviceBodyPass(
                                "a", lambda s: True, lambda s: None, provides=("x",)
                            ),
                            DeviceBodyPass(
                                "b", lambda s: True, lambda s: None, provides=("x",)
                            ),
                        ),
                    ),
                ),
                set(),
            )


if __name__ == "__main__":
    unittest.main()
