from __future__ import annotations

import copy
import operator
import unittest
from unittest.mock import patch

import torch

from test._cute_aux import _cpu_codegen

import helion
from helion._compiler import autotuner_heuristics
from helion._compiler.autotuner_heuristics import cute_row_prefetch
from helion._compiler.autotuner_heuristics.cute_row_packets import (
    CuteRowPacketHeuristic,
)
from helion._compiler.autotuner_heuristics.cute_row_packets import (
    append_row_packet_seeds,
)
from helion._compiler.autotuner_heuristics.cute_row_packets import row_packet_plan
from helion.exc import InvalidConfig
import helion.language as hl


@helion.kernel(backend="cute")
def _dependent_rows(
    source: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    batch, contexts, rows, features = source.shape
    features = hl.specialize(features)
    for b, context, r in hl.tile([batch, contexts, rows], block_size=[1, 1, None]):
        slot = slots[b.id].long()
        f = hl.arange(features)
        x = source[slot, context.id, r.index, f].float()
        w = weights[context.id, f].float()
        first = (x * w[None, :]).sum(-1)
        updated = x + first[:, None] * w[None, :]
        output[b.id, context.id, r] = updated.sum(-1)
        source[slot, context.id, r.index, f] = updated
    return output


@helion.kernel(backend="cute")
def _permuted_rows(
    bank: torch.Tensor,
    coefficient: torch.Tensor,
    remap: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
    outer, lanes, count, width = bank.shape
    width = hl.specialize(width)
    for lane, outer_tile, packet in hl.tile(
        [lanes, outer, count], block_size=[1, 1, None]
    ):
        entry = remap[outer_tile.id].long()
        column = hl.arange(width)
        values = bank[entry, lane.id, packet.index, column].float()
        factor = coefficient[lane.id, column].float()
        contraction = (values * factor[None, :]).sum(-1)
        replacement = values + contraction[:, None] * factor[None, :]
        answer[outer_tile.id, lane.id, packet] = replacement.sum(-1)
        bank[entry, lane.id, packet.index, column] = replacement
    return answer


@helion.kernel(backend="cute")
def _row_only(
    source: torch.Tensor, weights: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    rows, features = source.shape
    features = hl.specialize(features)
    for r in hl.tile(rows):
        f = hl.arange(features)
        x = source[r.index, f].float()
        w = weights[f].float()
        first = (x * w[None, :]).sum(-1)
        updated = x + first[:, None] * w[None, :]
        output[r] = updated.sum(-1)
        source[r.index, f] = updated
    return output


@helion.kernel(backend="cute")
def _independent_rows(
    source: torch.Tensor, weights: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    rows, features = source.shape
    features = hl.specialize(features)
    for r in hl.tile(rows):
        f = hl.arange(features)
        x = source[r.index, f].float()
        w = weights[f].float()
        first = (x * w[None, :]).sum(-1)
        second = (x + w[None, :]).sum(-1)
        output[r] = first + second
    return output


@helion.kernel(backend="cute")
def _ambiguous_context(
    source: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    batch, contexts, rows, features = source.shape
    features = hl.specialize(features)
    for b, context, r in hl.tile([batch, contexts, rows], block_size=[1, 1, None]):
        slot = slots[b.id].long()
        f = hl.arange(features)
        x = source[slot, context.id, r.index, f].float()
        y = source[b.id, context.id, r.index, f].float()
        w = weights[context.id, f].float()
        first = (x * w[None, :]).sum(-1)
        updated = y + first[:, None] * w[None, :]
        output[b.id, context.id, r] = updated.sum(-1)
        source[slot, context.id, r.index, f] = updated
    return output


def _args(
    features: int = 128, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((3, 5, 64, features), dtype=dtype),
        torch.empty((5, features), dtype=dtype),
        torch.arange(3, dtype=torch.int32),
        torch.empty((3, 5, 64)),
    )


def _bind(kernel, args):
    with _cpu_codegen():
        return kernel._bind_isolated(args)


def _bind_with_row_seed_stages(kernel, args):
    # Observe each append before later families extend the final seed pool.
    # Snapshots retain the exact prefix and detect mutations of caller seeds.
    stages = {}

    def observe(name, append):
        def record(env, ir, seeds):
            before = copy.deepcopy(seeds)
            result = append(env, ir, seeds)
            assert seeds == before
            assert name not in stages
            stages[name] = before, copy.deepcopy(result)
            return result

        return record

    with (
        patch.object(
            autotuner_heuristics,
            "append_row_packet_seeds",
            observe("packet", autotuner_heuristics.append_row_packet_seeds),
        ),
        patch.object(
            cute_row_prefetch,
            "append_row_prefetch_seeds",
            observe("prefetch", cute_row_prefetch.append_row_prefetch_seeds),
        ),
    ):
        bound = _bind(kernel, args)
    assert set(stages) == {"packet", "prefetch"}
    for before, result in stages.values():
        assert result[: len(before)] == before
        assert bound.config_spec.compiler_seed_configs[: len(result)] == result
    return bound, stages


class TestRowPackets(unittest.TestCase):
    def setUp(self):
        # Isolate the earlier family; combined append ordering is tested below.
        scope = patch(
            "helion._compiler.autotuner_heuristics.cute_row_prefetch.append_row_prefetch_seeds",
            side_effect=lambda env, ir, seeds: seeds,
        )
        scope.start()
        self.addCleanup(scope.stop)

    def plan(self, kernel, args):
        bound = _bind(kernel, args)
        plan = row_packet_plan(bound.env, bound.host_function.device_ir)
        return bound, plan

    def baseline(self, kernel, args):
        with patch.object(
            autotuner_heuristics,
            "append_row_packet_seeds",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            return _bind(kernel, args)

    def test_joint_four_carriers_and_prefix(self):
        args = _args()
        with patch.object(
            autotuner_heuristics,
            "append_row_packet_seeds",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            baseline, original_stages = _bind_with_row_seed_stages(
                _dependent_rows, args
            )
        bound, stages = _bind_with_row_seed_stages(_dependent_rows, args)
        plan = row_packet_plan(bound.env, bound.host_function.device_ir)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.traversal, (1, 2, 0))
        self.assertEqual((plan.feature_width, plan.row_width), (4, 4))
        old = original_stages["packet"][1]
        self.assertEqual(stages["packet"][0], old)
        seeds = stages["packet"][1]
        self.assertEqual(seeds[: len(old)], old)
        self.assertEqual(len(seeds), len(old) + 4)
        self.assertEqual(
            bound.config_spec.default_config(), baseline.config_spec.default_config()
        )
        expected = [(8, [4, 16], 2), (16, [4, 16], 4), (4, [2, 32], 2), (8, [2, 32], 4)]
        for seed, (block, threads, packet) in zip(
            seeds[len(old) :], expected, strict=True
        ):
            self.assertEqual(seed.config["block_sizes"], [block])
            self.assertEqual(seed.config["num_threads"], threads)
            self.assertEqual(seed.config["loop_orders"], [[1, 2, 0]])
            self.assertEqual(seed.config["cute_reduction_reloads"], ["register"])
            seq = bound.config_spec.cute_vector_widths
            widths = {
                s.block_id: v
                for s, v in zip(seq, seed.config["cute_vector_widths"], strict=True)
            }
            self.assertEqual(widths[plan.row], packet)
            self.assertEqual(widths[plan.reduction], 4)
            self.assertTrue(
                all(
                    v == 1
                    for key, v in widths.items()
                    if key not in (plan.row, plan.reduction)
                )
            )
            self.assertTrue(
                all(v == "blocked" for v in seed.config["cute_lane_layouts"])
            )
        with bound.env:
            generation = bound.config_spec.create_config_generation()
            before_generation = baseline.config_spec.create_config_generation()
            self.assertEqual(
                [(type(f), vars(f)) for f in generation.flat_spec],
                [(type(f), vars(f)) for f in before_generation.flat_spec],
            )
            self.assertEqual(
                generation._key_to_flat_indices, before_generation._key_to_flat_indices
            )
            for seed in seeds[len(old) :]:
                generation.strict_config_pair(seed)

    def test_names_and_axis_permutation(self):
        bound, stages = _bind_with_row_seed_stages(_permuted_rows, _args())
        plan = row_packet_plan(bound.env, bound.host_function.device_ir)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.traversal, (0, 2, 1))
        self.assertTrue(
            all(
                c.config["loop_orders"] == [[0, 2, 1]] for c in stages["packet"][1][-4:]
            )
        )

    def test_no_context_axes(self):
        bound, plan = self.plan(
            _row_only, (torch.empty(64, 128), torch.empty(128), torch.empty(64))
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.traversal, (0,))
        self.assertIn(
            CuteRowPacketHeuristic.name, bound.config_spec.autotuner_heuristics
        )

    def test_dtypes_and_extents(self):
        for extent in (64, 128, 256):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(extent=extent, dtype=dtype):
                    bound, plan = self.plan(_dependent_rows, _args(extent, dtype))
                    self.assertIsNotNone(plan)
                    self.assertEqual(
                        plan.feature_width,
                        16 // torch.empty((), dtype=dtype).element_size(),
                    )
                    self.assertIn(
                        CuteRowPacketHeuristic.name,
                        bound.config_spec.autotuner_heuristics,
                    )

    def test_feature_stride_declines(self):
        args = list(_args())
        args[0] = torch.empty((3, 5, 64, 256))[..., ::2]
        bound, plan = self.plan(_dependent_rows, tuple(args))
        self.assertIsNone(plan)
        self.assertNotIn(
            CuteRowPacketHeuristic.name, bound.config_spec.autotuner_heuristics
        )
        self.assertIn(
            "input_tensor_metadata", bound.env.compiler_fact_specialization_facts
        )

    def test_independent_reductions_decline(self):
        bound, plan = self.plan(
            _independent_rows, (torch.empty(64, 128), torch.empty(128), torch.empty(64))
        )
        self.assertIsNone(plan)
        self.assertNotIn(
            CuteRowPacketHeuristic.name, bound.config_spec.autotuner_heuristics
        )

    def test_ambiguous_anchor_declines(self):
        bound, plan = self.plan(_ambiguous_context, _args())
        self.assertIsNone(plan)
        self.assertNotIn(
            CuteRowPacketHeuristic.name, bound.config_spec.autotuner_heuristics
        )

    def test_overrides_dedup_and_disabled(self):
        args = _args()
        baseline = self.baseline(_dependent_rows, args)
        bound, plan = self.plan(_dependent_rows, args)
        self.assertIsNotNone(plan)
        old = copy.deepcopy(baseline.config_spec.compiler_seed_configs)
        old_image = copy.deepcopy(old)
        with bound.env:
            once = append_row_packet_seeds(
                bound.env, bound.host_function.device_ir, old
            )
            twice = append_row_packet_seeds(
                bound.env, bound.host_function.device_ir, once
            )
            self.assertEqual(once, twice)
            self.assertEqual(old, old_image)
            with patch.object(
                bound.env.settings, "autotune_config_overrides", {"block_sizes": [32]}
            ):
                self.assertIs(
                    append_row_packet_seeds(
                        bound.env, bound.host_function.device_ir, old
                    ),
                    old,
                )
        with patch.object(
            _dependent_rows.settings, "disable_autotuner_heuristics", True
        ):
            disabled = _bind(_dependent_rows, args)
        self.assertEqual(disabled.config_spec.compiler_seed_configs, [])
        self.assertIn(
            "input_tensor_metadata", disabled.env.compiler_fact_specialization_facts
        )

    def test_seed_decline_preserves_existing_pool(self):
        bound, stages = _bind_with_row_seed_stages(_dependent_rows, _args())
        plan = row_packet_plan(bound.env, bound.host_function.device_ir)
        self.assertIsNotNone(plan)
        ir = bound.host_function.device_ir
        original = stages["packet"][0]
        with bound.env:
            with patch.object(
                bound.env, "compiler_fact_specialization_facts", frozenset()
            ):
                self.assertIs(
                    append_row_packet_seeds(bound.env, ir, original), original
                )
            with patch.object(
                bound.config_spec,
                "normalized_config",
                side_effect=InvalidConfig("unsupported carrier"),
            ):
                self.assertIs(
                    append_row_packet_seeds(bound.env, ir, original), original
                )
            with patch.object(
                bound.settings, "autotune_config_overrides", {"block_sizes": [16]}
            ):
                extended = append_row_packet_seeds(bound.env, ir, original)
                self.assertEqual(extended[:-1], original)
                self.assertEqual(extended[-1].config["block_sizes"], [16])

    def test_unknown_structure_declines(self):
        bound, plan = self.plan(_dependent_rows, _args())
        self.assertIsNotNone(plan)
        ir = bound.host_function.device_ir
        grid = bound.config_spec.kernel_grid_fact
        for name, owner, attribute, replacement in (
            (
                "multiple_roots",
                bound.config_spec,
                "kernel_grid_fact",
                grid._replace(roots=grid.roots + grid.roots),
            ),
            ("multiple_graphs", ir, "graphs", [*ir.graphs, *ir.graphs]),
            ("rolled_reduction", bound.config_spec, "reduction_loops", [object()]),
        ):
            with self.subTest(name=name), patch.object(owner, attribute, replacement):
                self.assertIsNone(row_packet_plan(bound.env, ir))
        with patch.object(bound.env, "is_jagged_tile", return_value=True):
            self.assertIsNone(row_packet_plan(bound.env, ir))
        index_cast = next(
            n
            for n in ir.graphs[0].graph.nodes
            if n.target is torch.ops.prims.convert_element_type.default
            and n.args[1] == torch.int64
        )
        with patch.object(index_cast, "target", torch.ops.aten.abs.default):
            self.assertIsNone(row_packet_plan(bound.env, ir))

    def test_scaled_packet_coordinate_declines(self):
        bound, plan = self.plan(_dependent_rows, _args())
        self.assertIsNotNone(plan)
        ir = bound.host_function.device_ir
        graph = ir.graphs[0].graph
        feature = next(
            n for n in graph.nodes if n.target is torch.ops.prims.iota.default
        )
        users = tuple(feature.users)
        with graph.inserting_after(feature):
            scaled = graph.call_function(operator.mul, (feature, 2))
            scaled.meta["val"] = feature.meta["val"]
        try:
            for node in users:
                node.replace_input_with(feature, scaled)
            self.assertIsNone(row_packet_plan(bound.env, ir))
        finally:
            for node in users:
                node.replace_input_with(scaled, feature)
            graph.erase_node(scaled)

    def test_row_stride_cannot_request_wide_packet(self):
        args = list(_args())
        args[-1] = torch.empty_strided((3, 5, 64), (325, 65, 1))
        bound, plan = self.plan(_dependent_rows, tuple(args))
        self.assertIsNotNone(plan)
        self.assertEqual(plan.row_width, 1)
        self.assertNotIn(
            CuteRowPacketHeuristic.name, bound.config_spec.autotuner_heuristics
        )

    def test_view_offset_is_late_backend_authority(self):
        args = list(_args())
        args[-1] = torch.empty(3 * 5 * 64 + 1)[1:].view(3, 5, 64)
        self.assertEqual(args[-1].storage_offset(), 1)
        bound, plan = self.plan(_dependent_rows, tuple(args))
        self.assertIsNotNone(plan)
        self.assertEqual(plan.row_width, 4)
        self.assertIn(
            "input_tensor_metadata", bound.env.compiler_fact_specialization_facts
        )


if __name__ == "__main__":
    unittest.main()


def test_combined_row_packet_and_prefetch_preserve_original_prefix():
    args = _args()
    prefetch = "helion._compiler.autotuner_heuristics.cute_row_prefetch.append_row_prefetch_seeds"
    with patch(prefetch, side_effect=lambda env, ir, seeds: seeds):
        with patch.object(
            autotuner_heuristics,
            "append_row_packet_seeds",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            original, original_stages = _bind_with_row_seed_stages(
                _dependent_rows, args
            )
        packets, packet_stages = _bind_with_row_seed_stages(_dependent_rows, args)
    combined, combined_stages = _bind_with_row_seed_stages(_dependent_rows, args)
    old = original_stages["packet"][1]
    row = packet_stages["packet"][1]
    both = combined_stages["prefetch"][1]
    assert packet_stages["packet"][0] == old
    assert combined_stages["packet"] == packet_stages["packet"]
    assert combined_stages["prefetch"][0] == row
    assert row[: len(old)] == old and len(row) == len(old) + 4
    assert both[: len(row)] == row and len(both) == len(row) + 3
    assert packets.config_spec.default_config() == original.config_spec.default_config()
    assert (
        combined.config_spec.default_config() == original.config_spec.default_config()
    )
    assert both[-3]["cute_proven_bounds"] is True
    assert both[-3]["cute_lane_unroll"] == 8
