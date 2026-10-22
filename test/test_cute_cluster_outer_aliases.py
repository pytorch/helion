from __future__ import annotations

import ast
import unittest

from helion._compiler.cute import cluster_online_pair as pair
from helion._compiler.cute import rolled_cluster_online_pair as rolled


def _program() -> list[ast.stmt]:
    return ast.parse(
        """
row = cutlass.Int32(cute.arch.block_idx()[0])
sequence = row // 1024
raw_row = row + sequence
maximum_acc = cutlass.Float32(float('-inf'))
for offset in range(0, 4096, 2048):
    for lane in range(2):
        base = offset + cutlass.Int32(cute.arch.thread_idx()[0]) * 8 + lane * 1024
        mask = base < 32000
        for element in cutlass.range_constexpr(8):
            index = base + element
            value = cutlass.Float32((matrix.iterator + raw_row * 32000 + index).load() if raw_row < 4100 and mask else cutlass.BFloat16(0))
            term = cutlass.Float32(value) if mask else cutlass.Float32(float('-inf'))
            maximum_acc = cute.arch.fmax(maximum_acc, cutlass.Float32(term), nan=True)
reduce_lane = cutlass.Int32(cute.arch.thread_idx()[0])
reduced_max = _cute_grouped_reduce_cluster(maximum_acc, 'max', cutlass.Float32(float('-inf')), reduce_lane, buffer_a, barrier_a, group_span=256, cluster_n=2)
maximum = cutlass.Float32(reduced_max)
denominator_acc = cutlass.Float32(0)
for other_offset in range(0, 4096, 2048):
    for other_lane in range(2):
        other_base = other_offset + cutlass.Int32(cute.arch.thread_idx()[0]) * 8 + other_lane * 1024
        other_mask = other_base < 32000
        for other_element in cutlass.range_constexpr(8):
            other_index = other_base + other_element
            equivalent_row = row + sequence
            other_value = cutlass.Float32((matrix.iterator + equivalent_row * 32000 + other_index).load() if equivalent_row < 4100 and other_mask else cutlass.BFloat16(0))
            difference = other_value - maximum
            exponential = cute.math.exp2(cutlass.Float32(difference) * 1.4426950408889634, fastmath=True)
            other_term = cutlass.Float32(exponential) if other_mask else cutlass.Float32(0)
            denominator_acc = denominator_acc + cutlass.Float32(other_term)
other_reduce_lane = cutlass.Int32(cute.arch.thread_idx()[0])
reduced_sum = _cute_grouped_reduce_cluster(denominator_acc, 'sum', cutlass.Float32(0), other_reduce_lane, buffer_b, barrier_b, group_span=256, cluster_n=2)
denominator = cutlass.Float32(reduced_sum)
"""
    ).body


def _sites(body: list[ast.stmt]) -> tuple[list[ast.stmt], pair._Site, pair._Site]:
    working, owners = rolled._wrap_sites(body)
    sites = pair._find_sites(working)
    assert sites is not None and len(sites) == 2
    assert all(id(site.for_node) in owners for site in sites)
    return working, sites[0], sites[1]


class TestCuTeClusterOuterAliases(unittest.TestCase):
    def test_equivalent_scalar_row_addresses(self) -> None:
        body, first, second = _sites(_program())
        self.assertIsNotNone(rolled._prove(body, first, second))

    def test_reassigned_alias_and_dependencies_decline(self) -> None:
        for statement in (
            "raw_row = raw_row + 1",
            "sequence = sequence + 1",
            "if predicate:\n    sequence = sequence + 1",
            "for sequence in range(2):\n    pass",
        ):
            with self.subTest(statement=statement):
                body, first, second = _sites(_program())
                body.insert(second.top_idx, ast.parse(statement).body[0])
                sites = pair._find_sites(body)
                assert sites is not None and len(sites) == 2
                self.assertIsNone(rolled._prove(body, *sites))

    def test_memory_load_alias_declines(self) -> None:
        program = _program()
        aliases = [
            statement
            for statement in program
            if pair._is_name_assign(statement)
            and pair._assigned_name(statement) == "raw_row"
        ]
        self.assertEqual(len(aliases), 1)
        aliases[0].value = ast.parse(
            "(row_indices.iterator + row).load()", mode="eval"
        ).body
        body, first, second = _sites(program)
        self.assertIsNone(rolled._prove(body, first, second))

    def test_different_load_mask_declines(self) -> None:
        body, first, second = _sites(_program())
        for node in ast.walk(second.for_node):
            if isinstance(node, ast.Constant) and node.value == 4100:
                node.value = 4099
        self.assertIsNone(rolled._prove(body, first, second))

    def test_different_conversion_declines(self) -> None:
        body, first, second = _sites(_program())
        assignments = [
            node
            for node in ast.walk(second.for_node)
            if pair._is_name_assign(node) and pair._assigned_name(node) == "other_value"
        ]
        self.assertEqual(len(assignments), 1)
        cast = assignments[0].value
        assert isinstance(cast, ast.Call) and isinstance(cast.func, ast.Attribute)
        cast.func.attr = "Float16"
        self.assertIsNone(rolled._prove(body, first, second))


if __name__ == "__main__":
    unittest.main()
