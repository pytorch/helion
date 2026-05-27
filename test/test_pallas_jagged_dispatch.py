"""Unit tests for the pair-based jagged dispatch detector.

Tests the detector in isolation by constructing a fake ``CompileEnvironment``
with controlled ``jagged_tile_parent_ids``, so no kernel binding is needed.
End-to-end coverage is provided by the example tests that lower
``examples/jagged_*.py`` kernels through the Pallas backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import unittest

from helion._compiler.pallas.jagged_dispatch import detect_jagged_dispatch


@dataclass
class _FakeEnv:
    """Minimal stand-in — the detector reads only this one attribute."""

    jagged_tile_parent_ids: dict[int, list[int]] = field(default_factory=dict)


class TestJaggedDispatchDetect(unittest.TestCase):
    def test_no_jagged_tile_returns_none(self) -> None:
        # Non-jagged kernel: empty dict.
        env = _FakeEnv()
        self.assertIsNone(detect_jagged_dispatch(env))  # type: ignore[arg-type]

    def test_single_pair_dispatches(self) -> None:
        # jagged_sum-style: one jagged child, one parent.
        # tile_b has block_id=0; tile_k=hl.jagged_tile(nnz) has block_id=1
        # with parent_ids=[0].
        env = _FakeEnv(jagged_tile_parent_ids={1: [0]})
        self.assertEqual(detect_jagged_dispatch(env), 0)  # type: ignore[arg-type]

    def test_two_pairs_same_parent_dispatches(self) -> None:
        # jagged_mean-style: tile_m and tile_k both children of tile_b.
        env = _FakeEnv(jagged_tile_parent_ids={1: [0], 2: [0]})
        self.assertEqual(detect_jagged_dispatch(env), 0)  # type: ignore[arg-type]

    def test_many_pairs_same_parent_dispatches(self) -> None:
        # Unbounded number of children sharing one parent → still
        # dispatches.
        env = _FakeEnv(jagged_tile_parent_ids={1: [0], 2: [0], 3: [0], 4: [0]})
        self.assertEqual(detect_jagged_dispatch(env), 0)  # type: ignore[arg-type]

    def test_multi_parent_child_returns_none(self) -> None:
        # A jagged_tile whose nnz depends on TWO outer tiles.
        env = _FakeEnv(jagged_tile_parent_ids={2: [0, 1]})
        self.assertIsNone(detect_jagged_dispatch(env))  # type: ignore[arg-type]

    def test_zero_parent_child_returns_none(self) -> None:
        # Defensive: should never happen (jagged_tile always has a
        # parent), but the detector must not crash on len==0.
        env = _FakeEnv(jagged_tile_parent_ids={1: []})
        self.assertIsNone(detect_jagged_dispatch(env))  # type: ignore[arg-type]

    def test_multiple_items_axes_returns_none(self) -> None:
        # Two children, two distinct parents → kernel has more than one
        # items axis. Not supported.
        env = _FakeEnv(jagged_tile_parent_ids={2: [0], 3: [1]})
        self.assertIsNone(detect_jagged_dispatch(env))  # type: ignore[arg-type]

    def test_mixed_one_valid_one_multi_parent_returns_none(self) -> None:
        # Defensive: even a single multi-parent child taints the
        # dispatch — fall through.
        env = _FakeEnv(jagged_tile_parent_ids={2: [0], 3: [0, 1]})
        self.assertIsNone(detect_jagged_dispatch(env))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
