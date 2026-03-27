from __future__ import annotations

import logging
import unittest

import sympy

from helion._compiler.backend import TritonBackend
from helion.autotuner.config_generation import TRITON_MAX_TENSOR_NUMEL
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import TensorNumelConstraint


def _make_constraint(
    symbols: list[sympy.Symbol],
    numel_expr: sympy.Expr,
    block_indices: tuple[int, ...],
) -> TensorNumelConstraint:
    """Helper to build a TensorNumelConstraint from a sympy expression."""
    from helion._compiler.compile_environment import _make_numel_check

    constraint_expr = numel_expr <= TRITON_MAX_TENSOR_NUMEL
    return TensorNumelConstraint(
        check_fn=_make_numel_check(symbols, constraint_expr),
        block_indices=block_indices,
        expr_str=str(constraint_expr),
    )


def _check_numel_constraints(gen: ConfigGeneration, flat_config: list[object]) -> bool:
    """Return True if all tensor numel constraints are satisfied."""
    from typing import cast

    for constraint in gen.config_spec.tensor_numel_constraints:
        args = [
            cast("int", flat_config[gen.block_size_indices[i]])
            for i in constraint.block_indices
        ]
        if not constraint.check_fn(*args):
            return False
    return True


def _make_spec_and_gen(
    *,
    size_hints: list[int],
    min_sizes: list[int] | None = None,
    max_sizes: list[int] | None = None,
    constraints: list[TensorNumelConstraint] | None = None,
) -> tuple[ConfigSpec, ConfigGeneration]:
    """Create a ConfigSpec + ConfigGeneration with the given block sizes and constraints."""
    config_spec = ConfigSpec(backend=TritonBackend())
    for i, hint in enumerate(size_hints):
        config_spec.block_sizes.append(
            BlockSizeSpec(
                block_id=i,
                size_hint=hint,
                min_size=(min_sizes[i] if min_sizes else 1),
                max_size=(max_sizes[i] if max_sizes else None),
            )
        )
    if constraints:
        config_spec.tensor_numel_constraints.extend(constraints)
    gen = ConfigGeneration(config_spec)
    return config_spec, gen


class TestTensorNumelConstraintType(unittest.TestCase):
    """Tests for the TensorNumelConstraint NamedTuple."""

    def test_fields_accessible(self) -> None:
        c = TensorNumelConstraint(
            check_fn=lambda x: x <= 100,
            block_indices=(0,),
            expr_str="x <= 100",
        )
        self.assertEqual(c.block_indices, (0,))
        self.assertEqual(c.expr_str, "x <= 100")
        self.assertTrue(c.check_fn(50))
        self.assertFalse(c.check_fn(200))

    def test_tuple_unpacking(self) -> None:
        c = TensorNumelConstraint(
            check_fn=lambda x: True,
            block_indices=(0, 1),
            expr_str="expr",
        )
        check_fn, indices, expr_str = c
        self.assertEqual(indices, (0, 1))
        self.assertEqual(expr_str, "expr")


class TestCheckTensorNumelConstraints(unittest.TestCase):
    """Tests for constraint checking on flat configs."""

    def test_no_constraints_always_valid(self) -> None:
        _, gen = _make_spec_and_gen(size_hints=[256, 256])
        flat = gen.default_flat()
        self.assertTrue(_check_numel_constraints(gen, flat))

    def test_satisfied_constraint(self) -> None:
        b0, b1 = sympy.symbols("b0 b1")
        # b0 * b1 <= 1048576 — with defaults of 256, product is 65536
        constraint = _make_constraint([b0, b1], b0 * b1, (0, 1))
        _, gen = _make_spec_and_gen(size_hints=[256, 256], constraints=[constraint])
        flat = gen.default_flat()
        self.assertTrue(_check_numel_constraints(gen, flat))

    def test_violated_constraint(self) -> None:
        b0, b1 = sympy.symbols("b0 b1")
        # b0 * b1 * 16384 <= 1048576 — with defaults of 256 each, product = 256*256*16384 >> limit
        constraint = _make_constraint([b0, b1], b0 * b1 * 16384, (0, 1))
        _, gen = _make_spec_and_gen(size_hints=[256, 256], constraints=[constraint])
        # Manually construct a flat config with large block sizes
        flat = gen.default_flat()
        # Force large values into block size slots
        for i in gen.block_size_indices:
            flat[i] = 256
        self.assertFalse(_check_numel_constraints(gen, flat))

    def test_single_block_constraint(self) -> None:
        b0 = sympy.Symbol("b0")
        # b0 * 262144 <= 1048576 => b0 <= 4
        constraint = _make_constraint([b0], b0 * 262144, (0,))
        _, gen = _make_spec_and_gen(size_hints=[64], constraints=[constraint])
        flat = gen.default_flat()
        for i in gen.block_size_indices:
            flat[i] = 4
        self.assertTrue(_check_numel_constraints(gen, flat))
        for i in gen.block_size_indices:
            flat[i] = 8
        self.assertFalse(_check_numel_constraints(gen, flat))


class TestShrinkForNumelConstraints(unittest.TestCase):
    """Tests for the shrinking logic (both ConfigGeneration and ConfigSpec paths)."""

    def test_shrinks_until_satisfied(self) -> None:
        b0, b1 = sympy.symbols("b0 b1")
        # b0 * b1 * 16384 <= 1048576 => b0 * b1 <= 64
        constraint = _make_constraint([b0, b1], b0 * b1 * 16384, (0, 1))
        _, gen = _make_spec_and_gen(size_hints=[256, 256], constraints=[constraint])
        flat = gen.default_flat()
        # After shrinking, the constraint must hold
        self.assertTrue(_check_numel_constraints(gen, flat))
        bs0 = flat[gen.block_size_indices[0]]
        bs1 = flat[gen.block_size_indices[1]]
        assert isinstance(bs0, int) and isinstance(bs1, int)
        self.assertLessEqual(bs0 * bs1 * 16384, TRITON_MAX_TENSOR_NUMEL)

    def test_prefers_largest_block_size(self) -> None:
        """Verify the shrinking strategy halves the largest block first,
        producing balanced tile shapes."""
        b0, b1 = sympy.symbols("b0 b1")
        # b0 * b1 <= 1048576 => satisfied for 1024*1024, but let's use a tighter one
        # b0 * b1 * 4096 <= 1048576 => b0 * b1 <= 256
        constraint = _make_constraint([b0, b1], b0 * b1 * 4096, (0, 1))
        spec, gen = _make_spec_and_gen(
            size_hints=[512, 512],
            max_sizes=[512, 512],
            constraints=[constraint],
        )
        flat = gen.default_flat()
        bs0 = flat[gen.block_size_indices[0]]
        bs1 = flat[gen.block_size_indices[1]]
        assert isinstance(bs0, int) and isinstance(bs1, int)
        # Defaults are 32 each (2 dims, no reduction).  32*32*4096=4M > 1M.
        # Halve-largest-first: (32,32)->(16,32)->(16,16).  16*16*4096=1M.
        self.assertEqual(bs0, 16)
        self.assertEqual(bs1, 16)
        self.assertLessEqual(bs0 * bs1 * 4096, TRITON_MAX_TENSOR_NUMEL)

    def test_respects_min_size(self) -> None:
        b0 = sympy.Symbol("b0")
        # b0 * 524288 <= 1048576 => b0 <= 2, but min_size=16
        constraint = _make_constraint([b0], b0 * 524288, (0,))
        _, gen = _make_spec_and_gen(
            size_hints=[64], min_sizes=[16], constraints=[constraint]
        )
        flat = gen.default_flat()
        bs0 = flat[gen.block_size_indices[0]]
        assert isinstance(bs0, int)
        # Can't shrink below min_size=16, so constraint is unsatisfiable
        self.assertGreaterEqual(bs0, 16)

    def test_unsatisfiable_logs_warning(self) -> None:
        b0 = sympy.Symbol("b0")
        # b0 * 524288 <= 1048576 => b0 <= 2, but min_size=16 => can't satisfy
        constraint = _make_constraint([b0], b0 * 524288, (0,))
        _, gen = _make_spec_and_gen(
            size_hints=[64], min_sizes=[16], constraints=[constraint]
        )
        with self.assertLogs(
            "helion.autotuner.config_spec", level=logging.WARNING
        ) as cm:
            gen.default_flat()
        self.assertTrue(
            any("unsatisfiable" in msg for msg in cm.output),
            f"Expected 'unsatisfiable' warning, got: {cm.output}",
        )

    def test_config_spec_shrink_also_warns(self) -> None:
        b0 = sympy.Symbol("b0")
        constraint = _make_constraint([b0], b0 * 524288, (0,))
        spec, _ = _make_spec_and_gen(
            size_hints=[64], min_sizes=[16], constraints=[constraint]
        )
        with self.assertLogs(
            "helion.autotuner.config_spec", level=logging.WARNING
        ) as cm:
            spec.default_config()
        self.assertTrue(any("unsatisfiable" in msg for msg in cm.output))

    def test_multiple_constraints(self) -> None:
        b0, b1, b2 = sympy.symbols("b0 b1 b2")
        c1 = _make_constraint([b0, b1], b0 * b1 * 4096, (0, 1))
        c2 = _make_constraint([b1, b2], b1 * b2 * 4096, (1, 2))
        _, gen = _make_spec_and_gen(size_hints=[256, 256, 256], constraints=[c1, c2])
        flat = gen.default_flat()
        self.assertTrue(_check_numel_constraints(gen, flat))

    def test_config_spec_default_config_satisfies_constraints(self) -> None:
        """ConfigSpec.default_config() should produce a valid config."""
        b0, b1 = sympy.symbols("b0 b1")
        constraint = _make_constraint([b0, b1], b0 * b1 * 16384, (0, 1))
        spec, _ = _make_spec_and_gen(size_hints=[256, 256], constraints=[constraint])
        config = spec.default_config()
        block_sizes = config.config["block_sizes"]
        self.assertLessEqual(
            block_sizes[0] * block_sizes[1] * 16384, TRITON_MAX_TENSOR_NUMEL
        )


class TestDeduplication(unittest.TestCase):
    """Tests for constraint deduplication during extraction."""

    def test_identical_constraints_deduplicated(self) -> None:
        """Two constraints with the same expr_str should be deduplicated."""
        b0, b1 = sympy.symbols("b0 b1")
        c1 = _make_constraint([b0, b1], b0 * b1 * 4096, (0, 1))
        c2 = _make_constraint([b0, b1], b0 * b1 * 4096, (0, 1))
        # Verify they have the same expr_str
        self.assertEqual(c1.expr_str, c2.expr_str)
        # But direct append doesn't deduplicate — that's compile_environment's job.
        # Just verify the NamedTuple equality works as expected.
        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
        spec.block_sizes.append(BlockSizeSpec(block_id=1, size_hint=64))
        # Simulate dedup logic: only add if expr_str not seen
        seen: set[str] = set()
        for c in [c1, c2]:
            if c.expr_str not in seen:
                seen.add(c.expr_str)
                spec.tensor_numel_constraints.append(c)
        self.assertEqual(len(spec.tensor_numel_constraints), 1)

    def test_different_constraints_kept(self) -> None:
        b0, b1 = sympy.symbols("b0 b1")
        c1 = _make_constraint([b0, b1], b0 * b1 * 4096, (0, 1))
        c2 = _make_constraint([b0, b1], b0 * b1 * 8192, (0, 1))
        self.assertNotEqual(c1.expr_str, c2.expr_str)


class TestShrinkConfig(unittest.TestCase):
    """Test that shrink_config (used by random_flat) enforces numel constraints."""

    def test_random_flat_satisfies_constraints(self) -> None:
        b0, b1 = sympy.symbols("b0 b1")
        constraint = _make_constraint([b0, b1], b0 * b1 * 16384, (0, 1))
        _, gen = _make_spec_and_gen(size_hints=[256, 256], constraints=[constraint])
        # Generate several random configs — all must satisfy constraints
        for _ in range(10):
            flat = gen.random_flat()
            self.assertTrue(
                _check_numel_constraints(gen, flat),
                f"random_flat produced invalid config: "
                f"block_sizes={[flat[i] for i in gen.block_size_indices]}",
            )


class TestMixedSymbolExtraction(unittest.TestCase):
    """Constraints with non-block-size free symbols must be skipped."""

    def test_mixed_symbol_shape_skipped(self) -> None:
        """A tensor shape like block_size_0 * runtime_dim should not produce
        a constraint, because lambdify cannot substitute runtime_dim."""
        from types import SimpleNamespace

        from helion._compiler.compile_environment import CompileEnvironment

        b0 = sympy.Symbol("u0", integer=True)
        runtime_dim = sympy.Symbol("u8", integer=True)

        # Minimal mock of CompileEnvironment for _extract_tensor_numel_constraints
        env = object.__new__(CompileEnvironment)
        env.block_sizes = [SimpleNamespace(symbol=lambda: b0, block_id=0)]
        env.config_spec = SimpleNamespace(
            block_sizes=SimpleNamespace(
                block_id_to_index=lambda bid: bid,
            ),
            tensor_numel_constraints=[],
        )
        # Pure block-size shape: should produce a constraint
        # Mixed shape: should be skipped
        env.kernel_tensor_sizes = [
            [b0, sympy.Integer(16384)],
            [b0, runtime_dim],
        ]
        env._extract_tensor_numel_constraints()
        self.assertEqual(
            len(env.config_spec.tensor_numel_constraints),
            1,
            "Expected 1 constraint (pure block-size shape only); "
            "mixed-symbol shape should be skipped",
        )


class TestFixedPointOverlapping(unittest.TestCase):
    """Verify the fixed-point loop handles overlapping constraints."""

    def test_overlapping_constraints_all_satisfied(self) -> None:
        """Two constraints sharing block_sizes[1]: shrinking for C1 may
        violate C2, so a fixed-point pass is needed."""
        b0, b1, b2 = sympy.symbols("b0 b1 b2")
        # C1: b0 * b1 * 4096 <= 1048576  => b0*b1 <= 256
        # C2: b1 * b2 * 4096 <= 1048576  => b1*b2 <= 256
        c1 = _make_constraint([b0, b1], b0 * b1 * 4096, (0, 1))
        c2 = _make_constraint([b1, b2], b1 * b2 * 4096, (1, 2))
        _, gen = _make_spec_and_gen(size_hints=[256, 256, 256], constraints=[c1, c2])
        flat = gen.default_flat()
        self.assertTrue(_check_numel_constraints(gen, flat))
        bs = [flat[i] for i in gen.block_size_indices]
        assert all(isinstance(b, int) for b in bs)
        self.assertLessEqual(bs[0] * bs[1] * 4096, TRITON_MAX_TENSOR_NUMEL)
        self.assertLessEqual(bs[1] * bs[2] * 4096, TRITON_MAX_TENSOR_NUMEL)


if __name__ == "__main__":
    unittest.main()
