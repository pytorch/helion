from __future__ import annotations

import ast
import unittest

from helion._compiler.ast_read_writes import dead_assignment_elimination
from helion._compiler.ast_read_writes import definitely_does_not_have_side_effects


class TestAstReadWrites(unittest.TestCase):
    def test_compiler_shape_helpers_are_pure(self) -> None:
        for source in (
            "triton.cdiv(x, 4)",
            "triton.next_power_of_2(x)",
            "_cdiv(x, 4)",
            "_next_power_of_2(x)",
        ):
            expression = ast.parse(source, mode="eval").body
            self.assertTrue(definitely_does_not_have_side_effects(expression))

    def test_dead_shape_helper_assignments_are_removed_transitively(self) -> None:
        body = ast.parse(
            """
rdim = triton.next_power_of_2(block_size)
block_size = 128
"""
        ).body

        dead_assignment_elimination(body, ["rdim", "block_size"])

        self.assertEqual(body, [])


if __name__ == "__main__":
    unittest.main()
