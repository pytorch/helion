from __future__ import annotations

import ast

import torch

import helion
from helion._compiler.program_id import ForEachProgramID
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


def _dispatch_shape(tree: ast.AST) -> tuple[int, int]:
    pending = [(tree, 0)]
    branches = 0
    maximum_depth = 0
    while pending:
        node, depth = pending.pop()
        if isinstance(node, ast.If) and any(
            isinstance(part, ast.Name) and part.id == "pid_shared"
            for part in ast.walk(node.test)
        ):
            branches += 1
            depth += 1
        maximum_depth = max(maximum_depth, depth)
        pending.extend((child, depth) for child in ast.iter_child_nodes(node))
    return branches, maximum_depth


def independent_roots(
    x: torch.Tensor, outputs: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    out0, out1, out2, out3, out4 = outputs
    for tile in hl.tile(out0.numel(), block_size=256):
        out0[tile] = x[tile] + 1
    for tile in hl.tile(out1.numel(), block_size=256):
        out1[tile] = x[tile] + 2
    for tile in hl.tile(out2.numel(), block_size=256):
        out2[tile] = x[tile] + 3
    for tile in hl.tile(out3.numel(), block_size=256):
        out3[tile] = x[tile] + 4
    for tile in hl.tile(out4.numel(), block_size=256):
        out4[tile] = x[tile] + 5
    return outputs


class TestRootDispatchCodegen(TestCase):
    def test_balanced_dispatch_intervals(self):
        for roots in (1, 2, 3, 4, 5, 8, 33, 200):
            for pattern in ((1,), (0, 2, 0, 3, 1)):
                with self.subTest(roots=roots, pattern=pattern):
                    owner = ForEachProgramID("pid_shared")
                    expected = []
                    for root in range(roots):
                        body = ast.parse(
                            f"pid_shared -= {len(expected)}\n"
                            f"selected = ({root}, pid_shared)"
                        ).body
                        owner.case_bodies.append(body)
                        size = pattern[root % len(pattern)]
                        expected.extend((root, local_pid) for local_pid in range(size))
                        if root < roots - 1:
                            owner.case_branches.append(
                                ast.parse(
                                    f"if pid_shared < {len(expected)}: pass"
                                ).body[0]
                            )

                    tree = ast.Module(body=owner.codegen_dispatch(), type_ignores=[])
                    self.assertEqual(
                        _dispatch_shape(tree), (roots - 1, (roots - 1).bit_length())
                    )
                    code = compile(
                        ast.fix_missing_locations(tree), "<dispatch>", "exec"
                    )
                    for pid, selected in enumerate(expected):
                        scope = {"pid_shared": pid}
                        exec(code, scope)
                        self.assertEqual(scope["selected"], selected)

    def test_five_root_codegen(self):
        for static_shapes in (False, True):
            with self.subTest(static_shapes=static_shapes):
                kernel = helion.kernel(
                    independent_roots,
                    backend="triton",
                    static_shapes=static_shapes,
                    autotune_effort="none",
                )
                x = torch.empty(1024)
                outputs = tuple(torch.empty(size) for size in (0, 2, 257, 0, 1024))
                bound = kernel.bind((x, outputs))
                code = bound.to_code(helion.Config(pid_type="flat", num_warps=4))
                self.assertEqual(len(bound.host_function.device_ir.root_ids), 5)
                self.assertEqual(_dispatch_shape(ast.parse(code)), (4, 3))


@onlyBackends(["triton", "cute"])
class TestRootDispatchExecution(RefEagerTestBase, TestCase):
    @skipIfRefEager("checks generated root dispatch")
    def test_five_root_dispatch(self):
        for static_shapes in (False, True):
            for pid_type in ("flat", "persistent_blocked", "persistent_interleaved"):
                for sizes in ((65536,) * 5, (0, 2, 257, 0, 1024)):
                    with self.subTest(
                        static_shapes=static_shapes, pid_type=pid_type, sizes=sizes
                    ):
                        kernel = helion.kernel(
                            independent_roots,
                            static_shapes=static_shapes,
                            autotune_effort="none",
                        )
                        x = torch.arange(max(sizes), device=DEVICE, dtype=torch.float32)
                        outputs = tuple(
                            torch.full((size,), float("nan"), device=DEVICE)
                            for size in sizes
                        )
                        code, returned = code_and_output(
                            kernel, (x, outputs), pid_type=pid_type, num_warps=4
                        )
                        self.assertEqual(_dispatch_shape(ast.parse(code)), (4, 3))
                        for root, result in enumerate(returned):
                            self.assertIs(result, outputs[root])
                            torch.testing.assert_close(
                                result, x[: sizes[root]] + root + 1
                            )

                        x.add_(7)
                        for output in outputs:
                            output.fill_(float("nan"))
                        _, returned = code_and_output(
                            kernel, (x, outputs), pid_type=pid_type, num_warps=4
                        )
                        for root, result in enumerate(returned):
                            self.assertIs(result, outputs[root])
                            torch.testing.assert_close(
                                result, x[: sizes[root]] + root + 1
                            )
