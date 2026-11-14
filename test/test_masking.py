from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._compiler.aten_lowering import matmul_masked_value
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.language as hl
from helion.runtime.settings import _get_backend


class TestMatmulMaskedValue(unittest.TestCase):
    def test_requires_fast_math_and_two_zero_padded_operands(self) -> None:
        graph = torch.fx.Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        lhs.meta["masked_value"] = 0
        rhs.meta["masked_value"] = 0
        mm = graph.call_function(torch.ops.aten.mm.default, (lhs, rhs))

        env = types.SimpleNamespace(settings=types.SimpleNamespace(fast_math=False))
        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertIsNone(matmul_masked_value(mm))

            env.settings.fast_math = True
            self.assertEqual(matmul_masked_value(mm), 0)

            rhs.meta["masked_value"] = None
            self.assertIsNone(matmul_masked_value(mm))


@onlyBackends(["triton", "cute"])
class TestMasking(RefEagerTestBase, TestCase):
    def test_mask_dot(self):
        @helion.kernel(config={"block_sizes": [[32, 32], 32]}, dot_precision="ieee")
        def add1mm(x, y):
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n])
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k] + 1, y[tile_k, tile_n] + 1)
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([100, 100], device=DEVICE),
            torch.randn([100, 100], device=DEVICE),
        )
        code, result = code_and_output(
            add1mm,
            args,
        )
        torch.testing.assert_close(
            result, (args[0] + 1) @ (args[1] + 1), rtol=1e-2, atol=1e-1
        )

    def test_no_mask_views0(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m, 1], device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :][:, :, None].sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1, keepdim=True))
        self.assertNotIn("tl.where", code)

    def test_no_mask_views1(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].transpose(0, 1).sum(dim=0)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_no_mask_full0(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                v = torch.zeros_like(x[tile_m, :])
                out[tile_m] = v.sum(-1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.zeros_like(args[0]).sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_no_mask_full1(self):
        @helion.kernel(config={"block_size": [32, 32]})
        def fn(x):
            m, n = x.size()
            hl.specialize(n)
            out = torch.empty([m], device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                v = hl.zeros([tile_m, tile_n])
                out[tile_m] = v.sum(-1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.zeros_like(args[0]).sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_mask_offset(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = (x[tile_m, :] + 1).sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, (args[0] + 1).sum(dim=1))
        if _get_backend() == "cute":
            self.assertIn("if mask_0 and mask_1 else cutlass.Float32(0)", code)
        else:
            self.assertIn("tl.where", code)

    def test_no_mask_inductor_ops(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x0 = x[tile_m, :] + 1
                x1 = x0 - 1
                # +1-1 cancels out, so no masking needed
                out[tile_m] = x1.sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1))
        self.assertNotIn("tl.where", code)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_under_an_if_without_else(self):
        """A carry updated in a runtime ``if`` without ``else``, or a carry
        loop under one: the ``if`` passes the carry through on its other
        path, a result that has no branch output.  Summed in a later loop
        over its block, the carry the loop under the ``if`` leaves keeps its
        stale lanes out of the sum there too."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def add_one_if(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for _ in hl.tile(n, block_size=block_n):
                    if x[0, 0] > 0:
                        acc += 1
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def sum_if(x, flag):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                if flag[0] > 0:
                    for tile_n in hl.tile(n, block_size=block_n):
                        acc = acc + x[tile_m, tile_n]
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def later_loop_if(x, flag):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                if flag[0] > 0:
                    for tile_n in hl.tile(n, block_size=block_n):
                        acc = acc + x[tile_m, tile_n]
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    total = total + (acc + x[tile_m, tile_n]).sum(dim=1)
                out[tile_m] = total
            return out

        x = torch.randint(1, 5, [64, 20], device=DEVICE).float()
        zeros = torch.zeros([64], device=DEVICE)
        # The +1 reaches all 16 lanes in both iterations, padded ones
        # included (test_loop_carry_masking): 32 per row.
        torch.testing.assert_close(add_one_if(x), zeros + 32, rtol=0, atol=0)
        torch.testing.assert_close(add_one_if(-x), zeros, rtol=0, atol=0)
        for flag, expected in ((1.0, x.sum(dim=1)), (-1.0, zeros)):
            torch.testing.assert_close(
                sum_if(x, torch.tensor([flag], device=DEVICE)),
                expected,
                rtol=0,
                atol=0,
            )
        # The carry over the two 16-column tiles, then the later loop's sum
        # of it plus each tile's live columns.
        acc = torch.zeros([64, 16], device=DEVICE)
        for start in range(0, 20, 16):
            live = min(16, 20 - start)
            acc[:, :live] += x[:, start : start + live]
        later = torch.zeros([64], device=DEVICE)
        for start in range(0, 20, 16):
            live = min(16, 20 - start)
            later += (acc[:, :live] + x[:, start : start + live]).sum(dim=1)
        for flag, expected in ((1.0, later), (-1.0, x.sum(dim=1))):
            torch.testing.assert_close(
                later_loop_if(x, torch.tensor([flag], device=DEVICE)),
                expected,
                rtol=0,
                atol=0,
            )

    def test_if_pass_through_after_a_branch_local_variable(self):
        """The if branch assigns ``t``, which its graph returns but the
        conditional does not, so the result passing the outer ``v`` through
        is not the branch's second output: its masked value (1, from ``x +
        1``) keeps the mask before the sum.  With an else branch writing
        ``v`` and ``w``, the else outputs follow the if branch's ``t`` too,
        and the if side passes the outer ``w`` (masked value 2) through:
        either one taken for ``t`` would drop a mask the sum needs.  A scalar
        the if branch assigns before ``v`` (``t = 1``) is not an output of
        its graph, which returns only tensors: counted, it made ``v``'s slot
        name the branch-local ``_w = v * 2`` that follows."""

        @helion.kernel(config={"block_sizes": [4]}, static_shapes=True)
        def fn(x, flag):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                v = x[tile_m, :] + 1
                if flag[0] > 0:
                    t = v * 0
                    v = t - t
                out[tile_m] = v.sum(-1)
            return out

        @helion.kernel(config={"block_sizes": [4]}, static_shapes=True)
        def with_else(x, flag):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                v = x[tile_m, :] + 1
                w = x[tile_m, :] + 2
                if flag[0] > 0:
                    t = v * 0
                    v = t - t
                else:
                    v = v + 1
                    w = w * 0
                out[tile_m] = v.sum(-1) + w.sum(-1)
            return out

        @helion.kernel(config={"block_sizes": [4]}, static_shapes=True)
        def scalar_local(x, flag):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                if flag[0] > 0:
                    t = 1
                    v = x[tile_m, :] + t
                    _w = v * 2
                else:
                    v = x[tile_m, :] + 1
                out[tile_m] = v.sum(-1)
            return out

        # 20 columns in a 32-wide reduction: the sum must skip 12 lanes.
        x = torch.randint(1, 5, [64, 20], device=DEVICE).float()
        for flag, expected in (
            (1.0, torch.zeros([64], device=DEVICE)),
            (-1.0, (x + 1).sum(-1)),
        ):
            flag = torch.tensor([flag], device=DEVICE)
            torch.testing.assert_close(fn(x, flag), expected, rtol=0, atol=0)
            # v or w is x + 2, the other one 0.
            torch.testing.assert_close(
                with_else(x, flag), (x + 2).sum(-1), rtol=0, atol=0
            )
            torch.testing.assert_close(
                scalar_local(x, flag), (x + 1).sum(-1), rtol=0, atol=0
            )

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_masking(self):
        @helion.kernel(config={"block_sizes": [32, 32]})
        def fn(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for _ in hl.tile(n, block_size=block_n):
                    # The first iteration, this doesn't need a mask -- but the second does
                    acc += acc.sum(dim=1, keepdim=True)
                    acc += 1
                out[tile_m] = acc.sum(dim=1)
            return out

        for width in (33, 64, 100):
            with self.subTest(width=width):
                args = (torch.randn([100, width], device=DEVICE),)
                _, result = code_and_output(fn, args)
                # The reduction inside each tile sees only its live columns.
                # The final reduction sees the complete 32-element accumulator.
                expected_acc = torch.zeros((100, 32), device=DEVICE)
                for start in range(0, width, 32):
                    live_columns = min(32, width - start)
                    expected_acc += expected_acc[:, :live_columns].sum(
                        dim=1, keepdim=True
                    )
                    expected_acc += 1
                torch.testing.assert_close(
                    result, expected_acc.sum(dim=1), rtol=0, atol=0
                )

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_lane_uniform_updates_reach_padded_lanes(self):
        """Updates whose per-lane part is gone (a sum over the tile, a
        constant picked by a runtime ``if``, also where the other branch adds
        the tile's exp, and then another constant after the ``if``, a
        constant added in a nested while loop) reach a partial tile's padded
        lanes like ``acc += 1`` in test_loop_carry_masking."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def plus_row_sum(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n].sum(dim=1, keepdim=True)
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def plus_branch(x, beta: float):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for _ in hl.tile(n, block_size=block_n):
                    if beta > 0:
                        acc = acc + 1.0
                    else:
                        acc = acc + 2.0
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def exp_or_one(x, beta: float):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    if beta > 0:
                        acc = acc + torch.exp(x[tile_m, tile_n])
                    else:
                        acc = acc + 1.0
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def exp_or_one_then_one(x, beta: float):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    if beta > 0:
                        acc = acc + torch.exp(x[tile_m, tile_n])
                    else:
                        acc = acc + 1.0
                    acc = acc + 1.0
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def plus_one_while(x, steps):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for _ in hl.tile(n, block_size=block_n):
                    limit = steps[0]
                    step = torch.zeros([], device=x.device, dtype=torch.int32)
                    while step < limit:
                        acc = acc + 1.0
                        step = step + 1
                out[tile_m] = acc.sum(dim=1)
            return out

        # 20 columns: a full 16-column tile, then 4 columns and 12 padded
        # lanes.  Each tile's row sum reaches all 16 lanes.
        x = torch.randint(1, 5, [64, 20], device=DEVICE).float()
        torch.testing.assert_close(plus_row_sum(x), 16 * x.sum(dim=1), rtol=0, atol=0)
        zeros = torch.zeros([64], device=DEVICE)
        for beta, expected in ((1.0, 32), (-1.0, 64)):
            torch.testing.assert_close(
                plus_branch(x, beta), zeros + expected, rtol=0, atol=0
            )
        # The exp branch keeps the padded lanes out, the other adds 1 to all.
        torch.testing.assert_close(
            exp_or_one(x, 1.0), torch.exp(x).sum(dim=1), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(exp_or_one(x, -1.0), zeros + 32, rtol=0, atol=0)
        # 1.0 in the branch and after the if: 2 in every lane of both tiles.
        # (With beta > 0 the padded lanes still add exp(0): only a select of
        # the whole update could keep it out, and it would undo the 1.0 after
        # the if, so the pass leaves that update as it is.)
        torch.testing.assert_close(
            exp_or_one_then_one(x, -1.0), zeros + 64, rtol=0, atol=0
        )
        steps = torch.tensor([2], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(plus_one_while(x, steps), zeros + 64, rtol=0, atol=0)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_update_from_partial_tile_loads(self):
        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def exp_sum(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + torch.exp(x[tile_m, tile_n])
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def plus_one_sum(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + (x[tile_m, tile_n] + 1.0)
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def running_max(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.full([tile_m, block_n], float("-inf"))
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = torch.maximum(acc, x[tile_m, tile_n] - 1.0)
                out[tile_m] = acc.amax(dim=1)
            return out

        # 500 columns end in a 4-column tile whose 12 padded lanes load 0:
        # unmasked, each would add exp(0) or 0 + 1 to its row's sum, and its
        # -1 would beat every valid x - 1 <= -3 in the running max.
        x = -torch.randint(2, 10, [64, 500], device=DEVICE).float()
        torch.testing.assert_close(
            exp_sum(x), torch.exp(x).sum(dim=1), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            plus_one_sum(x), (x + 1.0).sum(dim=1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            running_max(x), (x - 1.0).amax(dim=1), rtol=0, atol=0
        )
        # 62 rows also end in a partial tile of rows.
        torch.testing.assert_close(
            plus_one_sum(x[:62]), (x[:62] + 1.0).sum(dim=1), rtol=0, atol=0
        )

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_updates_that_scale_or_repeat(self):
        """Carries that a mask after the loop does not cover (a decayed, a
        doubled, a halved or a gated accumulator, whose padded lanes the
        masking analysis takes for 0, and a decay divided by the tile's exp)
        and carries updated in a loop nested in another loop or nested in the
        carry's loop.  In the last tile's padded lanes the decay, the
        doubling and the halving apply as in the live lanes, while the gate
        and the divisor there are 1 and ``x`` and ``torch.exp(x)`` add
        0."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def decay(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc * 0.5 + x[tile_m, tile_n]
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def doubled(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = torch.add(x[tile_m, tile_n], acc, alpha=2.0)
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def halved(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = (acc + torch.exp(x[tile_m, tile_n])) / 2
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def gated(x, g):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc * g[tile_m, tile_n] + x[tile_m, tile_n]
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4, 1]}, static_shapes=True)
        def plus_one_twice(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for _ in hl.tile(2):
                    for tile_n in hl.tile(n, block_size=block_n):
                        acc = acc + (x[tile_m, tile_n] + 1.0)
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def decay_divided(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.full([tile_m, block_n], 1.0)
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = (acc * 0.5) / torch.exp(x[tile_m, tile_n])
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4, 1]}, static_shapes=True)
        def exp_in_a_nested_loop(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    for _ in hl.tile(2):
                        acc = acc + torch.exp(-x[tile_m, tile_n])
                out[tile_m] = acc.sum(dim=1)
            return out

        def lanes(x, g, pad):
            """The 16 lanes of ``acc * g + x`` after the loop, where in the
            last tile's padded lanes ``g`` is ``pad`` and ``x`` is 0."""
            acc = torch.zeros([x.size(0), 16], device=DEVICE)
            for start in range(0, x.size(1), 16):
                live = min(16, x.size(1) - start)
                gate = torch.full_like(acc, pad)
                gate[:, :live] = g[:, start : start + live]
                add = torch.zeros_like(acc)
                add[:, :live] = x[:, start : start + live]
                acc = acc * gate + add
            return acc

        # 500 columns end in a 4-column tile with 12 padded lanes, whose
        # loads would zero the gate, add exp(0) or add 0 + 1 (twice) to their
        # row's lanes.
        x = torch.randint(1, 10, [64, 500], device=DEVICE).float()
        g = torch.randint(0, 2, [64, 500], device=DEVICE).float()
        for kernel, scale in ((decay, 0.5), (doubled, 2.0)):
            torch.testing.assert_close(
                kernel(x),
                lanes(x, torch.full_like(x, scale), scale).sum(dim=1),
                rtol=1e-6,
                atol=1e-6,
            )
        torch.testing.assert_close(
            halved(x),
            lanes(torch.exp(x) / 2, torch.full_like(x, 0.5), 0.5).sum(dim=1),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            gated(x, g), lanes(x, g, 1.0).sum(dim=1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            plus_one_twice(x), 2 * (x + 1.0).sum(dim=1), rtol=0, atol=0
        )
        # Two tiles of 20 zeros: every lane halves twice, the 12 padded ones
        # of the second tile too (divided by 1 there), to 0.25: 4 per row.
        torch.testing.assert_close(
            decay_divided(torch.zeros([64, 20], device=DEVICE)),
            torch.full([64], 4.0, device=DEVICE),
            rtol=0,
            atol=0,
        )
        # Each live column adds its exp twice; the padded lanes add nothing.
        torch.testing.assert_close(
            exp_in_a_nested_loop(x),
            2 * torch.exp(-x).sum(dim=1),
            rtol=1e-5,
            atol=1e-5,
        )

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_sums_however_grouped(self):
        """Values added to a carry one after another count as one sum: ``acc
        + x + 1.0`` keeps the last tile's padded lanes out as ``acc + (x +
        1.0)`` does, and so do ``acc + 1.0 + torch.exp(x)`` and ``acc * 0.5 +
        x - 1.0``, whose decay still applies to them.  An infinite ``alpha``
        scales none of their loads' 0 into NaN."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def plus_x_plus_one(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n] + 1.0
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def plus_one_plus_exp(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + 1.0 + torch.exp(x[tile_m, tile_n])
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def decay_plus_x_minus_one(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc * 0.5 + x[tile_m, tile_n] - 1.0
                out[tile_m] = acc.sum(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def infinite_alpha(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = torch.add(acc, x[tile_m, tile_n], alpha=float("inf"))
                out[tile_m] = acc.sum(dim=1)
            return out

        # 500 columns end in a 4-column tile with 12 padded lanes, which
        # would add 1.0 (and exp(0)), or -1.0, to their row's lanes.
        x = torch.randint(1, 10, [64, 500], device=DEVICE).float()
        torch.testing.assert_close(
            plus_x_plus_one(x), (x + 1.0).sum(dim=1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            plus_one_plus_exp(x),
            (torch.exp(x) + 1.0).sum(dim=1),
            rtol=1e-5,
            atol=1e-5,
        )
        lanes = torch.zeros([64, 16], device=DEVICE)
        for start in range(0, 500, 16):
            live = min(16, 500 - start)
            lanes = lanes * 0.5
            lanes[:, :live] += x[:, start : start + live] - 1.0
        torch.testing.assert_close(
            decay_plus_x_minus_one(x), lanes.sum(dim=1), rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            infinite_alpha(x), torch.full([64], float("inf"), device=DEVICE)
        )

    @onlyBackends(["triton"])
    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_plus_a_load_broadcast_along_its_block(self):
        """A load broadcast along a second dim on the loop's block (``acc``
        ``[tile_m, bn, bn]`` plus ``x[tile_m, tile_n][:, None, :]``) stays out
        of the last tile's padded lanes of both dims, as its ``expand`` does.
        CuTe refuses a tile with two dims on one block."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def plus_broadcast(x, expand: bool):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    v = x[tile_m, tile_n][:, None, :]
                    if expand:
                        acc = acc + v.expand([tile_m, block_n, block_n])
                    else:
                        acc = acc + v
                out[tile_m] = acc.sum(dim=2).sum(dim=1)
            return out

        # 20 columns end in a 4-column tile: its 4 live lanes of the middle
        # dim each add the tile's row, the 12 padded ones nothing.
        x = torch.randint(1, 10, [64, 20], device=DEVICE).float()
        expected = 16 * x[:, :16].sum(dim=1) + 4 * x[:, 16:].sum(dim=1)
        for expand in (False, True):
            torch.testing.assert_close(
                plus_broadcast(x, expand), expected, rtol=0, atol=0
            )

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_rescaled_with_the_row_max_in_its_last_tile(self):
        """An online logsumexp keeps a per-lane accumulator that each tile
        rescales to the running max: the padded lanes of the last, partial
        tile are rescaled with the live ones and add none of the tile's
        ``exp``, so a row max in that tile leaves the result exact."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def fn(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    v = x[tile_m, tile_n]
                    m_new = torch.maximum(m_i, torch.amax(v, dim=1))
                    acc = acc * torch.exp(m_i - m_new)[:, None] + torch.exp(
                        v - m_new[:, None]
                    )
                    m_i = m_new
                out[tile_m] = torch.log(acc.sum(dim=1)) + m_i
            return out

        # 500 columns end in a 4-column tile, which holds each row's max.
        x = torch.randn([64, 500], device=DEVICE)
        x[:, -4:] += 5.0
        torch.testing.assert_close(
            fn(x), torch.logsumexp(x, dim=1), rtol=1e-5, atol=1e-5
        )

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_extremes_of_int32_loads(self):
        """A float carry's running max or min of int32 loads keeps its
        carried value in the padded lanes of the last, partial tile, as the
        int32 operand cannot hold the identity -inf or inf, and an int32
        carry's running max masks its operand to the int32 minimum: a padded
        lane's 0 would beat the negative data of the max and the positive
        data of the min."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def float_max(y):
            m, n = y.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=y.device)
            for tile_m in hl.tile(m):
                acc = hl.full([tile_m, block_n], float("-inf"))
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = torch.maximum(acc, y[tile_m, tile_n])
                out[tile_m] = acc.amax(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def float_min(y):
            m, n = y.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=y.device)
            for tile_m in hl.tile(m):
                acc = hl.full([tile_m, block_n], float("inf"))
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = torch.minimum(acc, y[tile_m, tile_n])
                out[tile_m] = acc.amin(dim=1)
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def int_max(y):
            m, n = y.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], dtype=y.dtype, device=y.device)
            for tile_m in hl.tile(m):
                acc = hl.full([tile_m, block_n], -(2**31), dtype=y.dtype)
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = torch.maximum(acc, y[tile_m, tile_n])
                out[tile_m] = acc.amax(dim=1)
            return out

        # 500 columns end in a 4-column tile whose 12 padded lanes load 0.
        y = -torch.randint(2, 10, [64, 500], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(float_max(y), y.amax(dim=1).float())
        torch.testing.assert_close(float_min(-y), (-y).amin(dim=1).float())
        torch.testing.assert_close(int_max(y), y.amax(dim=1))

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_tile_index_does_not_mask(self):
        @helion.kernel(config={"block_sizes": [32, 32], "indexing": "block_ptr"})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_size_n])
                for tile_n in hl.tile(0, n, block_size_n):
                    acc += x[tile_m.index, tile_n.index]
                out[tile_m.index] = acc.sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1))
        self.assertNotIn("tl.where", code)

    @staticmethod
    def _live_carry_sums() -> tuple[torch.Tensor, torch.Tensor]:
        """64 rows of 20 small integers, and each row's sum over its two tiles
        of 16 columns (a full tile, then 4 live columns) of the running carry
        ``acc += x[:, tile]``, each tile adding the carry's live lanes."""
        x = torch.randint(1, 5, [64, 20], device=DEVICE).float()
        acc = torch.zeros([64, 16], device=DEVICE)
        expected = torch.zeros([64], device=DEVICE)
        for start in range(0, 20, 16):
            live = min(16, 20 - start)
            acc[:, :live] += x[:, start : start + live]
            expected += acc[:, :live].sum(dim=1)
        return x, expected

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_reduced_inside_its_loop(self):
        """A sum of the carry over the loop's block inside the loop sees only
        the live lanes of a partial tile, as in test_loop_carry_masking: the
        carry's padded lanes hold the first tile's columns there, though its
        zero start gives them the masked value 0."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def fn(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n]
                    total = total + acc.sum(dim=1)
                out[tile_m] = total
            return out

        x, expected = self._live_carry_sums()
        torch.testing.assert_close(fn(x), expected, rtol=0, atol=0)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_reduced_in_a_branch_or_inner_loop_of_its_loop(self):
        """As test_loop_carry_reduced_inside_its_loop, with the sum in a
        runtime branch or an inner loop of the carry's loop: the reduction
        there needs the outer loop's mask as well."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def in_branch(x, flag):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n]
                    if flag[0] > 0:
                        total = total + acc.sum(dim=1)
                out[tile_m] = total
            return out

        @helion.kernel(config={"block_sizes": [16, 4, 8]}, static_shapes=True)
        def in_inner_loop(x, y):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n]
                    for tile_k in hl.tile(y.size(0)):
                        total = total + acc.sum(dim=1) * y[tile_k].sum()
                out[tile_m] = total
            return out

        x, expected = self._live_carry_sums()
        flag = torch.ones([1], dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(in_branch(x, flag), expected, rtol=0, atol=0)
        # Two 8-element tiles of y: each adds acc.sum(dim=1) * 8.
        y = torch.ones([16], device=DEVICE)
        torch.testing.assert_close(in_inner_loop(x, y), expected * 16, rtol=0, atol=0)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    # CuTe refuses a carry with a dim on its loop's block that the loop reads
    # other than to add to it (the lane-merged carry check).
    @onlyBackends(["triton"])
    def test_loop_carry_contracted_by_a_dot_inside_its_loop(self):
        """An ``hl.dot`` that contracts the carry's loop block reduces over it,
        though its result keeps a loop dim (the k loop's): the carry keeps its
        mask, and each tile's product takes only its live lanes, though the
        other operand's padded rows are 1."""

        @helion.kernel(config={"block_sizes": [16, 16, 16]}, static_shapes=True)
        def fn(x, w):
            m, n = x.size()
            k = w.size(1)
            block_n = hl.register_block_size(n)
            out = torch.empty([m, k], device=x.device)
            for tile_m in hl.tile(m):
                for tile_k in hl.tile(k):
                    acc = hl.zeros([tile_m, block_n])
                    total = hl.zeros([tile_m, tile_k])
                    for tile_n in hl.tile(n, block_size=block_n):
                        acc = acc + x[tile_m, tile_n]
                        total = total + hl.dot(acc, w[tile_n, tile_k] + 1.0)
                    out[tile_m, tile_k] = total
            return out

        x, _ = self._live_carry_sums()
        w = torch.randint(-2, 3, [20, 32], device=DEVICE).float()
        acc = torch.zeros([64, 16], device=DEVICE)
        expected = torch.zeros([64, 32], device=DEVICE)
        for start in range(0, 20, 16):
            live = min(16, 20 - start)
            acc[:, :live] += x[:, start : start + live]
            expected += acc[:, :live] @ (w[start : start + live] + 1.0)
        torch.testing.assert_close(fn(x, w), expected, rtol=0, atol=0)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    # CuTe refuses a carry with a dim on its loop's block that the loop reads
    # other than to add to it (the lane-merged carry check).
    @onlyBackends(["triton"])
    def test_loop_carry_reduced_along_a_repeated_block(self):
        """A carry with two dims on its loop's block summed along one of
        them, and an ``hl.dot`` contracting the carry with a matrix whose
        result keeps the block on its other axis: each reduces over the
        block though its result keeps it, so the carry keeps its mask and
        only the live lanes of the partial tile count."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def square_carry(x, y):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([block_n, block_n])
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + y[tile_n, tile_n]
                    total = total + (x[tile_m, tile_n] * acc.sum(1)[None, :]).sum(1)
                out[tile_m] = total
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def square_dot(x, w):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n]
                    total = total + hl.dot(acc, w[tile_n, tile_n] + 1.0).sum(1)
                out[tile_m] = total
            return out

        x, _ = self._live_carry_sums()
        y = torch.randint(-2, 3, [20, 20], device=DEVICE).float()
        acc_rows = torch.zeros([16, 16], device=DEVICE)
        acc = torch.zeros([64, 16], device=DEVICE)
        expected_rows = torch.zeros([64], device=DEVICE)
        expected_dot = torch.zeros([64], device=DEVICE)
        for start in range(0, 20, 16):
            live = min(16, 20 - start)
            cols = slice(start, start + live)
            acc_rows[:live, :live] += y[cols, cols]
            expected_rows += x[:, cols] @ acc_rows[:live, :live].sum(dim=1)
            acc[:, :live] += x[:, cols]
            expected_dot += (acc[:, :live] @ (y[cols, cols] + 1.0)).sum(dim=1)
        torch.testing.assert_close(square_carry(x, y), expected_rows, rtol=0, atol=0)
        torch.testing.assert_close(square_dot(x, y), expected_dot, rtol=0, atol=0)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_reduced_after_a_reshape_or_in_a_later_loop(self):
        """The carry's stale lanes reach a sum through shape-only ops
        (``acc.flatten().view(acc.shape)``), and a loop's result keeps them
        after the loop, where a later loop over the same block sums it: both
        sums keep the mask, and only the live lanes count."""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def reshaped(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n]
                    total = total + acc.flatten().view(acc.shape).sum(dim=1)
                out[tile_m] = total
            return out

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def later_loop(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n]
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    total = total + (acc + x[tile_m, tile_n]).sum(dim=1)
                out[tile_m] = total
            return out

        x, expected = self._live_carry_sums()
        torch.testing.assert_close(reshaped(x), expected, rtol=0, atol=0)
        acc = torch.zeros([64, 16], device=DEVICE)
        for start in range(0, 20, 16):
            live = min(16, 20 - start)
            acc[:, :live] += x[:, start : start + live]
        expected = torch.zeros([64], device=DEVICE)
        for start in range(0, 20, 16):
            live = min(16, 20 - start)
            expected += (acc[:, :live] + x[:, start : start + live]).sum(dim=1)
        torch.testing.assert_close(later_loop(x), expected, rtol=0, atol=0)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_reduced_through_a_flatten_across_graphs(self):
        """``acc.flatten()`` holds the carry's stale lanes in its one ``tile_m
        * bn`` dim, so a runtime branch or an inner loop given it needs the
        mask before summing it viewed back, as for the carry itself: the sums
        count only the live lanes.  (test_loop_carry_reduction_masks.py
        checks the masks of a flattened carry, which neither backend runs
        yet.)"""

        @helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
        def in_branch(x, flag):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n]
                    flat = acc.flatten()
                    if flag[0] > 0:
                        total = total + flat.view(tile_m, block_n).sum(dim=1)
                out[tile_m] = total
            return out

        @helion.kernel(config={"block_sizes": [16, 4, 8]}, static_shapes=True)
        def in_inner_loop(x, y):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    acc = acc + x[tile_m, tile_n]
                    flat = acc.flatten()
                    for tile_k in hl.tile(y.size(0)):
                        rows = flat.view(tile_m, block_n).sum(dim=1)
                        total = total + rows * y[tile_k].sum()
                out[tile_m] = total
            return out

        x, expected = self._live_carry_sums()
        flag = torch.ones([1], dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(in_branch(x, flag), expected, rtol=0, atol=0)
        # Two 8-element tiles of y: each adds the row sums times 8.
        y = torch.ones([16], device=DEVICE)
        torch.testing.assert_close(in_inner_loop(x, y), expected * 16, rtol=0, atol=0)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_inner_loop_values_without_earlier_lanes_need_no_mask(self):
        """An inner loop's input it only reads, and its own carry started in
        the outer loop, hold no lanes an earlier iteration of the outer loop
        left, so their sums over the outer loop's block in the inner loop need
        no mask (test_loop_carry_reduction_masks.py checks the kernels keep
        none): they give the sums of the live lanes."""

        @helion.kernel(config={"block_sizes": [16, 4, 8]}, static_shapes=True)
        def read_only(x, y):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    row = x[tile_m, tile_n]
                    for _tile_k in hl.tile(y.size(0)):
                        total = total + row.sum(dim=1)
                out[tile_m] = total
            return out

        @helion.kernel(config={"block_sizes": [16, 4, 8]}, static_shapes=True)
        def fresh_carry(x, y):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    inner = hl.zeros([tile_m, block_n])
                    for _tile_k in hl.tile(y.size(0)):
                        inner = inner + x[tile_m, tile_n]
                        total = total + inner.sum(dim=1)
                out[tile_m] = total
            return out

        x, _ = self._live_carry_sums()
        # Two 8-element tiles of y: the inner loop runs twice per column tile.
        y = torch.empty([16], device=DEVICE)
        for kernel, repeats in ((read_only, 2), (fresh_carry, 1 + 2)):
            torch.testing.assert_close(
                kernel(x, y), x.sum(dim=1) * repeats, rtol=0, atol=0
            )

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_result_summed_in_a_later_loop_from_any_nesting(self):
        """A value with a loop's stale lanes along a block reaches a later
        loop's sum over that block through ``c.sum(1)``, a reduction over
        another block in an enclosing graph (its mask decided without the
        loops over the summed one), out of an outer loop that reruns the
        carry's loop, and out of either branch of an ``if``: the sums count
        only the live lanes."""

        @helion.kernel(config={"block_sizes": [16, 4, 4]}, static_shapes=True)
        def enclosing_reduction(w, v):
            nx, ny = w.size()
            out = torch.empty([nx], device=w.device)
            block_y = hl.register_block_size(ny)
            for tile_g in hl.tile(nx):
                total = hl.zeros([tile_g])
                for tile_x in hl.tile(nx):
                    c = hl.zeros([block_y, tile_x])
                    for tile_y in hl.tile(ny, block_size=block_y):
                        c = c + w[tile_x, tile_y].T
                    r = c.sum(1)
                    for tile_y in hl.tile(ny, block_size=block_y):
                        total = total + (r + v[tile_y]).sum(0)
                out[tile_g] = total
            return out

        @helion.kernel(config={"block_sizes": [16, 64, 1]}, static_shapes=True)
        def nested_loop(x, y):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for _tile_k in hl.tile(y.size(0)):
                    acc = hl.zeros([tile_m, block_n])
                    for tile_n in hl.tile(n, block_size=block_n):
                        acc = acc + x[tile_m, tile_n]
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    total = total + (acc + x[tile_m, tile_n]).sum(dim=1)
                out[tile_m] = total
            return out

        @helion.kernel(config={"block_sizes": [16, 64]}, static_shapes=True)
        def branches(x, flag):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                if flag[0] > 0:
                    for tile_n in hl.tile(n, block_size=block_n):
                        acc = acc + x[tile_m, tile_n]
                else:
                    for tile_n in hl.tile(n, block_size=block_n):
                        acc = acc + 2 * x[tile_m, tile_n]
                total = hl.zeros([tile_m])
                for tile_n in hl.tile(n, block_size=block_n):
                    total = total + (acc + x[tile_m, tile_n]).sum(dim=1)
                out[tile_m] = total
            return out

        def later_loop_sums(x, acc_scale):
            # The carry over the two 16-column tiles, then the later loop's
            # sum of it plus each tile's live columns.
            acc = torch.zeros([x.size(0), 16], device=DEVICE)
            for start in range(0, 20, 16):
                live = min(16, 20 - start)
                acc[:, :live] += acc_scale * x[:, start : start + live]
            expected = torch.zeros([x.size(0)], device=DEVICE)
            for start in range(0, 20, 16):
                live = min(16, 20 - start)
                expected += (acc[:, :live] + x[:, start : start + live]).sum(dim=1)
            return expected

        w = torch.randint(1, 5, [8, 20], device=DEVICE).float()
        v = torch.randint(1, 5, [20], device=DEVICE).float()
        total = torch.zeros([], device=DEVICE)
        for x0 in range(0, 8, 4):
            c = torch.zeros([16, 4], device=DEVICE)
            for start in range(0, 20, 16):
                live = min(16, 20 - start)
                c[:live] += w[x0 : x0 + 4, start : start + live].T
            r = c.sum(1)
            for start in range(0, 20, 16):
                live = min(16, 20 - start)
                total += (r[:live] + v[start : start + live]).sum()
        torch.testing.assert_close(
            enclosing_reduction(w, v), total.expand(8), rtol=0, atol=0
        )
        x, _ = self._live_carry_sums()
        y = torch.empty([3], device=DEVICE)
        if _get_backend() != "tileir":
            # The TileIR toolchain keeps ``acc`` across the outer loop's three
            # iterations although each one resets it: the sums come out as if
            # the carry had run three times.  The generated code is Triton's,
            # which Triton compiles correctly.
            torch.testing.assert_close(
                nested_loop(x, y), later_loop_sums(x, 1), rtol=0, atol=0
            )
        for flag, scale in ((1, 1), (-1, 2)):
            flags = torch.tensor([flag], dtype=torch.int32, device=DEVICE)
            torch.testing.assert_close(
                branches(x, flags), later_loop_sums(x, scale), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
