"""CPU traces of ATen device assertion effects, errors and ordering."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import itertools
from unittest import mock

import pytest
import torch
from torch._higher_order_ops.effects import _EffectType
from torch._higher_order_ops.effects import _register_effectful_op

import helion
from helion._compiler import device_ir
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.aten_lowering import Lowering
from helion._compiler.triton.backend import TritonBackend
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfNotTriton
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cuda_metadata_trace
from helion._testing.distributed_kernels import fake_group as fake_group
from helion._testing.distributed_kernels import fake_world
from helion._testing.distributed_kernels import remote_views
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_LIB = torch.library.Library("helion_test", "FRAGMENT")
_LIB.define("check_device_predicate(Tensor cond) -> ()")
_LIB.impl("check_device_predicate", lambda cond: None, "CompositeExplicitAutograd")
_CHECK = torch.ops.helion_test.check_device_predicate.default
_register_effectful_op(_CHECK, _EffectType.ORDERED)


@contextmanager
def _native_assertions(*, decompose=False):
    """Model a backend that lowers ATen directly or decomposes to its own check."""
    default = torch.ops.aten._assert_async.default
    msg = torch.ops.aten._assert_async.msg

    class NativeAssertion(Lowering):
        def codegen(self, ctx, node):
            overload = {default: "default", msg: "message", _CHECK: "decomposed"}[
                node.target
            ]
            ctx.cg.add_statement(
                statement_from_string(
                    f'tl.device_assert({{cond}}, "backend {overload} overload")',
                    cond=ctx.env[node.args[0]],
                )
            )

    prior = TritonBackend.pre_inductor_lowering

    def lower(backend, node):
        return (
            NativeAssertion()
            if node.target in (default, msg, _CHECK)
            else prior(backend, node)
        )

    def override(table):
        if decompose:
            table[default] = _CHECK
        else:
            table.pop(default)

    with (
        mock.patch.object(
            TritonBackend,
            "decomp_table_overrides",
            side_effect=override,
        ),
        mock.patch.object(TritonBackend, "pre_inductor_lowering", lower),
    ):
        yield


def _printed_add_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    hl.device_print("combining", a)
    return a + b


def _checked_add_combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    torch._assert_async((a >= 0) & (b >= 0))
    return a + b


def _checked_add_combine_msg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    torch._assert_async((a >= 0) & (b >= 0), "negative operand")
    return a + b


@helion.kernel()
def _assert_in_device_loop(x: torch.Tensor, with_message: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        values = x[tile]
        if with_message:
            torch._assert_async(values.sum() >= 0, "negative sum")
        else:
            torch._assert_async(values.sum() >= 0)
        out[tile] = values
    return out


class TestDeviceAssertions(RefEagerTestDisabled, TestCase):
    def test_unsupported_device_assertion(self):
        # An inherited no-op must not erase either spelling before FX tracing.
        decompositions = dict(device_ir.select_decomp_table())
        decompositions[torch.ops.aten._assert_async.msg] = lambda *args: None
        for form in (False, True):
            with (
                self.subTest(with_message=form),
                mock.patch.object(
                    device_ir, "select_decomp_table", return_value=decompositions
                ),
                self.assertRaisesRegex(
                    helion.exc.BackendUnsupported,
                    r"aten\._assert_async\.msg in device code",
                ),
            ):
                cpu_bind(_assert_in_device_loop, (torch.ones(16), form))

    @skipIfNotTriton("this test installs a Triton backend decomposition")
    def test_backend_decomposition_can_handle_device_assertion(self):
        # A backend may already implement its own default-overload decomposition.
        # Exercise a real bind and prove that implementation is invoked.
        custom = mock.Mock(
            side_effect=lambda p: torch.ops.aten._assert_async.msg(p, "backend")
        )
        with (
            mock.patch(
                "helion._compiler.triton.backend.TritonBackend.decomp_table_overrides",
                side_effect=lambda table: table.update(
                    {torch.ops.aten._assert_async.default: custom}
                ),
            ),
            self.assertRaisesRegex(
                helion.exc.BackendUnsupported, r"aten\._assert_async\.msg"
            ),
        ):
            cpu_bind(_assert_in_device_loop, (torch.ones(16), False))
        self.assertTrue(custom.called)

        # Dropping the decomposition opts into lowering the default overload.
        # Check that the effect survives DCE all the way to emitted source.
        with _native_assertions():
            bound = cpu_bind(_assert_in_device_loop, (torch.ones(16), False))
            code = bound.to_code(bound.config_spec.default_config())
        self.assertIn("tl.device_assert(", code)
        self.assertIn("backend default overload", code)

    def test_unsupported_device_assertion_in_combine(self):
        @helion.kernel()
        def combine(x: torch.Tensor, scan: hl.constexpr, use_result: hl.constexpr):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                values = x[tile]
                if scan:
                    result = hl.associative_scan(_checked_add_combine_msg, values, 0)
                else:
                    result = hl.reduce(_checked_add_combine, values, 0)
                if use_result:
                    out[tile] = result
                else:
                    out[tile] = values
            return out

        for scan in (False, True):
            for use_result in (False, True):
                with (
                    self.subTest(scan=scan, use_result=use_result),
                    self.assertRaisesRegex(
                        helion.exc.BackendUnsupported,
                        r"device assertions in hl\.reduce or hl\.associative_scan combine functions",
                    ),
                ):
                    cpu_bind(combine, (torch.ones(16), scan, use_result))

    def test_printed_combine_rejected_even_when_result_unused(self):
        @helion.kernel()
        def combine(x: torch.Tensor, scan: hl.constexpr, use_result: hl.constexpr):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                values = x[tile]
                if scan:
                    result = hl.associative_scan(_printed_add_combine, values, 0)
                else:
                    result = hl.reduce(_printed_add_combine, values, 0)
                if use_result:
                    out[tile] = result
                else:
                    out[tile] = values
            return out

        for scan in (False, True):
            for use_result in (False, True):
                with (
                    self.subTest(scan=scan, use_result=use_result),
                    self.assertRaises(helion.exc.NotAllowedInHelperFunction),
                ):
                    cpu_bind(combine, (torch.ones(16), scan, use_result))

    def test_combine_respects_inline_asm_purity(self):
        @helion.kernel()
        def combine(x, fn: hl.constexpr, scan: hl.constexpr, use_result: hl.constexpr):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                values = x[tile]
                if scan:
                    result = hl.associative_scan(fn, values, 0)
                else:
                    result = hl.reduce(fn, values, 0)
                if use_result:
                    out[tile] = result
                else:
                    out[tile] = values
            return out

        def bind(scan, is_pure, use_result):
            def add(a, b):
                hl.inline_asm_elementwise(
                    "mov.b32 $0, $1;", "=f,f", [a], torch.float32, is_pure, pack=1
                )
                return a + b

            return cpu_bind(combine, (torch.ones(16), add, scan, use_result))

        for scan in (False, True):
            with self.subTest(scan=scan):
                # The outer result is unused, but the declared effect is not.
                with self.assertRaises(helion.exc.NotAllowedInHelperFunction):
                    bind(scan, is_pure=False, use_result=False)
                # The unused pure asm can disappear from a live addition.
                bound = bind(scan, is_pure=True, use_result=True)
                self.assertNotIn(
                    hl.inline_asm_elementwise,
                    {
                        n.target
                        for info in bound.host_function.device_ir.graphs
                        for n in info.graph.nodes
                    },
                )

    def test_unsupported_void_aten_operation(self):
        @helion.kernel()
        def with_effect(x: torch.Tensor, inplace: hl.constexpr):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                values = x[tile]
                if inplace:
                    torch._foreach_add_([values], 1)
                else:
                    torch.ops.aten._print.default("device marker")
                out[tile] = values
            return out

        for inplace, name in [
            (False, r"_print\.default"),
            (True, r"_foreach_add_\.Scalar"),
        ]:
            with (
                self.subTest(inplace=inplace),
                self.assertRaisesRegex(
                    helion.exc.BackendUnsupported, rf"aten\.{name} in device code"
                ),
            ):
                cpu_bind(with_effect, (torch.ones(16), inplace))

    def test_sparse_index_validation_is_not_discarded(self):
        @helion.kernel(static_shapes=True)
        def validate(crow, col, out):
            for i in hl.grid(1):
                torch.ops.aten._validate_compressed_sparse_indices.default(
                    True, crow, col, 1, 1, 0
                )
                out[i] = 17

        args = (
            torch.zeros(2, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty(1, dtype=torch.int64),
        )
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, r"_validate_compressed_sparse_indices"
        ):
            cpu_bind(validate, args)

    @skipIfNotTriton("requires native Triton assertion lowering")
    def test_assertion_input_orders_later_overwrite(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def prefetch(
            x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor, form: hl.constexpr
        ):
            for t in hl.tile(ids.size(0), block_size=1):
                row = ids[t.begin]
                if form == "message":
                    torch._assert_async(row >= 0, "row")
                else:
                    torch._assert_async(row >= 0)
                if row >= 0:
                    hl.prefetch(x, [row, slice(None)])
            for t in hl.tile(ids.size(0), block_size=16):
                ids[t] = z[t]
            return ids

        ids = torch.arange(32, dtype=torch.int32)
        args = (torch.ones(64, 256), ids, torch.full_like(ids, -1))
        for form in ("default", "message", "decomposed"):
            # CPU has no cross-root pipeline: refusal proves this read
            # orders the overwrite, unlike a hint-only prefetch address.
            with (
                self.subTest(form=form),
                _native_assertions(decompose=form == "decomposed"),
                self.assertRaises(helion.exc.LoopDependencyError),
            ):
                cpu_bind(prefetch, (*args, form))

    @pytest.mark.usefixtures("fake_group")
    @skipIfNotTriton("requires in-band Triton peer polling")
    def test_peer_poll_does_not_cross_assertion(self):
        @helion.kernel(autotune_effort="none", static_shapes=True)
        def ordered(
            symm: torch.Tensor,
            x: torch.Tensor,
            group: hl.ProcessGroupName,
            form: hl.constexpr,
            conditional: hl.constexpr,
        ):
            peers = remote_views(symm, group)
            out = torch.empty_like(x)
            for t in hl.tile(x.size(0)):
                symm[t] = x[t]
            for t in hl.tile(x.size(0)):
                first = peers[0][t]
                # The assertion has no data dependency on the first peer wait.
                total = x[t].sum()
                if (total != 0) if conditional else True:
                    if form == "message":
                        torch._assert_async(total >= 0, "first peer")
                    else:
                        torch._assert_async(total >= 0)
                out[t] = first + peers[1][t]
            return out

        x = torch.ones(512)
        group = fake_world(2)
        for form, conditional in itertools.product(
            ("default", "message", "decomposed"), (False, True)
        ):
            with (
                self.subTest(form=form, conditional=conditional),
                _native_assertions(decompose=form == "decomposed"),
                cuda_metadata_trace(),
            ):
                bound = cpu_bind(ordered, (x, x.clone(), group, form, conditional))
                code = bound.to_triton_code()
                self.assertIn(f"backend {form} overload", code)
                graph = ast.parse(code)
                checks = [
                    n.lineno
                    for n in ast.walk(graph)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "device_assert"
                ]
                waits = [
                    n.lineno
                    for n in ast.walk(graph)
                    if isinstance(n, ast.While)
                    and "inband_stale" in ast.unparse(n.test)
                ]
                self.assertEqual(len(checks), 1)
                boundary = checks[0]
                if conditional:
                    boundary = max(
                        n.lineno
                        for n in ast.walk(graph)
                        if isinstance(n, ast.If)
                        and n.end_lineno is not None
                        and n.lineno < checks[0] <= n.end_lineno
                    )
                self.assertTrue(any(n < boundary for n in waits), waits)
                self.assertTrue(any(n > checks[0] for n in waits), waits)
