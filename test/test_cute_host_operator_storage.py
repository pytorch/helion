from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.ast_extension import create
from helion._compiler.runtime_tensor_guards import (
    tensor_pairs_with_proven_disjoint_storage,
)
from helion._compiler.type_info import LiteralType
from helion._compiler.type_propagation import GlobalScope
from helion._compiler.type_propagation import LocalScope
from helion._compiler.type_propagation import TypePropagation
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _host_join_copy(x: torch.Tensor, select: int) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(x)
    second = torch.empty_like(x)
    if select > 0:
        target = first
    else:
        target = second
    for tile in hl.tile(x.size(0)):
        target[tile] = x[tile]
    return first, target


class _TransientStorageEffect:
    def __init__(self, owner, replacement):
        self.owner = owner
        self.replacement = replacement
        self.calls = []
        self.transient_view = None

    def _touch(self, operation):
        self.calls.append(operation)
        original = self.owner.data
        self.owner.data = self.replacement
        self.transient_view = self.owner[:]
        self.owner.data = original

    def __add__(self, other):
        assert other == 0
        self._touch("add")
        return 1

    def __eq__(self, other):
        assert other == 0
        self._touch("compare")
        return True


@pytest.mark.parametrize("operation", ["add", "compare", "identity"])
@pytest.mark.parametrize("opaque", [False, True])
def test_host_operator_effects_preserve_only_proved_storage(operation, opaque):
    kernel = helion.kernel(_host_join_copy, backend="cute", static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (torch.empty(32), 1))
        host = bound.host_function
        with bound.env, host:
            tensors = {
                name: host.local_types[name].fake_value
                for name in ("x", "first", "target")
            }
            pair = frozenset(("x", "target"))
            assert pair in tensor_pairs_with_proven_disjoint_storage(bound.env, tensors)
            effect = _TransientStorageEffect(tensors["first"], tensors["x"])
            original_storage = tensors["first"].untyped_storage()
            origin = host.local_types["first"].origin
            # Supply the opaque representative directly to the real executor.
            # This tests operator effects independently of public constexpr
            # admission, and does not replace the executor or mutation tracker.
            scope = LocalScope(
                GlobalScope(host),
                {"operand": LiteralType(origin, effect if opaque else 1)},
            )
            visitor = TypePropagation(host, scope)
            left = create(ast.Name, id="operand", ctx=ast.Load())
            right = create(ast.Constant, value=0)
            node = (
                create(ast.BinOp, left=left, op=ast.Add(), right=right)
                if operation == "add"
                else create(
                    ast.Compare,
                    left=left,
                    ops=[ast.Is() if operation == "identity" else ast.Eq()],
                    comparators=[right],
                )
            )
            result = visitor.visit(node)
            assert isinstance(result, LiteralType)
            assert result.value == (
                1 if operation == "add" else opaque and operation == "compare"
            )
            assert tensors["first"].untyped_storage() is original_storage
            if opaque and operation != "identity":
                assert effect.calls == [operation]
                assert effect.transient_view is not None
                assert (
                    effect.transient_view.untyped_storage()
                    is tensors["x"].untyped_storage()
                )
                # Storage returned to its original value, so snapshots alone
                # cannot notice the transient alias created by user Python.
                assert bound.env.tensor_storage_provenance(tensors["first"]) is None
                assert pair not in tensor_pairs_with_proven_disjoint_storage(
                    bound.env, tensors
                )
            else:
                assert not effect.calls
                assert pair in tensor_pairs_with_proven_disjoint_storage(
                    bound.env, tensors
                )
