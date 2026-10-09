"""Conservative value-observability facts for pure row-fragment graphs."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING

import torch

from ...language._tracing_ops import _mask_to
from ...language._tracing_ops import _new_var
from ...language.memory_ops import store

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    from torch.fx import Node


class RowValueRequirement(Enum):
    UNUSED = "unused"
    NUMERIC = "numeric"
    EXACT = "exact"


_VIEWS = frozenset(
    {
        _new_var,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.expand.default,
        torch.ops.prims.convert_element_type.default,
    }
)
_ZERO_ERASING = frozenset(
    {
        torch.ops.aten.exp.default,
        torch.ops.aten.exp2.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.cos.default,
        torch.ops.aten.cosh.default,
    }
)
_PREDICATES = frozenset(
    {
        torch.ops.aten.isnan.default,
        torch.ops.aten.isinf.default,
        torch.ops.aten.isfinite.default,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.ge.Scalar,
    }
)
_ZERO_PROPAGATING = frozenset(
    {
        _mask_to,
        torch.ops.aten.neg.default,
        torch.ops.aten.abs.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.amax.default,
        torch.ops.aten.amin.default,
    }
)
_BIT_OBSERVERS = frozenset(
    {
        torch.ops.aten.signbit.default,
        torch.ops.aten.copysign.Tensor,
        torch.ops.aten.copysign.Scalar,
        torch.ops.aten.view.dtype,
    }
)


class RowValueUses:
    """Prove whether canonical NaNs and zero signs can affect observable values.

    Direct value stores and sign-sensitive operations require exact recovery.
    Ordinary floating arithmetic does not promise a particular NaN payload,
    but its result must not feed a sign/bit observer after canonicalization.
    Zero signs are retained until a predicate or an operation such as exp maps
    both signs to the same value. In particular, a zero used as a divisor can
    produce opposite infinities, so that path is never relaxed.
    """

    def __init__(
        self, outputs: Iterable[Node], *, resolve: Callable[[Node], Node]
    ) -> None:
        self.resolve = resolve
        self.users: dict[Node, set[Node]] = defaultdict(set)
        seen: set[Node] = set()

        def visit(node: Node) -> None:
            node = resolve(node)
            if node in seen:
                return
            seen.add(node)
            for argument in node.all_input_nodes:
                source = resolve(argument)
                self.users[source].add(node)
                visit(source)

        for output in outputs:
            visit(output)

    def _has_bit_observer(self, node: Node, seen: set[Node]) -> bool:
        if node in seen:
            return False
        seen.add(node)
        if node.target in _BIT_OBSERVERS:
            return True
        return any(self._has_bit_observer(user, seen) for user in self.users[node])

    def _insensitive(self, source: Node, user: Node, active: set[Node]) -> bool:
        if user in active or user.target is store or user.target in _BIT_OBSERVERS:
            return False
        if user.target in _PREDICATES:
            return True
        if user.target in _ZERO_ERASING:
            return not self._has_bit_observer(user, set())
        propagates = user.target in _VIEWS or user.target in _ZERO_PROPAGATING
        if user.target in (torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar):
            denominator = user.args[1]
            propagates = not (
                isinstance(denominator, torch.fx.Node)
                and self.resolve(denominator) is source
            )
        if not propagates:
            return False
        users = self.users[user]
        return bool(users) and all(
            self._insensitive(user, descendant, active | {user}) for descendant in users
        )

    def requirement(self, node: Node) -> RowValueRequirement:
        node = self.resolve(node)
        users = self.users[node]
        if not users:
            return RowValueRequirement.UNUSED
        if all(self._insensitive(node, user, set()) for user in users):
            return RowValueRequirement.NUMERIC
        return RowValueRequirement.EXACT

    def numeric_only(self, node: Node) -> bool:
        return self.requirement(node) is RowValueRequirement.NUMERIC
