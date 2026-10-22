"""Prove ordered AB staging across two logical grouped K iterations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...exc import InvalidConfig
from ...language import _tracing_ops
from ...language import memory_ops
from ...language import view_ops
from .tcgen05_constants import TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo


def physical_grouped_ab_k(config: Mapping[str, object], logical_k: int) -> int:
    """Resolve only the staging tile; never change the logical configuration."""
    physical_k = config.get(TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY)
    if physical_k is None:
        return logical_k
    if type(physical_k) is not int or physical_k != 128 or logical_k != 64:
        raise InvalidConfig(
            f"{TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY}=128 requires logical K64"
        )
    return physical_k


def _strip_new_vars(value: object) -> object:
    while (
        isinstance(value, Node)
        and value.op == "call_function"
        and value.target is _tracing_ops._new_var
        and len(value.args) == 1
    ):
        value = value.args[0]
    return value


def _tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, Node):
        value = value.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _same_extent(value: object, expected: int, env: CompileEnvironment) -> bool:
    if isinstance(value, Node):
        if value.op != "call_function" or value.target is not _tracing_ops._get_symnode:
            return False
        value = value.meta.get("val")
    return isinstance(value, (int, torch.SymInt)) and env.known_equal(value, expected)


def prove_grouped_ab_k_loop(
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    mma: Node,
    *,
    lhs_load: Node,
    rhs_load: Node,
    static_k: int | None,
    zero_seed: bool,
    exclusive_region: bool,
    k_multiple: int = 128,
) -> bool:
    """Accept a closed direct-load addmm recurrence, with no other K effects.

    The caller proves grouped addresses, masks, output privacy and zero seeding.
    This proof additionally keeps the whole reduction's instruction order: its
    only result is the FP32 MMA carry, and its only tensor operations load the
    two direct operands and transpose the RHS.
    """
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import RootGraphInfo

    if (
        type(static_k) is not int
        or static_k <= 0
        or k_multiple <= 0
        or static_k % k_multiple
        or not zero_seed
        or not exclusive_region
        or mma.op != "call_function"
        or mma.target is not torch.ops.aten.addmm.default
        or len(mma.args) != 3
        or set(mma.kwargs) - {"alpha", "beta"}
    ):
        return False
    for name in ("alpha", "beta"):
        value = mma.kwargs.get(name, 1)
        if type(value) not in (int, float) or value != 1:
            return False
    acc, left, right = mma.args
    accumulator = _tensor(acc)
    result = _tensor(mma)
    lhs = _tensor(lhs_load)
    rhs = _tensor(rhs_load)
    if not (
        accumulator is not None
        and result is not None
        and accumulator.dtype == result.dtype == torch.float32
        and lhs is not None
        and rhs is not None
        and lhs.dtype == rhs.dtype
        and lhs.dtype in (torch.float16, torch.bfloat16)
        and _strip_new_vars(left) is lhs_load
        and lhs_load.target is memory_ops.load
        and rhs_load.target is memory_ops.load
    ):
        return False
    right = _strip_new_vars(right)
    if not (
        isinstance(right, Node)
        and right.op == "call_function"
        and right.target is torch.ops.aten.permute.default
        and len(right.args) == 2
        and _strip_new_vars(right.args[0]) is rhs_load
        and right.args[1] in ([1, 0], (1, 0))
    ):
        return False
    reduction = next((info for info in graphs if info.graph is mma.graph), None)
    root = next((info for info in graphs if isinstance(info, RootGraphInfo)), None)
    if not isinstance(reduction, ForLoopGraphInfo) or root is None:
        return False
    loops = [
        node
        for node in root.graph.nodes
        if node.op == "call_function"
        and node.target is _tracing_ops._for_loop
        and node.args[0] == reduction.graph_id
    ]
    if len(loops) != 1:
        return False
    begin, end = loops[0].args[1:3]
    if not (
        isinstance(begin, (list, tuple))
        and len(begin) == 1
        and type(begin[0]) is int
        and begin[0] == 0
        and isinstance(end, (list, tuple))
        and len(end) == 1
        and _same_extent(end[0], static_k, env)
    ):
        return False
    output = next(node for node in mma.graph.nodes if node.op == "output")
    returned = output.args[0]
    if isinstance(returned, (tuple, list)):
        if len(returned) != 1:
            return False
        returned = returned[0]
    if _strip_new_vars(returned) is not mma:
        return False

    dependencies: set[Node] = set()
    pending = [mma]
    while pending:
        node = pending.pop()
        if node in dependencies:
            continue
        dependencies.add(node)
        pending.extend(node.all_input_nodes)
    permitted = {
        _tracing_ops._new_var,
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        view_ops.subscript,
        torch.ops.aten.sym_size.int,
    }
    for node in mma.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node not in dependencies or node.op != "call_function":
            return False
        if node in (lhs_load, rhs_load, right, mma):
            continue
        if node.target not in permitted:
            return False
    return True
