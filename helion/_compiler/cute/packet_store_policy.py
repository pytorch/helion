"""Select cache hints for already-proven full global store packets."""

from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING

from .cutedsl_compat import fixed_l2_evict_last_store_policy_supported

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet


_STORE_BYTES = {
    "_cute_store_u8_vec": 1,
    "_cute_store_u16_vec": 2,
    "_cute_store_u32_vec": 4,
    "_cute_store_fp8e4m3fn_vec": 1,
    "_cute_store_fp8e5m2_vec": 1,
}
_STORE_HELPERS = {
    "_cute_store_u8_vec": "_cute_store_u8_vec_l2_evict_last",
    "_cute_store_u16_vec": "_cute_store_u16x8_l2_evict_last",
    "_cute_store_u32_vec": "_cute_store_u32x4_l2_evict_last",
    "_cute_store_fp8e4m3fn_vec": "_cute_store_fp8e4m3fn_vec_l2_evict_last",
    "_cute_store_fp8e5m2_vec": "_cute_store_fp8e5m2_vec_l2_evict_last",
}


def _store_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        isinstance(call.func, ast.Name)
        and call.func.id in _STORE_BYTES
        and len(call.args) == 2
        and not call.keywords
    ):
        return call
    return None


def _global_pointer(pointer: ast.expr, tensor_names: AbstractSet[str]) -> bool:
    tensors = {
        node.value.id
        for node in ast.walk(pointer)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }
    return len(tensors) == 1 and tensors <= tensor_names


def _collector_widths(loop: ast.For, widths: dict[str, int]) -> dict[str, int]:
    if not (
        isinstance(loop.iter, ast.Call)
        and ast.unparse(loop.iter.func) == "cutlass.range_constexpr"
        and len(loop.iter.args) == 1
        and not loop.iter.keywords
        and isinstance(loop.iter.args[0], ast.Constant)
        and type(loop.iter.args[0].value) is int
        and loop.iter.args[0].value > 0
        and not loop.orelse
        and not any(
            isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise))
            for node in ast.walk(loop)
        )
    ):
        return {}
    count = loop.iter.args[0].value
    names = Counter(node.id for node in ast.walk(loop) if isinstance(node, ast.Name))
    collected = {}
    for statement in loop.body:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        call = statement.value
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.attr == "append"
            and len(call.args) == 1
            and not call.keywords
        ):
            name = call.func.value.id
            if widths.get(name) == 0 and names[name] == 1:
                collected[name] = count
    return collected


def lower_packet_store_policy(
    body: list[ast.stmt],
    *,
    policy: str,
    global_tensor_names: AbstractSet[str],
    argument_names: AbstractSet[str] = frozenset(),
    target_device_capability: tuple[int, int] | None,
    cuda_version: str | None,
) -> list[ast.stmt]:
    """Hint exactly 16 bytes without changing any pointer or mask proof.

    The recognized compiler helpers already require a naturally aligned full
    packet and preserve their scalar tail fallback. This pass changes only
    the final store helper, after conversion/packing and memory scheduling.
    Global tensor arguments exclude shared/register pointers. Fresh constant
    collections establish width; unknown mutation or aliasing discards it.
    The recurrence-specific store helpers are left as selected by that pass.
    """
    if policy != "l2_evict_last" or not fixed_l2_evict_last_store_policy_supported(
        target_device_capability, cuda_version
    ):
        return body
    shadowed = set(argument_names) | {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    def rewrite(statements: list[ast.stmt], inherited: dict[str, int]) -> None:
        widths = dict(inherited)
        for statement in statements:
            keep: set[str] = set()
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.List)
                and not any(
                    isinstance(value, ast.Starred) for value in statement.value.elts
                )
            ):
                name = statement.targets[0].id
                widths[name] = len(statement.value.elts)
                keep.add(name)
            if isinstance(statement, ast.For):
                collected = _collector_widths(statement, widths)
                rewrite(statement.body, widths)
                rewrite(statement.orelse, widths)
                widths.update(collected)
                keep.update(collected)
            elif isinstance(statement, ast.If):
                rewrite(statement.body, widths)
                rewrite(statement.orelse, widths)
            elif isinstance(statement, ast.With):
                rewrite(statement.body, widths)
            call = _store_call(statement)
            if call is not None:
                assert isinstance(call.func, ast.Name)
                values = call.args[1]
                count = None
                if isinstance(values, ast.Name):
                    count = widths.get(values.id)
                    keep.add(values.id)
                elif isinstance(values, ast.List) and not any(
                    isinstance(value, ast.Starred) for value in values.elts
                ):
                    count = len(values.elts)
                if (
                    count is not None
                    and count * _STORE_BYTES[call.func.id] == 16
                    and call.func.id not in shadowed
                    and _global_pointer(call.args[0], global_tensor_names)
                ):
                    call.func.id = _STORE_HELPERS[call.func.id]
            used = {
                node.id for node in ast.walk(statement) if isinstance(node, ast.Name)
            }
            for name in used - keep:
                widths.pop(name, None)

    rewrite(body, {})
    return body
