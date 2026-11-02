"""Select cache hints for already-proven full global store packets."""

from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING

from .cutedsl_compat import fixed_l2_evict_last_store_policy_supported
from .persistent_branch_vec import PACKET_STORE_BYTES

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet


# ``cute_store_policy`` -> packet flush -> its cache-policy form.
# ``l2_evict_last``: ``st.global.L2::cache_hint`` with the validated retention
# descriptor (SM103 / CUDA 13).  ``streaming``: the portable ``.cs`` operator
# (evict-first at L1 and L2), for outputs written once: a 32 MB bf16 output
# stored by default displaced the two 32 MB inputs of the next launch of a
# back-to-back elementwise add from the 126 MB L2 (16.5 us per call); streamed,
# the inputs stay resident (14.9 us, below the Triton backend's 15.1).
_STORE_HELPERS = {
    "l2_evict_last": {
        "_cute_store_u16_vec": "_cute_store_u16x8_l2_evict_last",
        "_cute_store_u32_vec": "_cute_store_u32x4_l2_evict_last",
        "_cute_store_fp8e4m3fn_vec": "_cute_store_fp8e4m3fn_vec_l2_evict_last",
        "_cute_store_fp8e5m2_vec": "_cute_store_fp8e5m2_vec_l2_evict_last",
    },
    "streaming": {
        "_cute_store_u16_vec": "_cute_store_u16x8_streaming",
        "_cute_store_u32_vec": "_cute_store_u32x4_streaming",
        "_cute_store_fp8e4m3fn_vec": "_cute_store_fp8e4m3fn_vec_streaming",
        "_cute_store_fp8e5m2_vec": "_cute_store_fp8e5m2_vec_streaming",
    },
}


def _store_call(statement: ast.stmt) -> tuple[ast.Call, str, int | None] | None:
    """``(call, collector, lanes)`` of a packet flush statement.

    ``lanes`` is the number of collected lanes the flush writes when the
    values are a constant slice of the collector (``values[8:16]``: a lane
    loop wider than one packet flushes several), ``None`` for the whole list.
    """
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if not (
        isinstance(call.func, ast.Name)
        and call.func.id in PACKET_STORE_BYTES
        and len(call.args) == 2
        and not call.keywords
    ):
        return None
    values = call.args[1]
    if isinstance(values, ast.Name):
        return call, values.id, None
    if (
        isinstance(values, ast.Subscript)
        and isinstance(values.value, ast.Name)
        and isinstance(values.slice, ast.Slice)
        and values.slice.step is None
        and isinstance(values.slice.lower, ast.Constant)
        and isinstance(values.slice.upper, ast.Constant)
        and type(values.slice.lower.value) is int
        and type(values.slice.upper.value) is int
        and 0 <= values.slice.lower.value < values.slice.upper.value
    ):
        return (
            call,
            values.value.id,
            values.slice.upper.value - values.slice.lower.value,
        )
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
    target_device_capability: tuple[int, int] | None,
    cuda_version: str | None,
) -> list[ast.stmt]:
    """Rename full 16-byte global packet flushes to the selected policy's form.

    The packet emitters already proved alignment, coverage and the scalar
    tail; this only swaps the final helper.  A flush's width comes from a
    fresh ``[]`` appended once per ``cutlass.range_constexpr(N)`` iteration
    in the same statement list (or a constant slice of such a list, the
    wide-lane emitter's several packets per site); any other use of the list
    keeps the default store.  Flushes the async-store policy already renamed
    are left alone.
    """
    helpers = _STORE_HELPERS.get(policy)
    if helpers is None or (
        policy == "l2_evict_last"
        and not fixed_l2_evict_last_store_policy_supported(
            target_device_capability, cuda_version
        )
    ):
        return body
    # A rebound tensor parameter no longer names its global argument.
    global_tensor_names = frozenset(global_tensor_names) - {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    def split(call: ast.Call, collector: str, lanes: int) -> list[ast.stmt]:
        """Two 16-byte policy flushes for a 32-byte collector (a lane loop of
        16 bf16 or 8 fp32 lanes flushed whole by the hoisted-vector emitter):
        ``helper(ptr, vals[:half])`` and ``helper(ptr + half, vals[half:])``,
        the pointer advanced in elements like the wide-lane emitter does."""
        assert isinstance(call.func, ast.Name)
        half = lanes // 2
        pointer = ast.unparse(call.args[0])
        helper = helpers[call.func.id]
        return [
            ast.parse(f"{helper}({pointer}, {collector}[0:{half}])").body[0],
            ast.parse(
                f"{helper}(({pointer}) + {half}, {collector}[{half}:{lanes}])"
            ).body[0],
        ]

    def rewrite(statements: list[ast.stmt], inherited: dict[str, int]) -> None:
        widths = dict(inherited)
        result: list[ast.stmt] = []
        # Only a list emptied in this statement list may seed a collector: an
        # enclosing unrolled loop could already have appended to an outer one.
        fresh: set[str] = set()
        for statement in statements:
            replacement: list[ast.stmt] = [statement]
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
                fresh.add(name)
            if isinstance(statement, ast.For):
                collected = _collector_widths(
                    statement, {name: widths[name] for name in fresh & widths.keys()}
                )
                rewrite(statement.body, widths)
                rewrite(statement.orelse, widths)
                widths.update(collected)
                keep.update(collected)
            elif isinstance(statement, ast.If):
                rewrite(statement.body, widths)
                rewrite(statement.orelse, widths)
            elif isinstance(statement, ast.With):
                rewrite(statement.body, widths)
            store = _store_call(statement)
            if store is not None:
                call, collector, lanes = store
                assert isinstance(call.func, ast.Name)
                keep.add(collector)
                width = widths.get(collector, 0)
                if lanes is not None:
                    width = lanes if lanes <= width else 0
                itemsize = PACKET_STORE_BYTES[call.func.id]
                if width * itemsize == 16 and _global_pointer(
                    call.args[0], global_tensor_names
                ):
                    call.func.id = helpers[call.func.id]
                elif (
                    width * itemsize == 32
                    and lanes is None
                    and itemsize > 1
                    and _global_pointer(call.args[0], global_tensor_names)
                ):
                    replacement = split(call, collector, width)
            used = {
                node.id for node in ast.walk(statement) if isinstance(node, ast.Name)
            }
            for name in used - keep:
                widths.pop(name, None)
            result.extend(replacement)
        statements[:] = result

    rewrite(body, {})
    return body
