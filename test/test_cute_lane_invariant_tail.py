"""GPU-free coverage for the lane-invariant tail of a split lane loop.

``split_lane_loop_reductions`` (``tile_strategy.py``) lowers a lane loop
holding a ``_helion_lane_reduce`` marker into an accumulate pass, the
cross-thread combine, the lane-invariant tail once, and a consume pass for the
lane-varying side effects.  The tail used to be emitted between the passes
whatever the body's order, so ``out[tile_m, :] = row; out[tile_m, 0] = 0.0``
zeroed the element before the consume pass overwrote it.  A tail statement
that must stay after a consume-pass statement, by the lane-loop
distribution's ordering (registers, tensors, unknown effects, non-relaxed
atomics), now follows the consume pass together with the statements ordered
after it; one that must also stay before another consume-pass statement is
rejected; the rest keep their place between the passes.
"""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.tile_strategy import _lane_invariant_tail_after_consume
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = skipUnlessBackends(["cute"])


# --- the placement rules, on generated-code-shaped statements ---------------

_ROW_LOAD = (
    "row = (x.iterator + i * x.layout.stride[0] + lane * x.layout.stride[1]).load()"
)
_ROW_STORE = (
    "(out.iterator + i * out.layout.stride[0] + lane * out.layout.stride[1]).store(row)"
)
_ROW_STORE_AGAIN = "(out.iterator + i * out.layout.stride[0] + lane * out.layout.stride[1]).store(row * 2)"
_ZERO = "(out.iterator + i * out.layout.stride[0]).store(cutlass.Float32(0.0))"
_OTHER_STORE = "(out2.iterator + i * out2.layout.stride[0] + lane * out2.layout.stride[1]).store(row)"
_FLAG_STORE = (
    "(flags.iterator + i * flags.layout.stride[0]).store(cutlass.Float32(1.0))"
)
_FIRST_LOAD = "first = (out.iterator + i * out.layout.stride[0]).load()"
_TWICE = "twice = first * 2"
_TWICE_STORE = "(flags.iterator + i * flags.layout.stride[0]).store(twice)"
_SCALED_STORE = (
    "(out2.iterator + i * out2.layout.stride[0] + lane * out2.layout.stride[1])"
    ".store(row * first)"
)
_MYSTERY = "_helion_mystery(out.iterator + lane, row)"
_PACKED_MARKER_LOAD = (
    "load = _helion_persistent_branch_vec_load(1, 8, 'cutlass.BFloat16', '', "
    "packed.iterator + lane * packed.layout.stride[0], None)"
)
_CONVERT = "v = cutlass.Float32(load)"
_STATE_MARKER_STORE = (
    "_helion_persistent_branch_vec_store(1, 8, 'cutlass.BFloat16', "
    "next_state.iterator + lane * next_state.layout.stride[0], v, None)"
)
_OUT_MARKER_STORE = (
    "_helion_persistent_branch_vec_store(1, 8, 'cutlass.BFloat16', "
    "out.iterator + lane * out.layout.stride[0], v, None)"
)
_X_ZERO = "(x.iterator + i * x.layout.stride[0]).store(cutlass.Float32(0.0))"
_GEOMETRY_LANE = (
    "reduce_lane = cutlass.Int32(cute.arch.thread_idx()[0]) "
    "+ cutlass.Int32(cute.arch.thread_idx()[1]) * cutlass.Int32(cute.arch.block_dim()[0])"
)
_MYSTERY_VALUE = "v = _helion_mystery(out.iterator + i, 3)"
_TWICE_MYSTERY = "w = v * 2"
_TWICE_MYSTERY_STORE = "(flags.iterator + i * flags.layout.stride[0]).store(w)"
_UNRELATED_REGISTER = "k = i * 2"
_UNRELATED_STORE = "(flags.iterator + i * flags.layout.stride[0]).store(k)"


def _order(
    *lines: str,
    invariant: set[int],
    accumulated: set[int],
    consumed: set[int],
) -> set[int]:
    statements = ast.parse(textwrap.dedent("\n".join(lines))).body
    return _lane_invariant_tail_after_consume(
        list(statements),
        invariant=invariant,
        accumulated=accumulated,
        consumed=consumed,
        rename_groups={},
    )


def test_a_store_after_a_per_lane_store_of_its_tensor_follows_the_pass() -> None:
    after = _order(
        _ROW_LOAD, _ROW_STORE, _ZERO, invariant={2}, accumulated={0}, consumed={0, 1}
    )
    assert after == {2}


def test_a_store_before_a_per_lane_store_of_its_tensor_stays_between_the_passes() -> (
    None
):
    after = _order(
        _ROW_LOAD, _ZERO, _ROW_STORE, invariant={1}, accumulated={0}, consumed={0, 2}
    )
    assert after == set()


def test_a_store_into_another_tensor_stays_between_the_passes() -> None:
    after = _order(
        _ROW_LOAD,
        _ROW_STORE,
        _FLAG_STORE,
        _OTHER_STORE,
        invariant={2},
        accumulated={0},
        consumed={0, 1, 3},
    )
    assert after == set()


def test_a_store_between_two_per_lane_stores_of_its_tensor_is_rejected() -> None:
    with pytest.raises(exc.BackendUnsupported, match="between per-lane statements"):
        _order(
            _ROW_LOAD,
            _ROW_STORE,
            _ZERO,
            _ROW_STORE_AGAIN,
            invariant={2},
            accumulated={0},
            consumed={0, 1, 3},
        )


def test_the_statements_ordered_after_a_moved_load_move_with_it() -> None:
    # The load of ``out`` follows the per-lane stores; the register statement
    # and the store that read its result cannot stay ahead of it.
    after = _order(
        _ROW_LOAD,
        _ROW_STORE,
        _FIRST_LOAD,
        _TWICE,
        _TWICE_STORE,
        invariant={2, 3, 4},
        accumulated={0},
        consumed={0, 1},
    )
    assert after == {2, 3, 4}


def test_a_moved_load_the_pass_reads_is_rejected() -> None:
    with pytest.raises(exc.BackendUnsupported, match="between per-lane statements"):
        _order(
            _ROW_LOAD,
            _ROW_STORE,
            _FIRST_LOAD,
            _SCALED_STORE,
            invariant={2},
            accumulated={0},
            consumed={0, 1, 3},
        )


def test_a_store_ordered_before_an_accumulated_load_of_its_tensor_is_rejected() -> None:
    with pytest.raises(exc.BackendUnsupported, match="before the accumulate pass"):
        _order(_X_ZERO, _ROW_LOAD, invariant={0}, accumulated={1}, consumed={1})


def test_unknown_effects_in_the_pass_order_the_tail() -> None:
    # A call the ordering cannot see through keeps every memory access on its
    # side: the flag store follows one such call, and is rejected between two.
    after = _order(
        _ROW_LOAD,
        _MYSTERY,
        _FLAG_STORE,
        invariant={2},
        accumulated={0},
        consumed={0, 1},
    )
    assert after == {2}
    with pytest.raises(exc.BackendUnsupported, match="between per-lane statements"):
        _order(
            _ROW_LOAD,
            _MYSTERY,
            _FLAG_STORE,
            _MYSTERY,
            invariant={2},
            accumulated={0},
            consumed={0, 1, 3},
        )


def test_register_use_of_an_unknown_calls_result_moves_with_it() -> None:
    # The call touches ``out`` and follows the consume pass; the register
    # statement reading the value it binds, and the store reading that, go
    # with it instead of being emitted ahead of the definition they read.
    after = _order(
        _ROW_LOAD,
        _ROW_STORE,
        _MYSTERY_VALUE,
        _TWICE_MYSTERY,
        _TWICE_MYSTERY_STORE,
        invariant={2, 3, 4},
        accumulated={0},
        consumed={0, 1},
    )
    assert after == {2, 3, 4}
    # A register statement sharing no local name with the call keeps its
    # place between the passes.
    after = _order(
        _ROW_LOAD,
        _ROW_STORE,
        _MYSTERY_VALUE,
        _UNRELATED_REGISTER,
        _UNRELATED_STORE,
        invariant={2, 3, 4},
        accumulated={0},
        consumed={0, 1},
    )
    assert after == {2, 4}


def test_thread_geometry_reads_are_ordered_like_registers() -> None:
    # The persistent two-stage reduce derives its lane from ``thread_idx`` and
    # ``block_dim``; reading the launch geometry has no effect, so the
    # statement neither follows the loads before it nor precedes the store.
    after = _order(
        _ROW_LOAD,
        _ROW_STORE,
        _GEOMETRY_LANE,
        _OTHER_STORE,
        invariant={2},
        accumulated={0},
        consumed={0, 1, 3},
    )
    assert after == set()


def test_persistent_vector_markers_are_loads_and_stores_of_their_tensors() -> None:
    # The persistent-reduction markers expand into a vector load and a vector
    # store after the split: a store into ``out`` between a load of ``packed``
    # and a store into ``next_state`` keeps its place, and one after a marker
    # store into ``out`` follows the pass.
    after = _order(
        _PACKED_MARKER_LOAD,
        _CONVERT,
        _ZERO,
        _STATE_MARKER_STORE,
        invariant={2},
        accumulated={0, 1},
        consumed={0, 1, 3},
    )
    assert after == set()
    after = _order(
        _PACKED_MARKER_LOAD,
        _CONVERT,
        _OUT_MARKER_STORE,
        _ZERO,
        invariant={3},
        accumulated={0, 1},
        consumed={0, 1, 2},
    )
    assert after == {3}


# --- the rules in generated kernels -----------------------------------------


def _generate(kernel: object, args: tuple[torch.Tensor, ...], **config: object) -> str:
    with _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, args)
        return bound.to_code(helion.Config.from_dict(config))


def _kernel_function(code: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_"):
            return node
    raise AssertionError(code)


def _pass_indices(function: ast.FunctionDef) -> list[int]:
    """The positions of the lane passes among the kernel's statements."""
    return [
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.For)
    ]


def _store_calls(node: ast.AST, tensor: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "store"
        and f"{tensor}.iterator" in ast.unparse(call.func.value)
    ]


def _tail_store_index(
    function: ast.FunctionDef, tensor: str, value: Callable[[str], bool]
) -> int:
    """The position of the one tail statement storing such a ``value`` into
    ``tensor`` (possibly under an owner predicate)."""
    (index,) = [
        index
        for index, statement in enumerate(function.body)
        if not isinstance(statement, ast.For)
        and any(
            value(ast.unparse(call.args[0])) for call in _store_calls(statement, tensor)
        )
    ]
    return index


def _is_zero(value: str) -> bool:
    return value == "cutlass.Float32(0.0)"


def _is_a_sum(value: str) -> bool:
    return value.startswith(("cutlass.Float32(sum", "cutlass.Float32(v_"))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_zero(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    sums = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        out[tile_m, :] = row
        out[tile_m, 0] = 0.0
        sums[tile_m] = torch.sum(row, dim=1)
    return out, sums


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _zero_then_copy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    sums = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        out[tile_m, 0] = 0.0
        out[tile_m, :] = row
        sums[tile_m] = torch.sum(row, dim=1)
    return out, sums


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_zero_copy_again(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    sums = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        out[tile_m, :] = row
        out[tile_m, 0] = 0.0
        out[tile_m, :] = row * 2
        sums[tile_m] = torch.sum(row, dim=1)
    return out, sums


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_sum_into_the_row(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        out[tile_m, :] = row
        out[tile_m, 0] = torch.sum(row, dim=1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_then_flag(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out2 = torch.empty_like(x)
    flags = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        out[tile_m, :] = row
        flags[tile_m] = torch.sum(row, dim=1)
        out2[tile_m, :] = row * 2
    return out, out2, flags


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _softmax_then_zero(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        m = torch.amax(row, dim=1, keepdim=True)
        e = torch.exp(row - m)
        s = torch.sum(e, dim=1, keepdim=True)
        out[tile_m, :] = e / s
        out[tile_m, 0] = 0.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _zero_then_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        m = torch.amax(row, dim=1, keepdim=True)
        e = torch.exp(row - m)
        s = torch.sum(e, dim=1, keepdim=True)
        out[tile_m, 0] = 0.0
        out[tile_m, :] = e / s
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _softmax_zero_softmax_again(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        m = torch.amax(row, dim=1, keepdim=True)
        e = torch.exp(row - m)
        s = torch.sum(e, dim=1, keepdim=True)
        out[tile_m, :] = e / s
        out[tile_m, 0] = 0.0
        out[tile_m, :] = e * 2
    return out


# One program per row; its 32 (or 8) threads sweep the 256 columns in 8 (or
# 32) lane iterations, so every tile store of ``out`` is a consume-pass store.
_CONFIG = {
    "block_sizes": [1],
    "num_threads": [0, 32],
    "reduction_loops": [None],
    "cute_vector_widths": [1, 1],
}
_CONFIG_FEW_THREADS = {**_CONFIG, "num_threads": [0, 8]}
_ARGS = (torch.empty((8, 256)),)


@pytest.mark.parametrize(
    "config", [_CONFIG, _CONFIG_FEW_THREADS], ids=["threads32", "threads8"]
)
@pytest.mark.parametrize(
    "kernel", [_copy_then_zero, _softmax_then_zero], ids=["one_pass", "dependent"]
)
def test_store_after_the_per_lane_store_of_its_tensor_follows_the_consume_pass(
    kernel: object, config: dict[str, object]
) -> None:
    # The consume pass (the last lane pass) stores every element of ``out``;
    # the zeroing store that follows it in the body follows it in the kernel.
    code = _generate(kernel, _ARGS, **config)
    function = _kernel_function(code)
    passes = _pass_indices(function)
    assert len(passes) >= 2, code
    assert _store_calls(function.body[passes[-1]], "out"), code
    assert _tail_store_index(function, "out", _is_zero) > passes[-1], code


@pytest.mark.parametrize(
    "kernel", [_zero_then_copy, _zero_then_softmax], ids=["one_pass", "dependent"]
)
def test_store_before_the_per_lane_store_of_its_tensor_rejects_the_config(
    kernel: object,
) -> None:
    # Every thread of the row stores the zero, then thread 0 overwrites the
    # element in the consume pass: another thread's zero may land after it,
    # and the barrier between them cannot sit in the body the reduction split
    # rewrites.
    with pytest.raises(exc.BackendUnsupported, match="a body a later pass rewrites"):
        _generate(kernel, _ARGS, **_CONFIG)


def test_reduced_value_stored_into_the_copied_row_follows_the_consume_pass() -> None:
    # The store of the reduction result is owner-guarded (every thread holds
    # the combined value); the guard moves with it after the consume pass.
    code = _generate(_copy_then_sum_into_the_row, _ARGS, **_CONFIG)
    function = _kernel_function(code)
    passes = _pass_indices(function)
    index = _tail_store_index(function, "out", _is_a_sum)
    assert index > passes[-1], code
    assert isinstance(function.body[index], ast.If), code


def test_store_into_another_tensor_stays_between_the_passes() -> None:
    # A tail store the consume pass's accesses are independent of keeps its
    # place, ahead of the pass.
    code = _generate(_copy_then_flag, _ARGS, **_CONFIG)
    function = _kernel_function(code)
    passes = _pass_indices(function)
    assert _store_calls(function.body[passes[-1]], "out"), code
    assert _store_calls(function.body[passes[-1]], "out2"), code
    assert passes[0] < _tail_store_index(function, "flags", _is_a_sum) < passes[-1], (
        code
    )


@pytest.mark.parametrize(
    "kernel",
    [_copy_zero_copy_again, _softmax_zero_softmax_again],
    ids=["one_pass", "dependent"],
)
def test_store_between_two_per_lane_stores_of_its_tensor_rejects_the_config(
    kernel: object,
) -> None:
    # Two consume-pass stores of ``out`` surround the zeroing store: neither
    # side of the pass keeps the body's order.  The race check refuses it
    # first: the other threads' zeros race with thread 0's stores of the
    # element in the body the reduction split rewrites.
    with pytest.raises(exc.BackendUnsupported, match="a body a later pass rewrites"):
        _generate(kernel, _ARGS, **_CONFIG)
