"""CPU codegen tests: CuTe drops the stores of a scratch tensor no load reads.

With ``cute_scratch_forwarding``, ``cute/drop_scratch_stores.py`` removes the
stores of a kernel-local scratch tensor (``helion._compiler.scratch_tensors``)
once nothing in the device body loads it, and ``drop_unused_scratch_arguments``
then removes its kernel parameter, its launcher argument and its host
allocation.  The GPU side is in ``test_cute_scratch_forwarding_cuda.py``.
"""

from __future__ import annotations

import ast
import textwrap

import pytest
import torch

import helion
from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute.drop_scratch_stores import drop_scratch_stores
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.scratch_kernels import RELU_SINK_CONFIG
from helion._testing.scratch_kernels import ROW_SWEEPS_CONFIG
from helion._testing.scratch_kernels import relu_spatial_mean
from helion._testing.scratch_kernels import returned_unread_workspace
from helion._testing.scratch_kernels import row_softmax_bwd
from helion._testing.scratch_kernels import two_sweep_workspace
from helion._testing.scratch_kernels import unread_workspace

pytestmark = pytest.mark.cpu_only


def _code(fn: object, *args: torch.Tensor, **config: object) -> str:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        return bound.to_code(
            spec.normalized_config(
                helion.Config(**{**spec.default_config().config, **config})
            )
        )


def _uses(code: str, name: str) -> int:
    """The uses of ``name`` in ``code``, comments excluded."""
    return sum(
        1
        for node in ast.walk(ast.parse(code))
        if (isinstance(node, ast.Name) and node.id == name)
        or (isinstance(node, ast.arg) and node.arg == name)
    )


def _row_x() -> torch.Tensor:
    return torch.empty(8, 256, dtype=torch.bfloat16)


def test_an_unread_scratch_tensor_loses_its_stores_parameter_and_allocation() -> None:
    code = _code(unread_workspace, _row_x(), cute_scratch_forwarding=True)
    assert _uses(code, "ws") == 0


def test_the_knob_off_keeps_the_stores() -> None:
    code = _code(unread_workspace, _row_x())
    assert "ws = torch.empty(" in code
    assert any("ws.iterator" in line for line in code.splitlines())


def test_a_workspace_served_from_registers_disappears() -> None:
    # The second sweep reads the stored values from registers; the
    # workspace's stores, parameter and allocation go.
    rows = torch.empty(64, 1024, dtype=torch.bfloat16)
    code = _code(
        row_softmax_bwd,
        rows,
        rows,
        torch.empty(64),
        torch.empty(64),
        torch.empty(1),
        **ROW_SWEEPS_CONFIG,
        cute_reduction_reloads=["register"],
        cute_scratch_forwarding=True,
    )
    assert _uses(code, "probs_ws") == 0


def test_a_workspace_reloaded_from_memory_keeps_its_stores() -> None:
    rows = torch.empty(64, 1024, dtype=torch.bfloat16)
    code = _code(
        row_softmax_bwd,
        rows,
        rows,
        torch.empty(64),
        torch.empty(64),
        torch.empty(1),
        **ROW_SWEEPS_CONFIG,
        cute_reduction_reloads=["gmem"],
        cute_scratch_forwarding=True,
    )
    assert "probs_ws = torch.empty(" in code
    assert any(
        "probs_ws.iterator" in line and ".store(" in line for line in code.splitlines()
    )


def test_a_workspace_read_back_in_its_sweep_keeps_its_stores() -> None:
    code = _code(two_sweep_workspace, _row_x(), cute_scratch_forwarding=True)
    assert "ws = torch.empty(" in code
    assert any(
        "ws.iterator" in line and ".store(" in line for line in code.splitlines()
    )


def test_a_returned_workspace_keeps_its_stores() -> None:
    # The caller reads it: not kernel-local scratch.
    code = _code(returned_unread_workspace, _row_x(), cute_scratch_forwarding=True)
    assert "ws = torch.empty(" in code
    assert "ws.iterator" in code


def test_a_sunk_flush_goes_with_its_packet_list_and_appends() -> None:
    # The V-loop sink gathers ``act``'s lanes into a packet list and flushes
    # it after the lane loop (lane by lane where the mask fails).  With
    # ``act`` unread, the flush, the list, its appends and the lane stores
    # go, and so do its parameter and its allocation.
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    shift = torch.empty(120)
    flushed = _code(relu_spatial_mean, x, shift, **RELU_SINK_CONFIG)
    assert "_cute_store_u16_vec(act.iterator" in flushed
    assert "_vsink_store_vals" in flushed
    code = _code(
        relu_spatial_mean, x, shift, **RELU_SINK_CONFIG, cute_scratch_forwarding=True
    )
    assert _uses(code, "act") == 0
    assert "_vsink_store_vals" not in code


def _drop(source: str) -> str:
    """``drop_scratch_stores`` on a device body, ``ws`` the scratch tensor."""
    body = ast.parse(textwrap.dedent(source)).body
    module = ast.Module(body=drop_scratch_stores(body, {"ws"}), type_ignores=[])
    return ast.unparse(module)


def test_the_packet_lists_of_dead_flushes_and_their_loops_go() -> None:
    # The V-loop sink gathers a packet's lanes into a list in a lane loop,
    # then flushes it: with the flush, the list, its appends and the loop go.
    assert (
        _drop(
            """
        _vals = []
        for v in cutlass.range_constexpr(8):
            _vals.append(cutlass.Uint16(x[v]))
        _cute_store_u16_vec(ws.iterator + o, _vals)
        y = x[0]
        """
        )
        == "y = x[0]"
    )


def test_a_list_another_dead_list_reads_goes_with_it() -> None:
    source = """
        _b = []
        for v in cutlass.range_constexpr(8):
            _b.append(x[v])
        _cute_store_u16_vec(ws.iterator + o, _b)
        _a = []
        for v in cutlass.range_constexpr(8):
            _a.append(_b[v])
        _cute_store_u16_vec(ws.iterator + p, _a)
        """
    assert _drop(source) == ""
    # A read of ``_a`` keeps its appends, which keep ``_b``.
    kept = _drop(textwrap.dedent(source) + "y = _a[0]\n")
    assert "_b.append" in kept and "_a.append" in kept and "ws" not in kept


def test_stores_under_while_and_with_go() -> None:
    assert (
        _drop(
            """
        while i < n:
            (ws.iterator + i).store(v)
            i = i + 1
        with cute.arch.elect_one():
            (ws.iterator + o).store(v)
        """
        )
        == "while i < n:\n    i = i + 1\nwith cute.arch.elect_one():\n    pass"
    )


def test_a_load_keeps_every_store() -> None:
    source = """
        (ws.iterator + o).store(v)
        y = (ws.iterator + o).load()
        """
    assert _drop(source) == ast.unparse(ast.parse(textwrap.dedent(source)))


def test_a_scratch_tensor_read_in_another_ones_store_pointer_keeps_its_store() -> None:
    # ``ws``'s store pointer reads ``idx`` (``idx[o]``): a read of ``idx``,
    # not one of its stores, so ``idx``'s store stays, or the kept store of
    # ``ws`` would read memory nothing wrote.
    source = """
        (idx.iterator + o).store(v)
        (ws.iterator + idx[o]).store(w)
        y = (ws.iterator + p).load()
        """
    body = ast.parse(textwrap.dedent(source)).body
    module = ast.Module(body=drop_scratch_stores(body, {"ws", "idx"}), type_ignores=[])
    assert ast.unparse(module) == ast.unparse(ast.parse(textwrap.dedent(source)))


def test_a_deleted_list_keeps_its_appends() -> None:
    # A ``del`` of a value list binds its name too: dropping the init and the
    # appends would leave the ``del`` of an unbound name.
    source = """
        _vals = []
        _vals.append(x)
        _cute_store_u16_vec(ws.iterator + o, _vals)
        del _vals
        """
    assert _drop(source) == "_vals = []\n_vals.append(x)\ndel _vals"


def test_a_statement_at_two_positions_is_pruned_at_each() -> None:
    # Codegen may place one statement object at two positions, and each is
    # pruned on its own: the first keeps its loop, whose target ``y = i``
    # reads, and the second, whose loop nothing reads, goes.
    shared = ast.parse(
        "if c:\n    for i in range(4):\n        (ws.iterator + i).store(v)\n"
    ).body[0]
    body = [shared, statement_from_string("y = i"), shared]
    module = ast.Module(body=drop_scratch_stores(body, {"ws"}), type_ignores=[])
    assert ast.unparse(module) == "if c:\n    for i in range(4):\n        pass\ny = i"


def test_an_emptied_branch_with_an_effect_in_its_test_stays() -> None:
    source = """
        if (out.iterator + o).atomic_add(1) > 0:
            (ws.iterator + o).store(v)
        """
    assert _drop(source) == "if (out.iterator + o).atomic_add(1) > 0:\n    pass"


def test_a_load_sharing_the_store_pointer_node_keeps_the_store() -> None:
    # Codegen may splice one pointer node into a store and a load
    # (``statement_from_string`` does not copy its placeholders): the load's
    # ``ws`` is the store's node, a second occurrence of it.
    pointer = expr_from_string("ws.iterator + o")
    body = [
        statement_from_string("({pointer}).store(v)", pointer=pointer),
        statement_from_string("y = ({pointer}).load()", pointer=pointer),
    ]
    module = ast.Module(body=drop_scratch_stores(body, {"ws"}), type_ignores=[])
    assert (
        ast.unparse(module)
        == "(ws.iterator + o).store(v)\ny = (ws.iterator + o).load()"
    )


def test_an_aliased_list_keeps_its_appends() -> None:
    # ``_vals`` is bound to ``other``, which a later statement reads: only
    # the store goes.
    source = """
        other = []
        _vals = other
        _vals.append(x)
        _cute_store_u16_vec(ws.iterator + o, _vals)
        y = other[0]
        """
    assert _drop(source) == "other = []\n_vals = other\n_vals.append(x)\ny = other[0]"


def test_an_emptied_loop_stays_while_a_later_statement_reads_its_target() -> None:
    assert (
        _drop(
            """
        for i in range(4):
            (ws.iterator + i).store(v)
        y = i
        """
        )
        == "for i in range(4):\n    pass\ny = i"
    )
    # Read by the next iteration of an enclosing loop.
    assert (
        _drop(
            """
        for j in range(2):
            y = i
            for i in range(4):
                (ws.iterator + i).store(v)
        """
        )
        == "for j in range(2):\n    y = i\n    for i in range(4):\n        pass"
    )
    # A later loop that binds the name again does not read this one's.
    assert (
        _drop(
            """
        for i in range(4):
            (ws.iterator + i).store(v)
        for i in range(4):
            y = x[i]
        """
        )
        == "for i in range(4):\n    y = x[i]"
    )


def test_a_list_extended_in_place_keeps_its_init() -> None:
    # ``+=`` reads the list it extends.
    source = """
        _vals = []
        _vals += [x]
        _cute_store_u16_vec(ws.iterator + o, _vals)
        """
    assert _drop(source) == "_vals = []\n_vals += [x]"


def test_operands_with_an_effect_keep_their_statements() -> None:
    # Dropping the statement would drop the atomic its operand performs: a
    # store keeps every store of its tensor, an append its list.
    store = """
        (ws.iterator + o).store((out.iterator + o).atomic_add(1))
        (ws.iterator + p).store(v)
        """
    assert _drop(store) == ast.unparse(ast.parse(textwrap.dedent(store)))
    append = """
        _vals = []
        _vals.append((out.iterator + o).atomic_add(1))
        _cute_store_u16_vec(ws.iterator + o, _vals)
        """
    assert _drop(append) == "_vals = []\n_vals.append((out.iterator + o).atomic_add(1))"


def test_a_store_inside_a_try_keeps_every_store() -> None:
    # The pass does not look into a ``try`` (device code has none).
    source = """
        try:
            y = 1
        finally:
            (ws.iterator + o).store(v)
        (ws.iterator + p).store(v)
        """
    assert _drop(source) == ast.unparse(ast.parse(textwrap.dedent(source)))
