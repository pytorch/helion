from __future__ import annotations

import ast
from textwrap import indent
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ._serial_lane_cpu import _cpu_codegen
from ._serial_lane_model import run
from .test_cute_serial_coarsen import _recurrence
from .test_cute_serial_coarsen import args
from .test_cute_serial_coarsen import config
from .test_cute_serial_lane import _module
from helion import exc
from helion._compiler.cute.host_fastpath import KEY
from helion._compiler.cute.host_fastpath_proof import Integer
from helion._compiler.cute.host_fastpath_proof import Proof
from helion._testing import skipUnlessBackends
from helion.runtime.cute import host_fastpath
from helion.runtime.cute import serial_lane_guard

pytestmark = skipUnlessBackends(["cute"])

GUARD = """if p.toint() % 16 == 0:
    cute.copy(atom, source, target)
else:
    fallback()
"""


def _proof(body: str) -> Proof:
    proof = Proof({"x": ((128,), (1,), "torch.float32", 4)}, (1, 1, 1), (32, 1, 1))
    proof.block_body(ast.parse("p = x.iterator\n" + body + "\n" + GUARD).body, {})
    return proof


@pytest.mark.parametrize("dtype", ["Float32", "Float16", "BFloat16"])
@pytest.mark.parametrize("index", ["0", "3", "cutlass.Int32(2)"])
def test_bounded_local_register_data_keeps_address_facts(dtype, index):
    assert (
        _proof(
            f"r = cute.make_rmem_tensor((4,), cutlass.{dtype})\nr[{index}] = value"
        ).removed
        == 1
    )


@pytest.mark.parametrize(
    "body",
    [
        "r = external()\nr[0] = value",
        "r = cute.make_rmem_tensor((4,), unknown_dtype)\nr[0] = value",
        "r = cute.make_rmem_tensor((2, 2), cutlass.Float32)\nr[0] = value",
        "r = cute.make_rmem_tensor((4,), cutlass.Float32)\nr[4] = value",
        "r = cute.make_rmem_tensor((4,), cutlass.Float32)\nr[-1] = value",
        "r = cute.make_rmem_tensor((4,), cutlass.Float32)\nr[unknown] = value",
        "r = cute.make_rmem_tensor((4,), cutlass.Float32)\nr = x\nr[0] = value",
        "r = cute.make_rmem_tensor((4,), cutlass.Float32)\nr.layout.stride[0] = 2",
        "x[0] = value",
        "r = cute.make_rmem_tensor((4,), cutlass.Float32)\nr[0] += value",
        "r = cute.make_rmem_tensor((4,), cutlass.Float32)\nr[0] = (p := p + 1)",
        "r = cute.make_rmem_tensor((4,), cutlass.Float32)\ni = r[0]\np = p + i",
    ],
)
def test_unknown_alias_metadata_out_of_bounds_and_readback_remain_conservative(body):
    assert _proof(body).removed == 0


def test_alias_and_bounded_loop_keep_only_local_storage_fact():
    assert (
        _proof("""r = cute.make_rmem_tensor((4,), cutlass.Float32)
alias = r
for i in range(4):
    alias[i] = value
""").removed
        == 1
    )


def test_loop_carried_pointer_still_invalidated():
    proof = Proof({"x": ((128,), (1,), "torch.float32", 4)}, (1, 1, 1), (32, 1, 1))
    body = """p = x.iterator
r = cute.make_rmem_tensor((4,), cutlass.Float32)
for i in range(4):
    r[i] = value
    if p.toint() % 16 == 0:
        cute.copy(atom, source, target)
    else:
        fallback()
    p = p + 1
"""
    proof.block_body(ast.parse(body).body, {})
    assert proof.removed == 0


class _BytePointer:
    def __init__(self, address: int = 4096) -> None:
        self.address = address

    def __add__(self, offset: int) -> _BytePointer:
        return _BytePointer(self.address + 4 * offset)

    def toint(self) -> int:
        return self.address


def _execute_pointer_model(tree: ast.Module) -> tuple[list[str], int]:
    events: list[str] = []
    namespace: dict[str, object] = {
        "x": SimpleNamespace(iterator=_BytePointer()),
        "cute": SimpleNamespace(
            make_rmem_tensor=lambda shape, dtype: [0] * shape[0],
            copy=lambda *args: events.append("vector"),
        ),
        "cutlass": SimpleNamespace(Float32=float, Int32=int),
        "fallback": lambda: events.append("scalar"),
        "identity": lambda value: value,
        "atom": None,
        "source": None,
        "target": None,
    }
    exec(compile(ast.fix_missing_locations(tree), "<pointer-model>", "exec"), namespace)
    pointer = namespace["p"]
    assert isinstance(pointer, _BytePointer)
    return events, pointer.toint()


@pytest.mark.parametrize(
    "effect",
    [
        "ignored = (p := p + 1)",
        "ignored = identity(identity(p := p + 1))",
        "identity(p := p + 1)",
        "identity(identity(p := p + 1))",
        "alias = p\nignored = (alias := alias + 1)\np = alias",
        "i = 0\nignored = (i := 1)\np = p + i",
        "ignored = ((p := p + 1), 0)",
        "r[0] = identity(p := p + 1)",
        "r[((p := p + 1), 0)[1]] = 0.0",
        "((p := p + 1), r)[1][0] = 0.0",
    ],
)
def test_nested_assignment_writes_never_retain_stale_pointer_facts(effect):
    text = (
        "p = x.iterator\nr = cute.make_rmem_tensor((4,), cutlass.Float32)\n"
        + effect
        + "\nr[0] = 0.0\n"
        + GUARD
    )
    expected = _execute_pointer_model(ast.parse(text))
    assert expected == (["scalar"], 4100)
    tree = ast.parse(text)
    proof = Proof({"x": ((128,), (1,), "torch.float32", 4)}, (1, 1, 1), (32, 1, 1))
    tree.body = proof.block_body(tree.body, {})
    assert proof.removed == 0
    assert _execute_pointer_model(tree) == expected


def test_simple_assignment_can_use_its_old_value_without_nested_writes():
    text = (
        """p = x.iterator
r = cute.make_rmem_tensor((4,), cutlass.Float32)
p = p + 4
r[0] = 0.0
"""
        + GUARD
    )
    tree = ast.parse(text)
    proof = Proof({"x": ((128,), (1,), "torch.float32", 4)}, (1, 1, 1), (32, 1, 1))
    tree.body = proof.block_body(tree.body, {})
    assert proof.removed == 1
    assert _execute_pointer_model(tree) == _execute_pointer_model(ast.parse(text))
    assert _execute_pointer_model(tree) == (["vector"], 4112)


@pytest.mark.parametrize(
    "copy_call,removed",
    [
        ("cute.copy(atom, source, target)", 1),
        ("cute.autovec_copy(source, target)", 1),
        ("cute.autovec_copy_other(source, target)", 0),
        ("other.autovec_copy(source, target)", 0),
        ("copy(source, target)", 0),
        ("target[0] = source[0]", 0),
    ],
)
def test_only_exact_copy_and_autovec_copy_enable_guard_removal(copy_call, removed):
    text = "p = x.iterator\n" + GUARD.replace(
        "cute.copy(atom, source, target)", copy_call
    )
    tree = ast.parse(text)
    proof = Proof({"x": ((128,), (1,), "torch.float32", 4)}, (1, 1, 1), (32, 1, 1))
    tree.body = proof.block_body(tree.body, {})
    assert proof.removed == removed


@pytest.mark.parametrize("condition", ["p.toint() % 16 == 1", "value > 0.0"])
def test_autovec_does_not_erase_false_or_scalar_arithmetic_conditions(condition):
    text = "p = x.iterator\n" + GUARD.replace("p.toint() % 16 == 0", condition).replace(
        "cute.copy(atom, source, target)", "cute.autovec_copy(source, target)"
    )
    tree = ast.parse(text)
    original = ast.dump(tree)
    proof = Proof({"x": ((128,), (1,), "torch.float32", 4)}, (1, 1, 1), (32, 1, 1))
    tree.body = proof.block_body(tree.body, {})
    assert proof.removed == 0 and ast.dump(tree) == original


@pytest.mark.parametrize(
    "left,right", [(0, 16), (512, 16), (511, 16), (7, 3), (2**31 - 1, 16)]
)
def test_exact_nonnegative_modulo(left, right):
    proof = Proof({}, (1, 1, 1), (32, 1, 1))
    value = proof.value(ast.parse(f"{left} % {right}", mode="eval").body, {})
    expected = left % right
    assert value == Integer(expected, expected, expected)


@pytest.mark.parametrize(
    "expression",
    [
        "512 % 0",
        "512 % -16",
        "-1 % 16",
        "True % 16",
        "512 % True",
        "(i := 512) % 16",
        "512 % (d := 16)",
        "unknown % 16",
        "(2**31) % 16",
    ],
)
def test_invalid_unknown_or_named_assignment_modulo_remains_unknown(expression):
    proof = Proof({}, (1, 1, 1), (32, 1, 1))
    assert proof.value(ast.parse(expression, mode="eval").body, {}) is None


def test_nonexact_modulo_preserves_original_conservative_interval():
    proof = Proof({}, (1, 1, 1), (32, 1, 1))
    expression = ast.parse("i % 16", mode="eval").body
    assert proof.value(expression, {"i": Integer(0, 32, 16)}) == Integer(0, 15, 16)
    assert not proof.true(
        ast.parse("i % 16 == 0", mode="eval").body, {"i": Integer(0, 32, 16)}
    )


def test_exact_modulo_and_autovec_require_every_guard_clause():
    for width, removed in ((512, 1), (511, 0)):
        text = "p = x.iterator\n" + GUARD.replace(
            "p.toint() % 16 == 0", f"(p.toint() % 16 == 0) & ({width} % 16 == 0)"
        ).replace(
            "cute.copy(atom, source, target)", "cute.autovec_copy(source, target)"
        )
        proof = Proof({"x": ((128,), (1,), "torch.float32", 4)}, (1, 1, 1), (32, 1, 1))
        proof.block_body(ast.parse(text).body, {})
        assert proof.removed == removed


def _assert_conditional_pointer_model(text, expected, *, removed):
    original = _execute_pointer_model(ast.parse(text))
    assert original == expected
    tree = ast.parse(text)
    proof = Proof({"x": ((128,), (1,), "torch.float32", 4)}, (1, 1, 1), (32, 1, 1))
    tree.body = proof.block_body(tree.body, {})
    assert proof.removed == removed
    assert _execute_pointer_model(tree) == original


@pytest.mark.parametrize(
    "condition,use_else,events,address",
    [
        ("identity(p := p + 1)", False, ["scalar"], 4100),
        ("identity(identity(p := p + 1))", False, ["scalar"], 4100),
        ("True and identity(p := p + 1)", False, ["scalar"], 4100),
        ("False or identity(p := p + 1)", False, ["scalar"], 4100),
        ("not identity(p := p + 1)", True, ["scalar"], 4100),
        ("False and identity(p := p + 1)", True, ["vector"], 4096),
        ("True or identity(p := p + 1)", False, ["vector"], 4096),
    ],
)
def test_if_test_writes_invalidated_before_both_branches(
    condition, use_else, events, address
):
    body = "r[0] = 0.0\n" + GUARD.replace("atom, source, target", "None, None, None")
    text = "p = x.iterator\nr = cute.make_rmem_tensor((4,), cutlass.Float32)\n"
    text += f"if {condition}:\n" + indent("pass\n" if use_else else body, "    ")
    text += "else:\n" + indent(body if use_else else "pass\n", "    ")
    _assert_conditional_pointer_model(text, (events, address), removed=0)


def test_nested_if_test_assignment_does_not_escape_invalidation():
    body = "r[0] = 0.0\n" + GUARD
    text = "p = x.iterator\nr = cute.make_rmem_tensor((4,), cutlass.Float32)\n"
    text += "if identity(True):\n    if identity(p := p + 1):\n" + indent(
        body, "        "
    )
    _assert_conditional_pointer_model(text, (["scalar"], 4100), removed=0)


@pytest.mark.parametrize(
    "condition",
    ["True", "identity(True)", "True and identity(True)", "not identity(False)"],
)
def test_side_effect_free_if_test_retains_valid_branch_proof(condition):
    text = "p = x.iterator\nr = cute.make_rmem_tensor((4,), cutlass.Float32)\n"
    text += f"if {condition}:\n" + indent("r[0] = 0.0\n" + GUARD, "    ")
    _assert_conditional_pointer_model(text, (["vector"], 4096), removed=1)


@pytest.mark.parametrize(
    "setup,condition,update",
    [
        ("alias = p", "identity(alias := alias + 1)", "p = alias"),
        ("i = 0", "identity(i := 1)", "p = p + i"),
    ],
)
def test_if_test_invalidates_alias_and_integer_facts(setup, condition, update):
    text = (
        "p = x.iterator\nr = cute.make_rmem_tensor((4,), cutlass.Float32)\n"
        + setup
        + "\n"
    )
    text += f"if {condition}:\n" + indent(update + "\nr[0] = 0.0\n" + GUARD, "    ")
    _assert_conditional_pointer_model(text, (["scalar"], 4100), removed=0)


def _code(dtype, depth, factor, enabled, *, steps=16):
    values = args(dtype, steps=steps)
    raw = config(factor=factor, schedule=f"prefetch{depth}")
    raw.config["cute_serial_lane_tail_schedule"] = (
        "peel_final_group" if steps % depth == 0 else "guarded"
    )
    if enabled is not None:
        raw.config[KEY] = enabled
    with _cpu_codegen():
        bound = _recurrence._bind_isolated(values)
        canonical = bound._normalized_config_copy(raw)
        source = bound.to_code(raw)
        assert source == bound.to_code(canonical)
    return source, values, raw


def _functions(source):
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }


def _guards(node):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.If) and ".toint()" in ast.unparse(n.test)
    ]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("factor", [1, 2])
def test_real_repeated_prefetch_source_defaults_fallback_and_preload_stops(
    dtype, depth, factor
):
    from torch._inductor.codecache import PyCodeCache

    omitted, _, _ = _code(dtype, depth, factor, None)
    disabled, _, _ = _code(dtype, depth, factor, False)
    assert omitted == disabled
    source, values, raw = _code(dtype, depth, factor, True)
    old, new = _functions(omitted), _functions(source)
    for name, fn in old.items():
        if any(ast.unparse(d) == "cute.kernel" for d in fn.decorator_list):
            assert ast.dump(fn) == ast.dump(new[name])
    fast = next(fn for name, fn in new.items() if name.endswith("_host_fast"))
    assert not _guards(fast)
    assert _guards(new[fast.name.removesuffix("_host_fast")])
    stops = []

    class Stop(BaseException):
        pass

    def stop(text, **kwargs):
        stops.append(text)
        raise Stop

    with _cpu_codegen(), patch.object(PyCodeCache, "load", side_effect=stop):
        bound = _recurrence._bind_isolated(values)
        for selected in (raw, bound._normalized_config_copy(raw)):
            with pytest.raises(Stop):
                bound.compile_config(selected, allow_print=False)
    assert stops == [source, source]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [5, 16])
def test_actual_fast_body_cpu_math_and_tail_ownership(dtype, steps):
    source, values, _ = _code(dtype, 4, 2, True, steps=steps)
    original, _, _ = _code(dtype, 4, 2, False, steps=steps)
    tree = ast.parse(source)
    fast = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.endswith("_host_fast")
    )
    serial = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == fast.name.removesuffix("_host_fast")
    )
    # The existing CPU interpreter selects the serial symbol. Project only the
    # literal fast body under that symbol; fake aligned CPU storage is modeled.
    serial.body = fast.body
    expected, old_counts = run(original, values, reordered=True)
    actual, new_counts = run(ast.unparse(tree), values, reordered=True)
    assert torch.equal(expected, actual)
    assert old_counts == new_counts


def test_real_current_host_keeps_original_alias_grid_and_alignment_contract():
    source, values, _ = _code(torch.bfloat16, 4, 2, True)
    calls = []
    module = _module(source, calls)
    with (
        patch.object(serial_lane_guard, "_require_cuda"),
        patch.object(host_fastpath, "_is_cuda", return_value=True),
    ):
        module._recurrence(*values)
        assert calls[-1][0].__name__.endswith("_host_fast")
        grid = calls[-1][1]
        alias = (
            values[0]
            .view(torch.bfloat16)
            .flatten()[: values[2].numel()]
            .view(values[2].shape)
        )
        module._recurrence(values[0], values[1], alias)
        assert not calls[-1][0].__name__.endswith("_host_fast")
        assert calls[-1][1][0] == 2 * grid[0]
        offset = torch.empty(values[0].numel() + 1)[1:].view(values[0].shape)
        count = len(calls)
        with pytest.raises(exc.BackendUnsupported, match="launcher alignment"):
            module._recurrence(offset, values[1], values[2])
        assert len(calls) == count
        aligned = torch.empty(values[0].numel() + 4)[4:].view(values[0].shape)
        module._recurrence(aligned, values[1], values[2])
        assert calls[-1][0].__name__.endswith("_host_fast")
        module._recurrence(*values)
        assert calls[-1][0].__name__.endswith("_host_fast")
