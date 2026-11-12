"""Static shape evaluations see every guard added before them.

``ShapeEnv._maybe_evaluate_static``, the ``simplify`` it starts from and
``get_axioms`` cache their answers keyed on a version counter that a new guard
does not advance, although the guard's axioms can decide the answer.  The
CompileEnvironment's ShapeEnv advances that counter when a guard adds axioms.
"""

from __future__ import annotations

import pytest
import sympy
import torch
from torch._dynamo.source import ConstantSource
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.utils._sympy.functions import Max

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.type_info import NumericType
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _symint(shape_env: ShapeEnv, name: str, hint: int) -> torch.SymInt:
    source = ConstantSource(name)
    symbol = shape_env.create_symbol(
        hint, source=source, dynamic_dim=DimDynamic.DYNAMIC
    )
    symint = shape_env.create_symintnode(symbol, hint=hint, source=source)
    assert isinstance(symint, torch.SymInt)
    return symint


def test_a_guard_decides_a_query_repeated_after_it() -> None:
    env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
    a = _symint(env.shape_env, "a", 5)
    b = _symint(env.shape_env, "b", 6)
    query = sympy.Eq(a.node.expr, b.node.expr)
    assert env.shape_env._maybe_evaluate_static(query) is None
    # Installs the guard Ne(a, b): its axioms decide the query, though the
    # ShapeEnv's version, and so its cache, stays as it was.
    assert a != b
    assert env.shape_env._maybe_evaluate_static(query) is sympy.false


def test_a_guard_decides_a_query_whose_simplification_it_decides() -> None:
    env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
    a = _symint(env.shape_env, "a", 6)
    b = _symint(env.shape_env, "b", 5)
    difference = a.node.expr - b.node.expr
    query = sympy.Eq(Max(0, difference), difference)
    assert env.shape_env._maybe_evaluate_static(query) is None
    # ``ShapeEnv.simplify``, cached on the same version, decides
    # ``Max(0, a - b)`` by the guard ``a >= b``.
    assert a >= b
    assert env.shape_env._maybe_evaluate_static(query) is sympy.true


def test_statically_known_true_sees_a_guard_added_after_it() -> None:
    # The CuTe passes and Inductor's size checks ask the ShapeEnv directly.
    env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
    a = _symint(env.shape_env, "a", 5)
    b = _symint(env.shape_env, "b", 6)
    assert not statically_known_true(a != b)
    assert a != b
    assert statically_known_true(a != b)


def test_simplify_sees_a_guard_added_after_it() -> None:
    # ``known_nonnegative`` simplifies before it evaluates.
    env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
    a = _symint(env.shape_env, "a", 6)
    b = _symint(env.shape_env, "b", 5)
    difference = a.node.expr - b.node.expr
    assert env.shape_env.simplify(Max(0, difference)) != difference
    assert a >= b
    assert env.shape_env.simplify(Max(0, difference)) == difference


def test_get_axioms_sees_a_guard_added_after_it() -> None:
    # Inductor's index propagation reads the axioms without asking a static
    # query first.
    env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
    a = _symint(env.shape_env, "a", 5)
    b = _symint(env.shape_env, "b", 6)
    assert env.shape_env.get_axioms() == ()
    assert a != b
    guards = tuple(guard.expr for guard in env.shape_env.guards)
    assert guards and env.shape_env.get_axioms() == guards


def test_a_guard_records_the_frame_that_compared() -> None:
    # A guard records as its location the first frame outside torch, by which
    # Helion tells a kernel's comparisons from its compiler's: the refresh
    # puts no frame of its own there.
    env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
    a = _symint(env.shape_env, "a", 5)
    b = _symint(env.shape_env, "b", 6)
    assert a != b
    location = env.shape_env.guards[-1].sloc.framework_loc
    assert not isinstance(location, str)
    assert location.filename == __file__


@pytest.mark.parametrize("literal_first", [False, True])
def test_a_guard_makes_a_symbol_known_equal_to_a_literal(literal_first: bool) -> None:
    env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
    with env:
        a = _symint(env.shape_env, "a", 5)
        assert a == 5
        operands = (5, a) if literal_first else (a, 5)
        assert NumericType.known_equal(*operands)


def add_row_slice_twice(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    k = a.size(2)
    out = torch.empty_like(a)
    out2 = torch.empty_like(a)
    for tile0, tile1 in hl.tile([a.size(0), a.size(1)]):
        v = a[tile0, tile1, 0:k] + b[tile0, 0:k]
        out[tile0, tile1, 0:k] = v
        out2[tile0, tile1, 0:k] = v
    return out, out2


@skipUnlessBackends(["triton", "cute"])
@skipIfRefEager("Config tests not applicable in ref eager mode")
def test_a_slice_operand_broadcasts_by_the_guards_of_the_bind() -> None:
    # Placing ``b[tile0, 0:k]``'s dims beside ``[tile0, tile1, 0:k]`` asks
    # ``known_equal`` after the bind added the guards that decide it; the
    # cached, undecided answers broadcast it against the wrong dims
    # (ShapeMismatch "[u0, u1, u2] and [1, u0, u2]").
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(add_row_slice_twice),
            (torch.empty(8, 16, 2), torch.empty(8, 4)),
        )
        config = helion.Config(
            **(bound.config_spec.default_config().config | {"block_sizes": [8, 16]})
        )
        assert "out2" in bound.to_code(config)
