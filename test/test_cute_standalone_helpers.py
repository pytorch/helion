from __future__ import annotations

import ast
from contextlib import contextmanager
import dataclasses
import importlib.util
import sys
import textwrap
import types
import typing
from typing import TYPE_CHECKING

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._compiler.cute.standalone import _check_staged_operand_alignment
from helion._compiler.cute.standalone import _concretize_needed_scalars
from helion._compiler.cute.standalone import _drop_parameters
from helion._compiler.cute.standalone import _fit_repro_call
from helion._compiler.cute.standalone import _Helpers
from helion._compiler.cute.standalone import _pin_defaults
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl
from helion.runtime.cute.arg_signature import normalize_scalar
from helion.runtime.cute.standalone_launcher import _standalone_validate

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any


def _load_helper_modules(
    sources: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, source in sources.items():
        path = tmp_path / f"{name.rsplit('.', 1)[-1]}.py"
        path.write_text(textwrap.dedent(source))
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        package, _, child = name.rpartition(".")
        monkeypatch.setattr(sys.modules[package], child, module, raising=False)
        spec.loader.exec_module(module)


@contextmanager
def _standalone_helpers(
    helpers: _Helpers, sources: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, Any]]:
    # Like an exported module, with ``from __future__ import annotations``.
    future = ast.parse("from __future__ import annotations").body
    tree = ast.fix_missing_locations(
        ast.Module(body=[*future, *helpers.emit()], type_ignores=[])
    )
    code = compile(tree, "<standalone-helpers>", "exec")
    namespace: dict[str, Any] = {"types": types}
    with monkeypatch.context() as isolated:
        isolated.setitem(sys.modules, "helion", None)
        for name in sources:
            isolated.setitem(sys.modules, name, None)
        exec(code, namespace)
        yield namespace


def test_late_dependency_keeps_every_required_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "helion._standalone_test_shared": """
            def first():
                return 3

            def late():
                return 7
        """,
        "helion._standalone_test_consumer": """
            from helion._standalone_test_shared import late

            def consume():
                return late() * 2
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    shared = helpers.require("helion._standalone_test_shared", {"first"})
    consumer = helpers.require("helion._standalone_test_consumer", {"consume"})

    with _standalone_helpers(helpers, sources, monkeypatch) as exported:
        assert exported[shared].first() == 3
        assert exported[consumer].consume() == 14


def test_shared_dependency_initializes_before_its_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "helion._standalone_test_dependency": """
            OFFSET = 5
        """,
        "helion._standalone_test_left": """
            from helion._standalone_test_dependency import OFFSET

            INITIAL_VALUE = OFFSET + 1

            def value():
                return INITIAL_VALUE
        """,
        "helion._standalone_test_right": """
            from helion._standalone_test_dependency import OFFSET

            INITIAL_VALUE = OFFSET + 2

            def value():
                return INITIAL_VALUE
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    dependency = helpers.require("helion._standalone_test_dependency", {"OFFSET"})
    left = helpers.require("helion._standalone_test_left", {"value"})
    right = helpers.require("helion._standalone_test_right", {"value"})

    with _standalone_helpers(helpers, sources, monkeypatch) as exported:
        assert exported[dependency].OFFSET == 5
        assert exported[left].value() == 6
        assert exported[right].value() == 7


def test_dataclass_field_types_are_not_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "helion._standalone_test_state": """
            from dataclasses import dataclass
            import typing

            @dataclass
            class State:
                "Softmax state."

                scale: float
                threshold: typing.Final[float] = 0.5
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    state = helpers.require("helion._standalone_test_state", {"State"})

    with _standalone_helpers(helpers, sources, monkeypatch) as exported:
        state_type = exported[state].State
        assert [field.type for field in dataclasses.fields(state_type)] == [
            float,
            typing.Final[float],
        ]
        assert state_type(2.0) == state_type(2.0, 0.5)
        assert state_type.__doc__ == "Softmax state."


@pytest.mark.parametrize(
    "import_statement",
    (
        "from helion._standalone_test_dependency import increment",
        "from ._standalone_test_dependency import increment",
    ),
)
def test_helper_local_import_has_no_helion_dependency(
    import_statement: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "helion._standalone_test_dependency": """
            def increment(value):
                return value + 1
        """,
        "helion._standalone_test_consumer": f"""
            def value():
                {import_statement}
                return increment(8)
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    consumer = helpers.require("helion._standalone_test_consumer", {"value"})

    with _standalone_helpers(helpers, sources, monkeypatch) as exported:
        assert exported[consumer].value() == 9


@pytest.mark.parametrize(
    ("import_statement", "local_name"),
    (
        ("from . import _standalone_test_dependency as helper", "helper"),
        ("from helion import _standalone_test_dependency as helper", "helper"),
        ("from . import _standalone_test_dependency", "_standalone_test_dependency"),
        (
            "from helion import _standalone_test_dependency",
            "_standalone_test_dependency",
        ),
    ),
)
def test_helper_submodule_import(
    import_statement: str,
    local_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        "helion._standalone_test_dependency": """
            def increment(value):
                return value + 1
        """,
        "helion._standalone_test_consumer": f"""
            def value():
                {import_statement}
                return {local_name}.increment(8)
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    consumer = helpers.require("helion._standalone_test_consumer", {"value"})

    with _standalone_helpers(helpers, sources, monkeypatch) as exported:
        assert exported[consumer].value() == 9


@pytest.mark.parametrize(
    ("module_name", "helper_name"),
    (
        ("helion.runtime.cute.paired_sum", "try_paired_sum_cast"),
        ("helion.runtime.cute.single_sum", "try_single_sum_cast"),
    ),
)
def test_runtime_dependent_launch_helper_is_rejected(
    module_name: str,
    helper_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        "helion._standalone_test_consumer": f"""
            def value():
                from {module_name} import {helper_name}
                return {helper_name}
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    helpers.require("helion._standalone_test_consumer", {"value"})

    with pytest.raises(NotImplementedError, match="runtime-dependent launch plan"):
        helpers.emit()


def test_unaliased_dotted_helion_import_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        "helion._standalone_test_dependency": "VALUE = 9\n",
        "helion._standalone_test_consumer": """
            def value():
                import helion._standalone_test_dependency
                return helion._standalone_test_dependency.VALUE
        """,
    }
    _load_helper_modules(sources, tmp_path, monkeypatch)
    helpers = _Helpers()
    helpers.require("helion._standalone_test_consumer", {"value"})

    with pytest.raises(NotImplementedError, match="helper imports require an alias"):
        helpers.emit()


def test_under_aligned_staged_operands_are_refused() -> None:
    """The exported launcher cannot stage an under-aligned operand the way
    ``default_cute_launcher`` does, so export refuses one; an unstaged
    under-aligned tensor exports."""
    aligned = ("tensor", "torch.bfloat16", 1, (8,), (1,), 16)
    under_aligned = ("tensor", "torch.bfloat16", 1, (8,), (1,), 8)
    scalar = ("scalar", "float")
    _check_staged_operand_alignment(((0, False),), (aligned, under_aligned, scalar))
    with pytest.raises(NotImplementedError, match="under-aligned operand"):
        _check_staged_operand_alignment(
            ((0, False), (1, True)), (aligned, under_aligned, scalar)
        )


def test_drop_parameters_keeps_defaults_aligned() -> None:
    """Dropping parameters (callables the export compiles in) keeps every
    remaining default with its parameter: positional-only, defaulted and
    keyword-only ones included."""
    function = ast.parse(
        textwrap.dedent("""
            def kernel(a, fn, /, b, mod=None, c=1.0, *, hook=None, d=True):
                pass
        """)
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    _drop_parameters(function, {"fn", "mod", "hook"})
    assert ast.unparse(function.args) == "a, /, b, c=1.0, *, d=True"


def test_repro_caller_fits_the_entry_point() -> None:
    """The repro caller stops passing dropped callables and passes the
    hard-coded scalars their values instead of placeholders."""
    call = ast.parse(
        textwrap.dedent("""
            def call():
                from torch._dynamo.testing import rand_strided
                x = rand_strided(size=(8,), stride=(1,), dtype=torch.float32, device='cuda')
                fn = "UNSUPPORTED TYPE - REPLACE"
                check = False
                scale = 1.1
                shift = 1.1
                kernel(x, fn, check, scale, shift)
        """)
    ).body[0]
    assert isinstance(call, ast.FunctionDef)
    _fit_repro_call(call, "kernel", {"fn"}, {"check": True, "scale": float("inf")})
    lines = ast.unparse(call).splitlines()
    assert "    fn = 'UNSUPPORTED TYPE - REPLACE'" not in lines
    assert "    check = True" in lines
    assert "    scale = float('inf')" in lines
    assert "    shift = 1.1" in lines
    assert lines[-1] == "    kernel(x, check, scale, shift)"


def test_only_scalars_host_guards_need_are_made_concrete() -> None:
    """Capture keeps bool and float arguments symbolic and makes concrete only
    those whose values it needs (a host branch, the ``float()`` a constexpr
    launch parameter takes): not a bool the host code passes on nor a float
    it computes with. A guard on any other value is refused."""
    shape_env = ShapeEnv()
    with shape_env.ignore_fresh_unbacked_symbols():
        fakes: dict[str, object] = {
            "flag": shape_env.create_unbacked_symbool(),
            "other": shape_env.create_unbacked_symbool(),
            "scale": shape_env.create_unbacked_symfloat(),
            "shift": shape_env.create_unbacked_symfloat(),
            "eps": shape_env.create_unbacked_symfloat(),
        }
        nested = shape_env.create_unbacked_symbool()
    scalars: dict[str, bool | float] = {
        "flag": True,
        "other": False,
        "scale": 2.0,
        "shift": 1.0,
        "eps": 1e-6,
    }
    runs: list[dict[str, bool | float]] = []

    def host(
        flag: Any, other: Any, scale: Any, shift: Any, eps: Any
    ) -> tuple[Any, Any, tuple[str, object]]:
        if not flag:
            raise AssertionError("flag is True")
        assert scale > 0
        # A launch passes ``eps`` to a constexpr parameter, as ``capture`` does.
        return other, shift + 1.0, normalize_scalar(eps)

    def run(values: dict[str, bool | float]) -> None:
        runs.append(dict(values))
        host(**{name: values.get(name, fake) for name, fake in fakes.items()})

    assert _concretize_needed_scalars(run, scalars, fakes) == {
        "flag": True,
        "scale": 2.0,
        "eps": 1e-6,
    }
    assert runs == [
        {},
        {"flag": True},
        {"flag": True, "scale": 2.0},
        {"flag": True, "scale": 2.0, "eps": 1e-6},
    ]

    def run_nested(values: dict[str, bool | float]) -> None:
        bool(nested)

    with pytest.raises(NotImplementedError, match="top-level bool or float"):
        _concretize_needed_scalars(run_nested, scalars, fakes)


def _double(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _host_calls(
    x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    y = fn(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = y[tile] + 1.0
    return out


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _mapped(
    x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = fn(x[tile])
    return out


def _scale_mod(scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def mod(t: torch.Tensor) -> torch.Tensor:
        return t * scale

    return mod


@pytest.mark.parametrize("case", ["host_call", "closure_cell"])
def test_callable_used_by_host_code_is_refused(case: str) -> None:
    """The entry point omits callable arguments, so export refuses a kernel
    whose host code uses one: calls it, or reads a value its closure captures
    (lifted as a runtime input). Refused before capture runs."""
    x = torch.randn(128)
    kernel, fn = (
        (_host_calls, _double) if case == "host_call" else (_mapped, _scale_mod(2.0))
    )
    with cpu_target():
        bound = cpu_bind(kernel, (x, fn))
        with pytest.raises(NotImplementedError, match="callable argument the host"):
            bound.to_code(
                bound.config_spec.default_config(),
                options=helion.OutputCodeOptions(allow_helion_deps=False),
            )


def test_pinned_scalars_default_to_their_values() -> None:
    """A hard-coded parameter's default becomes its pinned value, so a call
    that leaves it out passes the constant check."""
    function = ast.parse(
        textwrap.dedent("""
            def kernel(x, check=False, /, scale=1.0, shift=0.0, *, _launcher=None):
                pass
        """)
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    _pin_defaults(function, {"check": True, "scale": float("inf")})
    assert (
        ast.unparse(function.args)
        == "x, check=True, /, scale=float('inf'), shift=0.0, *, _launcher=None"
    )


def test_float_constants_match_exactly() -> None:
    """A pinned bool or float matches like the launcher's constexpr keys: by
    type, and floats to the sign of zero, with every NaN matching NaN."""
    _standalone_validate(0.0, ("constant", 0.0))
    _standalone_validate(float("nan"), ("constant", float("nan")))
    _standalone_validate(True, ("constant", True))
    _standalone_validate(3, ("constant", 3))
    for arg, entry in (
        (-0.0, ("constant", 0.0)),
        (0.0, ("scalar_constexpr", "float", -0.0, -0.0)),
        (0, ("constant", 0.0)),
        (0, ("scalar_constexpr", "float", -0.0, -0.0)),
        (1, ("constant", True)),
        (1.0, ("constant", True)),
    ):
        with pytest.raises(ValueError, match="does not match the exported CuTe"):
            _standalone_validate(arg, entry)
