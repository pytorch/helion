from __future__ import annotations

import ast
import contextlib
import inspect
import textwrap
import typing
from typing import TYPE_CHECKING

import torch

from .. import exc
from .._compile_time import measure
from . import ast_extension
from .host_function import HostFunction
from .host_function import KernelDef
from .tensor_utils import patch_tensor_factories

if TYPE_CHECKING:
    import types

    from .compile_environment import CompileEnvironment


def _validate_ast(root: ast.FunctionDef) -> None:
    """Validate decorator ordering on the parsed kernel function."""
    # There must always be at least one decorator otherwise we would not have gotten this far
    if len(root.decorator_list) > 1:
        # Decorators are allowed before the helion kernel decorator
        # but are not allowed after
        def get_decorator_name(decorator: ast.expr) -> str:
            if isinstance(decorator, ast.Name):
                return decorator.id
            if isinstance(decorator, ast.Attribute):
                return get_decorator_name(decorator.value)
            if isinstance(decorator, ast.Call):
                return get_decorator_name(decorator.func)
            raise AssertionError(f"Unknown decorator: {decorator}")

        for idx, decorator in enumerate(root.decorator_list):
            # TODO(oulgen): this can break if someone did `import helion as helion2`
            if get_decorator_name(decorator) == "helion":
                if idx != len(root.decorator_list) - 1:
                    raise exc.DecoratorAfterHelionKernelDecorator


class KernelCompiler:
    """Orchestrates the frontend compilation pipeline.

    Creates a HostFunction and drives it through the compilation steps:

      1. Parse source into ExtendedAST
      2. Static loop unrolling
      3. Type propagation
      4. Config spec finalization
      5. Device IR lowering

    The HostFunction is the mutable compilation state that each step
    operates on.
    """

    def __init__(self, env: CompileEnvironment) -> None:
        self.env = env

    def compile(
        self,
        fn: types.FunctionType,
        fake_args: list[object],
        constexpr_args: dict[str, object],
    ) -> HostFunction:
        """Run the full compilation pipeline and return the compiled HostFunction."""
        hf = HostFunction(fn)
        with hf:
            self.parse(hf, fake_args, constexpr_args)
            with self._compilation_context():
                self.unroll(hf)
                self.propagate_types(hf)
                self.finalize_config()
                self.lower(hf)
        return hf

    def parse(
        self,
        hf: HostFunction,
        fake_args: list[object],
        constexpr_args: dict[str, object],
    ) -> None:
        with measure("HostFunction.parse_ast"):
            source_indented = inspect.getsource(hf.fn)
            source = textwrap.dedent(source_indented)
            hf._column_offset = source_indented.index(source[0])
            root = ast.parse(source)
            assert isinstance(root, ast.Module)
            function_defs = [
                stmt for stmt in root.body if isinstance(stmt, ast.FunctionDef)
            ]
            assert len(function_defs) == 1, (
                f"expected one function definition in parsed source, got "
                f"{[type(stmt).__name__ for stmt in root.body]}"
            )
            (root,) = function_defs
            root = ast_extension.convert(root)
            assert isinstance(root, ast.FunctionDef)
            assert isinstance(root, ast_extension.ExtendedAST)
            _validate_ast(root)
            params = inspect.signature(hf.fn).bind(*fake_args)
            params.apply_defaults()
            hf.definition = KernelDef(
                fn=hf.fn,
                constexpr_args=constexpr_args,
                name=root.name,
                args=root.args,
                body=root.body,
                params=params,
            )
            hf.location = root._location

    def unroll(self, hf: HostFunction) -> None:
        from .static_loop_unroller import unroll_static_loops

        with measure("HostFunction.unroll_static_loops"):
            unroll_static_loops(hf)

    def propagate_types(self, hf: HostFunction) -> None:
        from .type_propagation import propagate_types

        with measure("HostFunction.propagate_types"):
            propagate_types(hf)

    def finalize_config(self) -> None:
        # TODO(hinriksnaer): finalize_config_spec() accesses hf via
        # HostFunction.current() internally. pass hf explicitly?
        with measure("HostFunction.finalize_config_spec"):
            self.env.finalize_config_spec()

    def lower(self, hf: HostFunction) -> None:
        from .device_ir import lower_to_device_ir

        with (
            measure("HostFunction.lower_to_device_ir"),
            patch_tensor_factories(),
        ):
            hf.device_ir = lower_to_device_ir(hf)

    @contextlib.contextmanager
    def _compilation_context(self) -> typing.Generator[None, None, None]:
        with (
            # Disable autocast so that compilation reflects the actual dtypes
            # written in the kernel, not the caller's mixed-precision context.
            torch._C._DisableAutocast(),
            # When the PyTorch profiler is active, it may call guard_int()
            # on unbacked SymInts which adds entries to
            # shape_env.replacements, concretizing block-size variables.
            # suppress_guards() prevents this by skipping guard
            # installation (including replacements) in evaluate_expr.
            HostFunction._suppress_guards_if_profiler_enabled(self.env),
            torch.device(self.env.device),
        ):
            yield
