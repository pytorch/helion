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
from .host_function import KernelDefinition
from .tensor_utils import patch_tensor_factories

if TYPE_CHECKING:
    import types

    from ..runtime.tile_dependency import TileDependencySchedule
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
      3. Backend-specific AST customizations
      4. Type propagation
      5. Top-level tile-dependency analysis
      6. Tile-dependency schedule construction
      7. Config spec finalization
      8. Device IR lowering

    The HostFunction is the mutable compilation state that each step
    operates on.
    """

    def __init__(
        self,
        env: CompileEnvironment,
        *,
        tile_dependency_schedule: TileDependencySchedule | None = None,
    ) -> None:
        self.env = env
        self.backend = env.backend
        self.tile_dependency_schedule = tile_dependency_schedule

    def compile(
        self,
        fn: types.FunctionType,
        fake_args: list[object],
        constexpr_args: dict[str, object],
    ) -> HostFunction:
        """Run the full compilation pipeline and return the compiled HostFunction."""
        hf = self.parse(fn, fake_args, constexpr_args)
        with hf, self._compilation_context():
            self.unroll(hf)
            self.customize_ast(hf)
            self.propagate_types(hf)
            self.analyze_tile_dependencies(hf)
            self.schedule_tile_dependencies(hf)
            self.finalize_config()
            self.lower(hf)
        return hf

    def parse(
        self,
        fn: types.FunctionType,
        fake_args: list[object],
        constexpr_args: dict[str, object],
    ) -> HostFunction:
        with measure("HostFunction.parse_ast"):
            source_indented = inspect.getsource(fn)
            source = textwrap.dedent(source_indented)
            column_offset = source_indented.index(source[0])
            code = fn.__code__
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
            root = ast_extension.convert(root, code, column_offset)
            assert isinstance(root, ast.FunctionDef)
            assert isinstance(root, ast_extension.ExtendedAST)
            _validate_ast(root)
            params = inspect.signature(fn).bind(*fake_args)
            params.apply_defaults()

            definition = KernelDefinition(
                fn=fn,
                constexpr_args=constexpr_args,
                name=root.name,
                args=root.args,
                body=root.body,
                params=params,
            )
            return HostFunction(definition, root._location)

    def unroll(self, hf: HostFunction) -> None:
        from .static_loop_unroller import unroll_static_loops

        with measure("HostFunction.unroll_static_loops"):
            unroll_static_loops(hf)

    def customize_ast(self, hf: HostFunction) -> None:
        """Backend-specific AST customizations.

        Rewrites high-level patterns in the user's kernel AST into
        equivalent forms that compile to better code on the active
        backend.
        """
        with measure("HostFunction.customize_ast"):
            self.backend.customize_ast(hf)

    def propagate_types(self, hf: HostFunction) -> None:
        from .type_propagation import propagate_types

        with measure("HostFunction.propagate_types"):
            propagate_types(hf)

    def finalize_config(self) -> None:
        # TODO(hinriksnaer): finalize_config_spec() accesses hf via
        # HostFunction.current() internally. pass hf explicitly?
        with measure("HostFunction.finalize_config_spec"):
            self.env.finalize_config_spec()

    def analyze_tile_dependencies(self, hf: HostFunction) -> None:
        """Record cross-root dependencies for later scheduling/lowering passes."""
        from .loop_dependency_checker import analyze_top_level_tile_dependencies

        with measure("HostFunction.analyze_tile_dependencies"):
            hf.compiler_state.tile_dependency_analysis = (
                analyze_top_level_tile_dependencies(hf.body)
            )

    def schedule_tile_dependencies(self, hf: HostFunction) -> None:
        """Validate and build an explicitly requested tile-dependency schedule."""
        from .tile_dependency_schedule import build_tile_dependency_stage_graph

        with measure("HostFunction.schedule_tile_dependencies"):
            analysis = hf.compiler_state.tile_dependency_analysis
            assert analysis is not None
            unsynchronized = analysis.unsynchronized_tile_dependencies
            if unsynchronized and self.tile_dependency_schedule is None:
                raise exc.LoopDependencyError(unsynchronized[0].name)
            if unsynchronized and analysis.source_phase_starts:
                raise exc.TileDependencyScheduleError(
                    "mixing explicit hl.barrier() phases with implicit "
                    "tile-dependency stages is not supported yet"
                )
            schedule = build_tile_dependency_stage_graph(
                analysis, self.tile_dependency_schedule
            )
            hf.compiler_state.tile_dependency_schedule = schedule
            if not schedule.implicit_stage_starts:
                return
            if self.env.backend_name != "triton" or torch.version.hip is not None:
                dependency = next(
                    dependency
                    for dependency in analysis.unsynchronized_tile_dependencies
                    if dependency.consumer_root in schedule.implicit_stage_starts
                )
                raise exc.LoopDependencyError(dependency.name)
            reason = (
                "tile dependencies require a persistent blocked kernel for "
                "tile-dependency scheduling"
            )
            self.env.require_persistent_blocked_without_cooperative_barrier(reason)

    def lower(self, hf: HostFunction) -> None:
        from .device_ir import lower_to_device_ir

        factory_padding = (
            patch_tensor_factories()
            if self.env.backend.pad_factory_tensors_to_power_of_2
            else contextlib.nullcontext()
        )
        with (
            measure("HostFunction.lower_to_device_ir"),
            factory_padding,
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
