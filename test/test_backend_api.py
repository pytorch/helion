from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import pytest

import helion
from helion import backend_api
from helion._compiler import backend_registry
from helion._compiler.backend_registry import get_backend_class
from helion._compiler.backend_registry import registry_generation
from helion._compiler.backend_registry import unregister_compiler_backend
from helion._testing import onlyBackends
from helion.autotuner import benchmarking
from helion.backend_api import TritonBackend
from helion.language import _tracing_ops
from helion.language import reduce_ops

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(helion.__file__).resolve().parent.parent
_DOCS_PATH = Path(__file__).resolve().parents[1] / "docs" / "api" / "backend_api.md"

# Changing anything below changes the out-of-tree backend contract: decide
# whether BACKEND_API_VERSION must be bumped (see docs/api/backend_api.md).
_EXPECTED_ALL = [
    "Argument",
    "BACKEND_API_VERSION",
    "COVERED_BACKEND_HOOKS",
    "COVERED_STRATEGY_METHODS",
    "Backend",
    "BackendState",
    "BlockSizeSpec",
    "BoundKernel",
    "CodegenState",
    "CompactedShape",
    "CompileEnvironment",
    "Config",
    "ConfigSpec",
    "ConfigSpecFragment",
    "DeviceBodyContext",
    "DeviceFunction",
    "DeviceGridState",
    "DeviceIR",
    "DeviceLoopState",
    "FiniteSearch",
    "ForLoopGraphInfo",
    "GraphInfo",
    "HostFunction",
    "IfGraphInfo",
    "InductorOpOverrides",
    "LoopedReductionStrategy",
    "Lowering",
    "PersistentReductionStrategy",
    "ReductionLoopGraphInfo",
    "ReductionStrategy",
    "TileStrategy",
    "TileStrategyDispatch",
    "TritonBackend",
    "VectorWidthSpec",
    "codegen",
    "exc",
    "expr_from_string",
    "mask_to_op",
    "reduce_op",
    "register_compiler_backend",
    "statement_from_string",
    "symint_sympy_expr",
]

_BACKEND_HOOKS = {
    "name": "property -> 'str'",
    "dtype_str": "(self, dtype: 'torch.dtype') -> 'str'",
    "acc_type": "(self, dtype: 'torch.dtype') -> 'str'",
    "function_decorator": "property -> 'str'",
    "constexpr_type": "property -> 'str'",
    "default_launcher_name": "property -> 'str'",
    "library_imports": "property -> 'dict[str, str]'",
    "codegen_name": "property -> 'str'",
    "state": "property -> 'BackendState'",
    "validate_environment": "(self) -> 'None'",
    "pre_inductor_lowering": "(self, node: 'torch.fx.Node') -> 'Lowering | None'",
    "inductor_op_overrides": "(self) -> 'InductorOpOverrides'",
    "customize_ast": "(self, hf: 'HostFunction') -> 'bool'",
    "pre_codegen": "(self, graphs: 'list[GraphInfo]', config: 'Config', tile_strategy: 'TileStrategyDispatch') -> 'None'",
    "create_loop_strategy": "(self, fn: 'DeviceFunction', block_ids: 'list[int]', config: 'Config') -> 'TileStrategy'",
    "create_reduction_strategy": "(self, fn: 'DeviceFunction', block_id: 'int', reduction_loop: 'int | None') -> 'TileStrategy'",
    "program_id_expr": "(self, dim: 'int', *, index_dtype: 'str') -> 'str'",
    "grid_index_expr": "(self, offset_var: 'str', block_size_var: 'str', dtype: 'str', *, axis: 'int') -> 'str'",
    "loop_index_expr": "(self, offset_var: 'str', block_size_var: 'str', dtype: 'str', *, axis: 'int') -> 'str'",
    "arange_expr": "(self, offsets_var: 'str', lid: 'str', block_size_var: 'str', dtype: 'str', *, axis: 'int' = 0) -> 'str'",
    "range_str": "(self, begin: 'str | None', end: 'str', step: 'str | None') -> 'str | None'",
    "lane_index_expr": "(self, offset_var: 'str', elements_per_thread: 'int', *, axis: 'int') -> 'str'",
    "lane_offset_expr": "(self, lane_var: 'str') -> 'str'",
    "thread_in_tile_mask_expr": "(self, block_size_var: 'str', *, axis: 'int' = 0) -> 'str | None'",
    "scalar_load_expr": "(self, tensor_name: 'str', index_expr: 'str | None' = None) -> 'str'",
    "inline_constexpr": "(self, name: 'str', value: 'str') -> 'str'",
    "scalar_arg_preamble": "(self, arg: 'Argument') -> 'list[ast.AST]'",
    "finalize_device_body": "(self, body: 'list[ast.stmt]', ctx: 'DeviceBodyContext') -> 'tuple[list[ast.stmt], list[ast.arg]]'",
    "cast_expr": "(self, expr_str: 'str', dtype_str: 'str') -> 'str'",
    "cast_scalar_ast": "(self, x: 'ast.AST', target_dtype: 'torch.dtype') -> 'ast.AST'",
    "where_expr": "(self, mask: 'str', true_val: 'str', false_val: 'str') -> 'str'",
    "minimum_expr": "(self, a: 'str', b: 'str') -> 'str'",
    "zeros_expr": "(self, shape: 'str', dtype: 'str') -> 'str'",
    "full_expr": "(self, shape_dims: 'list[str]', value_expr: 'str', dtype: 'torch.dtype') -> 'str'",
    "reshape_expr": "(self, expr: 'str', shape: 'str') -> 'str'",
    "broadcast_to_expr": "(self, expr: 'str', shape: 'str') -> 'str'",
    "expands_broadcast_dims": "(self) -> 'bool'",
    "reduction_expr": "(self, input_name: 'str', reduction_type: 'str', dim: 'int', *, block_size_var: 'str | None' = None, threads_in_group: 'int | None' = None, dtype: 'torch.dtype | None' = None) -> 'str'",
    "reduction_combine_expr": "(self, reduction_type: 'str', acc: 'str', val: 'str', dtype: 'torch.dtype') -> 'str'",
    "reduction_index_expr": "(self, block_size_var: 'str', dtype: 'str', block_idx: 'int', *, axis: 'int') -> 'str'",
    "reduction_index_zero_expr": "(self, dtype: 'str') -> 'str'",
    "wrap_reduction_accumulator": "(self, acc_full: 'str', *, thread_count: 'int', loop_block_size: 'int', acc_dtype: 'torch.dtype') -> 'str'",
    "looped_reduction_thread_count": "(self, *, requested: 'int', block_size: 'int', block_index: 'int', config: 'Config', config_spec: 'ConfigSpec') -> 'int | None'",
    "register_reduction_loop_config_slots": "(self, env: 'CompileEnvironment', block_id: 'int', size_hint: 'int') -> 'None'",
    "max_reduction_threads": "(self) -> 'int | None'",
    "max_reduction_loop": "(self) -> 'int | None'",
    "reduction_block_size_is_inlined_constexpr": "(self) -> 'bool'",
    "launches_surplus_tile_threads": "(self) -> 'bool'",
    "launcher_keyword_args": "(self, config: 'Config', *, has_barrier: 'bool') -> 'list[str]'",
    "supports_config_key": "(self, key: 'str') -> 'bool'",
    "tunable_fragments": "(self) -> 'dict[str, ConfigSpecFragment]'",
    "supports_precompile": "(self) -> 'bool'",
    "adjust_block_size_constraints": "(self, block_specs: 'list[object]', ndim: 'int', block_sizes: 'list[object] | None' = None, kernel_tensor_sizes: 'dict[tuple[object, ...], int] | None' = None, min_element_bits: 'int' = 32) -> 'None'",
    "autotune": "(self, bound_kernel: 'BoundKernel[Any]', args: 'Sequence[object]', *, force: 'bool' = True, **kwargs: 'object') -> 'Config'",
    "get_do_bench": "(self) -> 'Callable[..., float | tuple[float, ...]] | None'",
    "get_interleaved_bench": "(self) -> 'Callable[..., list[float]] | None'",
}

_STRATEGY_HOOKS = {
    "TileStrategy.user_size": "(self, block_index: 'int') -> 'sympy.Expr'",
    "TileStrategy.codegen_grid": "(self, state: 'CodegenState') -> 'DeviceGridState'",
    "TileStrategy.codegen_device_loop": "(self, state: 'CodegenState') -> 'DeviceLoopState'",
    "TileStrategy.compact_shape": "(self, shapes: 'list[CompactedShape]') -> 'list[CompactedShape]'",
    "TileStrategy.codegen_preamble": "(self, state: 'CodegenState') -> 'None'",
    "ReductionStrategy.codegen_reduction": "(self, state: 'CodegenState', input_name: 'str', reduction_type: 'str', dim: 'int', fake_input: 'torch.Tensor', fake_output: 'torch.Tensor') -> 'ast.AST'",
}


_ABSTRACT_MEMBERS = frozenset(
    {
        "name",
        "dtype_str",
        "acc_type",
        "function_decorator",
        "constexpr_type",
        "default_launcher_name",
        "library_imports",
    }
)


def _restore_codegen_repair_state(test: unittest.TestCase) -> None:
    # Registering a backend that reuses Triton codegen marks "triton" for
    # codegen repair; put the repaired set back after the test.
    test.addCleanup(
        setattr,
        backend_registry,
        "_REPAIRED_CODEGEN_NAMES",
        backend_registry._REPAIRED_CODEGEN_NAMES,
    )


def _signature(cls: type, name: str) -> str:
    # eval_str=False keeps stringized annotations strings on every supported
    # Python (PEP 649 makes annotation evaluation lazier on newer versions),
    # so these pins are stable across interpreters.
    attr = inspect.getattr_static(cls, name)
    if isinstance(attr, property):
        fget = attr.fget
        assert fget is not None
        ret = inspect.signature(fget, eval_str=False).return_annotation
        return f"property -> {ret!r}"
    if isinstance(attr, (staticmethod, classmethod)):
        attr = attr.__func__
    return str(inspect.signature(attr, eval_str=False))


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestBackendApiSurface(unittest.TestCase):
    def test_exported_names(self) -> None:
        self.assertEqual(sorted(backend_api.__all__), sorted(_EXPECTED_ALL))
        public = {n for n in vars(backend_api) if not n.startswith("_")}
        public.discard("annotations")
        self.assertEqual(public, set(_EXPECTED_ALL))

    def test_version_lives_in_registry(self) -> None:
        self.assertIsInstance(backend_api.BACKEND_API_VERSION, int)
        self.assertIs(
            backend_api.BACKEND_API_VERSION, backend_registry.BACKEND_API_VERSION
        )

    def test_docs_version_matches_constant(self) -> None:
        if not _DOCS_PATH.exists():
            self.skipTest("docs not shipped with an installed package")
        doc = _DOCS_PATH.read_text()
        match = re.search(r":value: (\d+)", doc)
        self.assertIsNotNone(match, "docs :value: directive missing")
        self.assertEqual(int(match.group(1)), backend_api.BACKEND_API_VERSION)

    def test_covered_sets_are_the_pinned_ones(self) -> None:
        # The exported covered sets are exactly the names whose signatures are
        # pinned below; changing either changes the contract.
        self.assertEqual(list(backend_api.COVERED_BACKEND_HOOKS), list(_BACKEND_HOOKS))
        self.assertEqual(
            list(backend_api.COVERED_STRATEGY_METHODS), list(_STRATEGY_HOOKS)
        )

    def test_device_body_context_covered_fields(self) -> None:
        # The DeviceBodyContext fields finalize_device_body's docs cover.
        fields = {f.name for f in dataclasses.fields(backend_api.DeviceBodyContext)}
        self.assertLessEqual({"device_function", "args", "body_start"}, fields)

    def test_abstract_members(self) -> None:
        # Making a covered hook abstract breaks every out-of-tree subclass at
        # instantiation; pin the set so it can only change with a bump.
        self.assertEqual(
            set(backend_api.Backend.__abstractmethods__), set(_ABSTRACT_MEMBERS)
        )

    def test_codegen_decorator_signature(self) -> None:
        # The most-used call of the plugin pattern (rule 4).
        params = inspect.signature(backend_api.codegen).parameters
        self.assertEqual(
            [(p.name, p.kind) for p in params.values()],
            [
                ("original_fn", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                ("backend", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ],
        )

    def test_same_class_from_a_second_module_copy_is_explained(self) -> None:
        first = type(
            "Twin", (TritonBackend,), {"name": property(lambda s: "_test_twin")}
        )
        second = type(
            "Twin", (TritonBackend,), {"name": property(lambda s: "_test_twin")}
        )
        second.__module__ = "__main__"
        _restore_codegen_repair_state(self)
        backend_api.register_compiler_backend(first)
        self.addCleanup(unregister_compiler_backend, "_test_twin")
        with self.assertRaisesRegex(ValueError, "second copy of its module"):
            backend_api.register_compiler_backend(second)
        # Unrelated modules that happen to share a class name get no hint.
        second.__module__ = "other_plugin.backend"
        with self.assertRaises(ValueError) as ctx:
            backend_api.register_compiler_backend(second)
        self.assertNotIn("second copy", str(ctx.exception))

    def test_register_compiler_backend_overwrite_guard(self) -> None:
        # A different class under a registered name (e.g. a built-in) raises
        # and leaves the registry untouched (same class: test_backend_registry).
        class _TritonImpostor(TritonBackend):
            @property
            def name(self) -> str:
                return "triton"

        with self.assertRaisesRegex(ValueError, "already registered to"):
            backend_api.register_compiler_backend(_TritonImpostor)
        self.assertIs(get_backend_class("triton"), TritonBackend)

    def test_reloaded_backend_module_reregisters(self) -> None:
        # importlib.reload (or autoreload) defines a new class object under
        # the same name; it replaces the old one instead of raising. The class
        # registers through a decorator, which runs before the module rebinds
        # the name.
        source = (
            "from helion import backend_api\n\n"
            "def register(cls):\n"
            "    backend_api.register_compiler_backend(cls)\n"
            "    return cls\n\n"
            "@register\n"
            "class Reloadable(backend_api.TritonBackend):\n"
            "    @property\n"
            "    def name(self):\n"
            "        return '_test_reload'\n"
        )
        _restore_codegen_repair_state(self)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "_hreload_backend.py").write_text(source)
            sys.path.insert(0, tmp)
            self.addCleanup(sys.path.remove, tmp)
            self.addCleanup(sys.modules.pop, "_hreload_backend", None)
            self.addCleanup(unregister_compiler_backend, "_test_reload")
            module = importlib.import_module("_hreload_backend")
            first = get_backend_class("_test_reload")
            generation = registry_generation()
            importlib.reload(module)
        second = get_backend_class("_test_reload")
        self.assertIsNot(second, first)
        self.assertIs(second, module.Reloadable)
        self.assertGreater(registry_generation(), generation)

    def test_ops_are_the_internal_objects(self) -> None:
        # Backends attach @codegen handlers to these objects.
        self.assertIs(backend_api.reduce_op, reduce_ops._reduce)
        self.assertIs(backend_api.mask_to_op, _tracing_ops._mask_to)

    def test_backend_hook_signatures(self) -> None:
        for name, expected in _BACKEND_HOOKS.items():
            with self.subTest(name=name):
                self.assertEqual(_signature(backend_api.Backend, name), expected)

    def test_strategy_hook_signatures(self) -> None:
        for qualname, expected in _STRATEGY_HOOKS.items():
            cls_name, name = qualname.split(".")
            with self.subTest(name=qualname):
                cls = getattr(backend_api, cls_name)
                self.assertEqual(_signature(cls, name), expected)

    def test_reference_timers_accept_the_documented_keywords(self) -> None:
        # Helion's own timers take the keywords the docs say a plugin timer
        # is called with (test_backend_api_compile checks the call sites).
        single = inspect.signature(benchmarking.do_bench).parameters
        for name in ("warmup", "rep", "return_mode", "process_group_name"):
            self.assertIn(name, single)
        paired = inspect.signature(benchmarking.interleaved_bench).parameters
        for name in ("repeat", "desc", "max_total_ms"):
            self.assertIn(name, paired)

    def test_docs_list_every_covered_hook(self) -> None:
        if not _DOCS_PATH.exists():
            self.skipTest("docs not shipped with an installed package")
        doc = _DOCS_PATH.read_text()
        for name in [
            *backend_api.COVERED_BACKEND_HOOKS,
            *backend_api.COVERED_STRATEGY_METHODS,
        ]:
            with self.subTest(name=name):
                self.assertIn(f"`{name}`", doc)

    @pytest.mark.slow  # starts a fresh interpreter
    def test_import_first_in_fresh_process(self) -> None:
        # Plugins may import this module before anything else in helion.
        result = subprocess.run(
            [sys.executable, "-c", "import helion.backend_api"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            env=os.environ.copy(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
