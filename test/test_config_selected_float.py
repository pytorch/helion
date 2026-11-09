from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import math
from types import FunctionType
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch.utils._sympy.functions import TruncToInt

import helion
from helion._compiler.backend_registry import get_backend
from helion._compiler.device_function import DeviceFunction
from helion._compiler.inductor_lowering import APIFuncLowering
from helion._compiler.inductor_lowering import SympyExprLowering
from helion._testing import EXAMPLES_DIR
from helion._testing import cpu_render_target
from helion._testing import import_path
from helion._testing import matchesBackends
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.selected_tunable import emitted_device_arguments
from helion._testing.selected_tunable import nested_selected_float_results
from helion._testing.selected_tunable import render_scalar_operands
from helion._testing.tunable_fragments import IdentityFragment
from helion._testing.tunable_fragments import single_tunable_value
from helion.autotuner import EnumFragment
from helion.autotuner import LocalAutotuneCache
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.external import _ExternalKernelAdapter
from helion.autotuner.external import create_user_config_spec
import helion.language as hl
from helion.language._tracing_ops import _get_symnode
from helion.language.tunable_ops import _get_tunable
from helion.runtime.kernel import _SEARCHING_CANDIDATES

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def host_selected_float(x):
    value = hl.register_tunable("value", EnumFragment((-0.0, 2.5)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * value
    return out, value


@helion.kernel(autotune_effort="none", fast_math=False)
def device_selected_default(x, fragment: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = hl.register_tunable("value", fragment)
        out[tile] = x[tile] * value
    return out


@pytest.mark.parametrize(
    "kind", ["host", "nested", "integer", "boolean", "fraction", "custom"]
)
@onlyBackends(["triton", "cute"])
def test_omitted_registered_default_matches_explicit_render(kind):
    with cpu_render_target():
        if kind == "host":
            bound = cpu_bind(host_selected_float, (torch.empty(17),))
        elif kind == "nested":
            bound = cpu_bind(nested_selected_float_results, (torch.empty(17, 19),))
        else:
            fragment = {
                "integer": IntegerFragment(1, 8, 3),
                "boolean": BooleanFragment(),
                "fraction": EnumFragment((0.5, 1.0)),
                "custom": IdentityFragment(-0.0),
            }[kind]
            bound = cpu_bind(device_selected_default, (torch.empty(17), fragment))
        full = bound.config_spec.default_config()
        expected = full.config["value"]
        sparse = full.minimize(bound.config_spec)
        if kind == "custom":
            # Minimization cannot prove an arbitrary default protocol pure.
            # Explicit omission remains supported by normal compilation.
            assert sparse.config["value"].hex() == expected.hex()
            assert len(sparse.config) < len(full.config)
            sparse.config.pop("value")
        before = sparse.to_json()
        assert "value" not in sparse.config
        explicit = helion.Config.from_dict({**sparse.config, "value": expected})
        assert bound.to_code(sparse) == bound.to_code(explicit)
        normalized = bound._normalized_config_copy(sparse)
        actual = normalized.config["value"]
        assert type(actual) is type(expected)
        if type(expected) is float:
            assert actual.hex() == expected.hex()
        else:
            assert actual == expected
        with _capture_compiled_modules(bound) as loaded:
            first = bound.compile_config(sparse, allow_print=False)
            assert bound.compile_config(explicit, allow_print=False) is first
        assert len(loaded) == 1 and sparse.to_json() == before


@pytest.mark.parametrize("value", [0, False, -0.0])
@onlyBackends(["triton", "cute"])
def test_explicit_registered_value_does_not_call_default(value):
    with cpu_render_target():
        fragment = IdentityFragment(2.5)
        bound = cpu_bind(device_selected_default, (torch.empty(17), fragment))
        sparse = helion.Config(block_sizes=[16], value=value)
        with patch.object(
            fragment, "default", side_effect=AssertionError("explicit selection")
        ):
            normalized = bound._normalized_config_copy(sparse)
        actual = normalized.config["value"]
        assert type(actual) is type(value)
        assert actual.hex() == value.hex() if type(value) is float else actual == value


@onlyBackends(["triton", "cute"])
def test_minimize_keeps_custom_selection_and_removes_schedule_defaults():
    with cpu_render_target():
        fragment = IdentityFragment(-0.0)
        bound = cpu_bind(device_selected_default, (torch.empty(17), fragment))
        full = bound.config_spec.default_config()
        before = full.to_json()
        with patch.object(fragment, "default", wraps=fragment.default) as default:
            minimal = full.minimize(bound.config_spec)
        default.assert_called_once_with()
        assert minimal.config["value"].hex() == (-0.0).hex()
        assert len(minimal.config) < len(full.config)
        assert bound.to_code(minimal) == bound.to_code(full)
        assert full.to_json() == before

        # An omitted arbitrary default may change on each call. Minimization
        # must not evaluate it again to prove a smaller spelling equivalent.
        omitted = helion.Config.from_dict(full.config)
        omitted.config.pop("value")
        with patch.object(fragment, "default", wraps=fragment.default) as default:
            unchanged = omitted.minimize(bound.config_spec)
        default.assert_called_once_with()
        assert unchanged.to_json() == omitted.to_json()


@onlyBackends(["triton", "cute"])
def test_host_selected_float_executes_without_device_context():
    x = torch.empty(17)
    with cpu_render_target():
        bound = cpu_bind(host_selected_float, (x,))
        for selected in (-0.0, 0.0, 2.5):
            config = bound.config_spec.default_config()
            config.config["value"] = selected
            source = ast.parse(bound.to_code(config))
            functions = [n for n in source.body if isinstance(n, ast.FunctionDef)]
            host = next(n for n in functions if n.name == "host_selected_float")
            launcher = Mock()
            namespace = {"torch": torch, "hl": hl}
            if bound.settings.backend == "cute":
                import cutlass

                namespace["cutlass"] = cutlass
            namespace.update({n.name: object() for n in functions if n is not host})
            for default in host.args.kw_defaults:
                if isinstance(default, ast.Name):
                    namespace[default.id] = launcher
            module = ast.Module(
                [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), host],
                [],
            )
            # Execute the real Python wrapper; replace only the device launch.
            # A device-only SDK expression here must fail without an IR context.
            exec(
                compile(ast.fix_missing_locations(module), "<selected_host>", "exec"),
                namespace,
            )
            output, actual = namespace[host.name](x)
            assert output.shape == x.shape
            assert type(actual) is float and actual.hex() == selected.hex()
            launcher.assert_called_once()
            published = launcher.call_args.args[-1]
            assert type(published) is float and published.hex() == selected.hex()


def _evaluate_device_scalars(backend, expressions, dtype=torch.float32):
    if backend != "cute":
        import triton.language as tl
        from triton.runtime.interpreter import _patch_lang

        # Exercise Triton's scalar conversion on CPU, including its handling
        # of signed zero, instead of evaluating every literal as a Python float.
        scope = _patch_lang(FunctionType((lambda: None).__code__, {"tl": tl}))
        try:
            scalar_type = tl.float64 if dtype == torch.float64 else tl.float32
            return [
                float(
                    tl.full(
                        [], eval(code, {"tl": tl, **values}), scalar_type
                    ).handle.data.item()
                )
                for code, values in expressions
            ]
        finally:
            scope.restore()
    import cutlass
    from cutlass._mlir import ir
    from cutlass._mlir import passmanager
    from cutlass._mlir.dialects import func

    scalar_type = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.float64: cutlass.Float64,
    }[dtype]
    # Evaluate all emitted expressions in one real SDK insertion point. A
    # Python-only model would miss SDK interning of opposite floating zeros.
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "probe",
                ir.FunctionType.get([], [scalar_type.mlir_type] * len(expressions)),
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                outputs = [
                    scalar_type(eval(code, {"cutlass": cutlass, **values})).ir_value()
                    for code, values in expressions
                ]
                func.ReturnOp(outputs)
        passmanager.PassManager.parse("builtin.module(canonicalize)").run(
            module.operation
        )
        return [
            ir.FloatAttr(value.owner.attributes["value"]).value
            for value in list(block.operations)[-1].operands
        ]


@onlyBackends(["cute"])
def test_typed_scalar_constants_keep_both_zero_signs():
    overrides = get_backend("cute").inductor_op_overrides()
    # Check actual SDK conversions and folded IR for each floating width.
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        values = (-0.0, 0.0, -0.0)
        emitted = [(overrides.constant(value, dtype), {}) for value in values]
        actual = _evaluate_device_scalars("cute", emitted, dtype)
        assert [value.hex() for value in actual] == [value.hex() for value in values]


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@onlyBackends(["cute"])
def test_negative_zero_constant_has_requested_dtype(dtype):
    import cutlass
    from cutlass._mlir import ir

    scalar_type = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.float64: cutlass.Float64,
    }[dtype]
    expression = get_backend("cute").inductor_op_overrides().constant(-0.0, dtype)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            # Inspect the emitted value before any caller conversion can hide
            # a typed Float32 operand in narrower or wider arithmetic.
            actual = eval(expression, {"cutlass": cutlass})
            assert actual.ir_value().type == scalar_type.mlir_type


@helion.kernel(autotune_effort="none")
def mixed_selected_scalar(x, nested: hl.constexpr, domain: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        selected = hl.register_tunable("value", EnumFragment(domain))
        if nested:
            for column in hl.tile(x.size(1)):
                out[row, column] = x[row, column] * selected
        else:
            out[row, :] = x[row, :] * selected
    return out


@pytest.mark.parametrize("domain", [(2, 4.5), (4.5, 2)])
@pytest.mark.parametrize("nested", [False, True])
@onlyBackends(["triton", "cute"])
def test_mixed_selected_scalar_keeps_selected_type(domain, nested):
    with cpu_render_target():
        bound = cpu_bind(mixed_selected_scalar, (torch.empty(17, 19), nested, domain))
        symbol = single_tunable_value(bound, "value")._sympy_()
        original = APIFuncLowering.codegen
        for selected in domain:
            emitted = []

            def observe(self, ctx, node, *, emitted=emitted):
                result = original(self, ctx, node)
                if node.target is _get_tunable or (
                    node.target is _get_symnode
                    and isinstance(node.meta.get("val"), (torch.SymInt, torch.SymFloat))
                    and symbol in node.meta["val"]._sympy_().free_symbols
                ):
                    emitted.append(ast.unparse(result))
                return result

            config = bound.config_spec.default_config()
            config.config["value"] = selected
            with patch.object(APIFuncLowering, "codegen", observe):
                ast.parse(bound.to_code(config))
            assert emitted
            actual = [ast.literal_eval(code) for code in emitted]
            assert all(
                type(value) is type(selected) and value == selected for value in actual
            )


@onlyBackends(["triton", "cute"])
def test_nested_boolean_capture_preserves_selected_scalar():
    with cpu_render_target():
        bound = cpu_bind(
            mixed_selected_scalar, (torch.empty(17, 19), True, (True, False, 2))
        )
        symbol = single_tunable_value(bound, "value")._sympy_()
        original = APIFuncLowering.codegen
        for selected in (True, False, 2, True):
            emitted = []

            def observe(self, ctx, node, *, emitted=emitted):
                result = original(self, ctx, node)
                value = node.meta.get("val")
                if node.target is _get_tunable or (
                    node.target is _get_symnode
                    and isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool))
                    and value._sympy_() == symbol
                ):
                    emitted.append(ast.literal_eval(ast.unparse(result)))
                return result

            config = bound.config_spec.default_config()
            config.config["value"] = selected
            with patch.object(APIFuncLowering, "codegen", observe):
                ast.parse(bound.to_code(config))
            assert emitted and all(
                type(value) is type(selected) and value == selected for value in emitted
            ), (selected, emitted)
            if type(selected) is bool:
                # A derived Boolean predicate still evaluates as a predicate;
                # only the expression for the direct declaration is a value.
                with bound.env:
                    actual = bound.host_function.resolve_tunable_expr(~symbol, config)
                assert actual is (not selected)


@helion.kernel(autotune_effort="none", static_shapes=False)
def nested_selected_float(x, operation: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        selected = hl.register_tunable("value", EnumFragment((-0.0, 1.0)))
        if operation == "divide":
            value = selected / 2
        elif operation == "negate":
            value = -selected
        elif operation == "runtime":
            value = selected * x.size(0)
        else:
            value = selected
        for column in hl.tile(x.size(1)):
            out[row, column] = x[row, column] * value
    return out


@pytest.mark.parametrize("operation", ["direct", "divide", "negate"])
@onlyBackends(["triton", "cute"])
def test_nested_selected_float_preserves_signed_zero(operation):
    with cpu_render_target():
        bound = cpu_bind(nested_selected_float, (torch.empty(17, 19), operation))
        all_emitted = []
        all_expected = []
        for selected in (-0.0, 0.0, 1.0):
            config = bound.config_spec.default_config()
            config.config["value"] = selected
            _, operands = render_scalar_operands(bound, config)
            emitted = [ast.unparse(operand) for operand in operands]
            expected = (
                selected / 2
                if operation == "divide"
                else -selected
                if operation == "negate"
                else selected
            )
            assert emitted
            all_emitted.extend((code, {}) for code in emitted)
            all_expected.extend([expected] * len(emitted))
        actual = _evaluate_device_scalars(bound.settings.backend, all_emitted)
        assert [value.hex() for value in actual] == [
            value.hex() for value in all_expected
        ]


def _check_selected_runtime_expressions(x, selected_values):
    with cpu_render_target():
        bound = cpu_bind(nested_selected_float, (x, "runtime"))
        for selected in selected_values:
            config = bound.config_spec.default_config()
            config.config["value"] = selected
            source, expressions = render_scalar_operands(bound, config)
            inputs = emitted_device_arguments(bound, config, source, (x, "runtime"))
            for expression in expressions:
                code = ast.unparse(expression)
                names = {
                    node.id
                    for node in ast.walk(expression)
                    if isinstance(node, ast.Name)
                } - {"cutlass", "tl"}
                assert len(names) == 1, code
                (runtime_name,) = names
                assert inputs[runtime_name] == x.size(0)
                (actual,) = _evaluate_device_scalars(
                    bound.settings.backend,
                    [(code, {runtime_name: inputs[runtime_name]})],
                    dtype=x.dtype,
                )
                assert actual.hex() == (selected * x.size(0)).hex(), code


@onlyBackends(["triton", "cute"])
def test_selected_zero_keeps_runtime_expression_and_sign():
    _check_selected_runtime_expressions(torch.empty(17, 19), (-0.0, 0.0))


@onlyBackends(["triton", "cute"])
def test_selected_nonzero_float_keeps_runtime_precision():
    _check_selected_runtime_expressions(
        torch.empty(17, 19, dtype=torch.float64),
        (0.1 + 0.2, 1.2345678901234567),
    )


@onlyBackends(["triton", "cute"])
def test_combined_selected_expressions_do_not_reuse_inner_alias():
    x = torch.empty(17, 38)[:, ::2]
    with cpu_render_target():
        bound = cpu_bind(nested_selected_float_results, (x,))
        symbols = {
            symbol
            for value in bound.env.tunable_values.values()
            for symbol in value._sympy_().free_symbols
        }
        original = DeviceFunction.sympy_expr
        for selected in (-0.0, 0.0):
            printed = []

            def observe(self, expr, *, printed=printed):
                result = original(self, expr)
                if expr.free_symbols & symbols and expr.free_symbols - symbols:
                    printed.append(
                        (result, bool(symbols & self.expr_to_var_info.keys()))
                    )
                return result

            with patch.object(DeviceFunction, "sympy_expr", observe):
                code = bound.to_code(helion.Config(block_sizes=[8, 16], value=selected))
            ast.parse(code)
            assert printed
            for expression, cached_alias in printed:
                assert cached_alias, "Exercise reuse after a prior inner-loop use"
                names = {
                    node.id
                    for node in ast.walk(ast.parse(expression, mode="eval"))
                    if isinstance(node, ast.Name)
                }
                assert names.isdisjoint(map(str, symbols)), expression


@contextmanager
def _capture_compiled_modules(bound):
    loaded = []

    def load(source, **kwargs):
        module = ModuleType(f"selected_zero_{len(loaded)}")
        module.__file__ = f"selected_zero_{len(loaded)}.py"

        def compiled():
            return source

        compiled = FunctionType(
            compiled.__code__,
            compiled.__globals__
            | {
                f"_helion_{bound.kernel.name}": SimpleNamespace(
                    device_caches={0: object()}
                )
            },
            closure=compiled.__closure__,
        )
        setattr(module, bound.kernel.name, compiled)
        loaded.append(module)
        return module

    with (
        patch("torch._inductor.codecache.PyCodeCache.load", side_effect=load),
        patch.object(bound.env.backend, "setup_compile_cache_dir"),
        patch.object(bound.env.backend, "annotate_compiled_module"),
        patch.object(bound.env.backend, "after_compile"),
    ):
        yield loaded


@pytest.mark.parametrize("explicit_zero", [False, True])
@onlyBackends(["triton"])
def test_legacy_minimized_winner_clears_effective_default_cache(explicit_zero):
    with cpu_render_target():
        bound = cpu_bind(host_selected_float, (torch.empty(17),))
        defaults = bound.config_spec.default_config()
        # Model a search default distinct from generic normalization. TileIR's
        # current heuristic otherwise happens to use the same num_warps=4.
        defaults.config["num_warps"] = 8
        selected = 0.0 if explicit_zero else -0.0
        full = helion.Config.from_dict(defaults.config | {"value": selected})
        other = helion.Config.from_dict(defaults.config | {"value": -selected})
        # Preserve the sparse spelling produced before minimization checked
        # normalization. This also covers old saved autotune configurations.
        minimized = helion.Config(block_sizes=full.block_sizes)
        if explicit_zero:
            minimized.config["value"] = selected
        normalized = bound._normalized_config_copy(minimized)
        assert normalized.num_warps != full.num_warps
        before = minimized.to_json()
        with _capture_compiled_modules(bound) as loaded:
            first = bound.compile_config(full, allow_print=False)
            second = bound.compile_config(other, allow_print=False)
        assert first is not second
        first_jit = first.__globals__[f"_helion_{bound.kernel.name}"]
        other_jit = second.__globals__[f"_helion_{bound.kernel.name}"]
        with patch.object(bound.config_spec, "default_config", return_value=defaults):
            bound.env.backend.finalize_ephemeral_cache(bound, minimized)
        assert not first_jit.device_caches
        assert other_jit.device_caches
        first_key = bound._normalized_config_copy(full)
        second_key = bound._normalized_config_copy(other)
        assert first_key not in bound._compile_cache
        assert first_key not in bound._cache_path_map
        assert bound._compile_cache[second_key] is second
        assert bound._cache_path_map[second_key] == loaded[1].__file__
        assert minimized.to_json() == before


@pytest.mark.parametrize("reverse", [False, True])
@onlyBackends(["triton", "cute"])
def test_compiled_selected_zero_keeps_both_signs(reverse):
    with cpu_render_target():
        bound = cpu_bind(nested_selected_float_results, (torch.empty(17, 19),))
        values = (0.0, -0.0) if reverse else (-0.0, 0.0)
        configs = [helion.Config(block_sizes=[8, 16], value=v) for v in values]
        # Public Config equality remains Python numeric equality. Compilation
        # must nevertheless distinguish the selected values' device behavior.
        assert configs[0] == configs[1]
        with _capture_compiled_modules(bound) as loaded:
            first = bound.compile_config(configs[0], allow_print=False)
            assert bound.compile_config(configs[0], allow_print=False) is first
            assert not bound.has_compiled_config(configs[1])
            assert bound.get_cached_path(configs[1]) is None
            second = bound.compile_config(configs[1], allow_print=False)
            assert second is not first
            assert bound.compile_config(configs[1], allow_print=False) is second
            assert bound.compile_config(configs[0], allow_print=False) is first
            assert bound.get_cached_path(configs[0]) != bound.get_cached_path(
                configs[1]
            )
            with patch.object(
                bound.env.backend, "compiled_cache_key", side_effect=lambda _, fn: fn()
            ):
                assert bound.backend_cache_key(configs[0]) == first()
                assert bound.backend_cache_key(configs[1]) == second()
        assert len(loaded) == 2
        assert first() != second()
        if matchesBackends(["triton"]):
            bound.env.backend.finalize_ephemeral_cache(bound, configs[0])
            assert not bound.has_compiled_config(configs[0])
            assert bound.has_compiled_config(configs[1])
        else:
            with patch.object(
                bound.env.backend, "_compiled_kernels", return_value=[]
            ) as kernels:
                bound.env.backend.finalize_ephemeral_cache(bound, configs[0])
                bound.env.backend.finalize_ephemeral_cache(bound, configs[1])
            assert [call.args[0] for call in kernels.call_args_list] == [first, second]


@onlyBackends(["triton", "cute"])
def test_primed_selected_zero_keeps_source_and_render_state():
    with cpu_render_target():
        bound = cpu_bind(nested_selected_float_results, (torch.empty(17, 19),))
        configs = [helion.Config(block_sizes=[8, 16], value=v) for v in (-0.0, 0.0)]
        sources = [bound.to_code(config) for config in configs]
        assert sources[0] != sources[1]
        for config, source in zip(configs, sources, strict=True):
            bound.prime_rendered_code(config, source)
        with _capture_compiled_modules(bound) as loaded:
            token = _SEARCHING_CANDIDATES.set(True)
            try:
                primed = [
                    bound.compile_config(config, allow_print=False)
                    for config in configs
                ]
                assert [fn() for fn in primed] == sources
            finally:
                _SEARCHING_CANDIDATES.reset(token)
            # Each config must be rendered again once after leaving search;
            # the other sign's render-only mark must remain independent.
            rendered = [
                bound.compile_config(config, allow_print=False) for config in configs
            ]
            assert [fn() for fn in rendered] == sources
            assert all(a is not b for a, b in zip(primed, rendered, strict=True))
            assert len(loaded) == 4
            assert all(
                bound.compile_config(config, allow_print=False) is fn
                for config, fn in zip(configs, rendered, strict=True)
            )


def test_external_compile_cache_preserves_nested_float_types_and_paths():
    configs = [
        helion.Config(value=value)
        for value in (
            1,
            1.0,
            [-0.0, 1],
            [0.0, 1],
            {"left": (-0.0,), "right": (0.0,)},
            {"left": (0.0,), "right": (-0.0,)},
        )
    ]
    compiled = []

    def compile_fn(config):
        value = repr(config)

        def call():
            return value

        compiled.append(call)
        return call

    adapter = _ExternalKernelAdapter(
        create_user_config_spec({}), compile_fn, (), device=torch.device("cpu")
    )
    for config in configs:
        call = adapter.compile_config(config)
        assert call() == repr(config)
        assert adapter.compile_config(helion.Config.from_dict(config.config)) is call
    assert len(compiled) == len(configs)


@pytest.mark.parametrize("selected", [-0.0, 1.0])
@onlyBackends(["triton", "cute"])
def test_local_cache_persists_backend_key_for_minimized_config(tmp_path, selected):
    args = (torch.empty(17, 19),)
    with (
        cpu_render_target(),
        patch(
            "helion.autotuner.local_cache.extract_device",
            return_value=torch.device("cuda:0"),
        ),
        patch(
            "helion.autotuner.local_cache.get_device_name",
            return_value="cpu-test-target",
        ),
    ):
        bound = cpu_bind(nested_selected_float_results, args)
        config = bound._normalized_config_copy(
            helion.Config(block_sizes=[8, 16], value=selected)
        )
        minimized = config.minimize(bound.config_spec)
        assert len(minimized.config) < len(config.config)
        with patch("torch.cuda.is_available", return_value=True):
            cache = LocalAutotuneCache(BaseSearch(bound, args))
        path = tmp_path / "best_config"
        with patch.object(cache, "_get_local_cache_path", return_value=path):
            cache.put(minimized)
            assert "backend_cache_key" not in json.loads(path.read_text())
            with (
                _capture_compiled_modules(bound),
                patch.object(
                    bound.env.backend, "compiled_cache_key", return_value="compiled-key"
                ),
            ):
                bound.compile_config(config, allow_print=False)
                cache.put(minimized)
            data = json.loads(path.read_text())
            assert data["backend_cache_key"] == "compiled-key"
            assert data["config"] == minimized.to_json()


@pytest.mark.parametrize("selected", [None, -0.0, 0.0, 1.0])
@onlyBackends(["cute"])
def test_minimized_winner_finalizes_existing_cute_launcher(selected):
    with cpu_render_target():
        if selected is None:
            kernel = import_path(EXAMPLES_DIR / "add.py").add
            bound = cpu_bind(kernel, (torch.empty(16, dtype=torch.bfloat16),) * 2)
        else:
            bound = cpu_bind(nested_selected_float_results, (torch.empty(17, 19),))
        config = bound.config_spec.default_config()
        if selected is not None:
            config.config["value"] = selected
        minimized = config.minimize(bound.config_spec)
        if selected is not None:
            # Preserve the explicit selection, including the sign of zero.
            # Resolving an omitted user default is a separate renderer path.
            minimized.config["value"] = selected
        before = minimized.to_json()
        assert len(minimized.config) < len(config.config)
        assert bound.to_code(config) == bound.to_code(minimized)
        with _capture_compiled_modules(bound) as loaded:
            compiled = bound.compile_config(config, allow_print=False)
        persist = Mock()
        device_kernel = SimpleNamespace(
            _helion_cute_compiled_launchers={
                "winner": SimpleNamespace(persist_compiled=persist)
            }
        )
        with (
            patch.object(
                bound.env.backend, "_compiled_kernels", return_value=(device_kernel,)
            ) as kernels,
            patch.object(bound.env.backend, "setup_compile_cache_dir"),
            patch(
                "torch._inductor.codecache.PyCodeCache.load",
                side_effect=AssertionError("finalization must not compile"),
            ),
        ):
            bound.env.backend.finalize_ephemeral_cache(bound, minimized)
        kernels.assert_called_once_with(compiled, bound.kernel.name)
        persist.assert_called_once_with()
        assert len(loaded) == 1 and minimized.to_json() == before


@helion.kernel(autotune_effort="none")
def selected_float_truncation(x, nested: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        selected = hl.register_tunable("selected", EnumFragment((-4.5, 2.5)))
        value = math.trunc(selected)
        if nested:
            for column in hl.tile(x.size(1)):
                out[row, column] = x[row, column] * value
        else:
            out[row, :] = x[row, :] * value
    return out


@pytest.mark.parametrize("nested", [False, True])
@onlyBackends(["triton", "cute"])
def test_selected_float_truncation_emits_python_integer(nested):
    with cpu_render_target():
        bound = cpu_bind(selected_float_truncation, (torch.empty(3, 5), nested))
        originals = {
            APIFuncLowering: APIFuncLowering.codegen,
            SympyExprLowering: SympyExprLowering.codegen,
        }
        sources = []
        for selected in (-4.5, 2.5, -4.5):
            emitted = []

            def observe(self, ctx, node, *, emitted=emitted):
                code = originals[type(self)](self, ctx, node)
                value = node.meta.get("val")
                if isinstance(value, torch.SymInt) and value._sympy_().has(TruncToInt):
                    emitted.append(ast.literal_eval(code))
                return code

            config = bound.config_spec.default_config()
            config.config["selected"] = selected
            with (
                patch.object(APIFuncLowering, "codegen", observe),
                patch.object(SympyExprLowering, "codegen", observe),
            ):
                sources.append(bound.to_code(config))
            assert emitted
            assert all(
                type(value) is int and value == math.trunc(selected)
                for value in emitted
            )
        assert sources[0] == sources[2]
