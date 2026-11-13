from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.config_fragment_registration import HIGH_LITERAL_GLOBAL
from helion._testing.config_fragment_registration import check_shared_fragment_revisit
from helion._testing.config_fragment_registration import container_fragment_revisit
from helion._testing.config_fragment_registration import global_thread_fragment
from helion._testing.config_fragment_registration import grid_fragment
from helion._testing.config_fragment_registration import helper_returned_fragment
from helion._testing.cute_codegen import cpu_bind
from helion._testing.selected_tunable import emitted_device_arguments
from helion._testing.selected_tunable import render_scalar_operands
from helion._testing.tunable_fragments import render_tunable_config
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
from helion.autotuner.config_fragment import NumThreadsFragment
import helion.language as hl
from helion.language.tunable_ops import _get_tunable

pytestmark = pytest.mark.cpu_only


def _register_from_helper():
    return hl.register_tunable("helper_factor", EnumFragment((2, 16)))


@helion.kernel(autotune_effort="none")
def helper_registered_tunable(x, on_device: hl.constexpr):
    out = torch.empty_like(x)
    if not on_device:
        host_value = _register_from_helper()
    for tile in hl.tile(x.numel()):
        if on_device:
            value = _register_from_helper()
        else:
            value = host_value
        out[tile] = x[tile] + value
    return out


@pytest.mark.parametrize("on_device", [False, True])
@onlyBackends(["triton", "cute"])
def test_module_helper_registration_reports_direct_call_requirement(on_device):
    with (
        cpu_render_target(),
        pytest.raises(
            helion.exc.TypeInferenceError,
            match=r"hl\.register_tunable\(\).*Python helper function.*Return the fragment",
        ),
    ):
        cpu_bind(helper_registered_tunable, (torch.empty(37), on_device))
    # Reference-mode dispatch precedes the compilation-only helper diagnostic.
    reference = helion.kernel(
        helper_registered_tunable.fn,
        config={"helper_factor": 16},
        ref_mode=helion.RefMode.EAGER,
    )
    x = torch.arange(37, dtype=torch.float32)
    torch.testing.assert_close(reference(x, on_device), x + 16)


@onlyBackends(["triton", "cute"])
def test_module_helper_returning_fragment_keeps_selected_value():
    with cpu_render_target():
        bound = cpu_bind(helper_returned_fragment, (torch.empty(37), True))
        render_tunable_config(bound, "helper_factor", 16)


@helion.kernel(autotune_effort="none")
def nested_fragment(x, use_while: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        if use_while:
            index = 0
            while index < 3:
                value = hl.register_tunable("value", EnumFragment((2, 16)))
                result = result + value
                index += 1
        else:
            for _index in range(3):
                value = hl.register_tunable("value", EnumFragment((2, 16)))
                result = result + value
        out[tile] = result
    return out


@helion.kernel(autotune_effort="none")
def outer_registered_fragment(x, kind: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        if kind == "bool":
            value = hl.register_tunable("value", BooleanFragment())
        elif kind == "float":
            value = hl.register_tunable("value", EnumFragment((2.0, 4.0)))
        else:
            value = hl.register_tunable("value", EnumFragment((2, 16)))
        for column in hl.tile(x.size(1)):
            out[row, column] = x[row, column] * value
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def derived_outer_fragment(x, kind: hl.constexpr, operation: hl.constexpr):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        if kind == "bool":
            selected = hl.register_tunable("value", BooleanFragment())
        elif kind == "float":
            selected = hl.register_tunable("value", EnumFragment((2.5, 4.5)))
        else:
            selected = hl.register_tunable("value", EnumFragment((2, 16)))
        if operation == "not":
            value = not selected
        elif operation == "divide":
            value = selected / 2
        elif operation == "compare":
            value = selected > 3
        elif operation == "mixed":
            value = selected + row.block_size + x.size(0)
        else:
            value = selected + 1
        for column in hl.tile(x.size(1)):
            out[row, column] = x[row, column] * value
    return out


@pytest.mark.parametrize(
    "kind,values,operation",
    [
        (kind, values, operation)
        for kind, values in (("int", (2, 16)), ("float", (2.5, 4.5)))
        for operation in ("add", "divide", "compare")
    ]
    + [("bool", (False, True), "not")],
)
@onlyBackends(["triton", "cute"])
def test_derived_outer_tunable_preserves_selected_scalar(kind, values, operation):
    with cpu_render_target():
        bound = cpu_bind(derived_outer_fragment, (torch.empty(17, 19), kind, operation))
        for selected in values:
            config = bound.config_spec.default_config()
            config.config["value"] = selected
            _, emitted = render_scalar_operands(bound, config)
            expected = {
                "add": selected + 1,
                "divide": selected / 2,
                "compare": selected > 3,
                "not": not selected,
            }[operation]
            assert emitted
            assert all(
                isinstance(node, ast.Constant)
                and type(node.value) is type(expected)
                and node.value == expected
                for node in emitted
            )


@helion.kernel(autotune_effort="none", static_shapes=False)
def outer_block_size_fragment(x):
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        width = hl.register_tunable("width", EnumFragment((8, 16)))
        for column in hl.tile(x.size(1), block_size=width):
            out[row, column] = x[row, column] + column.block_size
    return out


@onlyBackends(["triton", "cute"])
def test_outer_tunable_sizes_nested_tile():
    with cpu_render_target():
        bound = cpu_bind(outer_block_size_fragment, (torch.empty(17, 19),))
        for width in (8, 16):
            config = bound.config_spec.default_config()
            config.config["width"] = width
            ast.parse(bound.to_code(config))


@pytest.mark.parametrize("kind,selected", [("int", 16), ("float", 4.5)])
@onlyBackends(["triton", "cute"])
def test_derived_tunable_keeps_runtime_and_block_symbols(kind, selected):
    with cpu_render_target():
        bound = cpu_bind(derived_outer_fragment, (torch.empty(17, 19), kind, "mixed"))
        for block_size in (16, 32):
            config = bound.config_spec.default_config()
            config.config["value"] = selected
            config.config["block_sizes"][0] = block_size
            source, residuals = render_scalar_operands(bound, config)
            for expr in residuals:
                code = ast.unparse(expr)
                names = {
                    item.id for item in ast.walk(expr) if isinstance(item, ast.Name)
                }
                # Some scalars are evaluated in the host wrapper, others on
                # device. Both paths must use the current runtime tensor extent.
                constants = {}
                if "_BLOCK_SIZE_0" in names:
                    (block,) = [
                        item.value
                        for item in ast.parse(source).body
                        if isinstance(item, ast.Assign)
                        and any(
                            isinstance(target, ast.Name)
                            and target.id == "_BLOCK_SIZE_0"
                            for target in item.targets
                        )
                    ]
                    if isinstance(block, ast.Call):
                        assert ast.unparse(block.func) == "tl.constexpr"
                        (block,) = block.args
                    constants["_BLOCK_SIZE_0"] = ast.literal_eval(block)
                    assert constants["_BLOCK_SIZE_0"] == block_size
                    names.remove("_BLOCK_SIZE_0")
                assert len(names) == 1, code
                (runtime_name,) = names
                for size in (17, 29):
                    inputs = emitted_device_arguments(
                        bound, config, source, (torch.empty(size, 19), kind, "mixed")
                    )
                    actual = eval(
                        code, {**constants, runtime_name: inputs[runtime_name]}
                    )
                    expected = selected + block_size + size
                    assert type(actual) is type(expected) and actual == expected


@helion.kernel(autotune_effort="none")
def direct_global_thread_fragment(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = hl.register_tunable("value", NumThreadsFragment(HIGH_LITERAL_GLOBAL))
        out[tile] = x[tile] + value
    return out


@pytest.mark.parametrize(
    "kind,values", [("int", (2, 16)), ("float", (2.0, 4.0)), ("bool", (False, True))]
)
@onlyBackends(["triton", "cute"])
def test_outer_tunable_used_in_nested_tile_uses_selected_value(kind, values):
    with cpu_render_target():
        bound = cpu_bind(outer_registered_fragment, (torch.empty(17, 19), kind))
        for value in values:
            config = bound.config_spec.default_config()
            config.config["value"] = value
            _, emitted = render_scalar_operands(bound, config)
            assert all(
                isinstance(node, ast.Constant)
                and type(node.value) is type(value)
                and node.value == value
                for node in emitted
            )


@onlyBackends(["triton", "cute"])
def test_device_constructor_preserves_typed_global_literal():
    with cpu_render_target():
        bound = cpu_bind(direct_global_thread_fragment, (torch.empty(37),))
        fragment = bound.config_spec.user_defined_tunables["value"]
        assert type(fragment.high) is int and fragment.high == 128
        _check_selected_read(bound, (0, 64))


def _check_selected_read(bound, values):
    env = bound.env
    reads = [
        node
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is _get_tunable
    ]
    assert reads
    for node in reads:
        scalar = node.meta["val"]
        expr = scalar._sympy_()
        assert expr.free_symbols <= env.tunable_symbols
        if isinstance(scalar, torch.SymInt):
            assert env.config_value_expressions[expr].arguments == ("value",)
            assert env.tunable_default_hint(expr) == max(values[0], 2)
    for value in values:
        render_tunable_config(bound, "value", value)


@pytest.mark.parametrize(
    "kind,values",
    [
        ("enum", (2, 16)),
        ("float", (2.0, 4.0)),
        ("bool", (False, True)),
        ("threads", (0, 64)),
        ("value_custom", (3, 7)),
    ],
)
@onlyBackends(["triton", "cute"])
def test_inline_fragment_revisit_preserves_declaration_and_selected_value(kind, values):
    with cpu_render_target():
        bound = cpu_bind(grid_fragment, (torch.empty(37), kind))
        _check_selected_read(bound, values)


@pytest.mark.parametrize("use_while", [False, True])
@onlyBackends(["triton", "cute"])
def test_nested_loop_revisit_preserves_config_provenance(use_while):
    with cpu_render_target():
        bound = cpu_bind(nested_fragment, (torch.empty(37), use_while))
        _check_selected_read(bound, (2, 16))


@pytest.mark.parametrize("form", ["closed", "stored"])
@onlyBackends(["triton", "cute"])
def test_helper_literal_global_domain_survives_revisit(form):
    with cpu_render_target():
        bound = cpu_bind(global_thread_fragment, (torch.empty(37), form, 64))
        fragment = bound.config_spec.user_defined_tunables["value"]
        assert type(fragment.high) is int and fragment.high == 128
        _check_selected_read(bound, (0, 64))


@onlyBackends(["triton", "cute"])
def test_revisit_domain_resolves_derived_global_literals():
    with cpu_render_target():
        bound = cpu_bind(global_thread_fragment, (torch.empty(37), "derived", 64))
        fragment = bound.config_spec.user_defined_tunables["value"]
        assert type(fragment.high) is int and fragment.high == 192
        _check_selected_read(bound, (0, 64))


@pytest.mark.parametrize("kind", ["threads", "enum", "custom", "cached_custom"])
@onlyBackends(["triton", "cute"])
def test_shared_fragment_revisit_observes_plain_metadata_changes(kind):
    check_shared_fragment_revisit(kind, False)


@pytest.mark.parametrize("kind", ["bool", "threads"])
@pytest.mark.parametrize("unpack", [False, True])
@onlyBackends(["triton", "cute"])
def test_container_helper_revisit_preserves_selected_builtin_domain(kind, unpack):
    with (
        patch.dict(container_fragment_revisit.fn.__globals__, _EXTERNAL_STATE=[]),
        cpu_render_target(),
    ):
        bound = cpu_bind(container_fragment_revisit, (torch.empty(37), kind, unpack))
        _check_selected_read(bound, (False, True) if kind == "bool" else (0, 64))
