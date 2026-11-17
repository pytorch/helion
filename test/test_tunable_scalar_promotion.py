from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.aten_lowering import AtenLowering
from helion._compiler.inductor_lowering import PointwiseLowering
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.selected_tunable import mixed_selected_nested_arithmetic
from helion._testing.selected_tunable import mixed_selected_scalar_uses
from helion._testing.selected_tunable import mixed_selected_tensor_forms
from helion._testing.selected_tunable import mixed_selected_tensor_product
from helion._testing.selected_tunable import selected_boolean_forms
from helion._testing.selected_tunable import selected_tensor_cast_chain
from helion._testing.selected_tunable import selected_tensor_unary
from helion.autotuner import EnumFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "operation", ["clamp", "full_like", "device_arithmetic", "host_arithmetic"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@onlyBackends(["triton", "cute"])
def test_mixed_domain_scalar_consumers_keep_selected_types(operation, dtype):
    x = torch.tensor([-3, 5], dtype=dtype)
    with cpu_render_target():
        bound = cpu_bind(mixed_selected_scalar_uses, (x, operation))
        before = _graph_snapshot(bound)
        sources = []
        for selected in (0.5, 2, 1, 0.5):
            config = bound.config_spec.default_config()
            config.config["selected"] = selected
            observed = []
            lowering = AtenLowering if operation == "full_like" else PointwiseLowering
            codegen = lowering.codegen

            def observe(self, ctx, node, *, codegen=codegen, observed=observed):
                if node.target in (
                    torch.ops.aten.full.default,
                    torch.ops.aten.clamp_max.default,
                    torch.ops.aten.mul.Tensor,
                ):
                    observed.append(type(node.args[1].meta["val"]))
                return codegen(self, ctx, node)

            with patch.object(lowering, "codegen", observe):
                source = bound.to_code(config)
            assert observed == [
                torch.SymFloat if type(selected) is float else torch.SymInt
            ]
            ast.parse(source)
            sources.append(source)
            reference = helion.kernel(
                mixed_selected_scalar_uses.fn,
                config=config,
                ref_mode=helion.RefMode.EAGER,
            )
            expected = (
                x.clamp(max=selected)
                if operation == "clamp"
                else torch.full_like(x, selected)
                if operation == "full_like"
                else x * (selected * 2)
            )
            torch.testing.assert_close(
                reference(x, operation), expected.to(dtype), rtol=0, atol=0
            )
            assert _graph_snapshot(bound) == before
        assert sources[0] == sources[-1]


@pytest.mark.parametrize(
    "operation,dtype",
    [
        ("augassign", torch.float32),
        ("where", torch.float32),
        ("method", torch.float32),
        ("augassign", torch.int32),
        ("method", torch.int32),
        ("method_alias", torch.int32),
    ],
)
@onlyBackends(["triton", "cute"])
def test_existing_tensor_operations_accept_mixed_selected_values(operation, dtype):
    x = torch.tensor([-3, 5], dtype=dtype)
    with cpu_render_target():
        bound = cpu_bind(mixed_selected_tensor_forms, (x, operation))
        before = _graph_snapshot(bound)
        for selected in (1, 0.5, 1):
            config = bound.config_spec.default_config()
            config.config["selected"] = selected
            observed = []
            codegen = PointwiseLowering.codegen

            def observe(self, ctx, node, *, codegen=codegen, observed=observed):
                if node.target is torch.ops.aten.mul.Tensor:
                    observed.append(self.buffer.get_dtype())
                return codegen(self, ctx, node)

            with patch.object(PointwiseLowering, "codegen", observe):
                ast.parse(bound.to_code(config))
            if operation != "where":
                assert observed == [(x * selected).dtype]
            reference = helion.kernel(
                mixed_selected_tensor_forms.fn,
                config=config,
                ref_mode=helion.RefMode.EAGER,
            )
            expected = (
                torch.where(x > 0, x, selected)
                if operation == "where"
                else x * selected
            )
            torch.testing.assert_close(
                reference(x, operation), expected.to(x.dtype), rtol=0, atol=0
            )
            assert _graph_snapshot(bound) == before


@onlyBackends(["triton", "cute"])
def test_boolean_fragment_keeps_integer_flag_spelling():
    x = torch.tensor([-3.0, 5.0])
    with cpu_render_target():
        bound = cpu_bind(selected_boolean_forms, (x, "branch"))
        before = _graph_snapshot(bound)
        for flag in (False, 0, True, 1):
            config = bound.config_spec.default_config()
            config.config["flag"] = flag
            ast.parse(bound.to_code(config))
            reference = helion.kernel(
                selected_boolean_forms.fn, config=config, ref_mode=helion.RefMode.EAGER
            )
            torch.testing.assert_close(
                reference(x, "branch"), x + 2 if flag else x - 3, rtol=0, atol=0
            )
            assert _graph_snapshot(bound) == before


@onlyBackends(["triton", "cute"])
def test_boolean_integer_spelling_still_retypes_tensor_arithmetic():
    x = torch.tensor([True, False])
    with cpu_render_target():
        bound = cpu_bind(selected_boolean_forms, (x, "arithmetic"))
        before = _graph_snapshot(bound)
        for flag in (False, 0, True, 1, True):
            config = bound.config_spec.default_config()
            config.config["flag"] = flag
            observed = []
            codegen = PointwiseLowering.codegen

            def observe(self, ctx, node, *, codegen=codegen, observed=observed):
                if node.target is torch.ops.aten.mul.Tensor:
                    observed.append(self.buffer.get_dtype())
                return codegen(self, ctx, node)

            with patch.object(PointwiseLowering, "codegen", observe):
                ast.parse(bound.to_code(config))
            assert observed == [(x * flag).dtype]
            assert _graph_snapshot(bound) == before


@onlyBackends(["triton", "cute"])
def test_where_refuses_an_erased_scalar_promotion_change():
    with cpu_render_target():
        bound = cpu_bind(
            mixed_selected_tensor_forms,
            (torch.tensor([-3, 5], dtype=torch.int32), "where"),
        )
        config = bound.config_spec.default_config()
        ast.parse(bound.to_code(config))
        config.config["selected"] = 0.5
        with pytest.raises(
            helion.exc.TypeInferenceError,
            match="where changes a traced scalar conversion dtype",
        ):
            bound.to_code(config)


@onlyBackends(["triton", "cute"])
def test_boolean_integer_spelling_does_not_bypass_captured_helper_proof():
    with cpu_render_target():
        bound = cpu_bind(selected_boolean_forms, (torch.tensor([-3.0, 5.0]), "helper"))
        config = bound.config_spec.default_config()
        ast.parse(bound.to_code(config))
        config.config["flag"] = 1
        with pytest.raises(helion.exc.TypeInferenceError, match="authored Call"):
            bound.to_code(config)


@helion.kernel(autotune_effort="none")
def selected_arithmetic(x, location: hl.constexpr, choices: hl.constexpr):
    out = torch.empty_like(x)
    if location == "host":
        selected = hl.register_tunable("selected", EnumFragment(choices))
    for row in hl.tile(x.size(0)):
        if location != "host":
            selected = hl.register_tunable("selected", EnumFragment(choices))
        if location == "nested":
            for column in hl.tile(x.size(1)):
                out[row, column] = x[row, column] * selected + 1
        else:
            out[row, :] = x[row, :] * selected + 1
    return out


def _graph_snapshot(bound):
    return [
        (
            str(info.graph),
            tuple(
                (node, id(node.meta.get("val")), id(node.meta.get("lowering")))
                for node in info.graph.nodes
            ),
            tuple(getattr(info, "node_args", ())),
        )
        for info in bound.host_function.device_ir.graphs
    ]


# Registration site and selected type order exercise different FX interfaces;
# each case alternates the same two configs to check the bound IR stays intact.
@pytest.mark.parametrize("location", ["host", "device", "nested"])
@pytest.mark.parametrize("choices", [(2, 4.5), (4.5, 2**54 + 1)])
@onlyBackends(["triton", "cute"])
def test_selected_type_controls_tensor_arithmetic(location, choices):
    x = torch.tensor([[3, 5]], dtype=torch.int64)
    with cpu_render_target():
        bound = cpu_bind(selected_arithmetic, (x, location, choices))
        before = _graph_snapshot(bound)
        sources = []
        for selected in (*choices, choices[0]):
            config = bound.config_spec.default_config()
            config.config["selected"] = selected
            expected = x * selected
            observed = []
            codegen = PointwiseLowering.codegen

            def observe(self, ctx, node, *, observed=observed, codegen=codegen):
                if node.target in (
                    torch.ops.aten.mul.Tensor,
                    torch.ops.aten.add.Tensor,
                ):
                    observed.append((node.target, self.buffer.get_dtype()))
                return codegen(self, ctx, node)

            with patch.object(PointwiseLowering, "codegen", observe):
                source = bound.to_code(config)
            sources.append(source)
            assert observed
            assert all(dtype == expected.dtype for _, dtype in observed)
            assert any(target is torch.ops.aten.mul.Tensor for target, _ in observed)
            reference = helion.kernel(
                selected_arithmetic.fn, config=config, ref_mode=helion.RefMode.EAGER
            )
            torch.testing.assert_close(
                reference(x, location, choices),
                (expected + 1).to(x.dtype),
                rtol=0,
                atol=0,
            )
            assert _graph_snapshot(bound) == before
            if type(selected) is int:
                assert str(selected) in source
        assert sources[0] == sources[2]


@onlyBackends(["triton", "cute"])
def test_selected_reverse_subtraction_retypes_the_original_scalar_schema():
    x = torch.tensor([[3, 5]], dtype=torch.int32)
    choices = (2, 4.5)
    with cpu_render_target():
        bound = cpu_bind(mixed_selected_nested_arithmetic, (x, choices, True))
        before = _graph_snapshot(bound)
        sources = []
        for selected in (*choices, choices[0]):
            config = bound.config_spec.default_config()
            config.config["selected"] = selected
            observed = []
            codegen = PointwiseLowering.codegen

            def observe(self, ctx, node, *, observed=observed, codegen=codegen):
                if node.target is torch.ops.aten.sub.Tensor:
                    observed.append(
                        (self.buffer.get_dtype(), node.meta["original_aten"]._schema)
                    )
                return codegen(self, ctx, node)

            with patch.object(PointwiseLowering, "codegen", observe):
                sources.append(bound.to_code(config))
            assert observed == [
                ((selected - x).dtype, torch.ops.aten.rsub.Scalar._schema)
            ]
            reference = helion.kernel(
                mixed_selected_nested_arithmetic.fn,
                config=config,
                ref_mode=helion.RefMode.EAGER,
            )
            torch.testing.assert_close(
                reference(x, choices, True), (selected - x).to(x.dtype), rtol=0, atol=0
            )
            assert _graph_snapshot(bound) == before
        assert sources[0] == sources[2]


@helion.kernel(autotune_effort="none")
def scalar_consumer(x, operation: hl.constexpr):
    selected = hl.register_tunable("selected", EnumFragment((2, 4.5)))
    if operation == "round":
        selected = round(selected)
    elif operation == "add":
        selected = selected + 1
    elif operation == "augassign":
        selected += 1
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = x[tile] * selected
        if operation == "floor":
            out[tile] = torch.floor(value)
        elif operation == "sin":
            out[tile] = torch.sin(value)
        else:
            out[tile] = value
    return out


@pytest.mark.parametrize("operation", ["round", "sin", "floor", "augassign"])
@onlyBackends(["triton", "cute"])
def test_cross_type_refuses_uncertified_source_or_lowering(operation):
    with cpu_render_target():
        bound = cpu_bind(scalar_consumer, (torch.tensor([3]), operation))
        config = bound.config_spec.default_config()
        # Same-type behavior remains unchanged, including the scalar operation.
        old = bound.to_code(config)
        with patch(
            "helion._compiler.selected_tunable_types.specialize_selected_tunable_types"
        ):
            assert bound.to_code(config) == old
        config.config["selected"] = 4.5
        with pytest.raises(helion.exc.TypeInferenceError, match="changes scalar type"):
            bound.to_code(config)


@onlyBackends(["triton", "cute"])
def test_host_scalar_add_preserves_selected_promotion():
    x = torch.tensor([3])
    with cpu_render_target():
        bound = cpu_bind(scalar_consumer, (x, "add"))
        before = _graph_snapshot(bound)
        for selected in (2, 4.5, 2):
            config = bound.config_spec.default_config()
            config.config["selected"] = selected
            ast.parse(bound.to_code(config))
            assert _graph_snapshot(bound) == before


@onlyBackends(["triton", "cute"])
def test_nondefault_source_branch_is_checked():
    with cpu_render_target():
        safe = cpu_bind(scalar_consumer, (torch.tensor([3]), "direct"))
        unsafe = cpu_bind(scalar_consumer, (torch.tensor([3]), "round"))
        config = safe.config_spec.default_config()
        config.config["selected"] = 4.5
        ast.parse(safe.to_code(config))
        with pytest.raises(helion.exc.TypeInferenceError, match="authored Call"):
            unsafe.to_code(config)


@onlyBackends(["triton", "cute"])
def test_boolean_and_integer_selected_types_remain_distinct():
    x = torch.tensor([[True, False]])
    with cpu_render_target():
        bound = cpu_bind(selected_arithmetic, (x, "device", (True, 2)))
        before = _graph_snapshot(bound)
        for selected in (True, 2, True):
            config = bound.config_spec.default_config()
            config.config["selected"] = selected
            observed = []
            codegen = PointwiseLowering.codegen

            def observe(self, ctx, node, *, observed=observed, codegen=codegen):
                if node.target is torch.ops.aten.mul.Tensor:
                    observed.append(self.buffer.get_dtype())
                return codegen(self, ctx, node)

            with patch.object(PointwiseLowering, "codegen", observe):
                bound.to_code(config)
            assert observed == [(x * selected).dtype]
            assert _graph_snapshot(bound) == before


@helion.kernel(autotune_effort="none")
def selected_in_conditional(x, enabled: bool):
    out = torch.empty_like(x)
    selected = hl.register_tunable("selected", EnumFragment((2, 4.5)))
    for tile in hl.tile(x.numel()):
        if enabled:
            out[tile] = x[tile] * selected
        else:
            out[tile] = x[tile] + selected
    return out


@helion.kernel(autotune_effort="none")
def selected_loop_carry(x):
    out = torch.empty_like(x)
    selected = hl.register_tunable("selected", EnumFragment((2, 4.5)))
    for tile in hl.tile(x.numel()):
        acc = torch.zeros_like(x[tile])
        for _ in hl.tile(2):
            acc = acc + x[tile] * selected
        out[tile] = acc
    return out


@pytest.mark.parametrize("conditional", [False, True])
@onlyBackends(["triton", "cute"])
def test_cross_type_control_flow_is_refused(conditional):
    x = torch.tensor([3])
    with cpu_render_target():
        kernel = selected_in_conditional if conditional else selected_loop_carry
        args = (x, True) if conditional else (x,)
        bound = cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        ast.parse(bound.to_code(config))
        before = _graph_snapshot(bound)
        config.config["selected"] = 4.5
        with pytest.raises(helion.exc.TypeInferenceError, match="changes scalar type"):
            bound.to_code(config)
        assert _graph_snapshot(bound) == before


@helion.kernel(autotune_effort="none")
def selected_in_helper(x, operation: hl.constexpr):
    selected = hl.register_tunable("selected", EnumFragment((2, 4.5)))

    if operation == "round":

        def hidden():
            return round(selected)

    elif operation == "cancel":

        def hidden():
            return round(selected) - selected

    else:

        def hidden():
            return selected

    value = hidden()
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * value
    return out


@pytest.mark.parametrize("operation", ["round", "cancel", "passthrough"])
@onlyBackends(["triton", "cute"])
def test_nested_helper_cannot_certify_selected_type(operation):
    x = torch.tensor([3], dtype=torch.int32)
    with cpu_render_target():
        bound = cpu_bind(selected_in_helper, (x, operation))
        config = bound.config_spec.default_config()
        ast.parse(bound.to_code(config))
        before = _graph_snapshot(bound)
        config.config["selected"] = 4.5
        with pytest.raises(helion.exc.TypeInferenceError, match="authored Call"):
            bound.to_code(config)
        assert _graph_snapshot(bound) == before


@onlyBackends(["triton", "cute"])
def test_selected_type_registration_uses_resolved_callable():
    x = torch.arange(-323, 323, dtype=torch.int32).reshape(17, 38)[:, ::2]
    with cpu_render_target():
        for nested in (False, True):
            bound = cpu_bind(mixed_selected_tensor_product, (x, nested))
            before = _graph_snapshot(bound)
            for selected in (2, 4.5, 2):
                config = helion.Config(
                    block_sizes=[8, 16] if nested else [8], selected=selected
                )
                observed = []
                codegen = PointwiseLowering.codegen

                def observe(self, ctx, node, *, observed=observed, codegen=codegen):
                    if node.target is torch.ops.aten.mul.Tensor:
                        observed.append(self.buffer.get_dtype())
                    return codegen(self, ctx, node)

                with patch.object(PointwiseLowering, "codegen", observe):
                    ast.parse(bound.to_code(config))
                assert observed == [(x * selected).dtype]
                assert _graph_snapshot(bound) == before


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("mode", ["chain", "alias_chain"])
@onlyBackends(["triton", "cute"])
def test_selected_tensor_cast_chain_preserves_explicit_dtypes(dtype, mode):
    x = torch.tensor([-3.5, 5.5], dtype=dtype)
    with cpu_render_target():
        bound = cpu_bind(selected_tensor_cast_chain, (x, mode))
        before = _graph_snapshot(bound)
        sources = []
        for selected in (0.5, 2, 0.5):
            config = helion.Config(block_sizes=[16], selected=selected)
            observed = []
            codegen = PointwiseLowering.codegen

            def observe(self, ctx, node, *, observed=observed, codegen=codegen):
                if node.target is torch.ops.prims.convert_element_type.default:
                    observed.append(node.args[1])
                return codegen(self, ctx, node)

            with patch.object(PointwiseLowering, "codegen", observe):
                sources.append(bound.to_code(config))
            expected_casts = [
                torch.int32,
                torch.float32 if mode == "alias_chain" else dtype,
            ]
            if mode == "alias_chain" and dtype != torch.float32:
                expected_casts.append(
                    dtype
                )  # Existing conversion into the output tensor.
            assert observed == expected_casts
            assert _graph_snapshot(bound) == before
        assert sources[0] == sources[-1]


@pytest.mark.parametrize("mode", ["erased", "erased_alias", "dependent_dtype"])
@onlyBackends(["triton", "cute"])
def test_selected_tensor_cast_requires_preserved_dtype_proof(mode):
    x = torch.tensor([-3, 5], dtype=torch.int32)
    with cpu_render_target():
        bound = cpu_bind(selected_tensor_cast_chain, (x, mode))
        config = helion.Config(block_sizes=[16], selected=2)
        ast.parse(bound.to_code(config))
        before = _graph_snapshot(bound)
        config.config["selected"] = 0.5
        with pytest.raises(helion.exc.TypeInferenceError, match="changes scalar type"):
            bound.to_code(config)
        assert _graph_snapshot(bound) == before


@pytest.mark.parametrize(
    "operation,dtype",
    [
        ("abs", torch.float32),
        ("absolute", torch.int32),
        ("method_abs", torch.int32),
        ("method_absolute", torch.float32),
        ("neg", torch.float32),
        ("negative", torch.int32),
        ("method_negative", torch.float32),
        ("method_neg", torch.int32),
    ],
)
@onlyBackends(["triton", "cute"])
def test_selected_tensor_unary_preserves_values_and_promoted_dtype(operation, dtype):
    x = torch.tensor([-3, 5], dtype=dtype)
    with cpu_render_target():
        bound = cpu_bind(selected_tensor_unary, (x, operation))
        before = _graph_snapshot(bound)
        sources = []
        for selected in (0.5, 2, 0.5):
            config = helion.Config(block_sizes=[16], selected=selected)
            observed = []
            codegen = PointwiseLowering.codegen

            def observe(self, ctx, node, *, observed=observed, codegen=codegen):
                if node.target in (
                    torch.ops.aten.abs.default,
                    torch.ops.aten.neg.default,
                ):
                    observed.append(self.buffer.get_dtype())
                return codegen(self, ctx, node)

            with patch.object(PointwiseLowering, "codegen", observe):
                sources.append(bound.to_code(config))
            assert observed == [(x * selected).dtype]
            reference = helion.kernel(
                selected_tensor_unary.fn, config=config, ref_mode=helion.RefMode.EAGER
            )
            value = x * selected
            expected = (value.abs() if "abs" in operation else -value) + selected
            torch.testing.assert_close(
                reference(x, operation), expected.to(x.dtype), rtol=0, atol=0
            )
            assert _graph_snapshot(bound) == before
        assert sources[0] == sources[-1]
