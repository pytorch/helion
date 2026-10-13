from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_coordinate_bounds import _context
from .test_cute_coordinate_subtree_sharing import SOURCE
from .test_cute_native_matmul_metadata import _stage
import helion
from helion import exc
from helion._compiler.cute import native_matmul_metadata as native
from helion._compiler.cute.chained_coordinate_bounds import (
    canonicalize_coordinate_bounds,
)
from helion._compiler.cute.chained_coordinate_sharing import (
    canonicalize_coordinate_sharing,
)
from helion._compiler.cute.coordinate_subtree_sharing import _Image
from helion._compiler.cute.coordinate_subtree_sharing import _Index
import helion.language as hl


def _sharing_context(enabled):
    cg, plan = _context()
    cg.device_function.config.config[native.KEY] = enabled
    cg.device_function.arguments = []
    cg.device_function.preamble = []
    cg.device_function.placeholder_args = set()
    cg.device_function.wrapper_only_params = []
    cg._extra_params = []
    cg.device_function.namespace = SimpleNamespace(_used_names=set())

    def new_var(prefix):
        used = cg.device_function.namespace._used_names
        name = prefix
        index = 0
        while name in used:
            index += 1
            name = f"{prefix}_{index}"
        used.add(name)
        return name

    cg.device_function.new_var = new_var
    return cg, plan


def _prepare(cg, plan):
    source = SOURCE.replace("for step", "if t < 128:\n    marker()\nfor step")
    original = ast.parse(source).body
    bounded, bounds = canonicalize_coordinate_bounds(cg, plan, original)
    assert bounds is not None
    shared, sharing = canonicalize_coordinate_sharing(
        cg, plan, bounded, before_bounds=original
    )
    assert sharing is not None
    cg.device_function.body = shared
    return source, original, bounded, bounds, sharing


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("image_name", ["preamble", "before", "after"])
@pytest.mark.parametrize("mutation", ["nodes", "roots", "replacement"])
def test_nested_image_facts_cannot_change(enabled, image_name, mutation):
    cg, plan = _sharing_context(enabled)
    with native.body_attempt(cg):
        source, original, bounded, bounds, sharing = _prepare(cg, plan)
        _stage(cg, plan, (source,))
        native.commit_body(
            cg,
            plan,
            (source,),
            original_body=original,
            coordinate_bounds=bounds,
            coordinate_sharing=sharing,
            bounded_body=bounded,
        )
    parent = sharing.argument_order if image_name == "preamble" else sharing.proof
    image = getattr(parent, image_name)
    if mutation == "replacement":
        object.__setattr__(parent, image_name, _Image(image.nodes, image.roots))
    elif mutation == "nodes":
        object.__setattr__(image, "nodes", (*image.nodes, ("extra", ())))
    else:
        object.__setattr__(image, "roots", (*image.roots, 0))
    with pytest.raises(exc.BackendUnsupported):
        native.validate_native_metadata(cg)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("location", ["preamble", "body"])
def test_changed_program_cannot_rewrite_nested_image_authority(enabled, location):
    cg, plan = _sharing_context(enabled)
    with native.body_attempt(cg):
        source, original, bounded, bounds, sharing = _prepare(cg, plan)
        _stage(cg, plan, (source,))
        native.commit_body(
            cg,
            plan,
            (source,),
            original_body=original,
            coordinate_bounds=bounds,
            coordinate_sharing=sharing,
            bounded_body=bounded,
        )
    extra = ast.parse("consume(12345)").body
    if location == "preamble":
        cg.device_function.preamble.extend(extra)
        updates = [(sharing.argument_order.preamble, cg.device_function.preamble)]
    else:
        cg.device_function.body.extend(extra)
        updates = [
            (sharing.proof.before, [*bounded, *extra]),
            (sharing.proof.after, cg.device_function.body),
        ]
    for image, body in updates:
        replacement = _Index(body).image
        object.__setattr__(image, "nodes", replacement.nodes)
        object.__setattr__(image, "roots", replacement.roots)
    with pytest.raises(exc.BackendUnsupported):
        native.validate_native_metadata(cg)


@pytest.mark.parametrize("enabled", [False, True])
def test_complete_original_bounds_then_sharing_with_optional_metadata(enabled):
    cg, plan = _sharing_context(enabled)
    with native.body_attempt(cg):
        source, original, bounded, bounds, sharing = _prepare(cg, plan)
        _stage(cg, plan, (source,))
        native.commit_body(
            cg,
            plan,
            (source,),
            original_body=original,
            coordinate_bounds=bounds,
            coordinate_sharing=sharing,
            bounded_body=bounded,
        )
    assert native.validate_native_metadata(cg) is enabled
    assert cg.cute_coordinate_sharing is sharing
    assert ast.dump(ast.Module(body=original, type_ignores=[])) == ast.dump(
        ast.parse(source)
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mutation", ["original", "intermediate", "installed"])
def test_each_body_transition_is_checked_before_optional_ledger(enabled, mutation):
    cg, plan = _sharing_context(enabled)
    with native.body_attempt(cg):
        source, original, bounded, bounds, sharing = _prepare(cg, plan)
        _stage(cg, plan, (source,))
        bodies = {
            "original": original,
            "intermediate": bounded,
            "installed": cg.device_function.body,
        }
        bodies[mutation].append(ast.Pass())
        with pytest.raises(exc.BackendUnsupported):
            native.commit_body(
                cg,
                plan,
                (source,),
                original_body=original,
                coordinate_bounds=bounds,
                coordinate_sharing=sharing,
                bounded_body=bounded,
            )
    assert cg.cute_coordinate_sharing is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "body",
        "rename",
        "argument",
        "extra",
        "wrapper",
        "reservation",
        "config",
        "proof",
    ],
)
def test_final_context_and_installed_body_obligation(enabled, mutation):
    cg, plan = _sharing_context(enabled)
    with native.body_attempt(cg):
        source, original, bounded, bounds, sharing = _prepare(cg, plan)
        _stage(cg, plan, (source,))
        native.commit_body(
            cg,
            plan,
            (source,),
            original_body=original,
            coordinate_bounds=bounds,
            coordinate_sharing=sharing,
            bounded_body=bounded,
        )
    df = cg.device_function
    if mutation == "body":
        df.body.append(ast.Pass())
    elif mutation == "rename":
        df._variable_renames[sharing.proof.name] = ["other"]
    elif mutation == "argument":
        df.arguments.append(SimpleNamespace(name=sharing.proof.name))
    elif mutation == "extra":
        cg._extra_params.append(sharing.proof.name)
    elif mutation == "wrapper":
        df.wrapper_only_params.append(sharing.proof.name)
    elif mutation == "reservation":
        df.namespace._used_names.remove(sharing.proof.name)
    elif mutation == "config":
        df.config.config[native.KEY] = not enabled
    else:
        object.__setattr__(sharing.proof, "interval", (0, 0))
    with pytest.raises(exc.BackendUnsupported):
        native.validate_native_metadata(cg)


def test_unrelated_namespace_growth_does_not_revoke_fresh_reservation():
    cg, plan = _sharing_context(False)
    with native.body_attempt(cg):
        source, original, bounded, bounds, sharing = _prepare(cg, plan)
        native.commit_body(
            cg,
            plan,
            (source,),
            original_body=original,
            coordinate_bounds=bounds,
            coordinate_sharing=sharing,
            bounded_body=bounded,
        )
    cg.device_function.new_var("later_legitimate_name")
    assert not native.validate_native_metadata(cg)


def test_unknown_launch_retains_original_body():
    cg, plan = _sharing_context(False)
    plan.threads = 0
    body = ast.parse(SOURCE).body
    after, receipt = canonicalize_coordinate_sharing(cg, plan, body)
    assert after is body and receipt is None


@pytest.mark.parametrize(
    "field", ["cute_coordinate_sharing", "cute_coordinate_sharing_commits"]
)
def test_metadata_disabled_obligation_cannot_be_removed(field):
    cg, plan = _sharing_context(False)
    source, original, bounded, bounds, sharing = _prepare(cg, plan)
    native.commit_body(
        cg,
        plan,
        (source,),
        original_body=original,
        coordinate_bounds=bounds,
        coordinate_sharing=sharing,
        bounded_body=bounded,
    )
    setattr(cg, field, None if field == "cute_coordinate_sharing" else ())
    with pytest.raises(exc.BackendUnsupported, match="removed"):
        native.validate_native_metadata(cg)


def test_absent_bounds_receipt_requires_identity_of_original_and_intermediate():
    cg, plan = _sharing_context(False)
    original = ast.parse(SOURCE).body
    bounded = [*original, ast.Pass()]
    result, sharing = canonicalize_coordinate_sharing(cg, plan, bounded)
    assert sharing is not None
    cg.device_function.body = result
    with pytest.raises(exc.BackendUnsupported, match="original-to-bounded"):
        native.commit_body(
            cg,
            plan,
            (SOURCE,),
            original_body=original,
            coordinate_sharing=sharing,
            bounded_body=bounded,
        )


def _argument_context():
    import torch

    from helion._compiler.device_function import ConstExprArg
    from helion._compiler.device_function import TensorArg

    cg, plan = _sharing_context(False)
    cg.device_function.arguments = [
        ConstExprArg("constant", "1.0"),
        TensorArg("first", torch.empty(1), "raw_first"),
        TensorArg("second", torch.empty(1), "raw_second"),
    ]
    cg.device_function.pallas_internal_scratch_name = lambda tensor: None
    return cg, plan


def test_original_sorted_args_transition_is_permitted_without_changing_bindings():
    from helion._compiler.device_function import DeviceFunction

    cg, plan = _argument_context()
    source, original, bounded, bounds, sharing = _prepare(cg, plan)
    native.commit_body(
        cg,
        plan,
        (source,),
        original_body=original,
        coordinate_bounds=bounds,
        coordinate_sharing=sharing,
        bounded_body=bounded,
    )
    before = tuple(cg.device_function.arguments)
    result = DeviceFunction.sorted_args(cg.device_function)
    assert tuple(map(id, result)) == tuple(map(id, [before[1], before[2], before[0]]))
    assert tuple(map(id, cg.device_function.arguments)) != tuple(map(id, before))
    assert not native.validate_native_metadata(cg)


@pytest.mark.parametrize(
    "mutation",
    ["other_order", "replacement", "append", "name", "host", "tensor", "type"],
)
def test_sorted_argument_admission_rejects_other_binding_changes(mutation):
    import torch

    from helion._compiler.device_function import ConstExprArg
    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.device_function import SymbolArgument
    from helion._compiler.device_function import TensorArg

    cg, plan = _argument_context()
    source, original, bounded, bounds, sharing = _prepare(cg, plan)
    native.commit_body(
        cg,
        plan,
        (source,),
        original_body=original,
        coordinate_bounds=bounds,
        coordinate_sharing=sharing,
        bounded_body=bounded,
    )
    DeviceFunction.sorted_args(cg.device_function)
    args = cg.device_function.arguments
    if mutation == "other_order":
        args[0], args[1] = args[1], args[0]
    elif mutation == "replacement":
        args[0] = TensorArg(args[0].name, args[0].fake_value, args[0]._host_str)
    elif mutation == "append":
        args.append(ConstExprArg("extra", "2.0"))
    elif mutation == "name":
        args[0].name = "changed"
    elif mutation == "host":
        args[0]._host_str = "other_raw"
    elif mutation == "tensor":
        args[0].fake_value = torch.empty(1)
    else:
        args[-1].__class__ = SymbolArgument
    with pytest.raises(exc.BackendUnsupported, match="argument"):
        native.validate_native_metadata(cg)


def test_unknown_argument_schema_declines_before_body_installation():
    cg, plan = _sharing_context(False)
    cg.device_function.arguments = [SimpleNamespace(name="foreign")]
    body = ast.parse(SOURCE).body
    after, receipt = canonicalize_coordinate_sharing(cg, plan, body)
    assert after is body and receipt is None


def _dce_context():
    from helion._compiler.device_function import ConstExprArg

    cg, plan = _sharing_context(False)
    df = cg.device_function
    df.arguments = [ConstExprArg("unused_a", "64"), ConstExprArg("unused_b", "128")]
    df.dce_vars = []
    df._tensor_args = {}
    df._tensor_descriptor_args = {}
    df._expr_args = {}
    df._tensor_properties = {}
    return cg, plan


def test_original_dce_removes_exact_unused_constexpr_projection():
    from helion._compiler.device_function import DeviceFunction

    cg, plan = _dce_context()
    source, original, bounded, bounds, sharing = _prepare(cg, plan)
    native.commit_body(
        cg,
        plan,
        (source,),
        original_body=original,
        coordinate_bounds=bounds,
        coordinate_sharing=sharing,
        bounded_body=bounded,
    )
    assert sharing.argument_order.removable_names == {"unused_a", "unused_b"}
    DeviceFunction.dead_code_elimination(cg.device_function)
    assert cg.device_function.arguments == []
    assert not native.validate_native_metadata(cg)


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "removed_fields",
        "new_body_use",
        "new_preamble_use",
        "placeholder",
        "preamble_change",
        "tensor",
        "live",
        "protected",
        "dynamic",
        "receipt_projection",
        "receipt_replacement",
        "receipt_dependencies",
    ],
)
def test_dce_projection_rejects_unproved_removal_or_changed_premise(mutation):
    import torch

    from helion._compiler.device_function import ConstExprArg
    from helion._compiler.device_function import SymbolArgument
    from helion._compiler.device_function import TensorArg

    cg, plan = _dce_context()
    df = cg.device_function
    if mutation == "tensor":
        df.arguments.append(TensorArg("raw", torch.empty(1), "raw"))
    elif mutation == "live":
        df.arguments.append(ConstExprArg("live", "32"))
        df.preamble = ast.parse("consume(live)").body
    elif mutation == "protected":
        df.arguments.append(ConstExprArg("protected", "32"))
        df.placeholder_args.add("protected")
    elif mutation == "dynamic":
        df.arguments.append(SymbolArgument("dynamic", "runtime_size"))
    source, original, bounded, bounds, sharing = _prepare(cg, plan)
    native.commit_body(
        cg,
        plan,
        (source,),
        original_body=original,
        coordinate_bounds=bounds,
        coordinate_sharing=sharing,
        bounded_body=bounded,
    )
    removed = df.arguments[0]
    df.arguments = []
    if mutation == "partial":
        df.arguments = [removed]
    elif mutation == "removed_fields":
        removed._host_str = "65"
    elif mutation == "new_body_use":
        df.body.extend(ast.parse("consume(unused_a)").body)
    elif mutation == "new_preamble_use":
        df.preamble = ast.parse("consume(unused_a)").body
    elif mutation == "placeholder":
        df.placeholder_args.add("unused_a")
    elif mutation == "preamble_change":
        df.preamble = ast.parse("consume_other()").body
    elif mutation == "receipt_projection":
        object.__setattr__(sharing.argument_order, "projected_ids", (id(removed),))
    elif mutation == "receipt_replacement":
        from dataclasses import replace

        object.__setattr__(sharing, "argument_order", replace(sharing.argument_order))
    elif mutation == "receipt_dependencies":
        object.__setattr__(sharing.argument_order, "dependencies", frozenset())
    with pytest.raises(exc.BackendUnsupported):
        native.validate_native_metadata(cg)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _finalization_kernel(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        output[row] = x[row] + 1
    return output


def test_actual_device_function_finalization_preserves_argument_authority():
    from helion._compiler.cute.chained_coordinate_sharing import ArgumentOrder
    from helion._compiler.device_function import ConstExprArg
    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.generate_ast import GenerateAST

    original_dce = DeviceFunction.dead_code_elimination
    original_call = DeviceFunction.codegen_function_call
    original_definition = DeviceFunction.codegen_function_def
    original_host_dce = GenerateAST.host_dead_code_elimination
    receipts = {}
    observed = []

    def check(df):
        receipts[id(df)].check(
            df.arguments,
            body=df.body,
            preamble=df.preamble,
            placeholder_args=frozenset(df.placeholder_args),
        )

    def dce(df):
        assert isinstance(df, DeviceFunction)
        assert id(df) not in receipts
        # Exercise the original constructor routes: one literal plus three
        # self-name host expressions with separate host size definitions.
        df.constexpr_arg_with_host_def("fixture_drop_a", 1)
        for suffix, value in (("b", 128), ("c", 64), ("d", 8)):
            name = f"fixture_drop_{suffix}"
            df.constexpr_arg(name)
            df.codegen.host_statements.extend(ast.parse(f"{name} = {value}").body)
        df.arguments.append(ConstExprArg("fixture_kept_literal", "16"))
        df.preamble.extend(
            ast.parse("fixture_kept_use = cutlass.Int32(fixture_kept_literal)").body
        )
        receipts[id(df)] = ArgumentOrder.capture(
            df.arguments,
            body=df.body,
            preamble=df.preamble,
            placeholder_args=frozenset(df.placeholder_args),
        )
        original_dce(df)
        check(df)
        names = {argument.name for argument in df.arguments}
        assert not {f"fixture_drop_{suffix}" for suffix in "abcd"} & names
        assert "fixture_kept_literal" in names
        observed.append("device_dce")

    def call(df):
        result = original_call(df)
        check(df)
        observed.append("sorted_call")
        return result

    def definition(df, **kwargs):
        result = original_definition(df, **kwargs)
        check(df)
        functions = [node for node in result if isinstance(node, ast.FunctionDef)]
        assert len(functions) == 1
        assert "fixture_kept_literal" not in {
            argument.arg for argument in functions[0].args.args
        }
        assert any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "fixture_kept_literal"
                for target in node.targets
            )
            for node in df.codegen.module_statements
        )
        assert not df._variable_renames
        observed.append("definition_and_literal_inline")
        return result

    def host_dce(cg):
        original_host_dce(cg)
        check(cg.device_function)
        observed.append("host_dce")

    with (
        _cpu_codegen(),
        patch.object(DeviceFunction, "dead_code_elimination", dce),
        patch.object(DeviceFunction, "codegen_function_call", call),
        patch.object(DeviceFunction, "codegen_function_def", definition),
        patch.object(GenerateAST, "host_dead_code_elimination", host_dce),
    ):
        bound = _finalization_kernel._bind_isolated((torch.empty(32),))
        code = bound.to_code(helion.Config(block_sizes=[32], num_warps=4))
    assert "fixture_kept_literal" in code
    assert observed == [
        "device_dce",
        "sorted_call",
        "definition_and_literal_inline",
        "host_dce",
    ]


@pytest.mark.parametrize("host_expression", ["unused_b", "runtime_size", "host_call()"])
def test_unused_constexpr_host_text_is_not_evaluated_or_required_literal(
    host_expression,
):
    from helion._compiler.device_function import DeviceFunction

    cg, plan = _dce_context()
    cg.device_function.arguments[1]._host_str = host_expression
    source, original, bounded, bounds, sharing = _prepare(cg, plan)
    native.commit_body(
        cg,
        plan,
        (source,),
        original_body=original,
        coordinate_bounds=bounds,
        coordinate_sharing=sharing,
        bounded_body=bounded,
    )
    DeviceFunction.dead_code_elimination(cg.device_function)
    assert not cg.device_function.arguments
    assert not native.validate_native_metadata(cg)


@pytest.mark.parametrize("enabled", [False, True])
def test_original_bounds_dependency_set_is_bound_before_optional_metadata(enabled):
    cg, plan = _dce_context()
    cg.device_function.config.config[native.KEY] = enabled
    original = ast.parse(SOURCE + "consume(unused_b)\n").body
    # This fixture deliberately supplies an unproved B0/B1 transition. Its
    # argument dependency must not be hidden by the proposed B1 body.
    proposed = ast.parse(SOURCE).body
    after, sharing = canonicalize_coordinate_sharing(
        cg, plan, proposed, before_bounds=original
    )
    assert sharing is not None
    assert sharing.argument_order.removable_names == {"unused_a"}
    cg.device_function.body = after
    with pytest.raises(exc.BackendUnsupported, match="original bounds dependencies"):
        native.commit_body(
            cg,
            plan,
            (SOURCE,),
            original_body=proposed,
            coordinate_sharing=sharing,
            bounded_body=proposed,
        )
