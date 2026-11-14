"""Literal collection membership uses values rather than compiler type wrappers."""

from __future__ import annotations

import ast
from contextlib import nullcontext

import pytest
import torch

import helion
from helion._compiler.type_info import LiteralType
from helion._compiler.type_info import SymBoolType
from helion._compiler.type_info import SymFloatType
from helion._compiler.type_info import SymIntType
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import scalar_additions
from helion._testing.constant_membership import approximate_comprehension_membership
from helion._testing.constant_membership import approximate_membership_alias
from helion._testing.constant_membership import collection_membership
from helion._testing.constant_membership import device_wrapped_membership
from helion._testing.constant_membership import filtered_collection_predicate
from helion._testing.constant_membership import filtered_collection_reduction
from helion._testing.constant_membership import literal_container_slice
from helion._testing.constant_membership import record_membership
from helion._testing.constant_membership import symbolic_collection_membership
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "kind,form",
    [
        ("list", "truth"),
        ("list", "not"),
        ("list", "bool"),
        ("list", "len"),
        ("tuple", "truth"),
        ("dict", "truth"),
        ("tuple", "len"),
        ("dict", "bool"),
    ],
)
def test_exact_outer_collection_ignores_filtered_child_length(kind, form):
    @helion.kernel(autotune_effort="none", static_shapes=False)
    def fn(x, limit: int, kind: hl.constexpr, form: hl.constexpr):
        filtered = [1 for value in (1, 2, 3) if value > limit]
        if kind == "list":
            outer = [filtered]
        elif kind == "tuple":
            outer = (filtered,)
        else:
            outer = {"child": filtered}
        if form == "bool" or form == "len":
            host_length = len(outer)
            host_truth = bool(outer)
        else:
            host_length = 1
            host_truth = True
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            if form == "truth":
                if outer:
                    bias = 3
                else:
                    bias = 7
            elif form == "not":
                bias = 7 if not outer else 3
            elif form == "bool":
                bias = 3 if bool(outer) else 7
            else:
                bias = len(outer) + 2
            out[tile] = x[tile] + (bias + host_length + host_truth)
        return out

    with cpu_render_target():
        x = torch.ones(7)
        for limit in (3, 0):
            bound = cpu_bind(fn, (x, limit, kind, form))
            for name, expected in (("host_length", 1), ("host_truth", True)):
                info = bound.host_function.local_types[name]
                assert isinstance(info, LiteralType) and info.value == expected
            assert scalar_additions(bound) == [5]
            ast.parse(bound.to_code(bound.config_spec.default_config()))


@pytest.mark.parametrize("form", ["tuple", "list", "dict"])
@pytest.mark.parametrize("mode", ["fast", "missing"])
def test_constant_collection_membership(form, mode):
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(collection_membership, backend="triton", static_shapes=True),
            (torch.ones(7), mode, form),
        )
        for name, expected in (
            ("member", mode == "fast"),
            ("nonmember", mode != "fast"),
        ):
            value = bound.host_function.local_types[name]
            assert isinstance(value, LiteralType)
            assert value.value is expected
        bound.to_code(bound.config_spec.default_config())


def test_symbolic_collection_membership_is_not_folded():
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(
                symbolic_collection_membership, backend="triton", static_shapes=False
            ),
            (torch.ones(7),),
        )
        assert isinstance(bound.host_function.local_types["member"], SymBoolType)
        assert isinstance(bound.host_function.local_types["nonmember"], SymBoolType)


@pytest.mark.parametrize("form", ["literal", "symbolic"])
def test_namedtuple_membership_uses_proved_values(form):
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(record_membership, static_shapes=False),
            (torch.ones(7), form),
        )
        for name, expected in (
            ("member", True),
            ("nonmember", False),
            ("field_member", False),
        ):
            value = bound.host_function.local_types[name]
            if form == "literal":
                assert isinstance(value, LiteralType) and value.value is expected
            else:
                assert isinstance(value, SymBoolType)
        if form == "literal":
            assert scalar_additions(bound) == [3, 13, 19]
        ast.parse(bound.to_code(bound.config_spec.default_config()))


def test_custom_namedtuple_membership_is_refused():
    with (
        cpu_render_target(),
        pytest.raises(
            helion.exc.TypeInferenceError,
            match="Host operations with custom record or container protocols",
        ),
    ):
        cpu_bind(
            helion.kernel(record_membership, static_shapes=False),
            (torch.ones(7), "custom"),
        )


# Each form exercises a distinct propagation edge, rather than a tuning grid:
# original list/dict, comprehension copy, slice, join, constructors, star expansion,
# nested provenance, collection arithmetic, host calls and argument expansion.
# No GPU compilation is involved.
@pytest.mark.parametrize(
    "form",
    [
        "list",
        "dict",
        "copy",
        "slice",
        "merge",
        "list_copy",
        "tuple_copy",
        "starred",
        "nested_copy",
        "dict_copy",
        "concat",
        "repeat",
        "dict_union",
        "sorted",
        "dict_kwargs",
        "call_starred",
        "nested_dict",
        "nested_unpack",
        "dict_scalar",
        "dict_string",
        "slice_key",
        "slice_key_merge",
    ],
)
def test_approximate_comprehension_membership_is_not_folded(form):
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(
                approximate_comprehension_membership,
                backend="triton",
                static_shapes=False,
            ),
            (torch.ones(7), form),
        )
    # The basic list filter removes every entry; the basic dict keeps keys
    # 2 and 3 while dropping key 1.
    # Representative contents prove neither membership nor nonmembership.
    for name in ("member", "nonmember", "kept", "excluded"):
        assert isinstance(bound.host_function.local_types[name], SymBoolType)


@pytest.mark.parametrize("form", ["list", "empty", "tuple", "nested", "reverse"])
def test_filtered_sequence_equality_uses_actual_contents(form):
    @helion.kernel(autotune_effort="none", static_shapes=False)
    def fn(x, limit: int, form: hl.constexpr):
        filtered = [value for value in (1, 2, 3) if value > limit]
        if form == "empty":
            equal = filtered == []
        elif form == "tuple":
            equal = tuple(filtered) == (2, 3)
        elif form == "nested":
            equal = (filtered,) == ([2, 3],)
        elif form == "reverse":
            equal = [2, 3] == filtered  # noqa: SIM300 -- test both operand orders
        else:
            equal = filtered == [2, 3]
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            if equal:
                out[tile] = x[tile] + 2
            else:
                out[tile] = x[tile] + 5
        return out

    with cpu_render_target():
        x = torch.ones(7)
        bound = fn.bind((x, 1, form))
        config = bound.config_spec.default_config()
        ast.parse(bound.to_code(config))
        assert isinstance(bound.host_function.local_types["equal"], SymBoolType)
        launches = []

        def observe(_kernel, _grid, *args, **_kwargs):
            launches.append(args)

        host = load_host_function(bound, config, observe)
        for limit in (1, 3, 0, 1):
            assert fn.bind((x, limit, form)) is bound
            host(x, limit, form)
            expected = [value for value in (1, 2, 3) if value > limit] == (
                [] if form == "empty" else [2, 3]
            )
            if bound.settings.backend == "pallas":
                # Pallas encodes a Boolean scalar in a one-element int buffer.
                scalars = [
                    arg.item()
                    for arg in launches[-1]
                    if isinstance(arg, torch.Tensor) and arg.numel() == 1
                ]
                assert scalars == [int(expected)]
            else:
                assert [arg for arg in launches[-1] if type(arg) is bool] == [expected]


def test_approximate_nested_item_preserves_exact_source_alias():
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(
                approximate_membership_alias, backend="triton", static_shapes=False
            ),
            (torch.ones(7),),
        )
    assert isinstance(bound.host_function.local_types["member"], SymBoolType)
    assert isinstance(bound.host_function.local_types["returned_member"], SymBoolType)
    source = bound.host_function.local_types["source_member"]
    assert isinstance(source, LiteralType) and source.value is True
    last = bound.host_function.local_types["last_member"]
    assert isinstance(last, LiteralType) and last.value is True
    assert (
        bound.host_function.local_types["choices"].origin
        is bound.host_function.local_types["original"].origin
    )


@pytest.mark.parametrize("form", ["zip", "enumerate"])
@pytest.mark.parametrize("exact", [False, True])
def test_device_wrapper_preserves_membership_uncertainty(form, exact):
    with (
        cpu_render_target(),
        nullcontext() if exact else pytest.raises(helion.exc.CantReadOnDevice),
    ):
        bound = cpu_bind(
            helion.kernel(device_wrapped_membership, backend="triton"),
            (torch.ones(7), form, exact),
        )
    if not exact:
        # A device iterator cannot consume representative contents. Computing
        # a host scalar predicate first remains supported and is tested below.
        return
    predicates = [
        node.value._type_info
        for statement in bound.host_function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "member"
            for target in node.targets
        )
    ]
    assert len(predicates) == 1
    if exact:
        assert isinstance(predicates[0], LiteralType) and predicates[0].value is True
    else:
        assert isinstance(predicates[0], SymBoolType)


def test_literal_container_slice_retains_binding_support():
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(literal_container_slice, backend="triton"),
            (torch.ones(7),),
        )
        bound.to_code(bound.config_spec.default_config())


@pytest.mark.parametrize("form", ["truth", "not", "bool", "len"])
@pytest.mark.parametrize("on_device", [False, True])
def test_filtered_collection_predicate_preserves_empty_result(form, on_device):
    with cpu_render_target():
        kernel = helion.kernel(filtered_collection_predicate, autotune_effort="none")
        x = torch.ones(7)
        bound = kernel.bind((x, 3, form, on_device))
        config = bound.config_spec.default_config()
        source = ast.parse(bound.to_code(config))
        device = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_helion_filtered_collection_predicate"
        )
        launches = []

        def observe(_kernel, _grid, *args, **_kwargs):
            launches.append(args)

        host = load_host_function(bound, config, observe)
        for limit in (3, 0, 3):
            assert kernel.bind((x, limit, form, on_device)) is bound
            host(x, limit, form, on_device)
            scalar = launches[-1][-1]
            length = len([1 for value in (1, 2, 3) if value > limit])
            if on_device:
                assert scalar == length
                if form != "len":
                    if bound.settings.backend == "pallas":
                        predicates = [
                            node.args[0]
                            for node in ast.walk(device)
                            if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "lax"
                            and node.func.attr == "cond"
                        ]
                    else:
                        predicates = [
                            node.test
                            for node in device.body
                            if isinstance(node, ast.If)
                        ]
                    assert len(predicates) == 1
                    # Resolve only the emitted scalar predicate; native loads
                    # and stores remain unexecuted in this CPU codegen test.
                    definitions = {
                        node.targets[0].id: node.value
                        for node in device.body
                        if isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                    }
                    argument = device.args.args[-1].arg
                    predicate_scalar = scalar
                    if bound.settings.backend == "pallas":
                        # Evaluate the emitted scalar-buffer load before the
                        # predicate, rather than assuming its index or value.
                        unpack = ast.Expression(definitions[argument])
                        predicate_scalar = eval(
                            compile(
                                ast.fix_missing_locations(unpack), "<unpack>", "eval"
                            ),
                            {argument: scalar},
                        ).item()

                    class ResolvePredicate(ast.NodeTransformer):
                        def __init__(self, value, assignments):
                            self.value = value
                            self.assignments = assignments

                        def visit_Name(self, node):
                            if node.id == device.args.args[-1].arg:
                                return ast.Constant(self.value)
                            if node.id in self.assignments:
                                return self.visit(
                                    ast.parse(
                                        ast.unparse(self.assignments[node.id]),
                                        mode="eval",
                                    ).body
                                )
                            return node

                    expression = ResolvePredicate(predicate_scalar, definitions).visit(
                        ast.parse(ast.unparse(predicates[0]), mode="eval")
                    )
                    actual = eval(
                        compile(
                            ast.fix_missing_locations(expression), "<predicate>", "eval"
                        )
                    )
                    assert bool(actual) == (
                        not length if form == "not" else bool(length)
                    )
            else:
                expected = length if form == "len" else 3 if length else 7
                assert scalar == expected
        assert len(launches) == 3


class _TruthyEmptyList(list):
    def __bool__(self):
        return True


def _opaque_filtered_collection(values):
    # The representative input is nonempty, unlike the actual filtered input.
    return [] if values else _TruthyEmptyList()


def _opaque_filtered_label(values):
    return "nonempty" if values else "empty"


@pytest.mark.parametrize("form", ["opaque", "opaque_merge", "generator", "label"])
def test_uncertain_collection_requires_safe_host_projection(form):
    @helion.kernel(autotune_effort="none")
    def fn(x, limit: int, form: hl.constexpr):
        filtered = [1 for value in (1, 2, 3) if value > limit]
        if form == "label":
            bias = len(_opaque_filtered_label(filtered))
        elif form == "generator":
            choices = (value for value in (1, 2, 3) if value > limit)
        else:
            choices = _opaque_filtered_collection(filtered)
            if form == "opaque_merge" and limit > 4:
                choices = []
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            if form == "label":
                out[tile] = x[tile] + bias
            elif choices:
                out[tile] = x[tile] + 3
            else:
                out[tile] = x[tile] + 7
        return out

    error = (
        helion.exc.TracedArgNotSupported
        if form == "label"
        else helion.exc.CantReadOnDevice
    )
    match = "non-numeric result" if form == "label" else None
    with cpu_render_target(), pytest.raises(error, match=match):
        cpu_bind(fn, (torch.ones(7), 3, form))


def test_host_collection_length_does_not_resolve_shadowed_len():
    @helion.kernel(autotune_effort="none")
    def fn(x, limit: int):
        len = 99  # noqa: A001 - the generated projection must not use this name
        filtered = [1 for value in (1, 2, 3) if value > limit]
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            if filtered:
                out[tile] = x[tile] + len
            else:
                out[tile] = x[tile]
        return out

    with cpu_render_target():
        x = torch.ones(7)
        bound = fn.bind((x, 3))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda _kernel, _grid, *args, **_kwargs: launches.append(args),
        )
        for limit, length in ((3, 0), (0, 3), (3, 0)):
            host(x, limit)
            assert launches[-1][-1] == length


class _EqualToBuiltinsMeta(type):
    __hash__ = type.__hash__

    def __eq__(self, other):
        return other is len or other is bool or self is other


class _EqualToBuiltins(metaclass=_EqualToBuiltinsMeta):
    def __new__(cls, values):
        return []


def test_collection_projection_recognizes_builtins_by_identity():
    @helion.kernel(autotune_effort="none")
    def fn(x, limit: int):
        filtered = [1 for value in (1, 2, 3) if value > limit]
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            if _EqualToBuiltins(filtered):
                out[tile] = x[tile] + 3
            else:
                out[tile] = x[tile] + 7
        return out

    with cpu_render_target(), pytest.raises(helion.exc.CantReadOnDevice):
        cpu_bind(fn, (torch.ones(7), 0))


@pytest.mark.parametrize("form", ["sum", "float", "any", "call", "shape"])
@onlyBackends(["triton", "cute"])
def test_filtered_host_reduction_uses_actual_scalar(form):
    kernel = helion.kernel(filtered_collection_reduction, autotune_effort="none")
    x = torch.ones(7)
    with cpu_render_target():
        bound = kernel.bind((x, 3, form))
        expected_type = {
            "sum": SymIntType,
            "shape": SymIntType,
            "float": SymFloatType,
            "any": SymBoolType,
            "call": SymIntType,
        }[form]
        assert isinstance(bound.host_function.local_types["bias"], expected_type)
        launches = []

        def observe(_kernel, _grid, *args, **_kwargs):
            launches.append(args)

        host = load_host_function(bound, bound.config_spec.default_config(), observe)
        for limit in (9, 0, 1, 9):
            assert kernel.bind((x, limit, form)) is bound
            host(x, limit, form)
            choices = x.shape if form == "shape" else (1, 2, 3)
            values = [value for value in choices if value > limit]
            expected = {
                "sum": sum(values),
                "shape": sum(values),
                "float": sum(values, 0.5),
                "any": any(values),
                "call": len(values),
            }[form]
            assert launches[-1][-1] == expected
            assert type(launches[-1][-1]) is type(expected)
        assert len(launches) == 4
