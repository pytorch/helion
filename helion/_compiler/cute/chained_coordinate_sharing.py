"""Live B1→B2 authority for bounded coordinate naming in a native body."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import exc
from .chained_coordinate_bounds import _excluded_names
from .chained_coordinate_bounds import _launch
from .coordinate_sdk_effects import CoordinateSDKEffects
from .coordinate_sdk_effects import coordinate_sdk_effects
from .coordinate_subtree_sharing import SubtreeSharingProof
from .coordinate_subtree_sharing import _Image
from .coordinate_subtree_sharing import _Index
from .coordinate_subtree_sharing import _Unsupported
from .coordinate_subtree_sharing import share_coordinate_subtree

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_function import Argument
    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


def _parameters(cg: GenerateAST) -> tuple[tuple[str, ...], ...]:
    df = cg.device_function
    return (
        tuple(argument.name for argument in df.arguments),
        tuple(cg._extra_params),
        tuple(df.wrapper_only_params),
    )


def _argument_fields(argument: Argument) -> tuple[object, ...]:
    from ..device_function import ConstExprArg
    from ..device_function import SymbolArgument
    from ..device_function import TensorArg
    from ..device_function import TensorDescriptorArg
    from ..device_function import TensorSizeArg
    from ..device_function import TensorStrideArg

    schemas: dict[type[Argument], set[str]] = {
        TensorArg: {"name", "fake_value", "_host_str"},
        TensorDescriptorArg: {"name", "fake_value", "_host_str", "permutation"},
        TensorSizeArg: {"name", "tensor_arg", "dim"},
        TensorStrideArg: {"name", "tensor_arg", "dim"},
        SymbolArgument: {"name", "_host_str"},
        ConstExprArg: {"name", "_host_str"},
    }
    fields = vars(argument)
    if type(argument) not in schemas or set(fields) != schemas[type(argument)]:
        raise ValueError("coordinate sharing argument schema changed")

    def field(value: object) -> object:
        if value is None or type(value) in (str, int, bool):
            return (type(value), value)
        if type(value) is list and all(type(item) is int for item in value):
            return (list, tuple(value))
        # Fake tensor / tensor-argument objects are bindings, not values to read
        # or compare with overloaded Tensor equality. The objects are retained.
        return (type(value), id(value))

    return (
        type(argument),
        tuple((key, field(value)) for key, value in sorted(fields.items())),
    )


@dataclass(frozen=True)
class ArgumentOrder:
    original: tuple[Argument, ...]
    fields: tuple[tuple[object, ...], ...]
    canonical_ids: tuple[int, ...]
    removable_names: frozenset[str]
    projected_ids: tuple[int, ...]
    projected_canonical_ids: tuple[int, ...]
    preamble: _Image
    placeholders: frozenset[str]
    dependencies: frozenset[str]

    @classmethod
    def capture(
        cls,
        arguments: Sequence[Argument],
        *,
        body: Sequence[ast.AST] = (),
        preamble: Sequence[ast.AST] = (),
        placeholder_args: frozenset[str] = frozenset(),
        dependencies: frozenset[str] = frozenset(),
    ) -> ArgumentOrder:
        from ..device_function import ConstExprArg

        original = tuple(arguments)
        if len({id(argument) for argument in original}) != len(original):
            raise ValueError("coordinate sharing duplicate argument binding")
        if len({argument.name for argument in original}) != len(original):
            raise ValueError("coordinate sharing duplicate argument name")
        fields = tuple(_argument_fields(argument) for argument in original)
        # Exact admitted native Argument classes have the original read-only
        # sort_key methods; this is the stable in-place DeviceFunction order.
        canonical = tuple(sorted(original, key=lambda argument: argument.sort_key()))
        names = _body_names((*preamble, *body))
        # The ordinary DeviceFunction DCE removes unused arguments. Admit only
        # its complete unused-constexpr projection, proved before the change.
        # A use in any AST context is retained, even if DCE regards it as dead.
        # Host text is never evaluated: self-name and other nonliteral host
        # expressions are equally unused under the original DCE predicate.
        removable = frozenset(
            argument.name
            for argument in original
            if type(argument) is ConstExprArg
            and argument.name not in names | placeholder_args | dependencies
        )
        return cls(
            original,
            fields,
            tuple(map(id, canonical)),
            removable,
            tuple(
                id(argument) for argument in original if argument.name not in removable
            ),
            tuple(
                id(argument) for argument in canonical if argument.name not in removable
            ),
            _Index(preamble).image,
            placeholder_args,
            dependencies,
        )

    def check(
        self,
        arguments: Sequence[Argument],
        *,
        body: Sequence[ast.AST] = (),
        preamble: Sequence[ast.AST] = (),
        placeholder_args: frozenset[str] = frozenset(),
    ) -> None:
        current = tuple(arguments)
        identities = tuple(map(id, current))
        original_ids = tuple(map(id, self.original))
        if identities not in (
            original_ids,
            self.canonical_ids,
            self.projected_ids,
            self.projected_canonical_ids,
        ):
            raise ValueError("coordinate sharing argument order/identity changed")
        # Retain and recheck removed objects too: mutating a dropped literal
        # cannot erase the original binding obligation.
        if (
            tuple(_argument_fields(argument) for argument in self.original)
            != self.fields
        ):
            raise ValueError("coordinate sharing argument semantic fields changed")
        if (
            self.preamble != _Index(preamble).image
            or self.placeholders != placeholder_args
        ):
            raise ValueError(
                "coordinate sharing DCE preamble/placeholder context changed"
            )
        if self.removable_names & _body_names((*preamble, *body)):
            raise ValueError("coordinate sharing removed constexpr acquired an AST use")


def _body_names(body: Sequence[ast.AST]) -> frozenset[str]:
    return frozenset(
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    )


def _image_facts(image: _Image) -> tuple[object, ...]:
    # Retain immutable field values, not a shared frozen-dataclass reference.
    return (type(image), id(image), image.nodes, image.roots)


def _argument_facts(order: ArgumentOrder) -> tuple[object, ...]:
    return (
        id(order),
        tuple(map(id, order.original)),
        order.fields,
        order.canonical_ids,
        order.removable_names,
        order.projected_ids,
        order.projected_canonical_ids,
        _image_facts(order.preamble),
        order.placeholders,
        order.dependencies,
    )


def _renames(cg: GenerateAST) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (key, tuple(group))
        for key, group in cg.device_function._variable_renames.items()
    )


def _proof_facts(proof: SubtreeSharingProof) -> tuple[object, ...]:
    return (
        _image_facts(proof.before),
        _image_facts(proof.after),
        proof.name,
        proof.expression,
        proof.scope,
        proof.anchor,
        proof.occurrences,
        proof.nodes_saved,
        proof.interval,
        proof.numeric_roots,
    )


@dataclass(frozen=True)
class CoordinateSharing:
    device: object
    plan: ChainedMatmulPlan
    config: object
    revision: object
    launch: tuple[tuple[int, int, int], tuple[int, int, int]]
    parameters: tuple[tuple[str, ...], ...]
    argument_order: ArgumentOrder
    argument_facts: tuple[object, ...]
    renames: tuple[tuple[str, tuple[str, ...]], ...]
    reserved_before: frozenset[str]
    proof: SubtreeSharingProof
    proof_facts: tuple[object, ...]
    sdk: CoordinateSDKEffects

    def check_context(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        body: Sequence[ast.AST] | None = None,
    ) -> None:
        from .native_matmul_metadata import _revision
        from .native_matmul_metadata import _snapshot

        df = cg.device_function
        if (
            type(self.proof) is not SubtreeSharingProof
            or type(self.sdk) is not CoordinateSDKEffects
            or type(self.argument_order) is not ArgumentOrder
            or self.argument_facts != _argument_facts(self.argument_order)
        ):
            raise exc.BackendUnsupported(
                "cute", "coordinate sharing context changed: receipt type"
            )
        try:
            self.argument_order.check(
                df.arguments,
                body=df.body if body is None else body,
                preamble=df.preamble,
                placeholder_args=frozenset(df.placeholder_args),
            )
        except (ValueError, _Unsupported) as error:
            raise exc.BackendUnsupported("cute", str(error)) from error
        current_parameters = _parameters(cg)
        checks = (
            ("device", self.device is df),
            ("plan", self.plan is plan),
            ("config", self.config == _snapshot(df.config.config)),
            ("revision", self.revision == _revision(plan)),
            ("launch", self.launch == _launch(cg, plan)),
            ("extra/wrapper parameters", self.parameters[1:] == current_parameters[1:]),
            ("renames", self.renames == _renames(cg)),
            (
                "original namespace reservations",
                self.reserved_before <= df.namespace._used_names,
            ),
            ("new reservation", self.proof.name in df.namespace._used_names),
            (
                "fresh rename/name exclusion",
                self.proof.name not in self.reserved_before | _excluded_names(cg),
            ),
            (
                "fresh parameter exclusion",
                self.proof.name
                not in {name for group in current_parameters for name in group},
            ),
            ("proof fields", self.proof_facts == _proof_facts(self.proof)),
        )
        changed = [name for name, valid in checks if not valid]
        if changed:
            raise exc.BackendUnsupported(
                "cute", "coordinate sharing context changed: " + ", ".join(changed)
            )
        try:
            self.sdk.check()
        except ValueError as error:
            raise exc.BackendUnsupported("cute", str(error)) from error

    def check(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        before: Sequence[ast.AST] | None = None,
    ) -> None:
        self.check_context(cg, plan)
        try:
            if before is not None:
                self.proof.check_before(before)
            self.proof.check_inverse(cg.device_function.body)
        except ValueError as error:
            raise exc.BackendUnsupported("cute", str(error)) from error

    def check_original_dependencies(self, body: Sequence[ast.AST]) -> None:
        if self.argument_order.dependencies != _body_names(body):
            raise exc.BackendUnsupported(
                "cute", "coordinate sharing original bounds dependencies changed"
            )


def canonicalize_coordinate_sharing(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    body: list[ast.stmt],
    *,
    before_bounds: Sequence[ast.AST] | None = None,
) -> tuple[list[ast.stmt], CoordinateSharing | None]:
    launch = _launch(cg, plan)
    if launch is None:
        return body, None
    sdk = coordinate_sdk_effects()
    if sdk is None:
        return body, None
    from .native_matmul_metadata import _revision
    from .native_matmul_metadata import _snapshot

    df = cg.device_function
    parameters, renames = _parameters(cg), _renames(cg)
    try:
        argument_order = ArgumentOrder.capture(
            df.arguments,
            body=body,
            preamble=df.preamble,
            placeholder_args=frozenset(df.placeholder_args),
            dependencies=_body_names(body if before_bounds is None else before_bounds),
        )
    except (ValueError, _Unsupported):
        return body, None
    reserved = frozenset(df.namespace._used_names)
    protected = reserved | frozenset(name for group in parameters for name in group)
    result, proof = share_coordinate_subtree(
        body,
        thread_block_dims=launch[0],
        block_grid_dims=launch[1],
        excluded_names=_excluded_names(cg),
        protected_names=protected,
        fresh_name=df.new_var,
        sdk_effects=sdk,
    )
    if proof is None:
        return body, None
    receipt = CoordinateSharing(
        df,
        plan,
        _snapshot(df.config.config),
        _revision(plan),
        launch,
        parameters,
        argument_order,
        _argument_facts(argument_order),
        renames,
        reserved,
        proof,
        _proof_facts(proof),
        sdk,
    )
    receipt.check_context(cg, plan, result)
    return result, receipt
