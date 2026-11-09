"""Lower explicit grid synchronization to ordered CuTe kernel launches."""

from __future__ import annotations

import ast
import copy
import dataclasses
from types import ModuleType
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...autotuner.config_fragment import ConfigSpecFragment
from ...language import _tracing_ops
from ...language.constexpr import specialize
from ...language.tunable_ops import register_tunable
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..type_info import BarrierResultType
from ..type_info import BlockSizeType
from ..type_info import CallableType
from ..type_info import ConfigFragmentType
from ..type_info import LiteralType
from ..type_info import NumericType
from ..type_info import SequenceType
from ..type_info import SliceType
from ..type_info import TensorAttributeType
from ..type_info import TensorType
from ..type_info import TypeInfo
from .materialized_fission import MaterializedFissionPlan
from .materialized_fission import _shape_only
from .materialized_fission_codegen import generate_materialized_fission
from .promote_output_axis import _PURE_FRAGMENT_CONSTRUCTORS
from .promote_output_axis import _fresh_tensors
from .promote_output_axis import _integer_metadata_call
from .promote_output_axis import _pure_scalar_call
from .promote_output_axis import _read_only_host_dsl

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...runtime.config import Config
    from ..device_ir import DeviceIR


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise exc.BackendUnsupported("cute", f"ordered grid phases: {detail}")


def _pure_callable_reference(node: ast.AST) -> bool:
    """Resolve only names and existing attributes of ordinary modules."""
    if isinstance(node, ast.Name):
        return True
    if not (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ExtendedAST)
        and isinstance(node.value._type_info, LiteralType)
        and type(node.value._type_info.value) is ModuleType
        and node.attr in vars(node.value._type_info.value)
    ):
        return False
    return _pure_callable_reference(node.value)


def _pure_metadata_type(info: TypeInfo | None, *, allow_fragment: bool = False) -> bool:
    if isinstance(info, ConfigFragmentType):
        return allow_fragment
    if isinstance(info, NumericType):
        return True
    if isinstance(info, SequenceType):
        return all(_pure_metadata_type(item) for item in info.element_types)
    return isinstance(info, LiteralType) and type(info.value) in (
        int,
        float,
        bool,
        str,
        type(None),
        torch.dtype,
        torch.device,
    )


def _pure_fragment_binding(node: ExtendedAST) -> bool:
    """A name copy or an exact built-in constructor can be replayed safely."""
    return isinstance(node, ast.Name) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ExtendedAST)
        and isinstance(node.func._type_info, CallableType)
        and any(
            node.func._type_info.value is cls for cls in _PURE_FRAGMENT_CONSTRUCTORS
        )
    )


def _pure_metadata_slice(node: ast.Subscript, tensor_names: set[str]) -> bool:
    """Slice a proved scalar sequence without admitting tensor/custom indexing."""
    owner = node.value
    key = node.slice
    if not (
        isinstance(owner, ExtendedAST)
        and isinstance(sequence := owner._type_info, SequenceType)
        and sequence.python_type in (tuple, list)
        and sequence.contents_exact
        and _pure_metadata_type(sequence)
        and isinstance(key, ast.Slice)
        and isinstance(key, ExtendedAST)
        and isinstance(key._type_info, SliceType)
        and key._type_info.contents_exact
        and _pure_host_metadata(owner, tensor_names)
    ):
        return False
    # Bounds have already been typed by ordinary Python slice propagation.
    # Check their expressions too: a folded integer from an opaque call or
    # mutable module attribute is not replay-safe metadata.
    return all(
        bound is None
        or (
            isinstance(bound, ExtendedAST)
            and (
                isinstance(info := bound._type_info, LiteralType)
                and type(info.value) in (int, bool, type(None))
                or isinstance(info, NumericType)
                and info.python_type in (int, bool)
            )
            and _pure_host_metadata(bound, tensor_names)
        )
        for bound in (key.lower, key.upper, key.step)
    )


def _pure_host_metadata(
    node: ast.AST, tensor_names: set[str], *, allow_fragment: bool = False
) -> bool:
    """Prove a host prelude value is pure scalar or shape metadata.

    Extends the shape-only grammar with the integer-metadata calls the
    freshness proof admits (``hl.register_tunable`` with its fragment,
    ``helion.cdiv`` / power-of-two helpers, ``min`` / ``max``). Each ordered
    stage re-executes the host prelude, and these derive the same value from
    the same config and shapes in every stage. The value itself must be a
    scalar (symbolic or literal), never a tensor. Numeric scaling and
    comparisons have the same replay contract as integer shape arithmetic.
    """
    if isinstance(node, ExtendedAST) and not _pure_metadata_type(
        node._type_info, allow_fragment=allow_fragment
    ):
        # User objects can overload arithmetic, indexing and truth tests.
        # A traced scalar result does not prove evaluating its inputs is pure.
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Not)
    ):
        return _pure_host_metadata(node.operand, tensor_names)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_pure_host_metadata(item, tensor_names) for item in node.elts)
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            return _pure_metadata_slice(node, tensor_names)
        return _pure_host_metadata(node.value, tensor_names) and _pure_host_metadata(
            node.slice, tensor_names
        )
    if isinstance(node, ast.BinOp):
        return (
            isinstance(
                node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
            )
            and _pure_host_metadata(node.left, tensor_names)
            and _pure_host_metadata(node.right, tensor_names)
        )
    if isinstance(node, ast.Compare):
        return all(
            _pure_host_metadata(value, tensor_names)
            for value in (node.left, *node.comparators)
        )
    if isinstance(node, ast.BoolOp):
        return all(_pure_host_metadata(value, tensor_names) for value in node.values)
    if isinstance(node, ast.Attribute) and _pure_callable_reference(node):
        # Canonical Torch dtype names are compiler constants. Arbitrary
        # module attributes, even scalar literals, are mutable host state;
        # dispatch does not guard their values before reusing a folded branch.
        assert isinstance(node, ExtendedAST)
        assert isinstance(node.value, ExtendedAST)
        info = node._type_info
        owner = node.value._type_info
        return (
            isinstance(info, LiteralType)
            and type(info.value) is torch.dtype
            and isinstance(owner, LiteralType)
            and owner.value is torch
        )
    # Compound nodes recurse above before the shape grammar can accept them;
    # this keeps overloaded operations on opaque literal operands out too.
    if _shape_only(node, tensor_names):
        return not isinstance(node, ast.Call) or all(
            _pure_host_metadata(arg, tensor_names) for arg in node.args
        )
    if not (
        isinstance(node, ast.Call)
        and isinstance(node, ExtendedAST)
        and isinstance(node.func, ExtendedAST)
        and isinstance(node.func._type_info, CallableType)
        and _pure_callable_reference(node.func)
        and (
            _integer_metadata_call(node, node.func._type_info)
            or _pure_scalar_call(node, node.func._type_info)
        )
    ):
        return False
    result = node._type_info
    callee = node.func._type_info.value
    if (
        isinstance(callee, type)
        and issubclass(callee, ConfigSpecFragment)
        and not (allow_fragment and isinstance(result, ConfigFragmentType))
    ):
        return False
    if isinstance(result, TensorType) or not isinstance(
        result, (NumericType, LiteralType)
    ):
        return False
    return all(
        _pure_host_metadata(
            value,
            tensor_names,
            allow_fragment=callee is register_tunable and index == 1,
        )
        for index, value in enumerate(node.args)
    ) and all(
        _pure_host_metadata(keyword.value, tensor_names) for keyword in node.keywords
    )


def _constant_prelude(
    statements: list[ast.stmt], tensor_names: set[str]
) -> list[ast.stmt]:
    """Fold only typed, effect-free constexpr branches before freshness analysis.

    The selected statements retain their traced types and origins. In
    particular this does not evaluate a Python helper again or speculate an
    untraced branch; shape/config guards already specialize the literal test.
    """
    result: list[ast.stmt] = []
    known_tensors = set(tensor_names)

    def fold(body: list[ast.stmt]) -> None:
        for statement in body:
            if isinstance(statement, ast.If):
                _require(
                    isinstance(statement.test, ExtendedAST)
                    and isinstance(statement.test._type_info, LiteralType)
                    and _pure_host_metadata(statement.test, known_tensors),
                    "host condition must be a pure constexpr boolean",
                )
                assert isinstance(statement.test, ExtendedAST)
                assert isinstance(statement.test._type_info, LiteralType)
                # The typed metadata proof excludes overloaded truth tests.
                branch = (
                    statement.body
                    if statement.test._type_info.value
                    else statement.orelse
                )
                fold(branch)
                continue
            result.append(statement)
            if (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ExtendedAST)
                and isinstance(statement.value._type_info, TensorType)
            ):
                known_tensors.update(
                    target.id
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                )

    fold(statements)
    return result


def _capturable_host_view(node: ast.AST, tensor_names: set[str]) -> bool:
    """Prove a host view whose exact object can be captured once.

    This does not make an input fresh or read-only. Both a writable parameter
    and a fresh output retain their original storage owner, and the existing
    explicit phase barriers still order every device access. The stage wrapper
    receives the same view object created by the outer host, not a new buffer.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node, ExtendedAST)
        and isinstance(node._type_info, TensorType)
        and isinstance(node.func, ExtendedAST)
    ):
        return False
    callee = node.func._type_info
    if (
        isinstance(callee, TensorAttributeType)
        and callee.attr() in ("reshape", "view", "flatten", "as_strided")
        and isinstance(node.func, ast.Attribute)
    ):
        owner = node.func.value
        arguments = node.args
    elif (
        isinstance(callee, CallableType)
        and any(callee.value is fn for fn in (torch.reshape, torch.flatten))
        and node.args
    ):
        owner, *arguments = node.args
    else:
        return False
    if not (
        isinstance(owner, ast.Name)
        and owner.id in tensor_names
        and isinstance(owner, ExtendedAST)
        and isinstance(owner._type_info, TensorType)
    ):
        return False

    if any(keyword.arg in (None, "out") for keyword in node.keywords) or not all(
        _pure_host_metadata(value, tensor_names)
        for value in (*arguments, *(k.value for k in node.keywords))
    ):
        return False
    base = owner._type_info.proxy()
    view = node._type_info.proxy()
    env = CompileEnvironment.current()
    if not (
        base.dtype == view.dtype
        and base.device == view.device
        and base.untyped_storage()._cdata == view.untyped_storage()._cdata
    ):
        return False
    if isinstance(callee, TensorAttributeType) and callee.attr() == "as_strided":
        # as_strided has an exact alias contract even when its extent, strides
        # and offset differ from the input. Torch checks the storage bounds
        # when the outer host creates this very object, which is then passed
        # unchanged to every stage. Never classify it as a fresh allocation.
        return True
    if not (
        env.known_equal(base.storage_offset(), view.storage_offset())
        and env.known_equal(base.numel(), view.numel())
    ):
        return False
    for tensor in (base, view):
        stride: int | torch.SymInt = 1
        for size, actual_stride in reversed(
            tuple(zip(tensor.shape, tensor.stride(), strict=True))
        ):
            if not env.known_equal(actual_stride, stride):
                return False
            stride = stride * size
    return True


def _specialized_argument_identity(statement: ast.Assign, host: HostFunction) -> bool:
    """Retain an integer argument's guarded specialization, not a new value.

    The original type propagation installs specialization guards, and ordinary
    codegen emits their constant value in both the wrapper and each stage.
    No arithmetic, different argument, opaque callable or storage rebind is an
    identity under this proof.
    """
    return (
        len(statement.targets) == 1
        and isinstance(target := statement.targets[0], ast.Name)
        and target.id in host.params.arguments
        and type(host.params.arguments[target.id]) in (int, torch.SymInt)
        and isinstance(call := statement.value, ast.Call)
        and len(call.args) == 1
        and not call.keywords
        and isinstance(argument := call.args[0], ast.Name)
        and argument.id == target.id
        and isinstance(call.func, ExtendedAST)
        and isinstance(callee := call.func._type_info, CallableType)
        and callee.value is specialize
        and _pure_callable_reference(call.func)
        and _read_only_host_dsl(call, callee)
        and isinstance(call, ExtendedAST)
        and isinstance(call._type_info, NumericType)
        and type(call._type_info.proxy()) in (int, torch.SymInt)
    )


def _ordered_phase_bundle(
    host: HostFunction, ir: DeviceIR
) -> tuple[HostFunction, MaterializedFissionPlan]:
    """Prove the explicit barrier phases can launch as ordered single roots."""
    _require(not ir.implicit_dependency_starts, "implicit dependency phases")
    _require(
        all(
            node.target is not _tracing_ops._if
            for graph in ir.graphs
            for node in graph.graph.nodes
        ),
        "runtime conditionals inside grid phases",
    )
    _require(
        all(
            node.target is not _tracing_ops._while_loop
            for graph in ir.graphs
            for node in graph.graph.nodes
        ),
        "runtime while loops inside grid phases",
    )
    _require(
        all(len(phase.roots) == 1 for phase in ir.phases),
        "each phase must contain one top-level grid loop",
    )
    roots = [phase.root_nodes[0] for phase in ir.phases]
    return _capture_phase_bundle(host, ir, roots)


def _capture_phase_bundle(
    host: HostFunction, ir: DeviceIR, roots: list[ast.For]
) -> tuple[HostFunction, MaterializedFissionPlan]:
    """Capture once after a caller has proved its root/phase ordering."""
    _require(
        len(roots) == len(ir.root_ids)
        and all(
            isinstance(root, ast.For)
            and isinstance(root, ExtendedAST)
            and root._loop_type is LoopType.GRID
            for root in roots
        ),
        "phase/root ownership mismatch",
    )
    first = host.body.index(roots[0])
    last = host.body.index(roots[-1])
    for statement in host.body[first : last + 1]:
        if statement in roots:
            continue
        _require(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ExtendedAST)
            and isinstance(statement.value._type_info, BarrierResultType),
            "host effects between grid phases",
        )
    tail = host.body[last + 1 :]
    _require(
        len(tail) == 1 and isinstance(tail[0], ast.Return), "post-grid host effects"
    )

    tensor_names = {
        name
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }
    prelude = _constant_prelude(host.body[:first], tensor_names)
    # Remove only the barriers validated above before proving that captured
    # allocations are fresh. The general freshness proof deliberately rejects
    # unrecognized host calls, including barriers.
    definition = dataclasses.replace(
        host.definition,
        body=[*prelude, *roots, *tail],
    )
    bundle = HostFunction(definition, host.location)
    bundle.compiler_state = host.compiler_state
    bundle.local_types = host.local_types
    bundle.device_ir = copy.copy(ir)
    bundle.device_ir.host_function = bundle

    bound_names = set(host.params.arguments)
    specialized_arguments: set[str] = set()
    fresh = _fresh_tensors(bundle, integer_metadata_calls=True)
    captures = []
    for statement in prelude:
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            continue
        if isinstance(statement, ast.Assert):
            _require(
                _pure_host_metadata(statement.test, tensor_names)
                and (
                    statement.msg is None
                    or _pure_host_metadata(statement.msg, tensor_names)
                ),
                "host assertion must contain only pure metadata",
            )
            continue
        _require(isinstance(statement, ast.Assign), "unsupported host prelude")
        assert isinstance(statement, ast.Assign)
        targets = [
            node.id
            for target in statement.targets
            for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        ]
        specializes_argument = _specialized_argument_identity(
            statement, host
        ) and not specialized_arguments.intersection(targets)
        _require(
            bool(targets)
            and len(set(targets)) == len(targets)
            and (not bound_names.intersection(targets) or specializes_argument),
            "rebound host metadata or allocation",
        )
        if specializes_argument:
            specialized_arguments.update(targets)
        bound_names.update(targets)
        assert isinstance(statement.value, ExtendedAST)
        value_type = statement.value._type_info
        if isinstance(value_type, TensorType):
            _require(
                len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and (
                    value_type.proxy() in fresh
                    or _capturable_host_view(statement.value, tensor_names)
                ),
                "host tensor prelude must allocate fresh storage or a proved contiguous view or explicit storage view",
            )
            captures.extend(targets)
            tensor_names.update(targets)
        else:
            _require(
                specializes_argument
                or isinstance(value_type, BlockSizeType)
                or _pure_host_metadata(
                    statement.value,
                    tensor_names,
                    # A fragment binding is inert metadata, just like the
                    # inline fragment argument to register_tunable. Arithmetic
                    # and truth tests on fragments remain outside the grammar.
                    allow_fragment=isinstance(value_type, ConfigFragmentType)
                    and _pure_fragment_binding(statement.value),
                ),
                "host metadata expression is not shape-only",
            )

    plan = MaterializedFissionPlan(
        root_index=len(prelude),
        source_root_key=ast.dump(roots[0]),
        materialized_names=tuple(captures),
        region_count=len(roots),
    )
    return bundle, plan


def ordered_phases_supported(host: HostFunction, ir: DeviceIR) -> bool:
    """Whether ``pid_type="flat"`` can lower these barrier phases."""
    try:
        _ordered_phase_bundle(host, ir)
    except exc.BackendUnsupported:
        return False
    return True


def generate_ordered_phases(
    host: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    *,
    store_transform: Callable[..., ast.AST] | None = None,
    load_transform: Callable[..., ast.AST] | None = None,
    extra_params: list[str] | None = None,
) -> ast.Module:
    """Bundle one complete root per explicit barrier phase.

    Separate launches implement the existing grid-wide ordering contract.
    The ordinary compiler has already checked each phase's memory dependencies;
    this path does not infer additional synchronization for unmarked loops.
    """
    bundle, plan = _ordered_phase_bundle(host, host.device_ir)
    return generate_materialized_fission(
        bundle,
        config,
        emit_repro_caller,
        plan,
        store_transform=store_transform,
        load_transform=load_transform,
        extra_params=extra_params,
    )
