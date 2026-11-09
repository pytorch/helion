from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from typing import cast
from typing import overload

import sympy
import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch.fx.experimental import proxy_tensor
from torch.utils._pytree import tree_map

from .. import exc
from .._compiler.ast_extension import ExtendedAST
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import AutoSize
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import ConfigValueExpression
from .._compiler.compile_environment import _symint_expr
from .._compiler.type_info import NumericType
from .._compiler.type_info import TileIndexType
from .._compiler.type_info import TypeInfo
from .._compiler.type_info import _to_proxy
from ..autotuner.config_fragment import _FRAGMENT_CHANGED_MESSAGE
from ..autotuner.config_fragment import BaseIntegerFragment
from ..autotuner.config_fragment import BooleanFragment
from ..autotuner.config_fragment import ConfigSpecFragment
from ..autotuner.config_fragment import NumThreadsFragment
from ..autotuner.config_fragment import _immutable_fragment_key
from ..autotuner.config_fragment import _plain_builtin_fragment_fields
from ..autotuner.config_fragment import _plain_builtin_fragment_key
from ..autotuner.config_fragment import assert_integer_power_of_two
from ..autotuner.config_spec import valid_keys
from ..exc import NotInsideKernel
from . import _decorators
from .loops import _normalize_begin_end

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.variable_origin import Origin

__all__ = [
    "register_block_size",
    "register_tunable",
]


@_decorators.api(is_device_only=False, cache_type=True, tiles_as_sizes=True)
def register_block_size(min_or_max: int, max_or_none: int | None = None, /) -> int:
    """
    Explicitly register a block size that should be autotuned and can be used for
    allocations and inside hl.tile(..., block_size=...).

    This is useful if you have two loops where you want them to share a block size,
    or if you need to allocate a kernel tensor before the hl.tile() loop.

    The signature can one of:
        hl.register_block_size(max)
        hl.register_block_size(min, max)

    Where min and max are integers that control the range of block_sizes searched by
    the autotuner.  Both may be symbolic shapes, in which case their size hints are
    used to bound the search.
    """
    raise exc.NotInsideKernel


@_decorators.ref(register_block_size)
def _(min_or_max: int, max_or_none: int | None = None, /) -> int:
    # In ref mode, always return the maximum value (full dimension size)
    if max_or_none is None:
        return min_or_max
    return max_or_none


@_decorators.type_propagation(register_block_size)
def _(
    min_or_max: TypeInfo, max_or_none: TypeInfo | None = None, /, *, origin: Origin
) -> TypeInfo:
    from .._compiler.type_info import BlockSizeType

    min_type, max_type = _normalize_begin_end(min_or_max, max_or_none, origin=origin)
    min_proxy = _to_proxy(min_type)
    max_proxy = _to_proxy(max_type)
    if not isinstance(max_proxy, (int, torch.SymInt)):
        raise exc.IncorrectTileUsage(
            f"expected max to be an integer or size, got {max_proxy!s}"
        )
    if not isinstance(min_proxy, (int, torch.SymInt)):
        raise exc.IncorrectTileUsage(
            f"expected min to be an integer or size, got {min_proxy!s}"
        )
    env = CompileEnvironment.current()
    # Host scalars (globals, int arguments, `*args` unpacked from a tuple) are
    # faked into SymInts during type propagation, so min may arrive symbolic even
    # when the user wrote a constant.  Both bounds only shape the autotuner search
    # space, so resolve them to hints the same way.
    min_proxy = env.size_hint(min_proxy)
    existing_block_id = (
        env.get_block_id(max_proxy) if isinstance(max_proxy, torch.SymInt) else None
    )
    if existing_block_id is None:
        result = TileIndexType.allocate(AutoSize(), origin)
        block_id = result.block_id
    else:
        block_id = existing_block_id
    loop_spec = env.config_spec.block_sizes.block_id_lookup(block_id)
    loop_spec.min_size = assert_integer_power_of_two(max(1, min_proxy))
    loop_spec.max_size = next_power_of_2(env.size_hint(max_proxy))
    loop_spec.declared_max_size = loop_spec.max_size
    return BlockSizeType(origin, env.block_sizes[block_id].var, block_id)


def _block_id_from_state(state: CodegenState) -> int:
    """Extract the block_id from the current state for nodes hl.register_block_size."""
    from .._compiler.type_info import BlockSizeType
    from .._compiler.type_info import SymIntType

    env = CompileEnvironment.current()
    if state.fx_node is not None:
        val = state.fx_node.meta["val"]
        if isinstance(val, BlockSizeType):
            return val.block_id
        assert isinstance(val, SymIntType)
        block_id = env.get_block_id(val.value)
        assert block_id is not None
        return block_id
    current_node = ExtendedAST.current()[-1]
    type_info = current_node._type_info
    if isinstance(type_info, BlockSizeType):
        return type_info.block_id
    assert isinstance(type_info, SymIntType)
    block_id = env.get_block_id(type_info.value)
    assert block_id is not None
    return block_id


@_decorators.codegen(register_block_size, "common")
def _(state: CodegenState) -> ast.AST:
    env = CompileEnvironment.current()
    block_size = env.config_spec.block_sizes.config_get(
        state.config.block_sizes, _block_id_from_state(state)
    )
    assert block_size is not None
    return expr_from_string(constant_repr(block_size))


@overload
def register_tunable(
    name: str, fragment: BaseIntegerFragment | NumThreadsFragment
) -> int: ...


@overload
def register_tunable(name: str, fragment: BooleanFragment) -> bool: ...


@overload
def register_tunable(name: str, fragment: ConfigSpecFragment) -> int | float | bool: ...


@_decorators.api(is_device_only=False)
def register_tunable(name: str, fragment: ConfigSpecFragment) -> int | float | bool:
    """
    Register a tunable parameter for autotuning.

    The fragment defines the search space and default value. Declare concrete
    domains using literal globals, ``hl.constexpr`` arguments, or
    ``hl.specialize`` for integer shapes and arguments. Arithmetic on literal
    globals is supported in kernel code; pass concrete arguments to Python
    helpers rather than computing domains from runtime values or control flow.

    Literal globals used to declare a domain are captured when a kernel is first
    bound. Reusing that binding keeps its original domain even if those globals
    later change. Use a new kernel to capture a different domain. Ordinary reads
    of the same globals in device computations remain runtime values and are
    read on each call.

    Symbolic domains are deprecated and emit a ``FutureWarning`` at the kernel
    source location, at most once per binding. Legacy fixed configurations remain
    supported when the fragment's default has a supported scalar type, but
    autotuning may fail when it evaluates a symbolic domain. Shape hints do not
    define domain bounds.

    Do not mutate a registered domain during compilation or reuse of a bound
    kernel. Ordinary built-in fragments are snapshotted at registration; custom
    fragments, subclasses, and built-ins with extra state retain their original
    objects and equality. Repeated declarations must agree, including the types
    of scalar choices. Built-in Boolean and thread-count fragments allow repeated
    construction with the same domain. For custom fragments, reuse one instance
    or define value equality.

    Integer and thread-count fragments return ``int`` and Boolean fragments
    return ``bool``. Enum and custom fragments return ``int | float | bool``;
    callers that previously relied on their ``int`` annotation may need to narrow
    the result's type.

    For module-level Python helpers, return the fragment from the helper and call
    ``hl.register_tunable`` directly in the kernel. Registration inside these
    helpers is not supported during compilation.

    A selection with a different scalar type from the default is supported for
    direct tensor ``+``, ``-``, ``*`` and ``/`` operands in device loops, including
    captured operands in nested loops, arithmetic tensor methods and tensor
    augmented assignments. ``torch.where`` is supported when its traced scalar
    conversion retains the selected value's promoted dtype. Config-local typing
    preserves integer selections without promoting the whole domain to floating point.
    Other cross-type uses, including scalar expressions such as ``round(value)``,
    local helpers capturing a selected value, type-changing loop carries and
    unsupported device control flow, raise
    ``TypeInferenceError`` instead of reusing the default's tensor casts. Use a
    domain whose choices have the same scalar type for these forms. Same-type
    selections retain the existing lowering behavior. Integer ``0``/``1`` Boolean
    fragment spellings also retain direct ``if flag:`` conditions; this exception
    does not bypass typing for Boolean tensor arithmetic.

    Args:
        name: The key for the tunable parameter in the Config().
        fragment: A ConfigSpecFragment that defines the search space (e.g., PowerOfTwoFragment)

    Returns:
        int | float | bool: The value assigned to this tunable parameter in the current configuration.

    """
    raise NotInsideKernel


def _tunable_name_conflicts(name: str) -> bool:
    """Whether ``name`` (or its plural) is a config key some registered backend
    owns; read live, so a backend registered after import counts."""
    keys = valid_keys()
    return name in keys or f"{name}s" in keys


def _concrete_fragment_metadata(value: object, env: CompileEnvironment) -> object:
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        literal = env.resolve_fragment_global(value)
        if literal is None:
            env.warn_symbolic_fragment_domain()
            return value
        return literal
    return value


def _snapshot_builtin_fragment(
    fragment: ConfigSpecFragment, env: CompileEnvironment
) -> ConfigSpecFragment:
    """Copy only complete known domains; preserve custom object semantics."""
    fields = _plain_builtin_fragment_fields(fragment)
    if fields is None:
        return fragment
    values = dict(fields)
    memo: dict[int, object] = {}

    def resolve(value: object) -> object:
        concrete = _concrete_fragment_metadata(value, env)
        if concrete is not value:
            memo[id(value)] = concrete
        return concrete

    # Only ordinary containers belong to these fields; never flatten a custom
    # pytree object or call a user copy protocol to infer a domain.
    def is_leaf(value: object) -> bool:
        return type(value) not in (list, tuple, dict)

    normalized = tree_map(resolve, values, is_leaf=is_leaf)
    if _immutable_fragment_key(normalized) is None:
        return fragment
    captured = object.__new__(type(fragment))
    # tree_map validates the resolved fields but rebuilds shared containers
    # separately. deepcopy preserves those aliases, including choices shared
    # with search_choices; memo replaces symbols before it could visit ShapeEnv.
    vars(captured).update(copy.deepcopy(values, memo))
    return captured


def _same_tunable_fragment_domain(
    left: ConfigSpecFragment, right: ConfigSpecFragment
) -> bool:
    # Dataclass equality conflates int/float/bool choices and signed zeros.
    # Compare complete typed domains even when their defaults have equal types.
    left_key = _plain_builtin_fragment_key(left)
    right_key = _plain_builtin_fragment_key(right)
    if (
        left_key is not None
        and right_key is not None
        and (type(left) is not type(right) or left_key != right_key)
    ):
        return False
    # These exact built-ins have identity equality but complete plain domains.
    # Additional instance state and subclasses retain their existing equality.
    if left != right:
        if type(left) is not type(right):
            return False
        left_key = _plain_builtin_fragment_key(left, for_identity_equality=True)
        return left_key is not None and left_key == _plain_builtin_fragment_key(
            right, for_identity_equality=True
        )
    return True


@_decorators.type_propagation(register_tunable)
def _register_tunable_type(
    name: TypeInfo, fragment: TypeInfo, *, origin: Origin
) -> TypeInfo:
    # Register the parameter and return its symbolic selected value.
    env = CompileEnvironment.current()

    try:
        fragment_val = fragment.as_literal()
        name_val = name.as_literal()
    except NotImplementedError:
        fragment_val = None
        name_val = None
    if not (isinstance(name_val, str) and isinstance(fragment_val, ConfigSpecFragment)):
        raise exc.RegisterTunableArgTypes(name, fragment)

    if _tunable_name_conflicts(name_val):
        raise exc.TunableNameConflict(name_val)

    captured = _snapshot_builtin_fragment(fragment_val, env)
    identity = id(fragment_val)
    metadata = (
        _plain_builtin_fragment_key(captured) if captured is not fragment_val else None
    )
    previous = env.tunable_fragment_snapshots.get(identity)
    if previous is not None and _plain_builtin_fragment_key(previous[1]) != metadata:
        raise exc.TypeInferenceError(_FRAGMENT_CHANGED_MESSAGE)
    if captured is not fragment_val:
        assert metadata is not None
        snapshot = env.tunable_fragment_snapshots.setdefault(
            identity, (fragment_val, captured)
        )
        fragment_val = snapshot[1]
    if (
        name_val in env.config_spec.user_defined_tunables
        and not _same_tunable_fragment_domain(
            env.config_spec.user_defined_tunables[name_val], fragment_val
        )
    ):
        raise exc.TunableNameConflict(name_val)

    env.config_spec.user_defined_tunables[name_val] = fragment_val

    python_type = type(fragment_val.default())
    if python_type is not int and python_type is not float and python_type is not bool:
        raise exc.TunableTypeNotSupported(python_type)
    symbolic_type = {int: torch.SymInt, float: torch.SymFloat, bool: torch.SymBool}[
        python_type
    ]
    for (registered_name, _), value in env.tunable_values.items():
        if registered_name == name_val and type(value) is not symbolic_type:
            raise exc.TunableNameConflict(name_val)
    site = (name_val, tuple(ExtendedAST.current()))
    if site in env.tunable_values:
        return NumericType.subtype(python_type)(origin, env.tunable_values[site])
    result = NumericType.subtype(python_type).new_unbacked(origin)
    env.tunable_values[site] = result.value
    # Codegen uses this provenance to annotate device arguments as constexpr.
    for symbol in result.value._sympy_().free_symbols:
        assert isinstance(symbol, sympy.Symbol)
        env.tunable_symbols.add(symbol)
    if isinstance(result.value, torch.SymInt):
        expr = _symint_expr(result.value)
        if expr is not None:
            env.config_value_expressions[expr] = ConfigValueExpression(
                "config", (name_val,)
            )
    return result


@_decorators.register_fake(register_tunable)
def _(name: str, fragment: ConfigSpecFragment) -> int | float | bool:
    raise exc.TypeInferenceError(
        "hl.register_tunable() is not supported inside a Python helper function "
        "during compilation. Return the fragment from the helper and call "
        "hl.register_tunable() directly in the kernel."
    )


@_decorators.api()
def _get_tunable(name: str) -> int | float | bool:
    """Read a registered config scalar in device IR without its fragment metadata."""
    raise AssertionError("this should never be called")


@_decorators.register_to_device_ir(register_tunable)
def _(
    tracer: proxy_tensor.PythonKeyTracer, name: str, fragment: ConfigSpecFragment
) -> torch.SymInt | torch.SymFloat | torch.SymBool:
    # Type propagation already checked the declaration, snapshotted its fragment,
    # and assigned the symbolic result. Keep that value, including Boolean type
    # and config provenance; a default would incorrectly specialize this use.
    result = ExtendedAST.current()[-1]._type_info
    assert isinstance(result, NumericType)
    assert name in CompileEnvironment.current().config_spec.user_defined_tunables
    proxy = tracer.create_proxy("call_function", _get_tunable, (name,), {})
    proxy_tensor.track_tensor_tree(result.value, proxy, constant=None, tracer=tracer)
    return result.value


def _codegen_tunable(state: CodegenState) -> ast.AST:
    name = state.proxy_arg(0)
    assert isinstance(name, str)
    config_value = state.config[name]
    assert isinstance(config_value, (int, float, bool))
    if not state.codegen.on_device:
        # Host registration executes outside the backend's IR context.
        return expr_from_string(constant_repr(config_value))
    return expr_from_string(
        CompileEnvironment.current().backend.scalar_literal_expr(config_value)
    )


@_decorators.codegen(register_tunable, "common")
def _register_tunable_codegen(state: CodegenState) -> ast.AST:
    return _codegen_tunable(state)


@_decorators.codegen(_get_tunable, "common")
def _(state: CodegenState) -> ast.AST:
    return _codegen_tunable(state)


@_decorators.ref(register_tunable)
def _(name: str, fragment: ConfigSpecFragment) -> int | float | bool:
    """Reference implementation of register_tunable."""
    from ..runtime.ref_mode import RefModeContext

    # Get config from the current RefModeContext
    context = RefModeContext.current()
    config = context.config

    # Determine the value to use
    value: object
    assert config is not None
    if name in config:
        value = config[name]
    else:
        value = fragment.default()

    # Match the compiler's exact scalar types before an integer clamp can
    # accidentally turn a subclass into a supported plain int.
    if type(value) not in (int, float, bool):
        raise exc.TunableTypeNotSupported(type(value))

    # For BaseIntegerFragment subclasses (IntegerFragment, PowerOfTwoFragment), apply clamp
    # This ensures the value is within valid bounds
    if isinstance(fragment, BaseIntegerFragment) and isinstance(value, (int, bool)):
        value = fragment.clamp(int(value))

    if type(value) not in (int, float, bool):
        raise exc.TunableTypeNotSupported(type(value))
    return cast("int | float | bool", value)
