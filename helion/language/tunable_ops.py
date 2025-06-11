from __future__ import annotations

from typing import TYPE_CHECKING

from torch._inductor.codegen.simd import constant_repr

from .. import exc
from .._compiler.ast_extension import expr_from_string
from ..autotuner.config_fragment import ConfigSpecFragment
from ..autotuner.config_spec import VALID_KEYS
from ..exc import NotInsideKernel
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.type_propagation import TypeInfo
    from .._compiler.variable_origin import Origin

__all__ = ["register_tunable"]


@_decorators.api(is_device_only=False)
def register_tunable(name: str, fragment: ConfigSpecFragment) -> int:
    """
    Register a tunable parameter for autotuning.

    This function allows you to define parameters that can be automatically tuned
    during the autotuning process. The fragment defines the search space and default value.

    :param name: The key for the tunable parameter in the Config().
    :param fragment: A ConfigSpecFragment that defines the search space (e.g., PowerOfTwoFragment)
    :return: The value assigned to this tunable parameter in the current configuration.
    """
    raise NotInsideKernel


@_decorators.type_propagation(register_tunable)
def _register_tunable_type(
    name: TypeInfo, fragment: TypeInfo, *, origin: Origin
) -> TypeInfo:
    # During type propagation, register the tunable parameter and return unbacked symint
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.type_propagation import NumericType

    env = CompileEnvironment.current()

    try:
        fragment_val = fragment.as_literal()
        name_val = name.as_literal()
    except NotImplementedError:
        fragment_val = None
        name_val = None
    if not (isinstance(name_val, str) and isinstance(fragment_val, ConfigSpecFragment)):
        raise exc.RegisterTunableArgTypes(name, fragment)
    del name, fragment

    if name_val in VALID_KEYS or f"{name_val}s" in VALID_KEYS:
        raise exc.TunableNameConflict(name_val)
    if (
        name_val in env.config_spec.user_defined_tunables
        and env.config_spec.user_defined_tunables[name_val] != fragment_val
    ):
        raise exc.TunableNameConflict(name_val)

    # register the value for tuning
    env.config_spec.user_defined_tunables[name_val] = fragment_val

    python_type = type(fragment_val.default())
    if not issubclass(python_type, (int, float, bool)):
        raise exc.TunableTypeNotSupported(python_type)
    return NumericType.subtype(python_type).new_unbacked(origin)


@_decorators.codegen(register_tunable)
def _register_tunable_codegen(state: CodegenState) -> ast.AST:
    name = state.proxy_arg(0)
    assert isinstance(name, str)
    config_value = state.config[name]
    assert isinstance(config_value, (int, float, bool))
    return expr_from_string(constant_repr(config_value))
