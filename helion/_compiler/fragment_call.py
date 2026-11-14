"""Conservative fragment-input guards for opaque Python calls."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import types
from typing import TYPE_CHECKING
from typing import cast

from ..autotuner.config_fragment import ConfigSpecFragment
from ..language.constexpr import ConstExpr
from .compile_environment import CompileEnvironment
from .compile_environment import RuntimeInputSpecialization
from .host_function import HostFunction
from .variable_origin import ArgumentOrigin
from .variable_origin import AttributeOrigin
from .variable_origin import ClosureOrigin
from .variable_origin import GetItemOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .variable_origin import Origin


@dataclass(frozen=True, eq=False, repr=False)
class _FragmentCallIdentity:
    value: object

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FragmentCallIdentity) and self.value is other.value

    def __repr__(self) -> str:
        # Identity affects only in-memory binding reuse. The existing input
        # domain and config-spec keys retain their persistent representation.
        return "_FragmentCallIdentity()"


def _has_fragment_call_children(value: object) -> bool:
    return (
        type(value) in (tuple, list, dict, frozenset)
        or isinstance(value, (types.FunctionType, types.MethodType))
        or (dataclasses.is_dataclass(value) and not isinstance(value, type))
        or (isinstance(value, tuple) and hasattr(type(value), "__match_args__"))
    )


def _fragment_call_inputs(
    values: Sequence[object],
) -> tuple[tuple[_FragmentCallIdentity, ...], bool]:
    result = []
    has_constexpr = False
    pending = [(value, frozenset()) for value in reversed(values)]
    while pending:
        value, ancestors = pending.pop()
        if isinstance(value, ConstExpr):
            has_constexpr = True
            if id(value) not in ancestors:
                pending.append((value.value, ancestors | {id(value)}))
        elif isinstance(value, ConfigSpecFragment):
            result.append(_FragmentCallIdentity(value))
        elif _has_fragment_call_children(value):
            identity = id(value)
            if identity in ancestors:
                continue
            ancestors = ancestors | {identity}
            if isinstance(value, types.MethodType):
                children = (value.__self__, value.__func__)
            elif isinstance(value, types.FunctionType):
                cells = []
                for cell in value.__closure__ or ():
                    try:
                        cells.append(cell.cell_contents)
                    except ValueError:
                        # A legal function can have an empty, unused cell.
                        continue
                children = (
                    *(value.__defaults__ or ()),
                    *(value.__kwdefaults__ or {}).values(),
                    *cells,
                )
            elif dataclasses.is_dataclass(value):
                # Match the fields admitted by TypeInfo.from_example.
                children = tuple(
                    getattr(value, field.name) for field in dataclasses.fields(value)
                )
            elif isinstance(value, dict):
                children = tuple(value.values())
            elif type(value) is frozenset:
                children = tuple(value)
            else:
                children = cast("Sequence[object]", value)
            pending.extend((child, ancestors) for child in reversed(children))
    return tuple(result), has_constexpr


def _fragment_call_input(values: Sequence[object]) -> tuple[_FragmentCallIdentity, ...]:
    return _fragment_call_inputs(values)[0]


def guard_fragment_call_input(origin: Origin) -> None:
    """Rebind fresh declarations without executing the opaque helper again.

    The call may inspect fragment identity regardless of its result type.
    Read original inputs, including callable defaults/closures and receivers;
    a lifted tracing wrapper does not retain all of that implicit state.
    """
    while isinstance(origin, (GetItemOrigin, AttributeOrigin, ClosureOrigin)):
        origin = origin.value
    if not isinstance(origin, ArgumentOrigin):
        return
    env = CompileEnvironment.current()
    value = env.runtime_arg_values_by_name[origin.name]
    fragments, has_constexpr = _fragment_call_inputs((value,))
    if not fragments and not has_constexpr:
        constexpr = origin.name in HostFunction.current().constexpr_args
        if not constexpr and type(value) in (types.FunctionType, tuple, list, dict):
            # The ordinary recursive key re-reads supported container entries,
            # callable defaults and cells. A later fragment changes that key;
            # opaque defaults, custom records and empty cells still need a guard.
            # Import locally: runtime.kernel imports TypeInfo during startup.
            from ..runtime.kernel import _nested_fast_dispatch_key

            if _nested_fast_dispatch_key(value) is not None:
                return
        elif type(value) is types.FunctionType:
            if constexpr:
                # A constexpr function keys by identity, not its implicit inputs.
                # Stateless functions are safe; defaults and cells can acquire
                # fragments without changing that function identity.
                if not (
                    value.__defaults__ or value.__kwdefaults__ or value.__closure__
                ):
                    return
        elif not _has_fragment_call_children(value):
            # Builtin callables and tensor metadata have no implicit fragment
            # inputs in this traversal. Structured receivers can acquire them.
            return
    # Even a ConstExpr(None) can later compare equal to a custom fragment.
    # Guard the whole argument so static iterations share one input schema.
    retains_input_identity = bool(fragments)
    env.has_opaque_fragment_inputs |= retains_input_identity
    env.register_runtime_input_specialization(
        f"fragment_call_input:{origin.name}",
        RuntimeInputSpecialization(
            sources=(origin.to_source(),),
            classifier_identity=("fragment_call_input", origin.name),
            classifier=_fragment_call_input,
            retains_input_identity=retains_input_identity,
        ),
    )
