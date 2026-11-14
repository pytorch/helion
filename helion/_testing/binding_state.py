"""A bound kernel's binding state on a ``CompileEnvironment`` stand-in.

The readers of a binding's layout facts (``keyed_hint``,
``live_example_input``, ``input_residues_hold_modulo``,
``traced_strides_hold`` and the proofs built on them) consult state the
binding records.  A unit test's stand-in that takes that state and the real
readers from here keeps working as the readers gain checks.
"""

from __future__ import annotations

import contextvars
from typing import Any
from typing import TypeVar
from typing import cast

from .._compiler.compile_environment import CompileEnvironment

_Env = TypeVar("_Env")

# The methods that read the state; a stand-in that is no CompileEnvironment
# gets the real ones bound to it.
_READERS = (
    "exact_input_metadata_holds",
    "keyed_hint",
    "live_example_input",
    "input_residues_hold_modulo",
    "traced_strides_hold",
)


def with_binding_state(env: _Env) -> _Env:
    """Give ``env`` the state of a bound kernel's own launches, which a test
    changes afterwards where it needs to: dispatch guards enforced, no input
    bound from an outer trace's FakeTensor, the inputs' bound base residues
    held to the signature's modulus, and no input at symbolic strides.

    A ``CompileEnvironment`` made without ``__init__`` (``object.__new__``)
    gets the private fields its properties read; any other stand-in gets
    plain attributes and the real readers, bound to it."""
    stub = cast("Any", env)
    if isinstance(env, CompileEnvironment):
        stub._dispatch_guards_enforced = contextvars.ContextVar(
            f"stub_dispatch_guards_enforced_{id(env)}", default=True
        )
        stub._input_alignment_limit = contextvars.ContextVar(
            f"stub_input_alignment_limit_{id(env)}", default=None
        )
        stub._symbolic_stride_sources = contextvars.ContextVar(
            f"stub_symbolic_stride_sources_{id(env)}", default=frozenset()
        )
    else:
        stub.dispatch_guards_enforced = True
        stub.input_alignment_limit = None
        stub.symbolic_stride_sources = frozenset()
        for name in _READERS:
            vars(env)[name] = getattr(CompileEnvironment, name).__get__(env)
    stub._binds_outer_trace_inputs = False
    return env
