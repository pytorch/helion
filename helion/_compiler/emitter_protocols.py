"""Core protocols between the compile pipeline and a backend's code emitters.

A backend contributes special-purpose code emitters through three ``Backend``
hooks, each returning plain ordered tuples built in the backend's own package:

* ``root_emitters()`` -> ``RootEmitter`` objects that ``GenerateAST.visit_For``
  tries in order before the ordinary lowering of a root loop;
* ``kernel_generators()`` -> ``KernelGenerator`` objects that ``_generate_ast``
  tries in order before the ordinary ``GenerateAST`` pass;
* ``retry_policy()`` -> the ``RetryPolicy`` under which ``generate_ast``
  re-generates a kernel after one of the backend's regeneration signals.

``EmittedBody`` is the one return contract for device-body emitters whose own
signatures differ (statement lists, source text, statements plus read names).
The adapters bind the wrapped emitter when the backend module is imported, so
a test that wants to intercept the body a ``codegen_*`` function renders
patches the adapter (``emit_*_body``), not the ``emit_*_device_body`` under it.
"""

from __future__ import annotations

import abc
import ast
import contextlib
import dataclasses
from typing import TYPE_CHECKING
from typing import ParamSpec

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .device_ir import GraphInfo
    from .generate_ast import GenerateAST
    from .host_function import HostFunction

P = ParamSpec("P")


@dataclasses.dataclass(frozen=True)
class RootEmitter(abc.ABC):
    """A special-purpose emitter that owns a whole root loop.

    ``try_emit`` is the dispatch step ``GenerateAST.visit_For`` runs for each
    registered emitter in order: ``plan`` says whether this emitter owns the
    root, ``emit`` renders it, and when the emitter's late validation refuses
    the root ``decline`` applies the family's own refusal semantics (which
    exception, whether the plan slot is cleared, or falling through to the
    next emitter).  Each family carries those semantics explicitly because the
    families never shared one.
    """

    name: str

    @abc.abstractmethod
    def plan(self, codegen: GenerateAST) -> object | None:
        """The plan recorded for this emitter's family, or ``None`` when it
        does not own the current root."""

    @abc.abstractmethod
    def emit(self, codegen: GenerateAST, plan: object) -> bool:
        """Render the root into ``codegen``; ``False`` when late validation
        refuses it."""

    @abc.abstractmethod
    def decline(self, codegen: GenerateAST, plan: object) -> None:
        """Apply the family's refusal: raise its exception, or return so the
        next emitter (and finally the ordinary lowering) is tried."""

    def try_emit(self, codegen: GenerateAST) -> bool:
        plan = self.plan(codegen)
        if plan is None:
            return False
        if self.emit(codegen, plan):
            return True
        self.decline(codegen, plan)
        return False


@dataclasses.dataclass(frozen=True)
class EmittedBody:
    """The device body a special-purpose emitter produced.

    ``reads`` are the kernel parameter names the body refers to that the
    emitter's caller must bind (the gated flash body's scalar tensors).
    """

    statements: list[ast.AST]
    reads: frozenset[str] = frozenset()

    @classmethod
    def from_source(cls, source: str) -> EmittedBody:
        """Parse an indented block of device source (bodies rendered as text
        carry a four-space indent)."""
        wrapped = ast.parse("if True:\n" + source)
        assert isinstance(wrapped.body[0], ast.If)
        return cls(list(wrapped.body[0].body))


def statements_adapter(
    emit: Callable[P, Sequence[ast.AST]],
) -> Callable[P, EmittedBody]:
    """``emit`` with the ``EmittedBody`` return contract (signature unchanged).

    ``emit`` is bound here, at definition time: patch the adapter to intercept
    the body a ``codegen_*`` function renders, not the wrapped emitter."""

    def adapted(*args: P.args, **kwargs: P.kwargs) -> EmittedBody:
        return EmittedBody(list(emit(*args, **kwargs)))

    return adapted


def source_adapter(emit: Callable[P, str]) -> Callable[P, EmittedBody]:
    """``emit`` rendering source text, with the ``EmittedBody`` return contract
    (``emit`` bound at definition time, as in ``statements_adapter``)."""

    def adapted(*args: P.args, **kwargs: P.kwargs) -> EmittedBody:
        return EmittedBody.from_source(emit(*args, **kwargs))

    return adapted


def statements_and_reads_adapter(
    emit: Callable[P, tuple[Sequence[ast.AST], Collection[str]]],
) -> Callable[P, EmittedBody]:
    """``emit`` returning statements plus the names they read, with the
    ``EmittedBody`` return contract (``emit`` bound at definition time, as in
    ``statements_adapter``)."""

    def adapted(*args: P.args, **kwargs: P.kwargs) -> EmittedBody:
        statements, reads = emit(*args, **kwargs)
        return EmittedBody(list(statements), frozenset(reads))

    return adapted


@dataclasses.dataclass(frozen=True)
class KernelTransforms:
    """The optional arguments of ``generate_ast`` that specialize one pass:
    memory transforms of a fusing caller, caller-built codegen graphs and
    counters, the host prefix a staged caller collects and a bounded-cache
    request."""

    store_transform: Callable[..., ast.AST] | None = None
    load_transform: Callable[..., ast.AST] | None = None
    extra_params: list[str] | None = None
    codegen_graphs: list[GraphInfo] | None = None
    memory_counters: dict[str, int] | None = None
    host_prefix: list[ast.AST] | None = None
    bounded_cache_request: object | None = None

    @property
    def carried(self) -> frozenset[str]:
        """The names of the specializing arguments this call carries."""
        return frozenset(
            field.name
            for field in dataclasses.fields(self)
            if (
                bool(self.extra_params)
                if field.name == "extra_params"
                else getattr(self, field.name) is not None
            )
        )


def _keep_config(config: Config) -> Config:
    return config


@dataclasses.dataclass(frozen=True)
class TransformPolicy:
    """Which specializing arguments a kernel generator cannot honor, and what
    happens when a call carries one (or ``config_conflict`` holds): ``decline``
    raises the generator's error or returns the config the later generators
    and the ordinary pass continue with."""

    excluded: frozenset[str] = frozenset()
    config_conflict: Callable[[Config], bool] | None = None
    decline: Callable[[Config], Config] = _keep_config

    def honors(self, config: Config, transforms: KernelTransforms) -> bool:
        if self.excluded & transforms.carried:
            return False
        return self.config_conflict is None or not self.config_conflict(config)


FORWARD_TRANSFORMS = TransformPolicy()


@dataclasses.dataclass(frozen=True)
class KernelGenerator:
    """A whole-kernel generator tried before the ordinary ``GenerateAST`` pass.

    ``_generate_ast`` runs the first generator whose ``applies`` holds and
    whose ``transform_policy`` honors the call; a generator whose policy does
    not honor it declines (raising, or rewriting the config the search
    continues with) and the next generator is tried.
    """

    name: str
    applies: Callable[[HostFunction, Config, KernelTransforms], bool]
    generate: Callable[[HostFunction, Config, bool, KernelTransforms], ast.Module]
    transform_policy: TransformPolicy = FORWARD_TRANSFORMS


@dataclasses.dataclass(frozen=True)
class RetryPolicy:
    """How ``generate_ast`` re-generates a kernel after a regeneration signal.

    ``signals`` are the exceptions that ask for another pass (a backend's
    policy sets the field's default); ``repair`` returns the config of that
    pass (or ``None`` to let the signal propagate) and may record on the
    compile environment what the next pass must withhold; ``scope`` brackets
    every pass of one ``generate_ast`` call so such records are restored
    afterwards.  The default retries nothing: ``except ()`` catches nothing.
    """

    signals: tuple[type[BaseException], ...] = ()

    def scope(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def repair(
        self, signal: BaseException, config: Config, *, caller_graphs: bool
    ) -> Config | None:
        return None


NO_RETRY = RetryPolicy()
