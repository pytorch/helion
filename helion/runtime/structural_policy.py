"""Backend-neutral structural policy contract (runtime).

A *structural policy* is a backend's frozen snapshot of the Settings that fix a
kernel's config schema before binding.  The autotuner caches, AOT heuristics and
measurement rows persist it beside each config and compare it on every read;
this module is the only thing they need to know about it.  The concrete policy
class, its versioned envelope schema and every persisted identity string belong
to the backend that owns the settings (today :mod:`helion.runtime.cute.structural_policy`
and :mod:`helion.runtime.cute.structural_config`); :meth:`helion._compiler.backend.Backend.structural_policy_type`
names that class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

if TYPE_CHECKING:
    from .config import Config
    from .settings import Settings


class StructuralPolicy(Protocol):
    """An immutable, versioned snapshot of a backend's structural settings.

    Implementations are value objects: equal policies compare equal, and
    :meth:`identity` is a stable digest of the persisted envelope (never
    Python ``hash()``), used in cache keys, file names and measurement rows.
    """

    def to_dict(self) -> dict[str, object]:
        """Return a detached, versioned envelope."""
        ...

    def to_json(self) -> str:
        """Canonical serialization, independent of mapping insertion order."""
        ...

    def identity(self) -> str:
        """A stable digest of the versioned effective policy."""
        ...

    @classmethod
    def from_settings(cls, settings: Settings) -> StructuralPolicy:
        """Snapshot the effective structural settings without activating them."""
        ...


class StructuralConfig:
    """A detached Config snapshot and the policy that defines its axes.

    The envelope is deliberately not a Mapping or an autotuner dimension;
    ordinary ``Config`` objects retain their existing format and behavior.
    Backends subclass this with their versioned envelope schema.
    """

    __slots__ = ()

    @property
    def policy(self) -> StructuralPolicy:
        """The policy whose config schema this envelope was recorded under."""
        raise NotImplementedError

    @property
    def config(self) -> Config:
        """Return a fresh ordinary Config; nested mutations cannot alter the record."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError

    def identity(self) -> str:
        raise NotImplementedError


class StructuralPolicyError(ValueError):
    """A recorded config policy cannot be used by the selected kernel."""


def require_same_structural_policy(
    actual: StructuralPolicy | None,
    expected: StructuralPolicy | None,
    *,
    context: str,
) -> None:
    if actual != expected:
        raise StructuralPolicyError(
            f"{context} has a different structural policy. "
            "Select its policy before binding a new Kernel; an existing bound "
            "kernel cannot change config schemas."
        )


def bound_structural_policy(
    kernel: object, *, captured: bool = False
) -> StructuralPolicy | None:
    """Read opt-in cache identity or the policy captured when a bound was traced."""
    from .kernel import BoundKernel

    if not isinstance(kernel, BoundKernel):
        return None
    return (
        kernel._structural_policy if captured else kernel.kernel.cute_structural_policy
    )
