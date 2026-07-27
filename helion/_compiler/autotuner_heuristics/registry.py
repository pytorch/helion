from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .common import HardwareTarget


class AutotunerHeuristic:
    """Base class for compiler-owned autotuner heuristics."""

    name: ClassVar[str]
    backend: ClassVar[str]
    promote_seed_to_default: ClassVar[bool] = False
    # Arches on which the promoted seed may become the autotune-off DEFAULT.
    # None = promote wherever the heuristic fires (back-compat; e.g. an
    # arch-fenced heuristic whose is_eligible already confines it). A tuple
    # restricts PROMOTION to those (device_kind, compute_capability) pairs while
    # leaving where the seed FIRES untouched — so an arch-agnostic seed can still
    # be offered everywhere as a search candidate but only defaulted on validated
    # arches.
    PROMOTE_TARGETS: ClassVar[tuple[HardwareTarget, ...] | None] = None

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        raise NotImplementedError

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        return None

    @classmethod
    def should_promote(cls, env: CompileEnvironment) -> bool:
        """Whether this heuristic's seed should become the autotune-off default
        on the current device. Gates the (possibly arch-agnostic) seed's
        promotion to ``PROMOTE_TARGETS`` without changing where the seed fires."""
        if not cls.promote_seed_to_default:
            return False
        if cls.PROMOTE_TARGETS is None:
            return True
        from .common import matches_hardware

        return matches_hardware(env, cls.PROMOTE_TARGETS)


AutotunerHeuristicType = type[AutotunerHeuristic]
