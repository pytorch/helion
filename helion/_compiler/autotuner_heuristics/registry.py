from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class AutotunerHeuristic:
    """Base class for compiler-owned autotuner heuristics."""

    name: ClassVar[str]
    backend: ClassVar[str]
    promote_seed_to_default: ClassVar[bool] = False

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        raise NotImplementedError

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        return None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        """Optional RANKED multi-seed hook. ``None`` (the default) means the
        heuristic plants only its single ``get_seed_config``. A heuristic that
        overrides this returns ``[primary, *alternates]`` — the loader plants the
        whole list as compiler seeds (``compiler_seed_configs`` is a list); the
        primary (``[0]``, the no-autotune Product-A config) must equal
        ``get_seed_config``. Alternates earn their slot only by covering the
        region's variance or speeding Product-B search convergence.
        """
        return None


AutotunerHeuristicType = type[AutotunerHeuristic]
