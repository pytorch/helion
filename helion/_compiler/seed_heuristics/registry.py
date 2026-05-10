from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class SeedHeuristic:
    """Base class for compiler-owned autotune seed heuristics."""

    name: ClassVar[str]
    backend: ClassVar[str]

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        raise NotImplementedError

    @classmethod
    def get_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        raise NotImplementedError


SeedHeuristicType = type[SeedHeuristic]
