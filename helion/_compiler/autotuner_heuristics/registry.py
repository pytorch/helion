from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .common import HardwareTarget
    from .common import NamedHardwareTarget


CompilerHeuristicSpecializationFact = Literal[
    "config_num_sm",
    "device_num_sm",
    "input_tensor_metadata",
]


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
    # Optional exact product targets for seeds validated on one named device.
    # This never limits where a seed fires, only where it becomes the default.
    PROMOTE_NAMED_TARGETS: ClassVar[frozenset[NamedHardwareTarget] | None] = None
    # Runtime facts that can change this heuristic's emitted seed configs.  The
    # bound-kernel cache adds these facts only after the heuristic actually
    # fires, so a shape- or SM-sensitive heuristic does not force unrelated
    # kernels on the same backend to specialize on those facts.
    #
    # ``device_num_sm`` is the physical count returned by get_num_sm(device).
    # ``config_num_sm`` is ConfigSpec.num_sm, which also applies the kernel's
    # persistent_reserved_sms setting.
    # ``input_tensor_metadata`` is the exact shape and stride of every input
    # tensor. Dynamic kernels request it when first-binding shape/layout hints
    # affect their seed population or the generated schedule's validity. It
    # intentionally creates a distinct bound kernel for each metadata tuple;
    # heuristics should request it only when coarser reuse would be unsound.
    CACHE_SPECIALIZATION_FACTS: ClassVar[
        frozenset[CompilerHeuristicSpecializationFact]
    ] = frozenset()

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        """Register correctness facts before optional seed generation.

        Overrides must be idempotent because callers may regenerate compiler
        seeds for the same bound kernel. This hook runs even when autotuner
        heuristics are disabled; it must not itself add or promote seed configs.
        Exceptions intentionally propagate and fail binding because silently
        dropping correctness facts could make later normalization unsound.

        Return the runtime facts that can change either the registration
        outcome or the registered correctness facts. An input-dependent hook
        must return its requirements even when this binding registers nothing,
        so a later binding that would register facts cannot reuse it. These
        requirements are independent of ``CACHE_SPECIALIZATION_FACTS``, which
        applies only when this heuristic actually emits seeds.
        """
        return frozenset()

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
        on the current device. Target gates do not change where the seed fires."""
        if not cls.promote_seed_to_default:
            return False
        if cls.PROMOTE_TARGETS is None and cls.PROMOTE_NAMED_TARGETS is None:
            return True
        from .common import matches_hardware

        return matches_hardware(
            env,
            cls.PROMOTE_TARGETS,
            named_targets=cls.PROMOTE_NAMED_TARGETS,
        )

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
