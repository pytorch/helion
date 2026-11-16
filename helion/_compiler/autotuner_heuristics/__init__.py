from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .common import dedupe_configs

if TYPE_CHECKING:
    import torch

    from ...runtime.config import Config
    from ..backend import Backend
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import AutotunerHeuristicType
    from .registry import CompilerHeuristicSpecializationFact


def _registered_backend(name: str) -> Backend | None:
    """The registered backend called ``name``; None for names no backend owns.

    Only the name-keyed entry point :func:`compiler_seed_specialization_facts`
    needs this lookup; the bind-time entry points use the live ``env.backend``.
    Synthetic backend names (tests) own no hooks.
    """
    from ..backend_registry import get_backend
    from ..backend_registry import list_backends

    if name not in list_backends():
        return None
    return get_backend(name)


def _builtin_heuristics() -> dict[str, tuple[AutotunerHeuristicType, ...]]:
    from ..backend_registry import get_backend
    from ..backend_registry import list_backends

    return {name: get_backend(name).autotuner_heuristics() for name in list_backends()}


def _current_registry_generation() -> int:
    from ..backend_registry import registry_generation

    return registry_generation()


# All active heuristics by backend, in each backend's ranked order (the
# ``Backend.autotuner_heuristics`` hook). This dict is the lookup
# ``get_heuristics`` reads at call time, which keeps it the one seam tests
# patch to plant synthetic registries. It is filled from the backends
# registered at import and refreshed in place by ``get_heuristics`` when the
# registry has changed since (``registry_generation``), so a backend
# registered later through ``register_compiler_backend`` is seen on the next
# lookup and an unregistered one is forgotten.
HEURISTICS_BY_BACKEND: dict[str, tuple[AutotunerHeuristicType, ...]] = (
    _builtin_heuristics()
)
_HEURISTICS_GENERATION: int = _current_registry_generation()

log: logging.Logger = logging.getLogger(__name__)


def _refresh_heuristics() -> None:
    global _HEURISTICS_GENERATION

    generation = _current_registry_generation()
    if generation == _HEURISTICS_GENERATION:
        return
    current = _builtin_heuristics()
    for name in list(HEURISTICS_BY_BACKEND):
        if name not in current:
            del HEURISTICS_BY_BACKEND[name]
    HEURISTICS_BY_BACKEND.update(current)
    _HEURISTICS_GENERATION = generation


def get_heuristics(backend: str) -> tuple[AutotunerHeuristicType, ...]:
    _refresh_heuristics()
    return HEURISTICS_BY_BACKEND.get(backend, ())


def compiler_promotion_specialization_key(
    backend: str,
    device: torch.device,
) -> tuple[tuple[str, str | None], ...]:
    """Return named-target facts that can change seeds or their promotion.

    Compute capability is already part of the bound-kernel specialization key.
    Heuristics may request exact product identity either for a promotion fence
    or because product policy changes their emitted seeds. Non-matching products
    share the ``None`` bucket; capability-only heuristics add no key at all.
    ``get_hardware_info`` is cached per canonical device.
    """
    registry_signature = []
    for heuristic in get_heuristics(backend):
        named_targets = heuristic.CACHE_NAMED_TARGETS
        if named_targets is None and heuristic.promote_seed_to_default:
            named_targets = heuristic.PROMOTE_NAMED_TARGETS
        if named_targets:
            registry_signature.append((heuristic.name, named_targets))
    if not registry_signature:
        return ()

    from ..._argument_device import _canonicalize_argument_device
    from ..._hardware import get_hardware_info

    try:
        hardware = get_hardware_info(_canonicalize_argument_device(device))
        hardware_identity = (
            hardware.device_kind,
            hardware.hardware_name,
            hardware.compute_capability,
        )
    except RuntimeError:
        hardware_identity = None
    return tuple(
        (
            heuristic_name,
            hardware_identity[1]
            if hardware_identity is not None
            and (
                hardware_identity in named_targets
                or (*hardware_identity[:2], None) in named_targets
            )
            else None,
        )
        for heuristic_name, named_targets in registry_signature
    )


def compiler_seed_specialization_facts(
    backend: str,
    fired_heuristics: tuple[str, ...] | list[str],
) -> frozenset[CompilerHeuristicSpecializationFact]:
    """Return device facts needed by compiler seeds that actually fired.

    Eligibility is known only after tracing the kernel.  Deferring this lookup
    until then avoids putting SM count in every bound-kernel key merely because
    one heuristic registered for the backend happens to depend on it.
    """
    fired = frozenset(fired_heuristics)
    heuristics = get_heuristics(backend)
    hooks = _registered_backend(backend)
    if hooks is not None:
        # Families appended after the complete ranked list still own facts.
        heuristics = (*heuristics, *hooks.seed_fact_heuristics())
    return frozenset(
        fact
        for heuristic in heuristics
        if heuristic.name in fired
        for fact in heuristic.CACHE_SPECIALIZATION_FACTS
    )


def compiler_seed_configs(
    env: CompileEnvironment,
    device_ir: DeviceIR,
) -> list[Config]:
    configs: list[Config] = []
    heuristics = get_heuristics(env.backend_name)
    backend = env.backend
    registered_fact_specialization_facts: set[CompilerHeuristicSpecializationFact] = (
        set()
    )
    for heuristic in heuristics:
        registered_fact_specialization_facts.update(
            heuristic.register_facts(env, device_ir)
        )
    registered_fact_specialization_facts.update(
        backend.register_seed_facts(env, device_ir)
    )
    env.compiler_fact_specialization_facts = frozenset(
        registered_fact_specialization_facts
    )
    env.config_spec.autotuner_heuristics = []
    env.config_spec.compiler_default_config = None
    env.config_spec.compiler_seed_timeout_retry_repetitions = None
    if env.settings.disable_autotuner_heuristics:
        return configs

    for heuristic in heuristics:
        try:
            if not heuristic.is_eligible(env, device_ir):
                continue

            # A heuristic may plant a RANKED multi-seed list (get_seed_configs);
            # the single get_seed_config is the primary (== the list's [0]). The
            # default base hook returns None, so existing single-seed heuristics
            # keep their exact behavior.
            ranked = heuristic.get_seed_configs(env, device_ir)
            if ranked is None:
                config = heuristic.get_seed_config(env, device_ir)
                ranked = [config] if config is not None else []
        except Exception as e:
            log.debug(
                "Autotuner heuristic %s failed while generating compiler seed config: %s",
                heuristic.name,
                e,
                exc_info=True,
            )
            continue
        ranked = [c for c in ranked if c is not None]
        if not ranked:
            continue
        configs.extend(ranked)
        if heuristic.should_promote(env):
            # The primary (rank-0) is the promoted default.
            env.config_spec.compiler_default_config = ranked[0]
        env.config_spec.autotuner_heuristics.append(heuristic.name)
    configs = dedupe_configs(configs)
    configs = backend.postprocess_seed_configs(env, device_ir, configs)
    return dedupe_configs(configs)


def register_compiler_coverage_groups(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    """Declare independent coverage after every ordinary field is finalized.

    Sampling and explicit-override policy belong to the generic autotuner.
    Keep declarations present when automatic heuristics are disabled.
    """
    env.backend.register_coverage_groups(env, device_ir)
