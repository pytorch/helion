from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterable
from typing import Sequence
from typing import cast

if TYPE_CHECKING:
    from ...autotuner.config_spec import BlockSizeSpec
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment

HardwareTarget = tuple[str, str | None]


def dedupe_configs(configs: Iterable[Config]) -> list[Config]:
    result: list[Config] = []
    seen: set[Config] = set()
    for config in configs:
        if config in seen:
            continue
        seen.add(config)
        result.append(config)
    return result


def matches_hardware(
    env: CompileEnvironment,
    targets: tuple[HardwareTarget, ...],
) -> bool:
    from ..._hardware import get_hardware_info

    hardware = get_hardware_info(env.device)
    return (hardware.device_kind, hardware.compute_capability) in targets or (
        hardware.device_kind,
        None,
    ) in targets


def clamp_block_size_targets(
    env: CompileEnvironment,
    block_dims: Sequence[tuple[int, int, int]],
) -> list[int] | None:
    """Clamp block-size targets against the live ConfigSpec constraints.

    Each entry in *block_dims* is ``(block_id, static_dim, target)``.
    Returns the clamped block sizes, or ``None`` if any axis cannot
    satisfy its floor/ceiling constraints.
    """
    block_sizes: list[int] = []
    for block_id, static_dim, target in block_dims:
        try:
            spec = cast(
                "BlockSizeSpec",
                env.config_spec.block_sizes.block_id_lookup(block_id),
            )
        except KeyError:
            return None
        candidate = min(target, static_dim)
        if candidate < 1:
            return None
        candidate = 1 << (candidate.bit_length() - 1)
        floor = max(spec.min_size, spec.autotuner_min)
        if candidate < floor:
            return None
        candidate = min(candidate, spec.max_size)
        if candidate < floor:
            return None
        block_sizes.append(candidate)
    return block_sizes
