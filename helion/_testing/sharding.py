"""Split test modules across CI shards by their recorded durations.

``HELION_TEST_SHARD=i/n`` (read by ``test/conftest.py``) makes one CI job run
the modules of shard ``i``. Whole modules stay together so module-level
kernels, fixtures and reference compiles are built once per shard. The shards
are balanced from the per-module seconds that ``scripts/shard_durations.py``
records in ``test/shard_durations.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

DURATIONS_FILE = Path(__file__).resolve().parents[2] / "test" / "shard_durations.json"
# Seconds a module costs even when every test in it skips: its import and
# collection. Without a floor the zero-second modules would tie and all land on
# the same shard.
MIN_WEIGHT = 1.0


def load_durations(path: Path = DURATIONS_FILE) -> dict[str, float]:
    """The recorded seconds per rootdir-relative test module, ``{}`` if absent."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {str(module): float(seconds) for module, seconds in data.items()}


def module_weights(
    modules: Iterable[str], durations: Mapping[str, float]
) -> dict[str, float]:
    """The weight of each module: its recorded duration (at least
    ``MIN_WEIGHT``), or the median of the recorded durations among ``modules``
    for one that has none (``MIN_WEIGHT`` when none is recorded), so a new
    module neither vanishes nor dominates a shard."""
    names = sorted(set(modules))
    recorded = sorted(durations[name] for name in names if name in durations)
    default = recorded[len(recorded) // 2] if recorded else MIN_WEIGHT
    return {name: max(durations.get(name, default), MIN_WEIGHT) for name in names}


def assign_shards(
    modules: Iterable[str], shards: int, durations: Mapping[str, float]
) -> dict[str, int]:
    """Assign each module to one of ``shards`` shards so their weights balance.

    Longest-processing-time-first: the heaviest module goes to the least
    loaded shard, then the next heaviest, and so on, which keeps the few
    heaviest modules apart. Ties break on the module name, so every process
    of a run computes the same assignment from the same modules.
    """
    if shards < 1:
        raise ValueError(f"shards must be positive, got {shards}")
    weights = module_weights(modules, durations)
    loads = [0.0] * shards
    assignment: dict[str, int] = {}
    for name in sorted(weights, key=lambda name: (-weights[name], name)):
        shard = min(range(shards), key=lambda index: (loads[index], index))
        loads[shard] += weights[name]
        assignment[name] = shard
    return assignment


def shard_loads(
    assignment: Mapping[str, int], shards: int, durations: Mapping[str, float]
) -> list[float]:
    """The summed weight of each shard under ``assignment``."""
    weights = module_weights(assignment, durations)
    loads = [0.0] * shards
    for name, shard in assignment.items():
        loads[shard] += weights[name]
    return loads
