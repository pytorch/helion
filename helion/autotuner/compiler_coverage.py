from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field
import json
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeAlias
from typing import cast

from ..exc import InvalidConfig
from ..runtime.config import Config
from .config_fragment import BooleanFragment
from .config_fragment import EnumFragment

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from .config_fragment import ConfigSpecFragment
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig


CoverageValue: TypeAlias = bool | int | str | None
MAX_COMPILER_COVERAGE_CONFIGS = 4
COMPILER_COVERAGE_POLICY_VERSION = 1


def same_value(left: object, right: object) -> bool:
    """Scalar modes are typed: False is not the integer 0."""
    return type(left) is type(right) and left == right


@dataclass(frozen=True, init=False)
class CoverageWitness:
    """One ranked ordinary carrier and its requested scalar mode.

    Keep the payload serialized so neither the caller nor a returned Config can
    mutate a registry entry through an aliased nested list.
    """

    _carrier_json: str = field(repr=False)
    value: CoverageValue

    def __init__(self, carrier: Config, value: CoverageValue) -> None:
        if type(value) not in (bool, int, str, type(None)):
            raise ValueError(f"Coverage mode must be a typed scalar, got {value!r}")
        payload = json.dumps(
            carrier.config, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        object.__setattr__(self, "_carrier_json", payload)
        object.__setattr__(self, "value", value)

    @property
    def carrier(self) -> Config:
        return Config.from_dict(json.loads(self._carrier_json))


@dataclass(frozen=True)
class CoverageDependency:
    """An explicit carrier value owned by an earlier registered group."""

    mechanism: str
    key: str
    value: CoverageValue

    def __post_init__(self) -> None:
        if type(self.mechanism) is not str or not self.mechanism:
            raise ValueError("Coverage dependency mechanism must be a nonempty string")
        if type(self.key) is not str or not self.key:
            raise ValueError("Coverage dependency key must be a nonempty string")
        if type(self.value) not in (bool, int, str, type(None)):
            raise ValueError("Coverage dependency value must be a typed scalar")


@dataclass(frozen=True)
class CompilerCoverageGroup:
    """Compiler-owned additive coverage for one optional scalar coordinate.

    The consumer registers its field and strict normalizer first, after proving
    eligibility under the bound structural policy. This registry does not
    establish lowering eligibility or replace that normalizer.
    """

    mechanism: str
    version: int
    key: str
    domain: tuple[CoverageValue, ...]
    legacy: CoverageValue
    witnesses: tuple[CoverageWitness, ...]
    dependencies: tuple[CoverageDependency, ...] = ()

    def __post_init__(self) -> None:
        if not self.mechanism or not self.key:
            raise ValueError("Coverage mechanism and key must be nonempty")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Coverage version must be a positive integer")
        object.__setattr__(self, "domain", tuple(self.domain))
        object.__setattr__(self, "witnesses", tuple(self.witnesses))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if any(type(entry) is not CoverageDependency for entry in self.dependencies):
            raise ValueError("Coverage dependencies must be immutable declarations")
        if len({entry.key for entry in self.dependencies}) != len(
            self.dependencies
        ) or len({entry.mechanism for entry in self.dependencies}) != len(
            self.dependencies
        ):
            raise ValueError("Duplicate or contradictory coverage dependencies")
        if any(
            type(value) not in (bool, int, str, type(None)) for value in self.domain
        ):
            raise ValueError("Coverage domain must contain only typed scalars")
        if not self.domain or len(self.domain) != len(set(self.domain)):
            raise ValueError("Coverage domain must be nonempty with distinct modes")
        if not same_value(self.domain[0], self.legacy):
            raise ValueError(
                "Coverage domain must start with its canonical legacy mode"
            )
        if len(self.witnesses) > MAX_COMPILER_COVERAGE_CONFIGS:
            raise ValueError(
                "At most four coverage witnesses may be declared per group"
            )
        for witness in self.witnesses:
            if not any(same_value(witness.value, value) for value in self.domain):
                raise ValueError(f"Unknown coverage mode {witness.value!r}")

    def validate_field(self, fragment: ConfigSpecFragment) -> None:
        if type(fragment) is BooleanFragment:
            domain: tuple[object, ...] = (False, True)
        elif type(fragment) is EnumFragment:
            domain = fragment.choices
            if (
                fragment.search_choices is not None
                or fragment.coverage_choices is not None
            ):
                raise ValueError("Coverage fields cannot change their sampling domain")
        else:
            raise ValueError(
                "Coverage owns only independent Boolean/Enum scalar fields"
            )
        if len(domain) != len(self.domain) or any(
            not same_value(left, right)
            for left, right in zip(domain, self.domain, strict=True)
        ):
            raise ValueError(f"Coverage domain does not match field {self.key!r}")
        if not same_value(fragment.default(), self.legacy):
            raise ValueError(f"Coverage field {self.key!r} must default to legacy mode")

    def validate_dependencies(self, earlier: Sequence[CompilerCoverageGroup]) -> None:
        """Require an explicit, acyclic and complete carrier declaration.

        Earlier registration is the dependency order. Transitive requirements
        must be repeated explicitly with the same typed values; no implicit
        modes are inserted into a carrier or normalization path.
        """
        if not self.dependencies:
            return
        prior = {group.mechanism: group for group in earlier}
        declared = {entry.key: entry for entry in self.dependencies}
        for entry in self.dependencies:
            group = prior.get(entry.mechanism)
            if group is None or group.key != entry.key:
                raise ValueError(
                    "Coverage dependency must name an earlier registered group/key"
                )
            if not any(same_value(entry.value, value) for value in group.domain):
                raise ValueError("Coverage dependency value is outside its domain")
            for required in group.dependencies:
                present = declared.get(required.key)
                if (
                    present is None
                    or present.mechanism != required.mechanism
                    or not same_value(present.value, required.value)
                ):
                    raise ValueError(
                        "Missing or contradictory transitive coverage dependency"
                    )
        for witness in self.witnesses:
            carrier = witness.carrier.config
            for entry in self.dependencies:
                if entry.key not in carrier or not same_value(
                    carrier[entry.key], entry.value
                ):
                    raise ValueError("Coverage carrier must match every dependency")
            for group in (*earlier, self):
                if group.key not in declared and not same_value(
                    carrier.get(group.key, group.legacy), group.legacy
                ):
                    raise ValueError("Coverage carrier has an undeclared mode")

    def policy(self) -> dict[str, object]:
        result: dict[str, object] = {
            "mechanism": self.mechanism,
            "version": self.version,
            "key": self.key,
            "domain": self.domain,
            "legacy": self.legacy,
            "witnesses": tuple(
                (witness._carrier_json, witness.value) for witness in self.witnesses
            ),
            "max_additions": MAX_COMPILER_COVERAGE_CONFIGS,
        }
        # Empty declarations retain the exact old policy/cache identity.
        if self.dependencies:
            result["dependencies"] = tuple(
                (entry.mechanism, entry.key, entry.value) for entry in self.dependencies
            )
        return result


def coverage_policy(
    groups: Sequence[CompilerCoverageGroup],
) -> dict[str, object] | None:
    if not groups:
        return None
    return {
        "version": COMPILER_COVERAGE_POLICY_VERSION,
        "initial_sampling": "projected-raw-slot-transfer",
        "groups": tuple(group.policy() for group in groups),
    }


@dataclass(frozen=True)
class CoverageOutcome:
    mechanism: str
    origin: Literal["declared", "cache"]
    requested: Config
    effective: Config | None
    outcome: str


def append_compiler_coverage(
    population: list[FlatConfig],
    generation: ConfigGeneration,
    *,
    cached_configs: Callable[[], Sequence[Config]],
    pin: Callable[[Config], None],
) -> tuple[list[FlatConfig], list[CoverageOutcome]]:
    """Append validated witnesses without changing base rows.

    No random draws, geometry repair, cross-product or retry is performed here.
    Invalid raw base rows remain for the ordinary search validation stage.
    Coupled carriers require an explicit dependency declaration. Warm-cache
    discovery retains its existing single-active-mechanism filter.
    """
    groups = generation.config_spec.compiler_coverage_groups
    result = list(population)
    outcomes: list[CoverageOutcome] = []
    seen: set[Config] = set()
    for flat in population:
        try:
            _flat, config = generation.canonicalize_flat(copy.deepcopy(flat))
            generation.config_spec.normalize(config.config)
        except (InvalidConfig, ValueError, TypeError, KeyError, AssertionError):
            continue
        seen.add(config)
    owned_keys = {group.key for group in groups}
    counts = {group.mechanism: 0 for group in groups}
    pinned: dict[str, set[Config]] = {group.mechanism: set() for group in groups}

    def old_values(config: Config) -> dict[str, object]:
        return {
            key: value for key, value in config.config.items() if key not in owned_keys
        }

    def append(
        group: CompilerCoverageGroup,
        witness: CoverageWitness,
        origin: Literal["declared", "cache"],
    ) -> None:
        requested = witness.carrier
        requested.config[group.key] = witness.value
        effective: Config | None = None
        try:
            carrier = witness.carrier
            dependencies = {entry.key: entry.value for entry in group.dependencies}
            if any(
                key not in carrier.config or not same_value(carrier.config[key], value)
                for key, value in dependencies.items()
            ):
                raise InvalidConfig("Coverage carrier must match every dependency")
            for other in groups:
                expected = dependencies.get(other.key, other.legacy)
                if not same_value(
                    carrier.config.get(other.key, other.legacy), expected
                ):
                    raise InvalidConfig(
                        "Coverage carrier must use ordinary legacy modes"
                    )
            _control_flat, control = generation.strict_config_pair(carrier)
            flat, effective = generation.strict_config_pair(requested)
            for entry in group.dependencies:
                if any(
                    entry.key not in config.config
                    or not same_value(config.config[entry.key], entry.value)
                    for config in (control, effective)
                ):
                    raise InvalidConfig(
                        "Coverage dependency is not effective after normalization"
                    )
            if group.dependencies and any(
                other.key != group.key
                and not same_value(
                    control.config.get(other.key, other.legacy),
                    effective.config.get(other.key, other.legacy),
                )
                for other in groups
            ):
                raise InvalidConfig("Coverage mode changed another carrier coordinate")
            if not same_value(
                effective.config.get(group.key, group.legacy), witness.value
            ):
                outcome = "mode_not_effective"
            elif old_values(control) != old_values(effective):
                outcome = "changed_carrier_geometry"
            elif effective in seen:
                outcome = "already_present"
            elif counts[group.mechanism] >= MAX_COMPILER_COVERAGE_CONFIGS:
                outcome = "addition_limit"
            else:
                result.append(copy.deepcopy(flat))
                seen.add(copy.deepcopy(effective))
                counts[group.mechanism] += 1
                outcome = "added"
            if outcome in ("added", "already_present"):
                group_pins = pinned[group.mechanism]
                if len(group_pins) < MAX_COMPILER_COVERAGE_CONFIGS:
                    group_pins.add(copy.deepcopy(effective))
                    pin(copy.deepcopy(effective))
        except (
            InvalidConfig,
            ValueError,
            TypeError,
            KeyError,
            AssertionError,
        ) as error:
            outcome = f"invalid: {error}"
        outcomes.append(
            CoverageOutcome(group.mechanism, origin, requested, effective, outcome)
        )

    for group in groups:
        for witness in group.witnesses:
            append(group, witness, "declared")

    if any(count < MAX_COMPILER_COVERAGE_CONFIGS for count in counts.values()):
        warm: dict[str, list[CoverageWitness]] = {
            group.mechanism: [] for group in groups
        }
        for config in cached_configs():
            active = [
                group
                for group in groups
                if not same_value(
                    config.config.get(group.key, group.legacy), group.legacy
                )
            ]
            if len(active) != 1:
                continue
            group = active[0]
            if counts[group.mechanism] >= MAX_COMPILER_COVERAGE_CONFIGS:
                continue
            value = config.config[group.key]
            if not any(same_value(value, mode) for mode in group.domain):
                continue
            carrier = copy.deepcopy(config)
            carrier.config.pop(group.key, None)
            warm[group.mechanism].append(
                CoverageWitness(carrier, cast("CoverageValue", value))
            )
        # Declared witnesses have priority. Warm entries then retain registry
        # order and their original cache order inside each mechanism's budget.
        for group in groups:
            for witness in warm[group.mechanism]:
                if counts[group.mechanism] >= MAX_COMPILER_COVERAGE_CONFIGS:
                    break
                append(group, witness, "cache")
    return result, outcomes
