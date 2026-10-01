from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from enum import Enum
import hashlib
import json
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

from .config import Config
from .cute_structural_policy import CUTE_STRUCTURAL_SETTINGS
from .cute_structural_policy import CuteStructuralPolicy

if TYPE_CHECKING:
    from .kernel import ConfigLike
    from .settings import Settings


class StructuralPolicyError(ValueError):
    """A recorded config policy cannot be used by the selected kernel."""


class _DefaultStructuralPolicy(Enum):
    OMITTED = "omitted"


DEFAULT_STRUCTURAL_POLICY = _DefaultStructuralPolicy.OMITTED
CuteStructuralPolicyRequest = (
    CuteStructuralPolicy | Literal["auto"] | None | _DefaultStructuralPolicy
)


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class CuteStructuralConfig:
    """A detached Config snapshot and the policy that defines its axes.

    The envelope is deliberately not a Mapping or an autotuner dimension.
    Ordinary Config dictionaries retain their existing format and behavior.
    """

    policy: CuteStructuralPolicy
    _config_json: str

    def __init__(
        self, config: Config | Mapping[str, object], policy: CuteStructuralPolicy
    ) -> None:
        if not isinstance(policy, CuteStructuralPolicy):
            raise TypeError("policy must be a CuteStructuralPolicy")
        if not isinstance(config, Mapping) or not all(
            isinstance(name, str) for name in config
        ):
            raise TypeError("config must be a mapping with string keys")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(
            self,
            "_config_json",
            json.dumps(
                dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        )

    @property
    def config(self) -> Config:
        """Return a fresh ordinary Config; nested mutations cannot alter the record."""
        return Config.from_dict(json.loads(self._config_json))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "helion.cute.structural_config",
            "version": 1,
            "policy": self.policy.to_dict(),
            "config": json.loads(self._config_json),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CuteStructuralConfig:
        if set(value) != {"schema", "version", "policy", "config"}:
            raise StructuralPolicyError("Expected a CuTe structural config envelope")
        if (
            value["schema"] != "helion.cute.structural_config"
            or type(value["version"]) is not int
            or value["version"] != 1
        ):
            raise StructuralPolicyError("Unsupported CuTe structural config version")
        policy = decode_structural_policy(value["policy"])
        config = value["config"]
        if not isinstance(config, Mapping):
            raise StructuralPolicyError("Expected ordinary config knobs in envelope")
        return cls(config, policy)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> CuteStructuralConfig:
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise StructuralPolicyError("Expected a CuTe structural config envelope")
        return cls.from_dict(decoded)

    def identity(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"helion.CuteStructuralConfig(config=helion.Config.from_dict({json.loads(self._config_json)!r}), "
            f"policy=helion.CuteStructuralPolicy.from_dict({self.policy.to_dict()!r}))"
        )


def decode_structural_policy(value: object) -> CuteStructuralPolicy:
    if not isinstance(value, Mapping):
        raise StructuralPolicyError("Expected a recorded CuTe structural policy")
    try:
        return CuteStructuralPolicy.from_dict(value)
    except (TypeError, ValueError) as error:
        raise StructuralPolicyError(str(error)) from error


def require_same_structural_policy(
    actual: CuteStructuralPolicy | None,
    expected: CuteStructuralPolicy | None,
    *,
    context: str,
) -> None:
    if actual != expected:
        raise StructuralPolicyError(
            f"{context} has a different CuTe structural policy. "
            "Select its policy before binding a new Kernel; an existing bound "
            "kernel cannot change config schemas."
        )


def bound_structural_policy(
    kernel: object, *, captured: bool = False
) -> CuteStructuralPolicy | None:
    """Read opt-in cache identity or the policy captured when a bound was traced."""
    from .kernel import BoundKernel

    if not isinstance(kernel, BoundKernel):
        return None
    return (
        kernel._structural_policy if captured else kernel.kernel.cute_structural_policy
    )


def config_artifact_json(config: Config, policy: CuteStructuralPolicy | None) -> str:
    """Policy-aware identity without changing legacy Config serialization."""
    if policy is None:
        return json.dumps(dict(config), sort_keys=True)
    return CuteStructuralConfig(config, policy).to_json()


def _seed_sequence(value: object) -> tuple[ConfigLike, ...]:
    if value is None:
        return ()
    if isinstance(value, (Config, dict, CuteStructuralConfig)):
        return (value,)
    return tuple(cast("Sequence[ConfigLike]", value))


def resolve_cute_structural_policy(
    settings: Settings,
    *,
    requested: CuteStructuralPolicy | None = None,
    automatic: CuteStructuralPolicy | None = None,
    configs: Sequence[ConfigLike] = (),
    seed_configs: Sequence[ConfigLike] = (),
    overrides: Mapping[str, object] | CuteStructuralConfig | None = None,
) -> CuteStructuralPolicy:
    """Resolve before binding; this function never changes the default policy.

    ``automatic`` is an explicit proposal supplied by the caller, not a new
    default. Unversioned positional configs/seeds/overrides retain the captured
    settings instead. Environment and explicit booleans always override an
    automatic proposal. A recorded/requested policy rejects conflicting values.
    """
    if settings.backend != "cute":
        raise StructuralPolicyError("CuTe structural policies require backend='cute'")
    if requested is not None and not isinstance(requested, CuteStructuralPolicy):
        raise TypeError("requested must be a CuteStructuralPolicy")
    values = [*configs, *_seed_sequence(settings.autotune_seed_configs), *seed_configs]
    effective_overrides = (
        settings.autotune_config_overrides if overrides is None else overrides
    )
    recorded = [
        value.policy for value in values if isinstance(value, CuteStructuralConfig)
    ]
    if isinstance(effective_overrides, CuteStructuralConfig):
        recorded.append(effective_overrides.policy)
    selected = requested or (recorded[0] if recorded else None)
    for policy in recorded:
        require_same_structural_policy(policy, selected, context="Config/seed envelope")
    if selected is not None:
        if not isinstance(selected, CuteStructuralPolicy):
            raise TypeError("requested must be a CuteStructuralPolicy")
        for name in CUTE_STRUCTURAL_SETTINGS:
            if getattr(settings.cute_structural_origins, name) != "default" and getattr(
                settings, name
            ) != getattr(selected, name):
                raise StructuralPolicyError(
                    f"Recorded CuTe structural policy conflicts with {name}"
                )
        return selected
    if automatic is not None and not isinstance(automatic, CuteStructuralPolicy):
        raise TypeError("automatic must be a CuteStructuralPolicy")
    if automatic is None or values or effective_overrides or overrides is not None:
        return settings.get_cute_structural_policy()
    return CuteStructuralPolicy(
        **{
            name: (
                getattr(automatic, name)
                if getattr(settings.cute_structural_origins, name) == "default"
                else getattr(settings, name)
            )
            for name in CUTE_STRUCTURAL_SETTINGS
        }
    )


def prepare_structural_policy(
    settings: Settings,
    configs: Sequence[ConfigLike],
    requested: CuteStructuralPolicy | None,
) -> tuple[Settings, Sequence[Config | dict[str, object]], CuteStructuralPolicy | None]:
    """Consume an explicit policy envelope before any binding/config interpretation.

    The ordinary unversioned path returns the original Settings and configs.
    A policy-bearing declaration deliberately derives settings and detached
    config snapshots; one Kernel owns exactly that selected policy.
    """
    # Ordinary custom seed sequences must not be iterated earlier than before.
    # Implicit policy selection from seeds uses a list/tuple or a single record.
    # A custom seed sequence can instead accompany an explicit policy/config.
    seed_value = settings.autotune_seed_configs
    seeds = (
        _seed_sequence(seed_value)
        if isinstance(seed_value, (list, tuple, CuteStructuralConfig))
        else ()
    )
    # Kernel already iterates this sequence once when constructing its config
    # list. Snapshot a custom sequence once so inspecting a new envelope does
    # not duplicate user iteration or interpret ordinary config knobs early.
    if not isinstance(configs, (list, tuple)):
        configs = [value for value in configs]  # noqa: C416 -- avoid user __len__/__length_hint__ calls
    inspected_configs = configs
    has_record = any(
        isinstance(value, CuteStructuralConfig)
        for value in (*inspected_configs, *seeds)
    )
    has_record = has_record or isinstance(
        settings.autotune_config_overrides, CuteStructuralConfig
    )
    if requested is None and not has_record:
        return settings, cast("Sequence[Config | dict[str, object]]", configs), None
    if seed_value is not None and not isinstance(
        seed_value, (Config, dict, CuteStructuralConfig, list, tuple)
    ):
        # A new policy declaration owns a snapshot. Iterating an arbitrary seed
        # provider during resolution and again during tuning could drop seeds
        # or observe a different policy on the second iteration.
        settings = settings.copy(autotune_seed_configs=_seed_sequence(seed_value))
    seeds = _seed_sequence(settings.autotune_seed_configs)
    policy = resolve_cute_structural_policy(
        settings, requested=requested, configs=configs
    )
    resolved = settings.copy(
        **{name: getattr(policy, name) for name in CUTE_STRUCTURAL_SETTINGS}
    )
    if any(isinstance(value, CuteStructuralConfig) for value in seeds):
        resolved.autotune_seed_configs = tuple(
            value.config if isinstance(value, CuteStructuralConfig) else value
            for value in seeds
        )
    if isinstance(settings.autotune_config_overrides, CuteStructuralConfig):
        resolved.autotune_config_overrides = (
            settings.autotune_config_overrides.config.config
        )
    return (
        resolved,
        tuple(
            value.config if isinstance(value, CuteStructuralConfig) else value
            for value in configs
        ),
        policy,
    )


def select_structural_policy(
    settings: Settings,
    configs: Sequence[ConfigLike],
    requested: CuteStructuralPolicyRequest = DEFAULT_STRUCTURAL_POLICY,
) -> tuple[Settings, Sequence[Config | dict[str, object]], CuteStructuralPolicy | None]:
    """Choose a config schema before tracing an ordinary CuTe search.

    Recorded declarations select their exact policy first. Automatic selection
    only applies when no unversioned declaration or opaque config callback can
    depend on the old axes. Explicit None preserves the legacy namespace.
    """
    # Settings and reference mode import compiler/runtime modules themselves;
    # resolve these dependencies only after the public constructor is entered.
    from .ref_mode import is_ref_mode_enabled
    from .settings import default_autotuner_fn

    if not (
        requested is DEFAULT_STRUCTURAL_POLICY
        or requested is None
        or isinstance(requested, CuteStructuralPolicy)
        or (isinstance(requested, str) and requested == "auto")
    ):
        raise TypeError("cute_structural_policy must be a policy, 'auto', or None")
    prepared, configs, recorded = prepare_structural_policy(
        settings,
        configs,
        requested if isinstance(requested, CuteStructuralPolicy) else None,
    )
    if recorded is not None:
        return prepared, configs, recorded
    if requested is None:
        return settings, configs, None
    if settings.backend != "cute":
        if requested == "auto":
            raise StructuralPolicyError("Explicit auto requires backend='cute'")
        return settings, configs, None
    if requested is DEFAULT_STRUCTURAL_POLICY and (
        settings.autotune_cache == "AOTAutotuneCache"
        or settings.autotune_effort == "none"
        or is_ref_mode_enabled(settings)
    ):
        return settings, configs, None

    if configs:
        return settings, configs, None
    seeds = settings.autotune_seed_configs
    if seeds is not None and not (
        type(seeds) in (list, tuple) and len(cast("Sequence[ConfigLike]", seeds)) == 0
    ):
        # A custom provider must remain unconsumed until ordinary autotuning.
        return settings, configs, None
    overrides = settings.autotune_config_overrides
    if type(overrides) is not dict or overrides:
        return settings, configs, None
    if (
        settings.autotuner_fn is not default_autotuner_fn
        or settings.autotune_config_filter is not None
    ):
        return settings, configs, None

    proposal = CuteStructuralPolicy(**dict.fromkeys(CUTE_STRUCTURAL_SETTINGS, True))
    policy = resolve_cute_structural_policy(settings, automatic=proposal)
    return prepare_structural_policy(settings, configs, policy)
