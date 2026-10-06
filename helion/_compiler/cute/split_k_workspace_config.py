from __future__ import annotations

from ...exc import InvalidConfig

WORKSPACE_KEY = "cute_split_k_workspace"
STAGES_KEY = "cute_split_k_stages"
WORKSPACE_CONFIG_KEYS = frozenset((WORKSPACE_KEY, STAGES_KEY))


def normalize_workspace_config(
    config: dict[str, object], *, available: bool, fix_invalid: bool
) -> None:
    if not available:
        if (
            config.get(WORKSPACE_KEY, False) is not False
            or config.get(STAGES_KEY, 2) != 2
        ) and not fix_invalid:
            raise InvalidConfig(
                "native split-K workspace requires a proved private FP32 reduction"
            )
        for key in WORKSPACE_CONFIG_KEYS:
            config.pop(key, None)
        return
    enabled = config.setdefault(WORKSPACE_KEY, False)
    stages = config.setdefault(STAGES_KEY, 2)
    if type(enabled) is not bool:
        if not fix_invalid:
            raise InvalidConfig(f"{WORKSPACE_KEY} must be a boolean")
        config[WORKSPACE_KEY] = False
    if type(stages) is not int or not 1 <= stages <= 16:
        if not fix_invalid:
            raise InvalidConfig(f"{STAGES_KEY} must be an integer in [1, 16]")
        config[STAGES_KEY] = 2
