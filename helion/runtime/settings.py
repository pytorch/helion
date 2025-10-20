from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import time
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol
from typing import Sequence
from typing import TypeVar
from typing import cast

import torch
from torch._environment import is_fbcode

from .. import exc
from ..autotuner.effort_profile import AutotuneEffort
from ..autotuner.effort_profile import get_effort_profile
from .ref_mode import RefMode

if TYPE_CHECKING:
    from ..autotuner.base_search import BaseAutotuner
    from .kernel import BoundKernel

    _T = TypeVar("_T")

    class AutotunerFunction(Protocol):
        def __call__(
            self, bound_kernel: BoundKernel, args: Sequence[object], **kwargs: object
        ) -> BaseAutotuner: ...


DotPrecision = Literal["tf32", "tf32x3", "ieee"]
PrecompileMode = Literal["spawn", "fork"] | None
_TRUE_LITERALS = frozenset({"1", "true", "yes", "on"})
_FALSE_LITERALS = frozenset({"0", "false", "no", "off"})


def _resolve_warning_name(name: str) -> type[exc.BaseWarning]:
    attr = name.strip()
    if not attr:
        raise ValueError("HELION_IGNORE_WARNINGS entries must be non-empty names")
    try:
        warning_cls = getattr(exc, attr)
    except AttributeError as err:
        raise ValueError(
            f"HELION_IGNORE_WARNINGS entry {name!r} is not a warning defined in helion.exc"
        ) from err
    if not isinstance(warning_cls, type) or not issubclass(
        warning_cls, exc.BaseWarning
    ):
        raise ValueError(
            f"HELION_IGNORE_WARNINGS entry {name!r} does not refer to a helion.exc.BaseWarning subclass"
        )
    return warning_cls


def _get_ignore_warnings() -> list[type[exc.BaseWarning]]:
    value = os.environ.get("HELION_IGNORE_WARNINGS")
    if not value:
        return []
    result: list[type[exc.BaseWarning]] = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        result.append(_resolve_warning_name(entry))
    return result


def _env_get_optional_int(var_name: str) -> int | None:
    value = os.environ.get(var_name)
    if value is None or (value := value.strip()) == "":
        return None
    try:
        parsed = int(value)
    except ValueError as err:
        raise ValueError(f"{var_name} must be an integer, got {value!r}") from err
    return parsed


def _env_get_int(var_name: str, default: int) -> int:
    result = _env_get_optional_int(var_name)
    return default if result is None else result


def _env_get_optional_float(var_name: str) -> float | None:
    value = os.environ.get(var_name)
    if value is None or (value := value.strip()) == "":
        return None
    try:
        return float(value)
    except ValueError as err:
        raise ValueError(f"{var_name} must be a float, got {value!r}") from err


def _env_get_bool(var_name: str, default: bool) -> bool:
    value = os.environ.get(var_name)
    if value is None or (value := value.strip()) == "":
        return default
    lowered = value.lower()
    if lowered in _TRUE_LITERALS:
        return True
    if lowered in _FALSE_LITERALS:
        return False
    raise ValueError(
        f"{var_name} must be one of {_TRUE_LITERALS | _FALSE_LITERALS}, got {value!r}"
    )


def _env_get_literal(
    var_name: str,
    default: _T,
    *,
    mapping: dict[str, object],
) -> _T:
    value = os.environ.get(var_name)
    if value is None:
        return default
    value = value.strip()
    if value in mapping:
        return cast("_T", mapping[value])
    if value == "":
        return default
    raise ValueError(
        f"{var_name} must be one of {', '.join(sorted(mapping))}, got {value!r}"
    )


def _get_index_dtype() -> torch.dtype:
    value = os.environ.get("HELION_INDEX_DTYPE")
    if value is None or (token := value.strip()) == "":
        return torch.int32
    try:
        dtype = getattr(torch, token)
    except AttributeError as err:
        raise ValueError(
            f"HELION_INDEX_DTYPE must map to a torch dtype attribute, got {value!r}"
        ) from err
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"HELION_INDEX_DTYPE {value!r} is not a torch.dtype")
    return dtype


def _get_autotune_log_level() -> int:
    value = os.environ.get("HELION_AUTOTUNE_LOG_LEVEL")
    if value is None or value.strip() == "":
        return logging.INFO
    text = value.strip()
    if text.lstrip("+-").isdigit():
        return int(text)
    upper = text.upper()
    level = logging.getLevelName(upper)
    if isinstance(level, int):
        return level
    raise ValueError(
        f"HELION_AUTOTUNE_LOG_LEVEL must be an integer or logging level name, got {value!r}"
    )


def _get_autotune_config_overrides() -> dict[str, object]:
    value = os.environ.get("HELION_AUTOTUNE_CONFIG_OVERRIDES")
    if not value or (value := value.strip()) == "":
        return {}
    if not value.startswith("{") and os.path.exists(value):
        value = open(value).read()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as err:
        raise ValueError(
            "HELION_AUTOTUNE_CONFIG_OVERRIDES must be valid JSON mapping of config keys to values"
        ) from err
    if not isinstance(parsed, dict):
        raise ValueError(
            "HELION_AUTOTUNE_CONFIG_OVERRIDES must decode to a JSON dictionary"
        )
    return parsed


def default_autotuner_fn(
    bound_kernel: BoundKernel, args: Sequence[object], **kwargs: object
) -> BaseAutotuner:
    from ..autotuner import LocalAutotuneCache
    from ..autotuner import search_algorithms

    autotuner_name = os.environ.get("HELION_AUTOTUNER", "PatternSearch")
    autotuner_cls = search_algorithms.get(autotuner_name)
    if autotuner_cls is None:
        raise ValueError(
            f"Unknown HELION_AUTOTUNER value: {autotuner_name}, valid options are: "
            f"{', '.join(search_algorithms.keys())}"
        )

    # Use autotune_max_generations from settings if kwarg is not explicitly provided
    if autotuner_name in ("PatternSearch", "DifferentialEvolutionSearch"):
        if bound_kernel.settings.autotune_max_generations is not None:
            kwargs.setdefault(
                "max_generations", bound_kernel.settings.autotune_max_generations
            )

    profile = get_effort_profile(bound_kernel.settings.autotune_effort)

    if autotuner_cls.__name__ == "PatternSearch":
        assert profile.pattern_search is not None
        kwargs.setdefault(
            "initial_population", profile.pattern_search.initial_population
        )
        kwargs.setdefault("copies", profile.pattern_search.copies)
        kwargs.setdefault("max_generations", profile.pattern_search.max_generations)
    elif autotuner_cls.__name__ == "DifferentialEvolutionSearch":
        assert profile.differential_evolution is not None
        kwargs.setdefault(
            "population_size", profile.differential_evolution.population_size
        )
        kwargs.setdefault(
            "max_generations", profile.differential_evolution.max_generations
        )
    elif autotuner_cls.__name__ == "RandomSearch":
        assert profile.random_search is not None
        kwargs.setdefault("count", profile.random_search.count)

    return LocalAutotuneCache(autotuner_cls(bound_kernel, args, **kwargs))  # pyright: ignore[reportArgumentType]


def _get_autotune_random_seed() -> int:
    if (seed := _env_get_optional_int("HELION_AUTOTUNE_RANDOM_SEED")) is not None:
        return seed
    return int(time.time() * 1000) % 2**32


def _get_ref_mode() -> RefMode:
    interpret = _env_get_bool("HELION_INTERPRET", False)
    return RefMode.EAGER if interpret else RefMode.OFF


@dataclasses.dataclass
class _Settings:
    # see __slots__ below for the doc strings that show up in help(Settings)
    ignore_warnings: list[type[exc.BaseWarning]] = dataclasses.field(
        default_factory=_get_ignore_warnings
    )
    index_dtype: torch.dtype = dataclasses.field(default_factory=_get_index_dtype)
    dot_precision: DotPrecision = dataclasses.field(
        default_factory=functools.partial(
            _env_get_literal,
            "TRITON_F32_DEFAULT",
            cast("DotPrecision", "tf32"),
            mapping={k: k for k in ("tf32", "tf32x3", "ieee")},
        )
    )  # pyright: ignore[reportAssignmentType]
    static_shapes: bool = dataclasses.field(
        default_factory=functools.partial(_env_get_bool, "HELION_STATIC_SHAPES", True)
    )
    autotune_log_level: int = dataclasses.field(default_factory=_get_autotune_log_level)
    autotune_compile_timeout: int = dataclasses.field(
        default_factory=functools.partial(
            _env_get_int, "HELION_AUTOTUNE_COMPILE_TIMEOUT", 60
        )
    )
    autotune_precompile: PrecompileMode = dataclasses.field(
        default_factory=functools.partial(
            _env_get_literal,
            "HELION_AUTOTUNE_PRECOMPILE",
            cast("PrecompileMode", "fork"),
            mapping={
                "spawn": "spawn",
                "fork": "fork",
                "": None,
                "0": None,
            },
        )
    )  # pyright: ignore[reportAssignmentType]
    autotune_precompile_jobs: int | None = dataclasses.field(
        default_factory=functools.partial(
            _env_get_optional_int,
            "HELION_AUTOTUNE_PRECOMPILE_JOBS",
        )
    )
    autotune_random_seed: int = dataclasses.field(
        default_factory=_get_autotune_random_seed
    )
    autotune_accuracy_check: bool = dataclasses.field(
        default_factory=functools.partial(
            _env_get_bool, "HELION_AUTOTUNE_ACCURACY_CHECK", True
        )
    )
    autotune_rebenchmark_threshold: float | None = dataclasses.field(
        default_factory=functools.partial(
            _env_get_optional_float,
            "HELION_REBENCHMARK_THRESHOLD",
        )
    )
    autotune_progress_bar: bool = dataclasses.field(
        default_factory=functools.partial(
            _env_get_bool, "HELION_AUTOTUNE_PROGRESS_BAR", True
        )
    )
    autotune_max_generations: int | None = dataclasses.field(
        default_factory=functools.partial(
            _env_get_optional_int,
            "HELION_AUTOTUNE_MAX_GENERATIONS",
        )
    )
    autotune_ignore_errors: bool = dataclasses.field(
        default_factory=functools.partial(
            _env_get_bool, "HELION_AUTOTUNE_IGNORE_ERRORS", False
        )
    )
    print_output_code: bool = dataclasses.field(
        default_factory=functools.partial(
            _env_get_bool, "HELION_PRINT_OUTPUT_CODE", False
        )
    )
    output_origin_lines: bool = dataclasses.field(
        default_factory=functools.partial(
            _env_get_bool, "HELION_OUTPUT_ORIGIN_LINES", True
        )
    )
    force_autotune: bool = dataclasses.field(
        default_factory=functools.partial(_env_get_bool, "HELION_FORCE_AUTOTUNE", False)
    )
    autotune_config_overrides: dict[str, object] = dataclasses.field(
        default_factory=_get_autotune_config_overrides
    )
    autotune_effort: AutotuneEffort = dataclasses.field(
        default_factory=functools.partial(
            _env_get_literal,
            "HELION_AUTOTUNE_EFFORT",
            cast("AutotuneEffort", "full"),
            mapping={key: key for key in ("none", "quick", "full")},
        )
    )  # pyright: ignore[reportAssignmentType]
    allow_warp_specialize: bool = dataclasses.field(
        default_factory=functools.partial(
            _env_get_bool, "HELION_ALLOW_WARP_SPECIALIZE", True
        )
    )
    debug_dtype_asserts: bool = dataclasses.field(
        default_factory=functools.partial(
            _env_get_bool, "HELION_DEBUG_DTYPE_ASSERTS", False
        )
    )
    ref_mode: RefMode = dataclasses.field(default_factory=_get_ref_mode)
    autotuner_fn: AutotunerFunction = default_autotuner_fn


class Settings(_Settings):
    """
    Settings can be passed to hl.kernel as kwargs and control the behavior of the
    compilation process. Unlike a Config, settings are not auto-tuned and set by the user.
    """

    __slots__ = {
        "ignore_warnings": (
            "Subtypes of exc.BaseWarning to ignore when compiling. "
            "Set HELION_IGNORE_WARNINGS=WarningA,WarningB (names from helion.exc) to configure via env."
        ),
        "index_dtype": (
            "The dtype to use for index variables. Default is torch.int32. "
            "Override with HELION_INDEX_DTYPE=torch.int64, etc."
        ),
        "dot_precision": "Precision for dot products, see `triton.language.dot`. Can be 'tf32', 'tf32x3', or 'ieee'.",
        "static_shapes": (
            "If True, use static shapes for all tensors. This is a performance optimization. "
            "Set HELION_STATIC_SHAPES=0 to disable."
        ),
        "autotune_log_level": (
            "Log level for autotuning using Python logging levels. Default is logging.INFO. "
            "Use HELION_AUTOTUNE_LOG_LEVEL to override or set 0 to disable output."
        ),
        "autotune_compile_timeout": "Timeout for Triton compilation in seconds used for autotuning. Default is 60 seconds.",
        "autotune_precompile": "Autotuner precompile mode: 'fork', 'spawn', or falsy/None to disable. Defaults to 'fork' on non-Windows platforms.",
        "autotune_precompile_jobs": "Maximum concurrent Triton precompile processes, default to cpu count.",
        "autotune_random_seed": "Seed used for autotuner random number generation. Defaults to HELION_AUTOTUNE_RANDOM_SEED or a time-based seed.",
        "autotune_accuracy_check": "If True, validate candidate configs against the baseline kernel output before accepting them during autotuning.",
        "autotune_rebenchmark_threshold": "If a config is within threshold*best_perf, re-benchmark it to avoid outliers. Defaults to effort profile value. Set HELION_REBENCHMARK_THRESHOLD to override.",
        "autotune_progress_bar": "If True, show progress bar during autotuning. Default is True. Set HELION_AUTOTUNE_PROGRESS_BAR=0 to disable.",
        "autotune_max_generations": "Override the maximum number of generations for Pattern Search and Differential Evolution Search autotuning algorithms with HELION_AUTOTUNE_MAX_GENERATIONS=N or @helion.kernel(autotune_max_generations=N).",
        "autotune_ignore_errors": (
            "If True, skip logging and raising autotune errors. "
            "Set HELION_AUTOTUNE_IGNORE_ERRORS=1 to enable globally."
        ),
        "print_output_code": "If True, print the output code of the kernel to stderr.",
        "output_origin_lines": (
            "If True, annotate generated Triton code with source-origin comments. "
            "Set HELION_OUTPUT_ORIGIN_LINES=0 to disable."
        ),
        "force_autotune": "If True, force autotuning even if a config is provided.",
        "autotune_config_overrides": (
            "Dictionary of config key/value pairs forced during autotuning. "
            "Accepts HELION_AUTOTUNE_CONFIG_OVERRIDES='{\"num_warps\":4}'."
        ),
        "allow_warp_specialize": "If True, allow warp specialization for tl.range calls on CUDA devices.",
        "debug_dtype_asserts": "If True, emit tl.static_assert checks for dtype after each device node.",
        "ref_mode": "Reference mode for kernel execution. Can be RefMode.OFF or RefMode.EAGER.",
        "autotuner_fn": (
            "Function to create an autotuner. "
            "Override by passing a callable to @helion.kernel(..., autotuner_fn=...)."
        ),
        "autotune_effort": "Autotuning effort preset. One of 'none', 'quick', 'full'.",
    }

    def __init__(self, **settings: object) -> None:
        """
        Initialize the Settings object with the provided dictionary of settings.
        """

        super().__init__(**settings)  # pyright: ignore[reportArgumentType]

        self._check_ref_eager_mode_before_print_output_code()

    def to_dict(self) -> dict[str, object]:
        """
        Convert the Settings object to a dictionary.

        Returns:
            dict[str, object]: A dictionary representation of the Settings object.
        """

        def shallow_copy(x: object) -> object:
            if isinstance(x, (list, dict)):
                return x.copy()
            return x

        # Only include fields that are meant to be public (repr=True)
        public_fields = {f.name for f in dataclasses.fields(self) if f.repr}
        return {
            k: shallow_copy(v)
            for k, v in dataclasses.asdict(self).items()
            if k in public_fields
        }

    def check_autotuning_disabled(self) -> None:
        msg = None
        if os.environ.get("HELION_DISALLOW_AUTOTUNING", "0") == "1":
            msg = "by HELION_DISALLOW_AUTOTUNING=1"
        if is_fbcode():
            from aiplatform.runtime_environment.runtime_environment_pybind import (  # type: ignore[import-untyped]
                RuntimeEnvironment,
            )

            if RuntimeEnvironment().get_mast_job_name() is not None:
                msg = "because autotuning is not allowed in MAST environment"
        if msg:
            raise exc.AutotuningDisallowedInEnvironment(msg)

    def get_rebenchmark_threshold(self) -> float:
        """
        Get the effective rebenchmark threshold.
        Uses the explicit setting if provided, otherwise falls back to the effort profile default.

        Returns:
            float: The rebenchmark threshold value.
        """
        if self.autotune_rebenchmark_threshold is not None:
            return self.autotune_rebenchmark_threshold

        from ..autotuner.effort_profile import get_effort_profile

        return get_effort_profile(self.autotune_effort).rebenchmark_threshold

    def _check_ref_eager_mode_before_print_output_code(self) -> None:
        """
        Check if ref eager mode is enabled before printing output code. If ref eager mode is enabled, raise an error.
        """
        if self.ref_mode == RefMode.EAGER and self.print_output_code:
            raise exc.RefEagerModeCodePrintError
