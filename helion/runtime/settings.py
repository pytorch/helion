from __future__ import annotations

import dataclasses
import logging
import os
import sys
import threading
import time
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol
from typing import Sequence
from typing import cast

import torch
from torch._environment import is_fbcode

from helion import exc
from helion.runtime.ref_mode import RefMode

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from ..autotuner.base_search import BaseAutotuner
    from .kernel import BoundKernel

    class _TLS(Protocol):
        default_settings: Settings | None

    class AutotunerFunction(Protocol):
        def __call__(
            self, bound_kernel: BoundKernel, args: Sequence[object], **kwargs: object
        ) -> BaseAutotuner: ...


_tls: _TLS = cast("_TLS", threading.local())


def set_default_settings(settings: Settings) -> AbstractContextManager[None, None]:
    """
    Set the default settings for the current thread and return a context manager
    that restores the previous settings upon exit.

    Args:
        settings: The Settings object to set as the default.

    Returns:
        AbstractContextManager[None, None]: A context manager that restores the previous settings upon exit.
    """
    prior = getattr(_tls, "default_settings", None)
    _tls.default_settings = settings

    class _RestoreContext:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *args: object) -> None:
            _tls.default_settings = prior

    return _RestoreContext()


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
    return LocalAutotuneCache(autotuner_cls(bound_kernel, args, **kwargs))  # pyright: ignore[reportArgumentType]


def _get_autotune_random_seed() -> int:
    value = os.environ.get("HELION_AUTOTUNE_RANDOM_SEED")
    if value is not None:
        return int(value)
    return int(time.time() * 1000) % 2**32


@dataclasses.dataclass
class _Settings:
    # see __slots__ below for the doc strings that show up in help(Settings)
    ignore_warnings: list[type[exc.BaseWarning]] = dataclasses.field(
        default_factory=list
    )
    index_dtype: torch.dtype = torch.int32
    dot_precision: Literal["tf32", "tf32x3", "ieee"] = cast(
        "Literal['tf32', 'tf32x3', 'ieee']",
        os.environ.get("TRITON_F32_DEFAULT", "tf32"),
    )
    static_shapes: bool = False
    use_default_config: bool = os.environ.get("HELION_USE_DEFAULT_CONFIG", "0") == "1"
    autotune_log_level: int = logging.INFO
    autotune_compile_timeout: int = int(
        os.environ.get("HELION_AUTOTUNE_COMPILE_TIMEOUT", "60")
    )
    autotune_precompile: bool = sys.platform != "win32"
    autotune_precompile_jobs: int | None = None
    autotune_random_seed: int = dataclasses.field(
        default_factory=_get_autotune_random_seed
    )
    autotune_accuracy_check: bool = (
        os.environ.get("HELION_AUTOTUNE_ACCURACY_CHECK", "1") == "1"
    )
    autotune_rebenchmark_threshold: float = float(
        os.environ.get("HELION_REBENCHMARK_THRESHOLD", "1.5")
    )
    autotune_progress_bar: bool = (
        os.environ.get("HELION_AUTOTUNE_PROGRESS_BAR", "1") == "1"
    )
    print_output_code: bool = os.environ.get("HELION_PRINT_OUTPUT_CODE", "0") == "1"
    force_autotune: bool = os.environ.get("HELION_FORCE_AUTOTUNE", "0") == "1"
    allow_warp_specialize: bool = (
        os.environ.get("HELION_ALLOW_WARP_SPECIALIZE", "1") == "1"
    )
    debug_dtype_asserts: bool = os.environ.get("HELION_DEBUG_DTYPE_ASSERTS", "0") == "1"
    ref_mode: RefMode = (
        RefMode.EAGER if os.environ.get("HELION_INTERPRET", "") == "1" else RefMode.OFF
    )
    autotuner_fn: AutotunerFunction = default_autotuner_fn


class Settings(_Settings):
    """
    Settings can be passed to hl.kernel as kwargs and control the behavior of the
    compilation process. Unlike a Config, settings are not auto-tuned and set by the user.
    """

    __slots__: dict[str, str] = {
        "ignore_warnings": "Subtypes of exc.BaseWarning to ignore when compiling.",
        "index_dtype": "The dtype to use for index variables. Default is torch.int32.",
        "dot_precision": "Precision for dot products, see `triton.language.dot`. Can be 'tf32', 'tf32x3', or 'ieee'.",
        "static_shapes": "If True, use static shapes for all tensors. This is a performance optimization.",
        "use_default_config": "For development only, skips all autotuning and uses the default config (which may be slow).",
        "autotune_log_level": "Log level for autotuning using Python logging levels. Default is logging.INFO. Use 0 to disable all output.",
        "autotune_compile_timeout": "Timeout for Triton compilation in seconds used for autotuning. Default is 60 seconds.",
        "autotune_precompile": "If True, precompile the kernel before autotuning. Requires fork-safe environment.",
        "autotune_precompile_jobs": "Maximum concurrent Triton precompile processes, default to cpu count.",
        "autotune_random_seed": "Seed used for autotuner random number generation. Defaults to HELION_AUTOTUNE_RANDOM_SEED or a time-based seed.",
        "autotune_accuracy_check": "If True, validate candidate configs against the baseline kernel output before accepting them during autotuning.",
        "autotune_rebenchmark_threshold": "If a config is within threshold*best_perf, re-benchmark it to avoid outliers. Default is 1.5x.  Set to <1 to disable.",
        "autotune_progress_bar": "If True, show progress bar during autotuning. Default is True. Set HELION_AUTOTUNE_PROGRESS_BAR=0 to disable.",
        "print_output_code": "If True, print the output code of the kernel to stderr.",
        "force_autotune": "If True, force autotuning even if a config is provided.",
        "allow_warp_specialize": "If True, allow warp specialization for tl.range calls on CUDA devices.",
        "debug_dtype_asserts": "If True, emit tl.static_assert checks for dtype after each device node.",
        "ref_mode": "Reference mode for kernel execution. Can be RefMode.OFF or RefMode.EAGER.",
        "autotuner_fn": "Function to create an autotuner",
    }
    assert __slots__.keys() == {field.name for field in dataclasses.fields(_Settings)}

    def __init__(self, **settings: object) -> None:
        """
        Initialize the Settings object with the provided dictionary of settings.
        If no settings are provided, the default settings are used (see `set_default_settings`).

        Args:
            settings: Keyword arguments representing various settings.
        """

        if defaults := getattr(_tls, "default_settings", None):
            settings = {**defaults.to_dict(), **settings}

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

        return {k: shallow_copy(v) for k, v in dataclasses.asdict(self).items()}

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

    def _check_ref_eager_mode_before_print_output_code(self) -> None:
        """
        Check if ref eager mode is enabled before printing output code. If ref eager mode is enabled, raise an error.
        """
        if self.ref_mode == RefMode.EAGER and self.print_output_code:
            raise exc.RefEagerModeCodePrintError

    @staticmethod
    def default() -> Settings:
        """
        Get the default Settings object. If no default settings are set, create a new one.

        Returns:
            Settings: The default Settings object.
        """
        result = getattr(_tls, "default_settings", None)
        if result is None:
            _tls.default_settings = result = Settings()
        return result
