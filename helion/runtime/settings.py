from __future__ import annotations

import dataclasses
import logging
import os
import sys
import threading
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol
from typing import cast

import torch

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from helion import exc

    class _TLS(Protocol):
        default_settings: Settings | None


_tls: _TLS = cast("_TLS", threading.local())


def set_default_settings(settings: Settings) -> AbstractContextManager[None, None]:
    """
    Set the default settings for the current thread and return a context manager
    that restores the previous settings upon exit.

    :param settings: The Settings object to set as the default.
    :return: A context manager that restores the previous settings upon exit.
    """
    prior = getattr(_tls, "default_settings", None)
    _tls.default_settings = settings

    class _RestoreContext:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *args: object) -> None:
            _tls.default_settings = prior

    return _RestoreContext()


@dataclasses.dataclass
class _Settings:
    # see __slots__ below for the doc strings that show up in help(Settings)
    ignore_warnings: list[type[exc.BaseWarning]] = dataclasses.field(
        default_factory=list
    )
    index_dtype: torch.dtype = torch.int32
    dot_precision: Literal["tf32", "tf32x3", "ieee"] = "tf32"
    static_shapes: bool = False
    use_default_config: bool = False
    autotune_log_level: int = logging.INFO
    autotune_compile_timeout: int = 60
    autotune_precompile: bool = sys.platform != "win32"
    print_output_code: bool = os.environ.get("HELION_PRINT_OUTPUT_CODE", "0") == "1"


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
        "autotune_log_level": "Log level for autotuning. 0 = no logging, 1 = only final config, 2 = default, 3 = verbose.",
        "autotune_compile_timeout": "Timeout for Triton compilation in seconds used for autotuning. Default is 60 seconds.",
        "autotune_precompile": "If True, precompile the kernel before autotuning. Requires fork-safe environment.",
        "print_output_code": "If True, print the output code of the kernel to stderr.",
    }
    assert __slots__.keys() == {field.name for field in dataclasses.fields(_Settings)}

    def __init__(self, **settings: object) -> None:
        """
        Initialize the Settings object with the provided dictionary of settings.
        If no settings are provided, the default settings are used (see `set_default_settings`).

        :param settings: Keyword arguments representing various settings.
        """
        if defaults := getattr(_tls, "default_settings", None):
            settings = {**defaults.to_dict(), **settings}
        # pyre-ignore[6]
        super().__init__(**settings)
        if os.getenv("HELION_USE_DEFAULT_CONFIG") == "1":
            self.use_default_config: bool = True

    def to_dict(self) -> dict[str, object]:
        """
        Convert the Settings object to a dictionary.

        :return: A dictionary representation of the Settings object.
        """

        def shallow_copy(x: object) -> object:
            if isinstance(x, (list, dict)):
                return x.copy()
            return x

        return {k: shallow_copy(v) for k, v in dataclasses.asdict(self).items()}

    @staticmethod
    def default() -> Settings:
        """
        Get the default Settings object. If no default settings are set, create a new one.

        :return: The default Settings object.
        """
        result = getattr(_tls, "default_settings", None)
        if result is None:
            _tls.default_settings = result = Settings()
        return result
