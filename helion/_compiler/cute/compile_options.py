"""Config-wide compiler options shared by ordinary and auxiliary launches."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

from .tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ...runtime.config import Config


CUTE_COMPILER_PIPELINE_KEY = "cute_compiler_pipeline"
CUTE_COMPILER_PIPELINE_OPTIONS = {"default": "", "opt2": "--opt-level 2"}
CUTE_COMPILER_PIPELINE_CHOICES = tuple(CUTE_COMPILER_PIPELINE_OPTIONS)


class _ConfigCompileOption(NamedTuple):
    key: str
    format_value: Callable[[object], str]
    ptxas: bool = False


_CONFIG_COMPILE_OPTIONS = (
    _ConfigCompileOption(
        CUTE_COMPILER_PIPELINE_KEY,
        lambda value: CUTE_COMPILER_PIPELINE_OPTIONS.get(cast("str", value), ""),
    ),
    _ConfigCompileOption(
        "advanced_controls_file",
        lambda value: f"--apply-controls {value}" if value else "",
        ptxas=True,
    ),
    _ConfigCompileOption(
        TCGEN05_CUBIN_LINEINFO_CONFIG_KEY,
        lambda value: "--generate-line-info" if value is True else "",
    ),
    _ConfigCompileOption(
        TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY,
        lambda value: "--enable-tvm-ffi" if value is True else "",
    ),
)
CONFIG_COMPILE_OPTION_KEYS = tuple(option.key for option in _CONFIG_COMPILE_OPTIONS)


def config_compile_options(config: Config, *, ptxas_options: Sequence[str] = ()) -> str:
    values = [
        (option, option.format_value(config.get(option.key)))
        for option in _CONFIG_COMPILE_OPTIONS
    ]
    ptxas = [
        *ptxas_options,
        *(value for option, value in values if option.ptxas and value),
    ]
    options = []
    for option, value in values:
        if option.ptxas:
            if ptxas:
                options.append(f"--ptxas-options='{' '.join(ptxas)}'")
                ptxas.clear()
        elif value:
            options.append(value)
    return " ".join(options)


def compile_options_keyword(options: str) -> str:
    """Append a literal launch keyword, retaining the legacy call when empty."""
    return f", cute_compile_options={options!r}" if options else ""
