from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import functools
import logging
import os
from typing import Callable
from typing import Generic
from typing import ParamSpec

LOG_ENV_VAR = "HELION_LOGS"

"""
Usage:

HELION_LOGS=+helion.runtime.kernel python test.py

will run test.py with helion.runtime.kernel logs enabled at logging.DEBUG level.
Aliases name groups of logs: ``all``; ``megakernel`` (a megakernel's schedule
and why each root barrier has no counter); ``timeline`` (also turns on
Settings.megakernel_timeline, under ./helion_timeline/, so autotuning a
megakernel raises: record a fixed or cached config).
"""


@dataclass
class LogRegistry:
    """
    This registry holds mappings for logging
    """

    # alias -> list of logs
    alias_map: dict[str, list[str]] = field(default_factory=dict)

    # log name -> log level
    log_levels: dict[str, int] = field(default_factory=dict)


# ``megakernel``: a megakernel's schedule, its root barriers and why, the
# transport of its cross-rank dependencies and why a ring falls back or an
# allocation keeps epoch words.
# ``timeline``: the paths megakernel timelines are written to; the setting reads
# the same value (Settings.megakernel_timeline) and records them.
_LOG_REGISTRY = LogRegistry(
    {
        "all": ["helion"],
        "megakernel": [
            "helion.megakernel",
            "helion._compiler.cross_loop_codegen",
            "helion._compiler.tile_dependency",
            "helion._compiler.triton.gluon_ring",
        ],
        "timeline": ["helion.runtime.timeline"],
    }
)


def log_entries(value: str) -> list[tuple[str, int]]:
    """The (log or alias, level) entries of a string like "foo.bar,+baz.fizz":
    [("foo.bar", logging.INFO), ("baz.fizz", logging.DEBUG)]."""
    entries = [e.strip() for e in value.split(",") if e.strip()]
    return [
        (entry.lstrip("+"), logging.DEBUG if entry.startswith("+") else logging.INFO)
        for entry in entries
    ]


def parse_log_value(value: str) -> None:
    """
    Given a string like "foo.bar,+baz.fizz" this function parses the string and converts
    it to a mapping of {"foo.bar": logging.INFO, "baz.fizz": logging.DEBUG}
    and updates the registry with this mapping
    """
    for alias, log_level in log_entries(value):
        if alias in _LOG_REGISTRY.alias_map:
            for log in _LOG_REGISTRY.alias_map[alias]:
                _LOG_REGISTRY.log_levels[log] = log_level
        else:
            _LOG_REGISTRY.log_levels[alias] = log_level


def init_logs_from_string(value: str) -> None:
    """
    Installs basic logging based on the input value
    """
    parse_log_value(value)

    for logger_name, level in _LOG_REGISTRY.log_levels.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt=f"%(asctime)s [{logger_name}] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        logger.addHandler(handler)
        logger.propagate = False


def init_logs() -> None:
    init_logs_from_string(os.environ.get(LOG_ENV_VAR, ""))


P = ParamSpec("P")


class LazyString(Generic[P]):
    def __init__(
        self, func: Callable[P, str], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        self._callable: Callable[[], str] = functools.partial(func, *args, **kwargs)

    def __str__(self) -> str:
        return self._callable()
