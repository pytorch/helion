from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
import os
from typing import Callable
from typing import ParamSpec

LOG_ENV_VAR = "HELION_LOGS"

"""
Usage:

HELION_LOGS=+helion.runtime.kernel python test.py

will run test.py with helion.runtime.kernel logs enabled at logging.DEBUG level
"""


@dataclass
class LogRegistery:
    """
    This registery holds mappings for logging
    """

    # alias -> list of logs
    alias_map: dict[str, list[str]] = field(default_factory=dict)

    # log name -> log level
    log_levels: dict[str, int] = field(default_factory=dict)


_LOG_REGISTERY = LogRegistery()


def parse_log_value(value: str) -> None:
    """
    Given a string like "foo.bar,+baz.fizz" this function parses this string and converts
    it to mapping of {"foo.bar": logging.INFO, "baz.fizz": logging.DEBUG}
    and updates the registery with this mapping
    """
    entries = [e.strip() for e in value.split(",") if e.strip()]
    for entry in entries:
        log_level = logging.DEBUG if entry.startswith("+") else logging.INFO
        alias = entry.lstrip("+")

        if alias in _LOG_REGISTERY.alias_map:
            for log in _LOG_REGISTERY.alias_map[alias]:
                _LOG_REGISTERY.log_levels[log] = log_level
        else:
            _LOG_REGISTERY.log_levels[alias] = log_level


def init_logs_from_string(value: str) -> None:
    """
    Installs basic logging based on the input value
    """
    parse_log_value(value)

    for logger_name, level in _LOG_REGISTERY.log_levels.items():
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


class LazyString:
    def __init__(
        self, func: Callable[P, str], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        self.func: Callable[P, str] = func
        self.args: tuple[object, ...] = args
        self.kwargs: object = kwargs

    def __str__(self) -> str:
        return self.func(*self.args, **self.kwargs)
