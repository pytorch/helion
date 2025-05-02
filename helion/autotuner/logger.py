from __future__ import annotations

import itertools
import logging
import sys
import time
from typing import Callable


class LambdaLogger:
    """
    A self-contained logger that does not propagates to the root logger and
    prints each record to stderr in the form:

        [<elapsed>s] <message>

    where *elapsed* is the whole-second wall-clock time since the logger
    instance was created.

    Takes lambas as arguments, which are called when the log is emitted.
    """

    _count: itertools.count[int] = itertools.count()

    def __init__(self, level: int) -> None:
        self.level = level
        self._logger: logging.Logger = logging.getLogger(
            f"{__name__}.{next(self._count)}"
        )
        self._logger.setLevel(level)
        self._logger.propagate = False
        self.reset()

    def reset(self) -> None:
        self._logger.handlers.clear()
        self._logger.addHandler(_make_handler())

    def __call__(
        self, *msg: str | Callable[[], str], level: int = logging.INFO
    ) -> None:
        """
        Log a message at a specified log level.

        :param msg: The message(s) to log. Can be strings or callables that return strings.
        :type msg: str | Callable[[], str]
        :param level: The log level for the message.
        :type level: int
        """
        if level >= self.level:
            self._logger.log(level, " ".join(map(_maybe_call, msg)))

    def warning(self, *msg: str | Callable[[], str]) -> None:
        return self(*msg, level=logging.WARNING)

    def debug(self, *msg: str | Callable[[], str]) -> None:
        return self(*msg, level=logging.DEBUG)


def _make_handler() -> logging.Handler:
    start = time.perf_counter()

    class _ElapsedFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            elapsed = int(time.perf_counter() - start)
            return f"[{elapsed}s] {record.getMessage()}"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ElapsedFormatter())
    return handler


def _maybe_call(fn: Callable[[], str] | str) -> str:
    """
    Call a callable or return the string directly.

    :param fn: A callable that returns a string or a string.
    :type fn: Callable[[], str] | str
    :return: The resulting string.
    :rtype: str
    """
    if callable(fn):
        return fn()
    return fn
