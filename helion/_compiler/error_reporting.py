from __future__ import annotations

from collections import defaultdict
import functools
import re
import sys
from typing import TYPE_CHECKING

from .. import exc
from ..exc import Base
from ..exc import BaseError
from ..exc import BaseWarning
from ..exc import ErrorCompilingKernel

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from .source_location import SourceLocation
    from .type_propagation import TypeNotAllowedOnDevice
    from helion.runtime.settings import Settings

    ErrorOrWarning = BaseError | BaseWarning


# pyre-ignore[9]
order_by_location: Callable[[Sequence[ErrorOrWarning]], list[ErrorOrWarning]] = (
    functools.partial(sorted, key=lambda e: e.location)
)


class ErrorReporting:
    def __init__(self, settings: Settings) -> None:
        """
        Initialize the ErrorReporting object with the provided settings.

        :param settings: The Settings object containing configuration for error reporting.
        """
        self.errors: list[BaseError] = []
        self.warnings: list[BaseWarning] = []
        self.ignores: tuple[type[BaseWarning], ...] = tuple(settings.ignore_warnings)
        self.type_errors: dict[SourceLocation, list[exc.TypePropagationError]] = (
            defaultdict(list)
        )
        self.printed_warning = 0

    def add(self, e: Base | type[Base]) -> None:
        """
        Add an error or warning to the respective list.

        :param e: An instance or type of BaseError or BaseWarning.
        :raises TypeError: If the provided object is not an error or warning.
        """
        if callable(e):
            e = e()
        if isinstance(e, BaseError):
            self.errors.append(e)
        elif isinstance(e, BaseWarning):
            if not isinstance(e, self.ignores):
                self.warnings.append(e)
        else:
            raise TypeError(f"expected error or warning, got {type(e)}")

    def add_type_error(self, type_info: TypeNotAllowedOnDevice) -> None:
        """
        Add a type error to the list of type errors and to the errors list if it's the first occurrence.

        :param type_info: The TypeNotAllowedOnDevice object containing information about the type error.
        """
        locations = type_info.locations
        similar_errors = self.type_errors[locations[0]]
        similar_errors.append(e := exc.TypePropagationError(type_info, similar_errors))
        if len(similar_errors) == 1:
            self.add(e)

    def ignore(self, e: type[BaseWarning]) -> None:
        """
        Add a warning type to the ignore list.

        :param e: The type of BaseWarning to ignore.
        """
        self.ignores = (*self.ignores, e)

    def raise_if_errors(self) -> None:
        """
        Raise an exception if there are any errors, after reporting pending warnings.

        :raises ErrorCompilingKernel: If there are multiple errors.
        :raises BaseError: If there is a single error.
        """
        sys.stderr.write(self.report(warnings_offset=self.printed_warning))
        self.printed_warning = len(self.warnings)
        if self.errors:
            if len(self.errors) > 1:
                raise ErrorCompilingKernel(len(self.errors), len(self.warnings))
            raise self.errors[0]

    def report(self, *, strip_paths: bool = False, warnings_offset: int = 0) -> str:
        """
        Generate a report of the errors and warnings.

        :param strip_paths: Whether to strip file paths from the report.
        :param warnings_offset: The offset to start reporting warnings from.
        :return: A string representation of the report.
        """
        report = "\n\n".join(
            {
                e.report(): None
                for e in order_by_location(
                    [*self.errors, *self.warnings[warnings_offset:]]
                )
            }
        )
        if strip_paths:
            report = re.sub(r'"[/\\][^"]*[/\\]', '".../', report)
        return report
