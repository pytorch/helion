from __future__ import annotations

from typing import Iterable
from typing import Iterator
from typing import Sequence


class OutputLines:
    """
    Helper to build source text while keeping track of the most recent newline so
    callers can inject annotations before the currently buffered statement.
    """

    def __init__(self, parent: object) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.last_newline = 0
        self.parent = parent
        self._skip_next_newline = False

    def extend(self, chunks: Iterable[str]) -> None:
        """Append text while tracking the index after the last newline."""
        concatenated = "".join(chunks)
        if not concatenated:
            return

        new_lines = concatenated.splitlines(keepends=True)
        self.lines.extend(new_lines)

        if new_lines[-1].endswith("\n"):
            self.last_newline = len(self.lines)
        elif len(new_lines) > 1:
            # Second to last line must end in newline if the last one did not.
            assert new_lines[-2].endswith("\n")
            self.last_newline = len(self.lines) - 1

    def append(self, text: str) -> None:
        self.extend([text])

    def insert_comments(self, comments: Sequence[str]) -> None:
        """Insert comment lines right before the current statement."""
        if not comments:
            return
        if self.lines and not self.lines[-1].endswith("\n"):
            self.lines[-1] = f"{self.lines[-1]}\n"
            self.last_newline = len(self.lines)
        indent = "    " * getattr(self.parent, "_indent", 0)
        insert_at = min(max(self.last_newline, 0), len(self.lines))
        for comment in comments:
            assert "\n" not in comment
            self.lines.insert(insert_at, f"{indent}{comment}\n")
            insert_at += 1
        self.last_newline = insert_at
        self._skip_next_newline = True

    def insert_annotation(self, annotation: str) -> None:
        self.insert_comments((f"# {annotation}",))

    def reset_last_location(self) -> None:
        self._skip_next_newline = False

    def insert_location_comment(self, location: object) -> None:
        # Base OutputLines does not track source locations; override when needed.
        return None

    def __bool__(self) -> bool:
        return bool(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self) -> Iterator[str]:
        return iter(self.lines)
