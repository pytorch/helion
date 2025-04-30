from __future__ import annotations

import collections
from contextlib import suppress
import linecache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import traceback


def _ensure_original_line(fs: traceback.FrameSummary) -> None:
    """
    Guarantee that ``fs._original_line`` exists and contains the
    *unmodified* source line (with leading whitespace preserved).

    Called by the back-ported ``format_frame_summary`` on interpreters
    where FrameSummary didn't have that attribute yet (<= 3.10).
    """
    if hasattr(fs, "_original_line"):
        return

    # 3.8/3.9/3.10 store the raw text in the private ``_line`` slot,
    # but it may be None when lookup_line=False.
    raw = getattr(fs, "_line", None)
    if raw is None and fs.filename and fs.lineno:
        raw = linecache.getline(fs.filename, fs.lineno)

    # Fall back to the stripped ``fs.line`` if we have nothing else.
    if raw is None:
        raw = fs.line or ""

    # Same public behaviour as 3.11's property:
    # "return the line as-is from the source, without modifying whitespace".
    fs._original_line = raw  # pyre-ignore[16]


def _byte_offset_to_character_offset(s: str, offset: int) -> int:
    """Convert a *byte* offset in UTF-8 source to a *character* offset."""
    as_utf8 = s.encode("utf-8")
    return len(as_utf8[:offset].decode("utf-8", errors="replace"))


_WIDE_CHAR_SPECIFIERS = "WF"


def _display_width(line: str, offset: int) -> int:
    """How many monospace columns ``line[:offset]`` would occupy."""
    if line.isascii():  # fast path
        return offset
    import unicodedata

    return sum(
        2 if unicodedata.east_asian_width(ch) in _WIDE_CHAR_SPECIFIERS else 1
        for ch in line[:offset]
    )


# pyre-ignore[2, 4]
_Anchors = collections.namedtuple(  # noqa: PYI024
    "_Anchors",
    [
        "left_end_offset",
        "right_start_offset",
        "primary_char",
        "secondary_char",
    ],
    defaults=["~", "^"],
)


def _extract_caret_anchors_from_line_segment(segment: str) -> _Anchors | None:
    """
    Heuristically decide where "primary" (^) and "secondary" (~) carets
    should be placed beneath *segment*, mimicking CPython 3.11.
    """
    import ast

    try:
        tree = ast.parse(segment)
    except SyntaxError:
        return None
    if len(tree.body) != 1:
        return None

    def normalize(off: int) -> int:
        return _byte_offset_to_character_offset(segment, off)

    statement = tree.body[0]

    if isinstance(statement, ast.Expr):
        expr = statement.expr  # pyre-ignore[16]
        #
        # 1.  Binary operator (a + b, a * b, ...)
        #
        if isinstance(expr, ast.BinOp):
            operator_start = normalize(expr.left.end_col_offset)  # pyre-ignore[6]
            operator_end = normalize(expr.right.col_offset)
            operator_str = segment[operator_start:operator_end]
            operator_offset = len(operator_str) - len(operator_str.lstrip())

            left_anchor = expr.left.end_col_offset + operator_offset  # pyre-ignore[58]
            right_anchor = left_anchor + 1
            if (
                operator_offset + 1 < len(operator_str)
                and not operator_str[operator_offset + 1].isspace()
            ):
                right_anchor += 1

            # skip spaces, parens, comment markers
            while left_anchor < len(segment) and (
                (ch := segment[left_anchor]).isspace() or ch in ")#"
            ):
                left_anchor += 1
                right_anchor += 1

            return _Anchors(  # pyre-ignore[20]
                normalize(left_anchor),
                normalize(right_anchor),
            )

        #
        # 2.  Subscript (a[index])
        #
        if isinstance(expr, ast.Subscript):
            left_anchor = normalize(expr.value.end_col_offset)  # pyre-ignore[6]
            right_anchor = normalize(expr.slice.end_col_offset + 1)  # pyre-ignore[58]

            while left_anchor < len(segment) and (
                (ch := segment[left_anchor]).isspace() or ch != "["
            ):
                left_anchor += 1
            while right_anchor < len(segment) and (
                (ch := segment[right_anchor]).isspace() or ch != "]"
            ):
                right_anchor += 1
            if right_anchor < len(segment):
                right_anchor += 1

            return _Anchors(left_anchor, right_anchor)  # pyre-ignore[20]

    return None  # fallback - no fancy anchors


def format_frame_summary(frame_summary):  # type: ignore[override]
    """Backport of Python 3.11's traceback.StackSummary.format_frame_summary()."""

    _ensure_original_line(frame_summary)

    row: list[str] = []

    # 1.  Header
    row.append(
        f'  File "{frame_summary.filename}", line {frame_summary.lineno}, in {frame_summary.name}\n'
    )

    # 2.  Source line(s)
    if frame_summary.line:
        stripped_line = frame_summary.line.strip()
        row.append(f"    {stripped_line}\n")

        line = frame_summary._original_line
        orig_line_len = len(line)
        frame_line_len = len(frame_summary.line.lstrip())
        stripped_characters = orig_line_len - frame_line_len

        if frame_summary.colno is not None and frame_summary.end_colno is not None:
            start_offset = _byte_offset_to_character_offset(line, frame_summary.colno)
            end_offset = _byte_offset_to_character_offset(line, frame_summary.end_colno)
            code_segment = line[start_offset:end_offset]

            anchors = None
            if frame_summary.lineno == frame_summary.end_lineno:
                with suppress(Exception):
                    anchors = _extract_caret_anchors_from_line_segment(code_segment)
            else:
                # multi-line span - ensure end_offset ends at end of physical line
                end_offset = len(line.rstrip())

            need_carets = end_offset - start_offset < len(stripped_line) or (
                anchors and anchors.right_start_offset - anchors.left_end_offset > 0
            )

            if need_carets:
                dp_start_offset = _display_width(line, start_offset) + 1
                dp_end_offset = _display_width(line, end_offset) + 1

                row.append("    ")  # noqa: FURB113
                row.append(" " * (dp_start_offset - stripped_characters))
                if anchors:
                    dp_left_end_offset = _display_width(
                        code_segment, anchors.left_end_offset
                    )
                    dp_right_start_offset = _display_width(
                        code_segment, anchors.right_start_offset
                    )
                    row.append(anchors.primary_char * dp_left_end_offset)  # noqa: FURB113
                    row.append(
                        anchors.secondary_char
                        * (dp_right_start_offset - dp_left_end_offset)
                    )
                    row.append(
                        anchors.primary_char
                        * (dp_end_offset - dp_start_offset - dp_right_start_offset)
                    )
                else:
                    row.append("^" * (dp_end_offset - dp_start_offset))
                row.append("\n")

    # 3.  Locals dump (if present)
    if frame_summary.locals:
        for name, value in sorted(frame_summary.locals.items()):
            row.append(f"     {name} = {value}\n")

    return "".join(row)
