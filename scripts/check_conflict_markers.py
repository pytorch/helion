"""Reject unresolved conflict markers, including outside merge/rebase state."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

# A bare equals underline is also valid Markdown/RST. Every intact conflict
# has an opening/closing marker; recognize those and diff3 ancestor markers
# without treating a seven-character documentation heading as a conflict.
# Git's conflict-marker-size attribute permits longer marker runs as well.
MARKER = re.compile(rb"^(?:<{7,}|>{7,}|\|{7,})(?:[ \t]|\r?$)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for path in args.filenames:
        with path.open("rb") as source:
            for number, line in enumerate(source, 1):
                if MARKER.match(line):
                    print(f"{path}:{number}: unresolved conflict marker")
                    failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
