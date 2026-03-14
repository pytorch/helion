#!/usr/bin/env python3
"""Read Helion PRINT_OUTPUT_CODE output from stdin and pretty-print the MSL.

Usage:
    HELION_PRINT_OUTPUT_CODE=1 HELION_USE_DEFAULT_CONFIG=1 \
      .venv/bin/pytest test/test_metal.py::TestMetalBackend::test_vector_add -x -s 2>&1 | \
      .venv/bin/python3 scripts/show_msl.py
"""

import sys

text = sys.stdin.read()
for line in text.splitlines():
    if "return ('" in line and "', '" in line:
        start = line.index("'") + 1
        end = line.index("'", start)
        msl = line[start:end].replace("\\n", "\n")
        print(msl)
        break
