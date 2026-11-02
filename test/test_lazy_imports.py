from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any
from typing import cast
import unittest
from unittest.mock import patch

import pytest

from helion import exc
from helion._testing import TestCase
from helion.autotuner import de_surrogate_hybrid
from helion.autotuner import surrogate_pattern_search

# No test here needs a GPU.
pytestmark = pytest.mark.cpu_only

_HINT = "requires scikit-learn. Install it with: pip install scikit-learn"


class TestLazyImports(TestCase):
    def test_import_helion_does_not_load_sklearn(self) -> None:
        # scikit-learn also loads SciPy and is slow to import. Only a new
        # interpreter shows what `import helion` loads. This test takes about
        # 2 s and is the only guard against the regression, so it is not
        # marked slow.
        script = """
            import sys
            import helion
            import helion.autotuner
            print(sorted(m for m in ("sklearn", "scipy") if m in sys.modules))
        """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")

    def test_surrogate_search_without_sklearn(self) -> None:
        # Without scikit-learn the surrogate searches give the install hint
        # when they are created.
        with (
            patch.object(de_surrogate_hybrid, "HAS_ML_DEPS", False),
            self.assertRaisesRegex(ImportError, f"DESurrogateHybrid {_HINT}"),
        ):
            de_surrogate_hybrid.DESurrogateHybrid(cast("Any", None), ())
        with (
            patch.object(surrogate_pattern_search, "HAS_ML_DEPS", False),
            self.assertRaisesRegex(exc.AutotuneError, f"LFBOPatternSearch {_HINT}"),
        ):
            surrogate_pattern_search.LFBOPatternSearch(cast("Any", None), ())


if __name__ == "__main__":
    unittest.main()
