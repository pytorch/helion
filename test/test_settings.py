from __future__ import annotations

import unittest

import helion


class TestSettingsValidation(unittest.TestCase):
    def test_autotune_effort_none_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "autotune_effort must be one of"):
            helion.Settings(autotune_effort=None)

    def test_autotune_effort_case_insensitive(self) -> None:
        settings = helion.Settings(autotune_effort="Quick")
        self.assertEqual(settings.autotune_effort, "quick")

    def test_negative_compile_timeout_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"Invalid value for autotune_compile_timeout: -1"
        ):
            helion.Settings(autotune_compile_timeout=-1)

    def test_autotune_precompile_jobs_negative_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"Invalid value for autotune_precompile_jobs: -1"
        ):
            helion.Settings(autotune_precompile_jobs=-1)

    def test_autotune_max_generations_negative_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"Invalid value for autotune_max_generations: -1"
        ):
            helion.Settings(autotune_max_generations=-1)

    def test_autotune_effort_invalid_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "autotune_effort must be one of"):
            helion.Settings(autotune_effort="super-fast")
