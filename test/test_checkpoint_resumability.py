from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.metrics import AutotuneMetrics
from helion.runtime.config import Config
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


def _make_mock_search(
    tmpdir: str,
    *,
    checkpoint_enabled: bool = True,
) -> PopulationBasedSearch:
    """Create a PopulationBasedSearch with mocked kernel and settings for testing."""
    settings = Settings()
    settings.autotune_checkpoint = checkpoint_enabled
    settings.autotune_log_level = 0

    config_spec = MagicMock()
    config_spec.create_config_generation.return_value = MagicMock(spec=ConfigGeneration)

    kernel = MagicMock()
    kernel.settings = settings
    kernel.config_spec = config_spec

    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.kernel = kernel
    search.settings = settings
    search.config_spec = config_spec
    search.args = []
    search.best_perf_so_far = float("inf")
    search._prepared = True
    search._precompile_tmpdir = None
    search._precompile_args_path = None
    search._checkpoint_key = "test_checkpoint_key_abc123"
    search._autotune_metrics = AutotuneMetrics()
    search._autotune_metrics.num_configs_tested = 10
    search.population = []
    search.config_gen = MagicMock(spec=ConfigGeneration)

    # Mock the log
    search.log = MagicMock()
    search.log.debug = MagicMock()

    return search


def _make_population_member(
    flat_values: list[object],
    perf: float = 1.0,
    status: str = "ok",
) -> PopulationMember:
    """Create a PopulationMember for testing."""
    config = Config(block_size=[32])
    return PopulationMember(
        fn=lambda: None,
        perfs=[perf],
        flat_values=flat_values,
        config=config,
        status=status,
        compile_time=0.5,
    )


class TestCheckpointPath:
    def test_returns_path_when_key_set(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            path = search._checkpoint_path()
            assert path is not None
            assert path.suffix == ".checkpoint"
            assert "test_checkpoint_key_abc123" in path.name

    def test_returns_none_when_no_key(self, tmp_path: Path) -> None:
        search = _make_mock_search(str(tmp_path))
        search._checkpoint_key = None
        assert search._checkpoint_path() is None


class TestSaveCheckpoint:
    def test_saves_valid_json(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            search.population = [
                _make_population_member([32, 4], perf=0.5),
                _make_population_member([64, 8], perf=0.8),
            ]
            search._save_checkpoint(3)

            path = search._checkpoint_path()
            assert path is not None
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["version"] == 1
            assert data["search_algorithm"] == "PopulationBasedSearch"
            assert data["generation"] == 3
            assert data["num_configs_tested"] == 10
            assert len(data["population"]) == 2
            assert data["best_perf"] == 0.5

    def test_saves_extra_data(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            search.population = [_make_population_member([32, 4], perf=1.0)]
            search._save_checkpoint(1, extra={"custom_key": [1, 2, 3]})

            path = search._checkpoint_path()
            assert path is not None
            data = json.loads(path.read_text())
            assert data["extra"]["custom_key"] == [1, 2, 3]

    def test_no_save_when_disabled(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path), checkpoint_enabled=False)
            search.population = [_make_population_member([32, 4], perf=1.0)]
            search._save_checkpoint(1)

            path = search._checkpoint_path()
            assert path is not None
            assert not path.exists()

    def test_no_save_when_no_key(self, tmp_path: Path) -> None:
        search = _make_mock_search(str(tmp_path))
        search._checkpoint_key = None
        search.population = [_make_population_member([32, 4], perf=1.0)]
        search._save_checkpoint(1)
        # Should not raise and should not create any file

    def test_atomic_write(self, tmp_path: Path) -> None:
        """Verify no temp files are left behind after save."""
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            search.population = [_make_population_member([32, 4], perf=1.0)]
            search._save_checkpoint(1)

            # Only the checkpoint file should exist, no tmp files
            files = list(tmp_path.iterdir())
            assert len(files) == 1
            assert files[0].suffix == ".checkpoint"


class TestLoadCheckpoint:
    def test_loads_valid_checkpoint(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            # Write a valid checkpoint
            search.population = [_make_population_member([32, 4], perf=0.5)]
            search._save_checkpoint(5)

            # Load it back
            data = search._load_checkpoint()
            assert data is not None
            assert data["generation"] == 5
            assert len(data["population"]) == 1

    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            assert search._load_checkpoint() is None

    def test_returns_none_when_disabled(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path), checkpoint_enabled=False)
            assert search._load_checkpoint() is None

    def test_returns_none_on_corrupted_json(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            path = search._checkpoint_path()
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not valid json{{{")

            assert search._load_checkpoint() is None

    def test_returns_none_on_version_mismatch(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            path = search._checkpoint_path()
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"version": 999, "search_algorithm": "PopulationBasedSearch"}
                )
            )

            assert search._load_checkpoint() is None

    def test_returns_none_on_algorithm_mismatch(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            path = search._checkpoint_path()
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"version": 1, "search_algorithm": "WrongAlgorithm"})
            )

            assert search._load_checkpoint() is None


class TestDeleteCheckpoint:
    def test_deletes_existing_file(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            search.population = [_make_population_member([32, 4], perf=1.0)]
            search._save_checkpoint(1)

            path = search._checkpoint_path()
            assert path is not None
            assert path.exists()

            search._delete_checkpoint()
            assert not path.exists()

    def test_noop_when_no_file(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            # Should not raise
            search._delete_checkpoint()

    def test_noop_when_no_key(self, tmp_path: Path) -> None:
        search = _make_mock_search(str(tmp_path))
        search._checkpoint_key = None
        # Should not raise
        search._delete_checkpoint()


class TestCheckpointRoundTrip:
    def test_save_and_load_preserves_population(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            search.population = [
                _make_population_member([32, 4], perf=0.3, status="ok"),
                _make_population_member([64, 8], perf=0.7, status="ok"),
                _make_population_member([128, 2], perf=float("inf"), status="error"),
            ]
            search._save_checkpoint(7)

            data = search._load_checkpoint()
            assert data is not None
            assert data["generation"] == 7
            assert len(data["population"]) == 3

            # Verify population entries
            pop = data["population"]
            assert pop[0]["flat_values"] == [32, 4]
            assert pop[0]["perfs"] == [0.3]
            assert pop[0]["status"] == "ok"
            assert pop[1]["flat_values"] == [64, 8]
            assert pop[2]["status"] == "error"

    def test_overwrite_existing_checkpoint(self, tmp_path: Path) -> None:
        with patch(
            "helion.autotuner.local_cache.get_helion_cache_dir", return_value=tmp_path
        ):
            search = _make_mock_search(str(tmp_path))
            search.population = [_make_population_member([32, 4], perf=1.0)]
            search._save_checkpoint(1)

            search.population = [_make_population_member([64, 8], perf=0.5)]
            search._save_checkpoint(2)

            data = search._load_checkpoint()
            assert data is not None
            assert data["generation"] == 2
            assert data["population"][0]["flat_values"] == [64, 8]


class TestEnvVarDisable:
    def test_env_var_disables_checkpoint(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"HELION_AUTOTUNE_CHECKPOINT": "0"}):
            settings = Settings()
            assert settings.autotune_checkpoint is False

    def test_env_var_enables_checkpoint(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"HELION_AUTOTUNE_CHECKPOINT": "1"}):
            settings = Settings()
            assert settings.autotune_checkpoint is True

    def test_default_is_enabled(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            # Remove the env var if it exists
            os.environ.pop("HELION_AUTOTUNE_CHECKPOINT", None)
            settings = Settings()
            assert settings.autotune_checkpoint is True
