from __future__ import annotations

import copy

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_horizontal_state_update import _affine_rows
import helion
from helion._compiler.cute.horizontal_state_update import HorizontalGeometry
from helion._compiler.cute.horizontal_state_update import InapplicableStateUpdate
from helion.exc import InvalidConfig


def _bind(rows=128, features=128):
    values = (
        torch.empty((3, 2, rows, features), dtype=torch.bfloat16),
        torch.empty((3, 2, rows), dtype=torch.bfloat16),
        torch.empty(features, dtype=torch.bfloat16),
        torch.tensor([2, 0, 1], dtype=torch.int64),
        torch.empty((3, 2, rows), dtype=torch.bfloat16),
    )
    return _affine_rows._bind_isolated(values)


def _selected(bound, row=None, **changes):
    selected = next(
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_state_update_schedule") == "horizontal_tma"
    )
    values = copy.deepcopy(selected.config)
    if row is not None:
        row_id, _ = bound.config_spec.cute_horizontal_state_row_tiles
        values["block_sizes"][
            bound.config_spec.block_sizes.block_id_to_index(row_id)
        ] = row
    return helion.Config(**dict(values, **changes))


@pytest.mark.parametrize("row", [4, 8, 256])
def test_manual_invalid_row_rejects_without_codegen(row):
    with _cpu_codegen():
        bound = _bind()
        with pytest.raises(InvalidConfig, match="horizontal state row tile"):
            bound.config_spec.create_config_generation().strict_config_pair(
                _selected(bound, row)
            )


@pytest.mark.parametrize(
    "changes",
    [{"pid_type": "xyz"}, {"pid_type": "persistent_blocked"}],
)
def test_manual_launch_dependency_rejects(changes):
    with _cpu_codegen():
        bound = _bind()
        with pytest.raises(InvalidConfig, match="horizontal state schedule"):
            bound.config_spec.create_config_generation().strict_config_pair(
                _selected(bound, **changes)
            )


@pytest.mark.parametrize("row", [16, 32, 64, 128])
def test_legal_and_partial_rows_preserved(row):
    with _cpu_codegen():
        bound = _bind(rows=65)
        requested = _selected(bound, row)
        _, actual = bound.config_spec.create_config_generation().strict_config_pair(
            requested
        )
        assert actual.config["block_sizes"] == [row]


@pytest.mark.parametrize("row,expected", [(4, 16), (8, 16), (256, 128)])
def test_sampler_repairs_only_selected_dependencies(row, expected):
    with _cpu_codegen():
        bound = _bind()
        requested = _selected(bound, row, pid_type="xyz", cute_cluster_n=2)
        bound.config_spec.normalize(requested.config, _fix_invalid=True)
        assert requested.config["block_sizes"] == [expected]
        assert requested.config["pid_type"] == "flat"
        assert requested.config["cute_cluster_n"] == 2
        _, actual = bound.config_spec.create_config_generation().strict_config_pair(
            requested
        )
        assert actual == requested


@pytest.mark.parametrize("features", [32, 128, 256, 512, 1024])
def test_resource_choices_match_original_geometry(features):
    with _cpu_codegen():
        bound = _bind(features=features)
        _, choices = bound.config_spec.cute_horizontal_state_row_tiles
        for row in (16, 32, 64, 128):
            try:
                HorizontalGeometry(row // 16, 2, 32, features, 2).check(48 * 1024)
            except InapplicableStateUpdate:
                assert row not in choices
                with pytest.raises(InvalidConfig, match="horizontal state row tile"):
                    bound.config_spec.normalize(_selected(bound, row).config)
            else:
                assert row in choices


@pytest.mark.parametrize("row", [4, 8, 256])
def test_ordinary_config_keeps_rows_and_pid(row):
    with _cpu_codegen():
        bound = _bind()
        values = copy.deepcopy(bound.config_spec.default_config().config)
        values.update(block_sizes=[row], pid_type="xyz")
        absent, explicit = copy.deepcopy(values), copy.deepcopy(values)
        explicit["cute_state_update_schedule"] = "ordinary"
        bound.config_spec.normalize(absent)
        bound.config_spec.normalize(explicit)
        assert absent == explicit
        assert absent["block_sizes"] == [row]
        assert absent["pid_type"] == "xyz"
        assert "cute_state_update_schedule" not in absent


@pytest.mark.parametrize("row,expected", [(4, 16), (8, 16), (256, 128)])
def test_actual_sampler_unflatten_repairs_row_dependency(row, expected):
    with _cpu_codegen():
        bound = _bind()
        generation = bound.config_spec.create_config_generation()
        flat = generation.flatten(_selected(bound))
        assert len(generation.block_size_indices) == 1
        flat[generation.block_size_indices[0]] = row
        actual = generation.unflatten(flat)
        assert actual.config["cute_state_update_schedule"] == "horizontal_tma"
        assert actual.config["block_sizes"] == [expected]
        assert generation.strict_config_pair(actual)[1] == actual


@pytest.mark.parametrize("cluster", [2, 8])
def test_inactive_raw_cluster_preserved(cluster):
    with _cpu_codegen():
        bound = _bind()
        request = _selected(bound, 128, cute_cluster_n=cluster)
        generation = bound.config_spec.create_config_generation()
        assert (
            generation.strict_config_pair(request)[1].config["cute_cluster_n"]
            == cluster
        )
        bound.config_spec.normalize(request.config, _fix_invalid=True)
        assert request.config["cute_cluster_n"] == cluster
