from __future__ import annotations

from unittest.mock import patch

import pytest

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_preparation_cut import _kda_fixture
from helion import exc
from helion._compiler.cute import chained_frontier_groups as groups
from helion._compiler.cute.chained_frontier_ownership import FrontierOwnership


def _config(columns, **kwargs):
    config = _kda_config(**kwargs)
    config.config["cute_chained_frontier_tile_columns"] = columns
    return config


def test_exact_six_ownership_changes_only_after_successful_emission():
    kernel, args = _kda_fixture()
    with patch.object(
        FrontierOwnership, "select", side_effect=AssertionError("default discovery")
    ):
        before = _source(kernel, args, _kda_config())
        assert _source(kernel, args, _config(0)) == before
    original = FrontierOwnership.select
    policies = []

    def observe(self, *args, **kwargs):
        selected = original(self, *args, **kwargs)
        if selected is not None:
            assert not self.activated
            policies.append(self)
        return selected

    with patch.object(FrontierOwnership, "select", observe):
        after = _source(kernel, args, _config(32))
    assert len(policies) == 1 and policies[0].activated
    tag = "chain_prepared_5_group"
    changes = [
        (
            f"{tag}_row = chain_prep_thread // 16 + {tag}_step * 8",
            f"{tag}_row = chain_prep_thread // 4",
        ),
        (
            f"{tag}_base = chain_prep_thread % 16 * 8",
            f"{tag}_base = chain_prep_thread % 4 * 8 + {tag}_step * 32",
        ),
    ]
    for index in (1, 2):
        prefix = f"{tag}_output_{index}"
        line = next(line for line in before.splitlines() if f"{prefix}_copy =" in line)
        changes.extend(
            [
                (
                    line,
                    line.replace(
                        "((8, 16), stride=(16, 1))", "((32, 4), stride=(4, 1))"
                    ),
                ),
                (
                    f"{prefix}_target[None, {tag}_step, 0]",
                    f"{prefix}_target[None, 0, {tag}_step]",
                ),
            ]
        )
    expected = before
    for old, new in changes:
        assert old != new and expected.count(old) == 1
        expected = expected.replace(old, new)
    assert after == expected


@pytest.mark.parametrize("columns", [128, 256])
def test_positive_ineffective_option_cannot_succeed_silently(columns):
    kernel, args = _kda_fixture()
    with pytest.raises(
        exc.BackendUnsupported, match="successfully emitted mixed-native"
    ):
        _source(kernel, args, _config(columns))


@pytest.mark.parametrize("retention", [False, True])
def test_failed_original_emitter_cannot_activate_ownership(retention):
    kernel, args = _kda_fixture()
    emitter = "emit_materialized_group" if retention else "emit_vector_group"
    original = getattr(groups, emitter)
    attempts = []

    def reject(*args, **kwargs):
        if kwargs.get("ownership") is not None:
            attempts.append(True)
            return None
        return original(*args, **kwargs)

    config = _config(32)
    config.config["cute_chained_operand_retention"] = retention
    with (
        patch.object(groups, emitter, reject),
        pytest.raises(
            exc.BackendUnsupported, match="successfully emitted mixed-native"
        ),
    ):
        _source(kernel, args, config)
    assert attempts
