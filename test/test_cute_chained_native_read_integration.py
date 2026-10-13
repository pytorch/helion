from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_preparation_cut import _kda_fixture
from helion import exc
from helion._compiler.cute import chained_frontier_groups as frontier
from helion._compiler.cute import chained_native_read_inputs as native


def _config(enabled, columns=32):
    config = _kda_config()
    config.config.update(
        cute_chained_native_vector_reads=enabled,
        cute_chained_frontier_tile_columns=columns,
    )
    return config


@pytest.mark.parametrize("columns", (0, 32))
def test_complete_kda_changes_only_three_proved_native_storage_reads(columns):
    kernel, args = _kda_fixture()
    with patch.object(
        native, "bind_retained_native_inputs", side_effect=AssertionError("default")
    ):
        before = _source(kernel, args, _config(False, columns))
        absent = _config(False, columns)
        absent.config.pop("cute_chained_native_vector_reads")
        assert _source(kernel, args, absent) == before
    selected = []
    original = native.bind_retained_native_inputs

    def bind(plan, frame, retention, published, first, stop, writes, **kwargs):
        result = original(
            plan, frame, retention, published, first, stop, writes, **kwargs
        )
        if result:
            assert all(item.matches(plan, published) for item in result)
            selected.append(result)
        return result

    with patch.object(native, "bind_retained_native_inputs", bind):
        after = _source(kernel, args, _config(True, columns))
    assert any(tuple(item.index for item in group) == (0, 1, 2) for group in selected)
    tag = "chain_prepared_5_group"
    restored = after
    for index in (0, 1, 2):
        prefix = f"{tag}_input_{index}"
        lines = restored.splitlines()
        setup = [line for line in lines if line.strip().startswith(f"{prefix}_")]
        loads = [
            line
            for line in lines
            if line.strip().startswith(f"cute.copy({prefix}_copy,")
        ]
        assert len(setup) == 4 and len(loads) == 1
        assert f"partition_S(chain_retained_operand_{index})" in setup[2]
        assert after.index(f"chain_retained_operand_{index} =") < after.index(setup[0])
        restored = "\n".join(line for line in lines if line not in (*setup, *loads))
        restored = restored.replace(
            f"{prefix}_values[{tag}_element]",
            f"chain_retained_operand_{index}[{tag}_row, ({tag}_base + {tag}_element)]",
        )
    assert "_group_input_" not in restored
    # Full syntax equality covers the original masks/typed-zero branches,
    # expressions/casts, destination partitions, layouts, events and barriers.
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(before))


@pytest.mark.parametrize("failure", ("binding", "producer"))
def test_native_option_cannot_activate_without_completed_original_producer(failure):
    kernel, args = _kda_fixture()
    target, name = (
        (native, "bind_retained_native_inputs")
        if failure == "binding"
        else (frontier, "emit_materialized_group")
    )
    with (
        patch.object(target, name, return_value=() if failure == "binding" else None),
        pytest.raises(exc.BackendUnsupported, match="native vector reads require"),
    ):
        _source(kernel, args, _config(True, 0))
