from __future__ import annotations

from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_preparation_pipeline import _args
from .test_cute_chained_preparation_pipeline import _config
from helion._compiler.cute import chained_pipeline_storage as storage


@pytest.mark.parametrize("steps", [0, 1, 4, 7])
@pytest.mark.parametrize("late", [False, True])
def test_cohort_revision_records_empty_host_shapes(steps: int, late: bool) -> None:
    args = _args("cpu", steps, late)
    config = _config(16, pipeline=True)
    config.config.update(
        cute_chained_preparation_cohorts=3,
        cute_chained_preparation_unroll=1,
    )
    original = storage.capture_storage_revision
    recorded = []

    def observe(*args, **kwargs):
        revision = original(*args, **kwargs)
        recorded.append(revision)
        return revision

    with (
        _cpu_codegen(),
        patch.object(storage, "capture_storage_revision", side_effect=observe),
    ):
        bound = _typed_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)):
            source = bound.to_code(config)
    assert len(recorded) == 1 and recorded[0] is not None
    assert any(0 in shape for _, shape in recorded[0].shapes) is (steps == 0)
    assert "chain_cohort =" in source
    assert "chain_loop_carry_" in source
