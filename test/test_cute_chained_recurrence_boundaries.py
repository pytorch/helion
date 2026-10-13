from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_recurrence_body import _code
from helion._compiler.cute import chained_body_program as bodies
from helion._compiler.cute import chained_recurrence_body as recurrence


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("point", ("span", "accept"))
@pytest.mark.parametrize("change", ("value", "remove", "order"))
def test_new_publications_are_bound_at_original_stage_return(dtype, point, change):
    bind = recurrence.bind_recurrence_body
    accept = recurrence.RecurrenceBody.accept
    span = bodies._completed_span
    captured, touched = [], []

    def capture(*args, **kwargs):
        body = bind(*args, **kwargs)
        captured.append(body)
        return body

    def mutate():
        if touched:
            return
        body = captured[-1]
        original = tuple(body.boundaries.items())
        assert body._completed_boundaries == original
        added = [node for node in body.boundaries if node not in dict(body._boundaries)]
        assert added
        node = added[0]
        if change == "value":
            body.boundaries[node] = "unemitted_result"
        elif change == "remove":
            body.boundaries.pop(node)
        else:
            value = body.boundaries.pop(node)
            remainder = tuple(body.boundaries.items())
            body.boundaries.clear()
            body.boundaries[node] = value
            body.boundaries.update(remainder)
        assert tuple(body.boundaries.items()) != original
        touched.append(original)

    def capture_span(*args):
        result = span(*args)
        if captured and point == "span":
            mutate()
        return result

    def accept_stage(self, stage, lines, completion, aliases):
        if point == "accept":
            mutate()
        return accept(self, stage, lines, completion, aliases)

    with (
        patch.object(recurrence, "bind_recurrence_body", capture),
        patch.object(recurrence.RecurrenceBody, "accept", accept_stage),
        patch.object(bodies, "_completed_span", capture_span),
        pytest.raises(Exception, match="recurrence.*(completion|publication) changed"),
    ):
        _code("completed", dtype)
    assert len(captured) == len(touched) == 1
    body = captured[0]
    assert body._cursor == 0 and body._pending is body.recurrence.stages[0]
    assert body._lines == [] and body._finished is None
    assert body._completed_boundaries == touched[0]


@pytest.mark.parametrize("kind", ("typed", "packed", "carry", "completed"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_original_boundary_capture_is_consumed_once_per_stage(kind, dtype):
    accept = recurrence.RecurrenceBody.accept
    records = []

    def accept_stage(self, stage, lines, completion, aliases):
        expected = tuple(self.boundaries.items())
        assert self._completed_boundaries == expected
        result = accept(self, stage, lines, completion, aliases)
        assert self._boundaries == expected and self._completed_boundaries is None
        records.append(stage)
        return result

    with patch.object(recurrence.RecurrenceBody, "accept", accept_stage):
        _code(kind, dtype)
    assert records
