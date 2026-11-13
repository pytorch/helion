from __future__ import annotations

import gc
import weakref

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_optional_lse import check_optional_lse_replay
from helion._testing.cute_optional_lse import initialize_paged_attention_inputs
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_optional_lse import optional_lse_fixture


@skipIfNotCUDA()
@pytest.mark.parametrize("form", ["reduced", "per_key", "mapped"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_optional_lse_outputs_and_changed_replay(form: str, return_lse: bool) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    kernel, args = optional_lse_fixture(form, return_lse, DEVICE)
    initialize_paged_attention_inputs(args)
    # Cover both math policies without doubling the native configuration grid.
    bound = helion.kernel(
        kernel.fn, backend="cute", static_shapes=True, fast_math=form != "reduced"
    ).bind(args)
    config = optional_lse_config()
    source = bound.to_code(config)
    expected_body = (
        "paged_attention_body("
        if form == "reduced"
        else "paged_attention_partial_body("
    )
    assert expected_body in source
    check_optional_lse_replay(bound, config, args, form, return_lse)


@skipIfNotCUDA()
def test_replay_graph_reset_precedes_delayed_collection() -> None:
    # No Helion compilation: this pins the CUDA graph lifetime shared by the
    # attention replay helpers. Graph resources must be released before Python
    # collects an older graph during a later capture.
    gc.collect()
    source = torch.ones(8, device=DEVICE)
    expected = source + 2

    def old_graph_cycle() -> tuple[list[object], weakref.ReferenceType]:
        with cuda_graph_for_replay() as graph:
            with torch.cuda.graph(graph):
                output = source + 1
            graph.replay()
            torch.testing.assert_close(output, source + 1, rtol=0, atol=0)
            cycle: list[object] = [graph]
            cycle.append(cycle)
            return cycle, weakref.ref(graph)

    cycle, previous = old_graph_cycle()
    assert previous() is not None
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            del cycle
            gc.collect()
            output = source + 2
        assert previous() is None
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
