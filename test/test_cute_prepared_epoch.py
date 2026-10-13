"""True original four-dot and same-dot split-K authority; no native execution."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from test import test_cute_chunk_recurrence as original

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import ReductionLoopBlockSizeSource
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chunk_recurrence
from helion._compiler.cute.prepared_continuation import PreparedContinuation
from helion._compiler.cute.prepared_epoch import ContractionSegment
from helion._compiler.cute.prepared_epoch import PreparedEpoch
from helion._compiler.device_function import DeviceFunction


def _capture(check):
    planner = chunk_recurrence._plan_chunk_recurrence
    observed = []

    def select(*args, **kwargs):
        plan = planner(*args, **kwargs, prepared_epoch=True)
        assert plan is not None and plan.prepared_epoch is not None
        plan.prepared_epoch.check()
        chunk_recurrence.validate_epoch_match(plan)
        check(plan.prepared_epoch)
        observed.append(plan)
        return plan

    with patch.object(chunk_recurrence, "_plan_chunk_recurrence", select):
        source = original._code(fp32_state=True, dv_partitions=2, pipeline="wide")
    assert len(observed) == 1
    return source


def test_epoch_retains_all_four_original_specs():
    def check(epoch):
        p, q, o, r = epoch.continuation.region.contractions
        assert epoch.projected.spec is p
        assert epoch.continuation.first is q
        assert epoch.continuation.second is o
        assert epoch.update is r
        assert p.node is epoch.projection.source
        assert o.accumulator is q.node
        assert epoch.projection.state in epoch.continuation.region.nodes
        left, right = epoch.projected.segments
        assert (left.begin, left.end, left.commit) == (0, 4, False)
        assert (right.begin, right.end, right.commit) == (4, 8, True)
        assert epoch.projected.initialized(left) is False
        assert epoch.projected.initialized(right) is True
        # A fake P/P relation is still rejected by the old continuation class.
        with pytest.raises(chain._UnsupportedChain, match="continuation relation"):
            PreparedContinuation(epoch.continuation.region, p, p)

    _capture(check)


def test_final_epoch_binding_rechecks_original_semantic_validator():
    planner = chunk_recurrence._plan_chunk_recurrence
    observed = []

    def select(*args, **kwargs):
        plan = planner(*args, **kwargs, prepared_epoch=True)
        assert plan is not None
        with patch.object(
            chunk_recurrence, "_validate_recurrence_semantics", return_value=False
        ) as validator:
            with pytest.raises(
                chain._UnsupportedChain, match="numerical match changed"
            ):
                chunk_recurrence.validate_epoch_match(plan)
            validator.assert_called_once()
        chunk_recurrence.validate_epoch_match(plan)
        observed.append(True)
        return plan

    with patch.object(chunk_recurrence, "_plan_chunk_recurrence", select):
        original._code(fp32_state=True, dv_partitions=2, pipeline="wide")
    assert observed == [True]


@pytest.mark.parametrize(
    "segments",
    [
        (),
        (ContractionSegment(0, 4, True),),
        (ContractionSegment(0, 4, False), ContractionSegment(5, 8, True)),
        (ContractionSegment(4, 8, True), ContractionSegment(0, 4, False)),
        (ContractionSegment(0, 4, False), ContractionSegment(3, 8, True)),
        (ContractionSegment(0, 4, False), ContractionSegment(4, 8, False)),
        (ContractionSegment(0, 4, False, True), ContractionSegment(4, 8, True)),
        (ContractionSegment(False, 4, False), ContractionSegment(4, 8, True)),
        (
            ContractionSegment(0, 4, 0),  # pyrefly: ignore [bad-argument-type]
            ContractionSegment(4, 8, True),
        ),
    ],
)
def test_incomplete_or_changed_original_prefix_rejects(segments):
    def check(epoch):
        with pytest.raises(chain._UnsupportedChain, match="contraction"):
            replace(epoch.projected, segments=segments)

    _capture(check)


@pytest.mark.parametrize("mutation", ["spec", "segment", "region", "graph"])
def test_same_shaped_or_stale_segment_authority_rejects(mutation):
    def check(epoch):
        segmented = epoch.projected
        if mutation == "spec":
            with pytest.raises(chain._UnsupportedChain, match="segmented contraction"):
                replace(segmented, spec=replace(segmented.spec))
        elif mutation == "segment":
            with pytest.raises(
                chain._UnsupportedChain, match="foreign contraction segment"
            ):
                segmented.initialized(replace(segmented.segments[0]))
        elif mutation == "region":
            original_region = segmented.region
            try:
                object.__setattr__(segmented, "region", replace(original_region))
                with pytest.raises(chain._UnsupportedChain, match="graph changed"):
                    segmented.check()
            finally:
                object.__setattr__(segmented, "region", original_region)
        else:
            node = segmented.spec.node
            original_args = node.args
            try:
                node.args = (*node.args[:2], node, node.args[3])
                with pytest.raises(chain._UnsupportedChain, match="graph changed"):
                    segmented.check()
            finally:
                node.args = original_args
        epoch.check()

    _capture(check)


def test_copied_epoch_update_spec_rejects():
    def check(epoch):
        with pytest.raises(chain._UnsupportedChain, match="epoch relation"):
            PreparedEpoch(
                epoch.projection,
                epoch.continuation,
                epoch.projected,
                replace(epoch.update),
            )

    _capture(check)


def test_default_constructor_keeps_epoch_absent_before_public_selection():
    planner = chunk_recurrence._plan_chunk_recurrence
    observed = []

    def capture(*args, **kwargs):
        plan = planner(*args, **kwargs)
        selected = kwargs.get("prepared_epoch", False)
        assert plan is not None
        if selected:
            assert plan.prepared_epoch is not None
        else:
            assert plan.prepared_epoch is None
        observed.append(selected)
        return plan

    with patch.object(chunk_recurrence, "_plan_chunk_recurrence", capture):
        original._code(fp32_state=True, dv_partitions=2, pipeline="wide")
    assert observed == [False, True]


def test_actual_reduction_config_mutation_rejects():
    def check(epoch):
        config = DeviceFunction.current().config
        env = CompileEnvironment.current()
        block_id = env.get_block_id(epoch.projected.spec.shape[2])
        assert block_id is not None
        source = env.block_sizes[block_id].block_size_source
        assert isinstance(source, ReductionLoopBlockSizeSource)
        assert DeviceFunction.current().resolved_block_size(block_id) == 128
        present = "reduction_loops" in config.config
        original_list = config.reduction_loops
        changed = list(original_list)
        while len(changed) <= source.reduction_loop:
            changed.append(None)
        try:
            config.config["reduction_loops"] = changed
            changed[source.reduction_loop] = 64
            assert DeviceFunction.current().config is config
            assert DeviceFunction.current().resolved_block_size(block_id) == 64
            with pytest.raises(chain._UnsupportedChain, match="K geometry changed"):
                epoch.projected.check()
        finally:
            if present:
                config.config["reduction_loops"] = original_list
            else:
                config.config.pop("reduction_loops")
        epoch.check()

    _capture(check)


def test_same_config_object_changed_result_tile_rejects():
    def check(epoch):
        config = DeviceFunction.current().config
        env = CompileEnvironment.current()
        block_id = env.get_block_id(epoch.projected.spec.shape[1])
        assert block_id is not None
        index = env.config_spec.block_sizes.block_id_to_index(block_id)
        blocks = config.block_sizes
        before = list(blocks)
        assert before[index] == 64
        try:
            blocks[index] = 32
            assert DeviceFunction.current().config is config
            assert config.block_sizes is blocks
            assert DeviceFunction.current().resolved_block_size(block_id) == 32
            with pytest.raises(chain._UnsupportedChain, match="graph changed"):
                epoch.projected.check()
        finally:
            blocks[:] = before
        epoch.check()

    _capture(check)
