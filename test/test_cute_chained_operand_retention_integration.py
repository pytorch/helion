from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention_emission import _config
from .test_cute_chained_operand_retention_emission import _typed_reuse
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_pipeline import _config as _pipeline_config
import helion
from helion import exc
from helion._compiler.cute import chained_pipeline_storage as storage
from helion._compiler.cute import chained_preparation_pipeline as preparation


def _enabled(config, enabled=True):
    return helion.Config.from_dict(
        {**config.config, "cute_chained_operand_retention": enabled}
    )


def _aliases(source):
    return {
        node.targets[0].id: ast.unparse(node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("chain_retained_operand_")
    }


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "masked,transpose", [(False, False), (True, False), (True, True)]
)
def test_complete_generic_route_preserves_default_and_publishes_native_aliases(
    dtype, masked, transpose
):
    args = (
        torch.empty((3, 32, 32), dtype=dtype),
        torch.empty((3, 32, 32), dtype=dtype),
        torch.empty((32, 32), dtype=torch.float32),
        masked,
        transpose,
    )
    with patch.object(
        preparation, "_retain_operands", side_effect=AssertionError("default discovery")
    ):
        ordinary = _source(_typed_reuse, args, _config())
        explicit = _source(_typed_reuse, args, _enabled(_config(), False))
    assert ordinary == explicit and not _aliases(ordinary)
    source = _source(_typed_reuse, args, _enabled(_config()))
    aliases = _aliases(source)
    assert len(aliases) == 2
    assert set(aliases.values()) == {"chain_0_a", "chain_0_b"}
    for value in aliases.values():
        assert source.index(f"{value} = cute.make_tensor") < source.index(
            f"= {value}\n"
        )
    assert "chain_slot_bars" in source
    assert "chain_sync.arrive_mbarrier" in source


def _kda_config(*, leaves=4, cohorts=3, rows=False, group=True):
    config = _pipeline_config(16, pipeline=True)
    config.config.update(
        block_sizes=[128],
        cute_chained_group_contractions=True,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_seed_tile_columns=32,
        cute_chained_pointwise_cache_layout="xor",
        cute_chained_pointwise_unroll=8,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=leaves,
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
        cute_chained_register_islands=True,
        cute_chained_vector_group=group,
        cute_chained_collective_retention=rows,
        cute_chained_operand_retention=True,
    )
    return config


@pytest.mark.parametrize(
    "leaves,cohorts,rows,group",
    [
        (1, 1, False, True),
        (1, 3, True, False),
        (4, 3, False, True),
        (4, 3, True, False),
    ],
)
def test_actual_kda_uses_final_frame_and_all_original_transports(
    leaves, cohorts, rows, group
):
    kernel, args = _kda_fixture()
    recorded = []
    original = storage.finalize_pipeline_storage

    def finalize(plan, pipeline, transports, **kwargs):
        result = original(plan, pipeline, transports, **kwargs)
        assert result is not None
        retained = pipeline.operand_retention
        assert retained is not None and retained.frame is pipeline.frame
        selected = retained.candidates
        assert len(selected) == 4
        assert tuple(item.row_offset for item in selected[:3]) == (0, 0, 32)
        # The general pass also finds the original narrowed inverse snapshot,
        # beyond the first-group triple selected in the manual experiment.
        assert selected[3].node.name == "convert_element_type_20"
        assert selected[3].logical_shape == (32, 32)
        assert selected[3].node is selected[3].operand
        assert len(pipeline.prepared_leaves) == (1 if leaves == 1 else 3)
        for leaf in pipeline.prepared_leaves:
            reads = tuple(
                a.event for a in pipeline.frame.actions if leaf.name in a.reads
            )
            assert leaf.read_events == reads
            assert (leaf.first_event, leaf.last_event) == (reads[0], reads[-1])
        assert kwargs["revision"].pipeline is pipeline
        if leaves == 4 and not rows and group:
            assert pipeline.frame.layout.allocated_bytes == 65536
        recorded.append((pipeline, result))
        return result

    with patch.object(storage, "finalize_pipeline_storage", finalize):
        source = _source(
            kernel,
            args,
            _kda_config(leaves=leaves, cohorts=cohorts, rows=rows, group=group),
        )
    assert len(recorded) == 1
    assert _aliases(source) == {
        "chain_retained_operand_0": "chain_0_a",
        "chain_retained_operand_1": "chain_0_b",
        "chain_retained_operand_2": "cute.domain_offset((32, 0), chain_0_b)",
        "chain_retained_operand_3": "chain_8_a",
    }
    assert "chain_register_island" in source
    assert ("chain_row_collective_" in source) is rows
    assert ("chain_prepared_5_group_output_2" in source) is group
    assert "chained_rectangular_leaf_tma" in source


@pytest.mark.parametrize("leaves,cohorts,rows", [(1, 3, True), (4, 3, True)])
def test_retention_does_not_fabricate_vector_group_activation(leaves, cohorts, rows):
    kernel, args = _kda_fixture()
    with pytest.raises(exc.BackendUnsupported, match="vector grouping requires"):
        _source(kernel, args, _kda_config(leaves=leaves, cohorts=cohorts, rows=rows))
