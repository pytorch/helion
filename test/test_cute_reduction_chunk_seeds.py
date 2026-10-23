from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from helion._compiler.autotuner_heuristics.cute import CuteReductionWideChunkHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteRolledRowLadderHeuristic
from helion._compiler.backend import CuteBackend
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import CuteLaneLayoutSpec
from helion.autotuner.config_spec import CuteVectorWidthSpec
from helion.autotuner.config_spec import NumThreadsSpec
from helion.autotuner.config_spec import ReductionLoopSpec


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("heuristic", "rows", "columns", "expected_chunk"),
    [
        (CuteRolledRowLadderHeuristic, 4096, 3072, 1024),
        (CuteReductionWideChunkHeuristic, 16, 4096, 2048),
        (CuteReductionWideChunkHeuristic, 16, 8191, 2048),
        (CuteReductionWideChunkHeuristic, 16, 50257, 16384),
        (CuteReductionWideChunkHeuristic, 16, 1048576, 262144),
        (CuteRolledRowLadderHeuristic, 16, 1048576, 65536),
    ],
)
def test_reduction_chunks_survive_seed_transfer(
    heuristic: type[CuteReductionWideChunkHeuristic | CuteRolledRowLadderHeuristic],
    rows: int,
    columns: int,
    expected_chunk: int,
    dtype: torch.dtype,
) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=rows))
    spec.reduction_loops.append(ReductionLoopSpec(block_id=1, size_hint=columns))
    for block_id, size in ((0, rows), (1, columns)):
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=size))
        spec.cute_vector_widths.append(
            CuteVectorWidthSpec(block_id=block_id, size_hint=size)
        )
        spec.cute_lane_layouts.append(CuteLaneLayoutSpec(block_id=block_id))
    env = MagicMock(config_spec=spec)
    graph = torch.fx.Graph()
    graph.placeholder("x").meta["val"] = torch.empty(
        (rows, columns), dtype=dtype, device="meta"
    )
    device_ir = MagicMock(graphs=[SimpleNamespace(graph=graph)])

    assert heuristic.is_eligible(env, device_ir)
    seed = heuristic.get_seed_config(env, device_ir)
    assert seed is not None
    assert seed.config["reduction_loops"] == [expected_chunk]
    assert spec.max_reduction_loop is not None
    assert expected_chunk <= spec.max_reduction_loop
    assert seed.config["cute_vector_widths"] == [1, 4 if dtype == torch.float32 else 8]
    spec.compiler_seed_configs = [seed]
    pairs = spec.create_config_generation().seed_flat_config_pairs()
    assert len(pairs) == 1
    assert pairs[0][1].config["reduction_loops"] == [expected_chunk]


@pytest.mark.parametrize(
    ("columns", "max_chunk", "eligible"),
    [
        (2049, None, False),
        (3072, None, False),
        (4095, None, False),
        (4096, None, True),
        (1048576, 1024, False),
    ],
)
def test_wide_chunk_eligibility_uses_rounded_capped_chunk(
    columns: int, max_chunk: int | None, eligible: bool
) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=16))
    spec.reduction_loops.append(ReductionLoopSpec(block_id=1, size_hint=columns))
    spec.max_reduction_loop = max_chunk
    assert (
        CuteReductionWideChunkHeuristic.is_eligible(
            MagicMock(config_spec=spec), MagicMock()
        )
        is eligible
    )
