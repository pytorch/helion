from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from helion import Config
from helion._compiler.autotuner_heuristics.cute import CuteReductionWideChunkHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteRolledClusterLadderHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteRolledRowLadderHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.flydsl.backend import FlyDSLBackend
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import CuteLaneLayoutSpec
from helion.autotuner.config_spec import CuteVectorWidthSpec
from helion.autotuner.config_spec import NumThreadsSpec
from helion.autotuner.config_spec import ReductionLoopSpec


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("heuristic", "rows", "columns", "max_chunk", "expected_chunk"),
    [
        (CuteRolledRowLadderHeuristic, 4096, 3072, 262144, 1024),
        (CuteRolledRowLadderHeuristic, 16, 40, 262144, 32),
        (CuteRolledRowLadderHeuristic, 16, 20, 262144, None),
        (CuteRolledRowLadderHeuristic, 16, 8192, 128, None),
        (CuteRolledRowLadderHeuristic, 16, 1048576, 3000, 2048),
        (CuteReductionWideChunkHeuristic, 16, 1048576, 3000, 2048),
        (CuteReductionWideChunkHeuristic, 16, 1048576, 4096, 4096),
        (CuteReductionWideChunkHeuristic, 16, 4096, 262144, 2048),
        (CuteReductionWideChunkHeuristic, 16, 8191, 262144, 2048),
        (CuteReductionWideChunkHeuristic, 16, 50257, 262144, 16384),
        (CuteReductionWideChunkHeuristic, 16, 1048576, 262144, 262144),
        (CuteRolledRowLadderHeuristic, 16, 1048576, 262144, 65536),
        (CuteRolledClusterLadderHeuristic, 16, 1048576, 3000, 2048),
        (CuteRolledClusterLadderHeuristic, 16, 1048576, 256, None),
    ],
)
def test_reduction_chunks_survive_seed_transfer(
    heuristic: type[
        CuteReductionWideChunkHeuristic
        | CuteRolledRowLadderHeuristic
        | CuteRolledClusterLadderHeuristic
    ],
    rows: int,
    columns: int,
    max_chunk: int,
    expected_chunk: int | None,
    dtype: torch.dtype,
) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.max_reduction_loop = max_chunk
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
    if expected_chunk is None:
        assert seed is None
        return
    assert seed is not None
    assert seed.config["reduction_loops"] == [expected_chunk]
    assert spec.max_reduction_loop is not None
    assert expected_chunk <= spec.max_reduction_loop
    expected_vec = 16 // dtype.itemsize if columns >= 1024 else 1
    assert seed.config.get("cute_vector_widths", [1, 1]) == [1, expected_vec]
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


@pytest.mark.parametrize(
    ("chunk", "cap", "expected"),
    [
        (1, 262144, 8),
        (3, 262144, 8),
        (4, 262144, 8),
        (7, 262144, 8),
        (3000, 3000, 2048),
        (2097152, 262144, 262144),
    ],
)
def test_reduction_seed_transfer_legalizes_chunk_budgets(
    chunk: int, cap: int, expected: int
) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.max_reduction_loop = cap
    spec.reduction_loops.append(ReductionLoopSpec(block_id=0, size_hint=4194304))
    seed = Config(reduction_loops=[chunk])
    spec.compiler_seed_configs = [seed]
    generation = spec.create_config_generation()
    compiler_pairs = generation.seed_flat_config_pairs()
    user_pairs = generation.user_seed_flat_config_pairs([seed])
    assert len(compiler_pairs) == len(user_pairs) == 1
    assert compiler_pairs[0][1].reduction_loops == [expected]
    assert user_pairs[0][1].reduction_loops == [expected]
    assert seed.reduction_loops == [chunk]
    # Direct normalization of a valid power-of-two chunk uses the same live cap.
    direct = Config(reduction_loops=[2097152])
    spec.normalize(direct)
    assert direct.reduction_loops == [2048 if cap == 3000 else 262144]


@pytest.mark.parametrize(("columns", "expected"), [(1000, None), (4194304, 1024)])
def test_extent_covering_chunks_share_persistent_encoding(
    columns: int,
    expected: int | None,
) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.reduction_loops.append(ReductionLoopSpec(block_id=0, size_hint=columns))
    for chunk in (None, columns, 2 * columns):
        seed = Config(reduction_loops=[chunk])
        spec.compiler_seed_configs = [seed]
        generation = spec.create_config_generation()
        compiler = generation.seed_flat_config_pairs()
        user = generation.user_seed_flat_config_pairs([seed])
        assert len(compiler) == len(user) == 1
        assert compiler[0][1].reduction_loops == [expected]
        assert user[0][1].reduction_loops == [expected]


@pytest.mark.parametrize("cap", [0, 1])
def test_ladder_declines_unusable_chunk_caps(cap: int) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=16))
    spec.reduction_loops.append(ReductionLoopSpec(block_id=1, size_hint=1048576))
    spec.max_reduction_loop = cap
    assert (
        CuteRolledRowLadderHeuristic.get_seed_config(
            MagicMock(config_spec=spec), MagicMock()
        )
        is None
    )


@pytest.mark.parametrize("chunk", [None, 128])
def test_indexed_reduction_keeps_warp_cap_without_backend_loop_cap(
    chunk: int | None,
) -> None:
    spec = ConfigSpec(backend=CuteBackend())
    spec.max_reduction_loop = None
    spec.reduction_loops.append(ReductionLoopSpec(block_id=0, size_hint=4096))
    spec.cute_indexed_reduction_block_ids.add(0)
    if chunk is not None:
        assert spec.reduction_loops[0]._encode_flat_value(spec, chunk) == 32
    assert spec.reduction_loops[0]._flat_fragment(spec).default() == 32
    config = Config(reduction_loops=[chunk])
    spec.normalize(config)
    assert config.reduction_loops == [32]


@pytest.mark.parametrize(
    ("chunk", "expected_seed", "expected_direct"),
    [
        (4, None, 4),
        (8192, 8192, 8192),
        (16384, 64, 64),
    ],
)
def test_other_backend_reduction_seed_policy_is_preserved(
    chunk: int,
    expected_seed: int | None,
    expected_direct: int,
) -> None:
    spec = ConfigSpec(backend=FlyDSLBackend())
    spec.reduction_loops.append(ReductionLoopSpec(block_id=0, size_hint=32768))
    seed = Config(reduction_loops=[chunk])
    spec.compiler_seed_configs = [seed]
    generation = spec.create_config_generation()
    compiler = generation.seed_flat_config_pairs()
    user = generation.user_seed_flat_config_pairs([seed])
    if expected_seed is None:
        assert not compiler and not user
    else:
        assert len(compiler) == len(user) == 1
        assert compiler[0][1].reduction_loops == [expected_seed]
        assert user[0][1].reduction_loops == [expected_seed]
    spec.normalize(seed)
    assert seed.reduction_loops == [expected_direct]
