from __future__ import annotations

import ast
from copy import deepcopy
from itertools import pairwise
import random
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.cute_population_contracts import checked_initial_population

import helion
from helion._compiler.autotuner_heuristics import cute as heuristics
from helion._compiler.cute.grouped_ab_k import physical_grouped_ab_k
from helion._compiler.cute.grouped_ab_k import prove_grouped_ab_k_loop
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_device_split_sizes

KEY = TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY
SHIFTED = "tcgen05_grouped_d_shift"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _observed_logical_k(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, int]:
    m, k = a.shape
    groups, n, _ = b.shape
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for group_tile, rows, columns in hl.tile(
        [groups, m, n], block_size=[1, block_m, block_n]
    ):
        group = group_tile.index.sum()
        start = offsets[group]
        length = offsets[group + 1] - offsets[group]
        row_index = start + rows.index
        valid = rows.index < length
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile(k, block_size=block_k):
            left = hl.load(a, [row_index, inner], extra_mask=valid[:, None])
            acc = torch.addmm(acc, left, b[group, columns, inner].T)
        hl.store(out, [row_index, columns], acc, extra_mask=valid[:, None])
    return out, block_k


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _unsupported_k_loop(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor, variant: int
) -> torch.Tensor:
    hl.specialize(variant)
    m, k = a.shape
    groups, n, _ = b.shape
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for group_tile, rows, columns in hl.tile(
        [groups, m, n], block_size=[1, block_m, block_n]
    ):
        group = group_tile.index.sum()
        start = offsets[group]
        length = offsets[group + 1] - offsets[group]
        row_index = start + rows.index
        valid = rows.index < length
        acc = hl.full(
            [rows, columns], 1.0 if variant == 1 else 0.0, dtype=torch.float32
        )
        begin = 64 if variant == 2 else 0
        for inner in hl.tile(begin, k, block_size=block_k):
            left = hl.load(a, [row_index, inner], extra_mask=valid[:, None])
            if variant == 3:
                left = left * 1.0
            acc = torch.addmm(
                acc,
                left,
                b[group, columns, inner].T,
                alpha=2 if variant == 4 else 1,
            )
            if variant == 5:
                acc = acc * 1.0
        hl.store(out, [row_index, columns], acc, extra_mask=valid[:, None])
    return out


def _args(
    *,
    m: int = 640,
    k: int = 256,
    n: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    offset_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((m, k), dtype=dtype),
        torch.empty((4, n, k), dtype=dtype),
        torch.tensor([0, 0, 1, 129, m], dtype=offset_dtype),
    )


def _config(cluster_m: int, *, stages: int = 2) -> helion.Config:
    return helion.Config(
        block_sizes=[256, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=cluster_m,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=stages,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=128,
        tcgen05_grouped_ab_tile_k=128,
    )


def _plans(source: str) -> list[dict[str, object]]:
    return next(
        ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )


@pytest.mark.parametrize("cluster_m", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_physical128_keeps_observed_logical64_and_both_loop_steps(
    cluster_m: int, dtype: torch.dtype, offset_dtype: torch.dtype
) -> None:
    args = _args(n=160, dtype=dtype, offset_dtype=offset_dtype)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, args)
        block = bound.config_spec.block_sizes[2]
        assert block.max_size == block.declared_max_size == 64
        requested = _config(cluster_m)
        config = bound.config_spec.normalized_config(requested)
        assert requested.block_sizes[2] == config.block_sizes[2] == 64
        source = bound.to_code(config)
        plans = _plans(source)
        grouped = next(
            p for p in plans if p["kind"] == "tcgen05_grouped_static_persistent"
        )
        ab = next(p for p in plans if p["kind"] == "tcgen05_ab_tma")
        assert grouped["bk"] == ab["bk"] == 128
        _validate_tcgen05_grouped_device_split_sizes(grouped, args[2])
        loops = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "tile_offset_2"
        ]
        assert len(loops) == 2
        for loop in loops:
            assert isinstance(loop.iter, ast.Call)
            assert ast.unparse(loop.iter.args[2]) == "cutlass.Int32(128)"
        namespace: dict[str, object] = {}
        exec(compile(source, "<physical grouped K128>", "exec"), namespace)
        launcher = Mock()
        result, observed = namespace["_observed_logical_k"](*args, _launcher=launcher)
        assert result.shape == (640, 160)
        assert observed == 64
        launcher.assert_called_once()
        assert block._fragment(bound.config_spec).high == 64


@pytest.mark.cuda
@pytest.mark.parametrize("cluster_m", [1, 2])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("immutable", [False, True])
def test_physical128_cuda_live_offsets_and_graph_replay(
    cluster_m: int, offset_dtype: torch.dtype, immutable: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family grouped MMA")
    generator = torch.Generator(device="cuda").manual_seed(972)
    n = 256 if immutable else 160
    a = torch.randn((640, 256), device="cuda", generator=generator).bfloat16()
    b = torch.randn((4, n, 256), device="cuda", generator=generator).bfloat16()
    b.add_(torch.arange(4, device="cuda")[:, None, None])
    offsets = torch.tensor([0, 0, 1, 129, 640], device="cuda", dtype=offset_dtype)
    boundaries = [0, 0, 1, 129, 640]
    args = (a, b, offsets)
    config = _config(cluster_m)
    if immutable:
        config.config["tcgen05_grouped_full_allocation_a"] = True
    bound = _observed_logical_k._bind_isolated(args)
    bound.set_config(config)

    def reference() -> torch.Tensor:
        result = torch.empty((640, n), device="cuda", dtype=torch.float32)
        for group, (begin, end) in enumerate(pairwise(boundaries)):
            result[begin:end] = a[begin:end].float() @ b[group].float().T
        return result

    result, observed = bound(*args)
    assert observed == config.block_sizes[2] == 64
    torch.testing.assert_close(result, reference(), rtol=1e-3, atol=1e-3)
    boundaries[1] = 1
    offsets[1] = 1
    result, observed = bound(*args)
    assert observed == 64
    torch.testing.assert_close(result, reference(), rtol=1e-3, atol=1e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, observed = bound(*args)
    assert observed == 64
    boundaries[1] = 0
    offsets[1] = 0
    graph.replay()
    torch.testing.assert_close(captured, reference(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("variant", [1, 2, 3, 5])
def test_noncanonical_k_loops_cannot_request_physical128(variant: int) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_unsupported_k_loop, (*_args(), variant))
        assert not bound.config_spec._cute_tcgen05_config.grouped_ab_k_fusion_supported
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_config(2))


@pytest.mark.parametrize("alpha,beta", [(2, 1), (1, 0), (True, 1)])
def test_nonunit_addmm_arguments_fail_the_closed_loop_proof(
    alpha: object, beta: object
) -> None:
    # The frontend currently rejects nondefault kwargs itself. Exercise the
    # backend guard against a future lowering supplying an otherwise valid FX
    # contraction with those kwargs.
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, _args())
        graphs = bound.host_function.device_ir.graphs
        mma = next(
            node
            for info in graphs
            for node in info.graph.nodes
            if node.target is torch.ops.aten.addmm.default
        )
        lhs = mma.args[1]
        rhs = mma.args[2].args[0]
        options = {
            "lhs_load": lhs,
            "rhs_load": rhs,
            "static_k": 256,
            "zero_seed": True,
            "exclusive_region": True,
        }
        assert prove_grouped_ab_k_loop(bound.env, graphs, mma, **options)
        original = mma.kwargs
        try:
            mma.kwargs = {"alpha": alpha, "beta": beta}
            assert not prove_grouped_ab_k_loop(bound.env, graphs, mma, **options)
        finally:
            mma.kwargs = original


@pytest.mark.parametrize(
    "mutation", ["load", "store", "output", "carry", "rhs", "begin", "end", "k_mask"]
)
def test_unmatched_k_dataflow_and_effects_fail_closed(mutation: str) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, _args())
        graphs = bound.host_function.device_ir.graphs
        mma = next(
            node
            for info in graphs
            for node in info.graph.nodes
            if node.target is torch.ops.aten.addmm.default
        )
        lhs, rhs = mma.args[1], mma.args[2].args[0]
        options = {
            "lhs_load": lhs,
            "rhs_load": rhs,
            "static_k": 256,
            "zero_seed": True,
            "exclusive_region": True,
        }
        assert prove_grouped_ab_k_loop(bound.env, graphs, mma, **options)
        graph = mma.graph
        output = next(node for node in graph.nodes if node.op == "output")
        if mutation == "output":
            output.args = ([mma, lhs],)
        elif mutation in ("begin", "end"):
            loop = next(
                node
                for info in graphs
                for node in info.graph.nodes
                if node.target is _tracing_ops._for_loop
            )
            values = list(loop.args)
            values[1 if mutation == "begin" else 2] = [
                64 if mutation == "begin" else 192
            ]
            loop.args = tuple(values)
        else:
            with graph.inserting_before(
                mma if mutation in ("rhs", "k_mask") else output
            ):
                if mutation == "load":
                    node = graph.call_function(memory_ops.load, lhs.args, lhs.kwargs)
                    node.meta.update(lhs.meta)
                elif mutation == "store":
                    graph.call_function(
                        memory_ops.store, (lhs.args[0], lhs.args[1], lhs)
                    )
                elif mutation == "carry":
                    node = graph.call_function(torch.ops.aten.mul.Tensor, (mma, 1.0))
                    node.meta.update(mma.meta)
                    output.args = ([node],)
                elif mutation == "rhs":
                    node = graph.call_function(
                        torch.ops.aten.mul.Tensor, (mma.args[2], 1.0)
                    )
                    node.meta.update(mma.args[2].meta)
                    mma.args = (mma.args[0], lhs, node)
                else:
                    predicate = graph.call_function(
                        torch.ops.aten.lt.Scalar, (lhs.args[1][1], 128)
                    )
                    masked = graph.call_function(
                        torch.ops.aten.logical_and.default, (lhs.args[2], predicate)
                    )
                    values = list(lhs.args)
                    values[2] = masked
                    lhs.args = tuple(values)
        assert not prove_grouped_ab_k_loop(bound.env, graphs, mma, **options)


def test_partial_physical_k_tail_fails_closed() -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, _args(k=192))
        assert not bound.config_spec._cute_tcgen05_config.grouped_ab_k_fusion_supported
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_config(2))


@pytest.mark.parametrize(
    "invalid_geometry", ["rows", "columns", "cluster_n", "source64"]
)
def test_physical_k_requires_an_existing_native_worklist_profile(
    invalid_geometry: str,
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, _args())
        config = _config(2)
        if invalid_geometry == "rows":
            config.block_sizes[0] = 128
        elif invalid_geometry == "columns":
            config.block_sizes[1] = 64
        elif invalid_geometry == "cluster_n":
            config.config["tcgen05_cluster_n"] = 2
        else:
            config.config["tcgen05_grouped_worklist_source_m_tile"] = 64
        with pytest.raises(InvalidConfig, match="requires"):
            bound.config_spec.normalized_config(config)
        if invalid_geometry == "source64":
            return  # The original dense-local admission rejects this carrier first.
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert config.get(KEY) is None
        assert config.block_sizes[2] == 64


@pytest.mark.parametrize("cluster_m,stages", [(1, 6), (2, 7)])
def test_physical_k_uses_its_own_smem_footprint(cluster_m: int, stages: int) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, _args())
        config = _config(cluster_m, stages=stages)
        legacy = deepcopy(config)
        legacy.config.pop(KEY)
        bound.config_spec.normalized_config(legacy)
        with pytest.raises(InvalidConfig, match="SMEM"):
            bound.config_spec.normalized_config(config)
        repaired = deepcopy(config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired[KEY] == 128 and repaired.block_sizes[2] == 64
        required = tcgen05_grouped_worklist_smem_bytes(
            group_count=4,
            device_split_sizes=True,
            sched_stage_count=1,
            bm=128 * cluster_m,
            bn=128,
            bk=128,
            dtype_bytes=2,
            output_dtype_bytes=4,
            ab_stages=repaired["tcgen05_ab_stages"],
            acc_stages=2,
            c_stages=2,
            cluster_m=cluster_m,
        )
        assert required <= 232448
        ast.parse(bound.to_code(repaired))


def test_none_keeps_legacy_source_and_fused_seeds_keep_logical_cap() -> None:
    args = _args(m=2048, k=512)
    device_tiles = heuristics._tcgen05_grouped_device_tile_seeds

    def fusion_era_tiles(*arguments, **keywords):
        # Later K64 source224/256 additions have their own prefix coverage.
        if (
            keywords["source_m_tile"] in (224, 256)
            and keywords.get("physical_k") is None
        ):
            return []
        return device_tiles(*arguments, **keywords)

    with (
        _forbid_cuda_work(),
        _target(),
        _mock_cuda_unavailable(),
        patch.object(
            heuristics,
            "_tcgen05_grouped_device_tile_seeds",
            side_effect=fusion_era_tiles,
        ),
    ):
        with patch(
            "helion._compiler.autotuner_heuristics.cute._tcgen05_grouped_physical_k_seeds",
            return_value=[],
        ):
            old = _cpu_bind(_observed_logical_k, args)
            old_seeds = deepcopy(
                [
                    seed
                    for seed in old.config_spec.compiler_seed_configs
                    if seed.get(SHIFTED) is not True
                ]
            )
            old_default = deepcopy(old.config_spec.compiler_default_config)
        bound = _cpu_bind(_observed_logical_k, args)
        all_seeds = bound.config_spec.compiler_seed_configs
        seeds = [seed for seed in all_seeds if seed.get(SHIFTED) is not True]
        shifted = [seed for seed in all_seeds if seed.get(SHIFTED) is True]
        assert all_seeds == [*seeds, *shifted]
        assert seeds[: len(old_seeds)] == old_seeds
        assert bound.config_spec.compiler_default_config == old_default
        added = seeds[len(old_seeds) :]
        assert added and all(s[KEY] == 128 and s.block_sizes[2] == 64 for s in added)
        assert bound.config_spec.block_sizes[2]._fragment(bound.config_spec).high == 64
        for seed in added:
            bound.config_spec.normalized_config(seed)
            ast.parse(bound.to_code(seed))
        legacy = _config(2)
        legacy.config.pop(KEY)
        explicit_none = deepcopy(legacy)
        explicit_none.config[KEY] = None
        assert bound.to_code(legacy) == bound.to_code(explicit_none)


def test_physical_k_seeds_reach_cold_full_population() -> None:
    args = _args(m=2048, k=512)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, args)
        state = random.getstate()
        try:
            random.seed(188)
            search = LFBOTreeSearch(
                bound,
                args,
                **LFBOTreeSearch.get_kwargs_from_profile(
                    get_effort_profile("full"), bound.settings
                ),
            )
            population = [
                search.config_gen.unflatten(row)
                for row in checked_initial_population(search)
            ]
            physical_seeds = [
                s
                for _, s in search.config_gen.seed_flat_config_pairs()
                if s.get(KEY) == 128
            ]
            assert physical_seeds and all(seed in population for seed in physical_seeds)
            assert all(seed.block_sizes[2] == 64 for seed in physical_seeds)
        finally:
            random.setstate(state)


@pytest.mark.parametrize(
    "physical,logical", [(64, 64), (True, 64), (128, 128), ("128", 64)]
)
def test_invalid_physical_geometry_is_rejected(physical: object, logical: int) -> None:
    with pytest.raises(InvalidConfig):
        physical_grouped_ab_k({KEY: physical}, logical)
