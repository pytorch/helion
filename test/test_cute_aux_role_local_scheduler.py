from __future__ import annotations

import ast
from collections import Counter
from unittest.mock import patch

import pytest
import torch

from .test_cute_batched_aux_tma import _batched_aux
from .test_cute_batched_aux_tma import _config
from .test_cute_batched_aux_tma import _cpu_codegen
from .test_cute_batched_aux_tma import _inputs
from .test_cute_batched_aux_tma import _rank_two_aux
from .test_cute_scheduler_mailbox import _plain_matmul
import helion
from helion import exc
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_ROLE_LOCAL_SCHEDULER_CONFIG_KEY as KEY,
)
from helion._compiler.program_id import Tcgen05PersistentProgramIDs
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = skipUnlessBackends(["cute"])


def _rank2_inputs(
    dtype: torch.dtype, m: int = 256, n: int = 512
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((m, 128), dtype=dtype),
        torch.empty((128, n), dtype=dtype),
        torch.empty((m, n), dtype=dtype),
    )


def _payload_calls(source: str) -> Counter[str]:
    """Payload operation ASTs are unchanged when only work transport changes."""
    result: Counter[str] = Counter()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {
            "copy",
            "gemm",
            "fence_acq_rel_cta",
            "cp_async_bulk_wait_group",
        }:
            result[ast.dump(node)] += 1
    return result


@pytest.mark.parametrize("rank", (2, 3))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("registers", (128, 256))
def test_default_identity_and_static_role_payloads(
    rank: int, dtype: torch.dtype, registers: int
) -> None:
    kernel = _rank_two_aux if rank == 2 else _batched_aux
    values = _rank2_inputs(dtype) if rank == 2 else _inputs(dtype=dtype)
    with _cpu_codegen():
        bound = kernel._bind_isolated(values)
        config = _config(block_sizes=[128, 128, 64], tcgen05_consumer_regs=registers)
        original = bound.to_code(config)
        explicit_false = bound.to_code(
            helion.Config.from_dict(config.config | {KEY: False})
        )
        enabled = helion.Config.from_dict(config.config | {KEY: True})
        source = bound.to_code(enabled)
        assert original == explicit_false
        assert "tcgen05_sched_pipeline" in original
        assert "tcgen05_sched_pipeline" not in source
        assert "tcgen05_work_tile_smem" not in source
        assert source.count("StaticPersistentTileScheduler.create(") == 4
        assert "cutlass.Int32(7)" in source and "block=(32, 8, 1)" in source
        assert _payload_calls(source) == _payload_calls(original)
        assert (
            "tcgen05_aux_pipeline.producer_tail(tcgen05_aux_pipeline_producer_state)"
            in source
        )
        assert "cute.arch.sync_threads()" in source
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            _, restored = generation.canonicalize_flat(generation.flatten(enabled))
        if rank == 3:
            assert restored.config[KEY] is True
            assert restored.config["tcgen05_consumer_regs"] == registers
        else:
            # This small rank2 family's existing search surface projects TMA
            # back to SIMT; explicit TMA remains a supported codegen path.
            assert restored.config["tcgen05_aux_load_mode"] == "simt"
            assert restored.config[KEY] is False


@pytest.mark.parametrize("order", ([0, 1, 2], [1, 2, 0], [2, 0, 1]))
def test_batched_permutations_have_same_four_scheduler_parameters(
    order: list[int],
) -> None:
    with _cpu_codegen():
        source = _batched_aux._bind_isolated(_inputs(batches=5)).to_code(
            _config(block_sizes=[128, 128, 64], loop_orders=[order], **{KEY: True})
        )
    params = [
        ast.dump(node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "PersistentTileSchedulerParams"
    ]
    assert len(params) == 4 and len(set(params)) == 1


@pytest.mark.parametrize(
    "overrides",
    (
        {"pid_type": "flat"},
        {"tcgen05_aux_load_mode": "simt"},
        {"tcgen05_persistence_model": "clc_persistent"},
        {"tcgen05_cluster_m": 2},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_warp_spec_store_warps": 1},
        {"tcgen05_warp_spec_c_input_warps": 0},
        {"tcgen05_warp_spec_scheduler_warps": 0},
        {"tcgen05_num_epi_warps": 2},
        {"tcgen05_layout_overrides_epi_tile_m": 64},
        {"tcgen05_l2_swizzle_size": 2},
        {"l2_groupings": [2]},
        {"tcgen05_grouped_mode": "static"},
        {"tcgen05_flat_role_coordinates": True},
    ),
)
def test_config_topology_guard(overrides: dict[str, object]) -> None:
    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(_inputs())
        state = bound.config_spec._cute_tcgen05_config
        config = _config(block_sizes=[128, 128, 64]).config
        assert state._aux_role_local_scheduler_config_supported(config)
        assert not state._aux_role_local_scheduler_config_supported(config | overrides)


@pytest.mark.parametrize("axis,size", (("m", 127), ("n", 511), ("k", 127)))
def test_static_scheduler_does_not_admit_padded_logical_tails(
    axis: str, size: int
) -> None:
    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(
            _inputs(
                m=size if axis == "m" else 128,
                n=size if axis == "n" else 512,
                k=size if axis == "k" else 128,
            )
        )
        state = bound.config_spec._cute_tcgen05_config
        assert not state._aux_role_local_scheduler_config_supported(_config().config)


def test_explicit_reject_and_search_repair() -> None:
    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(_inputs())
        config = _config(block_sizes=[128, 128, 64], **{KEY: True})
        with pytest.raises(
            exc.InvalidConfig, match="static.*AUX|AUX.*static|tcgen05_aux_role_local"
        ):
            bound.to_code(
                helion.Config.from_dict(config.config | {"l2_groupings": [2]})
            )
        with bound.env:
            values = dict(config.config)
            bound.config_spec.normalize(values)
            values["l2_groupings"] = [2]
            before = dict(values)
            bound.config_spec._cute_tcgen05_config.normalize_strategy(
                values, fix_invalid=True
            )
        assert values == before | {KEY: False}


def test_no_aux_plain_matmul_remains_ineligible() -> None:
    with _cpu_codegen():
        bound = _plain_matmul._bind_isolated(_rank2_inputs(torch.bfloat16)[:2])
        state = bound.config_spec._cute_tcgen05_config
        assert not state._aux_role_local_scheduler_config_supported(_config().config)
        assert state.aux_role_local_scheduler_seed_configs([_config()]) == [_config()]


def test_seed_prefix_and_search_domain_are_preserved() -> None:
    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(_inputs())
        state = bound.config_spec._cute_tcgen05_config
        with patch.object(
            state,
            "aux_role_local_scheduler_seed_configs",
            side_effect=lambda seeds: seeds,
        ):
            original = state.autotune_seed_configs()
        actual = state.autotune_seed_configs()
        eligible = [
            seed
            for seed in original
            if state._aux_role_local_scheduler_config_supported(seed.config)
        ]
        assert len(original) == 6 and len(eligible) == 6
        assert actual[: len(original)] == original
        assert [seed.config for seed in actual[len(original) :]] == [
            seed.config | {KEY: True} for seed in eligible
        ]
        assert (
            state.aux_role_local_scheduler_seed_configs(actual[len(original) :])
            == actual[len(original) :]
        )
        with bound.env:
            field = bound.config_spec._flat_fields()[KEY]
            assert isinstance(field, EnumFragment)
            assert field.default() is False and field.search_values() == [False, True]
            generation = ConfigGeneration(bound.config_spec)
            for seed in actual[len(original) :]:
                _, restored = generation.canonicalize_flat(generation.flatten(seed))
                assert restored.config[KEY] is True


def test_shared_loop_elision_is_a_final_codegen_requirement() -> None:
    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(_inputs())
        with (
            patch.object(
                Tcgen05PersistentProgramIDs,
                "_tcgen05_shared_loop_has_meaningful_work",
                return_value=True,
            ),
            pytest.raises(
                exc.BackendUnsupported, match="complete shared-loop elimination"
            ),
        ):
            bound.to_code(_config(block_sizes=[128, 128, 64], **{KEY: True}))


def test_aux_coordinates_must_be_proven_at_final_lowering() -> None:
    original = Tcgen05PersistentProgramIDs._role_local_dependency_stmts

    def without_aux_coordinates(
        self: Tcgen05PersistentProgramIDs,
        shared_body: list[ast.stmt],
        role_stmts: list[ast.stmt],
    ) -> list[ast.stmt]:
        if any("tcgen05_aux_l2_anchor" in ast.unparse(stmt) for stmt in role_stmts):
            return []
        return original(self, shared_body, role_stmts)

    with _cpu_codegen():
        bound = _batched_aux._bind_isolated(_inputs())
        with (
            patch.object(
                Tcgen05PersistentProgramIDs,
                "_role_local_dependency_stmts",
                without_aux_coordinates,
            ),
            pytest.raises(
                exc.BackendUnsupported, match="complete post-L2 tile coordinates"
            ),
        ):
            bound.to_code(_config(block_sizes=[128, 128, 64], **{KEY: True}))


@pytest.mark.parametrize("rank", (2, 3))
def test_runtime_fixture_emitted_geometry_has_multiple_waves(rank: int) -> None:
    m = 2048 if rank == 2 else 1024
    kernel = _rank_two_aux if rank == 2 else _batched_aux
    values = (
        _rank2_inputs(torch.bfloat16, m=m, n=2048)
        if rank == 2
        else _inputs(m=m, n=2048)
    )
    with _cpu_codegen():
        code = kernel._bind_isolated(values).to_code(
            _config(block_sizes=[128, 128, 64], **{KEY: True})
        )
    tree = ast.parse(code)
    mma_shapes = [
        ast.literal_eval(node.args[-2])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "make_trivial_tiled_mma"
    ]
    assert mma_shapes == [(128, 128)]
    blocks = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("_BLOCK_SIZE_")
    }
    assert blocks[f"_BLOCK_SIZE_{rank - 2}"] == blocks[f"_BLOCK_SIZE_{rank - 1}"] == 128
    assert (m // 128) * (2048 // 128) * (1 if rank == 2 else 3) == (
        256 if rank == 2 else 384
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rank", (2, 3))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("registers", (128, 256))
def test_static_aux_scheduler_runtime(
    rank: int, dtype: torch.dtype, registers: int
) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05 FP16/BF16 support")
    # Explicit M128/N128: >152 actual work tiles, exercising multiple work waves.
    kernel = _rank_two_aux if rank == 2 else _batched_aux
    for seed in range(5):
        torch.manual_seed(seed)
        leading = () if rank == 2 else (3,)
        m = 2048 if rank == 2 else 1024
        a = torch.randn((*leading, m, 128), device=DEVICE, dtype=dtype) * 0.05
        b = torch.randn((*leading, 128, 2048), device=DEVICE, dtype=dtype) * 0.05
        bias = torch.randn((*leading, m, 2048), device=DEVICE, dtype=dtype) * 0.1
        if rank == 2:
            args = (a, b, bias)
            reference = a.double() @ b.double() + bias.double()
        else:
            scale = torch.rand_like(bias)
            out = torch.empty_like(bias)
            args = (a, b, bias, scale, out)
            reference = (a.double() @ b.double()) * scale.double() + bias.double()
        frozen = tuple(value.clone() for value in args[: 3 if rank == 2 else 4])
        bound = kernel._bind_isolated(args)
        fn = bound.compile_config(
            _config(
                block_sizes=[128, 128, 64],
                tcgen05_consumer_regs=registers,
                **{KEY: True},
            )
        )
        output = fn(*args)
        torch.testing.assert_close(output.double(), reference, atol=0.003, rtol=0.01)
        first = output.clone()
        repeated = fn(*args)
        assert torch.equal(repeated, first)
        if rank == 2:
            assert repeated.data_ptr() != output.data_ptr()
            assert torch.equal(output, first)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn(*args)
        graph.replay()
        assert torch.equal(captured, first)
        if rank == 2:
            assert captured.data_ptr() not in (output.data_ptr(), repeated.data_ptr())
        for _repeat in range(3):
            captured.fill_(float("nan"))
            graph.replay()
            assert torch.equal(captured, first)
            if rank == 2:
                assert torch.equal(output, first)
                assert torch.equal(repeated, first)
        for value, original in zip(args[: len(frozen)], frozen, strict=True):
            assert torch.equal(value, original)
