from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_bounded_cache_codegen import _cpu_target
from test.test_cute_full_chunk_row_seeds import CASES
from test.test_cute_full_chunk_row_seeds import bind
from test.test_cute_full_chunk_row_seeds import kernel_ast
from test.test_cute_full_chunk_row_seeds import normalize_many
from test.test_cute_full_chunk_row_seeds import seeds_for

import helion
from helion._compiler.autotuner_heuristics.cute_full_chunk_row import (
    CuteFullChunkRowHeuristic,
)
from helion._compiler.autotuner_heuristics.cute_rolled_cluster_packet import (
    CuteRolledClusterPacketHeuristic,
)
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def cpu_only() -> Iterator[None]:
    initialized = torch.cuda.is_initialized()
    with (
        _cpu_target(),
        _forbid_native_compile() as native,
        patch.object(
            CuteRolledClusterPacketHeuristic, "get_seed_configs", return_value=[]
        ),
    ):
        yield
        assert all(mock.call_count == 0 for mock in native)
    assert torch.cuda.is_initialized() == initialized


def low_seeds(bound: BoundKernel) -> list[helion.Config]:
    assert bound.host_function is not None
    with bound.env:
        return CuteFullChunkRowHeuristic._low_thread_seeds(
            bound.env, bound.host_function.device_ir
        )


@pytest.mark.parametrize(("rows", "size", "dtype"), CASES)
def test_all_added_profiles_preserve_full_prefix_and_generate(
    rows: int, size: int, dtype: torch.dtype
) -> None:
    with patch.object(CuteFullChunkRowHeuristic, "_low_thread_seeds", return_value=[]):
        previous = bind(rows, size, dtype)
    bound = bind(rows, size, dtype)
    additions = low_seeds(bound)
    threads = [128, 256] if size == 4096 else [256, 128]
    assert len(additions) == 4 * len(threads)
    assert bound.config_spec.compiler_seed_configs == [
        *previous.config_spec.compiler_seed_configs,
        *additions,
    ]
    assert len(seeds_for(bound)) <= 17
    assert low_seeds(bound) == additions
    with previous.env:
        default = previous.config_spec.default_config()
        reference = previous.config_spec.autotune_reference_config()
        fingerprint = previous.config_spec.structural_fingerprint()
    with bound.env:
        assert bound.config_spec.default_config() == default
        assert bound.config_spec.autotune_reference_config() == reference
        assert bound.config_spec.structural_fingerprint() == fingerprint
    assert bound.to_code(default) == previous.to_code(default)
    assert [
        (s.num_threads[1], s["cute_replicated_reduction"], s.load_eviction_policies[0])
        for s in additions
    ] == [
        (t, replicated, policy)
        for t in threads
        for replicated in (False, True)
        for policy in ("l2_last", "l1_l2_last")
    ]
    for seed in additions:
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            flat, effective = generation.strict_config_pair(seed)
            assert generation.strict_unflatten(flat) == effective
            assert all(effective.config[k] == v for k, v in seed.config.items())
        assert seed["cute_independent_reduction"] is True
        assert seed["cute_reduction_reloads"] == ["register_prefetch"]
        assert seed["cute_pdl"] is True
        code = bound.to_code(effective)
        tree = kernel_ast(code)
        packet = min(8, 16 // dtype.itemsize)
        sums = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id.startswith("_helion_packet_sum_")
            and isinstance(n.value, ast.Call)
        ]
        assert len(sums) == packet
        assert f"ir.VectorType.get([{packet}]," in code
        assert f"block=({seed.num_threads[1]}, 1, 1)" in code
        assert "_consume_prefetch_" in code and "griddepcontrol_wait" in code
        loops = [n for n in tree.body if isinstance(n, ast.For)]
        assert len(loops) == 2
        assert "weight.iterator" in ast.unparse(loops[0])
        assert "weight.iterator" not in ast.unparse(loops[1])
        assert "x.iterator" not in ast.unparse(loops[1])
        assert ("_cute_grouped_reduce_shared_two_stage(" not in code) == seed[
            "cute_replicated_reduction"
        ]


@pytest.mark.parametrize(
    ("size", "dtype", "static", "layout"),
    [
        (4, torch.bfloat16, True, "dense"),
        (129, torch.bfloat16, True, "dense"),
        (4096, torch.bfloat16, True, "offset"),
        (4096, torch.bfloat16, True, "strided"),
        (4096, torch.bfloat16, True, "row_stride"),
        (4097, torch.float32, True, "dense"),
        (7168, torch.float32, False, "dense"),
        (4096, torch.float64, True, "dense"),
    ],
)
def test_unprofitable_shapes_keep_both_carriers_and_structural_identity(
    size: int, dtype: torch.dtype, static: bool, layout: str
) -> None:
    with patch.object(CuteFullChunkRowHeuristic, "_low_thread_seeds", return_value=[]):
        old = bind(3, size, dtype, static=static, layout=layout)
    bound = bind(3, size, dtype, static=static, layout=layout)
    additions = low_seeds(bound)
    assert bound.config_spec.compiler_seed_configs == [
        *old.config_spec.compiler_seed_configs,
        *additions,
    ]
    with old.env:
        fingerprint = old.config_spec.structural_fingerprint()
    with bound.env:
        assert bound.config_spec.structural_fingerprint() == fingerprint
    other = bind(3, 7168, dtype, static=static)
    with other.env:
        assert other.config_spec.structural_fingerprint() == fingerprint
    if layout == "offset":
        assert len(additions) == 8
        for seed in additions:
            code = bound.to_code(seed)
            assert "_consume_prefetch_" not in code
            assert not any(
                isinstance(n, ast.Call)
                and n.args
                and (
                    ast.unparse(n.func) == "cute.arch.load"
                    or ast.unparse(n.func).startswith("_cute_load_")
                )
                and "x.iterator" in ast.unparse(n.args[0])
                for n in ast.walk(ast.parse(code))
            )
    else:
        assert not additions


@pytest.mark.parametrize(
    ("matrix_count", "threads"),
    [(1, [128, 256]), (2, [256]), (3, [256]), (8, [])],
)
def test_shared_128_slot_budget_counts_all_original_loads(
    matrix_count: int, threads: list[int]
) -> None:
    tensors = tuple(
        torch.empty((3, 8192), dtype=torch.bfloat16) for _ in range(matrix_count)
    )
    bound = (
        bind(3, 8192, torch.bfloat16)
        if matrix_count == 1
        else helion.kernel(
            normalize_many, backend="cute", static_shapes=True
        )._bind_isolated(
            (tensors[0], tensors[1:], torch.empty(8192, dtype=torch.bfloat16))
        )
    )
    assert sorted({s.num_threads[1] for s in low_seeds(bound)}) == threads


@pytest.mark.parametrize(
    ("name", "domain", "count"),
    [
        ("num_threads", (512, 1024), 0),
        ("num_threads", (256,), 4),
        ("cute_vector_widths", (1,), 0),
        ("cute_vector_widths", (2,), 8),
        ("cute_lane_layouts", ("blocked",), 0),
        ("cute_reduction_reloads", ("register", "gmem"), 0),
    ],
)
def test_narrowed_live_sequence_domains(
    name: str, domain: tuple[object, ...], count: int
) -> None:
    bound = bind(1, 4096, torch.bfloat16)
    spec = bound.config_spec
    sequences = {
        "num_threads": spec.num_threads,
        "cute_vector_widths": spec.cute_vector_widths,
        "cute_lane_layouts": spec.cute_lane_layouts,
        "cute_reduction_reloads": spec.cute_reduction_reloads,
    }
    item = sequences[name].block_id_lookup(spec.reduction_loops[0].block_id)
    with patch.object(item, "_fragment", return_value=EnumFragment(domain)):
        assert len(low_seeds(bound)) == count


@pytest.mark.parametrize(
    ("key", "choices", "count"),
    [
        ("cute_independent_reduction", (False,), 0),
        ("cute_replicated_reduction", (False,), 4),
        ("cute_replicated_reduction", (True,), 4),
        ("cute_pdl", (False,), 8),
    ],
)
def test_boolean_domains_and_no_pdl_preference(
    key: str, choices: tuple[bool, ...], count: int
) -> None:
    bound = bind(1, 4096, torch.bfloat16)
    with bound.env:
        fields = bound.config_spec._flat_fields()
    fields[key] = EnumFragment(choices)
    with patch.object(bound.config_spec, "_flat_fields", return_value=fields):
        seeds = low_seeds(bound)
    assert len(seeds) == count
    assert all(seed[key] in choices for seed in seeds)


def test_policy_domain_caps_and_strict_roundtrip() -> None:
    bound = bind(1, 4096, torch.bfloat16)
    spec = bound.config_spec
    with patch.object(spec.load_eviction_policies, "inner", EnumFragment(("",))):
        assert not low_seeds(bound)
    with patch.object(
        spec.load_eviction_policies, "inner", EnumFragment(("", "l2_last"))
    ):
        assert len(low_seeds(bound)) == 4
    with patch.object(spec, "max_reduction_threads", 128):
        assert {s.num_threads[1] for s in low_seeds(bound)} == {128}
    with patch.object(spec, "max_reduction_loop", 1024):
        assert not low_seeds(bound)
    with bound.env:
        reference = spec.autotune_reference_config()
    with patch.object(ConfigGeneration, "strict_unflatten", return_value=reference):
        assert not low_seeds(bound)
