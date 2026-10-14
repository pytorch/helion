from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_bounded_cache_codegen import _cpu_target

import helion
from helion._compiler.autotuner_heuristics.cute_full_chunk_row import (
    CuteFullChunkRowHeuristic,
)
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


def normalize_rows(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for row in hl.tile(m):
        value = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(value * value, -1) + 1e-6)
        out[row, :] = (value * scale[:, None] * weight[:].to(torch.float32)).to(x.dtype)
    return out


def column_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(1), dtype=x.dtype, device=x.device)
    for column in hl.tile(x.size(1)):
        out[column] = x[:, column].sum(0)
    return out


def normalize_many(
    x: torch.Tensor, others: tuple[torch.Tensor, ...], weight: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for row in hl.tile(m):
        value = x[row, :].to(torch.float32)
        for other in others:
            value = value + other[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(value * value, -1) + 1e-6)
        out[row, :] = (value * scale[:, None] * weight[:].to(torch.float32)).to(x.dtype)
    return out


def normalize_row_bias(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for row in hl.tile(m):
        value = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(value * value, -1) + 1e-6)
        result = value * scale[:, None] * weight[:].to(torch.float32)
        out[row, :] = (result + bias[row][:, None]).to(x.dtype)
    return out


@pytest.fixture(autouse=True)
def cpu_only() -> Iterator[None]:
    initialized = torch.cuda.is_initialized()
    with _cpu_target(), _forbid_native_compile() as native:
        yield
        assert all(mock.call_count == 0 for mock in native)
    assert torch.cuda.is_initialized() == initialized


def bind(
    rows: int,
    size: int,
    dtype: torch.dtype,
    *,
    static: bool = True,
    layout: str = "dense",
) -> BoundKernel:
    if layout == "strided":
        x = torch.empty((rows, size * 2), dtype=dtype)[:, ::2]
    elif layout == "offset":
        x = torch.empty(rows * size + 1, dtype=dtype)[1:].view(rows, size)
    elif layout == "row_stride":
        x = torch.empty((rows, size + 1), dtype=dtype)[:, :size]
    else:
        x = torch.empty((rows, size), dtype=dtype)
    weight = torch.empty(size, dtype=dtype)
    kernel = helion.kernel(normalize_rows, backend="cute", static_shapes=static)
    return kernel._bind_isolated((x, weight))


CASES = [
    (rows, size, dtype)
    for dtype in (torch.bfloat16, torch.float16, torch.float32)
    for rows, size in ((1, 4096), (16, 7168))
] + [(128, 8192, torch.float32), (1024, 4096, torch.float32)]


def seeds_for(bound: BoundKernel, *, packets_only: bool = False) -> list[helion.Config]:
    assert bound.host_function is not None
    with bound.env:
        if packets_only:
            return (
                CuteFullChunkRowHeuristic._packet_seeds(
                    bound.env, bound.host_function.device_ir
                )
                if CuteFullChunkRowHeuristic.is_eligible(
                    bound.env, bound.host_function.device_ir
                )
                else []
            )
        return CuteFullChunkRowHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )


def kernel_ast(code: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
    )


def barrier_count(code: str) -> int:
    return sum(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.sync_threads"
        for node in ast.walk(kernel_ast(code))
    )


@pytest.mark.parametrize(("rows", "size", "dtype"), CASES)
def test_full_chunk_seeds_keep_legacy_prefix_default_and_packets(
    rows: int, size: int, dtype: torch.dtype
) -> None:
    with patch.object(CuteFullChunkRowHeuristic, "get_seed_configs", return_value=[]):
        old = bind(rows, size, dtype)
    bound = bind(rows, size, dtype)
    prior = old.config_spec.compiler_seed_configs
    current = bound.config_spec.compiler_seed_configs
    extras = seeds_for(bound)
    assert len(extras) == 8
    assert current == prior + extras
    assert current == bound.config_spec.compiler_seed_configs
    assert seeds_for(bound) == extras
    assert (
        old.config_spec.compiler_default_config
        == bound.config_spec.compiler_default_config
    )
    with old.env:
        old_generation = ConfigGeneration(old.config_spec)
        old_default = old_generation.unflatten(old_generation.default_flat())
        old_replicated = old.config_spec._flat_fields()["cute_replicated_reduction"]
        assert old_replicated.search_values() == [False]
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        default = generation.unflatten(generation.default_flat())
        replicated = bound.config_spec._flat_fields()["cute_replicated_reduction"]
        assert replicated.search_values() == [False, True]
        assert replicated.pattern_neighbors(False) == [True]
    assert default == old_default
    assert bound.to_code(default) == old.to_code(old_default)
    chunk = 1 << (size - 1).bit_length()
    for seed in extras:
        with bound.env:
            flat, effective = generation.strict_config_pair(seed)
            assert generation.strict_unflatten(flat) == effective
        for field, value in seed.config.items():
            assert effective.config[field] == value
        threads = seed.num_threads[1]
        vector = min(16 // dtype.itemsize, chunk // threads)
        assert seed.reduction_loops == [chunk]
        assert seed["cute_vector_widths"] == [vector, 1]
        code = bound.to_code(effective)
        assert f"ir.VectorType.get([{vector}]," in code
        assert f"block=({threads}, 1, 1)" in code
        assert (
            f"_REDUCTION_BLOCK_{bound.config_spec.reduction_loops[0].block_id} = {chunk}"
            in code
        )
        assert "_fuse_cache_0 = cute.make_rmem_tensor(" in code
        loops = [n for n in kernel_ast(code).body if isinstance(n, ast.For)]
        assert len(loops) == 2
        assert "x.iterator" not in ast.unparse(loops[1])
        prefetch = seed["cute_reduction_reloads"] == ["register_prefetch"]
        assert ("_consume_prefetch_" in code) == prefetch
        assert ("weight.iterator" in ast.unparse(loops[0])) == prefetch
        assert ("weight.iterator" in ast.unparse(loops[1])) != prefetch
        if seed.get("cute_replicated_reduction", False):
            assert barrier_count(code) == 1
            assert "_cute_grouped_reduce_shared_two_stage(" not in ast.unparse(
                kernel_ast(code)
            )
        else:
            assert barrier_count(code) == 0
            assert (
                ast.unparse(kernel_ast(code)).count(
                    "_cute_grouped_reduce_shared_two_stage("
                )
                == 1
            )
        if seed.get("cute_pdl", False):
            assert "griddepcontrol_wait" in code
            assert "l2_evict_last" in code
        if size != chunk:
            assert "else" in ast.unparse(loops[0])
        if dtype == torch.float32 and size == 7168:
            assert "range(2)" in code or threads == 512


@pytest.mark.parametrize(
    ("matrix_count", "threads"),
    [(1, [512, 1024]), (6, [512, 1024]), (7, [1024]), (14, [1024]), (15, [])],
)
def test_total_budget_counts_each_original_load(
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
    assert (
        sorted({s.num_threads[1] for s in seeds_for(bound, packets_only=True)})
        == threads
    )


@pytest.mark.parametrize(
    ("field", "block", "domain", "count"),
    [
        ("num_threads", "reduction", (0, 32), 0),
        ("num_threads", "reduction", (512,), 4),
        ("num_threads", "row", (0,), 0),
        ("cute_vector_widths", "reduction", (1,), 0),
        ("cute_vector_widths", "reduction", (1, 2), 8),
        ("cute_vector_widths", "row", (2,), 0),
        ("cute_lane_layouts", "reduction", ("blocked",), 0),
        ("cute_lane_layouts", "row", ("strided",), 0),
        ("cute_reduction_reloads", "reduction", ("auto", "gmem"), 0),
        ("cute_reduction_reloads", "reduction", ("register",), 2),
    ],
)
def test_narrowed_live_domains(
    field: str, block: str, domain: tuple[object, ...], count: int
) -> None:
    bound = bind(1, 4096, torch.bfloat16)
    spec = bound.config_spec
    sequence = {
        "num_threads": spec.num_threads,
        "cute_vector_widths": spec.cute_vector_widths,
        "cute_lane_layouts": spec.cute_lane_layouts,
        "cute_reduction_reloads": spec.cute_reduction_reloads,
    }[field]
    block_id = (
        spec.reduction_loops[0].block_id
        if block == "reduction"
        else spec.block_sizes[0].block_id
    )
    with patch.object(
        sequence.block_id_lookup(block_id),
        "_fragment",
        return_value=EnumFragment(domain),
    ):
        seeds = seeds_for(bound, packets_only=True)
    assert len(seeds) == count
    if field == "cute_vector_widths" and domain == (1, 2):
        assert all(s["cute_vector_widths"] == [2, 1] for s in seeds)


@pytest.mark.parametrize(
    "kind", ["replicated", "pdl", "policies", "unknown_threads", "thread_subset"]
)
def test_optional_fields_and_actual_search_subsets(kind: str) -> None:
    bound = bind(1, 4096, torch.bfloat16)
    spec = bound.config_spec
    with bound.env:
        fields = spec._flat_fields()
    if kind in ("replicated", "pdl"):
        fields[f"cute_{'replicated_reduction' if kind == 'replicated' else 'pdl'}"] = (
            EnumFragment((False,))
        )
        with patch.object(spec, "_flat_fields", return_value=fields):
            seeds = seeds_for(bound, packets_only=True)
        assert len(seeds) == 4 and not any(s.get("cute_pdl", False) for s in seeds)
    elif kind == "policies":
        with patch.object(spec.load_eviction_policies, "inner", EnumFragment(("",))):
            assert len(seeds_for(bound, packets_only=True)) == 4
    else:
        item = spec.num_threads.block_id_lookup(spec.reduction_loops[0].block_id)
        fragment = EnumFragment((0, 512, 1024), search_choices=(512,))
        with patch.object(item, "_fragment", return_value=fragment):
            if kind == "unknown_threads":
                with patch.object(fragment, "search_values", return_value=None):
                    assert seeds_for(bound, packets_only=True) == []
            else:
                assert {
                    s.num_threads[1] for s in seeds_for(bound, packets_only=True)
                } == {512}


@pytest.mark.parametrize(
    ("size", "dtype", "static", "layout"),
    [
        (7168, torch.bfloat16, False, "dense"),
        (4096, torch.bfloat16, True, "strided"),
        (4096, torch.bfloat16, True, "row_stride"),
        (4096, torch.float64, True, "dense"),
        (16, torch.bfloat16, True, "dense"),
        (131072, torch.bfloat16, True, "dense"),
    ],
)
def test_negative_layout_extent_dtype(
    size: int, dtype: torch.dtype, static: bool, layout: str
) -> None:
    assert (
        seeds_for(bind(3, size, dtype, static=static, layout=layout), packets_only=True)
        == []
    )


def test_unclassified_load_column_and_caps_decline() -> None:
    row_bias = helion.kernel(
        normalize_row_bias, backend="cute", static_shapes=True
    )._bind_isolated((torch.empty((3, 4096)), torch.empty(4096), torch.empty(3)))
    assert seeds_for(row_bias, packets_only=True) == []
    column = helion.kernel(
        column_sum, backend="cute", static_shapes=True
    )._bind_isolated((torch.empty((4096, 16)),))
    assert seeds_for(column, packets_only=True) == []
    bound = bind(1, 4096, torch.bfloat16)
    with patch.object(bound.config_spec, "max_reduction_threads", 128):
        assert seeds_for(bound, packets_only=True) == []
    with patch.object(bound.config_spec, "max_reduction_loop", 1024):
        assert seeds_for(bound, packets_only=True) == []
    with patch.object(bound.config_spec.reduction_loops[0], "allow_full_extent", False):
        assert seeds_for(bound, packets_only=True) == []


def test_offset_layout_never_gains_vector_alignment() -> None:
    bound = bind(1, 4096, torch.bfloat16, layout="offset")
    seeds = seeds_for(bound)
    assert len(seeds) == 8
    for seed in seeds:
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


def permuted_rows(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    indices = (hl.arange(n) + 1) % n
    for row in hl.tile(m):
        value = x[row, indices].to(torch.float32)
        scale = torch.rsqrt(torch.mean(value * value, -1) + 1e-6)
        out[row, :] = (value * scale[:, None] * weight[:].to(torch.float32)).to(x.dtype)
    return out


def test_non_slice_address_declines() -> None:
    bound = helion.kernel(
        permuted_rows, backend="cute", static_shapes=True
    )._bind_isolated((torch.empty((3, 4096)), torch.empty(4096)))
    assert seeds_for(bound, packets_only=True) == []


def test_block_id_mapping_and_policy_indices_are_not_positions() -> None:
    bound = bind(3, 7168, torch.bfloat16)
    spec = bound.config_spec
    original = seeds_for(bound)
    sequences = {
        "num_threads": spec.num_threads,
        "cute_vector_widths": spec.cute_vector_widths,
        "cute_lane_layouts": spec.cute_lane_layouts,
        "cute_reduction_reloads": spec.cute_reduction_reloads,
    }
    for seq in sequences.values():
        seq.reverse()
    reordered = seeds_for(bound)
    assert len(reordered) == len(original) == 8
    for before, after in zip(original, reordered, strict=True):
        for name in sequences:
            assert after[name] == list(reversed(before[name]))
        with bound.env:
            generation = ConfigGeneration(spec)
            flat, effective = generation.strict_config_pair(after)
            assert generation.strict_unflatten(flat) == effective
        assert f"block=({after.num_threads[0]}, 1, 1)" in bound.to_code(effective)
    facts = spec.memory_op_facts
    # Reverse fact traversal without changing their owned config slots.
    with patch.object(spec, "memory_op_facts", list(reversed(facts))):
        assert seeds_for(bound) == reordered


@pytest.mark.parametrize("target", [None, (8, 0), (9, 0), (10, 3)])
def test_pdl_profiles_follow_target_search_domain(
    target: tuple[int, int] | None,
) -> None:
    bound = bind(1, 4096, torch.bfloat16)
    with patch.object(bound.config_spec, "target_device_capability", target):
        seeds = seeds_for(bound)
    assert len(seeds) == (8 if target is not None and target >= (9, 0) else 5)
    assert any(seed.get("cute_replicated_reduction", False) for seed in seeds)


def test_disabled_heuristics_and_other_backend_remain_empty() -> None:
    bound = helion.kernel(
        normalize_rows,
        backend="cute",
        static_shapes=True,
        disable_autotuner_heuristics=True,
    )._bind_isolated((torch.empty((1, 4096)), torch.empty(4096)))
    assert bound.config_spec.compiler_seed_configs == []
    with patch.object(
        type(bound.env),
        "backend_name",
        new_callable=PropertyMock,
        return_value="triton",
    ):
        assert seeds_for(bound, packets_only=True) == []


@pytest.mark.parametrize(
    ("size", "dtype", "static", "layout"),
    [
        (129, torch.bfloat16, True, "dense"),
        (4097, torch.bfloat16, True, "dense"),
        (513, torch.float32, True, "dense"),
        (4096, torch.float64, True, "dense"),
        (7168, torch.bfloat16, False, "dense"),
        (4096, torch.bfloat16, True, "strided"),
        (4096, torch.bfloat16, True, "offset"),
    ],
)
def test_replication_carrier_keeps_shape_structure_and_reference_geometry(
    size: int, dtype: torch.dtype, static: bool, layout: str
) -> None:
    with patch.object(CuteFullChunkRowHeuristic, "get_seed_configs", return_value=[]):
        old = bind(3, size, dtype, static=static, layout=layout)
    bound = bind(3, size, dtype, static=static, layout=layout)
    seeds = seeds_for(bound)
    assert 1 <= len(seeds) <= 8
    packet_seeds = seeds_for(bound, packets_only=True)
    assert (
        bound.config_spec.compiler_seed_configs
        == old.config_spec.compiler_seed_configs + seeds
    )
    with bound.env:
        g = ConfigGeneration(bound.config_spec)
        reference = bound.config_spec.autotune_reference_config()
        default = bound.config_spec.default_config()
        assert bound.config_spec._flat_fields()[
            "cute_replicated_reduction"
        ].search_values() == [False, True]
        for seed in seeds:
            flat, effective = g.strict_config_pair(seed)
            assert g.strict_unflatten(flat) == effective
        if not packet_seeds:
            assert len(seeds) == 1
            expected = helion.Config.from_dict(
                reference.config | {"cute_replicated_reduction": True}
            )
            assert seeds == [expected]
    with old.env:
        assert reference == old.config_spec.autotune_reference_config()
        assert default == old.config_spec.default_config()
    assert bound.to_code(default) == old.to_code(default)
    other = bind(3, 7168 if size != 7168 else 4096, dtype, static=static)
    with bound.env:
        fingerprint = bound.config_spec.structural_fingerprint()
    with other.env:
        assert fingerprint == other.config_spec.structural_fingerprint()


def test_carrier_respects_restricted_reload_and_thread_defaults() -> None:
    bound = bind(1, 4096, torch.bfloat16)
    spec = bound.config_spec
    thread = spec.num_threads.block_id_lookup(spec.reduction_loops[0].block_id)
    reload = spec.cute_reduction_reloads[0]
    with (
        patch.object(thread, "_fragment", return_value=EnumFragment((32,))),
        patch.object(reload, "_fragment", return_value=EnumFragment(("gmem",))),
    ):
        seeds = seeds_for(bound)
        assert len(seeds) == 1
        with bound.env:
            g = ConfigGeneration(spec)
            flat, effective = g.strict_config_pair(seeds[0])
            assert g.strict_unflatten(flat) == effective
            assert effective["cute_reduction_reloads"] == ["gmem"]
            assert effective.num_threads[1] == 32
