"""CPU admission and source generation for operation-owned radix selection."""

from __future__ import annotations

import ast
from itertools import product
import operator
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.cute_population_contracts import checked_initial_population

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute_radix_selection import (
    CuteRadixSelectionHeuristic,
)
from helion._compiler.cute.ordered_selection import unordered_topk_args
from helion._compiler.cute.radix_selection import match_radix_selection_root
from helion._compiler.cute.radix_selection import radix_schedule
from helion._compiler.cute.radix_selection_config import BOOLEAN_KEYS
from helion._compiler.cute.radix_selection_config import CONFIG_KEYS
from helion._compiler.cute.radix_selection_config import PIPELINE_CHOICES
from helion.autotuner.pattern_search import PatternSearch
import helion.language as hl
from helion.runtime.ref_mode import RefMode

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        yield


def _canonical_rank(values: torch.Tensor) -> torch.Tensor:
    bits = values.float().view(torch.int32)
    magnitude = bits & 0x7FFFFFFF
    sign = bits >> 31
    rank = (magnitude ^ sign) - sign
    return torch.where(magnitude > 0x7F800000, 0x7FFFFFFF, rank)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _payload_selection(
    x: torch.Tensor,
    lengths: torch.Tensor,
    k: hl.constexpr,
    largest: hl.constexpr = True,
    sorted_result: hl.constexpr = False,
):
    rows = x.size(0)
    cols = hl.specialize(x.size(1))
    padded_cols = 1 << (cols - 1).bit_length()
    padded_k = 1 << (k - 1).bit_length()
    values = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, k), dtype=torch.int32, device=x.device)
    for row in hl.grid(rows):
        columns = hl.arange(padded_cols)
        length = lengths[row]
        valid = (columns < cols) & (columns < length)
        data = hl.load(x, [row, columns], extra_mask=valid)
        keys = torch.where(valid, _canonical_rank(data), -2147483648)
        positions = torch.topk(
            keys, k, dim=-1, largest=largest, sorted=sorted_result
        ).indices
        slots = hl.arange(padded_k)
        padded_positions = torch.nn.functional.pad(positions, (0, padded_k - k))
        active = (slots < k) & (padded_positions < cols) & (padded_positions < length)
        payload = hl.load(x, [row, padded_positions], extra_mask=active)
        hl.store(
            values,
            [row, slots],
            torch.where(active, payload, -float("inf")),
            extra_mask=slots < k,
        )
        hl.store(
            indices,
            [row, slots],
            torch.where(active, padded_positions, -1).to(torch.int32),
            extra_mask=slots < k,
        )
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _observed_keys(
    x: torch.Tensor, lengths: torch.Tensor, k: hl.constexpr, cross_slot: hl.constexpr
):
    cols = hl.specialize(x.size(1))
    padded_cols = 1 << (cols - 1).bit_length()
    values = torch.empty((x.size(0), k), dtype=torch.int32, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.grid(x.size(0)):
        columns = hl.arange(padded_cols)
        valid = (columns < cols) & (columns < lengths[row])
        data = hl.load(x, [row, columns], extra_mask=valid)
        keys = torch.where(valid, _canonical_rank(data), -2147483648)
        selected, original_indices = torch.topk(keys, k, dim=0, sorted=False)
        if cross_slot:
            selected = selected.flip(0)
        slots = hl.arange(k)
        hl.store(values, [row, slots], selected)
        hl.store(indices, [row, slots], original_indices)
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _external_output(x: torch.Tensor, lengths: torch.Tensor, output: torch.Tensor):
    cols = hl.specialize(x.size(1))
    padded_cols = 1 << (cols - 1).bit_length()
    for row in hl.grid(x.size(0)):
        columns = hl.arange(padded_cols)
        valid = (columns < cols) & (columns < lengths[row])
        data = hl.load(x, [row, columns], extra_mask=valid)
        keys = torch.where(valid, _canonical_rank(data), -2147483648)
        positions = torch.topk(keys, cols, sorted=False).indices
        slots = hl.arange(cols)
        chosen = hl.load(x, [row, positions], extra_mask=positions < cols)
        hl.store(output, [row, slots], chosen)
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _colliding_names(radix_thread: torch.Tensor, radix_row: torch.Tensor):
    cols = hl.specialize(radix_thread.size(1))
    output = torch.empty(
        (radix_thread.size(0), 7), dtype=torch.int64, device=radix_thread.device
    )
    for row in hl.grid(radix_thread.size(0)):
        columns = hl.arange(32)
        valid = (columns < cols) & (columns < radix_row[row])
        data = hl.load(radix_thread, [row, columns], extra_mask=valid)
        keys = torch.where(valid, _canonical_rank(data), -2147483648)
        positions = torch.topk(keys, 7, sorted=False).indices
        slots = hl.arange(7)
        hl.store(output, [row, slots], positions + 1)
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _raw_length_and_keys(x: torch.Tensor, lengths: torch.Tensor, k: hl.constexpr):
    cols = hl.specialize(x.size(1))
    logical_cols = 1 << (cols - 1).bit_length()
    ranks = torch.empty((x.size(0), k), dtype=torch.int32, device=x.device)
    positions = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    observed = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.grid(x.size(0)):
        columns = hl.arange(logical_cols)
        length = lengths[row]
        valid = (columns < cols) & (columns < length)
        data = hl.load(x, [row, columns], extra_mask=valid)
        keys = torch.where(valid, _canonical_rank(data), -2147483648)
        selected, ids = torch.topk(keys, k, sorted=False)
        slots = hl.arange(k)
        hl.store(ranks, [row, slots], selected)
        hl.store(positions, [row, slots], ids)
        hl.store(observed, [row, slots], length.to(torch.int64) + slots.to(torch.int64))
    return ranks, positions, observed


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _partial_logical_row(x: torch.Tensor, lengths: torch.Tensor):
    cols = hl.specialize(x.size(1))
    output = torch.empty((x.size(0), 7), dtype=torch.int64, device=x.device)
    for row in hl.grid(x.size(0)):
        columns = hl.arange(cols // 2)
        length = lengths[row]
        valid = (columns < cols) & (columns < length)
        data = hl.load(x, [row, columns], extra_mask=valid)
        keys = torch.where(valid, _canonical_rank(data), -2147483648)
        ids = torch.topk(keys, 7, sorted=False).indices
        hl.store(output, [row, hl.arange(7)], ids)
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _source_fanout(x: torch.Tensor, lengths: torch.Tensor):
    cols = hl.specialize(x.size(1))
    original = torch.empty_like(x)
    positions = torch.empty((x.size(0), cols), dtype=torch.int64, device=x.device)
    for row in hl.grid(x.size(0)):
        columns = hl.arange(cols)
        valid = (columns < cols) & (columns < lengths[row])
        data = hl.load(x, [row, columns], extra_mask=valid)
        keys = torch.where(valid, _canonical_rank(data), -2147483648)
        ids = torch.topk(keys, cols, sorted=False).indices
        hl.store(original, [row, columns], data)
        hl.store(positions, [row, columns], ids)
    return original, positions


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _custom_payload(
    x: torch.Tensor,
    lengths: torch.Tensor,
    k: hl.constexpr,
    offset: hl.constexpr,
    masked: hl.constexpr,
    smaller_output: hl.constexpr,
):
    cols = hl.specialize(x.size(1))
    logical_cols = 1 << (cols - 1).bit_length()
    padded_k = 1 << (k - 1).bit_length()
    result = torch.empty(
        (x.size(0), k - 1 if smaller_output else k), dtype=x.dtype, device=x.device
    )
    for row in hl.grid(x.size(0)):
        columns = hl.arange(logical_cols, dtype=torch.int64)
        length = lengths[row]
        valid = (columns < cols) & (columns < length)
        data = hl.load(x, [row, columns], extra_mask=valid)
        keys = torch.where(valid, _canonical_rank(data), -(1 << 31))
        positions = torch.topk(keys, k, sorted=False).indices
        positions = torch.nn.functional.pad(positions, (0, padded_k - k))
        slots = hl.arange(padded_k, dtype=torch.int64)
        active = (slots < k) & (positions < cols) & (positions < length)
        if masked:
            active = active & (slots % 2 == 0)
        if offset:
            positions = positions + 1
        payload = hl.load(x, [row, positions], extra_mask=active)
        hl.store(result, [row, slots], payload, extra_mask=slots < k)
    return result


def _args(dtype: torch.dtype = torch.bfloat16, cols: int = 33):
    return torch.empty((3, cols), dtype=dtype), torch.tensor([0, 5, cols]), 17


def _radix_config(bound, **changes):
    config = bound.config_spec.default_config()
    config.config.update(
        cute_selection_algorithm="radix",
        cute_radix_threads=1024,
        cute_radix_cluster_ctas=4,
        cute_radix_candidate_capacity=4096,
        cute_radix_histogram_mode="direct",
    )
    config.config.update(changes)
    return config


@pytest.mark.parametrize("sorted_value", [None, True, False, 0, "false"])
def test_unordered_parser_requires_literal_false(sorted_value) -> None:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    args = (source, 3, -1, True)
    if sorted_value is not None:
        args += (sorted_value,)
    node = graph.call_function(torch.ops.aten.topk.default, args)
    assert (unordered_topk_args(node) is not None) is (sorted_value is False)
    keyword = graph.call_function(
        torch.ops.aten.topk.default,
        (source, 3),
        {"dim": 0, "largest": False, "sorted": False},
    )
    assert unordered_topk_args(keyword) == (source, 3, 0, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("largest", [True, False])
def test_actual_rank_recipe_generates_bounded_cluster_source(dtype, largest) -> None:
    bound = _cpu_bind(_payload_selection, (*_args(dtype), largest, False))
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None
    assert graph.logical_cols == 64 and graph.cols == 33 and graph.k == 17
    assert graph.index_getters and not graph.value_getters
    source = bound.to_code(_radix_config(bound))
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Set) for node in ast.walk(tree))
    assert "_helion_radix_consumer" not in source
    assert "block=(1024, 1, 1)" in source
    assert "_helion_cute_cluster_shape = (4, 1, 1)" in source
    assert "cute.arch.block_idx()[0]) // 4" in source
    assert "num_bits_per_copy=128" in source
    assert (
        "range_constexpr(2)" in source
        if dtype != torch.float32
        else "range_constexpr(4)" in source
    )
    allocations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "allocate_tensor"
    ]
    sizes = [node.args[1].args[0].elts[0].value for node in allocations]
    assert set(sizes) <= {1, 2, 4, 8, 17, 256, 4096}
    assert 64 not in sizes
    if not largest:
        assert (
            "^ cutlass.Uint32(65535)" in source
            if dtype != torch.float32
            else "^ cutlass.Uint32(4294967295)" in source
        )


def test_live_key_values_keep_f32_int32_rank_and_int64_original_indices() -> None:
    bound = _cpu_bind(_observed_keys, (*_args(), False))
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None and graph.value_getters and graph.index_getters
    source = bound.to_code(_radix_config(bound))
    assert ".bitcast(cutlass.Int32)" in source
    assert "2139095040" in source
    assert "2147483647" in source
    assert "2147483648" in source
    assert "<< cutlass.Int64(32)" not in source
    assert not any(isinstance(node, ast.Set) for node in ast.walk(ast.parse(source)))


def test_cross_slot_consumer_and_sorted_selection_decline() -> None:
    flipped = _cpu_bind(_observed_keys, (*_args(), True))
    assert flipped.config_spec.cute_radix_selection_graph is None
    ordered = _cpu_bind(_payload_selection, (*_args(), True, True))
    assert ordered.config_spec.cute_radix_selection_graph is None
    with pytest.raises(exc.InvalidConfig):
        ordered.to_code(_radix_config(ordered))


@pytest.mark.parametrize("bad_getter", [True, 2, -1])
def test_original_tuple_getter_proof_is_typed_and_bounded(bad_getter) -> None:
    bound = _cpu_bind(_payload_selection, _args())
    info = bound.host_function.device_ir.graphs[0]
    selection = next(
        node for node in info.graph.nodes if node.target is torch.ops.aten.topk.default
    )
    getter = next(iter(selection.users))
    assert getter.target is operator.getitem
    getter.args = (selection, bad_getter)
    with bound.env, bound.host_function:
        assert match_radix_selection_root([info]) is None


def test_final_codegen_rechecks_the_current_recipe() -> None:
    bound = _cpu_bind(_payload_selection, _args())
    info = bound.host_function.device_ir.graphs[0]
    comparison = next(
        node for node in info.graph.nodes if node.target is torch.ops.aten.gt.Scalar
    )
    comparison.args = (comparison.args[0], 0x7F800001)
    with pytest.raises(exc.InvalidConfig):
        bound.to_code(_radix_config(bound))


def test_rank_recipe_requires_integer_comparison_constants() -> None:
    bound = _cpu_bind(_payload_selection, _args())
    info = bound.host_function.device_ir.graphs[0]
    comparison = next(
        node for node in info.graph.nodes if node.target is torch.ops.aten.gt.Scalar
    )
    comparison.args = (comparison.args[0], float(0x7F800000))
    with bound.env, bound.host_function:
        assert match_radix_selection_root([info]) is None


@pytest.mark.parametrize("wrapper", ["same_tensor", "dlpack"])
def test_readonly_proof_rejects_storage_aliases(wrapper) -> None:
    x, lengths, k = _args(cols=32)
    out = x if wrapper == "same_tensor" else torch.utils.dlpack.from_dlpack(x)
    bound = _cpu_bind(_external_output, (x, lengths, out))
    assert bound.config_spec.cute_radix_selection_graph is None
    independent = _cpu_bind(_external_output, (x, lengths, torch.empty_like(x)))
    assert independent.config_spec.cute_radix_selection_graph is not None


def test_packet_alignment_is_bound_cache_specialized() -> None:
    x, lengths, k = _args()
    aligned = _cpu_bind(_payload_selection, (x, lengths, k))
    unaligned_x = torch.empty((3, 34), dtype=x.dtype)[:, 1:]
    assert unaligned_x.data_ptr() % 16 == 2
    unaligned = _cpu_bind(_payload_selection, (unaligned_x, lengths, k))
    assert "num_bits_per_copy=128" in aligned.to_code(_radix_config(aligned))
    assert "num_bits_per_copy=128" not in unaligned.to_code(_radix_config(unaligned))


def test_literal_resource_budget_and_mode_validation() -> None:
    bound = _cpu_bind(_payload_selection, _args())
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None
    config = _radix_config(bound)
    plan = radix_schedule(graph, config.config)
    assert plan.shared_bytes == sum(
        ((count * dtype.itemsize + 15) // 16) * 16
        for _, dtype, count in plan.allocations
    )
    with (
        patch(
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
            return_value=1024,
        ),
        pytest.raises(exc.InvalidConfig),
    ):
        bound.to_code(config)
    for key, value in (
        ("cute_radix_threads", 64),
        ("cute_radix_cluster_ctas", 3),
        ("cute_radix_candidate_capacity", -1),
        ("cute_radix_histogram_mode", "unknown"),
        ("pid_type", "persistent_interleaved"),
    ):
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(_radix_config(bound, **{key: value}))


def test_direct_and_match_are_independent_histogram_modes() -> None:
    bound = _cpu_bind(_payload_selection, _args())
    direct = bound.to_code(_radix_config(bound))
    matched = bound.to_code(_radix_config(bound, cute_radix_histogram_mode="match"))
    assert "cute.arch.match_sync" not in direct
    assert "cute.arch.match_sync" in matched
    scalar = bound.to_code(
        _radix_config(
            bound,
            cute_radix_threads=256,
            cute_radix_cluster_ctas=1,
            cute_radix_candidate_capacity=0,
        )
    )
    assert "map_dsmem_ptr" not in scalar
    assert "candidate_indices" not in scalar


def test_argument_names_do_not_collide_with_generated_locals() -> None:
    x, lengths, k = _args(cols=32)
    bound = _cpu_bind(_colliding_names, (x, lengths))
    assert bound.config_spec.cute_radix_selection_graph is not None
    source = bound.to_code(_radix_config(bound))
    kernel = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    arguments = {argument.arg for argument in kernel.args.args}
    assigned = {
        node.id
        for node in ast.walk(kernel)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert arguments.isdisjoint(assigned)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_narrow_surrogate_preserves_all_f32_rank_order_and_equal_classes(dtype) -> None:
    words = torch.arange(1 << 16, dtype=torch.int32)
    values = words.to(torch.int16).view(dtype)
    magnitude = words & 0x7FFF
    rank16 = torch.where(words >= 0x8000, -magnitude, magnitude)
    infinity = 0x7C00 if dtype == torch.float16 else 0x7F80
    rank16 = torch.where(magnitude > infinity, 0x7FFF, rank16)
    rank32 = _canonical_rank(values)
    order = rank16.argsort()
    narrow = rank16[order].to(torch.int64)
    wide = rank32[order].to(torch.int64)
    same = narrow[1:] == narrow[:-1]
    assert torch.equal(wide[1:][same], wide[:-1][same])
    assert torch.all(wide[1:][~same] > wide[:-1][~same])


@pytest.mark.parametrize("largest", [False, True])
def test_legacy_defaults_and_initial_random_prefix_are_unchanged(largest) -> None:
    args = (*_args(), largest, False)
    with patch.object(
        CuteRadixSelectionHeuristic, "register_facts", return_value=frozenset()
    ):
        before = _cpu_bind(_payload_selection, args)
    after = _cpu_bind(_payload_selection, args)
    assert before.config_spec.default_config() == after.config_spec.default_config()
    assert (
        before.config_spec.compiler_seed_configs
        == after.config_spec.compiler_seed_configs
    )
    populations = []
    states = []
    for bound in (before, after):
        with bound.env:
            random.seed(41)
            search = PatternSearch(bound, args, initial_population=12)
            population = checked_initial_population(search)
            populations.append(
                [
                    {
                        key: value
                        for key, value in search.config_gen.canonicalize_flat(row)[
                            1
                        ].config.items()
                        if key not in CONFIG_KEYS
                    }
                    for row in population
                ]
            )
            states.append(random.getstate())
    assert states[0] == states[1]
    assert populations[1][: len(populations[0])] == populations[0]
    assert len(populations[1]) > len(populations[0])


def test_flags_require_typed_booleans_and_radix_and_trim_requires_largest() -> None:
    bound = _cpu_bind(_payload_selection, _args())
    for key in BOOLEAN_KEYS:
        for value in (0, 1, None, "true"):
            with pytest.raises(exc.InvalidConfig, match="Boolean"):
                bound.config_spec.normalized_config(
                    _radix_config(bound, **{key: value})
                )
        config = bound.config_spec.default_config()
        config.config[key] = True
        with pytest.raises(exc.InvalidConfig, match="requires the radix"):
            bound.config_spec.normalized_config(config)
    smallest = _cpu_bind(_payload_selection, (*_args(), False, False))
    with pytest.raises(exc.InvalidConfig, match="largest=True"):
        smallest.to_code(_radix_config(smallest, cute_radix_valid_domain=True))


_RADIX_SOURCE_FLAGS = tuple(
    key for key in BOOLEAN_KEYS if key != "cute_radix_cached_payload_inverse"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "modes", tuple(product((False, True), repeat=len(_RADIX_SOURCE_FLAGS)))
)
def test_all_flag_combinations_keep_typed_bounded_source_and_allocations(
    dtype: torch.dtype, modes: tuple[bool, ...]
) -> None:
    bound = _cpu_bind(_payload_selection, _args(dtype))
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None
    config = _radix_config(
        bound,
        cute_radix_candidate_capacity=1024,
        **dict(zip(_RADIX_SOURCE_FLAGS, modes, strict=True)),
    )
    plan = radix_schedule(graph, config.config)
    source = bound.to_code(config)
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Set) for node in ast.walk(tree))
    assert "_helion_radix_consumer" not in source
    assert (
        "scope='cluster'" in source
        if plan.ticket_stores
        else "scope='cluster'" not in source
    )
    declarations = {name: (kind, count) for name, kind, count in plan.allocations}
    if plan.cache_keys_enabled:
        key_type = torch.uint32 if dtype == torch.float32 else torch.uint16
        assert declarations["candidate_keys"] == (key_type, 1024)
        assert (
            "allocate_tensor(cutlass.Uint32" in source
            if dtype == torch.float32
            else "allocate_tensor(cutlass.Uint16" in source
        )
        assert "candidate_keys[" in source
    else:
        assert "candidate_keys" not in source
    assert ("selected_indices" in declarations) is not plan.ticket_stores
    assert ("metadata" in declarations) is not plan.ticket_stores
    assert plan.shared_bytes == sum(
        ((kind.itemsize * count + 15) // 16) * 16
        for kind, count in declarations.values()
    )
    if plan.scalar_candidates:
        assert "range_constexpr(1)" in source
    if plan.valid_domain:
        assert "clamped_length = cutlass.Int64(" in source
        assert "short_slot" in source


def test_zero_capacity_bypasses_key_and_scalar_cache_and_c1_avoids_dsm() -> None:
    bound = _cpu_bind(_payload_selection, _args())
    source = bound.to_code(
        _radix_config(
            bound,
            cute_radix_candidate_capacity=0,
            cute_radix_candidate_key_cache=True,
            cute_radix_scalar_candidates=True,
            cute_radix_ticket_stores=True,
            cute_radix_cluster_ctas=1,
        )
    )
    assert "candidate_keys" not in source and "candidate_indices" not in source
    assert "map_dsmem_ptr" not in source and "scope='cluster'" not in source
    assert "range_constexpr(1)" not in source
    assert "selected_indices" not in source


def test_resident_input_register_budget_preserves_reload_path() -> None:
    bound = _cpu_bind(_payload_selection, _args(cols=65537))
    reload = _radix_config(bound, cute_radix_threads=256, cute_radix_cluster_ctas=1)
    bound.config_spec.normalized_config(reload)
    resident = _radix_config(
        bound,
        cute_radix_threads=256,
        cute_radix_cluster_ctas=1,
        cute_radix_resident_input=True,
    )
    with pytest.raises(exc.InvalidConfig, match="register budget"):
        bound.config_spec.normalized_config(resident)


def test_int64_length_clamps_before_narrowing_and_raw_suffix_is_preserved() -> None:
    x = torch.randn(3, 33, dtype=torch.bfloat16)
    lengths = torch.tensor([-(1 << 40), 1 << 40, (1 << 32) + 7], dtype=torch.int64)
    args = (x, lengths, 17)
    bound = _cpu_bind(_raw_length_and_keys, args)
    config = _radix_config(bound, **dict.fromkeys(BOOLEAN_KEYS, True))
    source = bound.to_code(config)
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.startswith("radix_length_slot")
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.args[0]) == "cutlass.Int64"
        for node in ast.walk(ast.parse(source))
    )
    assert source.index("clamped_length = cutlass.Int64(") < source.index(
        "valid_count = cutlass.Int32("
    )
    assert "if radix_clamped_length < 0:" in source
    assert ".bitcast(cutlass.Int32)" in source
    assert "bitcast(cutlass.Int64)" not in source
    bound.set_config(config)
    with patch.object(bound.env.settings, "ref_mode", RefMode.EAGER):
        selected, ids, observed = bound.run_ref(*args)
    assert torch.equal(observed, lengths[:, None] + torch.arange(17))
    # Torch RefMode narrows an Int64 zero-dimensional weak scalar when it is
    # compared with hl.arange's Int32 vector. Native source must keep the raw
    # Int64 comparison; validate reference rank membership within that range.
    lengths = torch.tensor([-5, 19, 33], dtype=torch.int64)
    with patch.object(bound.env.settings, "ref_mode", RefMode.EAGER):
        selected, ids, observed = bound.run_ref(x, lengths, 17)
    raw = torch.zeros((3, 64), dtype=torch.bfloat16)
    valid = (torch.arange(64)[None, :] < 33) & (
        torch.arange(64)[None, :] < lengths[:, None]
    )
    raw[:, :33] = x
    expected = torch.where(valid, _canonical_rank(raw), -(1 << 31))
    assert torch.equal(selected, torch.gather(expected, 1, ids))
    assert torch.equal(
        selected.sort(1).values, expected.topk(17, 1).values.sort(1).values
    )


def test_trim_respects_the_declared_logical_domain() -> None:
    args = (torch.empty(3, 128), torch.empty(3, dtype=torch.int64))
    bound = _cpu_bind(_partial_logical_row, args)
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None and graph.logical_cols == 64 and graph.cols == 128
    config = _radix_config(bound, cute_radix_valid_domain=True)
    plan = radix_schedule(graph, config.config)
    assert plan.valid_cols == 64
    source = bound.to_code(config)
    assert "if radix_clamped_length > 64:" in source
    assert "radix_clamped_length = cutlass.Int64(64)" in source


def test_forwarded_raw_keeps_identity_prefix_masks_and_unmasked_loads_reread() -> None:
    args = _args()
    masked = _cpu_bind(_payload_selection, args)
    masked_source = masked.to_code(
        _radix_config(
            masked, cute_radix_ticket_stores=True, cute_radix_candidate_capacity=0
        )
    )
    unmasked_args = _args(cols=32)
    output = torch.empty_like(unmasked_args[0])
    unmasked = _cpu_bind(_external_output, (unmasked_args[0], unmasked_args[1], output))
    unmasked_source = unmasked.to_code(
        _radix_config(
            unmasked, cute_radix_ticket_stores=True, cute_radix_candidate_capacity=0
        )
    )

    def suffix_loads(source: str) -> int:
        return sum(
            any(
                isinstance(part, ast.Call)
                and isinstance(part.func, ast.Attribute)
                and part.func.attr == "load"
                for part in ast.walk(node.value)
            )
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id.startswith("radix_payload")
                for target in node.targets
            )
        )

    assert suffix_loads(masked_source) == 0
    assert suffix_loads(unmasked_source) > 0
    fanout = _cpu_bind(
        _source_fanout,
        (torch.empty(3, 16, dtype=torch.bfloat16), torch.empty(3, dtype=torch.int64)),
    )
    # A source vector observed at original slots crosses the selected-slot
    # ownership boundary. The initial strategy deliberately declines it.
    assert fanout.config_spec.cute_radix_selection_graph is None
    with pytest.raises(exc.InvalidConfig):
        fanout.to_code(_radix_config(fanout, cute_radix_ticket_stores=True))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("packet_bits", [128, 256])
@pytest.mark.parametrize("unroll", [1, 2, 4])
@pytest.mark.parametrize("resident_input", [False, True])
@pytest.mark.parametrize("payload_inverse", [False, True])
def test_packet_width_and_source_unroll_keep_uniform_candidate_loops(
    dtype, packet_bits, unroll, resident_input, payload_inverse
) -> None:
    args = _args(dtype, cols=35)
    bound = _cpu_bind(_payload_selection, args)
    config = _radix_config(
        bound,
        **{
            key: True
            for key in BOOLEAN_KEYS
            if key
            not in ("cute_radix_resident_input", "cute_radix_cached_payload_inverse")
        },
        cute_radix_resident_input=resident_input,
        cute_radix_cached_payload_inverse=payload_inverse,
        cute_radix_candidate_capacity=1024,
        cute_radix_packet_bits=packet_bits,
        cute_radix_scan_unroll=unroll,
    )
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None
    plan = radix_schedule(graph, config.config)
    assert plan.packet == packet_bits // (8 * args[0].element_size())
    assert plan.cta_cols % plan.packet == 0
    source = bound.to_code(config)
    assert f"num_bits_per_copy={packet_bits}" in source
    assert f"align({packet_bits // 8})" in source
    assert f"% {plan.packet} == 0" in source
    assert "cutlass.Int64(35)" in source
    if unroll > 1 and not resident_input:
        assert f"unroll={unroll}" in source
        # The cached scalar branch still uses ordinary uniform range rounds.
        assert "for radix_packet_round in range(radix_scan_rounds):" in source
    else:
        assert "unroll=" not in source
    if resident_input:
        assert ("radix_resident_values" in source) is not payload_inverse
        assert "radix_resident_keys" in source
        assert "if radix_packet_round < radix_scan_rounds:" in source
    assert "range_constexpr(1)" in source
    assert "_helion_radix_consumer" not in source


def test_wide_packet_has_separate_cache_alignment_fact_and_scalar_fallback() -> None:
    cols = 32
    arena = torch.empty(3 * cols + 8, dtype=torch.bfloat16)
    x = arena[: 3 * cols].view(3, cols)
    shifted = arena[8:].view(3, cols)
    assert x.data_ptr() % 32 == 0
    assert shifted.data_ptr() % 16 == 0 and shifted.data_ptr() % 32 == 16
    lengths = torch.empty(3, dtype=torch.int32)
    aligned = _cpu_bind(_payload_selection, (x, lengths, 17))
    unaligned = _cpu_bind(_payload_selection, (shifted, lengths, 17))
    key = "cute_radix_packet_alignment_32_v1"
    assert aligned.env.bound_runtime_input_specialization_results[key] == 0
    assert unaligned.env.bound_runtime_input_specialization_results[key] == 16
    for bound in (aligned, unaligned):
        assert "num_bits_per_copy=128" in bound.to_code(_radix_config(bound))
    config = _radix_config(aligned, cute_radix_packet_bits=256)
    assert "num_bits_per_copy=256" in aligned.to_code(config)
    scalar = unaligned.to_code(_radix_config(unaligned, cute_radix_packet_bits=256))
    assert "num_bits_per_copy=256" not in scalar
    assert "align(32)" not in scalar
    assert "input_members" in scalar and "selection_length" in scalar
    with patch.dict(
        aligned.env.bound_runtime_input_specialization_results, {key: None}
    ):
        assert "num_bits_per_copy=256" not in aligned.to_code(config)


def test_pipeline_coordinates_reject_untyped_or_unbounded_choices() -> None:
    bound = _cpu_bind(_payload_selection, _args())
    for key, choices in PIPELINE_CHOICES.items():
        for value in (True, None, "256", -1, 512):
            with pytest.raises(exc.InvalidConfig):
                bound.config_spec.normalized_config(
                    _radix_config(bound, **{key: value})
                )
        for value in choices:
            bound.config_spec.normalized_config(_radix_config(bound, **{key: value}))


def _device_kernel(source: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(decorator) == "cute.kernel" for decorator in node.decorator_list
        )
    )


def _long_row_branch(kernel: ast.FunctionDef) -> tuple[int, ast.If]:
    return next(
        (index, node)
        for index, node in enumerate(kernel.body)
        if isinstance(node, ast.If) and "valid_count" in ast.unparse(node.test)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_first_histogram_publishes_counters_and_all_row_branches_keep_final_wait(
    dtype, length_dtype
) -> None:
    x = torch.randn(3, 33, dtype=dtype)
    lengths = torch.tensor([16, 17, 18], dtype=length_dtype)
    bound = _cpu_bind(_payload_selection, (x, lengths, 17))
    config = _radix_config(
        bound,
        **dict.fromkeys(BOOLEAN_KEYS, True),
        cute_radix_packet_bits=256,
        cute_radix_scan_unroll=4,
    )
    kernel = _device_kernel(bound.to_code(config))
    index, branch = _long_row_branch(kernel)
    prefix = ast.Module(body=kernel.body[:index], type_ignores=[])
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(prefix)
        if isinstance(node, ast.Call)
    ]
    assert "cute.arch.sync_threads" in calls
    assert "cute.arch.cluster_arrive" not in calls
    assert "cute.arch.cluster_wait" not in calls
    short = ast.Module(body=branch.body, type_ignores=[])
    assert not any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.map_dsmem_ptr"
        for node in ast.walk(short)
    )
    assert "short_slot" in ast.unparse(short)
    assert ast.unparse(kernel.body[-1]) == "cute.arch.cluster_wait()"
    assert ast.unparse(kernel.body[-2]) == "cute.arch.cluster_arrive()"
    primary = next(
        node
        for node in branch.orelse
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "radix_primary_round"
    )
    assert primary.iter.args[0].value >= 2
    published = ast.unparse(primary)
    assert "cute.arch.cluster_arrive()" in published
    assert "if radix_primary_round == 1:" in published
    # These tails leave whole CTAs empty. Reference execution verifies the
    # logical <K/K/>K rows while every CTA still has the same barrier schedule.
    bound.set_config(config)
    with patch.object(bound.env.settings, "ref_mode", RefMode.EAGER):
        values, ids = bound.run_ref(x, lengths, 17)
    assert torch.equal((ids >= 0).sum(1), torch.tensor([16, 17, 17]))


def test_exact_long_read_bounds_and_full_store_guards_are_elided_only_when_proved() -> (
    None
):
    arguments = (
        torch.empty(3, 33, dtype=torch.bfloat16),
        torch.empty(3, dtype=torch.int64),
        17,
    )
    branches = {}
    for label, modes in (
        ("exact", (False, False, False)),
        ("offset", (True, False, False)),
        ("masked", (False, True, False)),
        ("smaller", (False, False, True)),
    ):
        bound = _cpu_bind(_custom_payload, (*arguments, *modes))
        source = bound.to_code(
            _radix_config(
                bound,
                **dict.fromkeys(BOOLEAN_KEYS, True)
                | {"cute_radix_cached_payload_inverse": False},
            )
        )
        _, branch = _long_row_branch(_device_kernel(source))
        branches[label] = branch

    def payload_guards(statements: list[ast.stmt]) -> list[str]:
        return [
            ast.unparse(node.test)
            for node in ast.walk(ast.Module(body=statements, type_ignores=[]))
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id.startswith("radix_payload")
                    for target in child.targets
                )
                for child in node.body
            )
        ]

    assert any("radix_row" in guard for guard in payload_guards(branches["exact"].body))
    assert all(
        "radix_row" not in guard for guard in payload_guards(branches["exact"].orelse)
    )
    assert any(
        "radix_row" in guard for guard in payload_guards(branches["offset"].orelse)
    )
    assert "_cute_python_mod" in ast.unparse(
        ast.Module(body=branches["masked"].orelse, type_ignores=[])
    )
    assert any(
        isinstance(node, ast.If)
        and ast.unparse(node.test).endswith("< 16")
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "store"
            for child in ast.walk(node)
        )
        for node in ast.walk(
            ast.Module(body=branches["smaller"].orelse, type_ignores=[])
        )
    )
