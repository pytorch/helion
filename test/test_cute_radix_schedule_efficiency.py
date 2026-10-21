"""Full-source output ownership, partial scan and short packet contracts."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_radix_selection import _args
from test.test_cute_radix_selection import _device_kernel
from test.test_cute_radix_selection import _payload_selection
from test.test_cute_radix_selection import _radix_config

from helion import exc
from helion._compiler.cute.radix_selection import radix_schedule
from helion._compiler.cute.radix_selection_codegen import _compact
from helion._compiler.cute.radix_selection_codegen import _compact_packet_work
from helion._compiler.cute.radix_selection_codegen import _histogram_choose
from helion._compiler.cute.radix_selection_codegen import _RadixCodegenSpec
from helion._compiler.cute.radix_selection_codegen import _Writer

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


def _primary(source: str) -> ast.For:
    return next(
        node
        for node in ast.walk(_device_kernel(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "radix_primary_round"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capacity", [0, 1024])
@pytest.mark.parametrize("resident", [False, True])
def test_deferred_output_has_one_owner_and_ignores_incomplete_candidate_band(
    dtype: torch.dtype, capacity: int, resident: bool
) -> None:
    bound = _cpu_bind(_payload_selection, _args(dtype))
    config = _radix_config(
        bound,
        cute_radix_threads=256,
        cute_radix_candidate_capacity=capacity,
        cute_radix_candidate_key_cache=capacity > 0,
        cute_radix_resident_input=resident,
        cute_radix_valid_domain=True,
        cute_radix_ticket_stores=True,
        cute_radix_boundary_early_exit=True,
        cute_radix_defer_output=True,
    )
    source = bound.to_code(config)
    primary = ast.unparse(_primary(source))
    assert "radix_above_ticket" not in primary
    assert "radix_coarse_above" not in primary
    assert "radix_output_source_column" not in primary
    assert "radix_coarse_prefix" not in source
    assert "radix_coarse_mask" not in source
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None
    plan = radix_schedule(graph, config.config)
    assert not {"coarse_prefix", "coarse_mask"}.intersection(
        name for name, _dtype, _count in plan.allocations
    )
    compact = _Writer()
    _compact(compact, _RadixCodegenSpec(plan, 32))
    text = "\n".join(compact.lines)
    assert "use_candidates = cutlass.Boolean(False)" in text
    assert "use_candidates = cutlass.Boolean(True)" not in text
    assert "candidate_counter[0]" not in text
    assert "coarse_member = cutlass.Boolean(True)" in text
    assert "input_members[vec]" in text
    assert "if equal_ticket < boundary_remaining:" in text
    # Histogram caching still runs in its original private band. Only the
    # final output owner must scan the full source, even after cache overflow.
    if capacity:
        assert "radix_candidate_counter[0]" in primary
        assert "radix_candidate_indices" in primary


@pytest.mark.parametrize(
    ("dtype", "threads", "cluster"),
    [
        (torch.float32, 256, 1),
        (torch.bfloat16, 512, 1),
        (torch.float32, 1024, 4),
        (torch.bfloat16, 1024, 4),
    ],
)
def test_named_scan_barrier_has_exact_warp_ownership_and_full_publication(
    dtype: torch.dtype, threads: int, cluster: int
) -> None:
    bound = _cpu_bind(_payload_selection, _args(dtype))
    config = _radix_config(
        bound,
        cute_radix_threads=threads,
        cute_radix_cluster_ctas=cluster,
        cute_radix_candidate_capacity=0,
        cute_radix_partial_scan=True,
        cute_radix_boundary_early_exit=True,
    )
    primary = _primary(bound.to_code(config))
    statements = primary.body[0].body
    index, scan = next(
        (index, node)
        for index, node in enumerate(statements)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "radix_thread < 256"
        and "cute.arch.barrier(" in ast.unparse(node)
    )
    barrier = next(
        node
        for node in ast.walk(scan)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.arch.barrier"
    )
    assert {kw.arg: kw.value.value for kw in barrier.keywords} == {
        "barrier_id": 1,
        "number_of_threads": 256,
    }
    assert "cute.arch.sync_threads()" not in ast.unparse(scan)
    assert "radix_warp_totals[radix_warp]" in ast.unparse(scan)
    assert "radix_primary_prefix[0]" in ast.unparse(scan)
    assert ast.unparse(statements[index + 1]) == "cute.arch.sync_threads()"
    if cluster > 1:
        assert (
            ast.unparse(statements[index + 2]) == "cute.arch.cluster_arrive_relaxed()"
        )
        assert ast.unparse(statements[index + 3]) == "cute.arch.cluster_wait()"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_short_packets_have_guarded_alignment_and_scalar_fallback(
    dtype: torch.dtype,
) -> None:
    bound = _cpu_bind(_payload_selection, _args(dtype, cols=32))
    source = bound.to_code(_radix_config(bound, cute_radix_packet_bits=64))
    assert "num_bits_per_copy=64" in source
    assert "align(8)" in source
    packet = 8 // dtype.itemsize
    assert f"% {packet} == 0" in source
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None
    plan = radix_schedule(graph, _radix_config(bound, cute_radix_packet_bits=64).config)
    assert plan.packet == packet
    arena = torch.empty(3 * 32 + 1, dtype=dtype)
    shifted = arena[1:].reshape(3, 32)
    assert shifted.data_ptr() % 8 != 0
    lengths = torch.tensor([0, 5, 32])
    unaligned = _cpu_bind(_payload_selection, (shifted, lengths, 17))
    fallback = unaligned.to_code(_radix_config(unaligned, cute_radix_packet_bits=64))
    assert "num_bits_per_copy=64" not in fallback
    assert "align(8)" not in fallback


@pytest.mark.parametrize(
    "field", ["cute_radix_defer_output", "cute_radix_partial_scan"]
)
def test_efficiency_coordinates_remain_typed_and_legacy_opt_in(field: str) -> None:
    bound = _cpu_bind(_payload_selection, _args())
    for value in (0, 1, None, "true"):
        with pytest.raises(exc.InvalidConfig, match="Boolean"):
            bound.config_spec.normalized_config(_radix_config(bound, **{field: value}))
    legacy = bound.config_spec.default_config()
    legacy.config[field] = True
    with pytest.raises(exc.InvalidConfig, match="requires the radix"):
        bound.config_spec.normalized_config(legacy)
    assert (
        bound.config_spec.default_config().config.get("cute_radix_packet_bits", 128)
        == 128
    )


def _compile_statements(statements: list[ast.stmt]):
    return compile(
        ast.Module(body=statements, type_ignores=[]), "<radix-source>", "exec"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scenario", ["first", "second", "ties", "zero_prefix"])
def test_emitted_deferred_boundary_predicates_keep_original_topk(
    dtype: torch.dtype, scenario: str
) -> None:
    bound = _cpu_bind(_payload_selection, _args(dtype))
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None
    config = _radix_config(
        bound,
        cute_radix_threads=256,
        cute_radix_cluster_ctas=4,
        cute_radix_candidate_capacity=0,
        cute_radix_valid_domain=True,
        cute_radix_ticket_stores=True,
        cute_radix_resident_input=True,
        cute_radix_boundary_early_exit=True,
        cute_radix_partial_scan=True,
        cute_radix_defer_output=True,
    )
    spec = _RadixCodegenSpec(radix_schedule(graph, config.config), 32)
    threshold = _Writer()
    _histogram_choose(
        threshold,
        spec,
        prefix="primary_prefix",
        mask="primary_mask",
        word="cutlass.Uint32",
        shift="primary_shift",
    )
    chosen = next(
        node
        for node in ast.walk(ast.parse("\n".join(threshold.lines)))
        if isinstance(node, ast.If)
        and all(
            term in ast.unparse(node.test)
            for term in ("bin_count > 0", "greater_count", "pass_remaining")
        )
    )
    choose_code = _compile_statements([chosen])
    compact = _Writer()
    _compact_packet_work(compact, spec, False)
    statements = ast.parse("\n".join(compact.lines)).body
    end = next(
        (
            index
            for index, node in enumerate(statements)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and (node.targets[0].id == "above_members")
        )
    )
    compact_code = _compile_statements(statements[:end])
    width = spec.primary_width
    shift = width - 8
    if scenario == "first":
        keys = [193 << shift | i for i in (1, 2)]
        keys += [191 << shift | i for i in (1, 2, 3)]
        keys += [189 << shift | i for i in (1, 2, 3)]
        k = 5
    elif scenario == "second":
        keys = [193 << shift | 1, 191 << shift | 160]
        keys += [191 << shift | 144] * 3
        keys += [191 << shift | 128] * 3
        if width == 32:
            keys = [
                key & 4278190080 | (key & 255) << 16 | i for i, key in enumerate(keys)
            ]
        k = 5
    elif scenario == "ties":
        keys = [191 << shift | 17] * 6
        keys += [191 << shift | 16] * 3
        keys += [193 << shift | 1]
        k = 4
    else:
        keys = [1, 2, 3, 4, 5, 6, 7, 8, 1 << shift | 1]
        k = len(keys)
    members = [True] * len(keys) + [False] * 3
    keys += [(1 << width) - 1] * 3
    state = {
        "cutlass": SimpleNamespace(Int32=int, Uint32=int, Boolean=bool),
        "primary_prefix": [0],
        "primary_mask": [0],
        "remaining": [k],
        "primary_done": [0],
        "coarse_prefix": [0],
        "coarse_mask": [0],
    }
    above = []
    rounds = 0
    for primary_round in range(width // 8):
        if state["primary_done"][0]:
            break
        rounds += 1
        prefix = state["primary_prefix"][0]
        mask = state["primary_mask"][0]
        remaining = state["remaining"][0]
        if primary_round == 1 and not spec.plan.defer_output:
            state["coarse_prefix"][0] = prefix
            state["coarse_mask"][0] = mask
            above += [
                index
                for index, (key, member) in enumerate(zip(keys, members, strict=True))
                if member and key & mask > prefix
            ]
        primary_shift = (width // 8 - 1 - primary_round) * 8
        histogram = [0] * 256
        for key, member in zip(keys, members, strict=True):
            if member and key & mask == prefix:
                histogram[key >> primary_shift & 255] += 1
        for digit, count in enumerate(histogram):
            state.update(
                thread=digit,
                bin_count=count,
                greater_count=sum(histogram[digit + 1 :]),
                pass_prefix=prefix,
                pass_mask=mask,
                pass_remaining=remaining,
                primary_shift=primary_shift,
            )
            exec(choose_code, state)
    equal = []
    for index, (key, member) in enumerate(zip(keys, members, strict=True)):
        state.update(
            input_keys=[key],
            input_members=[member],
            vec=0,
            primary_boundary=state["primary_prefix"][0],
            boundary_complete=bool(state["primary_done"][0]),
        )
        exec(compact_code, state)
        if state["take_above"]:
            above.append(index)
        if state["take_equal"]:
            equal.append(index)
    selected = above + equal[: state["remaining"][0]]
    expected = sorted(
        (key for key, member in zip(keys, members, strict=True) if member), reverse=True
    )[:k]
    assert len(set(selected)) == k
    assert sorted((keys[index] for index in selected), reverse=True) == expected
    if scenario in ("first", "zero_prefix"):
        assert rounds == 1 and state["primary_done"][0] == 1
    if scenario == "second":
        assert rounds == 2 and state["primary_done"][0] == 1
    if scenario == "ties":
        assert rounds == width // 8 and state["primary_done"][0] == 0
