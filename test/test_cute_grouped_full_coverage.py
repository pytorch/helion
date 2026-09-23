from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import itertools
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_gemm_split_sizes import _device_offsets_kernel
from test.test_cute_shared_rhs_grouped import _bind
from test.test_cute_shared_rhs_grouped import _explicit_offset_mm
from test.test_cute_shared_rhs_grouped import _OffsetPointer
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_shared_rhs_grouped import _target

import helion
from helion._compiler.autotuner_heuristics.cute import (
    CuteTcgen05GroupedSource64Heuristic,
)
from helion._compiler.autotuner_heuristics.cute import _tcgen05_grouped_worklist_config
from helion._compiler.autotuner_heuristics.cute import grouped_full_coverage_configs
from helion._compiler.cute.grouped_full_coverage import full_coverage_index_domain
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Generator

KEY = "tcgen05_grouped_full_coverage"
FLAG = "tcgen05_grouped_full_coverage"


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield
    torch.set_num_threads(previous)


def _config(mode: str | None = "fixed_tma_dense") -> helion.Config:
    result = _tcgen05_grouped_worklist_config(
        32, 64, 2, 240, runtime_direct=False, l2_swizzle_size=1
    )
    if mode is not None:
        result.config[KEY] = mode
    return result


def _inputs(m: int = 640, n: int = 128, k: int = 128) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.tensor([0, 1, m // 2, m - 1, m], dtype=torch.int32),
    )


def _generate(m: int = 640, n: int = 128, k: int = 128) -> tuple[str, str]:
    args = _inputs(m, n, k)
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        return bound.to_code(_config(None)), bound.to_code(_config())


@pytest.fixture(scope="module")
def sources() -> tuple[str, str]:
    return _generate()


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _inverse(candidate: str) -> tuple[ast.Module, dict[str, int]]:
    tree = ast.parse(candidate)
    counts = dict.fromkeys(("metadata", "flag", "init", "changed", "copy", "dense"), 0)

    class Inverse(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
            target = ast.unparse(node.targets[0])
            if target == FLAG:
                counts["flag"] += 1
                return None
            if target == "tcgen05_grouped_d_tensormap_group_changed":
                assert isinstance(node.value, ast.BoolOp)
                assert ast.unparse(node.value.values[0]) == f"not {FLAG}"
                assert len(node.value.values) == 2
                node.value = node.value.values[1]
                counts["changed"] += 1
            return self.generic_visit(node)

        def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
            test = ast.unparse(node.test)
            if test == f"not {FLAG}":
                assert len(node.body) == 1 and not node.orelse
                counts["init"] += 1
                return node.body[0]
            if test == FLAG:
                if len(node.orelse) == 3:
                    assert isinstance(node.orelse[-1], ast.While)
                    counts["dense"] += 1
                    return node.orelse
                assert len(node.body) == len(node.orelse) == 1
                fixed, dynamic = node.body[0], deepcopy(node.orelse[0])
                assert isinstance(dynamic, ast.Expr)
                assert isinstance(dynamic.value, ast.Call)
                assert len(dynamic.value.keywords) == 1
                assert dynamic.value.keywords[0].arg == "tma_desc_ptr"
                dynamic.value.keywords = []
                assert _dump(fixed) == _dump(dynamic)
                counts["copy"] += 1
                return node.orelse
            marker = next(
                (
                    i
                    for i, child in enumerate(node.body)
                    if isinstance(child, ast.Assign)
                    and ast.unparse(child.targets[0]) == "tcgen05_row_union_covered"
                ),
                None,
            )
            if marker is not None:
                assert len(node.body) - marker == 5
                assert isinstance(node.body[-2], ast.While)
                assert isinstance(node.body[-1], ast.If)
                node.body = node.body[:marker]
                counts["metadata"] += 1
            return self.generic_visit(node)

    result = Inverse().visit(tree)
    assert isinstance(result, ast.Module)
    return result, counts


@pytest.mark.parametrize("m,n,k", [(640, 128, 128), (32, 512, 64), (128, 256, 192)])
@skipUnlessBackends(["cute"])
def test_full_source_inverse_and_unchanged_wrapper(m: int, n: int, k: int) -> None:
    ordinary, dense = _generate(m, n, k)
    reversed_tree, counts = _inverse(dense)
    assert counts == {
        "metadata": 1,
        "flag": 1,
        "init": 2,
        "changed": 1,
        "copy": 1,
        "dense": 1,
    }
    assert _dump(reversed_tree) == _dump(ast.parse(ordinary))
    assert _plans(dense) == _plans(ordinary)
    assert "init_tensormap_from_atom(tma_atom_b" not in dense


@skipUnlessBackends(["cute"])
def test_off_is_exact_normalized_config_and_source() -> None:
    args = _inputs()
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        absent, explicit = _config(None), _config("off")
        before = bound.to_code(absent)
        assert bound.to_code(explicit) == before
        bound.config_spec.normalize(absent)
        bound.config_spec.normalize(explicit)
        assert absent.config == explicit.config and KEY not in explicit.config
        assert bound.host_function is not None
        with bound.env:
            source64 = CuteTcgen05GroupedSource64Heuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            )
        # Explicit off describes this requested config. The independent source64
        # heuristic appends one proved dense-local seed to the old seed prefix.
        assert source64 is not None
        seeds = bound.config_spec.compiler_seed_configs
        assert seeds[-1] == source64
        assert source64[KEY] == "fixed_tma_dense_local"
        assert all(KEY not in seed.config for seed in seeds[:-1])


@pytest.mark.parametrize(
    "updates",
    [
        {"tcgen05_ab_stages": 5},
        {"block_sizes": [256, 128, 128]},
        {"tcgen05_acc_stages": 1},
        {"tcgen05_c_stages": 4},
        {"tcgen05_consumer_regs": 256},
        {"tcgen05_sched_stage_count": 2},
        {"tcgen05_cluster_m": 2, "tcgen05_grouped_worklist_source_m_tile": 224},
        {"tcgen05_epilogue_layout": "module_helper_store_tail"},
        {"tcgen05_ab_producer_advance_mode": "skip"},
    ],
)
@pytest.mark.parametrize("mode", ["fixed_tma_dense", "fixed_tma_dense_local"])
@skipUnlessBackends(["cute"])
def test_unsupported_schedule_preserves_ordinary_repair(
    updates: dict[str, object],
    mode: str,
) -> None:
    args = _inputs()
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        selected = _config(mode)
        selected.config.update(deepcopy(updates))
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(deepcopy(selected))
        ordinary = deepcopy(selected.config)
        ordinary.pop(KEY)
        bound.config_spec.normalize(ordinary, _fix_invalid=True)
        bound.config_spec.normalize(selected, _fix_invalid=True)
        repaired_values = deepcopy(selected.config)
        repaired_mode = repaired_values.pop(KEY, None)
        assert repaired_values == ordinary
        if repaired_mode is not None:
            # Existing normalization can restore a supported schedule before
            # final validation. Every retained dense mode must really lower.
            assert repaired_mode == mode
            assert FLAG in bound.to_code(selected)


@pytest.mark.parametrize("bad", ["fp16", "int64", "m_tail", "n_tail", "a_offset"])
@pytest.mark.parametrize("mode", ["fixed_tma_dense", "fixed_tma_dense_local"])
@skipUnlessBackends(["cute"])
def test_binding_domain_declines_new_mode(bad: str, mode: str) -> None:
    a, b, offsets = _inputs()
    if bad == "fp16":
        a, b = a.to(torch.float16), b.to(torch.float16)
    elif bad == "int64":
        offsets = offsets.to(torch.int64)
    elif bad == "m_tail":
        a = a[:-1]
    elif bad == "n_tail":
        b = torch.empty((128, 160), dtype=b.dtype)
    else:
        a = torch.empty(a.numel() + 1, dtype=a.dtype)[1:].view(a.shape)
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, (a, b, offsets))
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_config(mode))


@pytest.mark.parametrize("mode", ["fixed_tma_dense", "fixed_tma_dense_local"])
@skipUnlessBackends(["cute"])
def test_group_specific_rhs_declines_new_mode(mode: str) -> None:
    a, _b, _offsets = _inputs()
    args = (
        a,
        torch.empty((8, 128, 128), dtype=a.dtype),
        torch.tensor([0, 1, 129, 640, 640, 640, 640, 640, 640], dtype=torch.int32),
    )
    with _target():
        bound = _bind(_device_offsets_kernel.fn, args)
        with pytest.raises(InvalidConfig, match="pure shared-RHS"):
            bound.to_code(_config(mode))


@pytest.mark.parametrize("change", ["alias", "group_epilogue", "extra_store"])
@pytest.mark.parametrize("mode", ["fixed_tma_dense", "fixed_tma_dense_local"])
@skipUnlessBackends(["cute"])
def test_unproved_alias_effect_or_group_dependence_declines(
    change: str, mode: str
) -> None:
    source = inspect.getsource(_explicit_offset_mm)
    if change == "alias":
        source = source.replace(
            "torch.zeros((m, n), dtype=a.dtype, device=a.device)", "a"
        )
    elif change == "group_epilogue":
        source = source.replace("acc.to(out.dtype)", "(acc + group).to(out.dtype)")
    else:
        source = source.replace(
            "    return out", "        a[group, group] = 0\n    return out"
        )
    assert source != inspect.getsource(_explicit_offset_mm)
    module = PyCodeCache.load("import torch\nimport helion.language as hl\n" + source)
    args = _inputs()
    with _target():
        bound = _bind(module._explicit_offset_mm, args)
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_config(mode))


def _metadata_code(source: str) -> Any:
    tree = ast.parse(source)
    setup = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Assign)
            and ast.unparse(child.targets[0]) == "tcgen05_grouped_raw_start"
            for child in node.body
        )
    )
    return compile(ast.Module(body=[setup], type_ignores=[]), "<metadata>", "exec")


def _metadata(
    code: Any, routing: list[int], stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    def convert(value: object, dtype: torch.dtype) -> torch.Tensor:
        return (
            value.to(dtype)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, dtype=dtype)
        )

    offsets = torch.zeros(len(routing) * stride, dtype=torch.int32)
    offsets[::stride] = torch.tensor(routing, dtype=torch.int32)
    starts = torch.full((4,), -1, dtype=torch.int32)
    sizes = torch.full((4, 4), -1, dtype=torch.int32)
    namespace = {
        "cutlass": SimpleNamespace(
            Int32=lambda x: convert(x, torch.int32),
            Int64=lambda x: convert(x, torch.int64),
            Boolean=lambda x: convert(x, torch.bool),
        ),
        "cute": SimpleNamespace(arch=SimpleNamespace(thread_idx=lambda: (0, 0, 0))),
        "group_offsets": SimpleNamespace(
            iterator=_OffsetPointer(offsets), layout=SimpleNamespace(stride=(stride,))
        ),
        "tcgen05_grouped_starts": starts,
        "tcgen05_grouped_problem_sizes": sizes,
    }
    exec(code, namespace)
    return starts, sizes


@pytest.mark.parametrize("stride", [1, 2])
@skipUnlessBackends(["cute"])
def test_actual_typed_metadata_and_union_preserve_holes_wrap_and_overlap(
    sources: tuple[str, str], stride: int
) -> None:
    ordinary, dense = map(_metadata_code, sources)
    corpus = [
        [0, 1, 193, 623, 640],
        [0, 1, 193, 623, 623],
        [320, 640, 0, 320, 320],
        [-17, 193, 657, 657, 657],
        [-(1 << 31), (1 << 31) - 1, 0, 640, 640],
        [320] * 5,
        [640, 639, 129, 1, 0],
    ]
    rng = random.Random(73216)
    values = [-(1 << 31), -17, 0, 1, 193, 320, 623, 639, 640, 657, (1 << 31) - 1]
    corpus.extend(rng.choices(values, k=5) for _ in range(40))
    for routing in corpus:
        starts, sizes = _metadata(ordinary, routing, stride)
        new_starts, new_sizes = _metadata(dense, routing, stride)
        covered = set()
        for start, end in itertools.pairwise(routing):
            extent = max((end - start + (1 << 31)) % (1 << 32) - (1 << 31), 0)
            first = min(max(start, 0), 640)
            length = min(max(extent + min(start, 0), 0), 640 - first)
            covered.update(range(first, first + length))
        if len(covered) == 640:
            assert new_starts.tolist() == [0, *starts.tolist()[1:]]
            expected = sizes.clone()
            expected[:, 1] = torch.tensor([640, 0, 0, 0], dtype=torch.int32)
            assert torch.equal(new_sizes, expected)
        else:
            assert torch.equal(new_starts, starts) and torch.equal(new_sizes, sizes)


def _run_dense_producer(
    code: Any, cta: int, grid: int
) -> tuple[list[list[int]], list[str]]:
    mailbox = torch.full((9,), -1, dtype=torch.int32)
    packets: list[list[int]] = []
    events: list[str] = []

    def acquire(state: object) -> None:
        events.append("acquire")

    def commit(state: object) -> None:
        events.append("commit")
        packets.append(mailbox.tolist())

    namespace = {
        "cutlass": SimpleNamespace(Int32=lambda x: torch.tensor(x, dtype=torch.int32)),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: (0, 0, cta),
                grid_dim=lambda: (1, 1, grid),
                lane_idx=lambda: 0,
                sync_warp=lambda: events.append("sync"),
            )
        ),
        "tcgen05_work_tile_smem": mailbox,
        "tcgen05_sched_pipeline": SimpleNamespace(
            producer_acquire=acquire, producer_commit=commit
        ),
        "tcgen05_sched_pipeline_producer_state": SimpleNamespace(
            advance=lambda: events.append("advance")
        ),
    }
    exec(code, namespace)
    return packets, events


@pytest.mark.parametrize("m,n", [(640, 128), (32, 512), (128, 256)])
@skipUnlessBackends(["cute"])
def test_actual_dense_producer_covers_tiles_once_and_preserves_terminal(
    m: int, n: int
) -> None:
    _ordinary, dense = _generate(m, n)
    tree = ast.parse(dense)
    scheduler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and node.body
        and isinstance(node.body[0], ast.If)
        and ast.unparse(node.body[0].test) == FLAG
        and len(node.body[0].body) == 3
    )
    branch = scheduler.body[0]
    assert isinstance(branch, ast.If)
    code = compile(
        ast.Module(body=[*branch.body, *scheduler.body[1:]], type_ignores=[]),
        "<dense-producer>",
        "exec",
    )
    rows, columns = m // 32, n // 128
    total = rows * columns
    for grid in (1, 3, 5, total + 1):
        tiles = []
        for cta in range(grid):
            packets, events = _run_dense_producer(code, cta, grid)
            assert packets and packets[-1][2] == 0
            assert events == ["acquire", "commit", "advance", "sync"] * len(packets)
            assert all(packet[2] == 1 for packet in packets[:-1])
            expected_indices = list(range(cta, total, grid))
            expected_tiles = [
                (index % rows, index // rows)
                if rows <= columns
                else (index // columns, index % columns)
                for index in expected_indices
            ]
            assert [(packet[0], packet[1]) for packet in packets[:-1]] == expected_tiles
            for packet in packets[:-1]:
                assert packet[3:7] == [0, -1, m, n]
                assert packet[8] == 0
                tiles.append((packet[0], packet[1]))
        assert sorted(tiles) == list(itertools.product(range(rows), range(columns)))


@pytest.mark.parametrize("mode", ["fixed_tma_dense", "fixed_tma_dense_local"])
@skipUnlessBackends(["cute"])
def test_named_config_roundtrip(mode: str) -> None:
    args = _inputs()
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        config = _config(mode)
        dense = bound.to_code(config)
        ordinary = bound.to_code(_config(None))
        restored = helion.Config.from_json(config.to_json())
        assert restored.config == config.config
        assert bound.to_code(restored) == dense
        assert dense != ordinary and restored.to_json() != _config(None).to_json()


@pytest.mark.parametrize(
    "args,expected",
    [
        ((4, 640, 128, 128, 32, 128, 64), True),
        ((8, 2560, 256, 512, 32, 128, 64), True),
        ((4, 639, 128, 128, 32, 128, 64), False),
        ((4, 640, 160, 128, 32, 128, 64), False),
        ((4, 640, 128, 127, 32, 128, 64), False),
        ((0, 640, 128, 128, 32, 128, 64), False),
        ((True, 640, 128, 128, 32, 128, 64), False),
        (((1 << 31) - 1, 1, 1, 1, 1, 1, 1), True),
        (((1 << 30) - 1, 2, 1, 1, 1, 1, 1), True),
        ((1 << 30, 2, 1, 1, 1, 1, 1), False),
        ((1, (1 << 30) + 1, 1, 1, 1, 1, 1), False),
        ((1, 1 << 25, 128, 64, 32, 128, 64), False),
    ],
)
def test_complete_index_domain(args: tuple[int, ...], expected: bool) -> None:
    assert full_coverage_index_domain(*args) is expected


@skipUnlessBackends(["cute"])
def test_coverage_records_are_independent_and_legacy_seeds_are_unchanged() -> None:
    args = _inputs()
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        assert bound.host_function is not None
        old = deepcopy([x.config for x in bound.config_spec.compiler_seed_configs])
        records = grouped_full_coverage_configs(
            bound.env, bound.host_function.device_ir
        )
        assert len(records) == 2 and KEY not in records[0].config
        assert records[1].config == records[0].config | {KEY: "fixed_tma_dense"}
        records[0].block_sizes[0] = 1
        assert records[1].block_sizes[0] == 256
        assert [x.config for x in bound.config_spec.compiler_seed_configs] == old
        disabled = helion.kernel(
            grouped_gemm_jagged.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
            cute_segmented_matmul_tiling=True,
            disable_autotuner_heuristics=True,
        )._bind_isolated(args)
        assert disabled.host_function is not None
        assert (
            len(
                grouped_full_coverage_configs(
                    disabled.env, disabled.host_function.device_ir
                )
            )
            == 2
        )
        # Keep explicit full-domain configs serializable. The shared coverage
        # facility, not the consumer, suppresses automatic proposals/append.
        assert KEY in disabled.config_spec._cute_tcgen05_config.optional_fragments(
            for_search=True
        )
        assert FLAG in disabled.to_code(_config())


@pytest.mark.parametrize("m,n,k", [(639, 128, 128), (640, 160, 128), (640, 128, 127)])
@skipUnlessBackends(["cute"])
def test_missing_full_tile_carrier_keeps_legacy_search_domain(
    m: int, n: int, k: int
) -> None:
    args = _inputs(m, n, k)
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        assert bound.host_function is not None
        assert not bound.config_spec._cute_tcgen05_config.grouped_full_coverage_eligible
        assert KEY not in bound.config_spec._cute_tcgen05_config.optional_fragments(
            for_search=True
        )
        assert (
            grouped_full_coverage_configs(bound.env, bound.host_function.device_ir)
            == []
        )


@pytest.mark.parametrize("k_min,k_max,has_carrier", [(16, 32, False), (128, 256, True)])
@skipUnlessBackends(["cute"])
def test_restricted_k_domain_exposes_only_a_reachable_carrier(
    k_min: int, k_max: int, has_carrier: bool
) -> None:
    tree = ast.parse(inspect.getsource(grouped_gemm_jagged.fn))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    function.decorator_list = []
    reductions = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "hl.tile(K)"
    ]
    assert len(reductions) == 1
    call = reductions[0].iter
    assert isinstance(call, ast.Call)
    call.keywords.append(
        ast.keyword(
            arg="block_size",
            value=ast.parse(
                f"hl.register_block_size({k_min}, {k_max})", mode="eval"
            ).body,
        )
    )
    restricted = PyCodeCache.load(
        "import torch\nimport helion.language as hl\n" + ast.unparse(tree)
    ).grouped_gemm_jagged
    args = _inputs(k=256)
    with _target():
        bound = _bind(restricted, args)
        assert bound.host_function is not None
        domain = bound.config_spec.block_sizes[2]._fragment(bound.config_spec)
        assert domain.low == k_min
        assert (
            bound.config_spec._cute_tcgen05_config.grouped_full_coverage_eligible
            is has_carrier
        )
        assert (
            KEY
            in bound.config_spec._cute_tcgen05_config.optional_fragments(
                for_search=True
            )
        ) is has_carrier
        assert (
            grouped_full_coverage_configs(bound.env, bound.host_function.device_ir)
            == []
        )
        records = grouped_full_coverage_configs(
            bound.env, bound.host_function.device_ir, block_k=128
        )
        assert bool(records) is has_carrier
        if has_carrier:
            assert records[0].block_sizes[2] == 128
            assert FLAG in bound.to_code(records[1])
            (group,) = (
                candidate
                for candidate in bound.config_spec.compiler_coverage_groups
                if candidate.key == KEY
            )
            assert [witness.value for witness in group.witnesses] == [
                "off",
                "fixed_tma_dense",
                "fixed_tma_dense_local",
            ]
            assert all(witness.carrier == records[0] for witness in group.witnesses)
            local = group.witnesses[-1].carrier
            local.config[KEY] = "fixed_tma_dense_local"
            assert FLAG in bound.to_code(local)


@pytest.mark.parametrize("m,n", [(32, 512), (128, 256)])
@pytest.mark.parametrize("mode", ["fixed_tma_dense", "fixed_tma_dense_local"])
@skipUnlessBackends(["cute"])
def test_supported_explicit_config_does_not_require_an_automatic_carrier(
    m: int, n: int, mode: str
) -> None:
    args = _inputs(m=m, n=n)
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        assert bound.host_function is not None
        state = bound.config_spec._cute_tcgen05_config
        assert state.grouped_full_coverage_supported
        assert not state.grouped_full_coverage_eligible
        assert KEY not in state.optional_fragments(for_search=True)
        assert (
            grouped_full_coverage_configs(bound.env, bound.host_function.device_ir)
            == []
        )
        assert FLAG in bound.to_code(_config(mode))
