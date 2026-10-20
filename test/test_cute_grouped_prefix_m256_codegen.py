from __future__ import annotations

import ast
import builtins
from copy import deepcopy
import importlib.util
from itertools import accumulate
from itertools import pairwise
from textwrap import dedent
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_grouped_d_shift import _config as _ordinary_config
from test.test_cute_grouped_large_device_seeds import _kernel
from test.test_cute_grouped_paired_clc_layout import _sdk
from test.test_cute_grouped_prefix_adaptive_config import _args
from test.test_cute_grouped_prefix_adaptive_config import _recipes
from test.test_cute_grouped_prefix_adaptive_n_codegen import _kernel_ast
from test.test_cute_grouped_prefix_adaptive_n_codegen import _plans

from helion._compiler.cute.grouped_prefix_adaptive_n_codegen import _is_role
from helion._compiler.cute.grouped_prefix_adaptive_n_codegen import _role_payload
from helion.runtime.cute.launcher import _append_cute_wrapper_plan

if TYPE_CHECKING:
    from pathlib import Path

    import helion
    from helion.autotuner.config_spec import ConfigSpec

KEY = "tcgen05_grouped_prefix_adaptive_n"
SOURCE = "tcgen05_grouped_worklist_source_m_tile"
BALANCED = "tcgen05_grouped_balanced_workers"


def _core(tree: ast.Module, width: int) -> ast.FunctionDef:
    return next(
        value
        for value in tree.body
        if isinstance(value, ast.FunctionDef)
        and value.name.endswith(f"_n{width}")
        and "_decode_" not in value.name
    )


def _recipe(spec: ConfigSpec, *, balanced: bool = False) -> helion.Config:
    if recipes := _recipes(spec, 256):
        return deepcopy(
            next(value for value in recipes if value.get(BALANCED, False) is balanced)
        )
    profile = spec._cute_tcgen05_config.get_grouped_prefix_adaptive_n_profile(
        {SOURCE: 256}
    )
    assert profile is not None
    config = _ordinary_config("mn")
    config.config.update(
        tcgen05_grouped_worklist_source_m_tile=256,
        tcgen05_grouped_worklist_mma_n=profile.primary.n,
        tcgen05_ab_stages=profile.primary.ab_stages,
        tcgen05_c_stages=profile.primary.c_stages,
        tcgen05_consumer_regs=256,
        cute_min_blocks_per_mp=1,
        tcgen05_l2_swizzle_size=8,
        tcgen05_grouped_l2_swizzle_alternate=4,
        tcgen05_grouped_l2_swizzle_mn_ratio=1,
        tcgen05_grouped_balanced_workers=balanced,
        tcgen05_grouped_prefix_adaptive_n=True,
    )
    return config


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    result: dict[str, str] = {}
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args(m=8192, k=256))
        spec = bound.config_spec
        for balanced in (False, True):
            config = spec.normalized_config(_recipe(spec, balanced=balanced))
            result[str(balanced)] = bound.to_code(config)
        profile = spec._cute_tcgen05_config.get_grouped_prefix_adaptive_n_profile(
            config.config
        )
        assert profile is not None
        for branch in (profile.primary, profile.alternate):
            raw = _recipe(spec)
            raw.config.pop(KEY)
            raw.config.update(
                tcgen05_grouped_balanced_workers=True,
                tcgen05_grouped_worklist_mma_n=branch.n,
                tcgen05_ab_stages=branch.ab_stages,
                tcgen05_c_stages=branch.c_stages,
            )
            if branch is profile.alternate:
                raw.config.update(
                    tcgen05_layout_strategy="explicit_epi_tile",
                    tcgen05_layout_overrides_epi_tile_m=128,
                    tcgen05_layout_overrides_epi_tile_n=32,
                    tcgen05_layout_overrides_d_store_box_n=32,
                )
            result[str(branch.n)] = bound.to_code(spec.normalized_config(raw))
        assert spec.block_sizes[2].declared_max_size == 64
    return result


def test_native_six_payloads_phases_and_common_header_are_preserved(
    sources: dict[str, str],
) -> None:
    source = sources["False"]
    tree = ast.parse(source)
    entry = _kernel_ast(tree)
    plans = _plans(source)
    profile = plans[0]["prefix_adaptive_n"]
    assert len(entry.args.args) == 15
    assert profile["widths"] == [224, 256]
    assert profile["ab_stages"] == [7, 6]
    assert profile["c_stages"] == [2, 3]
    assert profile["arena_bytes"] == 231424
    assert profile["compiler_smem_bytes"] == 232448
    assert profile["common_header_bytes"] == 560
    assert profile["output_tile"] == [128, 32]
    assert profile["tmem_columns"] == 512
    assert plans[1]["full_allocation_b_rank3"] is True
    assert plans[3]["bn"] == 256 and plans[3]["ab_stage_count"] == 6
    assert source.count("cute.arch.alloc_smem(") == 9
    assert "StaticPersistentGroupTileScheduler" not in source
    assert "tcgen05_sched_pipeline" not in source
    assert "tcgen05_work_tile_smem" not in source
    assert "warp_reduction_sum" not in source
    assert not any(isinstance(value, ast.Return) for value in ast.walk(entry))
    for width in (224, 256):
        native = _kernel_ast(ast.parse(sources[str(width)]))
        old_roles = [
            value
            for value in ast.walk(native)
            if isinstance(value, ast.If) and _is_role(value)
        ]
        core = _core(tree, width)
        roles = [
            value
            for value in core.body
            if isinstance(value, ast.If)
            and any(isinstance(child, ast.While) for child in value.body)
        ]
        assert len(old_roles) == len(roles) == 3
        assert not any(
            isinstance(value, ast.Call)
            and ast.unparse(value.func) == "cute.arch.alloc_smem"
            for value in ast.walk(core)
        )
        for old, new in zip(old_roles, roles, strict=True):
            old_loop = next(value for value in old.body if isinstance(value, ast.While))
            new_loop = next(value for value in new.body if isinstance(value, ast.While))
            payload = _role_payload(old_loop)
            assert ast.dump(ast.Module(payload, [])) == ast.dump(
                ast.Module(new_loop.body[-len(payload) - 1 : -1], [])
            )
        for call in (
            "tcgen05_ab_pipeline.producer_acquire",
            "tcgen05_ab_pipeline.consumer_release",
            "tcgen05_acc_pipeline.consumer_release",
            "tcgen05_c_pipeline.producer_acquire",
            "tcgen05_c_pipeline.producer_commit",
            "tcgen05_c_pipeline.producer_tail",
            "tcgen05_acc_pipeline.producer_tail",
            "tcgen05_tmem_allocator.free",
            "cute.copy",
            "cute.gemm",
        ):
            counts = [
                sum(
                    isinstance(value, ast.Call) and ast.unparse(value.func) == call
                    for value in ast.walk(root)
                )
                for root in (native, core)
            ]
            assert counts[0] == counts[1], call


@pytest.mark.parametrize(
    "groups,m,n,k,dtype,offset_dtype",
    [
        (4, 640, 800, 256, torch.float16, torch.int64),
        (32, 1024, 768, 256, torch.bfloat16, torch.int32),
    ],
)
def test_general_tail_dtypes_and_group_counts_use_proved_native_descriptors(
    groups: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(
            _kernel(),
            _args(groups=groups, m=m, n=n, k=k, dtype=dtype, offset_dtype=offset_dtype),
        )
        config = bound.config_spec.normalized_config(_recipe(bound.config_spec))
        source = bound.to_code(config)
        assert config.block_sizes == [256, 128, 64]
        assert bound.config_spec.block_sizes[2].declared_max_size == 64
    plans = _plans(source)
    assert plans[0]["group_count"] == groups and plans[0]["n_size"] == n
    assert plans[0]["prefix_adaptive_n"]["common_header_bytes"] <= 1024
    for plan in (plans[1], plans[3]):
        assert plan["full_allocation_a"] and plan["full_allocation_b"]
        assert plan["bk"] == 64
        if n % plan["bn"]:
            assert plan["full_allocation_b_rank3"] is True
    if offset_dtype is torch.int64:
        assert "tcgen05_grouped_raw_start = cutlass.Int64(" in source
        assert "cutlass.Int32(tcgen05_grouped_visible_extent)" in source


@pytest.mark.parametrize("width,features", [(224, 7), (256, 6)])
@pytest.mark.parametrize("counts", [(0, 2, 0, 1), (1,) * 16, (0,) * 31 + (9,)])
def test_emitted_panel_decoder_matches_native_order_and_empty_experts(
    sources: dict[str, str], width: int, features: int, counts: tuple[int, ...]
) -> None:
    tree = ast.parse(sources["False"])
    decoder = deepcopy(
        next(
            value
            for value in tree.body
            if isinstance(value, ast.FunctionDef)
            and value.name.endswith(f"_decode_n{width}")
        )
    )
    decoder.decorator_list = []
    groups = len(counts)
    prefixes = list(accumulate(counts))
    matrix = {(group, 0): count * 256 for group, count in enumerate(counts)}
    matrix.update({(group, 3): prefix for group, prefix in enumerate(prefixes)})
    state = {"lane": 0, "linear": 0}

    def ballot(predicate: bool) -> int:
        matches = [
            lane < groups and state["linear"] // features < prefixes[lane]
            for lane in range(32)
        ]
        assert predicate == matches[state["lane"]]
        return sum(int(value) << lane for lane, value in enumerate(matches))

    scope: dict[str, Any] = {
        "cutlass": SimpleNamespace(Int32=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                lane_idx=lambda: state["lane"],
                vote_ballot_sync=ballot,
                popc=lambda value: (value & ((1 << 32) - 1)).bit_count(),
                make_warp_uniform=lambda value: value,
            )
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([decoder], [])),
            "<native panel prefix>",
            "exec",
        ),
        scope,
    )
    expected = []
    for group, rows in enumerate(counts):
        panel = 8 if rows <= features else 4
        for panel_index, begin in enumerate(range(0, features, panel)):
            row_order = range(rows) if panel_index % 2 == 0 else reversed(range(rows))
            for row in row_order:
                for column in range(begin, min(begin + panel, features)):
                    expected.append((row, column, group, rows * 256))
    for linear, answer in enumerate(expected):
        state["linear"] = linear
        for lane in range(32):
            state["lane"] = lane
            assert scope[decoder.name](linear, matrix, features, groups) == answer


@pytest.mark.parametrize("balanced", [False, True])
def test_actual_global_choice_stride_and_inactive_cluster_bounds(
    sources: dict[str, str], balanced: bool
) -> None:
    tree = ast.parse(sources[str(balanced)])
    entry = deepcopy(_kernel_ast(tree))
    start = next(
        index
        for index, value in enumerate(entry.body)
        if isinstance(value, ast.Assign)
        and isinstance(value.targets[0], ast.Name)
        and value.targets[0].id.endswith("_rows")
    )
    entry.body = entry.body[start:]
    entry.decorator_list = []
    entry.args = ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    state = {"cap": 76, "z": 0}
    selected: list[tuple[int, tuple[Any, ...]]] = []
    scope: dict[str, Any] = {
        **vars(builtins),
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                grid_dim=lambda: (2, 1, state["cap"]),
                block_idx=lambda: (0, 0, state["z"]),
            )
        ),
    }
    for value in ast.walk(entry):
        if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load):
            scope.setdefault(value.id, object())
    for width in (224, 256):
        core = _core(tree, width)
        scope[core.name] = lambda *args, width=width: selected.append((width, args))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([entry], [])),
            "<actual M256 selector>",
            "exec",
        ),
        scope,
    )
    for cap in (1, 38, 71, 76):
        state["cap"] = cap
        for rows in (0, 1, 32, 44, 512, 4_194_304):
            scope["tcgen05_grouped_problem_sizes"] = {(15, 3): rows}
            totals = {224: rows * 7, 256: rows * 6}
            waves = {
                width: max(1, (count + cap - 1) // cap)
                for width, count in totals.items()
            }
            chosen = 224 if waves[224] * 224 <= waves[256] * 256 else 256
            count = totals[chosen]
            workers = max(
                1,
                min(
                    cap,
                    (count + waves[chosen] - 1) // waves[chosen] if balanced else count,
                ),
            )
            for z in range(cap):
                state["z"] = z
                selected.clear()
                scope[entry.name]()
                if rows == 0 or z >= workers:
                    assert not selected
                else:
                    assert len(selected) == 1 and selected[0][0] == chosen
                    # Native arguments, shared metadata, selected total/stride,
                    # payload and four common pipeline-state pointers.
                    assert selected[0][1][13:15] == (count, workers)


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_actual_complete_entry_and_both_cores_have_typed_verified_ir(
    offset_dtype: torch.dtype, tmp_path: Path
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args(m=8192, k=256, offset_dtype=offset_dtype))
        source = bound.to_code(
            bound.config_spec.normalized_config(_recipe(bound.config_spec))
        )
    tree = ast.parse(source)
    entry = _kernel_ast(tree)
    entry.decorator_list = [ast.parse("cute.jit", mode="eval").body]
    path = tmp_path / "complete_m256_entry.py"
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    with _forbid_cuda_work(), _sdk() as sdk:
        spec = importlib.util.spec_from_file_location("complete_m256_entry", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = sdk.func.FuncOp("complete_m256_prefix_entry", ([], []))
        with sdk.ir.InsertionPoint(function.add_entry_block()):
            scope: dict[str, Any] = {"cute": sdk.cute, "cutlass": sdk.cutlass}
            arguments: list[Any] = []
            for index, dtype, shape, strides in (
                (
                    0,
                    sdk.cutlass.Int32
                    if offset_dtype is torch.int32
                    else sdk.cutlass.Int64,
                    (17,),
                    (1,),
                ),
                (1, sdk.cutlass.BFloat16, (8192, 256), (256, 1)),
                (2, sdk.cutlass.BFloat16, (16, 1536, 256), (1536 * 256, 256, 1)),
                (3, sdk.cutlass.BFloat16, (8192, 1536), (1536, 1)),
            ):
                tensor = sdk.cute.make_tensor(
                    sdk.cute.make_ptr(
                        dtype,
                        (index + 1) * (1 << 36),
                        sdk.cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    sdk.cute.make_layout(shape, stride=strides),
                )
                scope[f"arg{index}"] = tensor
                arguments.append(tensor)
                for dim, (extent, step) in enumerate(zip(shape, strides, strict=True)):
                    scope[f"arg{index}_shape{dim}"] = extent
                    scope[f"arg{index}_stride{dim}"] = step
            body: list[str] = []
            extra: list[str] = []
            plans = _plans(source)
            grouped = plans[0]
            scope[grouped["total_clusters_arg"]] = (
                grouped["group_count"]
                * ((grouped["m_size"] + grouped["bm"] - 1) // grouped["bm"])
                * ((grouped["n_size"] + grouped["bn"] - 1) // grouped["bn"])
            )
            for plan in plans:
                _append_cute_wrapper_plan(body, extra, plan, num_sm=152)
            exec(
                compile(
                    ast.parse(dedent("\n".join(body))),
                    "<actual native M256 wrappers>",
                    "exec",
                ),
                scope,
            )
            arguments.extend(scope[name] for name in extra)
            assert len(arguments) == len(entry.args.args) == 15
            getattr(module, entry.name)(*arguments)
            sdk.func.ReturnOp([])
        assert sdk.module.operation.verify()
        text = str(sdk.module)
        assert "ballot" in text and "ctpop" in text
        assert "cute.gemm" in text
        if offset_dtype is torch.int64:
            assert "arith.trunci" in text


@pytest.mark.cuda
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_m256_live_graph_branch_changes_and_input_preservation(
    offset_dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family tcgen05")
    groups, rows, columns, reduction = 8, 4096, 768, 256
    generator = torch.Generator().manual_seed(419)
    left = torch.randint(0, 5, (rows, reduction), generator=generator).float() / 8
    right = (
        torch.randint(-4, 5, (groups, columns, reduction), generator=generator).float()
        / 8
    )
    right += torch.arange(groups)[:, None, None] / 2
    left, right = left.bfloat16(), right.bfloat16()
    a, weights = left.cuda(), right.cuda()
    uniform = list(range(0, rows + 1, rows // groups))
    skewed = list(accumulate([0, *([1] * 6), 0, rows - 6]))
    offsets = torch.tensor(uniform, dtype=offset_dtype, device="cuda")
    args = (a, weights, offsets)
    bound = _kernel()._bind_isolated(args)
    bound.set_config(_recipe(bound.config_spec))

    def expected(route: list[int]) -> torch.Tensor:
        return torch.cat(
            [
                (left[begin:end].double() @ right[group].double().T).bfloat16()
                for group, (begin, end) in enumerate(pairwise(route))
            ]
        )

    initial = bound(*args)
    torch.testing.assert_close(initial.cpu(), expected(uniform), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    for route in (skewed, uniform, [0] * groups + [rows]):
        offsets.copy_(torch.tensor(route, dtype=offset_dtype, device="cuda"))
        before = [value.clone() for value in args]
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured.cpu(), expected(route), rtol=0, atol=0)
        for value, original in zip(args, before, strict=True):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
    left.mul_(2)
    right.add_(0.5)
    a.copy_(left)
    weights.copy_(right)
    offsets.copy_(torch.tensor(skewed, dtype=offset_dtype, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured.cpu(), expected(skewed), rtol=0, atol=0)
