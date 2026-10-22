from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import inspect
from itertools import accumulate
from itertools import pairwise
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test._cute_binding import _require_cute_codegen
from test.cute_population_contracts import _target
from test.test_cute_grouped_d_shift import _config as _ordinary_config
from test.test_cute_grouped_large_device_seeds import _kernel
from test.test_cute_grouped_paired_clc_layout import _sdk
from test.test_cute_grouped_prefix_adaptive_config import _args
from test.test_cute_grouped_prefix_adaptive_config import _recipes

import helion
from helion._compiler.cute.grouped_prefix_adaptive_n_codegen import _is_role
from helion._compiler.cute.grouped_prefix_adaptive_n_codegen import _role_payload
from helion._testing import DEVICE
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.cute.launcher import _create_cute_wrapper

if TYPE_CHECKING:
    from pathlib import Path

KEY = "tcgen05_grouped_prefix_adaptive_n"
BALANCED = "tcgen05_grouped_balanced_workers"


def _recipe(spec):
    if recipes := _recipes(spec):
        return deepcopy(recipes[0])
    # Emission eligibility also covers explicit configs outside the current
    # sampler's logical block domain. It remains separate from seed ranking.
    profile = spec._cute_tcgen05_config.grouped_prefix_adaptive_n_profile
    assert profile is not None
    config = _ordinary_config("mn")
    config.config.update(
        tcgen05_ab_stages=profile.primary.ab_stages,
        tcgen05_c_stages=profile.primary.c_stages,
        tcgen05_grouped_worklist_mma_n=profile.primary.n,
        tcgen05_grouped_worklist_source_m_tile=128,
        tcgen05_grouped_balanced_workers=False,
        tcgen05_l2_swizzle_size=8,
        tcgen05_consumer_regs=256,
        cute_min_blocks_per_mp=1,
        tcgen05_grouped_prefix_adaptive_n=True,
    )
    return config


def _plans(source: str) -> list[dict[str, Any]]:
    return next(
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )


def _kernel_ast(tree: ast.Module) -> ast.FunctionDef:
    kernels = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(value) == "cute.kernel" for value in node.decorator_list)
    ]
    assert len(kernels) == 1
    return kernels[0]


def _core(tree: ast.Module, width: int) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.endswith(f"_n{width}")
    )


def _decoder(tree: ast.Module) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_decode")
    )


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    result = {}
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args())
        spec = bound.config_spec
        profile = spec._cute_tcgen05_config.grouped_prefix_adaptive_n_profile
        assert profile is not None
        assert spec.block_sizes[2].declared_max_size == 64
        recipe = _recipe(spec)
        for balanced in (False, True):
            recipe.config[BALANCED] = balanced
            result[str(balanced)] = bound.to_code(spec.normalized_config(recipe))
        for branch in (profile.primary, profile.alternate):
            ordinary = deepcopy(recipe)
            ordinary.config.pop(KEY)
            ordinary.config.update(
                tcgen05_grouped_balanced_workers=True,
                tcgen05_grouped_worklist_mma_n=branch.n,
                tcgen05_ab_stages=branch.ab_stages,
                tcgen05_c_stages=branch.c_stages,
            )
            result[str(branch.n)] = bound.to_code(spec.normalized_config(ordinary))
        ordinary.config[KEY] = False
        false_source = bound.to_code(spec.normalized_config(ordinary))
        ordinary.config.pop(KEY)
        assert false_source == bound.to_code(spec.normalized_config(ordinary))
    return result


def test_one_launch_one_arena_and_exact_operand_acc_output_payloads(
    sources: dict[str, str],
) -> None:
    tree = ast.parse(sources["False"])
    entry = _kernel_ast(tree)
    plans = _plans(sources["False"])
    assert plans[0]["balanced_workers"] is False
    assert plans[0]["prefix_adaptive_n"]["arena_bytes"] == 230400
    assert plans[0]["prefix_adaptive_n"]["tmem_columns"] == 256
    assert plans[0]["prefix_adaptive_n"]["output_tile"] == [64, 64]
    assert [plan["bn"] for plan in plans if plan["kind"] == "tcgen05_ab_tma"] == [
        192,
        256,
    ]
    assert sources["False"].count("cute.arch.alloc_smem(") == 1
    assert "StaticPersistentGroupTileScheduler.create" not in sources["False"]
    assert "tcgen05_work_tile_smem" not in sources["False"]
    assert "tcgen05_sched_pipeline" not in sources["False"]
    assert "warp_reduction_sum" not in sources["False"]
    assert not any(isinstance(node, ast.Return) for node in ast.walk(entry))
    for width in (192, 256):
        native = _kernel_ast(ast.parse(sources[str(width)]))
        before = [node for node in ast.walk(native) if _is_role(node)]
        core = _core(tree, width)
        after = [
            node
            for node in core.body
            if isinstance(node, ast.If)
            and any(isinstance(child, ast.While) for child in node.body)
        ]
        assert len(before) == len(after) == 3
        for previous, current in zip(before, after, strict=True):
            old_loop = next(
                node for node in previous.body if isinstance(node, ast.While)
            )
            new_loop = next(
                node for node in current.body if isinstance(node, ast.While)
            )
            payload = _role_payload(old_loop)
            assert ast.dump(ast.Module(payload, [])) == ast.dump(
                ast.Module(new_loop.body[-len(payload) - 1 : -1], [])
            )
            assert ast.unparse(new_loop.body[-1]).endswith(
                "tcgen05_grouped_virtual_grid[2]"
            )
        # The original C acquire, every copy/fence and the final drains remain.
        for call in (
            "tcgen05_acc_pipeline.consumer_release",
            "tcgen05_ab_pipeline.consumer_release",
            "tcgen05_c_pipeline.producer_acquire",
            "tcgen05_c_pipeline.producer_commit",
            "tcgen05_c_pipeline.producer_tail",
            "tcgen05_acc_pipeline.producer_tail",
            "tcgen05_tmem_allocator.free",
        ):
            counts = [
                sum(
                    isinstance(node, ast.Call) and ast.unparse(node.func) == call
                    for node in ast.walk(root)
                )
                for root in (native, core)
            ]
            assert counts[0] == counts[1], call


@pytest.mark.parametrize(
    "groups,m,n,k,dtype,offset_dtype",
    [
        (2, 384, 768, 256, torch.bfloat16, torch.int32),
        (8, 640, 768, 256, torch.float16, torch.int64),
        (32, 1024, 1536, 256, torch.bfloat16, torch.int64),
    ],
)
def test_general_tensor_extents_dtypes_and_live_metadata_emit_native_pair(
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
        spec = bound.config_spec
        recipe = _recipe(spec)
        recipe.config[BALANCED] = False
        source = bound.to_code(spec.normalized_config(recipe))
        profile = spec._cute_tcgen05_config.grouped_prefix_adaptive_n_profile
        assert profile is not None
        assert spec.block_sizes[2].declared_max_size == 64
    plans = _plans(source)
    assert plans[0]["group_count"] == groups
    assert plans[0]["m_size"] == m
    assert plans[0]["n_size"] == n
    assert plans[0]["k_total_size"] == k
    assert plans[0]["prefix_adaptive_n"]["arena_bytes"] == profile.arena_bytes
    assert all(plan.get("bk", 64) == 64 for plan in plans)
    for width in (192, 256):
        core = _core(ast.parse(source), width)
        loops = [
            node
            for node in ast.walk(core)
            if isinstance(node, ast.For)
            and any(
                isinstance(argument, ast.Name) and argument.id == "_BLOCK_SIZE_2"
                for argument in ast.walk(node.iter)
            )
        ]
        assert len(loops) == 2
    if offset_dtype is torch.int64:
        assert "tcgen05_grouped_raw_start = cutlass.Int64(" in source
    assert "cutlass.Int32(tcgen05_grouped_visible_extent)" in source


def test_unregistered_native_axes_reject_an_explicit_recipe(
    sources: dict[str, str],
) -> None:
    del sources
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        supported = _cpu_bind(_kernel(), _args())
        config = _recipe(supported.config_spec)
        unregistered = _cpu_bind(
            _kernel(),
            _args(groups=4, m=257, n=768, k=256, dtype=torch.float16),
        )
        assert (
            unregistered.config_spec._cute_tcgen05_config._matmul_block_indices()
            is None
        )
        with pytest.raises(InvalidConfig):
            unregistered.to_code(config)


@pytest.mark.parametrize("balanced", [False, True])
def test_actual_global_choice_workers_zero_tail_and_cluster_peers(
    sources: dict[str, str], balanced: bool
) -> None:
    entry = deepcopy(_kernel_ast(ast.parse(sources[str(balanced)])))
    start = next(
        index
        for index, node in enumerate(entry.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.endswith("_rows")
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
    globals_: dict[str, Any] = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                grid_dim=lambda: (2, 1, state["cap"]),
                block_idx=lambda: (0, 0, state["z"]),
            )
        ),
        "tcgen05_grouped_problem_sizes": {(15, 3): 16},
        "tcgen05_grouped_starts": object(),
        "_helion_grouped_prefix_smem": object(),
    }
    for argument in _kernel_ast(ast.parse(sources[str(balanced)])).args.args:
        globals_[argument.arg] = object()
    for width in (192, 256):
        name = _core(ast.parse(sources[str(balanced)]), width).name
        globals_[name] = lambda *arguments, width=width: selected.append(
            (width, arguments)
        )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([entry], [])),
            "<actual global selector>",
            "exec",
        ),
        globals_,
    )
    for cap in (1, 38, 66, 71, 76):
        state["cap"] = cap
        for rows in (0, 1, 16, 22, 64, 257, 8_388_608):
            globals_["tcgen05_grouped_problem_sizes"] = {(15, 3): rows}
            totals = {192: rows * 8, 256: rows * 6}
            waves = {
                width: max(1, (total + cap - 1) // cap)
                for width, total in totals.items()
            }
            width = 192 if waves[192] * 192 <= waves[256] * 256 else 256
            workers = max(
                1,
                min(
                    cap,
                    (totals[width] + waves[width] - 1) // waves[width]
                    if balanced
                    else totals[width],
                ),
            )
            for z in range(cap):
                state["z"] = z
                selected.clear()
                globals_[entry.name]()
                if rows == 0 or z >= workers:
                    assert not selected
                else:
                    assert len(selected) == 1 and selected[0][0] == width
                    assert selected[0][1][-2] == (2, 1, workers)
                    assert selected[0][1][-5] == totals[width]


def test_actual_host_wrapper_has_two_ab_descriptors_one_bounded_d_and_one_launch(
    sources: dict[str, str],
) -> None:
    source = sources["False"]
    plans = _plans(source)
    plans[0]["tcgen05_grouped_static_reserved_sms"] = 10
    kernel = SimpleNamespace(
        __name__=_kernel_ast(ast.parse(source)).name,
        _helion_cute_wrapper_plans=plans,
        _helion_cute_cluster_shape=(2, 1, 1),
    )
    schema = (
        ("tensor", "torch.int32", 1, (17,), (1,), 16),
        ("tensor", "torch.bfloat16", 2, (2048, 2048), (2048, 1), 16),
        ("tensor", "torch.bfloat16", 3, (16, 1536, 2048), (1536 * 2048, 2048, 1), 16),
        ("tensor", "torch.bfloat16", 2, (2048, 1536), (1536, 1), 16),
        ("wrapper_host_scalar", "tcgen05_grouped_total_clusters", "int"),
    )
    with _forbid_cuda_work():
        wrapper = _create_cute_wrapper(kernel, schema, (32, 8, 1), num_sm=152)
        host = inspect.getsource(wrapper)
    assert host.count(").launch(") == 1
    assert "cutlass.Int32(71)" in host
    assert "TensorMapManager" not in host and "update_tensormap" not in host
    assert "tma_atom_a_alternate" in host and "tma_atom_b_alternate" in host
    assert host.count("CopyBulkTensorTileS2GOp()") == 1
    assert "use_pdl" not in host


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_real_sdk_publication_scan_and_decoder_ir_remain_int32(
    offset_dtype: torch.dtype,
    tmp_path: Path,
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args(offset_dtype=offset_dtype))
        source = bound.to_code(
            bound.config_spec.normalized_config(_recipe(bound.config_spec))
        )
    tree = ast.parse(source)
    entry = _kernel_ast(tree)
    begin = next(
        index
        for index, node in enumerate(entry.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "tcgen05_grouped_metadata_thread_idx"
    )
    end = next(
        index
        for index, node in enumerate(entry.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.arch.sync_threads"
    )
    publisher = ast.FunctionDef(
        name="publish_prefix_cpu",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg=entry.args.args[0].arg),
                ast.arg(arg="tcgen05_grouped_problem_sizes"),
                ast.arg(arg="tcgen05_grouped_starts"),
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=deepcopy(entry.body[begin : end + 1]),
        decorator_list=[ast.parse("cute.jit", mode="eval").body],
        returns=None,
        type_comment=None,
    )
    module_path = tmp_path / "typed_prefix_publisher.py"
    module_path.write_text(
        "from __future__ import annotations\nimport cutlass\nimport cutlass.cute as cute\n"
        + ast.unparse(
            ast.fix_missing_locations(ast.Module([publisher, _decoder(tree)], []))
        )
        + "\n"
    )
    with _forbid_cuda_work(), _sdk() as sdk:
        spec = importlib.util.spec_from_file_location(
            "typed_prefix_publisher", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = sdk.func.FuncOp("prefix_publication_ir", ([], []))
        with sdk.ir.InsertionPoint(function.add_entry_block()):
            offsets = sdk.cute.make_tensor(
                sdk.cute.make_ptr(
                    sdk.cutlass.Int32
                    if offset_dtype is torch.int32
                    else sdk.cutlass.Int64,
                    0,
                    sdk.cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                sdk.cute.make_layout((17,), stride=(1,)),
            )
            matrix = sdk.cute.make_tensor(
                sdk.cute.make_ptr(
                    sdk.cutlass.Int32, 0, sdk.cute.AddressSpace.smem, assumed_align=16
                ),
                sdk.cute.make_layout((16, 4), stride=(4, 1)),
            )
            starts = sdk.cute.make_tensor(
                sdk.cute.make_ptr(
                    sdk.cutlass.Int32, 0, sdk.cute.AddressSpace.smem, assumed_align=16
                ),
                sdk.cute.make_layout((16,), stride=(1,)),
            )
            getattr(module, publisher.name)(offsets, matrix, starts)
            values = getattr(module, _decoder(tree).name)(
                sdk.cutlass.Int32(0), matrix, 8, 16
            )
            assert len(values) == 4
            sdk.func.ReturnOp([])
        assert sdk.module.operation.verify()
        text = str(sdk.module)
        assert "ballot" in text and "ctpop" in text
        if offset_dtype is torch.int64:
            assert "arith.trunci" in text


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _impure_grouped_loop(
    a: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor, variant: int
) -> torch.Tensor:
    hl.specialize(variant)
    rows, reduction = a.shape
    groups, columns, _ = weights.shape
    bm = hl.register_block_size(256)
    bn = hl.register_block_size(128)
    bk = hl.register_block_size(64)
    out = torch.empty((rows, columns), device=a.device, dtype=a.dtype)
    for group_tile, row_tile, column_tile in hl.tile(
        [groups, rows, columns], block_size=[1, bm, bn]
    ):
        group = group_tile.index.sum()
        start = offsets[group]
        length = offsets[group + 1] - offsets[group]
        row = start + row_tile.index
        valid = row_tile.index < length
        acc = hl.full(
            [row_tile, column_tile], 1.0 if variant == 1 else 0.0, dtype=torch.float32
        )
        for inner in hl.tile(64 if variant == 2 else 0, reduction, block_size=bk):
            if variant == 7:
                left = hl.load(
                    a,
                    [row, inner],
                    extra_mask=valid[:, None] & (inner.index < reduction - 1)[None, :],
                )
            else:
                left = hl.load(a, [row, inner], extra_mask=valid[:, None])
            if variant == 3:
                left = left * 1.0
            acc = torch.addmm(
                acc,
                left,
                weights[group, column_tile, inner].T,
                alpha=2 if variant == 4 else 1,
            )
            if variant == 5:
                acc = acc * 1.0
            if variant == 6:
                hl.store(out, [row, column_tile], acc, extra_mask=valid[:, None])
        hl.store(out, [row, column_tile], acc, extra_mask=valid[:, None])
    return out


@pytest.mark.parametrize("variant", range(1, 8))
def test_impure_or_transformed_reductions_cannot_enter_the_recipe(variant: int) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        args = _args(groups=4, m=640, n=768, k=256)
        pure = _cpu_bind(_kernel(), args)
        config = _recipe(pure.config_spec)
        try:
            impure = _cpu_bind(_impure_grouped_loop, (*args, variant))
        except AssertionError as failure:
            # The existing frontend rejects nondefault dot kwargs before the
            # native purity proof can run. It still cannot enter this recipe.
            assert variant == 4 and str(failure) == "dot kwargs not supported"
            return
        assert not impure.config_spec._cute_tcgen05_config.grouped_prefix_adaptive_n_supported
        with pytest.raises(InvalidConfig):
            impure.to_code(config)


@pytest.mark.parametrize(
    "counts", [(0, 2, 0, 1), (1,) * 16, (0,) * 31 + (3,), (1,) * 32]
)
@pytest.mark.parametrize("features", [4, 8])
def test_actual_decoder_matches_literal_group_tiles_and_empty_prefixes(
    sources: dict[str, str], counts: tuple[int, ...], features: int
) -> None:
    decoder = deepcopy(_decoder(ast.parse(sources["False"])))
    decoder.decorator_list = []
    groups = len(counts)
    prefixes = list(accumulate(counts))
    matrix = {(group, 0): count * 128 for group, count in enumerate(counts)}
    matrix.update({(group, 3): prefix for group, prefix in enumerate(prefixes)})
    state = {"lane": 0, "linear": 0}

    def i32(value: int) -> int:
        value &= (1 << 32) - 1
        return value - (1 << 32) if value >= (1 << 31) else value

    def ballot(predicate: bool) -> int:
        matches = [
            lane < groups and state["linear"] // features < prefixes[lane]
            for lane in range(32)
        ]
        assert predicate == matches[state["lane"]]
        return i32(sum(int(match) << lane for lane, match in enumerate(matches)))

    namespace = {
        "cutlass": SimpleNamespace(Int32=i32),
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
            "<actual group decoder>",
            "exec",
        ),
        namespace,
    )
    expected = [
        (row, column, group, count * 128)
        for group, count in enumerate(counts)
        for row in range(count)
        for column in range(features)
    ]
    for linear, answer in enumerate(expected):
        state["linear"] = linear
        for lane in range(32):
            state["lane"] = lane
            assert namespace[decoder.name](linear, matrix, features, groups) == answer


@pytest.mark.cuda
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_general_prefix_recipe_live_branch_changes_and_graph_input_preservation(
    offset_dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family tcgen05")
    _require_cute_codegen()
    groups, rows, columns, reduction = 16, 2048, 1536, 256
    generator = torch.Generator().manual_seed(953)

    def values() -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.randint(0, 9, (rows, reduction), generator=generator).float() / 8
        right = (
            torch.randint(
                -4, 5, (groups, columns, reduction), generator=generator
            ).float()
            / 8
        )
        right += torch.arange(groups)[:, None, None] / 2
        return left.bfloat16(), right.bfloat16()

    left, right = values()
    a, weights = left.cuda(), right.cuda()
    uniform = list(range(0, rows + 1, 128))
    skewed = list(accumulate([0, *([1] * 6), *([0] * 9), rows - 6]))
    offsets = torch.tensor(uniform, dtype=offset_dtype, device=DEVICE)
    args = (a, weights, offsets)
    bound = _kernel()._bind_isolated(args)
    recipe = _recipe(bound.config_spec)
    recipe.config[BALANCED] = False
    bound.set_config(recipe)
    initial = bound(*args)
    expected = torch.cat(
        [
            (left[begin:end].double() @ right[group].double().T).bfloat16()
            for group, (begin, end) in enumerate(pairwise(uniform))
        ]
    )
    torch.testing.assert_close(initial.cpu(), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = bound(*args)
    routing_cases = [
        uniform,
        skewed,
        [0] * (groups + 1),
        [0] * groups + [rows],
        uniform,
    ]
    if offset_dtype is torch.int64:
        wide = 1 << 35
        routing_cases.insert(-1, [-wide, -wide + 128, *([256] * 14), rows])
    for routing in routing_cases:
        left, right = values()
        a.copy_(left)
        weights.copy_(right)
        offsets.copy_(offsets.new_tensor(routing))
        output.fill_(float("nan"))
        graph.replay()
        expected = torch.full((rows, columns), float("nan"), dtype=torch.bfloat16)
        covered = torch.zeros(rows, dtype=torch.bool)
        logical = torch.arange(rows, dtype=offset_dtype)
        for group, (raw_start, raw_end) in enumerate(pairwise(routing)):
            start = torch.tensor(raw_start, dtype=offset_dtype)
            length = torch.tensor(raw_end, dtype=offset_dtype) - start
            indices = start + logical
            valid = (logical < length) & (indices >= 0) & (indices < rows)
            selected = indices[valid].long()
            if selected.numel():
                expected[selected] = (
                    left[selected].double() @ right[group].double().T
                ).bfloat16()
                covered[selected] = True
        observed = output.cpu()
        torch.testing.assert_close(observed[covered], expected[covered], rtol=0, atol=0)
        assert torch.isnan(observed[~covered]).all()
        torch.testing.assert_close(a.cpu(), left, rtol=0, atol=0)
        torch.testing.assert_close(weights.cpu(), right, rtol=0, atol=0)
        assert offsets.cpu().tolist() == routing
