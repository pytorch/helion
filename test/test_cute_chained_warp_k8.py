from __future__ import annotations

import ast
import importlib
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.chained_warp_mma import complete_warp_columns
from helion._compiler.cute.chained_warp_mma import emit_warp_mma
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.device_ir import GraphInfo


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _seeded_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    seed: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        (a.size(0), a.size(1), b.size(2)), dtype=torch.float32, device=a.device
    )
    for batch, row in hl.tile([a.size(0), a.size(1)], block_size=[1, None]):
        first = hl.dot(
            a[batch.id, row, :], b[batch.id, :, :], acc=seed[batch.id, row, :]
        )
        second = hl.dot(first.to(a.dtype), c[batch.id, :, :])
        output[batch.id, row, :] = second + first * 0.125
    return output


def _inputs(
    dtype: torch.dtype,
    *,
    rows: int = 16,
    columns: int = 8,
    reduction: int = 8,
    transpose: bool = False,
    device: str = "cpu",
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    views, backing = [], []
    for index, (m, n) in enumerate(
        ((rows, reduction), (reduction, columns), (columns, columns), (rows, columns))
    ):
        raw = torch.empty(
            (2, m + 2, n + 4),
            dtype=dtype if index < 3 else torch.float32,
            device=device,
        )
        if transpose and index < 3:
            raw = torch.empty((2, n + 4, m + 2), dtype=dtype, device=device)
            view = raw[:, 2 : n + 2, 1 : m + 1].transpose(1, 2)
        else:
            view = raw[:, 1 : m + 1, 2 : n + 2]
        views.append(view)
        backing.append(raw)
    return tuple(views), tuple(backing)


def _config(
    rows: int = 16, schedule: str = "coalesced", **kwargs: Any
) -> helion.Config:
    return helion.Config(
        block_sizes=[rows], num_warps=4, cute_chained_mma_schedule=schedule, **kwargs
    )


def _source(
    args: tuple[torch.Tensor, ...], config: helion.Config, *, capacity: int = 232448
) -> tuple[str, list[ChainedMatmulPlan]]:
    plans: list[ChainedMatmulPlan] = []
    original = chained_matmul.plan_chained_matmul

    def observe(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        if plan is not None:
            plans.append(plan)
        return plan

    with (
        _cpu_codegen(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
        patch.object(chained_matmul, "plan_chained_matmul", observe),
    ):
        source = _seeded_chain._bind_isolated(args).to_code(config)
    return source, plans


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("larger", (False, True))
def test_original_root_codegen_has_k8_atoms_and_original_joined_publication(
    dtype, larger
):
    args, _ = _inputs(
        dtype,
        rows=19 if larger else 16,
        columns=16 if larger else 8,
        reduction=8,
        transpose=larger,
    )
    source, plans = _source(
        args, _config(32 if larger else 16, "cp_async" if larger else "coalesced")
    )
    assert len(plans) == 1 and plans[0].warp_workspace is None
    assert "(16, 8, 8)" in source
    assert "num_matrices=2)" in source and "num_matrices=1)" in source
    assert source.count("_mma = cute.make_tiled_mma") == 2
    if larger:
        assert "(16, 8, 16)" in source
    tree = ast.parse(source)
    # Native shared seed and operand readers remain inside complete active warps.
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "chain_thread"
                for target in item.targets
            )
            for item in node.body
        )
    )
    for stage in range(2):
        compute = next(
            i
            for i, stmt in enumerate(function.body)
            if isinstance(stmt, ast.If)
            and any(
                isinstance(item, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == f"chain_{stage}_mma"
                    for target in item.targets
                )
                for item in stmt.body
            )
        )
        before = [ast.unparse(item) for item in function.body[:compute]]
        join = max(
            i for i, item in enumerate(before) if item == "cute.arch.sync_threads()"
        )
        assert all(
            isinstance(item, ast.Assign)
            and ast.unparse(item.targets[0]).startswith(f"chain_{stage}_c")
            for item in function.body[join + 1 : compute]
        )
        assert ast.unparse(function.body[compute + 1]) == "cute.arch.sync_threads()"
        assert f"cute.autovec_copy(chain_{stage}_acc, chain_{stage}_sc)" in ast.unparse(
            function.body[compute]
        )
    assert "chain_0_seed_coords" in source
    assert "chain_0_c[" in source  # FP32 fanout is retained after the narrowed input.


@pytest.mark.parametrize("schedule", ("cp_async_register", "cp_async_register_reuse"))
def test_k8_does_not_expand_other_protocols(schedule):
    args, _ = _inputs(torch.float16)
    source, plans = _source(args, _config(schedule=schedule))
    assert not plans and "(16, 8, 8)" not in source


@pytest.mark.parametrize("mode", ("packed", "packed_direct"))
def test_k8_retains_explicit_packed_result_veto(mode):
    args, _ = _inputs(torch.float16, columns=32)
    with pytest.raises(exc.BackendUnsupported, match="warp result transport"):
        _source(args, _config(cute_chained_warp_result_transport=mode))


def test_k8_does_not_use_capacity_rescue_even_with_all_mma_warps_active():
    args, _ = _inputs(torch.float16, columns=32)
    with (
        patch.object(
            chained_matmul,
            "plan_contraction_workspace",
            side_effect=AssertionError("K8 cannot pack C"),
        ),
        pytest.raises(exc.BackendUnsupported, match="shared bytes exceed"),
    ):
        _source(args, _config(), capacity=1)


@pytest.mark.parametrize("reduction", (0, 4, 12))
def test_incomplete_k_panels_decline_emission(reduction):
    with pytest.raises(ValueError, match="complete K8 or K16"):
        emit_warp_mma(
            "test",
            "cutlass.Float16",
            (16, 8, reduction),
            32,
            {"a": 1, "b": 0},
            ["test_acc.fill(0.0)"],
        )


@pytest.mark.parametrize("columns,admitted", ((16, True), (24, False), (32, True)))
def test_partial_active_warp_c_panel_rejects_with_actual_cute_ownership(
    columns, admitted
):
    import cutlass
    import cutlass.cute as cute

    assert complete_warp_columns(columns, 128) is admitted
    ir = importlib.import_module("cutlass._mlir.ir")
    warps = min(4, 2 ** (columns.bit_length() - 4))
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(
                        cutlass.Float16, cutlass.Float32, (16, 8, 8)
                    )
                ),
                atom_layout_mnk=(1, warps, 1),
            )
            cells = []
            for thread in range(32 * warps):
                fragment = mma.get_slice(thread).partition_C(
                    cute.make_identity_tensor((16, columns))
                )
                cells.extend(
                    tuple(map(int, fragment[i])) for i in range(cute.size(fragment))
                )
            assert len(cells) == len(set(cells))
            complete = set(cells) == {(r, c) for r in range(16) for c in range(columns)}
            assert complete is admitted
            if not admitted:
                assert sum(c >= columns for _, c in cells) == 128
        assert module.operation.verify()


@pytest.mark.parametrize("dtype_name", ("Float16", "BFloat16"))
@pytest.mark.parametrize(
    "shape,threads,padding", (((16, 8, 8), 32, 0), ((32, 16, 24), 64, 8))
)
@pytest.mark.parametrize("axes", ((0, 0), (0, 1), (1, 0), (1, 1)))
def test_actual_ldmatrix_packets_deliver_ordered_mma_fragments(
    dtype_name, shape, threads, padding, axes
):
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = {"Float16": cutlass.Float16, "BFloat16": cutlass.BFloat16}[dtype_name]
    rows, columns, reduction = shape
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 8))
                ),
                atom_layout_mnk=(1, threads // 32, 1),
            )
            covered = set()
            for thread in range(threads):
                c = mma.get_slice(thread).partition_C(
                    cute.make_identity_tensor((rows, columns))
                )
                cells = [tuple(map(int, c[i])) for i in range(cute.size(c))]
                assert not covered.intersection(cells)
                covered.update(cells)
            assert covered == {(r, c) for r in range(rows) for c in range(columns)}
            for role, extent, inner, matrices in (
                ("A", rows, axes[0], 2),
                ("B", columns, axes[1], 1),
            ):
                operand_shape = (extent, reduction)
                identity = cute.make_identity_tensor(operand_shape)
                stride = (reduction + padding, 1) if inner else (1, extent + padding)
                layout = cute.make_layout(operand_shape, stride=stride)
                capacity = extent * reduction + 8 * max(extent, reduction)
                copy = getattr(cute, f"make_tiled_copy_{role}")(
                    cute.make_copy_atom(
                        cute.nvgpu.warp.LdMatrix8x8x16bOp(
                            transpose=inner == 0, num_matrices=matrices
                        ),
                        dtype,
                    ),
                    mma,
                )
                for warp in range(threads // 32):
                    source = [
                        copy.get_slice(warp * 32 + lane).partition_S(identity)
                        for lane in range(32)
                    ]
                    fragment = [
                        getattr(mma.get_slice(warp * 32 + lane), f"partition_{role}")(
                            identity
                        )
                        for lane in range(32)
                    ]
                    dest = [
                        copy.get_slice(warp * 32 + lane).retile(fragment[lane])
                        for lane in range(32)
                    ]
                    for tile in range(cute.size(source[0], mode=[1])):
                        for k in range(reduction // 8):
                            packets = []
                            for address_lane in range(8 * matrices):
                                panel = source[address_lane][None, tile, k]
                                coordinates = [
                                    tuple(map(int, panel[i]))
                                    for i in range(cute.size(panel))
                                ]
                                offsets = [int(layout(coord)) for coord in coordinates]
                                assert len(offsets) == 8 and offsets == list(
                                    range(offsets[0], offsets[0] + 8)
                                )
                                assert (
                                    offsets[0] % 8 == 0
                                    and offsets[0] >= 0
                                    and offsets[-1] < capacity
                                )
                                assert all(
                                    0 <= r < extent and k * 8 <= c < (k + 1) * 8
                                    for r, c in coordinates
                                )
                                packets.append(coordinates)
                            # Independent PTX ldmatrix m8n8 delivery: groups of
                            # four lanes own adjacent half pairs of one row.
                            for lane in range(32):
                                delivered = []
                                for matrix in range(matrices):
                                    for half in range(2):
                                        r, c = lane // 4, 2 * (lane % 4) + half
                                        if inner == 0:
                                            r, c = c, r
                                        delivered.append(packets[matrix * 8 + r][c])
                                target = dest[lane][None, tile, k]
                                original = fragment[lane][None, tile, k]
                                assert delivered == [
                                    tuple(map(int, target[i]))
                                    for i in range(cute.size(target))
                                ]
                                assert delivered == [
                                    tuple(map(int, original[i]))
                                    for i in range(cute.size(original))
                                ]
        assert module.operation.verify()


@pytest.mark.parametrize("dtype_name", ("Float16", "BFloat16"))
@pytest.mark.parametrize("axes", ((0, 0), (0, 1), (1, 0), (1, 1)))
def test_exact_emitted_k8_copies_and_mma_lower_to_native_ir(dtype_name, axes):
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    manager = importlib.import_module("cutlass._mlir.passmanager")
    dtype = {"Float16": cutlass.Float16, "BFloat16": cutlass.BFloat16}[dtype_name]
    lines = emit_warp_mma(
        "test",
        f"cutlass.{dtype_name}",
        (16, 8, 24),
        32,
        {"a": axes[0], "b": axes[1]},
        ["test_acc.fill(0.125)"],
    )
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            scope = {
                "cutlass": cutlass,
                "cute": cute,
                "chain_thread": 0,
                "range": range,
            }
            for role, shape, axis, offset in (
                ("a", (16, 24), axes[0], 0),
                ("b", (8, 24), axes[1], 4096),
            ):
                stride = (shape[1] + 8, 1) if axis else (1, shape[0] + 8)
                scope[f"test_{role}"] = cute.make_tensor(
                    cute.make_ptr(
                        dtype, offset, cute.AddressSpace.smem, assumed_align=128
                    ),
                    cute.make_layout(shape, stride=stride),
                )
            tree = ast.parse("\n".join(lines))
            loop = tree.body[-1]
            assert isinstance(loop, ast.For)
            assert isinstance(loop.iter, ast.Call)
            assert ast.unparse(loop.iter.func) == "cutlass.range_constexpr"
            loop.iter.func = ast.Name(id="range", ctx=ast.Load())
            exec(
                compile(ast.fix_missing_locations(tree), "<emitted-k8>", "exec"), scope
            )
        manager.PassManager.parse(
            "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,convert-cute-to-core,canonicalize)"
        ).run(module.operation)
        assert module.operation.verify()
        lowered = str(module)
        assert lowered.count("nvvm.ldmatrix") == 6
        assert lowered.count("shape = #nvvm.shape<m = 16, n = 8, k = 8>") == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("larger", (False, True))
def test_gpu_guarded_k8_chain_seed_fanout_and_mixed_k(dtype, larger):
    torch.manual_seed(1984)
    args, backing = _inputs(
        dtype,
        rows=19 if larger else 16,
        columns=16 if larger else 8,
        reduction=8,
        transpose=larger,
        device="cuda",
    )
    a, b, c, seed = args
    for tensor in backing:
        tensor.fill_(-113.0)
    for tensor in args:
        tensor.copy_(torch.randn_like(tensor) * 0.125)
    snapshots = [tensor.clone() for tensor in backing]
    first = (a.double() @ b.double() + seed.double()).float()
    expected = (first.to(dtype).double() @ c.double() + first.double() * 0.125).float()
    config = _config(32 if larger else 16, "cp_async" if larger else "coalesced")
    bound = _seeded_chain._bind_isolated(args)
    source = bound.to_code(config)
    assert "(16, 8, 8)" in source and source.count("_mma = cute.make_tiled_mma") == 2
    compiled = bound.compile_config(config)
    guard_elements = 256
    output_backing = torch.empty(
        expected.numel() + 2 * guard_elements, device="cuda", dtype=torch.float32
    )
    guarded_output = output_backing[guard_elements:-guard_elements].view(expected.shape)
    original_empty = torch.empty

    def guarded_empty(shape, **kwargs):
        if tuple(shape) == tuple(expected.shape):
            assert kwargs["dtype"] is torch.float32
            return guarded_output
        return original_empty(shape, **kwargs)

    for poison in (37.0, -41.0):
        output_backing.fill_(poison)
        with patch("torch.empty", guarded_empty):
            actual = compiled(*args)
        assert actual.data_ptr() == guarded_output.data_ptr()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
        assert torch.all(output_backing[:guard_elements] == poison)
        assert torch.all(output_backing[-guard_elements:] == poison)
        for tensor, snapshot in zip(backing, snapshots, strict=True):
            assert torch.equal(tensor.view(torch.uint8), snapshot.view(torch.uint8))


def test_k8_does_not_discover_native_tcgen():
    args, _ = _inputs(torch.float16)
    with pytest.raises(exc.InvalidConfig, match="invalid chained MMA schedule"):
        _source(args, _config(schedule="tcgen05_tmem"))


def test_original_cold_seed_pool_reaches_native_k8_without_new_coordinates():
    args, _ = _inputs(torch.float16)
    with _cpu_codegen():
        bound = _seeded_chain._bind_isolated(args)
        spec = bound.config_spec
        assert spec.cute_chained_matmul_search_enabled
        assert not spec.cute_chained_tcgen05_search_enabled
        for schedule in ("coalesced", "cp_async"):
            seed = next(
                seed
                for seed in spec.compiler_seed_configs
                if seed.config.get("cute_chained_mma_schedule") == schedule
            )
            normalized = spec.normalized_config(seed)
            source = bound.to_code(normalized)
            assert "(16, 8, 8)" in source
            assert source.count("_mma = cute.make_tiled_mma") == 2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fixed_two(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        (a.size(0), a.size(1), b.size(2)), device=a.device, dtype=torch.float32
    )
    for batch in hl.tile(a.size(0), block_size=1):
        first = hl.dot(a[batch.id, :, :], b[batch.id, :, :])
        second = hl.dot(a[batch.id, :, :] * 0.5, b[batch.id, :, :])
        output[batch.id, :, :] = first + second
    return output


@pytest.mark.parametrize("rows,reduction", ((8, 8), (16, 4)))
def test_logical_m8_or_incomplete_k_remain_on_original_fallback(rows, reduction):
    with _cpu_codegen():
        source = _fixed_two._bind_isolated(
            (
                torch.empty((2, rows, reduction), dtype=torch.float16),
                torch.empty((2, reduction, 8), dtype=torch.float16),
            )
        ).to_code(helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced"))
    assert "_mma = cute.make_tiled_mma" not in source


def _k24_component_source(dtype: torch.dtype) -> str:
    dtype_name = "cutlass.BFloat16" if dtype is torch.bfloat16 else "cutlass.Float16"
    seed = [
        "test_seed_coords = test_thr.partition_C(cute.make_identity_tensor((16, 8)))",
        "for test_seed_index in cutlass.range_constexpr(cute.size(test_acc)):",
        "    test_seed_row, test_seed_col = test_seed_coords[test_seed_index]",
        "    test_acc[test_seed_index] = cutlass.Float32(seed[test_seed_row, test_seed_col])",
    ]
    emission = emit_warp_mma(
        "test", dtype_name, (16, 8, 24), 32, {"a": 1, "b": 0}, seed
    )
    body = [
        "chain_thread = cutlass.Int32(cute.arch.thread_idx()[0])",
        f"a_ptr = cute.arch.alloc_smem({dtype_name}, 512, alignment=128)",
        f"b_ptr = cute.arch.alloc_smem({dtype_name}, 384, alignment=128)",
        "test_a = cute.make_tensor(a_ptr, cute.make_layout((16, 24), stride=(32, 1)))",
        "test_b = cute.make_tensor(b_ptr, cute.make_layout((8, 24), stride=(1, 16)))",
        "for index in cutlass.range(chain_thread, 16 * 24, 32):",
        "    test_a[index // 24, index % 24] = a[index // 24, index % 24]",
        "for index in cutlass.range(chain_thread, 8 * 24, 32):",
        "    test_b[index // 24, index % 24] = b[index % 24, index // 24]",
        "cute.arch.sync_threads()",
        *emission,
        "for index in cutlass.range_constexpr(cute.size(test_acc)):",
        "    row, column = test_seed_coords[index]",
        "    output[row, column] = test_acc[index]",
        "cute.arch.sync_threads()",
    ]
    return "\n".join(
        [
            "from __future__ import annotations",
            "import cutlass",
            "import cutlass.cute as cute",
            "@cute.kernel",
            "def k24_component(a, b, seed, output):",
            *("    " + line for line in body),
            "",
        ]
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_k24_runtime_component_uses_exact_emitter_cpu(dtype):
    from torch._inductor.codecache import PyCodeCache

    source = _k24_component_source(dtype)
    assert "(16, 8, 8)" in source
    assert "cute.size(test_sa, mode=[2])" in source
    assert source.count("sync_threads()") == 2
    assert "test_acc[test_seed_index] = cutlass.Float32(seed[" in source
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        assert callable(PyCodeCache.load(source).k24_component)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_gpu_exact_emitted_k24_three_panels_with_fp32_seed(dtype):
    from torch._inductor.codecache import PyCodeCache

    from helion.runtime import default_cute_launcher

    torch.manual_seed(824)
    a_raw = torch.full((18, 28), -113, device="cuda", dtype=dtype)
    b_raw = torch.full((10, 28), -113, device="cuda", dtype=dtype)
    a = a_raw[1:17, 2:26]
    b = b_raw[1:9, 2:26].T
    a.copy_(torch.randn_like(a) * 0.125)
    b.copy_(torch.randn_like(b) * 0.125)
    seed = torch.randn((16, 8), dtype=torch.float32, device="cuda") * 0.125
    before = [tensor.clone() for tensor in (a_raw, b_raw, seed)]
    expected = (a.double() @ b.double() + seed.double()).float()
    storage = torch.empty(256, device="cuda", dtype=torch.float32)
    output = storage[64:192].view(16, 8)
    module = PyCodeCache.load(_k24_component_source(dtype))
    for poison in (37.0, -41.0):
        storage.fill_(poison)
        default_cute_launcher(
            module.k24_component, (1,), a, b, seed, output, block=(32, 1, 1)
        )
        torch.testing.assert_close(output, expected, atol=0.02, rtol=0.01)
        assert torch.all(storage[:64] == poison)
        assert torch.all(storage[192:] == poison)
        for tensor, saved in zip((a_raw, b_raw, seed), before, strict=True):
            assert torch.equal(tensor.view(torch.uint8), saved.view(torch.uint8))
