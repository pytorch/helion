from __future__ import annotations

from collections import Counter
from dataclasses import replace
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_load_only_preparation import _args as _m128_args
from .test_cute_chained_load_only_preparation import _load_only_recurrence
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_loop_tmem_carry_transport as carry_transport
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute.chained_loop_tmem_carry import plan_loop_tmem_carry
from helion._compiler.cute.chained_result_transport import M64_GROUP_WIDTHS
from helion._compiler.cute.chained_result_transport import M64_WIDTHS
from helion._compiler.cute.chained_result_transport import load_operation
from helion._compiler.cute.chained_result_transport import store_operation
from helion._compiler.cute.chained_seed_tiles import plan_seed_panels
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.chained_tcgen_stage import stage_geometry
from helion._compiler.cute.chained_tmem_transport import emit_packed_tmem_fragment


def _args(
    device: str = "cpu", *, dtype: torch.dtype = torch.bfloat16, steps: int = 3
) -> tuple[torch.Tensor, ...]:
    batch, rows, depth, columns = 2, 64, 128, 64
    shapes = (
        (batch, steps, rows, depth),
        (batch, steps, rows, depth),
        (batch, steps, rows, columns),
        (batch, steps, rows, rows),
        (batch, steps, depth, rows),
    )
    return (
        *(torch.empty(shape, device=device, dtype=dtype) for shape in shapes),
        torch.empty((batch, steps, rows), device=device),
        torch.empty((batch, depth, columns), device=device),
        torch.full((batch, steps + 1, rows, columns), 19.0, device=device, dtype=dtype),
        torch.empty((batch, depth, columns), device=device),
    )


def _config(**options: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "num_warps": 8,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_group_contractions": True,
            "cute_chained_preparation_pipeline": True,
            "cute_chained_loop_mma_rows": 64,
        }
        | options
    )


def _source(
    config: helion.Config | None = None,
    args: tuple[torch.Tensor, ...] | None = None,
) -> str:
    if args is None:
        args = _args()
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_load_only_recurrence, args)
        ):
            return bound.to_code(config or _config())


@pytest.mark.parametrize("width", M64_GROUP_WIDTHS)
def test_full_m64_orientation_and_default_geometry(width: int) -> None:
    direct = stage_geometry((64, width, 64), native_rows=64)
    transposed = StageGeometry((width, 64, 64), True, native_rows=64)
    assert direct is not None and direct.physical == transposed.physical == (
        64,
        width,
        64,
    )
    assert transposed.result_coordinates("i", "j") == ("j", "i")
    assert transposed.operand("a", "i", "j") == (1, ("j", "i"))
    assert stage_geometry((64, 128, 64)) == StageGeometry((64, 128, 64), True)
    assert M64_WIDTHS == (32, 64, 96, 128, 256)


@pytest.mark.parametrize(
    "shape", [(32, 128, 64), (64, 16, 64), (64, 48, 64), (64, 128, 15), (128, 128, 64)]
)
def test_m64_does_not_admit_padding_or_unknown_geometry(
    shape: tuple[int, int, int],
) -> None:
    assert stage_geometry(shape, native_rows=64) is None


@pytest.mark.parametrize("dtype_name", ["Float16", "BFloat16"])
@pytest.mark.parametrize("width", M64_GROUP_WIDTHS)
def test_actual_sparse_fp32_and_snapshot_order(dtype_name: str, width: int) -> None:
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    tcgen = importlib.import_module("cutlass.cute.nvgpu.tcgen05")
    helpers = importlib.import_module("cutlass.utils.blackwell_helpers")
    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = vars(cutlass)[dtype_name]
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen.CtaGroup.ONE,
                (64, width),
                tcgen.OperandSource.TMEM,
            )
            acc = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 32, cute.AddressSpace.tmem),
                mma.make_fragment_C(mma.partition_shape_C((64, width))).layout,
            )
            identity = mma.get_slice(0).partition_C(
                cute.make_identity_tensor((64, width))
            )
            namespace: dict[str, Any] = {
                "cutlass": cutlass,
                "cute": cute,
                "tcgen05": tcgen,
                "source_acc": acc,
                "source_identity": identity,
            }
            load = tcgen.make_tmem_copy(
                cute.make_copy_atom(
                    eval(load_operation((64, width)), namespace), cutlass.Float32
                ),
                acc,
            )
            store = tcgen.make_tmem_copy(
                cute.make_copy_atom(
                    eval(store_operation((64, width)), namespace), cutlass.Float32
                ),
                acc,
            )
            lines = emit_packed_tmem_fragment(
                "snapshot",
                "source",
                (64, width),
                f"cutlass.{dtype_name}",
                ("row", "column"),
                (),
                "0.0",
                destination="source_acc.iterator",
            )
            exec("\n".join(lines[:5]), namespace)
            packed, snapshot = namespace["snapshot_target"], namespace["snapshot_copy"]
            seen = []
            for thread in range(128):
                read = load.get_slice(thread).partition_D(identity)
                written = store.get_slice(thread).partition_S(identity)
                packed_coords = snapshot.get_slice(thread).partition_S(
                    namespace["snapshot_coords_tensor"]
                )
                source = [tuple(read[i]) for i in range(cute.size(read))]
                assert source == [tuple(written[i]) for i in range(cute.size(written))]
                assert source == [
                    (row, 2 * col + part)
                    for row, col in (
                        tuple(packed_coords[i]) for i in range(cute.size(packed_coords))
                    )
                    for part in (0, 1)
                ]
                seen.extend(source)
            assert Counter(seen) == Counter(
                (row, col) for row in range(64) for col in range(width)
            )
            operand = mma.make_fragment_A(
                helpers.make_smem_layout_a(mma, (64, width, width), dtype, 1).outer
            )
            for column in range(width):
                for row in range(64):
                    assert (
                        operand.layout(row + 64 * column)
                        == 2 * packed.layout(row + 64 * (column // 2)) + column % 2
                    )
        assert module.operation.verify()


@pytest.mark.parametrize(
    "total,offset,width", [(64, 0, 64), (128, 32, 96), (192, 64, 128), (256, 128, 128)]
)
def test_actual_grouped_seed_partitions_preserve_member_origin(
    total: int, offset: int, width: int
) -> None:
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    tcgen = importlib.import_module("cutlass.cute.nvgpu.tcgen05")
    helpers = importlib.import_module("cutlass.utils.blackwell_helpers")
    ir = importlib.import_module("cutlass._mlir.ir")
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = helpers.make_trivial_tiled_mma(
                cutlass.BFloat16,
                cutlass.BFloat16,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen.CtaGroup.ONE,
                (64, total),
                tcgen.OperandSource.SMEM,
            )
            acc = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 96, cute.AddressSpace.tmem),
                mma.make_fragment_C(mma.partition_shape_C((64, total))).layout,
            )
            panel = plan_seed_panels((64, total), offset, width)[0]
            namespace = {
                "cute": cute,
                "cutlass": cutlass,
                "tcgen05": tcgen,
                "group_acc": acc,
                "group_slice": mma.get_slice(0),
            }
            # Construct the production subview/copy once; only participant selection varies.
            exec("\n".join(panel.store_views("seed", "group", "0")[:3]), namespace)
            store, identity = namespace["seed_copy"], namespace["seed_identity"]
            load = tcgen.make_tmem_copy(
                cute.make_copy_atom(
                    eval(load_operation((64, width)), namespace), cutlass.Float32
                ),
                namespace["seed_segment"],
            )
            owners = []
            for thread in range(128):
                written = store.get_slice(thread).partition_S(identity)
                read = load.get_slice(thread).partition_D(identity)
                coordinates = [tuple(written[i]) for i in range(cute.size(written))]
                assert coordinates == [tuple(read[i]) for i in range(cute.size(read))]
                owners.extend(coordinates)
            assert Counter(owners) == Counter(
                (row, column)
                for row in range(64)
                for column in range(offset, offset + width)
            )
        assert module.operation.verify()


def test_grouped_loop_selects_exact_m64_carry_and_bounded_storage() -> None:
    carries, allocations = [], []
    original_carry = carry_transport.prepare_loop_tmem_carry
    original_storage = chained_pipeline_storage.finalize_pipeline_storage

    def carry(*args: Any, **kwargs: Any) -> Any:
        result = original_carry(*args, **kwargs)
        carries.append((args[1], args[3], result))
        return result

    def storage(*args: Any, **kwargs: Any) -> Any:
        result = original_storage(*args, **kwargs)
        allocations.append(result)
        return result

    with (
        patch.object(carry_transport, "prepare_loop_tmem_carry", carry),
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", storage),
    ):
        source = _source()
    assert len(carries) == len(allocations) == 1
    plan, groups, retained = carries[0]
    assert retained is not None
    assert [group.physical for group in groups] == [(64, 128, 128), (64, 192, 64)]
    assert retained.candidate.geometry.transpose
    assert retained.candidate.snapshot_shape == (64, 128)
    assert retained.candidate.member_offset == 64
    assert allocations[0] is not None and allocations[0].charged_bytes <= 232448
    assert "tcgen05.St16x128bOp" in source
    assert "tcgen05.St16x256bOp" in source
    assert "chain_resident_carry_" in source
    bad = replace(
        groups[-1],
        geometries=(groups[-1].geometries[0], StageGeometry((128, 64, 64), False)),
    )
    assert plan_loop_tmem_carry(plan, (*groups[:-1], bad)) is None


@pytest.mark.parametrize("failure", ["missing", "capacity"])
def test_m64_final_storage_veto_cannot_fall_back(failure: str) -> None:
    original = chained_pipeline_storage.finalize_pipeline_storage

    def reject(*args: Any, **kwargs: Any) -> Any:
        if failure == "missing":
            return None
        kwargs["capacity_bytes"] = 1
        return original(*args, **kwargs)

    with (
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", reject),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _source()


def test_m64_discovery_rejects_a_loop_without_full_row_geometry() -> None:
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(_m128_args())
        assert not bound.config_spec.cute_chained_loop_m64_search_enabled
        with pytest.raises(exc.InvalidConfig, match="eligible grouped"):
            bound.config_spec.normalized_config(dict(_config()))


def test_m64_ordinary_grouped_loop_uses_sparse_transport() -> None:
    source = _source(_config(num_warps=4, cute_chained_preparation_pipeline=False))
    assert "tcgen05.St16x256bOp" in source
    assert "tcgen05.Ld16x256bOp" in source
    assert "tcgen05.St32x32bOp" not in source
    assert "chain_resident_carry_" not in source


@pytest.mark.parametrize(
    "options",
    [
        {"cute_chained_group_contractions": False},
        {"cute_chained_warp_mma_rows": 64},
        {"cute_chained_snapshot_tile_columns": 32},
        {"cute_chained_drain_tile_columns": 32},
        {
            "cute_chained_completed_member_store": True,
            "cute_chained_preparation_pipeline": False,
        },
        {"cute_chained_loop_mma_rows": 32},
    ],
)
def test_incompatible_or_unknown_row_options_reject(options: dict[str, object]) -> None:
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(_args())
        with pytest.raises(exc.InvalidConfig):
            bound.config_spec.normalized_config(dict(_config(**options)))


def test_m64_completed_store_is_supported_with_a_prepared_pipeline() -> None:
    source = _source(_config(cute_chained_completed_member_store=True))
    assert "chain_store_0_register" in source
    assert "chain_2_c[" not in source


def test_m64_seeds_are_bounded_additions_and_default_source_is_unchanged() -> None:
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(_args())
        assert bound.config_spec.cute_chained_loop_m64_search_enabled
        parents = [
            bound.config_spec.normalized_config(
                {
                    "num_warps": 4,
                    "cute_chained_mma_schedule": "tcgen05_tmem",
                    "cute_chained_group_contractions": True,
                }
            )
        ]
        assert bound.host_function is not None
        completed = [
            helion.Config.from_dict(
                parents[0].config | {"cute_chained_completed_member_store": True}
            )
        ]
        assert (
            CuteChainedMatmulHeuristic._with_loop_m64_seeds(
                bound.env, bound.host_function.device_ir, completed
            )
            == completed
        )
        seeds = CuteChainedMatmulHeuristic._with_loop_m64_seeds(
            bound.env, bound.host_function.device_ir, parents
        )
        assert seeds[:1] == parents and len(seeds) == 3
        assert all(
            seed.config["cute_chained_loop_mma_rows"] == 64 for seed in seeds[1:]
        )
        for seed in seeds[1:]:
            bound.config_spec.normalized_config(dict(seed))
        default = _config(cute_chained_loop_mma_rows=128)
        normalized = bound.config_spec.normalized_config(dict(default))
        assert "cute_chained_loop_mma_rows" not in normalized
    args = _m128_args()
    assert _source(default, args) == _source(normalized, args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("prepared", [False, True])
def test_m64_grouped_carry_output_state_and_guards_gpu(
    dtype: torch.dtype, steps: int, prepared: bool
) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = _args("cuda", dtype=dtype, steps=steps)
    torch.manual_seed(7223)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(torch.randint(-2, 3, tensor.shape, device="cuda") * 0.015625)
    args[5].copy_(torch.randint(1, 4, args[5].shape, device="cuda") * 0.25)
    left, right, value, transform, update, gate, initial, output, final = args
    original = [tensor.clone() for tensor in args[:7]]
    expected, state = output.clone(), initial.double()
    for step in range(steps):
        entering = state.to(dtype).double()
        corrected = (
            (value[:, step].double() - right[:, step].double() @ entering)
            .to(dtype)
            .double()
        )
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ corrected + cross).to(dtype)
        state = (
            update[:, step].double() @ corrected
            + state * gate[:, step, -1, None, None].double()
        )
    config = _config(
        num_warps=8 if prepared else 4, cute_chained_preparation_pipeline=prepared
    )
    actual = _load_only_recurrence._bind_isolated(args).compile_config(config)(*args)
    torch.testing.assert_close(actual[0], expected, atol=0.002, rtol=0.01)
    torch.testing.assert_close(actual[1], state.float(), atol=0.002, rtol=0.01)
    for before, after in zip(original, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert torch.equal(final, actual[1])
