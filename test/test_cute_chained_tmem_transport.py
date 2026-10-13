from __future__ import annotations

import ast
import hashlib
import importlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_m64 import _m64_config
from .test_cute_chained_m64 import _m64_single
from .test_cute_chained_tcgen05 import _tcgen_code
from .test_cute_chained_tcgen05 import _tcgen_inputs
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_tmem_transport import emit_packed_tmem_fragment
from helion._compiler.cute.chained_tmem_transport import emit_tmem_operand_view

# Raw bridge-list SHA256 captured from real root codegen before extraction.
# Whole-kernel before/after hashes are retained in the task artifact, not pinned
# here: unrelated allocation/planning changes must not invalidate this contract.
_BRIDGE_DIGESTS = {
    (
        torch.bfloat16,
        32,
    ): "3080afd7978aa908b9b4220ec45ad5af158c229b2aa8b897d34e47bb2d594f63",
    (
        torch.bfloat16,
        64,
    ): "f3c2359ec687527efa7d11cfaaa98cad336964a51d9e43d03edbc87aac8f3677",
    (
        torch.bfloat16,
        128,
    ): "27609648eee18ee1e03987e1001c09f2fa56bb549ccd94f6f0de054a064924d4",
    (
        torch.bfloat16,
        256,
    ): "4d6d782a6b6a89863e2b48c67113c325cc973a5cccea372454a4591f0ffed129",
    (
        torch.float16,
        32,
    ): "747ecddea102ee030b19ef2fffee3d4727e6c146ffff7e8f356e87a112233a66",
    (
        torch.float16,
        64,
    ): "e01b452eaef1f7c93cdd30967b8109466ab9cdb46346b8df672e36db40a75b66",
    (
        torch.float16,
        128,
    ): "15c9207ec5a1527dcdea0914ed4f2e9a784a09e90f04f3fd4b8c2f502fd0a690",
    (
        torch.float16,
        256,
    ): "e308870839203e5741eac66612b1953cd8d5b9e9f81ad11ec8717a10eeb93d9d",
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [32, 64, 128, 256])
def test_real_root_bridge_preserves_raw_emission(dtype, width):
    records = []
    original = chained_tcgen05._bridge

    def record(*args, **kwargs):
        lines = original(*args, **kwargs)
        records.append(lines)
        return lines

    before = torch.cuda.is_initialized()
    with patch.object(chained_tcgen05, "_bridge", record):
        source = _tcgen_code(
            _tcgen_inputs("cpu", dtype, q=width, k=16, n=32), "plain", 32
        )
    assert (
        hashlib.sha256(json.dumps(records).encode()).hexdigest()
        == (_BRIDGE_DIGESTS[dtype, width])
    )
    dtype_name = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    assignments = {
        ast.dump(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
    }
    for line in emit_tmem_operand_view("chain_1", (128, 32, width), dtype_name):
        assert ast.dump(ast.parse(line).body[0]) in assignments
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_m64_remains_independent_and_does_not_select_packed_transport(dtype):
    # M64 currently admits independent dots, not a packed bridge. This extraction
    # must not broaden the root matcher or silently claim sparse-layout support.
    with (
        _cpu_codegen(),
        patch.object(chained_tcgen05, "emit_packed_tmem_fragment") as packed,
        patch.object(chained_tcgen05, "emit_tmem_operand_view") as view,
    ):
        source = _m64_single._bind_isolated(
            (torch.empty((64, 128), dtype=dtype), torch.empty((128, 64), dtype=dtype))
        ).to_code(_m64_config())
    packed.assert_not_called()
    view.assert_not_called()
    assert "tcgen05.Ld16x256bOp" in source
    assert "OperandSource.TMEM" not in source


def _emit(shape=(128, 32), dtype="cutlass.BFloat16", **kwargs):
    return emit_packed_tmem_fragment(
        "packed",
        "source",
        shape,
        dtype,
        ("row", "column"),
        ["original = source_values[packed_index]", "scaled = original * coefficient"],
        f"{dtype}(scaled) if column < valid_columns else {dtype}(0)",
        **kwargs,
    )


@pytest.mark.parametrize(
    "columns,repetition", [(16, 8), (32, 16), (48, 8), (64, 32), (96, 16), (128, 32)]
)
def test_packing_repetition_and_original_masked_expression(columns, repetition):
    lines = _emit((128, columns))
    source = "\n".join(lines)
    assert f"tcgen05.Repetition({repetition})" in source
    assert "source_values[packed_index]" in source
    assert "scaled = original * coefficient" in source
    assert "if column < valid_columns else cutlass.BFloat16(0)" in source
    ast.parse(source)


def test_role_local_transport_uses_explicit_destination_and_full_participant_barriers():
    execution = ChainedExecution(
        128,
        thread="consumer_thread",
        warp="consumer_warp",
        sync="consumer_barrier.arrive_and_wait()",
        tmem="unused_context_pointer",
    )
    lines = _emit(execution=execution, destination="resident_tmem + 64")
    source = "\n".join(lines)
    assert "get_slice(consumer_thread)" in source
    assert "cute.make_tensor(resident_tmem + 64, packed_layout)" in source
    assert source.count(execution.sync) == 2
    assert "chain_thread" not in source and "sync_threads" not in source
    assert "unused_context_pointer" not in source
    body = ast.parse(source).body
    assert isinstance(body[-5], ast.For)
    assert all(isinstance(node, ast.Expr) for node in body[-4:])
    assert ast.unparse(body[-4]) == execution.sync
    assert ast.unparse(body[-1]) == execution.sync


@pytest.mark.parametrize("threads", [32, 64, 256, 512])
def test_wider_or_partial_tmem_team_cannot_hide_role_barriers(threads):
    with pytest.raises(ValueError, match="128-thread"):
        _emit(execution=ChainedExecution(threads))


@pytest.mark.parametrize("shape", [(128, 0), (0, 32), (128, 31), (128, -32)])
def test_invalid_pair_packing_contract_rejects(shape):
    with pytest.raises(ValueError, match="positive even columns"):
        _emit(shape)


@pytest.mark.parametrize("dtype", ["cutlass.Float32", "cutlass.Int16"])
def test_non_floating_or_non_16bit_transport_rejects(dtype):
    with pytest.raises(ValueError, match="16-bit floating"):
        _emit(dtype=dtype)
    with pytest.raises(ValueError, match="16-bit floating"):
        emit_tmem_operand_view("stage", (128, 32, 32), dtype)


@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
@pytest.mark.parametrize("base_address", [0, 64, 320, 0x20040])
def test_emitted_operand_rebase_uses_typed_units_and_nonzero_base(
    dtype_name, base_address
):
    # Execute the emitted pointer expression: the factory result starts at zero,
    # and the external base is expressed in Float32, not BF16/FP16, units.
    namespace = {
        "chain_sm100": SimpleNamespace(
            make_smem_layout_a=lambda *args: SimpleNamespace(outer="operand-layout")
        ),
        "stage_mma": SimpleNamespace(
            make_fragment_A=lambda layout: SimpleNamespace(iterator=0, layout=layout)
        ),
        "cute": SimpleNamespace(
            make_tensor=lambda iterator, layout: SimpleNamespace(
                iterator=iterator, layout=layout
            )
        ),
        "cutlass": SimpleNamespace(
            Float32=SimpleNamespace(width=32),
            BFloat16=SimpleNamespace(width=16),
            Float16=SimpleNamespace(width=16),
        ),
        "resident_tmem": SimpleNamespace(toint=lambda: base_address),
    }
    lines = emit_tmem_operand_view(
        "stage", (128, 32, 64), f"cutlass.{dtype_name}", base="(resident_tmem)"
    )
    exec("\n".join(lines), namespace)
    assert namespace["stage_ra"].iterator == 2 * base_address
    assert namespace["stage_ra"].layout == "operand-layout"
    assert "chain_tptr" not in "\n".join(lines)


@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
@pytest.mark.parametrize("columns", [16, 32, 48, 64, 96, 128])
def test_actual_cute_packing_fragments_match_typed_operand_coordinates_cpu(
    dtype_name, columns
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    lines = _emit((128, columns), f"cutlass.{dtype_name}")
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            producer = blackwell_helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, columns),
                tcgen05.OperandSource.SMEM,
            )
            layout = producer.make_fragment_C(
                producer.partition_shape_C((128, columns))
            ).layout
            source = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem), layout
            )
            identity = producer.get_slice(0).partition_C(
                cute.make_identity_tensor((128, columns))
            )
            load = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(min(32, columns & -columns))),
                    cutlass.Float32,
                ),
                source,
            )
            coverage = set()
            for thread in range(128):
                coordinates = load.get_slice(thread).partition_D(identity)
                values = cute.make_rmem_tensor(coordinates.shape, cutlass.Float32)
                namespace: dict[str, Any] = {
                    "cute": cute,
                    "cutlass": cutlass,
                    "tcgen05": tcgen05,
                    "source_acc": source,
                    "source_identity": identity,
                    "source_coords": coordinates,
                    "source_values": values,
                    "chain_thread": thread,
                    "chain_tptr": cute.make_ptr(
                        cutlass.Float32, 256, cute.AddressSpace.tmem
                    ),
                }
                # Execute actual emitted setup and copy against CuTe MLIR;
                # source arithmetic is separately preserved by raw-code hashes.
                exec("\n".join(lines[:10]), namespace)
                packed_coords = namespace["packed_coords"]
                assert int(cute.size(coordinates)) == 2 * int(cute.size(packed_coords))
                for index in range(int(cute.size(coordinates))):
                    row, col = map(int, coordinates[index])
                    register = int(values.layout(index))
                    packed_row, packed_col = map(int, packed_coords[register // 2])
                    assert (row, col) == (packed_row, 2 * packed_col + register % 2)
                    coverage.add((row, col))
                exec(lines[-3], namespace)
            assert coverage == {(r, c) for r in range(128) for c in range(columns)}

            consumer = blackwell_helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, 32),
                tcgen05.OperandSource.TMEM,
            )
            weight_layout = blackwell_helpers.make_smem_layout_a(
                consumer, (128, 32, columns), dtype, 1
            )
            operand = consumer.make_fragment_A(weight_layout.outer)
            operand_coords = consumer.get_slice(0).partition_A(
                cute.make_identity_tensor((128, columns))
            )
            for index in range(128 * columns):
                row, column = map(int, operand_coords[index])
                packed_address = int(layout(((row, column // 2), 0, 0)))
                assert int(operand.layout(index)) == 2 * packed_address + column % 2
        assert module.operation.verify()
    assert torch.cuda.is_initialized() == before
