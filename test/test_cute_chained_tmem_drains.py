from __future__ import annotations

import ast
from dataclasses import replace
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_carry_layout import _carry
from .test_cute_chained_loop_tmem_carry_transport import _resident_carry_sequence
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_pipeline import _config
from .test_cute_chained_tmem_snapshot_mapping import _code
from helion._compiler.cute import chained_loop_tmem_carry_transport as pipeline
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_tcgen05 import _load_result
from helion._compiler.cute.chained_tmem_drains import FP32DrainPanels
from helion._compiler.cute.chained_tmem_drains import FP32SharedTarget
from helion._compiler.cute.chained_tmem_drains import ScalarPublication
from helion._compiler.cute.chained_tmem_drains import emit_streamed_fp32_drain


@pytest.mark.parametrize("width", [64, 96, 128, 160, 192, 224, 256])
def test_panel_shape_and_bounded_source(width):
    panels = FP32DrainPanels((128, width))
    publication = ScalarPublication(
        "index", "row", "column", (FP32SharedTarget("output", (128, width)),)
    )
    lines = emit_streamed_fp32_drain(
        panels, "drain", "original", publication, execution=ChainedExecution(128)
    )
    source = "\n".join(lines)
    assert panels.count == width // 32
    assert source.count("cute.make_rmem_tensor") == 1
    assert f"cutlass.range({width // 32}, unroll=1)" in source
    assert "Repetition(32)" in source
    assert source.count("fence_view_async_tmem_load") == 1
    assert not any(
        word in source
        for word in (
            "Float16",
            "alloc_smem",
            "align(",
            "sync_threads",
            "arrive_and_wait",
            "tmem_store",
        )
    )
    ast.parse(source)


@pytest.mark.parametrize(
    "shape",
    [
        (64, 128),
        (128, 16),
        (128, 32),
        (128, 48),
        (128, 80),
        (128, 144),
        (128, 288),
        (128,),
        (128, True),
        (128, 64.0),
        [128, 64],
    ],
)
def test_unsupported_or_partial_panels_reject(shape):
    with pytest.raises(ValueError):
        FP32DrainPanels(shape)


@pytest.mark.parametrize(
    "width,offset", [(16, 16), (32, 32), (48, 16), (64, 32), (128, 32), (192, 64)]
)
def test_default_carry_drain_is_exact_original_bytes(width, offset):
    carry = _carry(width, offset)
    prefix = carry.prefix
    expected = [
        *_load_result(prefix, carry.candidate.geometry.physical[:2]),
        f"for {prefix}_index in cutlass.range_constexpr(cute.size({prefix}_values)):",
        f"    {prefix}_row, {prefix}_column = {prefix}_coords[{prefix}_index]",
        f"    chain_loop_carry_{carry.candidate.carry_index}[{prefix}_row, {prefix}_column] = {prefix}_values[{prefix}_index]",
        "chain_tmem_barrier.arrive_and_wait()",
    ]
    assert carry.shared_transfer(upload=False) == [
        "if chain_thread < 128:",
        "\n".join("    " + line for line in expected),
        "cute.arch.sync_threads()",
    ]


@pytest.mark.parametrize(
    "change", ["panels", "shape", "arena", "member", "columns", "transpose"]
)
def test_carry_drain_retains_original_selection(change):
    carry = _carry(128, 32)
    carry = replace(carry, drain_panels=FP32DrainPanels((128, 128)))
    carry = replace(carry, _drain_facts=carry._current_drain_facts())
    assert "_drain_panel" in "\n".join(carry.shared_transfer(upload=False))
    if change == "panels":
        object.__setattr__(carry, "drain_panels", None)
    elif change == "shape":
        object.__setattr__(carry.drain_panels, "shape", (128, 64))
    elif change == "member":
        object.__setattr__(carry.candidate, "member_offset", 64)
    elif change == "transpose":
        object.__setattr__(carry.candidate.geometry, "transpose", True)
    else:
        name = "arena_offset" if change == "arena" else "required_columns"
        object.__setattr__(carry, name, getattr(carry, name) + 32)
    with pytest.raises(ValueError, match="selection changed"):
        carry.shared_transfer(upload=False)


@pytest.mark.parametrize("threads", [32, 64, 96, 160, 256])
def test_participant_contract_rejects(threads):
    with pytest.raises(ValueError, match="128 local"):
        emit_streamed_fp32_drain(
            FP32DrainPanels((128, 128)),
            "d",
            "s",
            ScalarPublication("i", "r", "c", ()),
            execution=ChainedExecution(threads),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_real_carry_adapter_activates_only_terminal_drain(dtype):
    original = pipeline.prepare_loop_tmem_carry
    records = []

    def selected(*args, **kwargs):
        assert kwargs["drain_tile_columns"] == 0
        result = original(*args, **(kwargs | {"drain_tile_columns": 32}))
        assert result is not None
        records.append(result)
        return result

    args = (
        *(
            torch.empty(shape, dtype=dtype)
            for shape in ((3, 128, 16), (3, 16, 64), (3, 64, 16), (3, 16, 64))
        ),
        torch.empty((128, 64), dtype=torch.float32),
    )
    config = _config(16, pipeline=True, consumer_warps=8)
    config.config["cute_chained_warp_mma_rows"] = 64
    control = _source(_resident_carry_sequence, args, config)
    with patch.object(pipeline, "prepare_loop_tmem_carry", selected):
        actual = _source(_resident_carry_sequence, args, config)
    (carry,) = records
    old = replace(carry, drain_panels=None, _drain_facts=None)
    before = "\n".join(old.shared_transfer(upload=False))
    after = "\n".join(carry.shared_transfer(upload=False))

    # The generated module's indentation is supplied by its ordinary wrapper.
    def normalized(text):
        return "\n".join(line.strip() for line in text.splitlines())

    assert normalized(actual).count(normalized(after)) == 1
    assert normalized(actual).replace(
        normalized(after), normalized(before)
    ) == normalized(control)
    assert carry.shared_transfer(upload=True) == old.shared_transfer(upload=True)
    assert carry.snapshot(ChainedExecution(256)) == old.snapshot(ChainedExecution(256))


@pytest.mark.parametrize("width", [64, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
def test_actual_dynamic_native_panel_loads_and_shared_publication(width, dtype_name):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    func = importlib.import_module("cutlass._mlir.dialects.func")
    scf = importlib.import_module("cutlass._mlir.dialects.scf")
    pm = importlib.import_module("cutlass._mlir.passmanager")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    publication = ScalarPublication(
        "index", "row", "column", (FP32SharedTarget("output", (128, width)),)
    )
    tree = ast.parse(
        "\n".join(
            emit_streamed_fp32_drain(
                FP32DrainPanels((128, width)),
                "drain",
                "original",
                publication,
                execution=ChainedExecution(128),
            )
        )
    )
    loop = tree.body[-1]
    assert isinstance(loop, ast.For)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("drain", ([ir.IntegerType.get_signless(32)] * 3, []))
        with ir.InsertionPoint(function.add_entry_block()):
            thread, origin, shared = map(cutlass.Int32, function.arguments)
            mma = blackwell_helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, width),
                tcgen05.OperandSource.SMEM,
            )
            layout = mma.make_fragment_C(mma.partition_shape_C((128, width))).layout
            ns: dict[str, Any] = {
                "cutlass": cutlass,
                "cute": cute,
                "tcgen05": tcgen05,
                "chain_thread": thread,
                "original_acc": cute.make_tensor(
                    cute.make_ptr(cutlass.Float32, origin, cute.AddressSpace.tmem),
                    layout,
                ),
                "original_slice": mma.get_slice(0),
                "output": cute.make_tensor(
                    cute.make_ptr(cutlass.Float32, shared, cute.AddressSpace.smem),
                    cute.make_layout((128, width), stride=(width, 1)),
                ),
            }
            exec(_code(tree.body[:-1]), ns)
            assert int(cute.size(ns["drain_values"])) == 32
            native_loop = scf.ForOp(
                *(cutlass.Int32(v).ir_value() for v in (0, width // 32, 1))
            )
            with ir.InsertionPoint(native_loop.body):
                ns["drain_panel"] = cutlass.Int32(native_loop.induction_variable)
                exec(_code(loop.body), ns)
                scf.YieldOp([])
            func.ReturnOp([])
        assert module.operation.verify()
        pm.PassManager.parse(
            "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,convert-cute-to-core,canonicalize)"
        ).run(module.operation)
        assert module.operation.verify()
        text = str(module)
        has_load_and_wait = (
            "nvvm.tcgen05.ld" in text and "nvvm.tcgen05.wait <load>" in text
        )
        assert has_load_and_wait
        assert "cvt." not in text and "scf.for" in text
