from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_snapshot_compile import _module
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_tmem_drains import FP32DrainPanels
from helion._compiler.cute.chained_tmem_drains import FP32SharedTarget
from helion._compiler.cute.chained_tmem_drains import PartitionedPublication
from helion._compiler.cute.chained_tmem_drains import ScalarPublication
from helion._compiler.cute.chained_tmem_drains import emit_streamed_fp32_drain


@pytest.mark.parametrize(
    "dtype,width,partitioned", [("BFloat16", 96, False), ("Float16", 128, True)]
)
def test_real_cute_ast_static_setup_dynamic_panel_containers(
    dtype, width, partitioned, tmp_path
):
    """Compile the actual shared renderer through the DSL AST, never launch.

    This is a small transport/container regression, not a full-kernel native or
    runtime qualification. Ordered address/payload coverage is separate.
    """
    from cuda.bindings.driver import CUstream
    import cutlass
    import cutlass.cute as cute

    publication = (
        PartitionedPublication("shared")
        if partitioned
        else ScalarPublication(
            "i", "r", "c", (FP32SharedTarget("shared", (128, width)),)
        )
    )
    lines = emit_streamed_fp32_drain(
        FP32DrainPanels((128, width)),
        "drain",
        "original",
        publication,
        execution=ChainedExecution(128),
    )
    source = (
        f"""from cuda.bindings.driver import CUstream
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils import blackwell_helpers

@cute.kernel
def device(output: cute.Tensor, origin: cutlass.Int32):
    chain_thread = cutlass.Int32(cute.arch.thread_idx()[0])
    mma = blackwell_helpers.make_trivial_tiled_mma(cutlass.{dtype}, cutlass.{dtype}, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, (128, {width}), tcgen05.OperandSource.SMEM)
    layout = mma.make_fragment_C(mma.partition_shape_C((128, {width}))).layout
    original_acc = cute.make_tensor(cute.make_ptr(cutlass.Float32, origin, cute.AddressSpace.tmem), layout)
    original_slice = mma.get_slice(0)
    shared = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {128 * width}, alignment=128), cute.make_layout((128, {width}), stride=({width}, 1)))
"""
        + "\n".join("    " + line for line in lines)
        + """
    cute.arch.sync_threads()
    output[chain_thread] = shared[chain_thread, 0]

@cute.jit
def host(output: cute.Pointer, origin: cutlass.Int32, stream: CUstream):
    tensor = cute.make_tensor(output, cute.make_layout(128))
    device(tensor, origin).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)
"""
    )
    module = _module(f"_fp32_drain_{dtype}_{width}_{partitioned}", source)
    initial = torch.cuda.is_initialized()
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        compiled = cute.compile(
            module.host,
            cute.runtime.make_ptr(
                cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4
            ),
            cutlass.Int32(64),
            CUstream(0),
            options=f"--dump-dir {tmp_path} --keep-ptx --keep-cubin --gpu-arch sm_103a",
        )
    assert "tcgen05.ld" in compiled.__ptx__
    assert "tcgen05.wait::ld" in compiled.__ptx__
    assert torch.cuda.is_initialized() == initial
