"""DeepSeek-V3 B1 NVFP4 attention output boundary, pretuned for TP4 B200.

The Helion megakernel fuses a rank-local row-parallel output projection,
one-shot all-reduce, residual addition, and RMSNorm. Readiness travels with
each BF16 payload word, eliminating a separate rank barrier.

Calls and graph replays must occur in identical collective order across ranks
on one ordered CUDA stream. Concurrent invocations are unsupported; separate
workspace/state pairs alone do not establish cross-rank launch ordering.

Run the benchmark with four local ranks::

    NVSHMEM_DISABLE_CUDA_VMM=1 \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      pretuned_kernels/megakernels/deepseek_v3_attention_nvfp4_tp/deepseek_v3_attention_nvfp4_tp.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    COMMUNICATION_BLOCK,
)
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    K_GROUP_BLOCK,
)
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    OUTPUT_BLOCK,
)
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    OUTPUT_FEATURES,
)
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    RING_SLOTS,
)
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import RMS_EPS
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    SIGNAL_PAD_BYTES,
)
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    WORLD_SIZE,
)
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    attention_boundary_source,
)
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    quantize_inputs,
)
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


NUM_SM_MULTIPLIER = 4
MAXNREG = 128


deepseek_v3_attention_nvfp4_tp = helion.kernel(
    attention_boundary_source,
    config=helion.Config(
        block_sizes=[OUTPUT_BLOCK, K_GROUP_BLOCK],
        cross_loop_pipeline="dynamic",
        maxnreg=MAXNREG,
        num_sm_multiplier=NUM_SM_MULTIPLIER,
        num_stages=2,
        num_warps=2,
        pid_type="persistent_blocked",
        range_multi_buffers=[None, True, None, None, None, None],
    ),
    static_shapes=False,
    backend="triton",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


def allocate_handoff_workspace(
    local_rank: int, group_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate state for one ordered stream or CUDA graph."""
    workspace_stride = (
        (OUTPUT_FEATURES + COMMUNICATION_BLOCK - 1) // COMMUNICATION_BLOCK
    ) * COMMUNICATION_BLOCK
    workspace = symm_mem.empty(
        (RING_SLOTS, WORLD_SIZE, workspace_stride),
        dtype=torch.uint32,
        device=torch.device("cuda", local_rank),
    )
    workspace.zero_()
    # The rendezvous is a host collective, not a completion fence for the
    # preceding device memset.  Complete every rank's initialization before
    # exposing peer aliases; otherwise a late zero can erase a first publish.
    torch.cuda.synchronize()
    symm_mem.rendezvous(workspace, group_name)
    ring_state = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    return workspace, ring_state


def _run(
    rank: int,
    local_rank: int,
    group: dist.ProcessGroup,
    *,
    verbose: bool,
) -> dict[str, Any]:
    import flashinfer
    import flashinfer.comm as flashinfer_comm
    from flashinfer.comm.mnnvl import TorchDistBackend
    from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp import _standalone

    activation_fp4, activation_scale, weight_fp4, weight_scale, alpha = quantize_inputs(
        rank
    )
    activation_bytes = activation_fp4.view(torch.uint8).reshape(-1)
    weight_bytes = weight_fp4.view(torch.uint8)
    activation_scale_bytes = activation_scale.reshape(-1).view(torch.int8)
    weight_scale_bytes = weight_scale.reshape(-1).view(torch.int8)
    alpha_value = float(alpha.item())
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260927)
    residual = torch.randn(
        (1, OUTPUT_FEATURES),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gamma = torch.randn(
        (OUTPUT_FEATURES,),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    symmetric_output = symm_mem.empty(
        (1, OUTPUT_FEATURES),
        dtype=torch.bfloat16,
        device=torch.device("cuda", local_rank),
    )
    symm_mem.rendezvous(symmetric_output, group.group_name)
    flashinfer_workspace = flashinfer_comm.create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=WORLD_SIZE,
        rank=rank,
        max_token_num=2048,
        hidden_dim=OUTPUT_FEATURES,
        dtype=torch.bfloat16,
        comm_backend=TorchDistBackend(group=dist.group.WORLD),
        group=dist.group.WORLD,
    )
    production_norm = torch.empty_like(residual)
    production_residual = torch.empty_like(residual)
    sglang_norm = torch.empty_like(residual)
    sglang_residual = torch.empty_like(residual)
    handoff_workspace, ring_state = allocate_handoff_workspace(
        local_rank, group.group_name
    )

    projection_args = (
        weight_bytes,
        activation_bytes,
        weight_scale_bytes,
        activation_scale_bytes,
        alpha_value,
        symmetric_output,
    )
    source_args = (
        *projection_args,
        handoff_workspace,
        ring_state,
        residual,
        gamma,
        hl.constexpr(rank),
        group.group_name,
    )

    def persistent() -> tuple[torch.Tensor, torch.Tensor]:
        return cast(
            "tuple[torch.Tensor, torch.Tensor]",
            deepseek_v3_attention_nvfp4_tp(*source_args),
        )

    def fused_tail(
        norm_out: torch.Tensor, residual_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flashinfer_comm.allreduce_fusion(
            input=symmetric_output,
            workspace=flashinfer_workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=gamma,
            rms_eps=RMS_EPS,
            use_oneshot=True,
            fp32_acc=True,
        )
        return norm_out, residual_out

    def standalone() -> tuple[torch.Tensor, torch.Tensor]:
        _standalone.attention_o_proj_local(*projection_args, group.group_name)
        reduced = torch.ops.symm_mem.one_shot_all_reduce(
            symmetric_output, "sum", group.group_name
        )
        return _standalone.residual_rms_norm(reduced, residual, gamma, group.group_name)

    def production() -> tuple[torch.Tensor, torch.Tensor]:
        torch.ops._C.cutlass_scaled_fp4_mm(
            symmetric_output,
            activation_fp4,
            weight_fp4,
            activation_scale,
            weight_scale,
            alpha,
        )
        return fused_tail(production_norm, production_residual)

    def sglang() -> tuple[torch.Tensor, torch.Tensor]:
        flashinfer.mm_fp4(
            activation_fp4,
            weight_fp4.T,
            activation_scale,
            weight_scale.T,
            alpha,
            out_dtype=torch.bfloat16,
            out=symmetric_output,
            backend="auto",
            enable_pdl=True,
        )
        return fused_tail(sglang_norm, sglang_residual)

    # Exercise the first tagged exchange immediately after initializing its
    # workspace; do not let a baseline collective accidentally serialize it.
    actual = persistent()
    torch.cuda.synchronize()
    dist.barrier()
    production_values = production()
    reference = (production_values[0].clone(), production_values[1].clone())

    def validate(_name: str, actual: object) -> None:
        norm, residual_value = cast("tuple[torch.Tensor, torch.Tensor]", actual)
        torch.testing.assert_close(norm, reference[0], rtol=2e-2, atol=0.25)
        torch.testing.assert_close(residual_value, reference[1], rtol=2e-2, atol=0.25)

    validate("distributed_helion", actual)
    gathered = [torch.empty_like(actual[1]) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, actual[1])
    for peer in gathered[1:]:
        torch.testing.assert_close(peer, gathered[0], rtol=0, atol=0)

    launches: dict[str, Callable[[], object]] = {
        "distributed_helion": persistent,
        "standalone_helion_one_shot": standalone,
        "vllm_cutlass_flashinfer": production,
        "sglang_flashinfer": sglang,
    }
    results = benchmark(launches, validate=validate)
    if verbose and rank == 0:
        for name, (warm, cold) in results.items():
            print(f"{name:>32s}: {warm:7.2f} us warm, {cold:7.2f} us cold L2")

    production_cold = min(
        results["vllm_cutlass_flashinfer"][1],
        results["sglang_flashinfer"][1],
    )
    persistent_cold = results["distributed_helion"][1]
    speedup = production_cold / persistent_cold
    return {
        "helion_wins": int(speedup > 1),
        "total": 1,
        "geomean": speedup,
        "best_speedup": speedup,
        "results": results,
    }


@torch.inference_mode()
def main(verbose: bool = True) -> dict[str, Any]:
    """Run TP4 correctness and paired cold-L2 measurements."""
    rank, local_rank, group = initialize(
        kernel_name="deepseek_v3_attention_nvfp4_tp",
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    try:
        return _run(rank, local_rank, group, verbose=verbose)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
