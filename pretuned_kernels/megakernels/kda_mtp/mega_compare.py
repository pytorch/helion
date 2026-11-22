# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""The Helion KDA verify megakernel against sgl-nvidia-mega's verify path and
SGLang's, on one GPU (no comms).

sgl-nvidia-mega (``SGL_NVIDIA_MEGA``, default ~/local/sgl-nvidia-mega): its
split-K input projection (PreparedProjection), the KdaSplit verify core
(PreparedKda.run_verify: split cache, per-token FP32 state checkpoints and conv
windows) and the output projection on the same split-K GEMM (its
PreparedOproj fuses the TP all-reduce; the local GEMM stands in for it).

python -m pretuned_kernels.megakernels.kda_mtp.mega_compare [rounds]
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

from pretuned_kernels._bench import capture_cuda_graph
from pretuned_kernels.megakernels.kda_mtp import _cold as cold
from pretuned_kernels.megakernels.kda_mtp import _reference as ref
from pretuned_kernels.megakernels.kda_mtp import kda_mtp_verify as kmv


def mega_call(base: dict):
    """sgl-nvidia-mega's verify step on a copy of ``base`` and its output."""
    sys.path.insert(
        0, os.environ.get("SGL_NVIDIA_MEGA", os.path.expanduser("~/local/sgl-nvidia-mega"))
    )
    from sgl_nvidia_mega.kimi_k3.sm103.gemm.projection import PreparedProjection
    from sgl_nvidia_mega.kimi_k3.sm103.kda.prepared import PreparedKda

    t = ref.clone_outputs(base)
    dev = t["hidden_states"].device
    # sgl-nvidia-mega treats slot <= 0 as padding: run the request on slot 1.
    conv_state, rec_state = t["conv_state"], t["recurrent_state"]
    conv_state[1].copy_(conv_state[0])
    rec_state[1].copy_(rec_state[0])
    slots = torch.tensor([1], device=dev, dtype=torch.int32)
    scratch_idx = torch.tensor([0], device=dev, dtype=torch.int32)
    conv_windows = torch.zeros(2, kmv.T, 3, 4608, device=dev, dtype=torch.bfloat16)
    checkpoints = torch.zeros(2, kmv.T, 12, 128, 128, device=dev, dtype=torch.float32)
    conv = t["conv_weight"].transpose(1, 2).reshape(4608, 1, 4)
    proxy = SimpleNamespace(
        conv1d=SimpleNamespace(weight=conv, bias=None),
        dt_bias=t["dt_bias"],
        A_log=t["a_log"],
        f_b_proj=SimpleNamespace(weight=t["f_b_weight"]),
        o_norm=SimpleNamespace(weight=t["norm_weight"], eps=1e-5),
        gate_lower_bound=-5.0,
    )
    proj = PreparedProjection(t["input_weight"], {kmv.T: (128, 8, 2, 6)}).prepare()
    oproj = PreparedProjection(t["output_weight"], {kmv.T: (128, 8, 2, 6)}).prepare()
    core = PreparedKda(proxy, {(1, kmv.T): (6288, 1)}, False).prepare_verify_caches(
        (conv_state, rec_state), (conv_windows, checkpoints)
    )
    x = t["hidden_states"]
    projected = torch.empty(kmv.T, 6288, device=dev, dtype=torch.bfloat16)
    local = torch.empty(kmv.T, 1536, device=dev, dtype=torch.bfloat16)
    out = torch.empty(kmv.T, 7168, device=dev, dtype=torch.bfloat16)

    def call():
        proj.run(x, projected)
        core.run_verify(
            projected,
            local,
            caches=(conv_state, rec_state),
            slots=slots,
            intermediates=(conv_windows, checkpoints),
            scratch_indices=scratch_idx,
        )
        oproj.run(local, out)

    return call, out


def main(rounds: int = 3) -> None:
    base = ref.make_inputs(1, kmv.H, kmv.T, cache_ring=True, seed=0)
    expected = ref.torch_reference(ref.clone_outputs(base), kmv.T, cache_ring=True)
    mega, mega_out = mega_call(base)
    mega()
    torch.cuda.synchronize()
    err = (mega_out.float() - expected.float().view_as(mega_out)).abs().max().item()
    print(f"sgl-nvidia-mega output max abs err {err:.3e}")
    candidate = ref.clone_outputs(base)
    out = torch.empty(kmv.T, 7168, device="cuda", dtype=torch.bfloat16)
    args = kmv.kernel_args(candidate, out)
    fn = kmv.compile_kernel(args)
    calls = {
        "mega": mega,
        "helion": lambda: fn(*args),
        "sglang": ref.make_sglang_call(
            ref.clone_outputs(base), kmv.T, side_stream=torch.cuda.Stream()
        ),
    }
    with torch.inference_mode():
        graphs = {name: capture_cuda_graph(call)[0] for name, call in calls.items()}
        for _ in range(rounds):
            single = cold.bench_clean([g.replay for g in graphs.values()], rep=400)
            steady, _ = cold.bench_steady(list(calls.values()), 20)
            print(
                "read-flush single replay: "
                + "  ".join(f"{n} {v:.2f} us" for n, v in zip(graphs, single))
            )
            print(
                "steady (20 x flush + call per graph): "
                + "  ".join(f"{n} {v:.2f} us" for n, v in zip(calls, steady))
            )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
