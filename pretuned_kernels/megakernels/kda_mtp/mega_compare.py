# pyrefly: ignore-errors
"""Time the Helion KDA verify megakernel against sgl-nvidia-mega's verify path
and SGLang's, on one GPU (no comms).

sgl-nvidia-mega (a checkout at ``$SGL_NVIDIA_MEGA``) runs the step as three
kernels: its split-K input projection (PreparedProjection), the KdaSplit verify
core (PreparedKda.run_verify: split cache, per-token FP32 state checkpoints and
conv windows) and the output projection on the same split-K GEMM (its
PreparedOproj fuses the TP all-reduce; the local GEMM stands in for it). Its
output's largest deviation from the PyTorch reference is printed before timing.
SGLang's verify path is timed too, so SGLang must be installed.

    SGL_NVIDIA_MEGA=/path/to/sgl-nvidia-mega \\
        python -m pretuned_kernels.megakernels.kda_mtp.mega_compare [rounds]
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

from pretuned_kernels.megakernels.kda_mtp import _cold as cold
from pretuned_kernels.megakernels.kda_mtp import _reference as ref
from pretuned_kernels.megakernels.kda_mtp import kda_mtp_verify as kmv
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

MEGA_ENV = "SGL_NVIDIA_MEGA"


def _import_mega() -> tuple[type, type]:
    """sgl-nvidia-mega's PreparedProjection and PreparedKda, from the checkout
    at ``$SGL_NVIDIA_MEGA``."""
    root = os.environ.get(MEGA_ENV)
    if not root:
        raise RuntimeError(f"set {MEGA_ENV} to an sgl-nvidia-mega checkout")
    sys.path.insert(0, root)
    try:
        from sgl_nvidia_mega.kimi_k3.sm103.gemm.projection import PreparedProjection
        from sgl_nvidia_mega.kimi_k3.sm103.kda.prepared import PreparedKda
    except ImportError as error:
        raise RuntimeError(
            f"cannot import sgl_nvidia_mega from {MEGA_ENV}={root}: {error}"
        ) from error
    return PreparedProjection, PreparedKda


def mega_call(
    base: dict, projection: type, kda: type
) -> tuple[Callable[[], None], torch.Tensor]:
    """sgl-nvidia-mega's verify step on ``base``, writing copies of its cache
    buffers, and its output."""
    t = ref.clone_outputs(base)
    dev = t["hidden_states"].device
    width = t["input_weight"].shape[0]
    # sgl-nvidia-mega runs the request on slot 1, a copy of base's request's
    # slot that no other system reads.
    conv_state, rec_state = t["conv_state"], t["recurrent_state"]
    request = int(t["state_indices"][0])
    conv_state[1].copy_(conv_state[request])
    rec_state[1].copy_(rec_state[request])
    slots = torch.tensor([1], device=dev, dtype=torch.int32)
    scratch_idx = torch.tensor([0], device=dev, dtype=torch.int32)
    conv_windows = torch.zeros(2, kmv.T, 3, 3 * kmv.S, device=dev, dtype=torch.bfloat16)
    checkpoints = torch.zeros(
        2, kmv.T, kmv.H, kmv.D, kmv.D, device=dev, dtype=torch.float32
    )
    conv = t["conv_weight"].transpose(1, 2).reshape(3 * kmv.S, 1, 4)
    proxy = SimpleNamespace(
        conv1d=SimpleNamespace(weight=conv, bias=None),
        dt_bias=t["dt_bias"],
        A_log=t["a_log"],
        f_b_proj=SimpleNamespace(weight=t["f_b_weight"]),
        o_norm=SimpleNamespace(weight=t["norm_weight"], eps=ref.EPS),
        gate_lower_bound=ref.LOWER_BOUND,
    )
    proj = projection(t["input_weight"], {kmv.T: (128, 8, 2, 6)}).prepare()
    oproj = projection(t["output_weight"], {kmv.T: (128, 8, 2, 6)}).prepare()
    core = kda(proxy, {(1, kmv.T): (width, 1)}, False).prepare_verify_caches(
        (conv_state, rec_state), (conv_windows, checkpoints)
    )
    x = t["hidden_states"]
    projected = torch.empty(kmv.T, width, device=dev, dtype=torch.bfloat16)
    local = torch.empty(kmv.T, kmv.S, device=dev, dtype=torch.bfloat16)
    out = torch.empty(kmv.T, ref.HIDDEN, device=dev, dtype=torch.bfloat16)

    def call() -> None:
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
    projection, kda = _import_mega()
    base = ref.make_inputs(1, kmv.H, kmv.T)
    expected = ref.torch_reference(ref.clone_outputs(base), kmv.T)
    mega, mega_out = mega_call(base, projection, kda)
    mega()
    torch.cuda.synchronize()
    err = (mega_out.float() - expected.float()).abs().max().item()
    print(f"sgl-nvidia-mega output max abs err {err:.3e}")
    cold.compare(
        {
            "mega": mega,
            "helion": kmv.prepare(base)[2],
            "sglang": ref.make_sglang_call(
                ref.clone_outputs(base), kmv.T, side_stream=torch.cuda.Stream()
            ),
        },
        rounds,
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
