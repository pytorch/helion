"""Codegen-only build of the Qwen3.5 TP4 GDN mixer megakernel on one GPU (no
launch), as ``qwen3_5_moe_tp4.compile_only`` does it for the MoE.

CUDA_VISIBLE_DEVICES=N PYTHONHASHSEED=0 python -m \
    pretuned_kernels.megakernels.qwen3_5_gdn_tp4.compile_only [--warm] [out.py]
"""

from __future__ import annotations

import sys

from pretuned_kernels.megakernels.qwen3_5_gdn_tp4 import _setup as ms
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4.qwen3_5_gdn_mixer_tp4 import (
    qwen3_5_gdn_mixer_tp4,
)
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.compile_only import fake_group
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.compile_only import generate
import torch


def mixer_args(device: torch.device, group_name: str) -> tuple:
    """qwen3_5_gdn_mixer_tp4's arguments on uninitialized tensors."""
    partial = torch.empty(ms.T, ms.HIDDEN, dtype=torch.bfloat16, device=device)
    return ms.kernel_args(
        ms.empty_layer(device),
        ms.empty_state(device),
        ms.empty_inputs(device),
        partial,
        group_name,
    )


def main() -> None:
    argv = [a for a in sys.argv[1:] if a != "--warm"]
    out = argv[0] if argv else "/tmp/qwen3_5_gdn_mixer_tp4_gen.py"
    with fake_group() as group_name:
        args = mixer_args(torch.device("cuda"), group_name)
        generate(qwen3_5_gdn_mixer_tp4, args, out, warm="--warm" in sys.argv)


if __name__ == "__main__":
    main()
