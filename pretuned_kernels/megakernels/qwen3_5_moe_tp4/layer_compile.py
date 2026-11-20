"""Codegen-only build of the merged Qwen3.5 TP4 layer megakernel (MoE + the next
layer's GDN mixer) on one GPU (no launch), as ``compile_only`` does it for the MoE.

CUDA_VISIBLE_DEVICES=N PYTHONHASHSEED=0 python -m \
    pretuned_kernels.megakernels.qwen3_5_moe_tp4.layer_compile [--warm] [out.py]

LAYER (default qwen3_5_layer_tp4): the merged kernel's module in this package.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING

from pretuned_kernels.megakernels.qwen3_5_gdn_tp4.compile_only import mixer_args
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.compile_only import fake_group
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.compile_only import generate
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.compile_only import moe_args
import torch

if TYPE_CHECKING:
    from types import ModuleType

PKG = "pretuned_kernels.megakernels"


def layer_module() -> ModuleType:
    """The merged kernel's module (LAYER, default qwen3_5_layer_tp4)."""
    name = os.environ.get("LAYER", "qwen3_5_layer_tp4")
    return importlib.import_module(f"{PKG}.qwen3_5_moe_tp4.{name}")


def separate_kernels(lmod: ModuleType) -> tuple:
    """The standalone MoE and GDN mixer kernels the merged kernel's halves match."""
    moe = importlib.import_module(f"{PKG}.qwen3_5_moe_tp4.{lmod.LAYER_MOE}")
    mixer = importlib.import_module(f"{PKG}.qwen3_5_gdn_tp4.{lmod.LAYER_MIXER}")
    return moe.qwen3_5_moe_tp4, mixer.qwen3_5_gdn_mixer_tp4


def merged_args(lmod: ModuleType, moe_vals: dict, mixer_vals: dict) -> list:
    """The merged kernel's positional arguments from the MoE's and the mixer's by
    name: the mixer's hidden_states and residual are dropped (the MoE's outputs
    feed them) and its names that the MoE also has carry LAYER_PREFIX."""
    args = dict(moe_vals)
    for name, value in mixer_vals.items():
        if name in ("hidden_states", "residual"):
            continue
        if name in moe_vals and name != "group_name":
            name = lmod.LAYER_PREFIX + name
        args[name] = value
    return [args[name] for name in lmod.qwen3_5_layer_tp4.signature.parameters]


def fake_layer_args(lmod: ModuleType, device: torch.device, group_name: str) -> list:
    """The merged kernel's arguments on uninitialized tensors."""
    moe, mixer = separate_kernels(lmod)
    moe_vals = dict(
        zip(moe.signature.parameters, moe_args(device, group_name), strict=True)
    )
    mixer_vals = dict(
        zip(mixer.signature.parameters, mixer_args(device, group_name), strict=True)
    )
    return merged_args(lmod, moe_vals, mixer_vals)


def main() -> None:
    argv = [a for a in sys.argv[1:] if a != "--warm"]
    out = argv[0] if argv else "/tmp/qwen3_5_layer_tp4_gen.py"
    lmod = layer_module()
    with fake_group() as group_name:
        args = fake_layer_args(lmod, torch.device("cuda"), group_name)
        generate(lmod.qwen3_5_layer_tp4, args, out, warm="--warm" in sys.argv)


if __name__ == "__main__":
    main()
