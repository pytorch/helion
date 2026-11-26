"""Where MPP's matmul lowering can run, and the preamble its kernels start with.

Kept apart from the codegen modules, which ``backend_registry`` reloads, so
test patches of it survive those reloads."""

from __future__ import annotations

import torch

#: The headers MPP's lowering includes (Metal 4, macOS 26).
MPP_HEADERS = ("metal_tensor", "MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h")


def mpp_preamble() -> list[str]:
    """The MSL a kernel with MPP ops starts with: a guard naming each
    missing header, the includes and the namespace."""
    lines: list[str] = []
    for header in MPP_HEADERS:
        lines += [
            f"#if !__has_include(<{header}>)",
            f'#error "Helion Metal MPP lowering requires <{header}>"',
            "#endif",
        ]
    lines += [f"#include <{header}>" for header in MPP_HEADERS]
    lines.append("using namespace mpp::tensor_ops;")
    return lines


def mpp_available(device: torch.device) -> bool:
    """Whether MPP's matmul lowering can run on ``device``: macOS 26 or
    newer, whose Metal compiler ships MPP's headers.  A kernel bound to CPU
    tensors, as when rendering MSL only, compiles nothing: assume it can.

    This reads the OS version only: a Metal language version lowered with
    ``PYTORCH_MPS_METAL_VERSION`` still reaches the preamble's ``#error``
    guards (:func:`mpp_preamble`) at launch."""
    return device.type != "mps" or torch.backends.mps.is_macos_or_newer(26, 0)
