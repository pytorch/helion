"""Helion as a torch.compile (Inductor) backend: after enable(), Inductor emits
the CUDA kernels Helion supports (pointwise, reductions and natively lowered
matmuls) as Helion kernels and the rest as Triton::

    import helion.inductor

    helion.inductor.enable()
    torch.compile(model)(x)

Helion compiles and autotunes each kernel on its first call; its errors are not
caught, so they fail the compile or that call. Needs the Inductor of torch 2.16
(nightly), and for matmuls BaseScheduling.lowers_native_matmul
(pytorch/pytorch#204025)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from helion._compat import requires_torch_version

if not requires_torch_version("2.16"):
    # Rather than an ImportError or AttributeError from Inductor internals, e.g.
    # config.strict_reduction, which torch 2.15 does not have.
    raise ImportError(
        f"helion.inductor needs torch 2.16 or later, not {torch.__version__}"
    )

from torch._inductor import config as inductor_config
from torch._inductor.codegen import common

from . import config
from .kernel import HelionKernel
from .overrides import HelionOverrides
from .overrides import HelionPrinter
from .scheduling import HelionCUDAScheduling
from .scheduling import HelionScheduling

__all__ = [
    "HelionCUDAScheduling",
    "HelionKernel",
    "HelionOverrides",
    "HelionPrinter",
    "HelionScheduling",
    "config",
    "disable",
    "enable",
]

if TYPE_CHECKING:
    from torch._inductor.custom_graph_pass import CustomGraphModulePass
    from torch._inductor.scheduler import BaseScheduling
    from torch._inductor.scheduler import Scheduler
    from torch.utils._config_module import ConfigModule

# While enabled: Inductor's CUDA registration, and its custom pass and config
# entries (absent unless something registered them).
_saved: (
    tuple[
        common.DeviceCodegen,
        dict[str, CustomGraphModulePass | None],
        dict[str, ConfigModule | None],
    ]
    | None
) = None


def enable() -> None:
    """Make HelionCUDAScheduling Inductor's CUDA scheduling where it would be
    CUDACombinedScheduling (config.cuda_backend "triton")."""
    global _saved
    if _saved is not None:
        return
    common.init_backend_registration()
    default = common.device_codegens["cuda"]
    passes = common.custom_backend_passes
    configs = common.custom_backend_codegen_configs
    _saved = (
        default,
        {"cuda": passes["cuda"]} if "cuda" in passes else {},
        {"cuda": configs["cuda"]} if "cuda" in configs else {},
    )

    def scheduling(scheduler: Scheduler | None) -> BaseScheduling:
        # Inductor's registration is a function of config.cuda_backend; keep its
        # Halide and Pallas schedulings.
        if inductor_config.cuda_backend == "triton":
            return HelionCUDAScheduling(scheduler)
        return default.scheduling(scheduler)

    common.register_backend_for_device(
        "cuda",
        scheduling,
        default.wrapper_codegen,
        default.cpp_wrapper_codegen,
        default.fx_wrapper_codegen,
        device_custom_pass=passes.get("cuda"),
        device_custom_config=cast("ConfigModule", config),
    )


def disable() -> None:
    """Restore the CUDA backend enable() replaced."""
    global _saved
    if _saved is None:
        return
    common.device_codegens["cuda"], passes, configs = _saved
    common.custom_backend_passes.pop("cuda", None)
    common.custom_backend_passes.update(passes)
    common.custom_backend_codegen_configs.pop("cuda", None)
    common.custom_backend_codegen_configs.update(configs)
    _saved = None
