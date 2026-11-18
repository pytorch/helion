"""Helion as an Inductor backend for CUDA, as Halide and Pallas are: with
torch._inductor.config.cuda_backend "helion", Inductor emits every CUDA kernel
(pointwise, reductions and natively lowered matmuls) as a Helion kernel::

    import helion.inductor

    torch.compile(model, options={"cuda_backend": "helion"})(x)

Settings of helion.inductor.config go in the same options with a "helion."
prefix: options={"cuda_backend": "helion", "helion.native_matmul": False}. A
kernel that Helion does not support yet fails the compile with Unsupported, as
with Halide.

Helion compiles and autotunes each kernel on its first call; its errors are not
caught, so they fail the compile or that call. Needs the Inductor of torch 2.16
(nightly), and for matmuls InductorChoices.use_native_matmul
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
from torch._inductor.choices import register_inductor_choices
from torch._inductor.codegen import common

from . import config
from .choices import HelionChoices
from .kernel import HelionKernel
from .overrides import HelionOverrides
from .overrides import HelionPrinter
from .passes import HelionPostGradPass
from .scheduling import HelionScheduling
from .scheduling import Unsupported

__all__ = [
    "HelionChoices",
    "HelionKernel",
    "HelionOverrides",
    "HelionPostGradPass",
    "HelionPrinter",
    "HelionScheduling",
    "Unsupported",
    "config",
]

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseScheduling
    from torch._inductor.scheduler import Scheduler
    from torch.utils._config_module import ConfigModule


def _register() -> None:
    """Make "helion" a value of torch._inductor.config.cuda_backend: Inductor's
    CUDA registration maps the others to their schedulings, and has no entry for
    this one. Also registers HelionChoices, helion.inductor.config as the
    options' "helion." namespace, which puts it in the FX graph cache key, and
    HelionPostGradPass as the CUDA custom backend pass, after any registered
    before it."""
    register_inductor_choices("helion", HelionChoices)
    common.custom_backend_codegen_configs["helion"] = cast("ConfigModule", config)
    common.init_backend_registration()
    default = common.device_codegens["cuda"]

    def scheduling(scheduler: Scheduler | None) -> BaseScheduling:
        if inductor_config.cuda_backend == "helion":
            return HelionScheduling(scheduler)
        return default.scheduling(scheduler)

    common.register_backend_for_device(
        "cuda",
        scheduling,
        default.wrapper_codegen,
        default.cpp_wrapper_codegen,
        default.fx_wrapper_codegen,
        device_custom_pass=HelionPostGradPass(common.custom_backend_passes.get("cuda")),
        device_custom_config=common.custom_backend_codegen_configs.get("cuda"),
    )


_register()
