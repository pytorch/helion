from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .. import exc
from .._compiler.metal.metal_launcher import (
    default_metal_launcher as default_metal_launcher,
)
from .config import Config as Config
from .cute.launcher import cute_cuda_graph as cute_cuda_graph
from .cute.launcher import default_cute_launcher as default_cute_launcher
from .cute.occupancy import get_max_active_clusters as get_max_active_clusters
from .kernel import Kernel as Kernel
from .kernel import OutputCodeOptions as OutputCodeOptions
from .kernel import kernel as kernel
from .pallas.launcher import default_pallas_launcher as default_pallas_launcher
from .settings import is_pallas_interpret as _module_is_pallas_interpret
from .triton.launcher import default_launcher as _triton_default_launcher
from .triton.launcher import get_num_sm as _triton_get_num_sm
from .triton.launcher import get_num_xcd as get_num_xcd
from .triton.launcher import set_triton_allocator as set_triton_allocator

if TYPE_CHECKING:
    import torch

    from .._compiler.flydsl.launcher import (
        default_flydsl_launcher as default_flydsl_launcher,
    )

log: logging.Logger = logging.getLogger(__name__)


def default_launcher(
    triton_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    _remote_copy_signal_dst: torch.Tensor | None = None,
    _remote_copy_signal_slots_per_program: int = 0,
    _remote_copy_process_group_name: str | None = None,
    _remote_barrier_signal_slots_per_program: int = 0,
    _remote_barrier_process_group_name: str | None = None,
    _remote_copy_scratch_specs: tuple[tuple[torch.Tensor, int], ...] = (),
    _persistent_state_specs: tuple[
        tuple[torch.Tensor, int, torch.dtype, bool], ...
    ] = (),
    _persistent_state_process_group_name: str | None = None,
    _persistent_state_rank_digest: str | None = None,
    _minimum_resident_programs: int = 0,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Thin in-process wrapper over the dependency-free
    :func:`helion.runtime.triton.launcher.default_launcher` that translates
    Triton's opaque "incompatible dimensions" error into
    :class:`helion.exc.ShapeMismatch`.
    """
    try:
        return _triton_default_launcher(
            triton_kernel,
            grid,
            *args,
            num_warps=num_warps,
            num_stages=num_stages,
            _remote_copy_signal_dst=_remote_copy_signal_dst,
            _remote_copy_signal_slots_per_program=_remote_copy_signal_slots_per_program,
            _remote_copy_process_group_name=_remote_copy_process_group_name,
            _remote_barrier_signal_slots_per_program=_remote_barrier_signal_slots_per_program,
            _remote_barrier_process_group_name=_remote_barrier_process_group_name,
            _remote_copy_scratch_specs=_remote_copy_scratch_specs,
            _persistent_state_specs=_persistent_state_specs,
            _persistent_state_process_group_name=_persistent_state_process_group_name,
            _persistent_state_rank_digest=_persistent_state_rank_digest,
            _minimum_resident_programs=_minimum_resident_programs,
            ptx_options=ptx_options,
            launch_cooperative_grid=launch_cooperative_grid,
            **kwargs,
        )
    except Exception as error:
        message = str(error)
        if "Cannot make_shape_compatible: incompatible dimensions" in message:
            raise exc.ShapeMismatch("kernel operands", message) from error
        if message.startswith("Cross-loop scheduling requires "):
            raise exc.InvalidConfig(message) from error
        raise


def get_num_sm(device: torch.device, *, reserved_sms: int = 0) -> int:
    """Number of SMs (persistent-kernel grid size) for any Helion device.

    Adds the CPU (Pallas-interpret) and TPU cases on top of the dependency-free
    GPU helper :func:`helion.runtime.triton.launcher.get_num_sm`. See that
    function for argument/return semantics.
    """
    if device.type == "cpu":
        if not _module_is_pallas_interpret():
            raise AssertionError("TODO: implement for other devices")
        return 1
    if device.type == "tpu":
        return 1
    return _triton_get_num_sm(device, reserved_sms=reserved_sms)


if not TYPE_CHECKING:  # keep type checkers flagging typos in helion.runtime

    def __getattr__(name: str) -> object:
        # Generated FlyDSL code (and code saved since v1.5.0) imports the
        # launcher by this name; the code lives in the FlyDSL package.
        if name == "default_flydsl_launcher":
            from .._compiler.flydsl.launcher import default_flydsl_launcher

            return default_flydsl_launcher
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
