"""Metal launches as data: a kernel's MSL, grid and arguments, for other runtimes.

:func:`capture_metal_launches` runs a Helion kernel's host code with a
launcher that records each Metal launch instead of compiling and
dispatching it, so another runtime can run the kernels Helion generates.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
from typing import TYPE_CHECKING
from typing import Literal

from ... import exc
from .metal_launcher import DEFAULT_BLOCK_DIMS
from .metal_launcher import check_launcher_kwargs

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import torch

    from ...runtime.config import Config
    from ...runtime.kernel import Kernel
    from .metal_jit import MSLKernel
    from .metal_jit import _MetalKernel


@dataclasses.dataclass(frozen=True, eq=False)
class MetalArgument:
    """One argument of a launch, in the order the kernel takes them.

    ``binding`` says how the kernel receives it: ``"buffer"``, a tensor bound
    as a device buffer; ``"scalar"``, an int or float bound by value;
    ``"baked"``, a host value baked into the source, which is not bound.
    Buffers and scalars take the ``[[buffer(i)]]`` slots in order.  For a
    buffer, ``written`` says the kernel stores to it, ``read`` that it uses
    it otherwise (a load, or an operand view), and ``shares_storage`` that
    another argument views the same memory.
    """

    name: str
    value: object
    binding: Literal["buffer", "scalar", "baked"]
    written: bool = False
    read: bool = False
    shares_storage: bool = False


@dataclasses.dataclass(frozen=True, eq=False)
class MetalLaunch:
    """One Metal kernel launch of a Helion kernel: the complete MSL source
    of the ``kernel void`` named ``name``, launched as ``grid`` threadgroups
    of ``threadgroup`` threads on ``arguments``."""

    name: str
    source: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    arguments: tuple[MetalArgument, ...]
    msl: MSLKernel = dataclasses.field(repr=False, compare=False)


def capture_metal_launches(
    kernel: Kernel[object],
    args: Sequence[object],
    config: Config | None = None,
    *,
    num_sm: int | None = None,
) -> list[MetalLaunch]:
    """The Metal launches ``kernel`` makes on ``args`` at ``config`` (None:
    the config ``BoundKernel.to_code`` picks), without compiling or
    dispatching any of them.

    The host code runs as it would, so it allocates its outputs, on the
    device of ``args``.  The MSL depends on the tensors' dtypes, shapes and
    strides (as the kernel specializes on them) and which share storage, not
    their contents.  It also depends on the GPU in two places: a register
    matmul's operand layout is probed on an Apple GPU when the tensors are
    on MPS, and assumed otherwise; and a persistent kernel
    (``pid_type="persistent_*"``) bakes in the GPU's core count, ``num_sm``
    (less the cores ``persistent_reserved_sms`` reserves, at least 1, as on
    a GPU).  Without it, the host code reads the count from the tensors' device
    (``helion.runtime.get_num_sm``), which only MPS tensors have, so a
    persistent kernel captured on CPU tensors takes ``num_sm``.  For MSL
    that will run, capture on MPS tensors, with the ``num_sm`` of the GPU
    that will run it if that is another; CPU tensors suffice to read the
    code.
    """
    args = kernel.normalize_args(*args)
    bound = kernel.bind(args)
    if bound.env.backend.name != "metal":
        raise exc.BackendUnsupported(
            bound.env.backend.name, "capture_metal_launches takes a Metal kernel"
        )
    return record_launches(bound.compile_config(config), args, num_sm=num_sm)


def record_launches(
    entry: Callable[..., object],
    args: Sequence[object],
    *,
    num_sm: int | None = None,
) -> list[MetalLaunch]:
    """The launches ``entry``, a Metal kernel's generated host function,
    makes on ``args``: recorded by its launcher, not compiled or
    dispatched.  ``num_sm`` is the core count a persistent kernel bakes in
    (:func:`capture_metal_launches`)."""
    launches: list[MetalLaunch] = []

    def record(
        metal_kernel: _MetalKernel,
        grid: tuple[int, ...],
        *launch_args: object,
        _block_dims: tuple[int, int, int] = DEFAULT_BLOCK_DIMS,
        _by_value: tuple[int, ...] = (),
        **kwargs: object,
    ) -> None:
        check_launcher_kwargs(kwargs)
        launches.append(metal_kernel.capture(grid, _block_dims, launch_args, _by_value))

    if num_sm is not None and num_sm < 1:
        raise ValueError(f"num_sm is a core count, at least 1: got {num_sm}")

    def run() -> None:
        # Imported here: helion.runtime imports this module.
        from ... import runtime

        runtime._num_sm_override.set(functools.partial(_core_count, num_sm))
        entry(*args, _launcher=record)

    # The override is set in a copy of the context, so it ends with the
    # capture and no other thread's host code reads it.
    contextvars.copy_context().run(run)
    return launches


def _core_count(
    num_sm: int | None, device: torch.device, reserved_sms: int, cluster_size: int
) -> int | None:
    """The core count a capture's host code reads (``get_num_sm``):
    ``num_sm`` less the reserved cores, at least 1 as on a GPU; without
    ``num_sm``, the device's own (None) on MPS, and refused on another
    device, which has no Apple GPU's count to read."""
    if num_sm is not None:
        return max(num_sm - reserved_sms, 1)
    if device.type != "mps":
        raise exc.BackendUnsupported(
            "metal",
            "a persistent kernel bakes in the GPU's core count, which "
            f"{device.type} tensors do not have: capture it on MPS tensors, "
            "or pass num_sm",
        )
    return None
