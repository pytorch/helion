from __future__ import annotations

from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    from .metal_jit import _MetalKernel


def default_metal_launcher(
    metal_kernel: _MetalKernel,
    grid: tuple[int, ...],
    *args: object,
    _block_dims: tuple[int, int, int] = (256, 1, 1),
    _by_value: tuple[int, ...] = (),
    **kwargs: object,
) -> None:
    """Launch a ``@metal_jit`` kernel: ``grid`` threadgroups of ``_block_dims``.

    ``_by_value`` holds the positions in ``args`` of the runtime scalars,
    which codegen passes as Python floats and the kernel binds by value.
    """
    kwargs.pop("num_warps", None)
    kwargs.pop("num_stages", None)
    if kwargs:
        raise exc.BackendUnsupported(
            "metal", f"unexpected launcher kwargs: {sorted(kwargs)}"
        )
    metal_kernel.launch(grid, _block_dims, args, _by_value)
