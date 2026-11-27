"""The zero-fill kernels of a generated module.

A backend may launch a kernel's host zero fills as kernels of their own, which
the generated module then holds beside the kernel named after its function.
The code generator names them after that kernel with ``zero_fill_kernel_name``,
and everything that handles a module's kernels finds them by the names that
function builds (``zero_fill_kernel_names``), so the generator and the rule
are one definition. This module imports nothing, so the generator, the
backend and the benchmark worker can all share it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

# The prefix of the names a generated module gives its zero-fill kernels.
ZERO_FILL_KERNEL_PREFIX = "_helion_zero_fill_"


def zero_fill_kernel_name(index: int, device_kernel: str) -> str:
    """The name of zero-fill kernel number ``index`` of the device kernel
    named ``device_kernel``. A kernel ``<kernel>`` has the device kernel
    ``_helion_<kernel>``, whose fill kernels are therefore
    ``_helion_zero_fill_<index>_helion_<kernel>``."""
    return _numbered_name(str(index), device_kernel)


def _numbered_name(number: str, device_kernel: str) -> str:
    """``zero_fill_kernel_name`` for the number written as ``number``."""
    return f"{ZERO_FILL_KERNEL_PREFIX}{number}_{device_kernel.lstrip('_')}"


def _zero_fill_number(name: str) -> str | None:
    """The number a zero-fill kernel's name would carry, as written: the
    text between the prefix and the next underscore where it is a number in
    the form ``zero_fill_kernel_name`` writes one, ASCII decimal digits
    without a leading zero; else None. The digits are compared as text, so
    a name with any count of them is answered."""
    if not name.startswith(ZERO_FILL_KERNEL_PREFIX):
        return None
    digits = name[len(ZERO_FILL_KERNEL_PREFIX) :].partition("_")[0]
    if digits.isascii() and digits.isdigit() and (digits == "0" or digits[0] != "0"):
        return digits
    return None


def zero_fill_kernel_names(
    module_globals: Mapping[str, object], kernel_name: str
) -> list[str]:
    """The names of the zero-fill kernels of the kernel ``kernel_name`` among
    a generated module's globals, sorted: exactly the names
    ``zero_fill_kernel_name`` builds for its device kernel,
    ``_helion_zero_fill_<n>_helion_<kernel_name>`` with ``n`` in decimal.

    The one rule for everything that has to find them: the backend, which
    gives a module's kernels its source hash and reads their compile-cache
    keys; the record of the kernels that carried that hash in the parent
    process (``precompile_future._source_hash_globals``); and the release
    of a job's kernels (``precompile_future._module_kernels``). A global
    under any other name is no zero-fill kernel of ``kernel_name`` to any
    of them, whatever it carries: a kernel without the prefix, the fill
    kernel of another kernel in the same globals (for a kernel ``add``,
    ``_helion_zero_fill_0_helion_helion_add`` belongs to a kernel
    ``helion_add``), or a name with anything but a number in its place.
    """
    device_kernel = f"_helion_{kernel_name}"
    return sorted(
        name
        for name in module_globals
        if (number := _zero_fill_number(name)) is not None
        and name == _numbered_name(number, device_kernel)
    )
