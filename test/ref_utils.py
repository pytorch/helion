"""Helper utilities for reference mode tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import helion
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path

if TYPE_CHECKING:
    from helion.runtime.settings import RefMode


def clear_kernel_caches_and_set_ref_mode(ref_mode: RefMode) -> None:
    """Clear kernel caches and set ref_mode on all kernels in examples."""
    # Get all Python files in the examples directory
    example_files = Path(EXAMPLES_DIR).glob("*.py")

    for example_file in example_files:
        try:
            # Import the module
            mod = import_path(example_file)

            # Find all Helion kernels in the module and update their settings
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, helion.Kernel):
                    # Reset the kernel to clear any cached bound kernels
                    attr.reset()
                    # Update the kernel's ref_mode setting
                    attr.settings.ref_mode = ref_mode
        except Exception:
            # Skip files that can't be imported or have issues
            pass
