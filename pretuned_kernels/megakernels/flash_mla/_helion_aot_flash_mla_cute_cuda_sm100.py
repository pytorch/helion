"""Expose the existing guarded B200 MLA preset to pretuned benchmark discovery."""

from __future__ import annotations

from copy import deepcopy

from pretuned_kernels.megakernels.flash_mla import flash_mla as mla
from pretuned_kernels.megakernels.flash_mla.flash_mla import (
    _cute_key as key_flash_mla_cute,
)


def autotune_flash_mla_cute(*args) -> dict[str, object]:
    """Use the same input contract and sole fixed config as the public CuTe entry."""
    key_flash_mla_cute(*args)
    (config,) = mla.flash_mla_cute.configs
    return deepcopy(config.config)
