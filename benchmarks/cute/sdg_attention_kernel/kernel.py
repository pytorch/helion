"""Run the retained SDG attention configs through ordinary Helion compilation.

This entry point matches the original experiment's ``kernel.py:run`` contract.
Dispatch depends only on host-visible layout metadata. Every call reads the
current queries, cache contents, page table and sequence lengths on the device.
"""

from __future__ import annotations

from functools import cache
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .headwise import paged_query_first_direct_attention as headwise
from .profile_contract import load_profiles
from .profile_contract import profile_signature
from .recurrence import paged_query_first_direct_attention as recurrence
from .rowwise import paged_query_first_direct_attention as rowwise
import helion
from helion.runtime.settings import _get_autotune_config_overrides

if TYPE_CHECKING:
    import torch

    from helion.runtime.kernel import Kernel

_FAMILIES = {
    "recurrence": recurrence,
    "rowwise": rowwise,
    "rowwise_extended": rowwise,  # Legacy name; retain the full six-choice domain.
    "headwise": headwise,
}
_PROFILE_PATH = Path(
    os.environ.get("SDG_ATTENTION_PROFILES", Path(__file__).with_name("configs.json"))
)
_PROFILES = load_profiles(_PROFILE_PATH)
_BY_NAME = {profile["name"]: profile for profile in _PROFILES}


_BY_SIGNATURE = {profile_signature(profile): profile for profile in _PROFILES}


def profile_names() -> tuple[str, ...]:
    """The supported, independently configured workload cases."""
    return tuple(_BY_NAME)


def configured_kernel(name: str) -> Kernel:
    """Keep stored configs fixed, including on repeated access after env changes."""
    if _get_autotune_config_overrides():
        raise ValueError(
            "HELION_AUTOTUNE_CONFIG_OVERRIDES cannot change fixed SDG profiles; "
            "unset it or provide an empty mapping."
        )
    return _configured_kernel(name)


@cache
def _configured_kernel(name: str) -> Kernel:
    profile = _BY_NAME[name]
    return helion.kernel(
        _FAMILIES[profile["family"]].fn,
        config=helion.Config.from_dict(profile["config"]),
        backend="cute",
        static_shapes=True,
        fast_math=True,
        force_autotune=False,
        autotune_force_persistent=False,
        autotune_config_overrides={},
    )


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    max_kv_len: int,
    window_left: int | None,
    return_lse: bool,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Execute one supported SDG layout with the caller's output allocation.

    The cache and E8M0 scales share their original backing storage. Query,
    output, key-cache and value-cache allocations must be distinct. The stored
    configs may enable PDL capture overlap; captured topology and pointer
    parameters must remain fixed, as documented by that CuTe config option.
    """
    tensors = (q, k, v, k_scale, v_scale, page_table, lengths, out)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("SDG attention inputs and output must share a device")
    signature = (
        tuple(
            (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
                tensor.storage_offset(),
            )
            for tensor in tensors
        ),
        max_kv_len,
        window_left,
        return_lse,
    )
    profile = _BY_SIGNATURE.get(signature)
    if profile is None:
        raise ValueError(f"Unsupported SDG attention layout; see {_PROFILE_PATH}")
    if k.untyped_storage() != k_scale.untyped_storage() or (
        v.untyped_storage() != v_scale.untyped_storage()
    ):
        raise ValueError("SDG cache data and scales must share their backing storage")
    storage = [tensor.untyped_storage() for tensor in (q, k, v, out)]
    if any(
        left == right for i, left in enumerate(storage) for right in storage[i + 1 :]
    ):
        raise ValueError("SDG query, output, key and value storage must be distinct")
    return configured_kernel(profile["name"])(
        q,
        k,
        v,
        k_scale,
        v_scale,
        page_table,
        lengths,
        window_left,
        128,
        return_lse,
        out,
    )
