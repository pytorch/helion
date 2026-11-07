"""The flash-attention search surface's facts (analysis result).

:class:`FlashSurfaceFacts` is what the attention detector proves about a
kernel (``FlashSearchSurface``) plus the two facts the binding adds (the
device's SM count and the ``fast_math`` setting) and the knob channel it
was bound under (:class:`FlashEnv`, the ``HELION_CUTE_FLASH_*`` variables
captured once at bind, overlaid with ``Settings.cute_flash_env_overrides``).
``ConfigSpec.flash_facts`` builds it from the spec's ``_cute_flash_*``
attributes; the three flash search builders in ``cute_flash`` (the config
resolver, the fragment surface, the seeds) take it as their first argument.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import overload

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .analysis.attention_match import FlashSearchSurface

FLASH_ENV_PREFIX = "HELION_CUTE_FLASH_"
_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class FlashEnv:
    """The ``HELION_CUTE_FLASH_*`` knob channel as the resolver reads it.

    ``values`` is the sorted ``(name, value)`` list of every variable with
    the prefix; the resolver reads nothing else from the environment.
    """

    values: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def capture(overrides: Mapping[str, str] | None = None) -> FlashEnv:
        """Snapshot the process environment, ``overrides`` taking precedence."""
        values = {
            name: value
            for name, value in os.environ.items()
            if name.startswith(FLASH_ENV_PREFIX)
        }
        if overrides:
            values.update(overrides)
        return FlashEnv(tuple(sorted(values.items())))

    @functools.cached_property
    def _mapping(self) -> dict[str, str]:
        return dict(self.values)

    @overload
    def get(self, name: str) -> str | None: ...

    @overload
    def get(self, name: str, default: str | _T) -> str | _T: ...

    def get(self, name: str, default: object = None) -> object:
        return self._mapping.get(name, default)


@dataclasses.dataclass(frozen=True)
class FlashSurfaceFacts:
    """Everything the flash search surface is built from.

    Field names and defaults are those of ``flash_surface.enable_cute_flash_search``
    and of the three builders' keywords (``dtype`` is the detector's
    ``io_dtype``).  ``tmem_rowmax_compatible`` stays ``None`` when the caller
    leaves the builders to derive it; the spec records its own derivation.
    ``block_size_targets`` maps block ids to the pinned tile sizes and is
    excluded from the hash.

    Hashability contract: ``flash_surface._cute_flash_autotune_fragments`` and
    ``test/cute_flash_testing.memoized_flash_fragments`` key memos on the
    facts, so every field must be hashable; a new list / dict / mapping field
    needs ``hash=False`` (or a tuple).
    """

    head_dim: int
    num_kv: int | None
    num_bh: int | None = None
    tensor_4d_heads: int | None = None
    dtype: torch.dtype = torch.float16
    block_size_targets: Mapping[int, int] = dataclasses.field(
        default_factory=dict, hash=False
    )
    is_causal: bool = False
    has_kv_tile_pruning: bool = False
    requires_ws_overlap: bool = False
    small_biased_candidate: bool = False
    standard_dense_output: bool = False
    dense_lse_output: bool = False
    standard_causal_output: bool = False
    tmem_rowmax_compatible: bool | None = None
    causal_resident_compatible: bool = False
    output_requires_tma: bool = False
    supports_tensor_4d_tma: bool = True
    has_row_epilogue: bool = False
    plain_row_body: bool = True
    has_score_modifiers: bool = False
    score_modifier_kinds: tuple[str, ...] = ()
    device_sm_count: int = 0
    head_dim_v: int | None = None
    kv_group: int = 1
    fast_math: bool = False
    has_list_walk: bool = False
    flex_has_score_region: bool = False
    flex_has_mask_region: bool = False
    flex_trivial_walk: bool = False
    q_seq: int | None = None
    kv_seq: int | None = None
    flex_aux_smem_candidates: tuple[int, ...] = ()
    flex_prefetch_regs: tuple[int, ...] = ()
    flex_coalesce_steps: tuple[int, ...] = ()
    flex_hook_ops: int = 0
    env: FlashEnv | None = None

    @staticmethod
    def from_surface(
        surface: FlashSearchSurface,
        *,
        device_sm_count: int,
        fast_math: bool,
        env: FlashEnv,
    ) -> FlashSurfaceFacts:
        """The detector's surface plus the binding's facts."""
        fields = surface._asdict()
        dtype = fields.pop("io_dtype")
        return FlashSurfaceFacts(
            dtype=dtype,
            device_sm_count=device_sm_count,
            fast_math=fast_math,
            env=env,
            **fields,
        )

    def as_kwargs(self) -> dict[str, Any]:
        """The facts as ``enable_cute_flash_search`` keywords."""
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }
