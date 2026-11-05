"""Immutable source constraints of the complete rank-two modular epilogue.

This certificate is source/FX geometry, not an alias fact or tensor owner. It is
created only from complete anchored store chains and consumed by the existing
ConfigSpec normalizer and seed admission. Final lowering retains its independent
readonly, alias, geometry and protocol guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .modular_aux import ModularAuxLoadMap
from .pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from .strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from .strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from .strategies import Tcgen05LayoutStrategy
from .tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from .tcgen05_constants import TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from .cute_epilogue import Tcgen05UnaryEpilogueChain


@dataclass(frozen=True)
class ModularAuxProtocolSubject:
    anchor: torch.fx.Node
    stores: tuple[torch.fx.Node, ...]
    maps: tuple[ModularAuxLoadMap, ...]
    block_ids: tuple[int, int]
    output_dtype: torch.dtype

    @staticmethod
    def inert_options() -> tuple[tuple[str, object], ...]:
        """The existing single-CTA, per-subtile SIMT protocol's pinned options.

        Placement remains its ordinary ConfigSpec proof. NM/compact geometry,
        grouped/leading scopes and logical/native tile identity retain their
        separate source and late guards; this is not a complete codegen proof.
        """
        return (
            ("tcgen05_cluster_m", 1),
            ("tcgen05_cluster_n", 1),
            (TCGEN05_AUX_LOAD_MODE_CONFIG_KEY, TCGEN05_AUX_LOAD_MODE_SIMT),
            (TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY, 0),
            (TCGEN05_EPI_SUBTILE_N_CONFIG_KEY, None),
            (TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY, 1),
            (TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY, 0),
            (TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY, False),
            (TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY, Tcgen05LayoutStrategy.DEFAULT.value),
        )

    def refusal(
        self,
        config: Mapping[str, object],
        indices: tuple[int, int] | None,
    ) -> str | None:
        if config.get("cute_collective_mma") is True:
            # Collective lowering owns its original coordinate protocol; this
            # constraint must not erase the legitimate source fallback domain.
            return None
        if config.get(TCGEN05_CTA_GROUP_CONFIG_KEY, "auto") not in ("auto", "one"):
            return "rank-two modular auxiliaries require a single-CTA source protocol"
        for key, value in self.inert_options():
            requested = config.get(key, value)
            if type(requested) is not type(value) or requested != value:
                if key == TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY:
                    return "mapped auxiliary loads require the ordinary per-subtile SIMT path"
                return "rank-two modular auxiliary loads require the plain single-CTA per-subtile protocol"
        for key in (
            "tcgen05_layout_overrides_epi_tile_m",
            "tcgen05_layout_overrides_epi_tile_n",
            "tcgen05_layout_overrides_d_store_box_n",
        ):
            if config.get(key) is not None:
                return "rank-two modular auxiliaries require implicit ordinary epilogue tiles"
        blocks = config.get("block_sizes")
        if (
            not isinstance(blocks, list)
            or indices is None
            or any(i >= len(blocks) for i in indices)
        ):
            return "rank-two modular auxiliaries require their original M/N block sizes"
        m, n = blocks[indices[0]], blocks[indices[1]]
        if type(m) is not int or type(n) is not int or m <= 0 or n <= 0:
            return "rank-two modular auxiliary tiles require positive static dimensions"
        if m > TCGEN05_ONE_CTA_MAX_BLOCK_M:
            return "rank-two modular auxiliaries require one native MMA M tile"
        for mapping in self.maps:
            reason = mapping.tile_reason(m, n)
            if reason is not None:
                return reason
        return None

    def repair(
        self,
        config: dict[str, object],
        indices: tuple[int, int] | None,
    ) -> None:
        if config.get("cute_collective_mma") is True:
            return
        if TCGEN05_CTA_GROUP_CONFIG_KEY in config:
            config[TCGEN05_CTA_GROUP_CONFIG_KEY] = "auto"
        for key, value in self.inert_options():
            if key in config:
                config[key] = value
        for key in (
            "tcgen05_layout_overrides_epi_tile_m",
            "tcgen05_layout_overrides_epi_tile_n",
            "tcgen05_layout_overrides_d_store_box_n",
        ):
            if key in config:
                config[key] = None
        blocks = config.get("block_sizes")
        if (
            not isinstance(blocks, list)
            or indices is None
            or any(i >= len(blocks) for i in indices)
        ):
            return
        blocks = list(blocks)
        for axis, index in enumerate(indices):
            value = blocks[index]
            if type(value) is not int or value <= 0:
                return
            if axis == 0:
                value = min(value, TCGEN05_ONE_CTA_MAX_BLOCK_M)
            # Search uses powers of two. Select the largest divisor no larger
            # than the sampled tile that every actual modular leaf can own.
            while value > 1 and any(
                mapping.extents[axis] % value
                or (mapping.split_axis == axis and mapping.period % value)
                for mapping in self.maps
            ):
                value //= 2
            blocks[index] = value
        config["block_sizes"] = blocks


def prove_modular_aux_protocol_subject(
    anchor: torch.fx.Node,
    stores: Sequence[tuple[torch.fx.Node, Tcgen05UnaryEpilogueChain]],
    block_ids: tuple[int, int],
) -> ModularAuxProtocolSubject | None:
    """The caller already proved the complete anchored stores and readonly DAG."""
    from .cute_epilogue import _ModularAuxiliaryTensorLoadExpr

    leaves = tuple(
        leaf
        for _, chain in stores
        for leaf in chain.auxiliary_tensor_loads
        if isinstance(leaf, _ModularAuxiliaryTensorLoadExpr)
    )
    if not leaves or any(
        not isinstance(leaf.mapping, ModularAuxLoadMap) for leaf in leaves
    ):
        return None
    maps = tuple(
        dict.fromkeys(
            leaf.mapping
            for leaf in leaves
            if isinstance(leaf.mapping, ModularAuxLoadMap)
        )
    )
    dtype: torch.dtype | None = None
    for store, _chain in stores:
        target = store.args[0] if store.args else None
        value = target.meta.get("val") if isinstance(target, torch.fx.Node) else None
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or tuple(value.shape) != maps[0].extents
            or tuple(value.stride()) != (maps[0].extents[1], 1)
            or any(mapping.extents != maps[0].extents for mapping in maps)
            or (dtype is not None and dtype != value.dtype)
        ):
            # Mixed-width store plans manufacture an explicit common epi tile.
            # The current modular transport deliberately refuses that protocol;
            # do not promote it into a native-only search with no valid config.
            return None
        dtype = value.dtype
    if dtype is None or not stores:
        return None
    return ModularAuxProtocolSubject(
        anchor, tuple(s for s, _ in stores), maps, block_ids, dtype
    )


def analyze_modular_aux_protocol_subject(
    graphs: Sequence[GraphInfo],
    anchor: torch.fx.Node,
    block_ids: tuple[int, int],
) -> tuple[bool, ModularAuxProtocolSubject | None]:
    """Distinguish no modular subject from an uncomposed modular-store protocol."""
    from .aux_tensor import analyze_tcgen05_matmul_store_chains
    from .cute_epilogue import _ModularAuxiliaryTensorLoadExpr

    stores = analyze_tcgen05_matmul_store_chains(graphs, anchor)
    if stores is None or not any(
        isinstance(leaf, _ModularAuxiliaryTensorLoadExpr)
        for _, chain in stores
        for leaf in chain.auxiliary_tensor_loads
    ):
        return False, None
    return True, prove_modular_aux_protocol_subject(anchor, stores, block_ids)
