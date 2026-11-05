"""A common FP32 accumulator coordinate contract for mixed-width output stores.

This is deliberately separate from shared-expression fanout. Each output keeps
its original expression, conversion, R2S layout, SMEM allocation and descriptor.
Only the unrounded accumulator's T2R coordinates are common to the stores.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from .strategies import tcgen05_epilogue_subtile_m
from .strategies import tcgen05_explicit_epilogue_tile_supported
from .tcgen05_constants import TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES
from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_c_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_fixed_smem_overhead_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from .cute_epilogue import Tcgen05UnaryEpilogueChain


@dataclass(frozen=True)
class MixedPrecisionStorePlan:
    """Original store identities/types sharing an FP32 T2R coordinate space."""

    stores: tuple[Node, ...]
    dtypes: tuple[torch.dtype, ...]
    input_dtype: torch.dtype
    device: torch.device

    def dtype_for(self, store: Node) -> torch.dtype | None:
        for original, dtype in zip(self.stores, self.dtypes, strict=True):
            if original is store:
                return dtype
        return None

    def default_tile(
        self, bm: int, bn: int, is_two_cta: bool
    ) -> tuple[int, int, int] | None:
        # N=32 is the existing 32-bit output subtile; unlike implicit output
        # tiling its integer coordinates do not change when D is 16-bit.
        tile_n = min(bn, 32)
        tile = (
            tcgen05_epilogue_subtile_m(bm=bm, is_two_cta=is_two_cta),
            tile_n,
            tile_n,
        )
        return (
            tile
            if tcgen05_explicit_epilogue_tile_supported(
                bm=bm, bn=bn, is_two_cta=is_two_cta, tile_shape=tile
            )
            else None
        )

    def seed_smem_upper_bound(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        input_dtype_bytes: int,
        ab_stages: int,
        c_stages: int,
        acc_stages: int = 2,
    ) -> int | None:
        """Price the simple ONE-CTA seed with the existing allocation model.

        All original outputs are priced separately, even when a masked SIMT
        store needs no C ring; this is an upper bound, never a free output.
        Seeds use post-wait GMEM auxiliaries and no extra warp/ring protocols.
        """
        tile = self.default_tile(bm, bn, False)
        if tile is None:
            return None
        tile_m, tile_n, _ = tile
        return (
            tcgen05_ab_smem_bytes_per_cta(
                bm=bm,
                bn=bn,
                bk=bk,
                dtype_bytes=input_dtype_bytes,
                ab_stages=ab_stages,
                cluster_m=1,
            )
            + sum(
                tcgen05_c_smem_bytes_per_cta(
                    epi_tile_m=tile_m,
                    epi_tile_n=tile_n,
                    dtype_bytes=dtype.itemsize,
                    c_stages=c_stages,
                )
                for dtype in self.dtypes
            )
            + tcgen05_fixed_smem_overhead_bytes(
                ab_stages=ab_stages, acc_stages=acc_stages, cluster_m=1
            )
            + TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES
        )


def prove_mixed_precision_stores(
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    stores: Sequence[tuple[Node, Tcgen05UnaryEpilogueChain]],
    *,
    input_dtype: torch.dtype,
    input_device: torch.device,
    m: int | torch.SymInt,
    n: int | torch.SymInt,
    input_loads: tuple[Node, Node],
) -> MixedPrecisionStorePlan | None:
    """Prove bounded ordinary row-major stores without borrowing fake aliases.

    The caller has already proved each complete epilogue and exact M/N store
    indices. This adds the mixed-width contract: identical rank-2 geometry,
    half/F32 final outputs, read-only operands, and disjoint destinations. It
    neither registers input classifiers nor publishes construction snapshots.
    """
    from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint
    from .register_loads import host_load_is_readonly

    if input_dtype not in (torch.float16, torch.bfloat16) or not 2 <= len(stores) <= 4:
        return None
    allowed = {input_dtype, torch.float32}
    destinations: list[torch.Tensor] = []
    dtypes: list[torch.dtype] = []
    loads = set(input_loads)
    for store, chain in stores:
        target = store.args[0] if store.args else None
        fake = target.meta.get("val") if isinstance(target, Node) else None
        if (
            not isinstance(fake, torch.Tensor)
            or fake.ndim != 2
            or fake.dtype not in allowed
            or fake.device != input_device
        ):
            return None
        if not (
            env.known_equal(fake.shape[0], m)
            and env.known_equal(fake.shape[1], n)
            # The initial mixed-width contract uses compact rows; this also
            # rejects internally overlapping/broadcast output views.
            and env.known_equal(fake.stride()[0], n)
            and env.known_equal(fake.stride()[1], 1)
        ):
            return None
        # Device scalars have their own lifetime/broadcast protocol; retaining
        # this initial contract to tensor-tile auxiliaries avoids adding one.
        if any(not leaf.is_host_scalar for leaf in chain.runtime_scalars):
            return None
        loads.update(leaf.load_node for leaf in chain.auxiliary_tensor_loads)
        destinations.append(fake)
        dtypes.append(fake.dtype)
    if set(dtypes) != allowed:
        return None
    allow_unbound = env.early_mma_metadata_scope_active
    if not all(
        host_load_is_readonly(load, env, graphs, allow_unbound=allow_unbound)
        for load in loads
    ):
        return None
    for left, right in itertools.combinations(destinations, 2):
        if left.untyped_storage() == right.untyped_storage():
            return None
        if not (
            env.tensor_storage_is_compiler_allocated(left)
            or env.tensor_storage_is_compiler_allocated(right)
            or runtime_tensors_are_proven_disjoint(
                env, left, right, allow_unbound=allow_unbound
            )
        ):
            return None
    return MixedPrecisionStorePlan(
        tuple(store for store, _ in stores), tuple(dtypes), input_dtype, input_device
    )
