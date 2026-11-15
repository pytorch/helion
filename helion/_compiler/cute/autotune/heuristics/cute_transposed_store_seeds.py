from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from .....autotuner.config_spec import _CUTE_LANE_UNROLL_CHOICES
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...transposed_stores import TransposedStore
from ...transposed_stores import transposed_stores
from .cute import _seq_config_list

if TYPE_CHECKING:
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR


class CuteTransposedStoreSeedsHeuristic(AutotunerHeuristic):
    """Seed the layouts that coalesce both sides of a transposed tile store.

    Without them the layout that coalesces the load scatters the transposed
    store, and the search settles on a compromise (``pad_vocab_and_transpose``
    of better-benchmark's ``sum_6119d597d188``: threads along the rows, so
    each warp load reads 16 rows).  The seeds run thread axis 0 along the
    store's other dim, which the loads and the untransposed stores have
    contiguous (``SEED_LANE_THREADS`` threads), give every thread vectors of
    the store's contiguous dim (``SEED_PACKET_BYTES``, a whole sector of the
    transposed rows, ``SEED_ROW_THREADS`` threads along it; a 1-byte dtype
    gets the widest vector width, 16 elements, half a sector) and issue every
    lane's loads first (``cute_lane_unroll``, ``cute/unroll_lane_loads.py``):
    small CTAs that each keep many loads in flight, each warp store writing
    whole sectors of 32 rows.  ``SEED_SHAPES`` lists the tiles, as columns
    of the other dim and bytes of the contiguous one: the first holds 64
    loads per thread, the second the same loads over twice the rows.  Only a
    rank-2 store qualifies (``transposed_stores``): a batched transpose
    tiled ``[1, rows, cols]`` gets no seeds.
    """

    name = "cute_transposed_store_seeds"
    backend = "cute"
    priority = 98

    SEED_LANE_THREADS = 64
    SEED_ROW_THREADS = 2
    SEED_PACKET_BYTES = 32
    SEED_SHAPES = ((256, 64), (128, 128))

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(cls._stores(env, device_ir))

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        stores = cls._stores(env, device_ir)
        if not stores:
            return []
        seeds = []
        for columns, row_bytes in cls.SEED_SHAPES:
            seed = cls._seed(env, stores[0], columns, row_bytes)
            if seed not in seeds:
                seeds.append(seed)
        return seeds

    @staticmethod
    def _stores(env: CompileEnvironment, device_ir: DeviceIR) -> list[TransposedStore]:
        """The transposed stores whose two blocks the config tiles, threads
        and vectorizes, and one loop orders (the seeds lead with the other
        dim; tiles of nested loops have no common order and no common
        lanes to unroll)."""
        spec = env.config_spec
        tiled = (
            set(spec.block_sizes.valid_block_ids())
            & set(spec.num_threads.valid_block_ids())
            & set(spec.cute_vector_widths.valid_block_ids())
        )
        ordered = [set(item.block_ids) for item in spec.loop_orders]
        return [
            store
            for store in transposed_stores(env, device_ir)
            if {store.other, store.contiguous} <= tiled
            and {store.other, store.contiguous} in ordered
        ]

    @classmethod
    def _seed(
        cls,
        env: CompileEnvironment,
        store: TransposedStore,
        want_columns: int,
        row_bytes: int,
    ) -> Config:
        spec = env.config_spec
        blocks = {store.other, store.contiguous}

        def size(block_id: int, want: int) -> int:
            fragment = spec.block_sizes.block_id_lookup(block_id)._fragment(spec)
            value = max(fragment.low, min(fragment.high, want))
            return 1 << (value.bit_length() - 1)

        rows = size(store.contiguous, row_bytes // store.itemsize)
        columns = size(store.other, want_columns)
        # The kernel's other tiles (another loop's) keep their defaults.
        sizes = {
            block_id: cast(
                "int",
                spec.block_sizes.block_id_lookup(block_id)._fragment(spec).default(),
            )
            for block_id in spec.block_sizes.valid_block_ids()
        }
        sizes.update({store.other: columns, store.contiguous: rows})
        lane_threads = min(columns, cls.SEED_LANE_THREADS)
        row_threads = min(rows, cls.SEED_ROW_THREADS)
        # The widest vector the config spec admits: a 1-byte dtype gets 16
        # elements, half a sector, as no width reaches 32.
        want = min(rows // row_threads, cls.SEED_PACKET_BYTES // store.itemsize)
        widths = spec.cute_vector_widths.block_id_lookup(store.contiguous)
        choices = cast("tuple[int, ...]", widths._fragment(spec).choices)
        packet = max(width for width in choices if width <= want)
        values: dict[str, object] = {
            "block_sizes": _seq_config_list(spec.block_sizes, sizes),
            "num_threads": _seq_config_list(
                spec.num_threads,
                {store.other: lane_threads, store.contiguous: row_threads},
            ),
            "cute_vector_widths": _seq_config_list(
                spec.cute_vector_widths, {store.other: 1, store.contiguous: packet}
            ),
            # Every column lane's loads first, each vector loop's scalar loads
            # split from its stores; a wider tile than the seeds ask for (a
            # registered block size) takes the largest factor there is, and
            # one column lane per thread has nothing to unroll.
            "cute_lane_unroll": min(
                _CUTE_LANE_UNROLL_CHOICES[-1], max(1, columns // lane_threads)
            ),
        }
        # Every tile with a vector-width slot has a lane-layout slot.
        values["cute_lane_layouts"] = _seq_config_list(
            spec.cute_lane_layouts, dict.fromkeys(blocks, "strided")
        )
        # Thread axis 0 runs along the other dim: it leads the loop order
        # (``_stores`` keeps the stores one loop order covers).
        orders = {}
        for item in spec.loop_orders:
            if set(item.block_ids) == blocks:
                first = item.block_ids.index(store.other)
                orders[item.block_id] = [first, 1 - first]
        values["loop_orders"] = _seq_config_list(spec.loop_orders, orders)
        return Config.from_dict(values)
