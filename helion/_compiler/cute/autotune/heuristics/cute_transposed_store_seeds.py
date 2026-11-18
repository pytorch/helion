from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from .....autotuner.config_spec import _CUTE_LANE_UNROLL_CHOICES
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...emitters.lane_packets import _CUTE_VECTOR_DTYPES
from ...flat_gather_lane import _packet_layout_aligned
from ...target_facts import widest_packet_bytes
from ...transposed_stores import BatchedLayoutStore
from ...transposed_stores import TransposedStore
from ...transposed_stores import batched_layout_stores
from ...transposed_stores import root_tile_begins_at_multiple
from ...transposed_stores import transposed_stores
from .cute import _seq_config_list

if TYPE_CHECKING:
    from .....autotuner.config_spec import ConfigSpec
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
    rank-2 store qualifies for these (``transposed_stores``).

    A batched layout change (``batched_layout_stores``: a store of rank 3 or
    more whose contiguous tile is not the one a load of the same tiles has
    contiguous, NCHW to channels-last and back) gets seeds of its own, not
    the rank-2 shape: with many column lanes per thread and
    ``cute_lane_unroll`` the copy of better-benchmark's
    ``pointwise_06377502b8f5`` spills its lanes to the stack.  Every thread
    takes ONE packet of one side's contiguous tile and one element of the
    other side's, and the threads run along the store's contiguous tile
    first, then along the load's:

    * ``BATCHED_STORE_SHAPES``, packets of the store's tile: 32 bytes, or 16
      bytes beside the packets of the threads next to it, so a warp stores
      whole sectors in runs of 64 bytes and each of its loads reads 16, 32 or
      8 consecutive elements of two or four rows;
    * ``BATCHED_LOAD_SHAPES``, packets of the load's tile, for the stores
      that take none (the 289-element rows a channels-last activation is
      stored into as ``[N, C, 17 * 17]``): a warp of threads along the
      store's tile stores 32 consecutive elements with each of its scalar
      stores.

    Both are planted where the lowering will prove them (``_batched_seed``).
    The batch tiles take their smallest block with a thread per element,
    lanes are blocked and the store policy is the default one: on that copy
    and on larger and smaller ones ``cute_store_policy="streaming"`` slows
    the packet stores down.  The other compiler seeds vectorize the
    innermost tile whichever side has it contiguous, and are slower there
    than the default config.
    """

    name = "cute_transposed_store_seeds"
    backend = "cute"
    priority = 98

    SEED_LANE_THREADS = 64
    SEED_ROW_THREADS = 2
    SEED_PACKET_BYTES = 32
    SEED_SHAPES = ((256, 64), (128, 128))
    # A batched layout change, with the packets on the store's contiguous
    # tile and on the load's: threads along the packet tile, bytes of the
    # packet each of them takes, threads along the other tile (one element
    # each).  Two threads of 32-byte stores and four of 16-byte ones both
    # store 64 contiguous bytes per warp row; 32 threads of scalar stores do
    # too.
    BATCHED_STORE_SHAPES = ((2, 32, 16), (2, 32, 32), (4, 16, 8))
    BATCHED_LOAD_SHAPES = ((1, 32, 32), (2, 32, 32), (2, 16, 32))
    # The threads of a seed's CTA: the batch tiles get what the two lane
    # tiles leave of it.
    BATCHED_SEED_CTA_THREADS = 512

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(cls._stores(env, device_ir) or cls._batched_stores(env, device_ir))

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
        seeds: list[Config] = []
        stores = cls._stores(env, device_ir)
        if stores:
            for columns, row_bytes in cls.SEED_SHAPES:
                seed = cls._seed(env, stores[0], columns, row_bytes)
                if seed not in seeds:
                    seeds.append(seed)
        batched_stores = cls._batched_stores(env, device_ir)
        if batched_stores:
            host_function = device_ir.host_function
            assert host_function is not None
            # The packet proof reads the stored tensor's provenance from the
            # host function, which codegen has current and the seed hooks do
            # not.
            with host_function:
                for batched in batched_stores:
                    for stored, shapes in (
                        (True, cls.BATCHED_STORE_SHAPES),
                        (False, cls.BATCHED_LOAD_SHAPES),
                    ):
                        for shape in shapes:
                            seed = cls._batched_seed(
                                env, device_ir, batched, stored, *shape
                            )
                            if seed is not None and seed not in seeds:
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

    @staticmethod
    def _batched_stores(
        env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[BatchedLayoutStore]:
        """The batched layout stores whose blocks the config tiles, threads
        and vectorizes and one loop orders, as for ``_stores``: the tiles of
        one ``hl.tile``."""
        spec = env.config_spec
        tiled = (
            set(spec.block_sizes.valid_block_ids())
            & set(spec.num_threads.valid_block_ids())
            & set(spec.cute_vector_widths.valid_block_ids())
        )
        ordered = [set(item.block_ids) for item in spec.loop_orders]
        return [
            store
            for store in batched_layout_stores(env, device_ir)
            if (blocks := {store.other, store.contiguous, *store.batch}) <= tiled
            and blocks in ordered
        ]

    @staticmethod
    def _block_size(spec: ConfigSpec, block_id: int, want: int) -> int:
        """The power of two nearest below ``want`` clamped to the sizes the
        search gives the block."""
        fragment = spec.block_sizes.block_id_lookup(block_id)._fragment(spec)
        value = max(fragment.low, min(fragment.high, want))
        return 1 << (value.bit_length() - 1)

    @staticmethod
    def _default_block_sizes(spec: ConfigSpec) -> dict[int, int]:
        """The default size of every block the config tiles."""
        return {
            block_id: cast(
                "int",
                spec.block_sizes.block_id_lookup(block_id)._fragment(spec).default(),
            )
            for block_id in spec.block_sizes.valid_block_ids()
        }

    @classmethod
    def _batched_seed(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
        store: BatchedLayoutStore,
        stored: bool,
        packet_threads: int,
        packet_bytes: int,
        element_threads: int,
    ) -> Config | None:
        """The seed of one of ``BATCHED_STORE_SHAPES`` (``stored``: the
        packets are the store's) or ``BATCHED_LOAD_SHAPES`` (the load's), or
        None where the lowering would not take its packets as packets.  A
        tile's lanes become one packet per thread
        (``memory_ops._cute_vector_load_ctx``) for a bf16, fp16 or fp32
        tensor (``_CUTE_VECTOR_DTYPES``: the dtypes it stores as packets;
        the seeds ask the same of a load), no wider than the target's
        widest (``widest_packet_bytes``: 32 bytes from SM100 on), at a
        vector width the config spec admits, when the tile is a root
        loop's, begins at a static multiple of the packet
        (``root_tile_begins_at_multiple``) and has a static multiple of it
        as its extent (no packet straddles the tensor's edge), and the
        tensor's base and the strides of its other dims are multiples of
        the packet (``_packet_layout_aligned``, the seed-time reading of
        the lowering's proof): the 289-element rows of a ``[N, C, 17 * 17]``
        buffer take no packets, the tiles of ``hl.tile([0, 8, 0], ...)``
        take 16-byte ones only, and a kernel of dynamic shapes takes none.
        A store with an ``extra_mask`` stays scalar, and a load with one
        keeps its packet only where the mask is uniform over it
        (``memory_ops._cute_lane_uniform_mask``), which a seed does not
        prove: neither gets packets of its own.  The same for a subscript
        that adds a scalar to a tile's index (``tile_c.index + 3``), a
        coordinate that lowering admits for loads only: a store with one
        on any dim keeps its lanes scalar whatever the scalar, and a load
        with one on its contiguous dim keeps the packets the scalar leaves
        aligned.  A tile shorter than the packet, or one whose smallest
        block needs more threads than the CTA has, gets no seed either."""
        spec = env.config_spec
        if stored:
            packet_tile, element_tile = store.contiguous, store.other
            tensor, lane_dim = store.tensor, store.lane_dim
            masked = store.masked
            shifted = store.shifted
        else:
            packet_tile, element_tile = store.other, store.contiguous
            tensor, lane_dim = store.load_tensor, store.load_lane_dim
            masked = store.load_masked
            shifted = store.load_shifted
        packet = packet_bytes // tensor.dtype.itemsize
        widths = spec.cute_vector_widths.block_id_lookup(packet_tile)
        if (
            masked
            or shifted
            or tensor.dtype not in _CUTE_VECTOR_DTYPES
            or packet_bytes > widest_packet_bytes(spec.target_device_capability)
            or packet not in cast("tuple[int, ...]", widths._fragment(spec).choices)
            or not root_tile_begins_at_multiple(env, device_ir, packet_tile, packet)
            or not env.block_sizes[packet_tile].known_multiple(packet)
            or not _packet_layout_aligned(env, tensor, lane_dim, packet)
        ):
            return None
        # A larger smallest block takes more threads: every thread keeps one
        # packet of the one tile and one element of the other.
        rows = cls._block_size(spec, packet_tile, packet_threads * packet)
        columns = cls._block_size(spec, element_tile, element_threads)
        lane_threads = rows // packet * columns
        if rows % packet or lane_threads > cls.BATCHED_SEED_CTA_THREADS:
            return None
        # The kernel's other tiles (another loop's) keep their defaults.
        sizes = cls._default_block_sizes(spec)
        sizes.update({element_tile: columns, packet_tile: rows})
        threads = {element_tile: columns, packet_tile: rows // packet}
        budget = cls.BATCHED_SEED_CTA_THREADS // lane_threads
        for block_id in store.batch:
            # The smallest block the search gives a batch tile, a thread per
            # element while the CTA has them.
            sizes[block_id] = cls._block_size(spec, block_id, 1)
            threads[block_id] = min(sizes[block_id], budget)
            budget //= threads[block_id]
        blocks = {store.other, store.contiguous, *store.batch}
        orders = {}
        for item in spec.loop_orders:
            if set(item.block_ids) == blocks:
                # Thread axis 0 runs along the store's contiguous tile, axis 1
                # along the load's, the batch tiles after them.
                lead = [
                    item.block_ids.index(store.contiguous),
                    item.block_ids.index(store.other),
                ]
                orders[item.block_id] = [
                    *lead,
                    *(i for i in range(len(item.block_ids)) if i not in lead),
                ]
        values: dict[str, object] = {
            "block_sizes": _seq_config_list(spec.block_sizes, sizes),
            "num_threads": _seq_config_list(spec.num_threads, threads),
            "cute_vector_widths": _seq_config_list(
                spec.cute_vector_widths,
                dict.fromkeys(blocks, 1) | {packet_tile: packet},
            ),
            "cute_lane_layouts": _seq_config_list(
                spec.cute_lane_layouts, dict.fromkeys(blocks, "blocked")
            ),
            "loop_orders": _seq_config_list(spec.loop_orders, orders),
        }
        if spec.supports_config_key("cute_store_policy"):
            values["cute_store_policy"] = "default"
        return Config.from_dict(values)

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
        rows = cls._block_size(spec, store.contiguous, row_bytes // store.itemsize)
        columns = cls._block_size(spec, store.other, want_columns)
        # The kernel's other tiles (another loop's) keep their defaults.
        sizes = cls._default_block_sizes(spec)
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
