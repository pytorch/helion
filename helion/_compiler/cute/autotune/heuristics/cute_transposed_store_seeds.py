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
from ...transposed_stores import transposed_accesses
from ...transposed_stores import transposed_stores
from .cute import _seq_config_list

if TYPE_CHECKING:
    import torch

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

    Those shapes pay where the store takes its packets.  Where a load the
    store transposes takes packets (``transposed_accesses``;
    ``_load_packet_seed`` says when), one more seed, ``LOAD_PACKET_SHAPE``,
    lays the threads out the other way: two threads along the load's
    contiguous dim on thread axis 0, each with four 32-byte packets of it,
    the two threads' packets side by side (strided lanes), and 32 threads
    along the store's contiguous dim, one element each, 16 of them to a
    warp.  Every store instruction of a warp then writes whole 32-byte
    sectors: the transposed store's scalars 16 consecutive two-byte elements
    for each of the two threads, and an untransposed store of the tile,
    which takes the load's lanes as packets (16 bytes of bf16 beside an
    fp32 load), two packets side by side.  The case is the tied LM-head
    weight of better-benchmark's ``pointwise_52fe1925ccda``, fp32
    ``[50265, 1024]`` cast to bf16 and stored as is and transposed: no
    packet divides the 50265 elements of a transposed row, so the shapes
    above render scalar loads and stores there, and a search that starts
    from them settles on the rows on thread axis 0, a third slower than
    this seed.  The thread layout carries it.  One packet thread (the
    untransposed packets no longer pair up), blocked lanes (they are not
    side by side) and four or more packet threads (fewer than 16 rows left
    to a warp) each cost a quarter or more; the packets per thread, the
    rows per CTA, the lane unroll, a CTAs-per-SM minimum and the store
    policy move it by about a percent.  The seed names no load policy: the
    weight is larger than L2, which no policy keeps resident, and the
    choice is left to the search.

    Which seed leads follows the side that takes packets.  Where the
    transposed rows take none, this seed was the fastest of the kernel's
    seeds by a wide margin on the kernels timed; where they do, the shapes
    above stayed a few percent ahead, and this seed is one more candidate.

    This seed and the batched ones below are planted where a proof says
    their packets are packets (``_packet``), and a config renders such a
    layout only in a kernel whose CTA is the seed's tiles' alone.  They
    share one admission by the kernel's structure (``_layout_holds``): a
    pointwise kernel, the seed's tiles the only blocks with threads, no
    store into a tensor the kernel loads.  Their accesses are read from the
    graphs in which the lowering takes packets (``transposed_accesses``): a
    root loop's, and a branch's that codegen folds away, not a branch on a
    tensor's value or a nested loop.

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

    Both are planted where the lowering will prove them (``_batched_seed``)
    in a kernel that renders their layout (``_layout_holds``).
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
    # A transposed store whose load takes the packets: threads along the
    # load's contiguous tile, bytes of a packet, packets per thread, threads
    # along the store's contiguous tile (one element each).
    LOAD_PACKET_SHAPE = (2, 32, 4, 32)
    # A batched layout change, with the packets on the store's contiguous
    # tile and on the load's: threads along the packet tile, bytes of the
    # packet each of them takes, threads along the other tile (one element
    # each).  Two threads of 32-byte stores and four of 16-byte ones both
    # store 64 contiguous bytes per warp row; 32 threads of scalar stores do
    # too.
    BATCHED_STORE_SHAPES = ((2, 32, 16), (2, 32, 32), (4, 16, 8))
    BATCHED_LOAD_SHAPES = ((1, 32, 32), (2, 32, 32), (2, 16, 32))
    # The most threads of the CTA of a seed planted on a packet proof: a
    # batched seed's batch tiles get what its two lane tiles leave of it.
    SEED_CTA_THREADS = 512

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
        host_function = device_ir.host_function
        if stores:
            for columns, row_bytes in cls.SEED_SHAPES:
                seed = cls._seed(env, stores[0], columns, row_bytes)
                if seed not in seeds:
                    seeds.append(seed)
            assert host_function is not None
            # The packet proof reads the loaded and stored tensors'
            # provenance from the host function, which codegen has current
            # and the seed hooks do not.
            with host_function:
                seed = cls._load_packet_seed(env, device_ir, stores[0])
            if seed is not None and seed not in seeds:
                seeds.append(seed)
        batched_stores = cls._batched_stores(env, device_ir)
        if batched_stores:
            assert host_function is not None
            # As above, for the stored and loaded tensors.
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

    @classmethod
    def _batched_stores(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[BatchedLayoutStore]:
        """The batched layout stores whose blocks the config tiles, threads
        and vectorizes and one loop orders, as for ``_stores``: the tiles of
        one ``hl.tile``, in a kernel that renders the seeds' layout
        (``_layout_holds``)."""
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
            and cls._layout_holds(spec, blocks)
        ]

    @staticmethod
    def _layout_holds(spec: ConfigSpec, blocks: set[int]) -> bool:
        """Whether a config that gives the tiles ``blocks`` a seed's threads
        and packets renders that layout, by the kernel's structure: the
        admission of the seeds whose packets a proof admits (``_packet``),
        the batched ones and ``_load_packet_seed``'s.

        - The kernel is pointwise (``pointwise_facts``).  Beside a reduction
          of the tile's lanes such a config does not render.  The rule is
          wider than that on purpose: a kernel with an accumulator, a scan,
          an atomic or inline code is not pointwise either and was not
          measured (a tile-wide atomic beside the copy keeps the layouts).
        - ``blocks`` are the only blocks with threads.  A reduction dim or
          the untiled dim of ``y[tile, :]`` takes thread axis 0 (a batched
          seed's CTA grows to 1024 threads, and the two packet threads of
          ``LOAD_PACKET_SHAPE`` are left axis 1); beside another root
          loop's 16 threads the packets of a tile of two or four threads
          render as scalar lanes.
        - The kernel stores into no tensor it loads.  The code generator
          refuses a tensor loaded as packets and stored transposed in one
          lane loop, whichever load or store of the kernel that is; a
          store of the elements just loaded renders the layout and is
          declined with it on purpose, as no in-place kernel was measured.
        """
        names: dict[str, set[str | None]] = {"load": set(), "store": set()}
        for fact in spec.memory_op_facts:
            names[fact.kind].add(fact.tensor_name)
        return (
            bool(spec.pointwise_facts)
            and set(spec.num_threads.valid_block_ids()) == blocks
            and names["load"].isdisjoint(names["store"])
        )

    @staticmethod
    def _block_size(spec: ConfigSpec, block_id: int, want: int) -> int:
        """The power of two nearest below ``want`` clamped to the sizes the
        search gives the block."""
        fragment = spec.block_sizes.block_id_lookup(block_id)._fragment(spec)
        value = max(fragment.low, min(fragment.high, want))
        return 1 << (value.bit_length() - 1)

    @staticmethod
    def _other_tile_first(spec: ConfigSpec, store: TransposedStore) -> list[object]:
        """The loop orders with the store's other tile first: thread axis 0
        runs along it (``_stores`` keeps the stores one loop order covers)."""
        blocks = {store.other, store.contiguous}
        orders = {}
        for item in spec.loop_orders:
            if set(item.block_ids) == blocks:
                first = item.block_ids.index(store.other)
                orders[item.block_id] = [first, 1 - first]
        return _seq_config_list(spec.loop_orders, orders)

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
        None where the lowering would not take its packets as packets
        (``_packet``).  A tile shorter than the packet, or one whose
        smallest block needs more threads than the CTA has, gets no seed
        either."""
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
        packet = cls._packet(
            env, device_ir, tensor, lane_dim, packet_tile, packet_bytes, masked, shifted
        )
        if packet is None:
            return None
        # A larger smallest block takes more threads: every thread keeps one
        # packet of the one tile and one element of the other.
        rows = cls._block_size(spec, packet_tile, packet_threads * packet)
        columns = cls._block_size(spec, element_tile, element_threads)
        lane_threads = rows // packet * columns
        if rows % packet or lane_threads > cls.SEED_CTA_THREADS:
            return None
        # A size for every block: a block size has no default for
        # ``_seq_config_list`` to fill in.
        sizes = cls._default_block_sizes(spec)
        sizes.update({element_tile: columns, packet_tile: rows})
        threads = {element_tile: columns, packet_tile: rows // packet}
        budget = cls.SEED_CTA_THREADS // lane_threads
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

    @staticmethod
    def _packet(
        env: CompileEnvironment,
        device_ir: DeviceIR,
        tensor: torch.Tensor,
        lane_dim: int,
        packet_tile: int,
        packet_bytes: int,
        masked: bool,
        shifted: bool,
    ) -> int | None:
        """The elements of a ``packet_bytes`` packet of ``tensor`` along
        ``lane_dim``, the dim the tile ``packet_tile`` indexes, or None where
        the lowering would not take the tile's lanes as packets there, for
        a load or store with or without an ``extra_mask`` (``masked``) or a
        tile's index plus a scalar as a subscript (``shifted``).

        A tile's lanes become one packet per thread
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
        aligned.
        """
        spec = env.config_spec
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
        return packet

    @classmethod
    def _load_packet_seed(
        cls, env: CompileEnvironment, device_ir: DeviceIR, store: TransposedStore
    ) -> Config | None:
        """The seed of ``LOAD_PACKET_SHAPE`` for the transposed ``store``, or
        None where its config would not render the layout it names:

        - the store transposes a load in a graph where the lowering takes
          packets (``transposed_accesses``: a store that only another store
          witnesses has none), and every such load takes the packets, of
          one width, as the lowering takes them (``_packet``: a masked
          load, one shifted along its contiguous dim, an int32 or fp64 one,
          a row no packet divides and a kernel of dynamic shapes get none);
        - every store of the tile the other way round takes the loads'
          lanes as one packet of its own dtype (16 bytes of bf16 beside an
          fp32 load), by the same proof;
        - the kernel renders the layout (``_layout_holds``): pointwise, the
          two tiles the only blocks with threads, no store into a tensor it
          loads;
        - the search gives the tiles blocks as large as the seed's.  A
          larger smallest block (``_block_size``) takes more packets per
          thread along the load's tile and more threads along the store's,
          while they fit a CTA of ``SEED_CTA_THREADS``; a tile whose
          largest block is smaller than the seed's (an extent of half the
          seed's tile or less) gets no seed.

        The seed names tiles, threads, vector widths, lane layouts and the
        loop order, and opens no knob of the search."""
        spec = env.config_spec
        if not cls._layout_holds(spec, {store.other, store.contiguous}):
            return None
        packet_threads, packet_bytes, lanes, element_threads = cls.LOAD_PACKET_SHAPE
        accesses = transposed_accesses(device_ir, store)
        packets = {
            cls._packet(
                env,
                device_ir,
                load.tensor,
                load.lane_dim,
                store.other,
                packet_bytes,
                load.masked,
                load.shifted,
            )
            for load in accesses
            if not load.stored
        }
        # No load, a load that takes no packets, or loads of two widths.
        if len(packets) != 1 or None in packets:
            return None
        (packet,) = cast("set[int]", packets)
        # A store of the tile the other way round takes the loads' lanes as
        # one packet of its own dtype.
        if any(
            cls._packet(
                env,
                device_ir,
                stored.tensor,
                stored.lane_dim,
                store.other,
                packet * stored.tensor.dtype.itemsize,
                stored.masked,
                stored.shifted,
            )
            != packet
            for stored in accesses
            if stored.stored
        ):
            return None
        # A larger smallest block takes more: more packets per thread along
        # the load's tile, more threads along the store's.
        want = packet_threads * lanes * packet
        columns = cls._block_size(spec, store.other, want)
        rows = cls._block_size(spec, store.contiguous, element_threads)
        if (
            columns < want
            or rows < element_threads
            or packet_threads * rows > cls.SEED_CTA_THREADS
        ):
            return None
        # A size for every block, as in ``_batched_seed``.
        sizes = cls._default_block_sizes(spec)
        sizes.update({store.other: columns, store.contiguous: rows})
        return Config.from_dict(
            {
                "block_sizes": _seq_config_list(spec.block_sizes, sizes),
                "num_threads": _seq_config_list(
                    spec.num_threads,
                    {store.other: packet_threads, store.contiguous: rows},
                ),
                "cute_vector_widths": _seq_config_list(
                    spec.cute_vector_widths,
                    {store.other: packet, store.contiguous: 1},
                ),
                # Strided: the two threads' packets alternate along the tile.
                "cute_lane_layouts": _seq_config_list(
                    spec.cute_lane_layouts,
                    {store.other: "strided", store.contiguous: "blocked"},
                ),
                # Thread axis 0 runs along the load's contiguous tile.
                "loop_orders": cls._other_tile_first(spec, store),
            }
        )

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
        values["loop_orders"] = cls._other_tile_first(spec, store)
        return Config.from_dict(values)
