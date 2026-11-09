from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
from typing import Hashable
from typing import Literal
from typing import TypeVar
from typing import cast
import uuid


@dataclass(frozen=True, eq=False)
class _ConfigObjectIdentity:
    value: object

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ConfigObjectIdentity) and self.value is other.value


def _config_value_key(value: object) -> Hashable:
    """In-memory identity for values that may affect generated code.

    Preserve order in mappings and bits in floating-point values. Hashable
    opaque objects retain their equality contract: this includes complex
    numbers, frozensets, and numeric types that are not Python float subclasses.
    Unhashable list and dict subclasses use identity plus their contents:
    sharing less is safer than
    dropping their custom equality or invoking arbitrary copy hooks. As with
    other opaque tunables, mutation of custom attributes or external state is
    outside the cache contract.
    """
    kind = type(value)
    if kind is float:
        return kind, struct.pack("!d", cast("float", value))
    if isinstance(value, float):
        return kind, struct.pack("!d", value), value
    if kind in (list, tuple):
        return kind, tuple(
            _config_value_key(item)
            for item in cast("list[object] | tuple[object, ...]", value)
        )
    if kind in (dict, OrderedDict):
        return kind, tuple(
            (_config_value_key(key), _config_value_key(item))
            for key, item in cast("dict[object, object]", value).items()
        )
    if kind.__hash__ is None:
        if isinstance(value, list):
            return (
                kind,
                _ConfigObjectIdentity(value),
                tuple(_config_value_key(item) for item in value),
            )
        if isinstance(value, dict):
            return (
                kind,
                _ConfigObjectIdentity(value),
                tuple(
                    (_config_value_key(key), _config_value_key(item))
                    for key, item in value.items()
                ),
            )
    return kind, cast("Hashable", value)


def _config_key(config: Config) -> Hashable:
    """Ignore option order; preserve code-relevant identity within each value."""
    return frozenset(
        (name, _config_value_key(value)) for name, value in config.config.items()
    )


_ConfigValue = TypeVar("_ConfigValue")


def _copy_config_structure(value: _ConfigValue) -> _ConfigValue:
    """Copy known config containers while preserving opaque leaf objects."""
    if type(value) is OrderedDict:
        return cast(
            "_ConfigValue",
            OrderedDict(
                (key, _copy_config_structure(item)) for key, item in value.items()
            ),
        )
    if type(value) is dict:
        return cast(
            "_ConfigValue",
            {key: _copy_config_structure(item) for key, item in value.items()},
        )
    if type(value) is list:
        return cast("_ConfigValue", [_copy_config_structure(item) for item in value])
    if type(value) is tuple:
        return cast(
            "_ConfigValue", tuple(_copy_config_structure(item) for item in value)
        )
    return value


IndexingLiteral = Literal["pointer", "tensor_descriptor", "block_ptr"]
PidTypeLiteral = Literal[
    "flat",
    "xyz",
    "persistent_blocked",
    "persistent_interleaved",
]
CrossLoopPipelineLiteral = Literal["barrier", "static", "dynamic"]
EvictionPolicyLiteral = Literal["", "first", "last"]
LoadCacheModifierLiteral = Literal["", ".cg"]
StoreCacheModifierLiteral = Literal["", ".cs", ".wt"]
CuteAsyncLoadCacheLiteral = Literal["cg", "ca"]
CuteAsyncStorePolicyLiteral = Literal["default", "l2_evict_last"]
CuteStorePolicyLiteral = Literal["default", "streaming", "l2_evict_last"]
CuteExactDivFormLiteral = Literal["redo", "plain"]
CuteAffineScanScheduleLiteral = Literal[
    "ordinary",
    "direct_m16n8_v1",
    "direct_m16n16_v1",
]
CuteHostPairedSumLiteral = Literal["off", "mapped", "narrow"]
CutePdlReleaseLiteral = Literal["auto", "entry", "none", "exit", "last_wave"]
NumSmMultiplierLiteral = int
MaxnregLiteral = int | None


class Config(Mapping[str, object]):
    config: dict[str, object]

    def __init__(
        self,
        *,
        # Core properties
        block_sizes: list[int] | None = None,
        num_threads: list[int] | int | None = None,
        loop_orders: list[list[int]] | None = None,
        flatten_loops: list[bool] | None = None,
        l2_groupings: list[int] | None = None,
        reduction_loops: list[int | None] | None = None,
        range_unroll_factors: list[int] | None = None,
        range_warp_specializes: list[bool | None] | None = None,
        range_num_stages: list[int] | None = None,
        range_multi_buffers: list[bool | None] | None = None,
        range_flattens: list[bool | None] | None = None,
        static_ranges: list[bool] | None = None,
        pallas_load_buffer_count: list[int] | None = None,
        load_eviction_policies: (
            EvictionPolicyLiteral | list[EvictionPolicyLiteral] | None
        ) = None,
        load_cache_modifiers: list[LoadCacheModifierLiteral] | None = None,
        store_cache_modifiers: list[StoreCacheModifierLiteral] | None = None,
        cute_store_policy: CuteStorePolicyLiteral | None = None,
        cute_exact_div_form: CuteExactDivFormLiteral | None = None,
        cute_async_load_stages: int | None = None,
        cute_async_load_lookahead: int | None = None,
        cute_async_load_group_rows: int | None = None,
        cute_async_load_cache: CuteAsyncLoadCacheLiteral | None = None,
        cute_async_store_policy: CuteAsyncStorePolicyLiteral | None = None,
        cute_bf16x2_recurrence: bool | None = None,
        cute_signed_bitfield_bf16: bool | None = None,
        cute_proven_bounds: bool | None = None,
        cute_affine_scan_schedule: CuteAffineScanScheduleLiteral | None = None,
        cute_rng_packet: bool | None = None,
        cute_independent_reduction: bool | None = None,
        cute_replicated_reduction: bool | None = None,
        cute_vector_packet_unroll: bool | None = None,
        cute_vloop_sink: bool | None = None,
        cute_lane_unroll: int | None = None,
        cute_wide_packet_hints: bool | None = None,
        cute_pdl: bool | None = None,
        cute_pdl_release: CutePdlReleaseLiteral | None = None,
        cute_pdl_spread: bool | None = None,
        cute_warp_aggregated_atomics: bool | None = None,
        cute_packet_prefetch: int | None = None,
        cute_reduction_pipeline_depth: int | None = None,
        cute_host_paired_sum: CuteHostPairedSumLiteral | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
        pid_type: PidTypeLiteral | None = None,
        cross_loop_pipeline: CrossLoopPipelineLiteral | None = None,
        num_sm_multiplier: NumSmMultiplierLiteral | None = None,
        maxnreg: MaxnregLiteral = None,
        host_tensor_descriptors: bool | None = None,
        indexing: IndexingLiteral | list[IndexingLiteral] | None = None,
        atomic_indexing: IndexingLiteral | list[IndexingLiteral] | None = None,
        advanced_controls_file: str | None = None,
        epilogue_subtile: int | None = None,
        xcd_remap: bool | None = None,
        reduction_fusion: bool | None = None,
        # For user-defined properties
        **kwargs: object,
    ) -> None:
        """
        Initialize a Config object.

        Args:
            block_sizes: Controls tile sizes for hl.tile invocations.
            num_threads: Target thread count per axis (backend-specific).
            loop_orders: Permutes iteration order of tiles.
            l2_groupings: Reorders program IDs for L2 cache locality.
            reduction_loops: Configures reduction loop behavior.
            range_unroll_factors: Loop unroll factors for tl.range calls.
            range_warp_specializes: Warp specialization for tl.range calls.
            range_num_stages: Number of stages for tl.range calls.
            range_multi_buffers: Controls disallow_acc_multi_buffer for tl.range calls.
            range_flattens: Controls flatten parameter for tl.range calls.
            static_ranges: Whether to use tl.static_range instead tl.range.
            pallas_load_buffer_count: Pallas-only load buffer count (1 or 2) for
                each input tensor. Tensors without an existing DMA route use the
                ordinary path.
            load_eviction_policies: Eviction policies for load operations. A single
                value applies to every load; a list specifies one value per load.
                Valid values are "", "first", and "last". CuTe also supports
                "nc_l2_first" for proven read-only 16-byte or 32-byte packets.
            load_cache_modifiers: Cache modifiers for load operations ("", ".cg").
            store_cache_modifiers: Cache modifiers for store operations ("", ".cs", ".wt").
            cute_store_policy: Cache policy for proven full aligned 16-byte CuTe
                global store packets ("default", "streaming" = evict-first
                ``st.global.cs`` for outputs written once, or "l2_evict_last").
            cute_async_load_stages: Shared-memory ring stages for eligible CuTe
                in-place 16-byte state loads. Zero disables the transformation.
            cute_async_load_lookahead: Async-copy groups kept ahead of compute.
            cute_async_load_group_rows: Per-thread row iterations in each group.
            cute_exact_div_form: How CuTe lowers an fp32 division by a
                loop-invariant divisor (a row sum, a scalar argument); both
                forms give the IEEE quotient bit for bit. ``"redo"`` (default)
                speculates the Markstein sequence over the loop and re-runs
                the loop with ``div.rn`` on a hard case (a loop the redo
                cannot take keeps its plain division), ``"plain"`` keeps one
                ``div.rn`` per element.
            cute_async_load_cache: PTX cp.async cache policy ("cg" or "ca").
            cute_async_store_policy: Cache policy for the matching in-place
                16-byte state store ("default" or "l2_evict_last").
            cute_bf16x2_recurrence: Pack a structurally proven BF16 rank-one
                recurrence into native BF16x2 operations.
            cute_signed_bitfield_bf16: Convert proved signed byte fields to BF16
                using the existing vector load/store packets. Disabled by default.
            cute_proven_bounds: Remove CuTe index guards only when exact launch
                dimensions and cache-specialized tensor sizes prove them true.
            cute_affine_scan_schedule: Physical schedule for a compatible affine
                scan. ``"ordinary"`` disables the direct lowering;
                ``"direct_m16n8_v1"`` and ``"direct_m16n16_v1"`` select the
                measured direct schedule profiles.
            cute_independent_reduction: Sum each vector lane independently in
                FP32 before combining lanes, preserving the element expression.
                This changes summation order. Disabled by default.
            cute_replicated_reduction: Replicate single-use whole-CTA sums in
                each warp using one shared-memory barrier. Disabled by default.
            cute_vector_packet_unroll: Fully unroll small straight-line vector
                packet loops. Disabled by default to retain compact code.
            cute_vloop_sink: Sink a grid constexpr vector loop into the serial
                reduction nest it wraps (column sums): one vector load per row,
                per-lane register accumulators, one grouped cross-thread combine
                per row tile. Disabled by default. Kernels where nothing can be
                sunk generate exactly the knob-off code.
            cute_lane_unroll: Unroll factor (1, 2, 4, 8, 16) for the row lane
                loop of a sunk vector nest; every copy's vector loads issue
                before any copy's accumulates. Only applies with cute_vloop_sink.
            cute_wide_packet_hints: Carry the ``l2_last`` / ``l1_l2_first`` /
                ``l1_l2_last`` load eviction policies onto 32-byte CuTe packets
                (``cute_vector_widths`` 8 for fp32, 16 for bf16/fp16) through
                one 256-bit cache-hinted load; SM100+ with CUDA 13 or newer,
                elsewhere inert. Off by default: a hinted 32-byte packet then
                loads without the hint, as it always has.
            cute_pdl: Launch the kernel as a programmatic dependent of the
                previous kernel in the stream: its launch and prologue overlap
                that kernel's tail and it waits (``griddepcontrol.wait``) ahead
                of its first global memory access; when its own host zero fills
                launch as dependent fill kernels, ahead of its first access to a
                filled tensor instead (the fill kernel releases only after its
                own wait, so every earlier kernel has completed), which lets a
                split-K kernel's operand loads overlap the fill of its atomic
                output. Off by default. Kernels that
                already take part in dependent launch (native plans launched
                with it, bodies with their own wait or release, the
                materialized-fission producer) and pre-sm_90 targets generate
                exactly the knob-off code.
            cute_pdl_release: Where a ``cute_pdl`` kernel releases its own
                dependents (``griddepcontrol.launch_dependents``), which is what
                lets the next kernel of a sequence launch early too. ``entry``
                releases right after the wait, so the dependents' CTAs become
                resident while this kernel runs; ``exit`` releases after the
                body's last statement; ``none`` leaves the implicit trigger at
                grid completion, after which a dependent starts later than a
                plain stream edge would have launched it; ``last_wave`` releases
                at entry in the CTAs of the launch's last wave only (the CTAs
                whose dispatch-order index lies within the resident capacity of
                the grid's end; the earlier CTAs count as released when they
                exit), so the dependents launch as the last wave starts and
                fill the slots its tail frees, which is plain ``entry`` for a
                grid that fits in one wave and keeps the middle waves of a
                larger grid undisturbed. ``auto`` (the default) picks ``entry``
                while the launch grid is known to fit in one wave of resident
                CTAs (the SM count times the CTAs an SM holds, from the launch
                block or ``cute_min_blocks_per_mp``) and ``last_wave`` past it
                or when the grid is sized at runtime; the search leaves the
                placement at ``auto``. Every placement reads the same data: a
                dependent's own wait blocks until this grid has completed and
                flushed. On the tcgen05 GEMM kernels, whose lowering waits in
                its warp roles and releases after the last MMA, ``entry`` and
                ``last_wave`` add the release right after each of those waits
                and ``exit`` / ``none`` keep the late release alone. Inert when
                ``cute_pdl`` is off.
            cute_pdl_spread: Pad a ``cute_pdl`` kernel's shared memory so that
                at most ``ceil(CTAs / SMs)`` of its CTAs fit on one SM, its
                density on an idle device. A programmatic dependent launches
                while its primary runs and the hardware dispatches its CTAs
                onto whatever SMs have room; when the primary's CTAs leave none
                beside them, a small dependent's CTAs all land on the few SMs
                the primary left idle and run packed there once it completes.
                With the pad the CTAs beyond the idle SMs wait for the
                primary's to retire and take their SMs one each. Off by
                default; inert when ``cute_pdl`` is off, when the kernel's own
                allocation already holds its CTAs to that density, when the
                grid is sized at runtime, and on the families that own their
                dependent launch.
            cute_warp_aggregated_atomics: Combine relaxed Int32 zero/one ticket
                reservations for equal addresses within each active warp. Each
                lane retains its own valid fetch-add return value. Off by default.
            cute_packet_prefetch: Prefetch 2, 4, or 8 independent vector packets
                in a proved complete tile; 0 (the default) keeps the original order.
                Requires cute_proven_bounds and storage-disjointness proof.
            cute_reduction_pipeline_depth: Shared-memory ring depth (2 or 4) for
                the proved CuTe pipelined resident-reduction schedule. Defaults
                to 2. Depth 4 prefetches three row groups ahead and requires
                enough shared memory for all four slots and reduction scratch.
            cute_host_paired_sum: Fuse a proved host FP32 sum(0)/cast pair or
                terminal sum/cast. ``"off"`` (the default) retains Torch calls;
                ``"mapped"`` and ``"narrow"`` change independent-column
                ownership while preserving the audited FP32 sum tree. Unknown
                Torch implementations or unsupported bindings use the original
                operations. Available only for typed, independent host pairs
                or a terminal sum/cast with an effect-free return prefix.
            num_warps: Number of warps per block.
            num_stages: Number of stages for software pipelining.
            pid_type: Program ID type strategy ("flat", "xyz", "persistent_blocked", "persistent_interleaved").
            cross_loop_pipeline: Execution strategy for kernels with
                compiler-inferred cross-loop dependencies. ``"barrier"`` uses
                grid synchronization. ``"static"`` and ``"dynamic"`` (Triton
                only) execute the same compiler-derived dependency schedule
                with fixed worker ownership or one-shot packet dispatch,
                respectively. Unsupported kernels reject this field.
            num_sm_multiplier: Positive integer multiplier for the number of SMs
                in persistent kernels. The autotuner searches powers of two, but
                explicit configurations may use intermediate values.
                Controls multi-occupancy by launching N * num_sms thread blocks instead of just num_sms.
            maxnreg: None or a positive integer maximum number of registers per
                thread (1--256). The autotuner searches None, 32, 64, 128, and
                256, but explicit configurations may use intermediate values.
                Lower values allow higher occupancy but may hurt performance.
                Used with persistent kernels to ensure multi-occupancy can be
                achieved.
            host_tensor_descriptors: Construct tensor descriptors once in the host
                launcher instead of independently in every device program. This
                only affects operations using tensor-descriptor indexing.
            indexing: Indexing strategy for load and store operations. Can be:
                - A single strategy string (all loads/stores use this strategy):
                  indexing="block_ptr"  # backward compatible
                - A list of strategies (one per load/store operation, must specify all):
                  indexing=["pointer", "block_ptr", "tensor_descriptor"]
                - Empty/omitted (all loads/stores default to "pointer")
                Valid strategies: "pointer", "tensor_descriptor", "block_ptr"
                ("block_ptr" needs Triton < 3.9; newer Triton lowers it as "pointer")
            atomic_indexing: Indexing strategy for atomic operations (e.g., hl.atomic_add).
                Same format as ``indexing`` (a single string or a list per atomic op).
                Defaults to "pointer" when omitted.
            advanced_controls_file: Path to a PTXAS control file applied during compilation, or empty string for none.
            epilogue_subtile: Split factor for the epilogue (post-matmul pointwise + store) along
                the N dimension. None = disabled (default), valid values are 2 or 4.
            xcd_remap: AMD CDNA only. Remap program IDs into contiguous per-XCD regions to
                improve L2 locality on multi-XCD GPUs (MI300/MI350). Supported for pid_type
                "flat", "persistent_blocked", and "persistent_interleaved"; composes with
                ``l2_groupings``.
            reduction_fusion: When multiple independent reductions in a looped
                (rolled) reduction read the same operand, fold them into one loop
                with a single shared load instead of re-reading the operand per
                reduction. Trades global-memory bandwidth for register pressure,
                so it is autotuned. Defaults to False.
            **kwargs: Additional user-defined configuration parameters.
        """
        self.config = {}
        core_props = {
            "block_sizes": block_sizes,
            "num_threads": num_threads,
            "loop_orders": loop_orders,
            "flatten_loops": flatten_loops,
            "l2_groupings": l2_groupings,
            "reduction_loops": reduction_loops,
            "range_unroll_factors": range_unroll_factors,
            "range_warp_specializes": range_warp_specializes,
            "range_num_stages": range_num_stages,
            "range_multi_buffers": range_multi_buffers,
            "range_flattens": range_flattens,
            "static_ranges": static_ranges,
            "pallas_load_buffer_count": pallas_load_buffer_count,
            "load_eviction_policies": load_eviction_policies,
            "load_cache_modifiers": load_cache_modifiers,
            "store_cache_modifiers": store_cache_modifiers,
            "cute_store_policy": cute_store_policy,
            "cute_exact_div_form": cute_exact_div_form,
            "cute_async_load_stages": cute_async_load_stages,
            "cute_async_load_lookahead": cute_async_load_lookahead,
            "cute_async_load_group_rows": cute_async_load_group_rows,
            "cute_async_load_cache": cute_async_load_cache,
            "cute_async_store_policy": cute_async_store_policy,
            "cute_bf16x2_recurrence": cute_bf16x2_recurrence,
            "cute_signed_bitfield_bf16": cute_signed_bitfield_bf16,
            "cute_proven_bounds": cute_proven_bounds,
            "cute_affine_scan_schedule": cute_affine_scan_schedule,
            "cute_rng_packet": cute_rng_packet,
            "cute_independent_reduction": cute_independent_reduction,
            "cute_replicated_reduction": cute_replicated_reduction,
            "cute_vector_packet_unroll": cute_vector_packet_unroll,
            "cute_vloop_sink": cute_vloop_sink,
            "cute_lane_unroll": cute_lane_unroll,
            "cute_wide_packet_hints": cute_wide_packet_hints,
            "cute_pdl": cute_pdl,
            "cute_pdl_release": cute_pdl_release,
            "cute_pdl_spread": cute_pdl_spread,
            "cute_warp_aggregated_atomics": cute_warp_aggregated_atomics,
            "cute_packet_prefetch": cute_packet_prefetch,
            "cute_reduction_pipeline_depth": cute_reduction_pipeline_depth,
            "cute_host_paired_sum": cute_host_paired_sum,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "indexing": indexing,
            "atomic_indexing": atomic_indexing,
            "pid_type": pid_type,
            "cross_loop_pipeline": cross_loop_pipeline,
            "num_sm_multiplier": num_sm_multiplier,
            "maxnreg": maxnreg,
            "host_tensor_descriptors": host_tensor_descriptors,
            "advanced_controls_file": advanced_controls_file,
            "epilogue_subtile": epilogue_subtile,
            "xcd_remap": xcd_remap,
            "reduction_fusion": reduction_fusion,
        }
        for key, value in core_props.items():
            if value is not None:
                self.config[key] = value
        self.config.update(kwargs)

    def __getitem__(self, key: str) -> object:
        return self.config[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)

    def __repr__(self) -> str:
        return f"helion.{self.__str__()}"

    def __str__(self) -> str:
        args = [f"{key}={value!r}" for key, value in sorted(self.config.items())]
        return f"Config({', '.join(args)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return self.config == other.config

    def __hash__(self) -> int:
        return hash(frozenset([(k, _to_hashable(v)) for k, v in self.config.items()]))

    def __getstate__(self) -> dict[str, object]:
        return dict(self.config)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.config = dict(state)

    def to_json(self) -> str:
        """Convert the config to a JSON string."""
        return json.dumps(self.config, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, object]) -> Config:
        """Create a Config from a plain dictionary."""
        obj = Config()
        obj.config = dict(config_dict)
        return obj

    @classmethod
    def from_json(cls, json_str: str) -> Config:
        """Create a Config object from a JSON string."""
        config_dict = json.loads(json_str)
        return cls(**config_dict)  # Changed to use dictionary unpacking

    def minimize(self, config_spec: object) -> Config:
        """
        Return a new Config with values matching effective defaults removed.

        Matching defaults are removed only when normalization preserves the
        selected config. Dependent or unprovable forms retain their fields.

        Args:
            config_spec: The ConfigSpec that defines the defaults for this kernel.

        Returns:
            A new Config with default values removed.
        """
        from ..autotuner.config_fragment import _plain_builtin_fragment_key
        from ..autotuner.config_spec import ConfigSpec
        from ..exc import InvalidConfig

        assert isinstance(config_spec, ConfigSpec)
        default_config = config_spec.default_config()

        # Arbitrary default protocols may have side effects. Keep explicit
        # custom values while minimizing the independent schedule fields.
        custom_keys = {
            key
            for key, fragment in config_spec.user_defined_tunables.items()
            if _plain_builtin_fragment_key(fragment) is None
        }
        if not custom_keys.issubset(self.config):
            # Proving an omitted custom value would call its default again.
            return Config.from_dict(self.config)
        required_keys = {"block_sizes"} | custom_keys

        minimal: dict[str, object] = {}
        for key, value in self.config.items():
            # Keep value if it differs from the default or is required
            default_value = default_config.config.get(key)
            if (
                _config_value_key(value) != _config_value_key(default_value)
                or key in required_keys
            ):
                minimal[key] = value

        candidate = Config.from_dict(minimal)
        try:
            expected = config_spec.normalized_config(self)
            normalized = config_spec.normalized_config(candidate)
            if _config_key(normalized) == _config_key(expected):
                return candidate

            # Effective search defaults may differ from normalization's omitted
            # defaults (for example after shrinking a small kernel's warp count).
            # Restore those fields rather than change the selected configuration.
            for key, value in self.config.items():
                if key not in minimal and (
                    (key in expected) != (key in normalized)
                    or _config_value_key(expected.config.get(key))
                    != _config_value_key(normalized.config.get(key))
                ):
                    minimal[key] = value
            candidate = Config.from_dict(minimal)
            if _config_key(config_spec.normalized_config(candidate)) == _config_key(
                expected
            ):
                return candidate
        except InvalidConfig:
            # Minimization is optional; leave rejected or dependent forms for
            # the usual compilation validation instead of guessing defaults.
            pass
        return Config.from_dict(self.config)

    def save(self, path: str | Path) -> None:
        """Save the config to a JSON file."""
        # Write to temp dir and rename to make the operation atomic
        # in case we are in a multithreaded environment
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        tmp = Path(path).parent / f"tmp.{uuid.uuid4()!s}"
        tmp.write_text(self.to_json())
        os.rename(str(tmp), str(path))

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a config from a JSON file."""
        return cls.from_json(Path(path).read_text())

    @property
    def block_sizes(self) -> list[int]:
        return cast("list[int]", self.config["block_sizes"])

    @property
    def loop_orders(self) -> list[list[int]]:
        return cast("list[list[int]]", self.config.get("loop_orders", []))

    @property
    def num_threads(self) -> list[int]:
        value = self.config.get("num_threads", [])
        if isinstance(value, int):
            return [value]
        return cast("list[int]", value)

    @property
    def flatten_loops(self) -> list[bool]:
        return cast("list[bool]", self.config.get("flatten_loops", []))

    @property
    def reduction_loops(self) -> list[int | None]:
        return cast("list[int | None]", self.config.get("reduction_loops", []))

    @property
    def num_warps(self) -> int:
        from ..autotuner.config_spec import DEFAULT_NUM_WARPS

        return cast("int", self.config.get("num_warps", DEFAULT_NUM_WARPS))

    @property
    def num_stages(self) -> int:
        from ..autotuner.config_spec import DEFAULT_NUM_STAGES

        return cast("int", self.config.get("num_stages", DEFAULT_NUM_STAGES))

    @property
    def l2_groupings(self) -> list[int]:
        return cast("list[int]", self.config.get("l2_groupings", []))

    @property
    def pid_type(self) -> PidTypeLiteral:
        return cast("PidTypeLiteral", self.config.get("pid_type", "flat"))

    @property
    def cross_loop_pipeline(self) -> CrossLoopPipelineLiteral:
        return cast(
            "CrossLoopPipelineLiteral",
            self.config.get("cross_loop_pipeline", "barrier"),
        )

    @property
    def xcd_remap(self) -> bool:
        return cast("bool", self.config.get("xcd_remap", False))

    @property
    def num_sm_multiplier(self) -> int:
        from ..autotuner.config_spec import DEFAULT_NUM_SM_MULTIPLIER

        return cast(
            "int", self.config.get("num_sm_multiplier", DEFAULT_NUM_SM_MULTIPLIER)
        )

    @property
    def maxnreg(self) -> int | None:
        from ..autotuner.config_spec import DEFAULT_MAXNREG

        return cast("int | None", self.config.get("maxnreg", DEFAULT_MAXNREG))

    @property
    def host_tensor_descriptors(self) -> bool:
        return cast("bool", self.config.get("host_tensor_descriptors", False))

    @property
    def reduction_fusion(self) -> bool:
        return cast("bool", self.config.get("reduction_fusion", False))

    @property
    def range_unroll_factors(self) -> list[int]:
        return cast("list[int]", self.config.get("range_unroll_factors", []))

    @property
    def advanced_controls_file(self) -> str:
        return cast("str", self.config.get("advanced_controls_file", ""))

    @property
    def range_warp_specializes(self) -> list[bool | None]:
        return cast("list[bool | None]", self.config.get("range_warp_specializes", []))

    @property
    def range_num_stages(self) -> list[int]:
        return cast("list[int]", self.config.get("range_num_stages", []))

    @property
    def range_multi_buffers(self) -> list[bool | None]:
        return cast("list[bool | None]", self.config.get("range_multi_buffers", []))

    @property
    def range_flattens(self) -> list[bool | None]:
        return cast("list[bool | None]", self.config.get("range_flattens", []))

    @property
    def static_ranges(self) -> list[bool]:
        return cast("list[bool]", self.config.get("static_ranges", []))

    @property
    def pallas_load_buffer_count(self) -> list[int]:
        return cast("list[int]", self.config.get("pallas_load_buffer_count", []))

    @property
    def load_eviction_policies(
        self,
    ) -> EvictionPolicyLiteral | list[EvictionPolicyLiteral]:
        return cast(
            "EvictionPolicyLiteral | list[EvictionPolicyLiteral]",
            self.config.get("load_eviction_policies", []),
        )

    @property
    def load_cache_modifiers(self) -> list[LoadCacheModifierLiteral]:
        return cast(
            "list[LoadCacheModifierLiteral]",
            self.config.get("load_cache_modifiers", []),
        )

    @property
    def store_cache_modifiers(self) -> list[StoreCacheModifierLiteral]:
        return cast(
            "list[StoreCacheModifierLiteral]",
            self.config.get("store_cache_modifiers", []),
        )

    @property
    def cute_async_load_stages(self) -> int:
        return cast("int", self.config.get("cute_async_load_stages", 0))

    @property
    def cute_async_load_lookahead(self) -> int:
        return cast("int", self.config.get("cute_async_load_lookahead", 4))

    @property
    def cute_async_load_group_rows(self) -> int:
        return cast("int", self.config.get("cute_async_load_group_rows", 2))

    @property
    def cute_async_load_cache(self) -> CuteAsyncLoadCacheLiteral:
        return cast(
            "CuteAsyncLoadCacheLiteral",
            self.config.get("cute_async_load_cache", "cg"),
        )

    @property
    def cute_async_store_policy(self) -> CuteAsyncStorePolicyLiteral:
        return cast(
            "CuteAsyncStorePolicyLiteral",
            self.config.get("cute_async_store_policy", "default"),
        )

    @property
    def cute_store_policy(self) -> CuteStorePolicyLiteral:
        return cast(
            "CuteStorePolicyLiteral",
            self.config.get("cute_store_policy", "default"),
        )

    @property
    def cute_wide_packet_hints(self) -> bool:
        return self.config.get("cute_wide_packet_hints") is True

    @property
    def cute_exact_div_form(self) -> CuteExactDivFormLiteral:
        return cast(
            "CuteExactDivFormLiteral",
            self.config.get("cute_exact_div_form", "redo"),
        )

    @property
    def cute_bf16x2_recurrence(self) -> bool:
        return cast("bool", self.config.get("cute_bf16x2_recurrence", False))

    @property
    def cute_proven_bounds(self) -> bool:
        return cast("bool", self.config.get("cute_proven_bounds", False))

    @property
    def cute_affine_scan_schedule(self) -> CuteAffineScanScheduleLiteral:
        return cast(
            "CuteAffineScanScheduleLiteral",
            self.config.get("cute_affine_scan_schedule", "ordinary"),
        )

    @property
    def cute_packet_prefetch(self) -> int:
        return cast("int", self.config.get("cute_packet_prefetch", 0))

    @property
    def cute_reduction_pipeline_depth(self) -> int:
        return cast("int", self.config.get("cute_reduction_pipeline_depth", 2))

    @property
    def indexing(self) -> IndexingLiteral | list[IndexingLiteral]:
        return cast(
            "IndexingLiteral | list[IndexingLiteral]", self.config.get("indexing", [])
        )

    @property
    def atomic_indexing(self) -> IndexingLiteral | list[IndexingLiteral]:
        return cast(
            "IndexingLiteral | list[IndexingLiteral]",
            self.config.get("atomic_indexing", []),
        )

    @property
    def epilogue_subtile(self) -> int | None:
        return cast("int | None", self.config.get("epilogue_subtile", None))


def _to_hashable(x: object) -> object:
    if isinstance(x, list):
        return tuple([_to_hashable(i) for i in x])
    if isinstance(x, dict):
        return tuple(sorted([(k, _to_hashable(v)) for k, v in x.items()]))
    return x
