from __future__ import annotations

import dataclasses
import functools
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from torch._inductor.runtime.runtime_utils import next_power_of_2

from .._compat import supports_amd_cdna_tunables
from .._compat import supports_maxnreg
from .._compat import supports_tensor_descriptor
from .._compat import use_tileir_tunables
from ..exc import InvalidConfig
from .block_id_sequence import BlockIdSequence
from .block_id_sequence import _BlockIdItem
from .block_id_sequence import _PowerOfTwoBlockIdItem
from .config_fragment import BlockSizeFragment
from .config_fragment import BooleanFragment
from .config_fragment import ConfigSpecFragment
from .config_fragment import EnumFragment
from .config_fragment import IntegerFragment
from .config_fragment import ListOf
from .config_fragment import NumWarpsFragment
from .config_fragment import PermutationFragment
from .config_fragment import PowerOfTwoFragment
from .config_fragment import assert_integer_power_of_two
import helion

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..runtime.config import IndexingLiteral
    from ..runtime.config import PidTypeLiteral
    from .config_generation import ConfigGeneration

DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 1
DEFAULT_NUM_CTAS = 1
DEFAULT_OCCUPANCY = 1
AMD_CDNA_TUNABLES = ("waves_per_eu", "matrix_instr_nonkdim")
TILEIR_TUNABLES = ("num_ctas", "occupancy")
VALID_KEYS: frozenset[str] = frozenset(
    [
        "block_sizes",
        "loop_orders",
        "l2_groupings",
        "reduction_loops",
        "flatten_loops",
        "range_unroll_factors",
        "range_warp_specializes",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
        "static_ranges",
        "num_warps",
        "num_stages",
        "pid_type",
        "num_sm_multiplier",
        "maxnreg",
        "indexing",
        "load_eviction_policies",
        *AMD_CDNA_TUNABLES,
        *TILEIR_TUNABLES,
    ]
)
VALID_PID_TYPES = ("flat", "xyz", "persistent_blocked", "persistent_interleaved")
MIN_NUM_SM_MULTIPLIER = 1
MAX_NUM_SM_MULTIPLIER = 128
DEFAULT_NUM_SM_MULTIPLIER = 1
# maxnreg values: None means no limit, otherwise limit to this many registers per thread
# Lower values allow higher occupancy but may hurt performance for register-heavy kernels
VALID_MAXNREG = (None, 32, 64, 128, 256)
DEFAULT_MAXNREG = None


# For tileir backend or AMD ROCM, eviction policies are not supported.
# This is a function to avoid CUDA initialization at import time.
@functools.cache
def get_valid_eviction_policies() -> tuple[str, ...]:
    if not use_tileir_tunables() and not supports_amd_cdna_tunables():
        return ("", "first", "last")
    return ("",)


VALID_WAVES_PER_EU = (1, 2, 3, 4)
VALID_MATRIX_INSTR_NONKDIM = (0, 16, 32)


@dataclasses.dataclass
class ConfigSpec:
    block_sizes: BlockIdSequence[BlockSizeSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    loop_orders: BlockIdSequence[LoopOrderSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    l2_groupings: BlockIdSequence[L2GroupingSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    flatten_loops: BlockIdSequence[FlattenLoopSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    reduction_loops: BlockIdSequence[ReductionLoopSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    range_unroll_factors: BlockIdSequence[RangeUnrollFactorSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    range_warp_specialize: BlockIdSequence[RangeWarpSpecializeSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    range_num_stages: BlockIdSequence[RangeNumStagesSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    range_multi_buffers: BlockIdSequence[RangeMultiBufferSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    range_flattens: BlockIdSequence[RangeFlattenSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    static_ranges: BlockIdSequence[StaticRangeSpec] = dataclasses.field(
        default_factory=BlockIdSequence
    )
    user_defined_tunables: dict[str, ConfigSpecFragment] = dataclasses.field(
        default_factory=dict
    )
    allowed_pid_types: tuple[PidTypeLiteral, ...] = dataclasses.field(
        default_factory=functools.partial(tuple, VALID_PID_TYPES)
    )
    grid_block_ids: list[int] = dataclasses.field(default_factory=list)
    load_eviction_policies: ListOf = dataclasses.field(
        default_factory=lambda: ListOf(
            EnumFragment(choices=get_valid_eviction_policies()), length=0
        )
    )
    indexing: ListOf = dataclasses.field(
        default_factory=lambda: ListOf(
            EnumFragment(choices=ConfigSpec._valid_indexing_types()),
            length=0,
        )
    )
    waves_per_eu: ConfigSpecFragment | None = dataclasses.field(
        default_factory=lambda: (
            EnumFragment(choices=VALID_WAVES_PER_EU)
            if supports_amd_cdna_tunables()
            else None
        )
    )
    matrix_instr_nonkdim: ConfigSpecFragment | None = dataclasses.field(
        default_factory=lambda: (
            EnumFragment(choices=VALID_MATRIX_INSTR_NONKDIM)
            if supports_amd_cdna_tunables()
            else None
        )
    )
    num_ctas: ConfigSpecFragment | None = dataclasses.field(
        default_factory=lambda: (
            PowerOfTwoFragment(1, 2, DEFAULT_NUM_CTAS)
            if use_tileir_tunables()
            else None
        )
    )
    occupancy: ConfigSpecFragment | None = dataclasses.field(
        default_factory=lambda: (
            PowerOfTwoFragment(1, 8, DEFAULT_OCCUPANCY)
            if use_tileir_tunables()
            else None
        )
    )

    @staticmethod
    def _valid_indexing_types() -> tuple[IndexingLiteral, ...]:
        if supports_tensor_descriptor():
            return ("pointer", "tensor_descriptor")
        if use_tileir_tunables():
            # block_ptr is not supported for tileir backend
            return ("pointer",)
        return ("pointer", "block_ptr")

    def _remove_duplicates(self) -> None:
        self.loop_orders._remove_duplicates()
        self.l2_groupings._remove_duplicates()
        self.flatten_loops._remove_duplicates()
        self.range_unroll_factors._remove_duplicates()
        self.range_warp_specialize._remove_duplicates()
        self.range_num_stages._remove_duplicates()
        self.range_multi_buffers._remove_duplicates()
        self.range_flattens._remove_duplicates()
        self.static_ranges._remove_duplicates()

    def disallow_pid_type(self, pid_type: PidTypeLiteral) -> None:
        """Disallow a pid_type from being used in the config."""

        self.allowed_pid_types = tuple(
            [x for x in self.allowed_pid_types if x != pid_type]
        )
        assert self.allowed_pid_types

    def normalize(
        self, config: helion.Config | dict[str, object], *, _fix_invalid: bool = False
    ) -> None:
        """Normalize the config to match the block_sizes and validate the config.

        Args:
            config: The config to normalize (modified in place).
            _fix_invalid: If True, silently fix invalid combinations instead of raising
                errors. Used internally during autotuning config generation.
        """
        if isinstance(config, helion.Config):
            self.normalize(config.config, _fix_invalid=_fix_invalid)
            return

        for name in (
            "block_size",
            "loop_order",
            "reduction_loop",
            "l2_grouping",
            "flatten_loop",
            "range_unroll_factor",
            "range_warp_specialize",
            "range_num_stage",
            "range_multi_buffer",
            "range_flatten",
            "static_range",
        ):
            if name in config:
                names = f"{name}s"
                if names in config:
                    raise InvalidConfig(f"Cannot specify both {name} and {names}")
                config[names] = [config.pop(name)]

        for name, mapping, flatten in [
            ("block_sizes", self.block_sizes, True),
            ("flatten_loops", self.flatten_loops, True),
            ("l2_groupings", self.l2_groupings, True),
            ("loop_orders", self.loop_orders, False),
            ("reduction_loops", self.reduction_loops, True),
            ("range_unroll_factors", self.range_unroll_factors, True),
            ("range_warp_specializes", self.range_warp_specialize, True),
            ("range_num_stages", self.range_num_stages, True),
            ("range_multi_buffers", self.range_multi_buffers, True),
            ("range_flattens", self.range_flattens, True),
            ("static_ranges", self.static_ranges, True),
        ]:
            config[name] = mapping._normalize(
                name, config.get(name, ()), flatten=flatten
            )

        # Disable range_* configs for static ranges
        static_range_block_ids = [
            block_id
            for block_id in self.static_ranges.valid_block_ids()
            if self.static_ranges.config_get(
                cast("list[bool]", config.get("static_ranges", [])),
                block_id,
            )
        ]
        if static_range_block_ids:
            for name, mapping in (
                ("range_unroll_factors", self.range_unroll_factors),
                ("range_warp_specializes", self.range_warp_specialize),
                ("range_num_stages", self.range_num_stages),
                ("range_multi_buffers", self.range_multi_buffers),
                ("range_flattens", self.range_flattens),
            ):
                config[name] = mapping._reset_config_to_default(
                    name, config.get(name, ()), block_ids=static_range_block_ids
                )

        for name in (
            "loop_orders",
            "l2_groupings",
            "flatten_loops",
            "reduction_loops",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "indexing",
        ):
            if not config.get(name):
                config.pop(name, None)

        config.setdefault("num_warps", DEFAULT_NUM_WARPS)
        config.setdefault("num_stages", DEFAULT_NUM_STAGES)
        config.setdefault(
            "load_eviction_policies", self.load_eviction_policies.default()
        )
        config.setdefault("indexing", self.indexing.default())
        for key in AMD_CDNA_TUNABLES:
            if (fragment := getattr(self, key)) is not None:
                config.setdefault(key, fragment.default())
            elif key in config:
                raise InvalidConfig(f"{key} is not supported on this target hardware")
        for key in TILEIR_TUNABLES:
            if (fragment := getattr(self, key)) is not None:
                config.setdefault(key, fragment.default())
            elif key in config:
                raise InvalidConfig(f"{key} is not supported on this target hardware")

        if "pid_type" in config:
            if config["pid_type"] not in VALID_PID_TYPES:
                raise InvalidConfig(
                    f"Invalid value for 'pid_type': {config['pid_type']!r} must be one of {list(VALID_PID_TYPES)!r}"
                )
        else:
            config["pid_type"] = VALID_PID_TYPES[0]

        # Validate num_sm_multiplier is a power of two in range
        if "num_sm_multiplier" in config:
            val = config["num_sm_multiplier"]
            if (
                not isinstance(val, int)
                or val < MIN_NUM_SM_MULTIPLIER
                or val > MAX_NUM_SM_MULTIPLIER
                or (val & (val - 1)) != 0  # not a power of two
            ):
                raise InvalidConfig(
                    f"Invalid value for 'num_sm_multiplier': {val!r} must be a power of two between {MIN_NUM_SM_MULTIPLIER} and {MAX_NUM_SM_MULTIPLIER}"
                )
        else:
            config["num_sm_multiplier"] = DEFAULT_NUM_SM_MULTIPLIER

        # Only validate maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if supports_maxnreg():
            if "maxnreg" in config:
                if config["maxnreg"] not in VALID_MAXNREG:
                    raise InvalidConfig(
                        f"Invalid value for 'maxnreg': {config['maxnreg']!r} must be one of {list(VALID_MAXNREG)!r}"
                    )
            else:
                config["maxnreg"] = VALID_MAXNREG[0]
        else:
            # Remove maxnreg on AMD if present
            config.pop("maxnreg", None)

        # Handle num_sm_multiplier and maxnreg for non-persistent pid_types
        # These options only make sense for persistent kernels
        pid_type = config["pid_type"]
        if pid_type in ("flat", "xyz"):
            # Handle num_sm_multiplier
            num_sm_multiplier = config.get(
                "num_sm_multiplier", DEFAULT_NUM_SM_MULTIPLIER
            )
            if num_sm_multiplier != DEFAULT_NUM_SM_MULTIPLIER:
                if _fix_invalid:
                    # Silently fix during autotuning config generation
                    config.pop("num_sm_multiplier", None)
                else:
                    # Raise error for user-specified invalid combinations
                    raise InvalidConfig(
                        f"num_sm_multiplier={num_sm_multiplier} can only be used with persistent "
                        f"pid_type ('persistent_blocked' or 'persistent_interleaved'), "
                        f"got pid_type={pid_type!r}"
                    )
            else:
                # Remove default value from config
                config.pop("num_sm_multiplier", None)

            # Handle maxnreg - only makes sense for persistent kernels (and only on non-AMD and non-Intel GPU)
            if supports_maxnreg():
                maxnreg = config.get("maxnreg", DEFAULT_MAXNREG)
                if maxnreg != DEFAULT_MAXNREG:
                    if _fix_invalid:
                        # Silently fix during autotuning config generation
                        config.pop("maxnreg", None)
                    else:
                        # Raise error for user-specified invalid combinations
                        raise InvalidConfig(
                            f"maxnreg={maxnreg} can only be used with persistent "
                            f"pid_type ('persistent_blocked' or 'persistent_interleaved'), "
                            f"got pid_type={pid_type!r}"
                        )
                else:
                    # Remove default value from config
                    config.pop("maxnreg", None)

        # Set default values for grid indices when pid_type is not persistent
        if pid_type in ("flat", "xyz") and self.grid_block_ids:
            for name, mapping in (
                ("range_unroll_factors", self.range_unroll_factors),
                ("range_warp_specializes", self.range_warp_specialize),
                ("range_num_stages", self.range_num_stages),
                ("range_multi_buffers", self.range_multi_buffers),
                ("range_flattens", self.range_flattens),
            ):
                config[name] = mapping._reset_config_to_default(
                    name, config.get(name, ()), block_ids=self.grid_block_ids
                )

        range_warp_specializes = cast(
            "list[bool | None]", config.get("range_warp_specializes", [])
        )

        if range_warp_specializes and any(range_warp_specializes):
            # Only one range_warp_specializes is allowed, take the first one
            # Prefer warp specialize on outermost loop
            first_idx = range_warp_specializes.index(True)
            for i in range(first_idx + 1, len(range_warp_specializes)):
                range_warp_specializes[i] = None

            range_unroll_factors = cast(
                "list[int]", config.get("range_unroll_factors", [])
            )
            if range_unroll_factors and range_unroll_factors[first_idx] > 1:
                if range_unroll_factors[first_idx]:
                    range_unroll_factors[first_idx] = 0

                config["range_unroll_factors"] = range_unroll_factors

        config["range_warp_specializes"] = range_warp_specializes
        # Allow tunable parameter keys in addition to VALID_KEYS
        allowed_keys = VALID_KEYS | {*self.user_defined_tunables.keys()}
        if invalid_keys := ({*config} - allowed_keys):
            raise InvalidConfig(f"Invalid config keys {sorted(invalid_keys)!r}")

    def create_config_generation(
        self, *, overrides: Mapping[str, object] | None = None
    ) -> ConfigGeneration:
        from .config_generation import ConfigGeneration

        return ConfigGeneration(self, overrides=overrides)

    def default_config(self) -> helion.Config:
        return self.flat_config(lambda x: x.default())

    # ---- shared field iterators (used by flat_config & flat_key_layout) ----

    def _flat_sequence_fields(
        self,
    ) -> tuple[tuple[str, BlockIdSequence[Any]], ...]:
        """(key, BlockIdSequence) for list-based fields in flat_config() order."""
        return (
            ("block_sizes", self.block_sizes),
            ("loop_orders", self.loop_orders),
            ("flatten_loops", self.flatten_loops),
            ("l2_groupings", self.l2_groupings),
            ("reduction_loops", self.reduction_loops),
            ("range_unroll_factors", self.range_unroll_factors),
            ("range_warp_specializes", self.range_warp_specialize),
            ("range_num_stages", self.range_num_stages),
            ("range_multi_buffers", self.range_multi_buffers),
            ("range_flattens", self.range_flattens),
            ("static_ranges", self.static_ranges),
        )

    def _scalar_flat_fragments(self) -> Iterator[tuple[str, ConfigSpecFragment]]:
        """Yield (key, fragment) for scalar/ListOf fields in flat_config() order."""
        yield "num_warps", (
            NumWarpsFragment(1, 32, DEFAULT_NUM_WARPS)
            if not supports_amd_cdna_tunables()
            else NumWarpsFragment(1, 16, DEFAULT_NUM_WARPS)
        )
        yield "num_stages", (
            IntegerFragment(1, 8, DEFAULT_NUM_STAGES)
            if not supports_amd_cdna_tunables()
            else IntegerFragment(1, 4, DEFAULT_NUM_STAGES)
        )
        yield "indexing", self.indexing
        yield "pid_type", EnumFragment(self.allowed_pid_types)
        yield "num_sm_multiplier", PowerOfTwoFragment(
            MIN_NUM_SM_MULTIPLIER,
            MAX_NUM_SM_MULTIPLIER,
            DEFAULT_NUM_SM_MULTIPLIER,
        )
        yield "load_eviction_policies", self.load_eviction_policies
        if use_tileir_tunables():
            assert self.num_ctas is not None, "num_ctas is required for tileir backend"
            assert self.occupancy is not None, (
                "occupancy is required for tileir backend"
            )
            # tileir overrides num_stages/num_warps and adds num_ctas/occupancy;
            # num_warps is unused in tileir backend, set to 4 as placeholder
            yield "num_stages", EnumFragment(choices=tuple(range(1, 11)))
            yield "num_warps", NumWarpsFragment(4, 4)
            yield "num_ctas", self.num_ctas
            yield "occupancy", self.occupancy
        if supports_maxnreg():
            yield "maxnreg", EnumFragment(VALID_MAXNREG)
        yield from self.user_defined_tunables.items()

    # ---- public API built on the shared iterators ----

    def flat_key_layout(self) -> list[tuple[str, int]]:
        """Return (key_name, num_flat_entries) for each field in flat_config() order.

        Built from the same helpers that flat_config() uses so there is
        exactly one place where field ordering lives.
        """
        layout: list[tuple[str, int]] = []
        for key, seq in self._flat_sequence_fields():
            if seq:
                layout.append((key, len(seq)))
        layout.extend((key, 1) for key, _ in self._scalar_flat_fragments())
        return layout

    def flat_config(self, fn: Callable[[ConfigSpecFragment], object]) -> helion.Config:
        """Map a flattened version of the config using the given function."""
        config: dict[str, Any] = {}
        for key, seq in self._flat_sequence_fields():
            config[key] = seq._flat_config(self, fn)
        for key, fragment in self._scalar_flat_fragments():
            config[key] = fn(fragment)

        for name in (
            "loop_orders",
            "flatten_loops",
            "reduction_loops",
            "l2_groupings",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "indexing",
        ):
            if not config.get(name):
                config.pop(name, None)
        self.normalize(config, _fix_invalid=True)
        return helion.Config(**config)


class LoopOrderSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PermutationFragment:
        return PermutationFragment(len(self.block_ids))

    def _normalize(self, name: str, value: object) -> list[int]:
        if type(value) is not list:
            if not isinstance(value, tuple):
                raise InvalidConfig(f"{name} must be a list, got {value!r}")
            value = [*value]
        length = len(self.block_ids)
        if len(value) != length:
            raise InvalidConfig(f"{name} must be length {length}, got {len(value)}")
        if {*value} != {*range(length)}:
            raise InvalidConfig(f"{name} must be permutation, got {value!r}")
        return value

    def _fill_missing(self) -> list[int]:
        """Provide a value when not provided by the user."""
        return [*range(len(self.block_ids))]


class L2GroupingSpec(_PowerOfTwoBlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PowerOfTwoFragment:
        return PowerOfTwoFragment(1, 64, 1)

    def _fill_missing(self) -> int:
        return 1


class BlockSizeSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
        min_size: int = 1,
        max_size: int | None = None,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint
        self.min_size: int = min_size
        bounded_hint = max(size_hint, 1)
        self.max_size: int = (
            next_power_of_2(bounded_hint) if max_size is None else max_size
        )
        if self.max_size < self.min_size:
            self.max_size = self.min_size
        assert self.min_size <= self.max_size

    def __repr__(self) -> str:
        fields: list[str] = []
        for field, default in (
            ("block_id", None),
            ("size_hint", None),
            ("min_size", 1),
            ("max_size", next_power_of_2(self.size_hint)),
        ):
            value = getattr(self, field)
            if value != default:
                fields.append(f"{field}={value!r}")
        return f"BlockSizeSpec({', '.join(fields)})"

    def update_min(self, value: int) -> None:
        self.min_size = assert_integer_power_of_two(max(value, self.min_size))
        if self.max_size < self.min_size:
            self.max_size = self.min_size

    def update_max(self, value: int) -> None:
        clamped = max(value, 1)
        self.max_size = assert_integer_power_of_two(min(clamped, self.max_size))

    def update_hint(self, value: int) -> None:
        self.size_hint = value
        self.update_max(next_power_of_2(max(value, 1)))

    def _fragment(self, base: ConfigSpec) -> BlockSizeFragment:
        total_ndim = len(base.block_sizes)
        reduction_numel = _product(
            [next_power_of_2(spec.size_hint) for spec in base.reduction_loops]
        )
        if total_ndim <= 2 and reduction_numel <= 128:
            default = 32
        elif reduction_numel <= 256:
            default = 16
        else:
            default = 1
        return BlockSizeFragment(
            self.min_size,
            self.max_size,
            default,
        )


class FlattenLoopSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> BooleanFragment:
        return BooleanFragment()

    def _normalize(self, name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}") from None
        return value

    def _fill_missing(self) -> bool:
        return False


class ReductionLoopSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _flat_config(
        self, base: ConfigSpec, fn: Callable[[ConfigSpecFragment], object]
    ) -> int | None:
        low = 8  # TODO(jansel): is smaller needed?
        high = next_power_of_2(max(low, self.size_hint))
        default = min(high, 4096)
        value = fn(BlockSizeFragment(low, high, default))
        assert isinstance(value, int)

        if not (low <= value <= high):
            raise InvalidConfig(
                f"Invalid value for reduction loop {low} <= {value} <= {high}"
            )
        if value >= self.size_hint:
            return None  # max size becomes persistent reduction
        return value

    def _normalize(self, name: str, value: object) -> int | None:
        if value is None:
            return None
        return super()._normalize(name, value)

    def _fill_missing(self) -> None:
        return None


class _OptionalIntSpec(_BlockIdItem):
    def _normalize(self, name: str, value: object) -> int:
        if not isinstance(value, int):
            raise InvalidConfig(f"{name} must be an integer, got {value!r}")
        return value

    def _fill_missing(self) -> int:
        """Provide a value when not provided by the user."""
        return 0


class _OptionalBoolSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        return EnumFragment((None, False, True))

    def _normalize(self, name: str, value: object) -> bool | None:
        if value is not None and not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean or None, got {value!r}")
        return value

    def _fill_missing(self) -> None:
        """Provide a value when not provided by the user."""
        return None


class RangeUnrollFactorSpec(_OptionalIntSpec):
    def _fragment(self, base: ConfigSpec) -> IntegerFragment:
        return IntegerFragment(0, 4, 0)


class RangeWarpSpecializeSpec(_OptionalBoolSpec):
    pass


class RangeNumStagesSpec(_OptionalIntSpec):
    def _fragment(self, base: ConfigSpec) -> IntegerFragment:
        return IntegerFragment(0, 4, 0)


class RangeMultiBufferSpec(_OptionalBoolSpec):
    pass


class RangeFlattenSpec(_OptionalBoolSpec):
    pass


class StaticRangeSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> BooleanFragment:
        return BooleanFragment()

    def _normalize(self, name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}")
        return value

    def _fill_missing(self) -> bool:
        """Provide a value when not provided by the user."""
        return False


def _product(seq: Sequence[int]) -> int:
    """Return the product of the elements in the sequence."""
    return functools.reduce(operator.mul, seq, 1)
