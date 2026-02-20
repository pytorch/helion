from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from torch._inductor.runtime.runtime_utils import next_power_of_2

from .._compat import supports_amd_cdna_tunables
from .._compat import supports_maxnreg
from .._compat import supports_tensor_descriptor
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
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .._compiler.backend import Backend
    from ..runtime.config import IndexingLiteral
    from ..runtime.config import PidTypeLiteral
    from .config_generation import ConfigGeneration

DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 1
BACKEND_TUNABLE_KEYS: frozenset[str] = frozenset(
    {"waves_per_eu", "matrix_instr_nonkdim", "num_ctas", "occupancy"}
)
# All config keys whose support depends on the backend.  The base Backend
# class rejects these by default; each backend subclass opts in selectively.
BACKEND_SPECIFIC_KEYS: frozenset[str] = BACKEND_TUNABLE_KEYS | {"elements_per_thread"}
VALID_KEYS: frozenset[str] = frozenset(
    [
        "block_sizes",
        "elements_per_thread",
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
        "epilogue_subtiling",
        *BACKEND_TUNABLE_KEYS,
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
def get_valid_eviction_policies(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and not supports_amd_cdna_tunables():
        return ("", "first", "last")
    return ("",)


VALID_EPILOGUE_SUBTILE_SIZES = (None, 2, 4)


class ConfigSpec:
    def __init__(
        self,
        *,
        backend: Backend,
        user_defined_tunables: Mapping[str, ConfigSpecFragment] | None = None,
    ) -> None:
        self.backend = backend
        self.backend_name = backend.name
        self.max_reduction_threads = backend.max_reduction_threads()
        self.user_defined_tunables = (
            {} if user_defined_tunables is None else dict(user_defined_tunables)
        )

        self.block_sizes: BlockIdSequence[BlockSizeSpec] = BlockIdSequence()
        self.elements_per_thread: BlockIdSequence[ElementsPerThreadSpec] = (
            BlockIdSequence()
        )
        self.loop_orders: BlockIdSequence[LoopOrderSpec] = BlockIdSequence()
        self.l2_groupings: BlockIdSequence[L2GroupingSpec] = BlockIdSequence()
        self.flatten_loops: BlockIdSequence[FlattenLoopSpec] = BlockIdSequence()
        self.reduction_loops: BlockIdSequence[ReductionLoopSpec] = BlockIdSequence()
        self.range_unroll_factors: BlockIdSequence[RangeUnrollFactorSpec] = (
            BlockIdSequence()
        )
        self.range_warp_specialize: BlockIdSequence[RangeWarpSpecializeSpec] = (
            BlockIdSequence()
        )
        self.range_num_stages: BlockIdSequence[RangeNumStagesSpec] = BlockIdSequence()
        self.range_multi_buffers: BlockIdSequence[RangeMultiBufferSpec] = (
            BlockIdSequence()
        )
        self.range_flattens: BlockIdSequence[RangeFlattenSpec] = BlockIdSequence()
        self.static_ranges: BlockIdSequence[StaticRangeSpec] = BlockIdSequence()

        self.allowed_pid_types: tuple[PidTypeLiteral, ...] = tuple(VALID_PID_TYPES)
        self.grid_block_ids: list[int] = []
        self.load_eviction_policies = ListOf(
            EnumFragment(choices=get_valid_eviction_policies(self.backend_name)),
            length=0,
        )
        self.indexing = ListOf(
            EnumFragment(choices=self.valid_indexing_types()),
            length=0,
        )
        self.backend_tunable_fragments = self.backend.tunable_fragments()
        unknown_tunables = set(self.backend_tunable_fragments) - BACKEND_TUNABLE_KEYS
        if unknown_tunables:
            raise RuntimeError(
                f"Backend {self.backend_name!r} returned unknown tunables: {sorted(unknown_tunables)!r}"
            )
        self.epilogue_subtiling = ListOf(
            EnumFragment(choices=VALID_EPILOGUE_SUBTILE_SIZES), length=0
        )
        # Maps epilogue_subtiling idx -> block idx for conditioning on whether to subtile
        self.epilogue_subtiling_block_ids: list[int] = []

    def valid_indexing_types(self) -> tuple[IndexingLiteral, ...]:
        if supports_tensor_descriptor():
            return ("pointer", "tensor_descriptor")
        if not self.backend.supports_block_ptr_indexing():
            return ("pointer",)
        return ("pointer", "block_ptr")

    def _remove_duplicates(self) -> None:
        self.elements_per_thread._remove_duplicates()
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

    def supports_config_key(self, key: str) -> bool:
        return self.backend.supports_config_key(key)

    def supported_config_keys(self) -> frozenset[str]:
        return frozenset(key for key in VALID_KEYS if self.supports_config_key(key))

    def unsupported_config_keys(self, config: Mapping[str, object]) -> list[str]:
        return sorted(
            key
            for key in config
            if key in VALID_KEYS and not self.supports_config_key(key)
        )

    def is_supported_config(self, config: Mapping[str, object]) -> bool:
        return not self.unsupported_config_keys(config)

    def register_epilogue_subtiling(
        self, store_count: int, block_ids: list[int]
    ) -> None:
        assert store_count == len(block_ids), (
            f"store_count ({store_count}) must match len(block_ids) ({len(block_ids)})"
        )
        self.epilogue_subtiling = ListOf(
            EnumFragment(choices=VALID_EPILOGUE_SUBTILE_SIZES), length=store_count
        )
        self.epilogue_subtiling_block_ids = block_ids

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

        if unsupported := self.unsupported_config_keys(config):
            raise InvalidConfig(
                f"Unsupported config keys for backend {self.backend_name!r}: {unsupported}"
            )

        for name, mapping, flatten in [
            ("block_sizes", self.block_sizes, True),
            ("elements_per_thread", self.elements_per_thread, True),
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
            if not self.supports_config_key(name):
                if name in config:
                    raise InvalidConfig(
                        f"{name} is not supported on backend {self.backend_name!r}"
                    )
                config.pop(name, None)
                continue
            config[name] = mapping._normalize(
                name, config.get(name, ()), flatten=flatten
            )
        if self.supports_config_key("elements_per_thread"):
            elements_per_thread = cast(
                "list[int]", config.get("elements_per_thread", [])
            )
            if all(value == 1 for value in elements_per_thread):
                config.pop("elements_per_thread", None)
        else:
            config.pop("elements_per_thread", None)

        # Cap reduction loops at the backend's max reduction thread count
        if self.max_reduction_threads is not None and self.reduction_loops:
            max_threads = self.max_reduction_threads
            reduction_loops = config.get("reduction_loops", [])
            if isinstance(reduction_loops, list):
                new_loops = list(reduction_loops)
                changed = False
                for i, spec in enumerate(self.reduction_loops):
                    if i >= len(new_loops):
                        break
                    if (new_loops[i] is None and spec.size_hint > max_threads) or (
                        new_loops[i] is not None and new_loops[i] > max_threads
                    ):
                        new_loops[i] = max_threads
                        changed = True
                if changed:
                    config["reduction_loops"] = new_loops

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
            "epilogue_subtiling",
        ):
            if not config.get(name):
                config.pop(name, None)

        config.setdefault("num_warps", DEFAULT_NUM_WARPS)
        config.setdefault("num_stages", DEFAULT_NUM_STAGES)
        config.setdefault(
            "load_eviction_policies", self.load_eviction_policies.default()
        )
        config.setdefault("indexing", self.indexing.default())
        for key, fragment in self.backend_tunable_fragments.items():
            config.setdefault(key, fragment.default())
        # Handle epilogue_subtiling specially: if the config has a value from a previous
        # compilation with a different number of stores, reset it to the current default
        # Required for branching logic, test_error_in_non_taken_branch
        if "epilogue_subtiling" in config:
            if config["epilogue_subtiling"] and len(
                config["epilogue_subtiling"]  # pyrefly: ignore [bad-argument-type]
            ) != len(self.epilogue_subtiling_block_ids):
                # Config is from a different compilation, reset to current default
                config["epilogue_subtiling"] = self.epilogue_subtiling.default()
        else:
            config["epilogue_subtiling"] = self.epilogue_subtiling.default()

        epilogue_subtiling = cast(
            "list[int | None]", config.get("epilogue_subtiling", [])
        )
        block_sizes_config = cast("list[int]", config.get("block_sizes", []))
        flatten_loops_config = cast("list[bool]", config.get("flatten_loops", []))
        if epilogue_subtiling and self.epilogue_subtiling_block_ids:
            for i, block_id in enumerate(self.epilogue_subtiling_block_ids):
                if epilogue_subtiling[i] is not None:
                    # Check if the loop is flattened (offset_var not available)
                    if self.flatten_loops.config_get(
                        flatten_loops_config, block_id, False
                    ):
                        epilogue_subtiling[i] = None
                        continue
                    block_size = self.block_sizes.config_get(
                        block_sizes_config, block_id, None
                    )
                    # Block size must be > 16 and divisible by 2
                    if block_size is None or block_size <= 16 or block_size % 2 != 0:
                        epilogue_subtiling[i] = None

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
        # Allow tunable parameter keys in addition to backend-supported keys.
        allowed_keys = self.supported_config_keys() | {
            *self.user_defined_tunables.keys()
        }
        if invalid_keys := ({*config} - allowed_keys):
            raise InvalidConfig(f"Invalid config keys {sorted(invalid_keys)!r}")

    def create_config_generation(
        self, *, overrides: Mapping[str, object] | None = None
    ) -> ConfigGeneration:
        from .config_generation import ConfigGeneration

        return ConfigGeneration(self, overrides=overrides)

    def default_config(self) -> helion.Config:
        return self.flat_config(lambda x: x.default())

    def flat_config(self, fn: Callable[[ConfigSpecFragment], object]) -> helion.Config:
        """Map a flattened version of the config using the given function."""
        config: dict[str, Any] = {
            "block_sizes": self.block_sizes._flat_config(self, fn),
            "loop_orders": self.loop_orders._flat_config(self, fn),
            "flatten_loops": self.flatten_loops._flat_config(self, fn),
            "l2_groupings": self.l2_groupings._flat_config(self, fn),
            "reduction_loops": self.reduction_loops._flat_config(self, fn),
            "range_unroll_factors": self.range_unroll_factors._flat_config(self, fn),
            "range_warp_specializes": self.range_warp_specialize._flat_config(self, fn),
            "range_num_stages": self.range_num_stages._flat_config(self, fn),
            "range_multi_buffers": self.range_multi_buffers._flat_config(self, fn),
            "range_flattens": self.range_flattens._flat_config(self, fn),
            "static_ranges": self.static_ranges._flat_config(self, fn),
            "num_warps": fn(NumWarpsFragment(1, 32, DEFAULT_NUM_WARPS))
            if not supports_amd_cdna_tunables()
            else fn(NumWarpsFragment(1, 16, DEFAULT_NUM_WARPS)),
            "num_stages": fn(IntegerFragment(1, 8, DEFAULT_NUM_STAGES))
            if not supports_amd_cdna_tunables()
            else fn(IntegerFragment(1, 4, DEFAULT_NUM_STAGES)),
            "indexing": fn(self.indexing),
            "pid_type": fn(EnumFragment(self.allowed_pid_types)),
            "num_sm_multiplier": fn(
                PowerOfTwoFragment(
                    MIN_NUM_SM_MULTIPLIER,
                    MAX_NUM_SM_MULTIPLIER,
                    DEFAULT_NUM_SM_MULTIPLIER,
                )
            ),
            "load_eviction_policies": fn(self.load_eviction_policies),
            "epilogue_subtiling": fn(self.epilogue_subtiling),
        }
        if self.supports_config_key("elements_per_thread"):
            config["elements_per_thread"] = self.elements_per_thread._flat_config(
                self, fn
            )

        if {"num_ctas", "occupancy"} <= set(self.backend_tunable_fragments):
            num_ctas_fragment = self.backend_tunable_fragments["num_ctas"]
            occupancy_fragment = self.backend_tunable_fragments["occupancy"]
            # num_warps is not used in tileir backend, set to 4 as placeholder
            tileir_config = {
                "num_stages": fn(EnumFragment(choices=tuple(range(1, 11)))),
                "num_warps": fn(NumWarpsFragment(4, 4)),
                "num_ctas": fn(num_ctas_fragment),
                "occupancy": fn(occupancy_fragment),
            }
            config.update(tileir_config)
        else:
            config.update(
                {
                    key: fn(fragment)
                    for key, fragment in self.backend_tunable_fragments.items()
                }
            )

        # Only include maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if supports_maxnreg():
            config["maxnreg"] = fn(EnumFragment(VALID_MAXNREG))
        # Add tunable parameters
        config.update(
            {key: fn(fragment) for key, fragment in self.user_defined_tunables.items()}
        )

        for name in (
            "loop_orders",
            "elements_per_thread",
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
            "epilogue_subtiling",
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


class ElementsPerThreadSpec(_PowerOfTwoBlockIdItem):
    def __init__(self, *, block_id: int, size_hint: int) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _fragment(self, base: ConfigSpec) -> PowerOfTwoFragment:
        max_ept = min(max(self.size_hint, 1), 256)
        return PowerOfTwoFragment(
            1,
            next_power_of_2(max_ept),
            1,
        )

    def _fill_missing(self) -> int:
        return 1


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
        # Cap default at the backend's max reduction threads so that
        # large reductions default to looped rather than persistent.
        if base.max_reduction_threads is not None:
            if self.size_hint > base.max_reduction_threads:
                default = min(default, base.max_reduction_threads)
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
