from __future__ import annotations

import contextlib
import copy
import functools
import itertools
import operator
import random
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast

from .._compat import warps_to_threads
from ..exc import InvalidConfig
from .block_id_sequence import BlockIdSequence
from .config_fragment import Category
from .config_fragment import ConfigSpecFragment
from .config_fragment import PowerOfTwoFragment
from .config_spec import shrink_block_sizes_for_numel_constraints
from helion._dist_utils import sync_seed

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .. import Config
    from . import ConfigSpec

FlatConfig = list[object]


TRITON_MAX_TENSOR_NUMEL = 1048576


class ConfigGeneration:
    def __init__(
        self,
        config_spec: ConfigSpec,
        *,
        overrides: Mapping[str, object] | None = None,
        advanced_controls_files: list[str] | None = None,
        process_group_name: str | None = None,
    ) -> None:
        def _collect_spec(spec: ConfigSpecFragment) -> object:
            """
            Collect a configuration specification fragment.

            Args:
                spec: The configuration specification fragment.

            Returns:
                The default value of the fragment.
            """
            self.flat_spec.append(spec)
            return spec.default()

        super().__init__()
        self.config_spec = config_spec
        self.process_group_name = process_group_name
        self._advanced_controls_files = advanced_controls_files
        self.flat_spec: list[ConfigSpecFragment] = []
        config_spec.flat_config(
            _collect_spec,
            advanced_controls_files=advanced_controls_files,
        )
        assert self.flat_spec, "No config values to tune"
        self._override_values = dict(overrides or {})
        self.block_size_indices: list[int] = [
            i
            for i, spec in enumerate(self.flat_spec)
            if spec.category() == Category.BLOCK_SIZE
        ]
        self.num_threads_indices: list[int] = []
        self._cute_num_thread_block_pairs: list[tuple[int, int]] = []
        self._cute_block_index_by_id: dict[int, int] = {}
        self._cute_num_thread_index_by_id: dict[int, int] = {}
        self._cute_flatten_loop_groups: list[tuple[int, list[int]]] = []
        if self.config_spec.backend_name == "cute":
            self._init_cute_num_thread_pairs()
        self.num_warps_index: int = next(
            (
                i
                for i, spec in enumerate(self.flat_spec)
                if spec.category() == Category.NUM_WARPS
            ),
            -1,
        )
        self.min_block_size: int = (
            max([spec.min_size for spec in config_spec.block_sizes])
            if config_spec.block_sizes
            else 1
        )

    def _init_cute_num_thread_pairs(self) -> None:
        """Pair each CuTe num_threads flat slot with its block_size slot."""
        try:
            block_indices, _ = self._key_to_flat_indices["block_sizes"]
            num_thread_indices, _ = self._key_to_flat_indices["num_threads"]
        except KeyError:
            return
        self.num_threads_indices = num_thread_indices
        block_index_by_id = {
            spec.block_id: block_indices[i]
            for i, spec in enumerate(self.config_spec.block_sizes)
            if i < len(block_indices)
        }
        num_thread_index_by_id = {
            spec.block_id: num_thread_indices[i]
            for i, spec in enumerate(self.config_spec.num_threads)
            if i < len(num_thread_indices)
        }
        self._cute_block_index_by_id = block_index_by_id
        self._cute_num_thread_index_by_id = num_thread_index_by_id
        self._cute_num_thread_block_pairs = [
            (num_thread_indices[i], block_index_by_id[spec.block_id])
            for i, spec in enumerate(self.config_spec.num_threads)
            if i < len(num_thread_indices) and spec.block_id in block_index_by_id
        ]
        try:
            flatten_indices, _ = self._key_to_flat_indices["flatten_loops"]
        except KeyError:
            return
        self._cute_flatten_loop_groups = [
            (
                flatten_indices[i],
                [
                    block_id
                    for block_id in spec.block_ids
                    if block_id in block_index_by_id
                    and block_id in num_thread_index_by_id
                ],
            )
            for i, spec in enumerate(self.config_spec.flatten_loops)
            if i < len(flatten_indices)
        ]

    @functools.cached_property
    def overridden_flat_indices(self) -> set[int]:
        """Return flat_spec indices that are frozen by config overrides."""
        if not self._override_values:
            return set()
        result: set[int] = set()
        for key in self._override_values:
            if key in self._key_to_flat_indices:
                indices, _ = self._key_to_flat_indices[key]
                result.update(indices)
        return result

    @functools.cached_property
    def _key_to_flat_indices(self) -> dict[str, tuple[list[int], bool]]:
        """Build mapping from config key names to (flat_spec indices, is_sequence).

        Derived from ConfigSpec.flat_key_layout().
        """
        mapping: dict[str, tuple[list[int], bool]] = {}
        idx = 0
        for key, count, is_sequence in self.config_spec.flat_key_layout(
            advanced_controls_files=self._advanced_controls_files
        ):
            mapping[key] = (list(range(idx, idx + count)), is_sequence)
            idx += count
        assert idx == len(self.flat_spec), (
            f"flat_key_layout() total ({idx}) != flat_spec length ({len(self.flat_spec)})"
        )
        return mapping

    def _apply_overrides(self, config: Config) -> Config:
        if not self._override_values:
            return config
        for key, value in self._override_values.items():
            config.config[key] = copy.deepcopy(value)
        self.config_spec.normalize(config.config)
        return config

    @staticmethod
    def _largest_power_of_two_at_most(value: int) -> int:
        return 1 << (max(value, 1).bit_length() - 1)

    def _repair_cute_num_threads(self, flat_config: FlatConfig) -> None:
        """Keep CuTe launch-thread choices compatible with tuned block sizes."""
        if not self._cute_num_thread_block_pairs:
            return

        for num_threads_idx, block_size_idx in self._cute_num_thread_block_pairs:
            num_threads = flat_config[num_threads_idx]
            block_size = flat_config[block_size_idx]
            if (
                type(num_threads) is not int
                or num_threads == 0
                or type(block_size) is not int
                or block_size <= 0
            ):
                continue
            if num_threads > block_size:
                num_threads = self._largest_power_of_two_at_most(block_size)
            while num_threads > 1 and block_size % num_threads != 0:
                num_threads //= 2
            flat_config[num_threads_idx] = max(num_threads, 1)

        for flatten_idx, block_ids in self._cute_flatten_loop_groups:
            if flat_config[flatten_idx] is not True:
                continue
            group: list[tuple[int, int, bool, int]] = []
            for block_id in block_ids:
                block_size = flat_config[self._cute_block_index_by_id[block_id]]
                num_threads_idx = self._cute_num_thread_index_by_id[block_id]
                num_threads = flat_config[num_threads_idx]
                if (
                    type(block_size) is not int
                    or block_size <= 0
                    or type(num_threads) is not int
                ):
                    group = []
                    break
                resolved_threads = num_threads if num_threads > 0 else block_size
                group.append(
                    (num_threads_idx, block_size, num_threads == 0, resolved_threads)
                )
            if not group:
                continue
            thread_product = functools.reduce(
                operator.mul, (item[3] for item in group), 1
            )
            auto_positions = [i for i, item in enumerate(group) if item[2]]
            while thread_product > 1024 and auto_positions:
                largest_pos = max(auto_positions, key=lambda i: group[i][3])
                num_threads_idx, block_size, is_auto, resolved_threads = group[
                    largest_pos
                ]
                if resolved_threads <= 1:
                    auto_positions.remove(largest_pos)
                    continue
                next_threads = resolved_threads // 2
                while next_threads > 1 and block_size % next_threads != 0:
                    next_threads //= 2
                if next_threads == resolved_threads:
                    auto_positions.remove(largest_pos)
                    continue
                flat_config[num_threads_idx] = next_threads
                group[largest_pos] = (
                    num_threads_idx,
                    block_size,
                    is_auto,
                    next_threads,
                )
                thread_product = (thread_product // resolved_threads) * next_threads

        explicit_indices = [
            idx
            for idx, _ in self._cute_num_thread_block_pairs
            if type(flat_config[idx]) is int and cast("int", flat_config[idx]) > 0
        ]
        thread_product = functools.reduce(
            operator.mul,
            (cast("int", flat_config[idx]) for idx in explicit_indices),
            1,
        )
        while thread_product > 1024 and explicit_indices:
            largest_idx = max(
                explicit_indices,
                key=lambda idx: cast("int", flat_config[idx]),
            )
            largest = cast("int", flat_config[largest_idx])
            if largest <= 1:
                break
            flat_config[largest_idx] = largest // 2
            thread_product //= 2

    def flatten(self, config: Config) -> FlatConfig:
        """Inverse of unflatten: convert a Config to a FlatConfig."""
        result = self.default_flat()
        flat_fields = self.config_spec._flat_fields()
        for key, (indices, is_sequence) in self._key_to_flat_indices.items():
            if key not in config.config:
                continue
            value = config.config[key]
            if is_sequence:
                assert isinstance(value, list)
                field = flat_fields[key]
                assert isinstance(field, BlockIdSequence)
                encoded_values = field._encode_flat_values(self.config_spec, value)
                for idx, encoded_value in zip(indices, encoded_values, strict=True):
                    result[idx] = encoded_value
            else:
                assert len(indices) == 1
                result[indices[0]] = value
        self._repair_cute_num_threads(result)
        return result

    def unflatten(self, flat_values: FlatConfig) -> Config:
        """
        Convert a flat configuration back into a full configuration.

        Args:
            flat_values: The flat configuration values.

        Returns:
            The full configuration object.
        """

        def get_next_value(spec: ConfigSpecFragment) -> object:
            i = next(count)
            assert type(self.flat_spec[i]) is type(spec)
            return flat_values[i]

        assert len(flat_values) == len(self.flat_spec)
        self._repair_cute_num_threads(flat_values)
        count: itertools.count[int] = itertools.count()
        config = self.config_spec.flat_config(
            get_next_value,
            advanced_controls_files=self._advanced_controls_files,
        )
        assert next(count) == len(flat_values)
        config = self._apply_overrides(config)
        # Overrides may reintroduce pointer stores that break subtiled outputs
        self.config_spec.fix_epilogue_subtile_store_indexing(config.config)
        return config

    def block_numel(self, flat_config: FlatConfig) -> int:
        return functools.reduce(
            operator.mul,
            [cast("int", flat_config[i]) for i in self.block_size_indices],
            1,
        )

    def _shrink_for_numel_constraints(self, flat_config: FlatConfig) -> None:
        """Shrink block sizes in flat_config to satisfy numel constraints."""
        constraints = self.config_spec.tensor_numel_constraints
        if not constraints:
            return
        block_sizes = [cast("int", flat_config[i]) for i in self.block_size_indices]
        min_sizes = [
            max(self.flat_spec[i].get_minimum(), self.min_block_size)
            for i in self.block_size_indices
        ]
        shrink_block_sizes_for_numel_constraints(constraints, block_sizes, min_sizes)
        for idx, fi in enumerate(self.block_size_indices):
            flat_config[fi] = block_sizes[idx]

    def shrink_config(
        self, flat_config: FlatConfig, max_elements_per_thread: int
    ) -> None:
        """
        Fully random configs tend to run out of resources and tile a long time to compile.
        Here we shrink the config to a reasonable size.

        Args:
            flat_config: config to mutate in place
            max_elements_per_thread: maximum number of elements per thread
        """
        if self.num_warps_index < 0 or not self.block_size_indices:
            return
        num_threads = warps_to_threads(cast("int", flat_config[self.num_warps_index]))
        # Respect Triton's maximum tensor element limit
        triton_limit = TRITON_MAX_TENSOR_NUMEL
        theoretical_max_elements = max_elements_per_thread * num_threads
        max_elements = min(theoretical_max_elements, triton_limit)
        while self.block_numel(flat_config) > max_elements:
            changes = 0
            for i in self.block_size_indices:
                val = flat_config[i]
                assert isinstance(val, int)
                threshold = max(self.flat_spec[i].get_minimum(), self.min_block_size)
                if val // 2 >= threshold:
                    flat_config[i] = val // 2
                    changes += 1
            if changes == 0:
                break
        self._shrink_for_numel_constraints(flat_config)
        self._repair_cute_num_threads(flat_config)

    def default_flat(self) -> FlatConfig:
        """
        Retrieve the default flat configuration.

        Returns:
            The default flat configuration values.
        """
        config = [spec.default() for spec in self.flat_spec]
        self._shrink_for_numel_constraints(config)
        self._repair_cute_num_threads(config)
        return config

    def seed_flat_config_pairs(self) -> list[tuple[FlatConfig, Config]]:
        """Return ConfigSpec-provided seeds as flat and normalized configs.

        ``ConfigSpec.autotune_seed_configs()`` is compiler-owned and must
        return configs that match the live spec structurally. ``InvalidConfig``
        means overrides make a seed inapplicable; other flatten/unflatten
        exceptions are programming errors and intentionally surface.
        """
        result: list[tuple[FlatConfig, Config]] = []
        seen: set[Config] = set()
        for config in self.config_spec.autotune_seed_configs():
            try:
                flat = self.flatten(config)
                normalized = self.unflatten(flat)
            except InvalidConfig:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append((flat, normalized))
        return result

    def user_seed_flat_config_pairs(
        self,
        user_seed_configs: Sequence[Config],
        log_func: Callable[[str], None] | None = None,
    ) -> list[tuple[FlatConfig, Config]]:
        """Return user-provided seed configs as flat and normalized configs."""
        result: list[tuple[FlatConfig, Config]] = []
        seen: set[Config] = set()
        for i, config in enumerate(user_seed_configs):
            try:
                flat = self.flatten(config)
                normalized = self.unflatten(flat)
            except (
                InvalidConfig,
                ValueError,
                TypeError,
                KeyError,
                AssertionError,
            ) as e:
                if log_func is not None:
                    log_func(f"Failed to transfer autotune seed config {i + 1}: {e}")
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append((flat, normalized))
        return result

    def random_flat(self) -> FlatConfig:
        """
        Generate a random flat configuration.

        Returns:
            A random flat configuration.
        """

        with sync_seed(process_group_name=self.process_group_name):
            config = [spec.random() for spec in self.flat_spec]
            self.shrink_config(config, PowerOfTwoFragment(1, 2048, 32).random())
            self._repair_cute_num_threads(config)
            return config

    def random_config(self) -> Config:
        errors: dict[str, int] = {}
        for _ in range(64):
            try:
                return self.unflatten(self.random_flat())
            except InvalidConfig as e:
                msg = str(e)
                errors[msg] = errors.get(msg, 0) + 1
                continue
        summary = "; ".join(f"{msg} (x{n})" for msg, n in errors.items())
        raise InvalidConfig(
            f"failed to generate a valid random config after 64 attempts: {summary}"
        )

    def random_population_flat(
        self,
        n: int,
        *,
        user_seed_configs: Sequence[Config] = (),
        log_func: Callable[[str], None] | None = None,
    ) -> list[FlatConfig]:
        if n <= 0:
            return [self.default_flat()]
        default_flat = self.default_flat()
        result = [default_flat]

        # Initial population order is default -> user seed configs -> compiler seeds
        # -> random.  This preserves user seed priority without dropping built-in
        # backend/compiler seeds from ConfigSpec.autotune_seed_configs().
        for flat, _config in self.user_seed_flat_config_pairs(
            user_seed_configs, log_func
        ):
            if any(flat == existing for existing in result):
                continue
            result.append(flat)
            if len(result) >= n:
                return result[:n]

        for flat, _config in self.seed_flat_config_pairs():
            if any(flat == existing for existing in result):
                continue
            result.append(flat)
            if len(result) >= n:
                return result[:n]

        result.extend(self.random_flat() for _ in range(n - len(result)))
        return result

    def random_population(
        self,
        n: int,
        *,
        user_seed_configs: Sequence[Config] = (),
        log_func: Callable[[str], None] | None = None,
    ) -> list[Config]:
        result: list[Config] = []
        attempts = 0
        for flat in self.random_population_flat(
            n, user_seed_configs=user_seed_configs, log_func=log_func
        ):
            try:
                result.append(self.unflatten(flat))
            except InvalidConfig:
                attempts += 1
        # Retry to fill the population to the requested size
        while len(result) < n and attempts < 64:
            with contextlib.suppress(InvalidConfig):
                result.append(self.unflatten(self.random_flat()))
            attempts += 1
        return result

    def differential_mutation(
        self,
        x: FlatConfig,
        a: FlatConfig,
        b: FlatConfig,
        c: FlatConfig,
        crossover_rate: float,
    ) -> FlatConfig:
        """
        The main op in differential evolution, randomly combine `x` with `a + (b - c)`.
        """
        overridden = self.overridden_flat_indices
        result = [*x]
        mutated = False
        for i, spec in enumerate(self.flat_spec):
            if i not in overridden and random.random() < crossover_rate:
                result[i] = spec.differential_mutation(a[i], b[i], c[i])
                mutated = True
        if not mutated:
            eligible = [i for i in range(len(self.flat_spec)) if i not in overridden]
            if eligible:
                i = random.choice(eligible)
                result[i] = self.flat_spec[i].differential_mutation(a[i], b[i], c[i])
        # TODO(jansel): can this be larger? (too large and Triton compile times blow up)
        self.shrink_config(result, 8192)
        self._repair_cute_num_threads(result)
        return result

    def encode_config(self, flat_config: FlatConfig) -> list[float]:
        """
        Encode a flat configuration into a numerical vector for ML models.

        This is used by surrogate-assisted algorithms (e.g., DE-Surrogate) that need
        to represent configurations as continuous vectors for prediction models.

        Args:
            flat_config: The flat configuration values to encode.

        Returns:
            A list of floats representing the encoded configuration.
        """
        encoded: list[float] = []

        for flat_idx, spec in enumerate(self.flat_spec):
            value = flat_config[flat_idx]
            encoded_value = spec.encode(value)
            assert len(encoded_value) == spec.dim()
            encoded.extend(encoded_value)

        return encoded
