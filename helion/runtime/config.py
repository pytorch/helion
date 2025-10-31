from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Literal
from typing import cast
import uuid

from ..autotuner.config_spec import DEFAULT_NUM_STAGES
from ..autotuner.config_spec import DEFAULT_NUM_WARPS

IndexingLiteral = Literal["pointer", "tensor_descriptor", "block_ptr"]
PidTypeLiteral = Literal["flat", "xyz", "persistent_blocked", "persistent_interleaved"]
EvictionPolicyLiteral = Literal["", "first", "last"]


class Config(Mapping[str, object]):
    config: dict[str, object]

    def __init__(
        self,
        *,
        # Core properties
        block_sizes: list[int] | None = None,
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
        load_eviction_policies: list[EvictionPolicyLiteral] | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
        pid_type: PidTypeLiteral | None = None,
        indexing: IndexingLiteral | list[IndexingLiteral] | None = None,
        # For user-defined properties
        **kwargs: object,
    ) -> None:
        """
        Initialize a Config object.

        Args:
            block_sizes: Controls tile sizes for hl.tile invocations.
            loop_orders: Permutes iteration order of tiles.
            l2_groupings: Reorders program IDs for L2 cache locality.
            reduction_loops: Configures reduction loop behavior.
            range_unroll_factors: Loop unroll factors for tl.range calls.
            range_warp_specializes: Warp specialization for tl.range calls.
            range_num_stages: Number of stages for tl.range calls.
            range_multi_buffers: Controls disallow_acc_multi_buffer for tl.range calls.
            range_flattens: Controls flatten parameter for tl.range calls.
            static_ranges: Whether to use tl.static_range instead tl.range.
            load_eviction_policies: Eviction policies for load operations ("", "first", "last").
            num_warps: Number of warps per block.
            num_stages: Number of stages for software pipelining.
            pid_type: Program ID type strategy ("flat", "xyz", "persistent_blocked", "persistent_interleaved").
            indexing: Indexing strategy for load and store operations. Can be:
                - A single strategy string (all loads/stores use this strategy):
                  indexing="block_ptr"  # backward compatible
                - A list of strategies (one per load/store operation, must specify all):
                  indexing=["pointer", "block_ptr", "tensor_descriptor"]
                - Empty/omitted (all loads/stores default to "pointer")
                Valid strategies: "pointer", "tensor_descriptor", "block_ptr"
            **kwargs: Additional user-defined configuration parameters.
        """
        self.config = {}
        core_props = {
            "block_sizes": block_sizes,
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
            "load_eviction_policies": load_eviction_policies,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "indexing": indexing,
            "pid_type": pid_type,
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
    def from_json(cls, json_str: str) -> Config:
        """Create a Config object from a JSON string."""
        config_dict = json.loads(json_str)
        return cls(**config_dict)  # Changed to use dictionary unpacking

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
    def flatten_loops(self) -> list[bool]:
        return cast("list[bool]", self.config.get("flatten_loops", []))

    @property
    def reduction_loops(self) -> list[int | None]:
        return cast("list[int | None]", self.config.get("reduction_loops", []))

    @property
    def num_warps(self) -> int:
        return cast("int", self.config.get("num_warps", DEFAULT_NUM_WARPS))

    @property
    def num_stages(self) -> int:
        return cast("int", self.config.get("num_stages", DEFAULT_NUM_STAGES))

    @property
    def l2_groupings(self) -> list[int]:
        return cast("list[int]", self.config.get("l2_groupings", []))

    @property
    def pid_type(self) -> PidTypeLiteral:
        return cast("PidTypeLiteral", self.config.get("pid_type", "flat"))

    @property
    def range_unroll_factors(self) -> list[int]:
        return cast("list[int]", self.config.get("range_unroll_factors", []))

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
    def load_eviction_policies(self) -> list[EvictionPolicyLiteral]:
        return cast(
            "list[EvictionPolicyLiteral]", self.config.get("load_eviction_policies", [])
        )

    @property
    def indexing(self) -> IndexingLiteral | list[IndexingLiteral]:
        return cast(
            "IndexingLiteral | list[IndexingLiteral]", self.config.get("indexing", [])
        )


def _to_hashable(x: object) -> object:
    if isinstance(x, list):
        return tuple([_to_hashable(i) for i in x])
    if isinstance(x, dict):
        return tuple(sorted([(k, _to_hashable(v)) for k, v in x.items()]))
    return x
