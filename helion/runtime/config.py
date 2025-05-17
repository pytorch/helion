from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Literal
from typing import cast

from helion.autotuner.config_spec import DEFAULT_NUM_STAGES
from helion.autotuner.config_spec import DEFAULT_NUM_WARPS

IndexingLiteral = Literal["pointer", "tensor_descriptor", "block_ptr"]


class Config(Mapping[str, object]):
    config: dict[str, object]

    def __init__(
        self,
        *,
        # Core properties
        block_sizes: list[int | list[int]] | None = None,
        loop_orders: list[list[int]] | None = None,
        reduction_loops: list[int | None] | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
        l2_grouping: int | None = None,
        use_yz_grid: bool | None = None,
        indexing: IndexingLiteral | None = None,
        # For user-defined properties
        **kwargs: object,
    ) -> None:
        """
        Initialize a Config object.

        Args:
            block_sizes: Controls tile sizes for hl.tile invocations.
            loop_orders: Permutes iteration order of tiles.
            reduction_loops: Configures reduction loop behavior.
            num_warps: Number of warps per block.
            num_stages: Number of stages for software pipelining.
            l2_grouping: Reorders program IDs for L2 cache locality.
            use_yz_grid: Whether to use yz grid dimensions.
            indexing: Indexing strategy ("pointer", "tensor_descriptor", "block_ptr").
            **kwargs: Additional user-defined configuration parameters.
        """
        self.config = {}
        core_props = {
            "block_sizes": block_sizes,
            "loop_orders": loop_orders,
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "indexing": indexing,
            "l2_grouping": l2_grouping,
            "use_yz_grid": use_yz_grid,
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
        args = [f"{key}={value!r}" for key, value in self.config.items()]
        return f"Config({', '.join(args)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return self.config == other.config

    def __hash__(self) -> int:
        return hash(frozenset([(k, _list_to_tuple(v)) for k, v in self.config.items()]))

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
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a config from a JSON file."""
        return cls.from_json(Path(path).read_text())

    @property
    def block_sizes(self) -> list[int | list[int]]:
        return cast("list[int | list[int]]", self.config["block_sizes"])

    @property
    def loop_orders(self) -> list[list[int]]:
        return cast("list[list[int]]", self.config.get("loop_orders", []))

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
    def l2_grouping(self) -> int:
        return cast("int", self.config.get("l2_grouping", 1))

    @property
    def use_yz_grid(self) -> bool:
        return cast("bool", self.config.get("use_yz_grid", False))

    @property
    def indexing(self) -> IndexingLiteral:
        return self.config.get("indexing", "pointer")  # type: ignore


def _list_to_tuple(x: object) -> object:
    if isinstance(x, list):
        return tuple([_list_to_tuple(i) for i in x])
    return x
