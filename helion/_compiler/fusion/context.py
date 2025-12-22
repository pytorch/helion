"""Fusion context and specifications."""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode


@dataclass
class EpilogueSpec:
    """Specification for epilogue fusion."""
    epilogue_nodes: list["BaseSchedulerNode"]
    accumulator_name: str | set[str]

    @property
    def accumulator_names(self) -> set[str]:
        if isinstance(self.accumulator_name, set):
            return self.accumulator_name
        return {self.accumulator_name}


@dataclass
class PrologueSpec:
    """Specification for prologue fusion."""
    prologue_nodes: list["BaseSchedulerNode"]
    input_name: str


@dataclass
class FusionContext:
    """Holds fusion state during code generation."""
    epilogues: dict[str, list[EpilogueSpec]]
    prologues: dict[str, list[PrologueSpec]]
    store_map: dict[int, str] = field(default_factory=dict)
    epilogue_closures: dict[str, str] = field(default_factory=dict)
    prologue_closures: dict[str, str] = field(default_factory=dict)

    def register_closure(self, buffer_name: str, *, epilogue: bool = True) -> str:
        """Register an external buffer as a closure parameter."""
        closures = self.epilogue_closures if epilogue else self.prologue_closures
        prefix = "epilogue" if epilogue else "prologue"
        if buffer_name not in closures:
            closures[buffer_name] = f"{prefix}_closure_{len(closures)}"
        return closures[buffer_name]

    def get_epilogue_specs(self, tensor_name: str) -> list[EpilogueSpec]:
        return self.epilogues.get(tensor_name, [])

    def get_prologue_specs(self, input_name: str) -> list[PrologueSpec]:
        return self.prologues.get(input_name, [])

    def get_epilogue_for_store(self, store_index: int) -> list[EpilogueSpec]:
        if store_index not in self.store_map:
            return []
        return self.get_epilogue_specs(self.store_map[store_index])

    def get_all_epilogue_specs(self) -> list[EpilogueSpec]:
        return [spec for specs in self.epilogues.values() for spec in specs]

    @property
    def is_multi_output(self) -> bool:
        return bool(self.store_map)

    @property
    def all_closures(self) -> dict[str, str]:
        return {**self.epilogue_closures, **self.prologue_closures}


_current_ctx: ContextVar[FusionContext | None] = ContextVar("_fusion_ctx", default=None)


@contextlib.contextmanager
def fusion_context(
    epilogues: dict[str, list[EpilogueSpec]],
    prologues: dict[str, list[PrologueSpec]],
    store_index_to_buffer: dict[int, str] | None = None,
) -> Iterator[FusionContext]:
    """Context manager for fusion operations."""
    ctx = FusionContext(
        epilogues=epilogues,
        prologues=prologues,
        store_map=store_index_to_buffer or {},
    )
    token = _current_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _current_ctx.reset(token)


def get_current_context() -> FusionContext | None:
    return _current_ctx.get()
