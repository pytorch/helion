"""Explicit independent argument banks for repeated CUDA graph calls."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING

import torch

from .. import exc
from ..runtime.settings import _env_get_bool
from .benchmarking import _BATCH_MEMORY_FRACTION
from .benchmarking import _BENCHMARK_CUDAGRAPH_ENV
from .benchmarking import _resolve_l2_flush_mode
from .kernel_args import _clone_tensor_leaves
from .kernel_args import _input_tree_leaves

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


class InputBankError(exc.InvalidAPIUsage):
    """A requested benchmark context cannot be honored; do not rank it as a config."""

    def __reduce__(self) -> tuple[object, ...]:
        # args already contains the formatted message and worker source location.
        # Unpickling must not run InvalidAPIUsage's formatting a second time.
        return _restore_input_bank_error, (type(self), self.args), self.__dict__


class InputBankMemoryError(InputBankError):
    """The bank allocation budget is exhausted before accepting a timing sample."""

    message = "{0}"


def _restore_input_bank_error(
    error_type: type[InputBankError], args: tuple[object, ...]
) -> InputBankError:
    error = RuntimeError.__new__(error_type)
    RuntimeError.__init__(error, *args)
    return error


def check_input_bank_mode(
    count: int, *, wall_clock: bool = False, l2_flush_mode: str | None = None
) -> None:
    """Never silently substitute a different timing context for bank rotation."""
    if count == 1:
        return
    if wall_clock or not _env_get_bool(_BENCHMARK_CUDAGRAPH_ENV, False):
        raise InputBankError(
            "autotune_benchmark_input_banks > 1 requires the built-in CUDA "
            "event timer and HELION_BENCHMARK_CUDAGRAPH=1"
        )
    if _resolve_l2_flush_mode(l2_flush_mode) not in ("replay", "off"):
        raise InputBankError(
            "autotune_benchmark_input_banks > 1 requires the replay or off L2 regime"
        )


@dataclass(frozen=True)
class InputBankCycle:
    """One indivisible cycle; no timer can observe a partial rotation.

    Retaining this tuple through graph capture keeps every call's output alive.
    A failed call raises instead of returning a shortened cycle for timing.
    """

    fn: Callable[..., object]
    banks: tuple[Sequence[object], ...]
    extra_input_bytes: int = 0
    memory_budget_bytes: int | None = None
    device: torch.device | None = None

    def __call__(self) -> tuple[object, ...]:
        return tuple(starmap(self.fn, self.banks))

    def warm(self, *, retained_cycles: int = 1) -> tuple[object, ...]:
        """Warm complete cycles and budget their simultaneously live outputs."""
        if retained_cycles < 1:
            raise ValueError("retained_cycles must be positive")

        def run() -> tuple[object, ...]:
            return tuple(output for _ in range(retained_cycles) for output in self())

        if self.memory_budget_bytes is None:
            return run()
        before = torch.cuda.memory_allocated(self.device)
        try:
            outputs = run()
        except torch.OutOfMemoryError as error:
            raise InputBankMemoryError(
                "benchmark input-bank outputs exceed device memory"
            ) from error
        retained = max(0, torch.cuda.memory_allocated(self.device) - before)
        if self.extra_input_bytes + retained > self.memory_budget_bytes:
            raise InputBankMemoryError(
                "benchmark input banks and retained outputs exceed the CUDA graph memory budget"
            )
        return outputs


def clone_input_banks(
    args: Sequence[object], count: int
) -> tuple[Sequence[object], ...]:
    """Bank zero is the worker's private input; other banks preserve all aliases."""
    return (args, *(_clone_tensor_leaves(args) for _ in range(count - 1)))


def _tensor_operands(args: Sequence[object]) -> list[torch.Tensor]:
    return [
        value for value in _input_tree_leaves(args) if isinstance(value, torch.Tensor)
    ]


def tensor_storage_bytes(tensors: Sequence[torch.Tensor]) -> int:
    """Count each aliased allocation once, including mixed-dtype views."""
    sizes: dict[tuple[torch.device, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        pointer = (tensor.device, storage.data_ptr())
        sizes[pointer] = max(sizes.get(pointer, 0), storage.nbytes())
    return sum(sizes.values())


def make_input_bank_cycle(
    fn: Callable[..., object], args: Sequence[object], count: int
) -> InputBankCycle:
    """Allocate banks before warmup/timing, using the graph pool's budget share."""
    tensors = _tensor_operands(args)
    if (
        not tensors
        or any(value.device.type != "cuda" for value in tensors)
        or len({value.device for value in tensors}) != 1
    ):
        raise InputBankError("rotating benchmark input banks require CUDA tensors")
    required = (count - 1) * tensor_storage_bytes(tensors)
    # Return unused allocator blocks before consulting the same free-memory
    # budget used by captured output batches. Live finalist copies stay alive.
    with torch.cuda.device(tensors[0].device):
        torch.cuda.empty_cache()
        free, _total = torch.cuda.mem_get_info(tensors[0].device)
    if required > free * _BATCH_MEMORY_FRACTION:
        raise InputBankMemoryError(
            f"{count} benchmark input banks need {required} additional bytes; "
            "they exceed the CUDA graph memory budget"
        )
    try:
        banks = clone_input_banks(args, count)
    except torch.OutOfMemoryError as error:
        raise InputBankMemoryError(
            "benchmark input banks exceed device memory"
        ) from error
    return InputBankCycle(
        fn, banks, required, int(free * _BATCH_MEMORY_FRACTION), tensors[0].device
    )
