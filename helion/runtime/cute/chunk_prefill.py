"""Host-wrapper construction and ABI checks for a proved fused recurrence."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ... import exc

if TYPE_CHECKING:
    from _thread import LockType
    from collections.abc import Callable
    from collections.abc import Sequence


def validate_plan(plan: dict[str, object]) -> None:
    if plan.get("device_abi") == 2:
        if (
            plan.get("threads") != 1024
            or plan.get("chunk_size") != 32
            or plan.get("numerical_policy") != "centered_bt32_fp32_rhs_v2"
            or plan.get("task_order", "identity")
            not in ("identity", "longest_first_precompute")
            or plan.get("schedule", "single") != "single"
        ):
            raise exc.BackendUnsupported(
                "cute", "unsupported centered BT32 schedule ABI"
            )
    elif (
        plan.get("device_abi") != 1
        or plan.get("threads") != 512
        or plan.get("chunk_size", 16) != 16
        or plan.get("numerical_policy", "native_bt16_bf16_rhs_v1")
        != "native_bt16_bf16_rhs_v1"
    ):
        raise exc.BackendUnsupported("cute", "unsupported fused prefill schedule ABI")
    if plan.get("task_order", "identity") not in (
        "identity",
        "longest_first",
        "longest_first_precompute",
    ):
        raise exc.BackendUnsupported("cute", "invalid fused prefill task order")
    if plan.get("schedule", "single") not in (
        "single",
        "prefix_tail_2",
        "prefix_tail_4",
    ):
        raise exc.BackendUnsupported("cute", "invalid fused prefill stream schedule")
    if plan.get("schedule", "single") != "single":
        for key in ("prefix_count", "sequence_groups"):
            value = plan.get(key)
            if type(value) is not int or value <= 0:
                raise exc.BackendUnsupported("cute", f"invalid fused prefill {key}")
        if cast("int", plan["sequence_groups"]) > cast("int", plan["sequences"]):
            raise exc.BackendUnsupported(
                "cute", "too many fused prefill sequence groups"
            )
    for key in ("heads", "sequences", "total_tokens"):
        value = plan.get(key)
        if type(value) is not int or value <= 0:
            raise exc.BackendUnsupported("cute", f"invalid fused prefill {key}")


def validate_args(plan: dict[str, object], args: Sequence[object]) -> None:
    from ..._compiler.cute.chunk_prefill import _disjoint_storage

    validate_plan(plan)
    keys = (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "a_log",
        "dt",
        "initial_state",
        "cu_seqlens",
        "out",
        "final_state",
    )
    tensors = []
    for key in keys:
        index = plan.get(f"{key}_idx")
        if type(index) is not int or not 0 <= index < len(args):
            raise exc.BackendUnsupported(
                "cute", "invalid fused prefill tensor argument"
            )
        tensor = args[index]
        if not isinstance(tensor, torch.Tensor):
            raise exc.BackendUnsupported(
                "cute", "fused prefill requires tensor arguments"
            )
        tensors.append(tensor)
    heads, sequences, tokens = plan["heads"], plan["sequences"], plan["total_tokens"]
    assert (
        isinstance(heads, int)
        and isinstance(sequences, int)
        and isinstance(tokens, int)
    )
    rows = heads * tokens
    activation = (rows, 128)
    state_shape = (sequences, heads, 128, 128)
    shapes = (
        activation,
        activation,
        activation,
        activation,
        (rows,),
        (heads,),
        (heads, 128),
        state_shape,
        (sequences + 1,),
        activation,
        state_shape,
    )
    dtypes = (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.float32,
        tensors[8].dtype,
        torch.bfloat16,
        torch.float32,
    )
    if tensors[8].dtype not in (torch.int32, torch.int64) or any(
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != tensors[0].device
        for tensor, shape, dtype in zip(tensors, shapes, dtypes, strict=True)
    ):
        raise exc.BackendUnsupported(
            "cute", "fused prefill tensor ABI does not match the proved region"
        )
    if plan.get("device_abi") == 2 and (
        tensors[8].dtype is not torch.int64
        or cast("int", heads) % 8
        or any(not tensor.is_contiguous() for tensor in tensors)
        or any(tensor.data_ptr() % 16 for tensor in tensors)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "centered BT32 requires contiguous 16-byte-aligned tensors, "
            "int64 cu_seqlens, and a head count divisible by 8",
        )
    if not _disjoint_storage(tensors):
        raise exc.BackendUnsupported(
            "cute", "fused prefill requires disjoint aligned output storage"
        )


def get_host(plan: dict[str, object]) -> Callable[..., object]:
    """Resolve the device implementation only from a validated numerical ABI."""
    validate_plan(plan)
    if plan.get("device_abi") == 2:
        from ..._compiler.cute.chunk_prefill_bt32.device import host
    else:
        from ..._compiler.cute.chunk_prefill_tmem import host
    return host


def append_host_call(body: list[str], plan: dict[str, object]) -> None:
    validate_plan(plan)
    if plan.get("device_abi") == 2:
        _append_bt32_host_call(body, plan)
        return
    heads, tokens = plan["heads"], plan["total_tokens"]
    assert isinstance(heads, int) and isinstance(tokens, int)

    def arg(key: str) -> str:
        index = plan[f"{key}_idx"]
        assert type(index) is int
        return f"arg{index}"

    def view(key: str, shape: tuple[int, ...], stride: tuple[int, ...]) -> str:
        name = f"_prefill_{key}"
        body.append(
            f"    {name} = cute.make_tensor({arg(key)}.iterator, "
            f"cute.make_layout({shape!r}, stride={stride!r}))"
        )
        return name

    activation_shape = (1, tokens, heads, 128)
    activation_stride = (128, heads * 128, 128, 1)
    q, k, v, gate, output = (
        view(key, activation_shape, activation_stride)
        for key in ("q", "k", "v", "g", "out")
    )
    beta = view("beta", (1, tokens, heads), (heads, heads, 1))
    call_args = (
        q,
        k,
        v,
        gate,
        arg("a_log"),
        arg("dt"),
        beta,
        arg("cu_seqlens"),
        "_prefill_order"
        if plan.get("task_order") == "longest_first_precompute"
        else "None",
        "None",
        arg("initial_state"),
        output,
        arg("final_state"),
        "stream",
        f"cutlass.Float32({arg('scale')})",
        "None",
        "None",
        "None",
        "cutlass.Int32(0)",
        "True",
        f"cutlass.Float32({arg('gate_scale')})",
        "512",
        "cutlass.BFloat16",
        repr(
            "identity"
            if plan.get("task_order") == "longest_first_precompute"
            else plan.get("task_order", "identity")
        ),
    )
    body.append("    _helion_cute_kernel_tag = 'chunk_prefill_sm100'")
    if plan.get("schedule", "single") == "single":
        body.append(f"    _helion_chunk_prefill_host({', '.join(call_args)})")
        return
    prefixes = cast("int", plan["prefix_count"])
    groups = cast("int", plan["sequence_groups"])
    sequences = cast("int", plan["sequences"])
    average = (tokens + sequences * prefixes - 1) // (sequences * prefixes)
    segment_tokens = ((average + 15) // 16) * 16
    for group in range(groups):
        begin, end = group * sequences // groups, (group + 1) * sequences // groups
        for stage in range(prefixes + 1):
            stage_args = list(call_args)
            stage_args[10] = (
                arg("initial_state")
                if stage == 0
                else f"_prefill_state{(stage - 1) % 2}"
            )
            stage_args[12] = (
                arg("final_state")
                if stage == prefixes
                else f"_prefill_state{stage % 2}"
            )
            stage_args[13] = f"_prefill_stream{group}"
            stage_args.extend(
                (
                    str(stage * segment_tokens),
                    str(segment_tokens if stage < prefixes else -1),
                    f"cutlass.Int32({begin})",
                    str(end - begin),
                )
            )
            body.append(f"    _helion_chunk_prefill_host({', '.join(stage_args)})")


def _append_bt32_host_call(body: list[str], plan: dict[str, object]) -> None:
    validate_plan(plan)
    heads, tokens = plan["heads"], plan["total_tokens"]
    assert isinstance(heads, int) and isinstance(tokens, int)

    def arg(key: str) -> str:
        index = plan[f"{key}_idx"]
        assert type(index) is int
        return f"arg{index}"

    def view(key: str, shape: tuple[int, ...], stride: tuple[int, ...]) -> str:
        name = f"_prefill_{key}"
        body.append(
            f"    {name} = cute.make_tensor({arg(key)}.iterator, "
            f"cute.make_layout({shape!r}, stride={stride!r}))"
        )
        return name

    activation_shape = (1, tokens, heads, 128)
    activation_stride = (128, heads * 128, 128, 1)
    q, k, v, gate, output = (
        view(key, activation_shape, activation_stride)
        for key in ("q", "k", "v", "g", "out")
    )
    beta = view("beta", (1, tokens, heads), (heads, heads, 1))
    call_args = (
        q,
        k,
        v,
        gate,
        arg("a_log"),
        arg("dt"),
        beta,
        arg("cu_seqlens"),
        "_prefill_order"
        if plan.get("task_order") == "longest_first_precompute"
        else "None",
        "None",
        arg("initial_state"),
        output,
        arg("final_state"),
        "stream",
        f"cutlass.Float32({arg('scale')})",
        "None",
        "None",
        "None",
        "0",
        "True",
        f"cutlass.Float32({arg('gate_scale')})",
        "1024",
        "cutlass.BFloat16",
        repr(
            "identity"
            if plan.get("task_order") == "longest_first_precompute"
            else plan.get("task_order", "identity")
        ),
    )
    body.extend(
        (
            "    _helion_cute_kernel_tag = 'chunk_prefill_sm100'",
            f"    _helion_chunk_prefill_host({', '.join(call_args)})",
        )
    )


@dataclass(eq=False)
class PrefillResources:
    """Mutable state, order and synchronization for one origin/capture context."""

    states: tuple[torch.Tensor, ...]
    streams: tuple[torch.cuda.Stream, ...]
    ready: torch.cuda.Event | None
    done: tuple[torch.cuda.Event, ...]
    lock: LockType = field(default_factory=threading.Lock, repr=False)
    order: torch.Tensor | None = None
    order_kernel: object | None = None

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.states, *((self.order,) if self.order is not None else ()))

    @property
    def device(self) -> torch.device:
        return self.tensors[0].device

    def fork(self, origin: torch.cuda.Stream) -> None:
        if self.ready is not None:
            self.ready.record(origin)
            for stream in self.streams:
                stream.wait_event(self.ready)

    def join(self, origin: torch.cuda.Stream) -> None:
        for stream, done in zip(self.streams, self.done, strict=True):
            done.record(stream)
            origin.wait_event(done)


def prefill_resources(
    cute_kernel: object, plan: dict[str, object], args: Sequence[object]
) -> PrefillResources:
    from .launcher import _cuda_stream_capture_context
    from .launcher import _track_cute_cuda_graph_cache_entry

    initial = cast("torch.Tensor", args[cast("int", plan["initial_state_idx"])])
    stream, capture_id = _cuda_stream_capture_context(initial.device)
    groups = (
        cast("int", plan["sequence_groups"])
        if plan.get("schedule", "single") != "single"
        else 0
    )
    precompute = plan.get("task_order") == "longest_first_precompute"
    key = (initial.device, stream, capture_id, tuple(initial.shape), groups, precompute)
    cache = cast(
        "OrderedDict[tuple[object, ...], PrefillResources]",
        cast("Any", cute_kernel).__dict__.setdefault(
            "_helion_cute_prefill_resources", OrderedDict()
        ),
    )
    resources = cache.get(key)
    if resources is None:
        order = None
        order_kernel = None
        if precompute:
            from ..._compiler.cute.sequence_order import stable_sequence_order

            order = torch.empty(
                (initial.shape[0],), device=initial.device, dtype=torch.int32
            )
            order_kernel = stable_sequence_order
        resources = PrefillResources(
            states=(torch.empty_like(initial), torch.empty_like(initial))
            if groups
            else (),
            streams=tuple(
                torch.cuda.Stream(device=initial.device) for _ in range(groups)
            ),
            ready=torch.cuda.Event() if groups else None,
            done=tuple(torch.cuda.Event() for _ in range(groups)),
            order=order,
            order_kernel=order_kernel,
        )
        cache[key] = resources
    cache.move_to_end(key)
    if capture_id is not None:
        # The retained value owns streams/events as well as scratch tensors.
        # Raw captures remain retained; managed captures release after reset.
        _track_cute_cuda_graph_cache_entry(
            cute_kernel,
            "_helion_cute_prefill_resources",
            key,
            resources,
            resources.tensors,
        )
    else:
        eager_keys = [key for key in cache if key[2] is None]
        while len(eager_keys) > 8:
            cache.pop(eager_keys.pop(0))
    return resources
