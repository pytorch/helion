"""The four providers, each bound to one cell's inputs.

Every provider runs exactly the operator module's math on the same host
inputs.  None falls back to another implementation: a failure is recorded as
that provider's result.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast

import torch
from torch.utils._pytree import tree_map

from .inputs import from_mlx
from .inputs import to_mlx

if TYPE_CHECKING:
    from types import ModuleType

    import mlx.core as mx


def mlx_available() -> bool:
    return importlib.util.find_spec("mlx") is not None


@dataclass
class Bound:
    """One provider's call on one cell, plus how to complete and read it."""

    call: Callable[[], object]
    complete: Callable[[object], None]
    sync: Callable[[], None]
    to_host: Callable[[object], object]
    events: Callable[[], tuple[torch.mps.Event, torch.mps.Event]] | None


def _torch_sync() -> None:
    torch.mps.synchronize()


def _mps_events() -> tuple[torch.mps.Event, torch.mps.Event]:
    return torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)


def _torch_bound(call: Callable[[], object]) -> Bound:
    return Bound(
        call=call,
        complete=lambda out: _torch_sync(),
        sync=_torch_sync,
        to_host=lambda out: tree_map(lambda t: t.detach().cpu(), out),
        events=_mps_events,
    )


def _mlx_bound(call: Callable[[], object]) -> Bound:
    import mlx.core as mx

    return Bound(
        call=call,
        complete=mx.eval,
        sync=mx.synchronize,
        to_host=lambda out: tree_map(from_mlx, out, is_leaf=_is_mlx_array),
        events=None,
    )


def _is_mlx_array(value: object) -> bool:
    import mlx.core as mx

    return isinstance(value, mx.array)


def torch_call(
    op: ModuleType, fn: Callable[..., object], inputs: dict[str, torch.Tensor]
) -> Callable[[], object]:
    """A zero-argument call of ``fn`` (a forward) on ``inputs``.

    For a backward operator the forward runs once here, and each call computes
    the gradients of ``op.GRAD_WRT`` for the cotangent ``dy`` through the
    retained graph, as TritonBench times backward operators.
    """
    if op.MODE == "fwd":
        args = tuple(inputs.values())
        return lambda: fn(*args)
    leaves = {
        name: t.detach().requires_grad_(name in op.GRAD_WRT)
        for name, t in inputs.items()
        if name != "dy"
    }
    out = cast("torch.Tensor", fn(*leaves.values()))
    wrt = [leaves[name] for name in op.GRAD_WRT]
    dy = inputs["dy"]
    return lambda: torch.autograd.grad(out, wrt, dy, retain_graph=True)


def bind(
    provider: str, op: ModuleType, host: dict[str, torch.Tensor], device: dict
) -> Bound:
    """Bind ``provider`` to a cell.  ``device`` caches converted inputs."""
    if provider == "mlx":
        if "mlx" not in device:
            device["mlx"] = {name: to_mlx(t) for name, t in host.items()}
        args = tuple(device["mlx"].values())
        fn = op.mlx if op.MODE == "fwd" else mlx_grads(op, tuple(device["mlx"]))
        return _mlx_bound(lambda: fn(*args))
    if "mps" not in device:
        device["mps"] = {name: t.to("mps") for name, t in host.items()}
    inputs = device["mps"]
    if provider == "torch_eager":
        return _torch_bound(torch_call(op, op.eager, inputs))
    if provider == "torch_compile":
        torch._dynamo.reset()
        if op.MODE == "bwd":
            # Replaying a retained graph needs buffers the backward did not
            # donate; TritonBench sets the same for its backward compiles.
            torch._functorch.config.donated_buffer = False
        # TritonBench's compile variants use max-autotune without CUDA graphs.
        compiled = torch.compile(
            op.eager, fullgraph=True, mode="max-autotune-no-cudagraphs"
        )
        return _torch_bound(torch_call(op, compiled, inputs))
    assert provider == "helion", provider
    # Like benchmarks/run.py: each input gets fresh bound kernels and static
    # shapes.  The backend is pinned here, not through HELION_BACKEND, so an
    # import made before the benchmark started cannot select another one.
    for kernel in op.HELION_KERNELS:
        kernel.reset()
        kernel.settings.static_shapes = True
        kernel.settings.backend = "metal"
    return _torch_bound(torch_call(op, op.helion, inputs))


def mlx_grads(op: ModuleType, names: tuple[str, ...]) -> Callable[..., object]:
    """MLX's backward for a backward operator: ``mx.vjp`` of the forward.

    MLX keeps no graph between calls, so each call traces the forward again
    and evaluates whatever of it the vjp needs; that is the cost an MLX user
    pays for gradients.
    """
    import mlx.core as mx

    primal_names = [name for name in names if name != "dy"]
    wrt = [primal_names.index(name) for name in op.GRAD_WRT]

    def grads(*args: mx.array) -> tuple[mx.array, ...]:
        primals = list(args[: len(primal_names)])
        dy = args[names.index("dy")]

        def forward(*diff: mx.array) -> mx.array:
            full = list(primals)
            for i, value in zip(wrt, diff, strict=True):
                full[i] = value
            return op.mlx(*full)

        _, vjps = mx.vjp(forward, [primals[i] for i in wrt], [dy])
        return tuple(vjps)

    return grads


def helion_evidence(op: ModuleType) -> list[dict[str, str]]:
    """Backend and config of each Helion kernel bound for this cell.

    ``bind`` resets the kernels, so everything bound since ran this cell.  An
    empty list means no Helion kernel ran.
    """
    return [
        {
            "kernel": kernel.name,
            "backend": bound.env.backend.name,
            "config": repr(bound._config),
            "autotune_effort": kernel.settings.autotune_effort,
        }
        for kernel in op.HELION_KERNELS
        for bound in kernel._bound_kernels.values()
    ]


def floor_call(runtime: str) -> tuple[Callable[[], object], Callable[[object], None]]:
    """The smallest call and completion for a runtime, for ``sync_floor_ms``."""
    if runtime == "mlx":
        import mlx.core as mx

        one = mx.ones(1)
        mx.eval(one)
        return lambda: one + 1, mx.eval
    one = torch.ones(1, device="mps")
    return lambda: one + 1, lambda out: _torch_sync()


def runtime_of(provider: str) -> str:
    return "mlx" if provider == "mlx" else "torch"
