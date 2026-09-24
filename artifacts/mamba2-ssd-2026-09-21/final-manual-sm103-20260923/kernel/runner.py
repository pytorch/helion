"""Private four-stage native Mamba-forward runner with exact cast locations.

Public inputs are read-only. Internal stages mutate only their returned scratch
buffers. Scratch is per-stream for eager calls and per-capture for CUDA graphs;
every invocation allocates a fresh output. No public operation dispatch changes.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING
from typing import Any

from helion_kernel_library.configs import get_platform
import torch

from . import kernels
import helion

if TYPE_CHECKING:
    from collections.abc import Callable

Inputs = tuple[torch.Tensor, ...]
StageArgs = tuple[Any, ...]
StageResult = torch.Tensor | tuple[torch.Tensor, ...]
Configs = dict[str, dict[str, Any]]
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
HELION_KERNELS = kernels.HELION_KERNELS
HELION_REFERENCES = kernels.HELION_REFERENCES


def _config_key(inputs: Inputs) -> dict[str, int | str]:
    x, _dt, _a, b, _c, _d = inputs
    return {
        "backend": "cute",
        "dtype": str(x.dtype).removeprefix("torch."),
        "batch": x.shape[0],
        "seq_len": x.shape[1],
        "heads": x.shape[2],
        "head_dim": x.shape[3],
        "groups": b.shape[2],
        "dstate": b.shape[3],
        "chunk_size": 128,
    }


def _validate_inputs(inputs: Inputs, chunk_size: int = 128) -> None:
    if chunk_size != 128:
        raise ValueError("The CuTe SSD experiment currently supports chunk_size=128")
    x, _dt, _a, b, _c, _d = inputs
    _batch, length, heads, _dim = x.shape
    groups, state = b.shape[2:]
    if heads % groups or state != 128 or length % 128:
        raise ValueError(
            "CuTe SSD requires heads divisible by groups, state128, and full chunk128"
        )
    if any(not value.is_contiguous() for value in inputs):
        raise ValueError("CuTe SSD requires contiguous native inputs")


def _meta(inputs: Inputs) -> tuple[tuple[Any, ...], ...]:
    return tuple((v.shape, v.stride(), v.dtype, v.device) for v in inputs)


def load_pretuned_configs(inputs: Inputs) -> tuple[Configs, Path]:
    path = _CONFIG_DIR / f"{get_platform(inputs[0].device)}.json"
    key = _config_key(inputs)
    for entry in json.loads(path.read_text()):
        if entry["key"] == key:
            return entry["configs"], path
    raise KeyError(f"No pre-tuned large CuTe SSD config for {key} in {path}")


def save_pretuned_configs(
    inputs: Inputs, configs: Configs, *, autotune: dict[str, Any]
) -> Path:
    """Save four native stages in this artifact only; no public configs touched."""
    if set(configs) != {kernel.fn.__name__ for kernel in HELION_KERNELS}:
        raise ValueError("Expected exactly the four native Mamba stage configs")
    path = _CONFIG_DIR / f"{get_platform(inputs[0].device)}.json"
    key = _config_key(inputs)
    entry = {"key": key, "configs": configs, "autotune": autotune}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        contents = handle.read()
        entries = json.loads(contents) if contents else []
        entries = [existing for existing in entries if existing["key"] != key]
        entries.append(entry)
        entries.sort(key=lambda value: json.dumps(value["key"], sort_keys=True))
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(entries, indent=2) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
    return path


class Workspace:
    """Baseline-typed scratch, independent of CUDA stream bookkeeping."""

    def __init__(self, inputs: Inputs, *, tuning: bool = False) -> None:
        _validate_inputs(inputs)
        x, _dt, _a, b, _c, _d = inputs
        batch, length, heads, dim = x.shape
        groups, state = b.shape[2:]
        chunks = length // 128
        self.native_shape = x.shape
        self.out_shape = tuple(x.shape)

        def alloc(shape: tuple[int, ...], dtype: torch.dtype = x.dtype) -> torch.Tensor:
            # Only tuning needs initialized untouched regions. NaNs make the
            # tuner's mutation comparison reject an otherwise unchanged buffer.
            factory = torch.zeros if tuning else torch.empty
            return factory(shape, dtype=dtype, device=x.device)

        self.terminal = alloc((batch, chunks, heads), torch.float32)
        self.prefix = alloc((batch, chunks, heads, 128), torch.float32)
        self.rawdt = alloc((batch, chunks, heads, 128), torch.float32)
        # Native P,N storage makes X.T @ weighted_B a genuine M64 dot.
        # Carry/output consume this layout without a global transpose or copy.
        self.states = alloc((batch, chunks, heads, dim, state), torch.float32)
        self.incoming = alloc((batch, chunks, heads, dim, state))
        self.cb = alloc((batch, chunks, groups, 128, 128), torch.float32)

    def stage_args(self, inputs: Inputs) -> tuple[StageArgs, ...]:
        x, dt, a, b, c, d = inputs
        # Every public tensor comes from this invocation, not workspace creation.
        return (
            (x, b, dt, a, 128),
            (self.states, self.terminal, self.incoming),
            (b, c, 128),
            (x, c, d, self.prefix, self.rawdt, self.cb, self.incoming, 128),
        )

    def accept_result(self, index: int, result: StageResult) -> None:
        if index == 0:
            # All four are genuine fresh FP32 allocations returned by state.
            assert isinstance(result, tuple) and len(result) == 4
            assert all(isinstance(value, torch.Tensor) for value in result)
            self.states, self.terminal, self.prefix, self.rawdt = result
        elif index == 2:
            assert isinstance(result, torch.Tensor)
            self.cb = result


class _StreamRunner:
    def __init__(
        self,
        inputs: Inputs,
        configs: Configs | None,
        *,
        autotune: bool = False,
        settings: helion.Settings | None = None,
        compiled_fns: list[Callable[..., Any]] | None = None,
        stage_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.meta = _meta(inputs)
        self.stream = torch.cuda.current_stream(inputs[0].device)
        self.lock = threading.Lock()
        self.workspace = Workspace(inputs, tuning=autotune)
        self.configs: Configs = {}
        self.autotune_stage_seconds: dict[str, float] = {}
        self.autotune_stage_seeds: dict[str, int] = {}
        self.fns: list[Callable[..., Any]] = []
        self.generated_code: dict[str, str] = {}
        if compiled_fns is not None:
            assert configs is not None
            self.configs, self.fns = configs, compiled_fns
            return
        for index, (kernel, baseline) in enumerate(
            zip(HELION_KERNELS, HELION_REFERENCES, strict=True)
        ):
            values = self.workspace.stage_args(inputs)[index]
            base_settings: helion.Settings = settings or kernel.settings
            options: dict[str, Any] = base_settings.to_dict()
            stage_seed = base_settings.autotune_random_seed + index
            options.update(
                backend="cute", static_shapes=True, autotune_baseline_fn=baseline
            )
            if autotune:
                options["autotune_effort"] = "full"
                # Each stage receives its own fresh deterministic seed from the
                # caller's run seed; do not replace generic compiler seeds.
                options["autotune_random_seed"] = stage_seed
            private = helion.aot_kernel(kernel.fn, **options)
            name = kernel.fn.__name__
            bound = private.bind(values)
            if autotune:
                started = time.perf_counter()
                config = bound.autotune(values, force=True)
                self.autotune_stage_seconds[name] = time.perf_counter() - started
                self.autotune_stage_seeds[name] = stage_seed
            else:
                if configs is None or name not in configs:
                    raise KeyError(f"Missing pre-tuned large CuTe SSD stage: {name}")
                config = helion.Config(**configs[name])
            fn = bound.compile_config(config)
            self.workspace.accept_result(index, fn(*values))
            self.fns.append(fn)
            self.configs[name] = dict(config)
            self.generated_code[name] = bound.to_code(config)
            if stage_callback is not None:
                stage_callback(
                    {
                        "stage": index,
                        "kernel": name,
                        "config": dict(config),
                        "autotune_seconds": self.autotune_stage_seconds.get(name),
                        "autotune_seed": self.autotune_stage_seeds.get(name),
                        "autotune_force": autotune,
                        "code": self.generated_code[name],
                    }
                )

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        if _meta(inputs) != self.meta:
            raise ValueError("Input shape, strides, dtype or device changed")
        with self.lock:
            for index, fn in enumerate(self.fns):
                values = self.workspace.stage_args(inputs)[index]
                out = fn(*values)
                self.workspace.accept_result(index, out)
            return out.reshape(self.workspace.native_shape)


class Mamba2SSDCuteNative:
    def __init__(
        self,
        inputs: Inputs,
        *,
        chunk_size: int = 128,
        configs: Configs | None = None,
        autotune: bool = False,
        settings: helion.Settings | None = None,
        stage_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        _validate_inputs(inputs, chunk_size)
        self._device_index = inputs[0].device.index
        self.config_file: str | None = None
        if not autotune and configs is None:
            configs, path = load_pretuned_configs(inputs)
            self.config_file = str(path)
        primary = _StreamRunner(
            inputs,
            configs,
            autotune=autotune,
            settings=settings,
            stage_callback=stage_callback,
        )
        self.configs = primary.configs
        self._compiled_fns = primary.fns
        self.autotune_stage_seconds = primary.autotune_stage_seconds
        self.autotune_stage_seeds = primary.autotune_stage_seeds
        self.generated_code = primary.generated_code
        self._meta = primary.meta
        self._streams = {
            torch._C._cuda_getCurrentRawStream(self._device_index): primary
        }
        self._stream_lock = threading.Lock()

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        if _meta(inputs) != self._meta:
            raise ValueError("Input shape, strides, dtype or device changed")
        if torch.cuda.is_current_stream_capturing():
            # Capture-local scratch keeps independently captured graphs from
            # racing even when captured on the same stream. No warmup launches.
            runner = _StreamRunner(
                inputs, self.configs, compiled_fns=self._compiled_fns
            )
            return runner(*inputs)
        stream = torch._C._cuda_getCurrentRawStream(self._device_index)
        with self._stream_lock:
            runner = self._streams.get(stream)
            if runner is None:
                runner = _StreamRunner(
                    inputs, self.configs, compiled_fns=self._compiled_fns
                )
                self._streams[stream] = runner
        return runner(*inputs)
