"""Pipeline-local configuration dispatch, independent of global AOT state."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Mapping
import contextlib
import contextvars
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any

import torch

from .config import Config

if TYPE_CHECKING:
    from .kernel import BoundKernel
    from .kernel import Kernel

PipelineKeyFunction = Callable[["Kernel[Any]", tuple[object, ...]], object]
PipelineInitialConfig = Callable[["Kernel[Any]", tuple[object, ...]], Config]


class PipelineBudgetExhausted(RuntimeError):
    """Stop between complete pipeline measurements, retaining the incumbent."""


def argument_metadata(value: object) -> object:
    """Serializable metadata, never tensor values or storage addresses."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("Pipeline tuning supports strided tensor arguments")
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "storage_offset": value.storage_offset(),
        }
    if isinstance(value, (tuple, list)):
        return [argument_metadata(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Pipeline argument dictionaries require string keys")
        return {key: argument_metadata(item) for key, item in sorted(value.items())}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    raise TypeError(f"Unsupported pipeline argument type: {type(value).__name__}")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class PipelineConfig:
    """One coherent bundle; applying it never replaces a kernel's AOT config.

    Keys identify kernel source and runtime argument metadata. Use ``activate``
    around ordinary pipeline calls, including graph capture. Unseen stage keys
    are rejected unless the caller explicitly allows new stages.
    """

    def __init__(
        self,
        configs: Mapping[str, Config] | None = None,
        *,
        key_fn: PipelineKeyFunction | None = None,
    ) -> None:
        self._key_fn = key_fn
        self.configs: Mapping[str, Config] = MappingProxyType(
            {
                key: Config(**copy.deepcopy(dict(config)))
                for key, config in (configs or {}).items()
            }
        )

    def with_config(self, key: str, config: Config) -> PipelineConfig:
        return PipelineConfig({**self.configs, key: config}, key_fn=self._key_fn)

    def copy(self) -> PipelineConfig:
        return PipelineConfig(self.configs, key_fn=self._key_fn)

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            key: copy.deepcopy(dict(config))
            for key, config in sorted(self.configs.items())
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Mapping[str, object]],
        *,
        key_fn: PipelineKeyFunction | None = None,
    ) -> PipelineConfig:
        return cls(
            {key: Config(**dict(config)) for key, config in value.items()},
            key_fn=key_fn,
        )

    def digest(self) -> str:
        return _digest(self.to_dict())

    @contextlib.contextmanager
    def activate(
        self,
        *,
        key_fn: PipelineKeyFunction | None = None,
        initial_config: PipelineInitialConfig | None = None,
        allow_new: bool = False,
    ) -> Generator[PipelineScope, None, None]:
        scope = PipelineScope(
            self,
            key_fn=key_fn or self._key_fn,
            initial_config=initial_config,
            allow_new=allow_new,
        )
        with scope.activate():
            yield scope


@dataclass
class PipelineStage:
    key: str
    kernel: Kernel[Any]
    bound: BoundKernel[Any]
    args: tuple[object, ...]
    config: Config
    source_hash: str
    argument_metadata: object
    config_source: str = "bundle"
    config_space_identity: tuple[str, str] = ("", "")

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kernel": f"{self.kernel.fn.__module__}.{self.kernel.fn.__qualname__}",
            "config": dict(self.config),
            "source_hash": self.source_hash,
            "arguments": self.argument_metadata,
            "config_source": self.config_source,
            "config_space_identity": list(self.config_space_identity),
        }


_active_pipeline: contextvars.ContextVar[PipelineScope | None] = contextvars.ContextVar(
    "helion_active_pipeline", default=None
)


def _saved_config(
    kernel: Kernel[Any], bound: BoundKernel[Any], args: tuple[object, ...]
) -> tuple[Config, str]:
    if bound._config is not None:
        return bound._config, "bound_kernel"
    if kernel.configs:
        return kernel.configs[0], "kernel_configs"
    if kernel.settings.autotune_cache == "AOTAutotuneCache":
        # Resolve only an existing heuristic. Constructing the AOT cache would
        # permit collect/evaluate fallbacks to launch a nested autotuner.
        from ..autotuner.aot_cache import find_heuristic_file
        from ..autotuner.aot_kernel import _flatten_key_value

        path = find_heuristic_file(
            kernel.fn.__code__.co_filename, kernel_name=kernel.name
        )
        if path is not None:
            spec = importlib.util.spec_from_file_location("helion_pipeline_aot", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load pipeline AOT heuristic: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            selector = getattr(module, f"autotune_{kernel.name}", None)
            user_key = getattr(kernel, "_aot_user_key", None)
            key_args = _flatten_key_value(user_key(*args)) if user_key else args
            if selector is not None:
                try:
                    selected = selector(*key_args)
                except KeyError:
                    # An exact saved selector may not contain shapes introduced
                    # by a candidate's new topology. Use an explicit default;
                    # never enter the AOT collection/autotuning fallback.
                    selected = None
                if selected is not None:
                    return Config(**dict(selected)), f"saved_aot:{path}"
    return bound.config_spec.autotune_reference_config(), "reference_default"


class PipelineScope:
    """Resolve and compile stages once, then freeze for whole-graph measurement."""

    def __init__(
        self,
        bundle: PipelineConfig,
        *,
        key_fn: PipelineKeyFunction | None = None,
        initial_config: PipelineInitialConfig | None = None,
        allow_new: bool = True,
    ) -> None:
        self.configs = dict(bundle.configs)
        self._config_sources = dict.fromkeys(bundle.configs, "bundle")
        self.key_fn = key_fn or bundle._key_fn
        self.initial_config = initial_config
        self.allow_new = allow_new
        self.stages: dict[str, PipelineStage] = {}
        self.trace: list[str] = []
        self._sources: dict[int, str] = {}
        self._bound_spaces: dict[int, tuple[str, str]] = {}
        self._compiled: dict[tuple[str, int, str], Callable[..., object]] = {}
        self._frozen = False

    @property
    def bundle(self) -> PipelineConfig:
        return PipelineConfig(self.configs, key_fn=self.key_fn)

    def freeze(self) -> None:
        self.allow_new = False
        self._frozen = True

    @contextlib.contextmanager
    def activate(self) -> Generator[PipelineScope, None, None]:
        if _active_pipeline.get() is not None:
            raise RuntimeError("Nested pipeline configuration scopes are not supported")
        token = _active_pipeline.set(self)
        try:
            yield self
        finally:
            _active_pipeline.reset(token)

    def call(self, kernel: Kernel[Any], args: tuple[object, ...]) -> object:
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "Pipeline configuration scopes require eager host dispatch"
            )
        normalized = kernel.normalize_args(*args)
        bound = kernel.bind(normalized)
        if bound.env.process_group_name is not None or kernel.settings.distributed:
            raise ValueError("Distributed pipeline tuning is not supported")
        source_hash = self._sources.get(id(kernel))
        if source_hash is None:
            source_hash = hashlib.sha256(kernel.kernel_source().encode()).hexdigest()
            self._sources[id(kernel)] = source_hash
        metadata = argument_metadata(normalized)
        space = self._bound_spaces.get(id(bound))
        if space is None:
            space = (
                bound.env.backend.name,
                bound.config_spec.structural_fingerprint_hash(
                    advanced_controls_files=bound.settings.autotune_search_acf or None
                ),
            )
            self._bound_spaces[id(bound)] = space
        sharing_key = (
            self.key_fn(kernel, normalized) if self.key_fn else (space, metadata)
        )
        key = f"{kernel.fn.__module__}.{kernel.fn.__qualname__}:" + _digest(
            (source_hash, sharing_key)
        )
        if key in self.stages and self.stages[key].config_space_identity != space:
            raise ValueError(
                "A shared pipeline key aliases incompatible backends or config spaces"
            )
        if key not in self.configs:
            if not self.allow_new:
                raise RuntimeError(
                    f"Unseen pipeline stage key after preparation: {key}"
                )
            if self.initial_config is not None:
                selected = self.initial_config(kernel, normalized)
                source = "initial_config"
            else:
                selected, source = _saved_config(kernel, bound, normalized)
            self.configs[key] = Config(**copy.deepcopy(dict(selected)))
            self._config_sources[key] = source
        config = self.configs[key]
        compilation_key = key, id(bound), config.to_json()
        if compilation_key not in self._compiled:
            if self._frozen:
                raise RuntimeError("A pipeline stage required compilation after freeze")
            self._compiled[compilation_key] = bound.compile_config(
                config, allow_print=False
            )
        stage = PipelineStage(
            key,
            kernel,
            bound,
            normalized,
            config,
            source_hash,
            metadata,
            self._config_sources[key],
            space,
        )
        self.stages.setdefault(key, stage)
        self.trace.append(key)
        return self._compiled[compilation_key](*normalized)
