"""Disk-backed runners which materialize ordinary bindings on frontend access."""

from __future__ import annotations

import dataclasses
import threading
from typing import TYPE_CHECKING
from typing import Generic
from typing import TypeVar
import weakref

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_map
from torch.utils._pytree import tree_map_only

from .._argument_device import _canonicalize_argument_device
from .._argument_device import _find_argument_device as _find_device
from .._compiler.backend_registry import get_backend_class
from .config import Config
from .cute_structural_config import CuteStructuralConfig
from .generated_code_cache import _argument_key
from .generated_code_cache import compiled_kernel_cache_key
from .kernel import BoundKernel
from .kernel import _input_tensor_aliases
from .kernel import _load_code

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from typing import Hashable

    from .kernel import CompiledConfig
    from .kernel import ConfigLike
    from .kernel import Kernel
    from .kernel import _CompilerSeedSpecializationExtractor
    from .settings import Settings

_R = TypeVar("_R")


@dataclasses.dataclass(frozen=True)
class _CachedTensorArgument:
    reference: weakref.ReferenceType[torch.Tensor]
    identity: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool
    static_indices: tuple[int, ...]

    @classmethod
    def capture(cls, value: torch.Tensor) -> _CachedTensorArgument:
        return cls(
            weakref.ref(value),
            id(value),
            tuple(int(dim) for dim in value.shape),
            tuple(int(stride) for stride in value.stride()),
            value.dtype,
            value.device,
            value.requires_grad,
            tuple(getattr(value, "_dynamo_static_indices", ())),
        )


class _CachedBoundKernel(Generic[_R]):
    """A disk runner that delegates frontend operations to a normal binding.

    The private input guard protects only this runner. It never participates in
    the kernel's specialization schema or the autotuner's cache identity.
    """

    def __init__(
        self,
        kernel: Kernel[_R],
        args: tuple[object, ...],
        signature: tuple[Hashable, ...],
        artifact: tuple[Config, Config, str],
    ) -> None:
        self.kernel = kernel
        self._base_spec_key = signature
        self._reset_generation = kernel._reset_generation
        self._cache_managed = True
        self._dispatch_generation: int | None = None
        self._compiler_seed_specialization_extractors: tuple[
            _CompilerSeedSpecializationExtractor, ...
        ] = ()
        self._compiler_seed_specialization_results: tuple[Hashable, ...] = ()
        self._artifact = artifact
        self._input_key = _argument_key(args), _input_tensor_aliases(args)
        self._binding_args = tree_map_only(
            torch.Tensor, _CachedTensorArgument.capture, args
        )
        self._materialize_lock = threading.RLock()
        self._delegate: BoundKernel[_R] | None = None
        self._introspection_bound: BoundKernel[_R] | None = None
        self._compiled_run: CompiledConfig | None = None
        self._compiled_path: str | None = None
        self._run: Callable[..., _R] | None = None
        self._config: Config | None = None
        self._backend = get_backend_class(kernel.settings.backend)()
        self._backend.validate_environment()
        self._device = _canonicalize_argument_device(_find_device(args))

    @property
    def settings(self) -> Settings:
        return self.kernel.settings

    @property
    def configs(self) -> list[Config]:
        return self.kernel.configs

    def __getattr__(self, name: str) -> object:
        # Frontend-dependent APIs retain the ordinary BoundKernel behavior.
        # The facade never partially initializes or mutates that object.
        return getattr(self._materialize(), name)

    def _matches_inputs(self, args: Sequence[object]) -> bool:
        return (
            _argument_key(tuple(args)),
            _input_tensor_aliases(args),
        ) == self._input_key

    def _specialize_extra(self) -> list[Callable[[Sequence[object]], Hashable]]:
        return []

    def _record_runtime_input_specialization_results(
        self,
        extractors: Sequence[Callable[[Sequence[object]], Hashable]],
        results: Sequence[Hashable],
    ) -> bool:
        return not extractors and not results

    def _restore_binding_args(self) -> tuple[tuple[object, ...], bool]:
        expired = False
        restored: dict[int, torch.Tensor] = {}

        def restore(value: object) -> object:
            nonlocal expired
            if not isinstance(value, _CachedTensorArgument):
                return value
            tensor = value.reference()
            if tensor is not None:
                return tensor
            expired = True
            if value.identity not in restored:
                # Metadata-only introspection must not allocate a user's tensor
                # again, especially on a GPU. This fake input is never launched.
                with FakeTensorMode():
                    fake = torch.empty_strided(
                        value.shape,
                        value.stride,
                        dtype=value.dtype,
                        device=value.device,
                        requires_grad=value.requires_grad,
                    )
                for dim in value.static_indices:
                    torch._dynamo.mark_static(fake, dim)
                restored[value.identity] = fake
            return restored[value.identity]

        return tree_map(restore, self._binding_args), expired

    def _materialize(self, args: tuple[object, ...] | None = None) -> BoundKernel[_R]:
        # Every path which needs both locks uses this order, including bind().
        with self.kernel._bind_lock, self._materialize_lock:
            if self._delegate is not None:
                return self._delegate
            expired = False
            if args is None:
                args, expired = self._restore_binding_args()
            if expired or self._reset_generation != self.kernel._reset_generation:
                if self._introspection_bound is None:
                    self._introspection_bound = BoundKernel(
                        self.kernel,
                        args,
                        base_spec_key=self._base_spec_key,
                        is_distributed=False,
                        cache_managed=False,
                    )
                return self._introspection_bound

            signature = self.kernel._base_specialization_key(args)
            current_key = self.kernel._get_bound_kernel_cache_key(args, signature)
            current = (
                None
                if current_key is None
                else self.kernel._bound_kernels.get(current_key)
            )
            if current is not None and not isinstance(current, _CachedBoundKernel):
                self._delegate = current
                return current
            normal = BoundKernel(
                self.kernel,
                args,
                base_spec_key=signature,
                is_distributed=False,
                artifact_key=compiled_kernel_cache_key(self.kernel, args, signature),
            )
            has_native_schema = any(
                not isinstance(bound, _CachedBoundKernel)
                and bound._base_spec_key == signature
                for bound in self.kernel._bound_kernels.values()
            )
            if not has_native_schema:
                extra_fns = normal._specialize_extra()
                compiler_seed_fns = normal._compiler_seed_specialization_extractors
                aliases = {
                    alias_signature: alias
                    for alias_signature, alias in self.kernel._specialization_aliases.items()
                    if alias.canonical_signature == signature
                }
                # Validate projections before replacing the provisional schema.
                # A frontend failure leaves every exact disk runner intact.
                extra_results = tuple(extractor(args) for extractor in extra_fns)
                compiler_seed_results = tuple(
                    extractor(args) for extractor in compiler_seed_fns
                )
                hash((signature, extra_results, compiler_seed_results))
                with self.kernel._specialize_extra_lock:
                    self.kernel._specialize_extra[signature] = extra_fns
                    self.kernel._compiler_seed_specialize_extra[signature] = (
                        compiler_seed_fns
                    )
                    for alias_signature, alias in aliases.items():
                        self.kernel._specialize_extra[alias_signature] = [alias]
                        self.kernel._compiler_seed_specialize_extra[alias_signature] = (
                            compiler_seed_fns
                        )
                    if extra_fns:
                        self.kernel._has_specialization_extras = True
                    self.kernel._specialization_generation += 1
                affected = {signature, *aliases}
                for key in list(self.kernel._bound_kernels):
                    if key.specialization_key in affected:
                        self.kernel._bound_kernels.pop(key)
                for key, bound in list(self.kernel._dispatch_cache.items()):
                    if bound._base_spec_key in affected:
                        self.kernel._dispatch_cache.pop(key)
                self.kernel._prepared_call = None
            else:
                # A different native specialization already established this
                # schema. Preserve it and its bindings, as ordinary bind() does.
                for key, bound in list(self.kernel._bound_kernels.items()):
                    if bound is self:
                        self.kernel._bound_kernels.pop(key)
                for key, bound in list(self.kernel._dispatch_cache.items()):
                    if bound is self:
                        self.kernel._dispatch_cache.pop(key)
                if (
                    self.kernel._prepared_call is not None
                    and self.kernel._prepared_call.bound is self
                ):
                    self.kernel._prepared_call = None
            cache_key = self.kernel._create_bound_kernel_cache_key(
                normal, args, signature, snapshot_runtime_results=True
            )
            self.kernel._bound_kernels[cache_key] = normal
            self._delegate = normal
            return normal

    def _requested_config(self, config: ConfigLike | None) -> Config:
        if config is None:
            return self._config or self._artifact[0]
        if isinstance(config, Config):
            return config
        if isinstance(config, CuteStructuralConfig):
            return self._materialize()._normalize_config(config)
        return Config.from_dict(config)

    def compile_config(
        self, config: ConfigLike | None = None, *, allow_print: bool = True
    ) -> CompiledConfig:
        if self._delegate is not None:
            return self._delegate.compile_config(config, allow_print=allow_print)
        requested = self._requested_config(config)
        if requested not in self._artifact[:2]:
            return self._materialize().compile_config(
                requested, allow_print=allow_print
            )
        with self._materialize_lock:
            if self._compiled_run is None:
                module = _load_code(
                    self.kernel,
                    self._backend,
                    self._artifact[2],
                    self._device.index or 0,
                    extra=(
                        repr(self._base_spec_key)
                        if self.settings.static_shapes
                        and self._backend.requires_shape_specialized_module
                        else ""
                    ),
                )
                self._compiled_run = getattr(module, self.kernel.name)
                self._compiled_path = module.__file__
        return self._call_compiled_config

    def _call_compiled_config(self, *args: object) -> _R:
        if len(args) != self.kernel._num_params:
            args = self.kernel.normalize_args(*args)
        if (
            self._delegate is None
            and self._reset_generation == self.kernel._reset_generation
            and not torch.compiler.is_compiling()
            and self._matches_inputs(args)
        ):
            assert self._compiled_run is not None
            return self._compiled_run(*args)
        if (
            self._delegate is None
            and self._reset_generation == self.kernel._reset_generation
            and not torch.compiler.is_compiling()
            and self.kernel._base_specialization_key(args) == self._base_spec_key
        ):
            self._materialize(args)
        # compile_config() returns a callable for this particular config,
        # independently of a later set_config() or normal binding selection.
        bound = self.kernel.bind(args)
        return bound.compile_config(self._artifact[0])(*args)

    def get_cached_path(self, config: ConfigLike | None = None) -> str | None:
        if self._delegate is not None:
            return self._delegate.get_cached_path(config)
        if self._requested_config(config) in self._artifact[:2]:
            return self._compiled_path
        return self._materialize().get_cached_path(config)

    def set_config(self, config: ConfigLike) -> None:
        requested = self._requested_config(config)
        if self._delegate is not None or requested not in self._artifact[:2]:
            self._materialize().set_config(requested)
        else:
            self.compile_config(requested)
        self._config = requested
        self._run = self._call_cached

    def to_triton_code(
        self,
        config: ConfigLike | None = None,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str:
        return self._materialize().to_triton_code(
            config,
            emit_repro_caller=emit_repro_caller,
            output_origin_lines=output_origin_lines,
        )

    def autotune(
        self,
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        normalized_args = self.kernel.normalize_args(*args)
        if (
            self._reset_generation != self.kernel._reset_generation
            or torch.compiler.is_compiling()
            or self.kernel._base_specialization_key(normalized_args)
            != self._base_spec_key
        ):
            return self.kernel.bind(normalized_args).autotune(
                normalized_args, force=force, **kwargs
            )
        if (
            self._delegate is not None
            or not self._matches_inputs(normalized_args)
            or force
            or self.settings.force_autotune
            or self.settings.autotune_handoff
            or kwargs
            or len(self.configs) > 1
        ):
            normal = self._materialize(normalized_args)
            return normal.autotune(normalized_args, force=force, **kwargs)
        if len(self.configs) == 1:
            (config,) = self.configs
        elif self.settings.autotune_effort == "none":
            config = self._artifact[0]
        else:
            # Custom selectors still choose their configuration. Frontend
            # access or a different winner naturally materializes the delegate.
            self.settings.check_autotuning_disabled()
            config = self.settings.autotuner_fn(
                self.kernel.bind(normalized_args), normalized_args
            ).autotune(skip_cache=False)
        self.set_config(config)
        return config

    def _call_cached(self, *args: object) -> _R:
        if len(args) != self.kernel._num_params:
            args = self.kernel.normalize_args(*args)
        if (
            self._delegate is not None
            or self._reset_generation != self.kernel._reset_generation
            or torch.compiler.is_compiling()
        ):
            return self.kernel.bind(args)(*args)
        if not self._matches_inputs(args):
            if self.kernel._base_specialization_key(args) != self._base_spec_key:
                return self.kernel.bind(args)(*args)
            return self._materialize(args)(*args)
        assert self._compiled_run is not None
        return self._compiled_run(*args)

    def __call__(self, *args: object) -> _R:
        if len(args) != self.kernel._num_params:
            args = self.kernel.normalize_args(*args)
        if (
            self._reset_generation != self.kernel._reset_generation
            or torch.compiler.is_compiling()
        ):
            return self.kernel.bind(args)(*args)
        if self._run is None:
            self.autotune(args, force=False)
        return self._call_cached(*args)
