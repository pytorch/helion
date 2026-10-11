"""Disk-backed runners which materialize ordinary bindings on frontend access."""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import TYPE_CHECKING
from typing import Generic
from typing import TypeVar
from typing import cast
import weakref

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_map
from torch.utils._pytree import tree_map_only

from .._argument_device import _canonicalize_argument_device
from .._argument_device import _find_argument_device as _find_device
from .._compiler.backend_registry import get_backend_class
from .._compiler.compile_environment import _concrete_tensor_satisfies_alignment_guard
from .._compiler.compile_environment import (
    tensor_descriptor_layout_signature_from_strides,
)
from .config import Config
from .cute_structural_config import CuteStructuralConfig
from .generated_code_cache import binding_input_key
from .generated_code_cache import binding_variant_key
from .generated_code_cache import compiled_kernel_cache_key
from .generated_code_cache import load_binding_schema
from .generated_code_cache import load_compiled_kernel
from .kernel import BoundKernel
from .kernel import _CompilerSeedSpecializationExtractor
from .kernel import _load_code
from .kernel import _PreparedMetadataSpecializationExtractor
from .kernel import _tensor_descriptor_extent_class

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from typing import Hashable

    from .._compiler.autotuner_heuristics.registry import (
        CompilerHeuristicSpecializationFact,
    )
    from ..autotuner.base_cache import BoundKernelInMemoryCacheKey
    from .kernel import CompiledConfig
    from .kernel import ConfigLike
    from .kernel import Kernel
    from .kernel import _PreparedCall
    from .settings import Settings

_R = TypeVar("_R")

log: logging.Logger = logging.getLogger(__name__)

# Frontend state and APIs which a runner obtains from an ordinary binding.
# Any other missing attribute is an AttributeError, never a frontend compile.
_FORWARDED_ATTRIBUTES = frozenset(
    {
        "_cache_path_map",
        "_compile_cache",
        "_debug_str",
        "_env",
        "_fixed_config_for_td_layout_guards",
        "_generated_code_input_key",
        "_generated_source_cache_keys",
        "_get_host_semantic_fingerprint",
        "_get_host_semantic_input_normalization",
        "_implicit_config",
        "_normalize_config",
        "_normalized_config_copy",
        "_require_implicit_config",
        "_runtime_arg_values_for_codegen",
        "_runtime_tensor_refs_by_name",
        "_semantic_dimension_key",
        "_structural_policy",
        "_user_provided_config",
        "backend_cache_key",
        "bench_compile_config",
        "config_envelope",
        "config_spec",
        "ensure_config_exists",
        "env",
        "extra_cache_key",
        "fake_args",
        "format_kernel_decorator",
        "host_function",
        "is_cacheable",
        "maybe_log_repro",
        "run_ref",
        "supports_subprocess_benchmark",
        "to_code",
    }
)


def _guard_value(guard: list[object]) -> Callable[[Sequence[object]], Hashable]:
    """Rebuild a projection exactly as ``BoundKernel._specialize_extra`` reads it.

    Size, stride and descriptor-layout projections compare tensor metadata that
    prepared calls already guard. Like the frontend's own extractors, they are
    wrapped by ``_guard_extractor``.
    """
    kind, *operands = guard
    if kind == "arg":
        (index,) = cast("list[int]", operands)

        def arg(args: Sequence[object], _index: int = index) -> Hashable:
            return cast("Hashable", args[_index])

        return arg
    base = _guard_value(cast("list[object]", operands[0]))
    if kind == "item":

        def getitem(
            args: Sequence[object],
            _base: Callable[[Sequence[object]], Hashable] = base,
            _item: int | str = cast("int | str", operands[1]),
        ) -> Hashable:
            value = _base(args)
            if isinstance(value, dict):
                return cast("Hashable", value[_item])
            if isinstance(_item, str):
                return cast("Hashable", getattr(value, _item))
            return cast("Sequence[Hashable]", value)[_item]

        return getitem
    if kind == "td_layout":
        ndim, element_size, extent_cap = cast("list[int | None]", operands[1:])

        def td_layout(
            args: Sequence[object],
            _base: Callable[[Sequence[object]], Hashable] = base,
            _ndim: int = cast("int", ndim),
            _element_size: int = cast("int", element_size),
            _extent_cap: int | None = extent_cap,
        ) -> Hashable:
            tensor = cast("torch.Tensor", _base(args))
            if tensor.ndim != _ndim:
                return ("ndim", tensor.ndim)
            return (
                tensor_descriptor_layout_signature_from_strides(
                    tensor.stride(), _element_size
                ),
                tuple(
                    _tensor_descriptor_extent_class(int(size), _extent_cap)
                    for size in tensor.size()
                ),
                all(int(size) < 2**31 for size in tensor.size()),
            )

        return td_layout
    if kind == "td_alignment":

        def td_alignment(
            args: Sequence[object],
            _base: Callable[[Sequence[object]], Hashable] = base,
            _requires_zero_storage_offset: bool = cast("bool", operands[1]),
        ) -> Hashable:
            return _concrete_tensor_satisfies_alignment_guard(
                cast("torch.Tensor", _base(args)), _requires_zero_storage_offset
            )

        return td_alignment

    def tensor_property(
        args: Sequence[object],
        _base: Callable[[Sequence[object]], Hashable] = base,
        _size: bool = kind == "size",
        _dim: int = cast("int", operands[1]),
    ) -> Hashable:
        value = _base(args)
        tensors = value if isinstance(value, (list, tuple)) else (value,)
        result = tuple(
            cast("torch.Tensor", tensor).size(_dim)
            if _size
            else cast("torch.Tensor", tensor).stride(_dim)
            for tensor in tensors
        )
        return result if isinstance(value, (list, tuple)) else result[0]

    return tensor_property


def _guard_extractor(guard: list[object]) -> Callable[[Sequence[object]], Hashable]:
    extractor = _guard_value(guard)
    if guard[0] in ("size", "stride", "td_layout"):
        return _PreparedMetadataSpecializationExtractor(extractor)
    return extractor


def load_cached_kernel(
    kernel: Kernel[_R],
    args: tuple[object, ...],
    signature: tuple[Hashable, ...],
    artifact_key: str,
    cache_key: BoundKernelInMemoryCacheKey | None,
) -> BoundKernel[_R] | None:
    """Return a runner when a saved binding's guards accept these inputs."""
    schema = load_binding_schema(artifact_key)
    if schema is None:
        return None
    extractors = tuple(
        _guard_extractor(cast("list[object]", guard)) for guard in schema["guards"]
    )
    compiler_seed_extractors = tuple(
        _CompilerSeedSpecializationExtractor(
            cast("CompilerHeuristicSpecializationFact", fact), reserved_sms
        )
        for fact, reserved_sms in cast(
            "list[tuple[str, int]]", schema["compiler_seed_facts"]
        )
    )
    extra_results = tuple(extractor(args) for extractor in extractors)
    compiler_seed_results = tuple(
        extractor(args) for extractor in compiler_seed_extractors
    )
    if cache_key is not None and (
        cache_key.extra_results != extra_results
        or cache_key.compiler_seed_results != compiler_seed_results
    ):
        # Another binding already published this signature's schema, and the
        # saved guards disagree with it. The published schema is authoritative.
        return None
    artifact = load_compiled_kernel(
        binding_variant_key(artifact_key, schema, extra_results, compiler_seed_results)
    )
    if artifact is None:
        return None
    runner = _CachedBoundKernel(
        kernel,
        args,
        signature,
        artifact,
        extractors,
        compiler_seed_extractors,
        compiler_seed_results,
    )
    # The runner forwards frontend APIs to an ordinary binding on demand.
    return cast("BoundKernel[_R]", runner)


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

    The runner publishes the saved specialization guards under the same
    in-memory key as an ordinary binding, so dispatch accepts exactly the
    inputs that binding would accept.
    """

    _is_disk_runner = True

    def __init__(
        self,
        kernel: Kernel[_R],
        args: tuple[object, ...],
        signature: tuple[Hashable, ...],
        artifact: tuple[Config, Config, str],
        extractors: tuple[Callable[[Sequence[object]], Hashable], ...],
        compiler_seed_extractors: tuple[_CompilerSeedSpecializationExtractor, ...],
        compiler_seed_results: tuple[Hashable, ...],
    ) -> None:
        self.kernel = kernel
        self._base_spec_key = signature
        self._reset_generation = kernel._reset_generation
        self._cache_managed = True
        self._dispatch_generation: int | None = None
        self._direct_prepared_call: _PreparedCall | None = None
        self._extractors = extractors
        self._compiler_seed_specialization_extractors = compiler_seed_extractors
        self._compiler_seed_specialization_results = compiler_seed_results
        self._artifact = artifact
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
        if name not in _FORWARDED_ATTRIBUTES:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        if self._delegate is None:
            log.debug("%s.%s requires the Helion frontend", self.kernel.name, name)
        return getattr(self._materialize(), name)

    def _is_current(self) -> bool:
        """Whether this runner, not a delegate or a rebinding, owns its launches."""
        return (
            self._delegate is None
            and self._reset_generation == self.kernel._reset_generation
            and not torch.compiler.is_compiling()
        )

    def _specialize_extra(self) -> list[Callable[[Sequence[object]], Hashable]]:
        return list(self._extractors)

    def _record_runtime_input_specialization_results(
        self,
        extractors: Sequence[Callable[[Sequence[object]], Hashable]],
        results: Sequence[Hashable],
    ) -> bool:
        # Saved schemas never contain runtime input specializations.
        return True

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
                    log.debug("Compiling %s for introspection", self.kernel.name)
                    self._introspection_bound = BoundKernel(
                        self.kernel,
                        args,
                        base_spec_key=self._base_spec_key,
                        is_distributed=False,
                        cache_managed=False,
                    )
                return self._introspection_bound

            log.debug(
                "Replacing cached %s binding with a compiled one", self.kernel.name
            )
            signature = self.kernel._base_specialization_key(args)
            current_key = self.kernel._get_bound_kernel_cache_key(args, signature)
            current = (
                None
                if current_key is None
                else self.kernel._bound_kernels.get(current_key)
            )
            if current is not None and not current._is_disk_runner:
                self._evict()
                self._delegate = current
                self._run = None
                return current
            input_key = binding_input_key(self.kernel, args)
            normal = BoundKernel(
                self.kernel,
                args,
                base_spec_key=signature,
                is_distributed=False,
                artifact_key=(
                    None
                    if input_key is None
                    else compiled_kernel_cache_key(
                        self.kernel, args, signature, input_key
                    )
                ),
                input_key=input_key,
            )
            has_native_schema = any(
                not bound._is_disk_runner and bound._base_spec_key == signature
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
                # A frontend failure leaves every disk runner intact.
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
                self._evict()
            cache_key = self.kernel._create_bound_kernel_cache_key(
                normal, args, signature, snapshot_runtime_results=True
            )
            self.kernel._bound_kernels[cache_key] = normal
            self._delegate = normal
            # Launches now go through the delegate. Clear the direct callable so
            # stale dispatch state cannot keep using this facade.
            self._run = None
            return normal

    def _evict(self) -> None:
        """Remove this runner from dispatch before a delegate replaces it."""
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
            # Like an ordinary binding, the compiled callable serves every
            # input which dispatch maps to this binding.
            return self._compiled_run

    def get_cached_path(self, config: ConfigLike | None = None) -> str | None:
        if self._delegate is not None:
            return self._delegate.get_cached_path(config)
        if self._requested_config(config) in self._artifact[:2]:
            return self._compiled_path
        return self._materialize().get_cached_path(config)

    def set_config(self, config: ConfigLike) -> None:
        requested = self._requested_config(config)
        if self._delegate is None and requested in self._artifact[:2]:
            run = self.compile_config(requested)
        else:
            self._materialize().set_config(requested)
            run = None
        self._config = requested
        self._run = run

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
        if not self._is_current() or self.kernel.bind(normalized_args) is not self:
            return self.kernel.bind(normalized_args).autotune(
                normalized_args, force=force, **kwargs
            )
        if (
            force
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

    def _prepare_direct_call(self, args: tuple[object, ...]) -> None:
        """Publish a direct-call guard like ``BoundKernel._prepare_direct_call``.

        Compiler seed facts depend only on the device or exact tensor metadata,
        which the prepared argument guard compares, so they share the guard.
        """
        run = self._run
        if (
            run is None
            or self.kernel._key_fn is not None
            or not (
                self.kernel._has_specialization_extras
                or self._compiler_seed_specialization_extractors
            )
        ):
            return
        try:
            fast_entry = self.kernel._fast_dispatch_key_and_guards(args)
            if fast_entry is None:
                return
            bound = cast("BoundKernel[_R]", self)
            with self.kernel._bind_lock:
                if (
                    not self._is_current()
                    or self.kernel._bind(args) is not bound
                    or self._run is not run
                ):
                    return
                entry = self.kernel._prepare_dispatch_entry(args, bound, fast_entry)
                if entry is not None and entry[0] is not None:
                    self._direct_prepared_call = entry[0]
        except Exception:
            # Preparation runs after the real kernel call. It is optional and
            # must not turn a successful launch into a user-visible failure.
            return

    def __call__(self, *args: object) -> _R:
        if len(args) != self.kernel._num_params:
            args = self.kernel.normalize_args(*args)
        if not self._is_current():
            return self.kernel.bind(args)(*args)
        if (
            (prepared := self._direct_prepared_call) is not None
            and prepared.matches(self.kernel, args)
            and (run := self._run) is not None
        ):
            return run(*args)
        if (
            self.kernel._has_specialization_extras
            or self._compiler_seed_specialization_extractors
        ):
            # As in BoundKernel.__call__, direct calls revalidate value guards.
            rebound = self.kernel.bind(args)
            if rebound is not self:
                return rebound(*args)
        if self._run is None:
            if self._config is None:
                self.autotune(args, force=False)
            else:
                # set_config() chose a config this runner did not save.
                self._materialize(args).set_config(self._config)
        if not self._is_current() or (run := self._run) is None:
            return self.kernel.bind(args)(*args)
        result = run(*args)
        self._prepare_direct_call(args)
        return result
