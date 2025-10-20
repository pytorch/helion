from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import itertools
import logging
import operator
import re
import sys
import types
from typing import TYPE_CHECKING
from typing import Callable
from typing import Generic
from typing import Hashable
from typing import TypeVar
from typing import cast
from typing import overload
from typing_extensions import Protocol

import torch
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch._inductor.codecache import PyCodeCache
from torch._inductor.codecache import compiled_fx_graph_hash
from torch._subclasses import FakeTensor
from torch.utils.weak import WeakIdKeyDictionary

from .. import exc
from .._compiler.ast_extension import unparse
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.generate_ast import generate_ast
from .._compiler.host_function import HostFunction
from .._compiler.inductor_lowering_extra import patch_inductor_lowerings
from .._compiler.output_header import assert_no_conflicts
from .._compiler.output_header import get_needed_imports
from .._compiler.variable_origin import ArgumentOrigin
from .._logging import LazyString
from .._utils import counters
from ..language.constexpr import ConstExpr
from .config import Config
from .ref_mode import RefModeContext
from .ref_mode import is_ref_mode_enabled
from .settings import Settings

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Sequence

    from torch._guards import Source

    from ..autotuner import ConfigSpec
    from ..autotuner.base_cache import BoundKernelInMemoryCacheKey

    ConfigLike = Config | dict[str, object]

log: logging.Logger = logging.getLogger(__name__)
_R = TypeVar("_R")
CompiledConfig = Callable[..., _R]

# Cache for GraphModule hashes
_graph_module_hash_cache: WeakIdKeyDictionary = WeakIdKeyDictionary()


class Kernel(Generic[_R]):
    def __init__(
        self,
        fn: Callable[..., _R],
        *,
        configs: list[ConfigLike] | None = None,
        settings: Settings | None,
        key: Callable[..., Hashable] | None = None,
    ) -> None:
        """
        Initialize the Kernel object.  This is typically called from the `@helion.kernel` decorator.

        Args:
            fn: The function to be compiled as a Helion kernel.
            configs: A list of configurations to use for the kernel.
            settings: The settings to be used by the Kernel. If None, a new `Settings()` instance is created.
            key: Optional callable that returns an extra hashable component for specialization.
        """
        super().__init__()
        assert isinstance(fn, types.FunctionType)
        assert_no_conflicts(fn)
        self.name: str = fn.__name__
        self.fn: types.FunctionType = fn
        self.signature: inspect.Signature = inspect.signature(fn)
        self.settings: Settings = settings or Settings()
        self._key_fn: Callable[..., Hashable] | None = key
        self.configs: list[Config] = [
            Config(**c) if isinstance(c, dict) else c  # pyright: ignore[reportArgumentType]
            for c in configs or []
        ]
        self._bound_kernels: dict[BoundKernelInMemoryCacheKey, BoundKernel] = {}
        self._specialize_extra: dict[
            Hashable, list[Callable[[Sequence[object]], Hashable]]
        ] = {}
        if any(
            param.kind
            in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for param in self.signature.parameters.values()
        ):
            raise TypeError(
                f"Kernel({self.name}) cannot have *args, **kwargs, or keyword-only arguments"
            )

        self._annotations: list[object] = []
        for param in self.signature.parameters.values():
            ann = param.annotation
            if isinstance(ann, str) and re.search(r"constexpr", ann, re.IGNORECASE):
                self._annotations.append(ConstExpr)
            else:
                self._annotations.append(ann)

    def _get_bound_kernel_cache_key(
        self, args: tuple[object, ...], signature: tuple[Hashable, ...]
    ) -> BoundKernelInMemoryCacheKey | None:
        from ..autotuner.base_cache import BoundKernelInMemoryCacheKey

        extra_fns = self._specialize_extra.get(signature)
        if extra_fns is not None:
            extra_results: tuple[Hashable, ...] = tuple([s(args) for s in extra_fns])
            return BoundKernelInMemoryCacheKey(signature, extra_results)
        return None

    def _create_bound_kernel_cache_key(
        self,
        bound_kernel: BoundKernel,
        args: tuple[object, ...],
        signature: tuple[Hashable, ...],
    ) -> BoundKernelInMemoryCacheKey:
        from ..autotuner.base_cache import BoundKernelInMemoryCacheKey

        self._specialize_extra[signature] = extra_fns = bound_kernel._specialize_extra()
        extra_results: tuple[Hashable, ...] = tuple([s(args) for s in extra_fns])
        return BoundKernelInMemoryCacheKey(signature, extra_results)

    def bind(self, args: tuple[object, ...]) -> BoundKernel[_R]:
        """
        Bind the given arguments to the Kernel and return a BoundKernel object.

        Args:
            args: The arguments to bind to the Kernel.

        Returns:
            BoundKernel: A BoundKernel object with the given arguments bound.
        """
        if not isinstance(args, tuple):
            assert isinstance(args, list), "args must be a tuple or list"
            args = tuple(args)
        if len(args) > len(self.signature.parameters):
            raise TypeError(
                f"Too many arguments passed to the kernel, expected: {len(self.signature.parameters)} got: {len(args)}."
            )
        signature = self.specialization_key(args)
        cache_key = self._get_bound_kernel_cache_key(args, signature)
        bound_kernel = (
            None if cache_key is None else self._bound_kernels.get(cache_key, None)
        )
        if bound_kernel is None:
            normalized_args: tuple[object, ...] = self.normalize_args(*args)
            if len(normalized_args) != len(args):
                # we had default args that needed to be applied
                bound_kernel = self.bind(normalized_args)
            else:
                bound_kernel = BoundKernel(self, args)
            if cache_key is None:
                cache_key = self._create_bound_kernel_cache_key(
                    bound_kernel, args, signature
                )
            self._bound_kernels[cache_key] = bound_kernel
        return bound_kernel

    def specialization_key(self, args: Sequence[object]) -> tuple[Hashable, ...]:
        """
        Generate a specialization key for the given arguments.

        This method generates a unique key for the arguments based on their types
        and the corresponding extractor functions defined in `_specialization_extractors`.

        Args:
            args: The arguments to generate a specialization key for.

        Returns:
            Hashable: A hashable key representing the specialization of the arguments.
        """
        result = []
        assert len(args) <= len(self._annotations)
        for value, annotation in zip(args, self._annotations, strict=False):
            if isinstance(value, ConstExpr):
                result.append(value.value)
            elif annotation is ConstExpr:
                result.append(value)
            else:
                result.append(self._specialization_key(value))
        if self._key_fn is not None:
            return (*result, self._key_fn(*args))
        return (*result,)

    def _specialization_key(self, obj: object) -> Hashable:
        """
        Helper used to generate a specialization key for the given object.

        This method determines a unique key for the object based on its type
        and the corresponding extractor function defined in `_specialization_extractors`.

        Args:
            obj: The argument to generate a specialization key for.

        Returns:
            Hashable: A hashable key representing the specialization of the object.
        """
        try:
            extractor = _specialization_extractors[type(obj)]
        except KeyError:
            if isinstance(obj, torch.fx.GraphModule):
                # GraphModule subclasses need special handling
                extractor = _specialization_extractors[torch.fx.GraphModule]
            elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
                # this is a namedtuple
                extractor = _specialization_extractors["namedtuple"]
            elif dataclasses.is_dataclass(obj):
                extractor = _specialization_extractors["dataclass"]
            else:
                raise TypeError(
                    f"unsupported argument type: {type(obj).__name__}"
                ) from None
        return extractor(self, obj)

    def normalize_args(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        """
        Normalize the given arguments and keyword arguments according to the function signature.

        Args:
            args: The positional arguments to normalize.
            kwargs: The keyword arguments to normalize.

        Returns:
            tuple[object, ...]: A tuple of normalized positional arguments.
        """
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return tuple(bound_args.args)

    def autotune(
        self,
        args: Sequence[object],
        *,
        force: bool = True,
        **options: object,
    ) -> Config:
        """
        Perform autotuning to find the optimal configuration for the kernel.  This uses the
        default setting, you can call helion.autotune.* directly for more customization.

        If config= or configs= is provided to helion.kernel(), the search will be restricted to
        the provided configs.  Use force=True to ignore the provided configs.

        Mutates (the bound version of) self so that `__call__` will run the best config found.

        Args:
            args: Example arguments used for benchmarking during autotuning.
            force: If True, force full autotuning even if a config is provided.
            options: Additional keyword options forwarded to the autotuner.

        Returns:
            Config: The best configuration found during autotuning.
        """
        args = self.normalize_args(*args)
        return self.bind(args).autotune(args, force=force, **options)

    def __call__(self, *args: object, **kwargs: object) -> _R:
        """
        Call the Kernel with the given arguments and keyword arguments.

        Args:
            args: The positional arguments to pass to the Kernel.
            kwargs: The keyword arguments to pass to the Kernel.

        Returns:
            _R: The result of the Kernel function call.
        """
        if kwargs:
            args = self.normalize_args(*args, **kwargs)
        return self.bind(args)(*args)

    def reset(self) -> None:
        """
        Clears the cache of bound kernels, meaning subsequent calls will
        recompile and re-autotune.
        """
        self._bound_kernels.clear()


class BoundKernel(Generic[_R]):
    def __init__(
        self,
        kernel: Kernel[_R],
        args: tuple[object, ...],
    ) -> None:
        """
        Initialize a BoundKernel object.

        This constructor sets up the environment, compiles the kernel function, and prepares
        the arguments for execution.

        Args:
            kernel: The Kernel object to bind.
            args: A tuple of arguments to bind to the kernel.
        """
        super().__init__()
        self.kernel = kernel
        self._run: Callable[..., _R] | None = None
        self._config: Config | None = None
        self._compile_cache: dict[Config, CompiledConfig] = {}
        self.env = CompileEnvironment(_find_device(args), self.kernel.settings)

        if is_ref_mode_enabled(self.kernel.settings):
            self.fake_args = []  # type: ignore[assignment]
            self.host_function = None  # type: ignore[assignment]
            return

        with self.env:
            assert len(args) == len(self.kernel.signature.parameters)
            self.fake_args: list[object] = []
            constexpr_args = {}
            for name, arg, annotation in zip(
                self.kernel.signature.parameters,
                args,
                self.kernel._annotations,
                strict=False,
            ):
                if isinstance(arg, ConstExpr):
                    assert not isinstance(arg.value, torch.Tensor), (
                        "ConstExpr cannot be a tensor"
                    )
                    self.fake_args.append(arg.value)
                    constexpr_args[name] = arg.value
                elif annotation is ConstExpr:
                    assert not isinstance(arg, torch.Tensor), (
                        "ConstExpr cannot be a tensor"
                    )
                    self.fake_args.append(arg)
                    constexpr_args[name] = arg
                else:
                    self.fake_args.append(self.env.to_fake(arg, ArgumentOrigin(name)))
            with (
                _maybe_skip_dtype_check_in_meta_registrations(),
                patch_inductor_lowerings(),
            ):
                self.host_function: HostFunction = HostFunction(
                    self.kernel.fn, self.fake_args, constexpr_args
                )

    @property
    def settings(self) -> Settings:
        """
        Retrieve the settings associated with the kernel.

        Returns:
            Settings: The settings of the kernel.
        """
        return self.kernel.settings

    @property
    def config_spec(self) -> ConfigSpec:
        """
        Retrieve the configuration specification for the kernel.

        Returns:
            ConfigSpec: The configuration specification.
        """
        return self.env.config_spec

    @property
    def configs(self) -> list[Config]:
        """
        Alias for `self.kernel.configs`.

        Returns:
            list[Config]: The list of configurations.
        """
        return self.kernel.configs

    def format_kernel_decorator(self, config: Config, settings: Settings) -> str:
        """Return the @helion.kernel decorator snippet capturing configs and settings that influence Triton code generation."""
        return f"@helion.kernel(config={config.__repr__()}, static_shapes={settings.static_shapes})"

    def to_triton_code(
        self,
        config: ConfigLike | None = None,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str:
        """
        Generate Triton code for the kernel based on the given configuration.

        Args:
            config: The configuration to use for code generation.
            emit_repro_caller: Emits a main function to call the triton kernel with example inputs.

        Returns:
            str: The generated Triton code as a string.
        """
        if config is None:
            config = self._require_implicit_config()
        with self.env:
            if not isinstance(config, Config):
                config = Config(**config)  # pyright: ignore[reportArgumentType]
            self.env.config_spec.normalize(config)
            root = generate_ast(self.host_function, config, emit_repro_caller)
            if output_origin_lines is None:
                output_origin_lines = self.settings.output_origin_lines
            return get_needed_imports(root) + unparse(
                root, output_origin_lines=output_origin_lines
            )

    def compile_config(
        self, config: ConfigLike | None = None, *, allow_print: bool = True
    ) -> CompiledConfig:
        """
        Compile the kernel for a specific configuration.

        Args:
            config: The configuration to compile the kernel with.
            allow_print: Set to suppress printing the output code when autotuning.

        Returns:
            CompiledConfig: A callable object representing the compiled kernel.
        """
        if config is None:
            config = self._require_implicit_config()
        if not isinstance(config, Config):
            config = Config(
                **config  # pyright: ignore[reportArgumentType]
            )
        if (rv := self._compile_cache.get(config)) is not None:
            return rv
        try:
            triton_code = self.to_triton_code(
                config, emit_repro_caller=self.settings.print_output_code
            )
            module = PyCodeCache.load(triton_code)
        except Exception:
            log.warning(
                "Helion compiler triton codegen error for %s",
                self.format_kernel_decorator(config, self.settings),
                exc_info=True,
            )
            raise
        if allow_print:
            log.info("Output code: \n%s", triton_code)
            log.info("Output code written to: %s", module.__file__)
            log.debug("Debug string: \n%s", LazyString(lambda: self._debug_str()))
            if self.settings.print_output_code:
                print(triton_code, file=sys.stderr)
        rv = getattr(module, self.kernel.name)
        self._compile_cache[config] = rv
        return rv

    def _debug_str(self) -> str:
        """
        Generate a debug string for the kernel.

        Returns:
            str: A string containing debug information about the kernel.
        """
        if self.host_function is None:
            # In ref mode, host_function is not created
            return f"<BoundKernel {self.kernel.fn.__name__} in ref mode>"
        with self.env:
            return self.host_function.debug_str()

    def autotune(
        self,
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        """
        Perform autotuning to find the optimal configuration for the kernel.  This uses the
        default setting, you can call helion.autotune.* directly for more customization.

        If config= or configs= is provided to helion.kernel(), the search will be restricted to
        the provided configs.  Use force=True to ignore the provided configs.

        Mutates self so that `__call__` will run the best config found.

        Args:
            args: Example arguments used for benchmarking during autotuning.
            force: If True, force full autotuning even if a config is provided.
            kwargs: Additional keyword options forwarded to the autotuner.

        Returns:
            Config: The best configuration found during autotuning.
        """
        force = force or self.settings.force_autotune
        if not force and self.kernel.configs:
            if len(self.kernel.configs) == 1:
                (config,) = self.kernel.configs
            else:
                # We have finite predetermined configs, no need to precompile
                self.settings.autotune_precompile = None

                from ..autotuner import FiniteSearch

                config = FiniteSearch(self, args, self.configs).autotune()
        else:
            self.settings.check_autotuning_disabled()
            config = self.settings.autotuner_fn(self, args, **kwargs).autotune(
                skip_cache=force
            )

        self.set_config(config)
        return config

    def set_config(self, config: ConfigLike) -> None:
        """
        Set the configuration for the kernel and compile it.

        Mutates self so that `__call__` will run the provided config.

        Args:
            config: The configuration to set.
        """
        if not isinstance(config, Config):
            config = Config(
                **config  # pyright: ignore[reportArgumentType]
            )
        self._run = self.compile_config(config)
        self._config = config

    def _specialize_extra(self) -> list[Callable[[Sequence[object]], Hashable]]:
        """
        Returns a list of functions that will be called to generate extra specialization keys.
        This is used to specialize on the values hl.specialize()'ed arguments.

        Returns:
            list[Callable[[Sequence[object]], Hashable]]: A list of functions that generate extra specialization keys.
        """
        if not self.env.specialized_vars:
            return []

        def make_extractor(v: Source) -> Callable[[Sequence[object]], Hashable]:
            if isinstance(v, TensorPropertySource):
                assert v.prop == TensorProperty.SIZE
                index = v.idx
                assert index is not None
                inner = make_extractor(v.base)

                return lambda args: cast("torch.Tensor", inner(args)).size(index)
            if isinstance(v, LocalSource):
                index = arg_name_to_index[v.local_name]
                return operator.itemgetter(index)
            raise exc.SpecializeArgType(v)

        arg_name_to_index: dict[str, int] = {
            n: i for i, n in enumerate(self.kernel.signature.parameters.keys())
        }
        extractors = []
        for v in sorted(self.env.specialized_vars, key=lambda v: v.name):
            source = self.env.shape_env.var_to_sources[v][0]
            extractors.append(make_extractor(source))
        return extractors

    def _implicit_config(self) -> Config | None:
        """
        Returns a single config that is implicitly used by this kernel, if any.
        """
        configs = self.kernel.configs
        if self._config is not None:
            return self._config
        if len(configs) == 1:
            return configs[0]
        if len(configs) == 0 and self.kernel.settings.autotune_effort == "none":
            config = self.config_spec.default_config()
            if not is_ref_mode_enabled(self.kernel.settings):
                kernel_decorator = self.format_kernel_decorator(config, self.settings)
                print(
                    f"Using default config: {kernel_decorator}",
                    file=sys.stderr,
                )
            return config
        return None

    def _require_implicit_config(self) -> Config:
        """
        Returns the implicit config for this kernel, or raises an error if no implicit config is available.
        """
        if (config := self._implicit_config()) is None:
            raise RuntimeError("no config provided and no implicit config available")
        return config

    def run_ref(self, *args: object) -> _R:  # pyright: ignore[reportReturnType]
        # Unwrap ConstExpr arguments
        clean_args = []
        for arg in args:
            if isinstance(arg, ConstExpr):
                clean_args.append(arg.value)
            else:
                clean_args.append(arg)

        # Pass the config to RefModeContext
        with RefModeContext(self.env, self._config):
            result = self.kernel.fn(*clean_args)
            return cast("_R", result)

    def __call__(self, *args: object) -> _R:
        """
        Execute the kernel with the given arguments.

        Args:
            args: The arguments to pass to the kernel.

        Returns:
            _R: The result of the kernel execution.
        """
        if is_ref_mode_enabled(self.kernel.settings):
            if (config := self._implicit_config()) is not None:
                self._config = config
            return self.run_ref(*args)

        if self._run is None:
            if (config := self._implicit_config()) is not None:
                self.set_config(config)
            else:
                self.autotune(args, force=False)
            assert self._run is not None

        assert self._config is not None
        counters["best_config_decorator"][
            self.format_kernel_decorator(self._config, self.settings)
        ] = 1

        return self._run(*args)


class _KernelDecorator(Protocol):
    def __call__(
        self,
        fn: Callable[..., _R],
    ) -> Kernel[_R]: ...


@overload
def kernel(
    fn: Callable[..., _R],
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    key: Callable[..., Hashable] | None = None,
    **settings: object,
) -> Kernel[_R]: ...


@overload
def kernel(
    fn: None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    key: Callable[..., Hashable] | None = None,
    **settings: object,
) -> _KernelDecorator: ...


def kernel(
    fn: Callable[..., _R] | None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    key: Callable[..., Hashable] | None = None,
    **settings: object,
) -> Kernel[_R] | _KernelDecorator:
    """
    Decorator to create a Kernel object from a Python function.

    Args:
        fn: The function to be wrapped by the Kernel. If None, a decorator is returned.
        config: A single configuration to use for the kernel. Refer to the
            ``helion.Config`` class for details.
        configs: A list of configurations to use for the kernel. Can only specify
            one of config or configs. Refer to the ``helion.Config`` class for
            details.
        key: Optional callable returning a hashable that augments the specialization key.
        settings: Keyword arguments representing settings for the Kernel.
            Can also use settings=Settings(...) to pass a Settings object
            directly. Refer to the ``helion.Settings`` class for available
            options.

    Returns:
        object: A Kernel object or a decorator that returns a Kernel object.

    See Also:
        - :class:`~helion.Settings`: Controls compilation behavior and debugging options
        - :class:`~helion.Config`: Controls GPU execution parameters and optimization strategies
    """
    if config is not None:
        assert not configs, "Cannot specify both config and configs"
        configs = [config]
    elif configs is None:
        configs = []

    if settings_obj := settings.get("settings"):
        assert len(settings) == 1, "settings must be the only keyword argument"
        assert isinstance(settings_obj, Settings), "settings must be a Settings object"
    else:
        settings_obj = Settings(**settings)

    if fn is None:
        return functools.partial(
            kernel, configs=configs, settings=settings_obj, key=key
        )
    return Kernel(fn, configs=configs, settings=settings_obj, key=key)


def _tensor_key(fn: Kernel, obj: torch.Tensor) -> Hashable:
    # NOTE: If a machine has two different gpu types on the same machine,
    # obj.device.type will incorrectly hit
    if fn.settings.static_shapes:
        return (
            obj.dtype,
            obj.device.type,
            (*obj.size(),),
            (*obj.stride(),),
        )
    return (
        obj.dtype,
        obj.device.type,
        # 0, 1, or >=2 specialization
        tuple([min(s, 2) for s in obj.size()]),
    )


def _sequence_key(fn: Kernel, obj: Sequence) -> Hashable:
    return type(obj), tuple([fn._specialization_key(item) for item in obj])


def _mapping_key(
    fn: Kernel, obj: dict[str | int, object], real_type: type[object]
) -> Hashable:
    return real_type, tuple(
        sorted((k, fn._specialization_key(v)) for k, v in obj.items())
    )


def _number_key(fn: Kernel, n: float | bool) -> object:
    return type(n)


def _function_key(fn: Kernel, obj: types.FunctionType) -> object:
    if obj.__closure__:
        closures = [
            fn._specialization_key(cell.cell_contents) for cell in obj.__closure__
        ]
        return (obj.__code__, *closures)
    return obj.__code__


def _graph_module_key(fn: Kernel, obj: torch.fx.GraphModule) -> Hashable:
    """Generate a specialization key for GraphModule arguments."""
    # Check if already cached
    if obj in _graph_module_hash_cache:
        return _graph_module_hash_cache[obj]

    # Check for unsupported operations
    unsupported_ops = {
        node.op
        for node in itertools.chain(
            obj.graph.find_nodes(op="call_module"),
            obj.graph.find_nodes(op="get_attr"),
        )
    }
    if unsupported_ops:
        raise exc.GraphModuleUnsupportedOps(", ".join(sorted(unsupported_ops)))

    _graph_module_hash_cache[obj] = rv = str(compiled_fx_graph_hash(obj, [], {}, []))
    return rv


_specialization_extractors: dict[
    type[object] | str, Callable[[Kernel, object], Hashable]
] = {  # pyright: ignore[reportAssignmentType]
    torch.Tensor: _tensor_key,
    torch.nn.Parameter: _tensor_key,
    FakeTensor: _tensor_key,
    torch.dtype: lambda fn, x: x,
    torch.device: lambda fn, x: x,
    int: _number_key,
    float: _number_key,
    bool: _number_key,
    str: lambda fn, x: x,
    list: _sequence_key,
    tuple: _sequence_key,
    dict: lambda fn, x: _mapping_key(fn, x, type(x)),  # pyright: ignore[reportArgumentType]
    "namedtuple": lambda fn, x: _mapping_key(fn, x._asdict(), type(x)),  # pyright: ignore[reportAttributeAccessIssue]
    "dataclass": lambda fn, x: _mapping_key(fn, dataclasses.asdict(x), type(x)),  # pyright: ignore[reportArgumentType]
    types.FunctionType: _function_key,
    types.BuiltinFunctionType: lambda fn, x: x,
    torch.fx.GraphModule: _graph_module_key,
    ConstExpr: lambda fn, x: x.value,  # pyright: ignore[reportAttributeAccessIssue]
    type(None): lambda fn, x: None,
}


def _find_device(args: tuple[object, ...]) -> torch.device:
    """
    Extract the device from the arguments.

    Args:
        args: The arguments to extract the device from.

    Returns:
        torch.device: The extracted device
    """
    for arg in args:
        if isinstance(arg, torch.device):
            return arg
        if isinstance(arg, torch.Tensor):
            return arg.device
        if isinstance(arg, (tuple, list)):
            for item in arg:
                try:
                    return _find_device(item)
                except exc.NoTensorArgs:
                    pass
        elif isinstance(arg, dict):
            for item in arg.values():
                try:
                    return _find_device(item)
                except exc.NoTensorArgs:
                    pass
    raise exc.NoTensorArgs


def _maybe_skip_dtype_check_in_meta_registrations() -> (
    contextlib.AbstractContextManager[None, None]
):
    if hasattr(torch.fx.experimental._config, "skip_dtype_check_in_meta_registrations"):  # pyright: ignore[reportAttributeAccessIssue]
        return torch.fx.experimental._config.patch(  # pyright: ignore[reportAttributeAccessIssue]
            skip_dtype_check_in_meta_registrations=True
        )
    return contextlib.nullcontext()
