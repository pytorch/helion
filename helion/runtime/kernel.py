from __future__ import annotations

import ast
import functools
import inspect
import types
from typing import TYPE_CHECKING
from typing import Callable
from typing import overload

import torch
from torch._inductor.codecache import PyCodeCache

from .. import exc
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.generate_ast import generate_ast
from .._compiler.host_function import HostFunction
from .._compiler.output_header import assert_no_conflicts
from .._compiler.output_header import get_needed_imports
from .._compiler.variable_origin import ArgumentOrigin
from .config import Config
from .settings import Settings

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Sequence

    from ..autotuner import ConfigSpec

    ConfigLike = Config | dict[str, object]


class Kernel:
    def __init__(
        self,
        fn: types.FunctionType,
        *,
        configs: list[ConfigLike] | None = None,
        settings: Settings | None,
    ) -> None:
        """
        Initialize the Kernel object.  This is typically called from the `@helion.kernel` decorator.

        :param fn: The function to be compiled as a Helion kernel.
        :param settings: The settings to be used by the Kernel. If None, default settings are used.
        """
        super().__init__()
        assert_no_conflicts(fn)
        self.name: str = fn.__name__
        self.fn = fn
        self.signature: inspect.Signature = inspect.signature(fn)
        self.settings: Settings = settings or Settings.default()
        self.configs: list[Config] = [*map(Config, configs or ())]
        # pyre-fixme[11]: BoundKernel undefined?
        self._bound_kernels: dict[Hashable, BoundKernel] = {}
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

    def bind(self, args: tuple[object, ...]) -> BoundKernel:
        """
        Bind the given arguments to the Kernel and return a BoundKernel object.

        :param args: The arguments to bind to the Kernel.
        :return: A BoundKernel object with the given arguments bound.
        """
        if not isinstance(args, tuple):
            assert isinstance(args, list), "args must be a tuple or list"
            args = tuple(args)
        signature = self._specialization_key(args)
        bound_kernel = self._bound_kernels.get(signature)
        if bound_kernel is None:
            normalized_args: tuple[object, ...] = self.normalize_args(*args)
            if len(normalized_args) != len(args):
                # we had default args that needed to be applied
                bound_kernel = self.bind(normalized_args)
            else:
                bound_kernel = BoundKernel(self, args)
            self._bound_kernels[signature] = bound_kernel
        return bound_kernel

    def _specialization_key(self, obj: object) -> Hashable:
        """
        Generate a specialization key for the given arg.

        This method determines a unique key for the object based on its type
        and the corresponding extractor function defined in `_specialization_extractors`.

        :param obj: The argument to generate a specialization key for.
        :return: A hashable key representing the specialization of the object.
        """
        try:
            extractor = _specialization_extractors[type(obj)]
        except KeyError:
            raise TypeError(
                f"unsupported argument type: {type(obj).__name__}"
            ) from None
        return extractor(self, obj)

    def normalize_args(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        """
        Normalize the given arguments and keyword arguments according to the function signature.

        :param args: The positional arguments to normalize.
        :param kwargs: The keyword arguments to normalize.
        :return: A tuple of normalized positional arguments.
        """
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return tuple(bound_args.args)

    def autotune(
        self,
        args: Sequence[object],
        **options: object,
    ) -> Config:
        """
        Perform autotuning to find the optimal configuration for
        the kernel.  This uses the default setting, you can call
        helion.autotune.* directly for more customization.

        Mutates (the bound version of) self so that `__call__` will run the best config found.

        :param args: Example arguments used for benchmarking during autotuning.
        :type args: list[object]
        :return: The best configuration found during autotuning.
        :rtype: Config
        """
        args = self.normalize_args(*args)
        return self.bind(args).autotune(args, **options)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """
        Call the Kernel with the given arguments and keyword arguments.

        :param args: The positional arguments to pass to the Kernel.
        :param kwargs: The keyword arguments to pass to the Kernel.
        :return: The result of the Kernel function call.
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


class BoundKernel:
    # pyre-fixme[11]: Kernel undefined?
    def __init__(self, kernel: Kernel, args: tuple[object, ...]) -> None:
        """
        Initialize a BoundKernel object.

        This constructor sets up the environment, compiles the kernel function, and prepares
        the arguments for execution.

        :param kernel: The Kernel object to bind.
        :type kernel: Kernel
        :param args: A tuple of arguments to bind to the kernel.
        :type args: tuple[object, ...]
        """
        super().__init__()
        self.kernel = kernel
        self._run: Callable[..., object] | None = None
        self._compile_cache: dict[Config, Callable[..., object]] = {}
        self.env = CompileEnvironment(_find_device(args), self.kernel.settings)
        with self.env:
            assert len(args) == len(self.kernel.signature.parameters)
            self.fake_args: list[object] = [
                # TODO(jansel): Support hl.constexpr
                self.env.to_fake(arg, ArgumentOrigin(name))
                for name, arg in zip(self.kernel.signature.parameters, args)
            ]
            self.host_fn: HostFunction = HostFunction(self.kernel.fn, self.fake_args)
        if len(kernel.configs) == 1:
            self.set_config(kernel.configs[0])

    @property
    def settings(self) -> Settings:
        """
        Retrieve the settings associated with the kernel.

        :return: The settings of the kernel.
        :rtype: Settings
        """
        return self.kernel.settings

    @property
    def config_spec(self) -> ConfigSpec:
        """
        Retrieve the configuration specification for the kernel.

        :return: The configuration specification.
        :rtype: ConfigSpec
        """
        return self.env.config_spec

    @property
    def configs(self) -> list[Config]:
        """
        Alias for `self.kernel.configs`.

        :return: The list of configurations.
        :rtype: list[Config]
        """
        return self.kernel.configs

    def to_triton_code(self, config: ConfigLike) -> str:
        """
        Generate Triton code for the kernel based on the given configuration.

        :param config: The configuration to use for code generation.
        :type config: Config or dict[str, object]
        :return: The generated Triton code as a string.
        :rtype: str
        """
        with self.env:
            config = Config(config)
            self.env.config_spec.normalize(config)
            root = generate_ast(self.host_fn, config)
            return get_needed_imports(root) + ast.unparse(root)

    def compile_config(self, config: ConfigLike) -> Callable[..., object]:
        """
        Compile the kernel for a specific configuration.

        :param config: The configuration to compile the kernel with.
        :type config: Config or dict[str, object]
        :return: A callable object representing the compiled kernel.
        :rtype: Callable[..., object]
        """
        if not isinstance(config, Config):
            config = Config(config)
        if (rv := self._compile_cache.get(config)) is not None:
            return rv
        module = PyCodeCache.load(self.to_triton_code(config))
        rv = getattr(module, self.kernel.name)
        self._compile_cache[config] = rv
        return rv

    def _debug_str(self) -> str:
        """
        Generate a debug string for the kernel.

        :return: A string containing debug information about the kernel.
        :rtype: str
        """
        with self.env:
            return self.host_fn.debug_str()

    def autotune(
        self,
        args: Sequence[object],
        **kwargs: object,
    ) -> Config:
        """
        Perform autotuning to find the optimal configuration for
        the kernel.  This uses the default setting, you can call
        helion.autotune.* directly for more customization.

        Mutates self so that `__call__` will run the best config found.

        :param args: Example arguments used for benchmarking during autotuning.
        :type args: list[object]
        :return: The best configuration found during autotuning.
        :rtype: Config
        """
        if self.kernel.configs:
            from ..autotuner import FiniteSearch

            config = FiniteSearch(self, args, self.configs).autotune()
        else:
            from ..autotuner import DifferentialEvolutionSearch

            config = DifferentialEvolutionSearch(
                self,
                args,
                # pyre-ignore[6]
                **kwargs,
            ).autotune()
        self.set_config(config)
        return config

    def set_config(self, config: ConfigLike) -> None:
        """
        Set the configuration for the kernel and compile it.

        Mutates self so that `__call__` will run the provided config.

        :param config: The configuration to set.
        :type config: ConfigLike
        """
        if not isinstance(config, Config):
            config = Config(config)
        self._run = self.compile_config(config)

    def __call__(self, *args: object) -> object:
        """
        Execute the kernel with the given arguments.

        :param args: The arguments to pass to the kernel.
        :type args: object
        :return: The result of the kernel execution.
        :rtype: object
        """
        if self._run is None:
            if not self.configs and self.settings.use_default_config:
                self.set_config(self.config_spec.default_config())
            else:
                self.autotune(args)
            assert self._run is not None
        return self._run(*args)


@overload
def kernel(
    fn: Callable[..., object],
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    **settings: object,
) -> Kernel: ...


@overload
def kernel(
    fn: None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    **settings: object,
) -> Callable[[Callable[..., object]], Kernel]: ...


def kernel(
    fn: Callable[..., object] | None = None,
    *,
    config: ConfigLike | None = None,
    configs: list[ConfigLike] | None = None,
    **settings: object,
) -> object:
    """
    Decorator to create a Kernel object from a Python function.

    :param fn: The function to be wrapped by the Kernel. If None, a decorator is returned.
    :type fn: Callable[..., object] | None
    :param config: A single configuration to use for the kernel.
    :param configs: A list of configurations to use for the kernel.  Can only specify one of config or configs.
    :param settings: Keyword arguments representing settings for the Kernel.
                    Can also use settings=Settings(...) to pass a Settings object directly.
    :return: A Kernel object or a decorator that returns a Kernel object.
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
        return functools.partial(kernel, configs=configs, settings=settings_obj)
    return Kernel(fn, configs=configs, settings=settings_obj)


def _tensor_key(fn: Kernel, obj: torch.Tensor) -> Hashable:
    if fn.settings.static_shapes:
        return (
            obj.dtype,
            obj.device,
            (*obj.size(),),
            (*obj.stride(),),
        )
    return (
        obj.dtype,
        obj.device,
        # 0, 1, or >=2 specialization
        tuple([min(s, 2) for s in obj.size()]),
    )


def _sequence_key(fn: Kernel, obj: Sequence) -> Hashable:
    return type(obj), tuple([fn._specialization_key(item) for item in obj])


def _number_key(fn: Kernel, n: float | bool) -> object:
    if fn.settings.static_shapes:
        return n
    return type(n)


def _function_key(fn: Kernel, obj: types.FunctionType) -> object:
    if obj.__closure__:
        closures = [
            fn._specialization_key(cell.cell_contents) for cell in obj.__closure__
        ]
        return (obj.__code__, *closures)
    return obj.__code__


_specialization_extractors: dict[type[object], Callable[[Kernel, object], Hashable]] = {
    torch.Tensor: _tensor_key,
    torch.nn.Parameter: _tensor_key,
    torch.dtype: lambda fn, x: x,
    torch.device: lambda fn, x: x,
    int: _number_key,
    float: _number_key,
    bool: _number_key,
    str: lambda fn, x: x,
    list: _sequence_key,
    tuple: _sequence_key,
    dict: lambda fn, x: tuple(
        sorted((k, fn._specialization_key(v)) for k, v in x.items())
    ),
    types.FunctionType: _function_key,
}


def _find_device(args: tuple[object, ...]) -> torch.device:
    """
    Extract the device from the arguments.

    :param args: The arguments to extract the device from.
    :return: The extracted device
    """
    for arg in args:
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
