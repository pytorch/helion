"""External autotuner API for non-Helion kernels."""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING
from typing import Any

import torch

from helion.autotuner.base_search import _AutotunableKernel
from helion.autotuner.config_spec import ConfigSpec
from helion.runtime.config import Config
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import helion
    from helion.autotuner.config_fragment import ConfigSpecFragment


class UserConfigSpec(ConfigSpec):
    """A ConfigSpec that only includes user-defined tunables.

    The base ConfigSpec.flat_config() always injects Triton/Helion-specific
    tunables (num_warps, num_stages, pid_type, etc.) into the search space.
    This subclass overrides flat_config to only emit the tunables the caller
    defined, keeping the search space minimal and DSL-agnostic.
    """

    def flat_key_layout(self) -> list[tuple[str, int]]:
        return [(key, 1) for key in self.user_defined_tunables]

    def flat_config(self, fn: Callable[[ConfigSpecFragment], object]) -> helion.Config:
        return Config.from_dict(
            {key: fn(fragment) for key, fragment in self.user_defined_tunables.items()}
        )

    def normalize(
        self,
        config: helion.Config | dict[str, object],
        *,
        _fix_invalid: bool = False,
    ) -> None:
        pass


def create_user_config_spec(
    tunables: dict[str, ConfigSpecFragment],
) -> UserConfigSpec:
    return UserConfigSpec(user_defined_tunables=dict(tunables))


@dataclasses.dataclass
class _FakeEnv:
    device: torch.device


class _ExternalKernelAdapter(_AutotunableKernel):
    """Adapter making user-provided callables satisfy the ``_AutotunableKernel`` protocol.

    Private for now; could be made public/subclassable once the interface
    stabilises.  Subclassing would let DSL authors override the diagnostic
    hooks: ``get_cached_path``, ``to_triton_code``, ``format_kernel_decorator``,
    and ``maybe_log_repro``.
    """

    def __init__(
        self,
        config_spec: UserConfigSpec,
        compile_fn: Callable[[Config], Callable[..., Any]],
        args: Sequence[Any],
        baseline_fn: Callable[..., Any] | None = None,
        device: torch.device | None = None,
        **settings_kwargs: object,
    ) -> None:
        self._config_spec = config_spec
        self._configs: list[Config] = []
        self._compile_fn = compile_fn
        self._compile_cache: dict[Config, Callable[..., Any]] = {}

        self._env = _FakeEnv(
            device
            or next(
                (
                    arg.device
                    for arg in args
                    if isinstance(arg, torch.Tensor) and arg.is_cuda
                ),
                torch.device("cuda"),
            )
        )
        self._settings = Settings(**settings_kwargs)
        if baseline_fn is not None:
            self._settings.autotune_baseline_fn = baseline_fn
        self._settings.autotune_precompile = None

    @property
    def config_spec(self) -> UserConfigSpec:
        return self._config_spec

    @property
    def env(self) -> _FakeEnv:
        return self._env

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def configs(self) -> list[Config]:
        return self._configs

    def compile_config(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        allow_print: bool = True,
    ) -> Callable[..., Any]:
        if config is None:
            config = self.config_spec.default_config()
        if not isinstance(config, Config):
            config = Config.from_dict(config)

        if config not in self._compile_cache:
            self._compile_cache[config] = self._compile_fn(config)
        return self._compile_cache[config]

    def format_kernel_decorator(self, config: Config, settings: Settings) -> str:
        return f"config={config!r}"

    def get_cached_path(self, config: Config | None = None) -> str | None:
        return None

    def to_triton_code(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str | None:
        return None

    def maybe_log_repro(
        self,
        log_func: Callable[[str], None],
        args: Sequence[object],
        config: Config | None = None,
    ) -> None:
        pass


SETTINGS_KWARGS = {
    f.name for f in dataclasses.fields(Settings) if f.name.startswith("autotune_")
}


def _split_and_validate_kwargs(
    search_cls: type[object],
    kwargs: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split kwargs into Settings and search kwargs, then validate search keys.

    Keys listed in ``SETTINGS_KWARGS`` are routed to ``Settings``. Remaining
    keys are treated as algorithm kwargs and are validated against the selected
    search class constructor.
    """
    settings_kw = {k: v for k, v in kwargs.items() if k in SETTINGS_KWARGS}
    search_kw = {k: v for k, v in kwargs.items() if k not in SETTINGS_KWARGS}

    signature = inspect.signature(search_cls.__init__)
    search_params = {
        name
        for name, param in signature.parameters.items()
        if name not in {"self", "kernel", "args"}
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if not any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        if unknown_search_keys := sorted(set(search_kw) - search_params):
            raise ValueError(
                f"Unknown search kwargs for algorithm {search_cls.__name__}: "
                f"{unknown_search_keys}. Allowed search kwargs: "
                f"{sorted(search_params)}. Allowed settings kwargs: "
                f"{sorted(SETTINGS_KWARGS)}."
            )

    return settings_kw, search_kw


def autotune(
    tunables: dict[str, ConfigSpecFragment],
    compile_fn: Callable[[Config], Callable[..., Any]],
    args: Sequence[Any],
    *,
    baseline_fn: Callable[..., Any] | None = None,
    algorithm: str = "PatternSearch",
    device: torch.device | None = None,
    **kwargs: object,
) -> Config:
    """Autotune a kernel using Helion's search algorithms.

    Args:
        tunables: Dict mapping parameter names to ConfigSpecFragment instances
            that define the search space.
        compile_fn: ``Config -> Callable``.  Given a config, return a callable
            that runs the kernel.  The callable must accept ``*args``.
        args: The arguments that will be passed to ``compile_fn(config)()``.
        baseline_fn: Optional ``*args -> output``.  Reference implementation
            for accuracy validation.  If not provided, the default config is
            compiled and used as the baseline (same behavior as
            ``@helion.kernel``).
        algorithm: Name of the search algorithm.  One of ``"PatternSearch"``,
            ``"LFBOPatternSearch"``, ``"DifferentialEvolutionSearch"``,
            ``"DESurrogateHybrid"``, ``"RandomSearch"``, ``"FiniteSearch"``.
        device: CUDA device.  Auto-detected from *args* if not given.
        **kwargs: Split automatically -- keys in ``SETTINGS_KWARGS`` go to
            ``Settings`` (e.g. ``autotune_accuracy_check``), everything else
            goes to the search algorithm constructor (e.g.
            ``max_generations``, ``initial_population``). Unknown search kwargs
            raise ``ValueError`` with the allowed keys for the selected
            algorithm.

    Note:
        External kernels are not compatible with autotune cache wrappers.

    Returns:
        The best ``Config`` found.

    Example::

        from helion.autotuner import PowerOfTwoFragment
        from helion.autotuner import autotune

        best = autotune(
            tunables={"block": PowerOfTwoFragment(32, 512, 128)},
            compile_fn=my_compile,
            args=(input_tensor,),
            algorithm="PatternSearch",
            max_generations=5,
        )
    """
    from helion.autotuner import search_algorithms

    search_cls = search_algorithms.get(algorithm)
    if search_cls is None:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}."
            f" Available: {list(search_algorithms.keys())}"
        )

    settings_kw, search_kw = _split_and_validate_kwargs(search_cls, kwargs)

    return search_cls(
        _ExternalKernelAdapter(
            create_user_config_spec(tunables),
            compile_fn,
            args,
            baseline_fn=baseline_fn,
            device=device,
            **settings_kw,
        ),
        args,
        **search_kw,
    ).autotune()
