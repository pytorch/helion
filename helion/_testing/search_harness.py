"""A stand-in kernel on which the autotuner tests build and prepare searches
without a real kernel or a compile."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from ..autotuner.base_search import BaseSearch
from ..autotuner.benchmark_provider import LocalBenchmarkProvider
from ..autotuner.search_surface import BackendSearchSurface
from ..runtime.config import Config
from . import DEVICE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..autotuner.base_search import _AutotunableKernel
    from ..runtime.settings import Settings


def mock_kernel(settings: Settings, **attrs: object) -> SimpleNamespace:
    """A stand-in for a bound kernel with what building a search and its
    benchmark provider reads: a config spec whose default config is
    ``block_sizes=[1]``, on a backend that precompiles and finds every config
    viable, and an environment on ``DEVICE`` outside any process group.
    ``attrs`` adds or replaces kernel attributes, such as the
    ``compile_config`` of a search that compiles configs."""
    config_spec = SimpleNamespace(
        default_config=lambda: Config(block_sizes=[1]),
        search_surface=BackendSearchSurface(),
        cute_flash_search_enabled=False,
        compiler_seed_timeout_retry_repetitions=None,
        create_config_generation=lambda **kwargs: SimpleNamespace(
            invalid_config_count=0
        ),
        backend=SimpleNamespace(
            supports_precompile=lambda: True,
            autotune_config_is_viable=lambda config_spec, config: True,
            should_deduplicate_generated_sources=lambda config_spec: False,
            get_do_bench=lambda: None,
            classify_autotune_exception=lambda error: None,
            probe_long_autotune_kernels=lambda config_spec: False,
        ),
    )
    kernel = SimpleNamespace(
        settings=settings,
        config_spec=config_spec,
        env=SimpleNamespace(process_group_name=None, device=DEVICE),
        format_kernel_decorator=lambda config, settings: "decorator",
        to_triton_code=lambda config: "code",
        maybe_log_repro=lambda log_func, args, config=None: None,
        supports_subprocess_benchmark=lambda: False,
    )
    vars(kernel).update(attrs)
    return kernel


def prepare_search(search: BaseSearch) -> None:
    """``search._prepare()`` without the accuracy baseline, which would
    compile and run the kernel's default config."""
    with patch.object(
        LocalBenchmarkProvider, "_compute_baseline", return_value=(None, [], None)
    ):
        search._prepare()


def mock_search(
    settings: Settings, args: Sequence[object] = (), **kernel_attrs: object
) -> BaseSearch:
    """A prepared ``BaseSearch`` on ``args`` of ``mock_kernel(settings,
    **kernel_attrs)``. It is built through ``__init__``, rather than by
    setting a search's attributes by hand, so attributes a later ``__init__``
    adds need no change here."""
    kernel = mock_kernel(settings, **kernel_attrs)
    search = BaseSearch(cast("_AutotunableKernel", kernel), args)
    prepare_search(search)
    return search
