"""Shared helpers and caches for the CuTe flash, flex and collective tests.

The flash structural-coverage design costs seconds to a minute per
``ConfigGeneration`` and the CPU render of a flash kernel a fraction of a
second per config, so read-only tests across ``test_cute_flash_*`` modules
share warmed generations and rendered sources through the process-wide
caches here instead of rebuilding them per case. Tests that mutate or patch a
spec, a generation, or a kernel must build their own.

The GPU modules compile each kernel configuration once and read the generated
source back from the compiled module (``compiled_source`` and
``compile_and_run``) instead of rendering it a second time, and the seed
modules render one representative seed per codegen branch
(``seeds_per_codegen_branch``) instead of every seed in a search space.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import torch

from .._compiler.cute import cute_flash
from .._compiler.cute import cute_flash_gated
from .._compiler.cute.autotune.config_generation_ext import cute_generation_extension
from .._compiler.cute.autotune.flash_surface import enable_cute_flash_search
from .._compiler.cute.backend import CuteBackend
from ..autotuner.config_fragment import EnumFragment
from ..autotuner.config_generation import ConfigGeneration
from ..autotuner.config_spec import BlockSizeSpec
from ..autotuner.config_spec import ConfigSpec
from ..runtime.config import Config
from .cute_aux import cpu_codegen

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Hashable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .._compiler.cute.cute_flash_gated import GatedAttentionMatch
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import ConfigLike
    from ..runtime.kernel import Kernel

__all__ = [
    "bind_flash_kernel",
    "compile_and_run",
    "compile_and_source",
    "compiled_source",
    "flash_config_spec",
    "frozen_overrides",
    "gated_attention_match",
    "memoized_flash_fragments",
    "render_flash_config",
    "seeds_per_codegen_branch",
    "shared_flash_generation",
]

_FRAGMENT_CACHE: dict[object, dict[str, Any]] = {}
_FRAGMENT_VERIFIED: set[object] = set()


def _flash_environment() -> tuple[tuple[str, str], ...]:
    """The ``HELION_`` environment the flash fragments may read."""
    return tuple(
        sorted(
            (key, value)
            for key, value in os.environ.items()
            if key.startswith("HELION_")
        )
    )


@contextlib.contextmanager
def memoized_flash_fragments() -> Iterator[None]:
    """Memoize ``flash_fragments_from_facts`` while a coverage design is built.

    The structural-coverage design calls it tens of thousands of times with
    only a handful of distinct inputs (the facts and one override pair per
    pipeline family), dominating its cost. The cache lives for the process
    (keyed by the facts, the keyword overrides and the ``HELION_``
    environment, so a monkeypatched environment never sees another test's
    fragments; the facts carry the knob channel as well); the first repeat of
    each key is re-verified against the real function so nondeterminism would
    still fail, and every hit returns a fresh dict. The key does not see
    patched module state: do not use it under a patched flash target policy
    or a patched module constant the fragments read (build such a generation
    unmemoized instead).
    """
    real = cast("Callable[..., dict[str, Any]]", cute_flash.flash_fragments_from_facts)
    environment = _flash_environment()

    def wrapper(facts: object, **kwargs: object) -> dict[str, Any]:
        key = (environment, facts, tuple(sorted(kwargs.items())))
        hit = _FRAGMENT_CACHE.get(key)
        if hit is None:
            hit = _FRAGMENT_CACHE[key] = real(facts, **kwargs)
        elif key not in _FRAGMENT_VERIFIED:
            _FRAGMENT_VERIFIED.add(key)
            assert real(facts, **kwargs) == hit
        return dict(hit)

    # Nested blocks stay sound: an inner block sees the outer wrapper as
    # ``real`` and its hits short-circuit before reaching it.
    with patch.object(cute_flash, "flash_fragments_from_facts", wrapper):
        yield


def flash_config_spec(
    *,
    head_dim: int,
    num_kv: int,
    dtype: torch.dtype,
    is_causal: bool,
    num_bh: int = 64,
    tensor_4d_heads: int | None = None,
    standard_dense_output: bool | None = None,
    standard_causal_output: bool | None = None,
    requires_ws_overlap: bool = False,
    supports_tensor_4d_tma: bool = True,
    output_requires_tma: bool = False,
    fast_math: bool = False,
) -> ConfigSpec:
    """A CuTe flash ``ConfigSpec`` for the generic structural search.

    The target capability is left unset so the surface is independent of the
    local GPU; hardware-specific measured seeds are covered by
    ``test_cute_flash_arch.py``.
    """
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=None)
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    enable_cute_flash_search(
        spec,
        head_dim=head_dim,
        num_kv=num_kv,
        num_bh=num_bh,
        tensor_4d_heads=tensor_4d_heads,
        dtype=dtype,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=is_causal,
        requires_ws_overlap=requires_ws_overlap,
        standard_dense_output=(
            not is_causal if standard_dense_output is None else standard_dense_output
        ),
        standard_causal_output=(
            is_causal if standard_causal_output is None else standard_causal_output
        ),
        supports_tensor_4d_tma=supports_tensor_4d_tma,
        output_requires_tma=output_requires_tma,
        fast_math=fast_math,
    )
    return spec


def _freeze(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def frozen_overrides(
    overrides: Mapping[str, object] | None,
) -> tuple[tuple[str, object], ...]:
    """A hashable, order-independent form of a config (override) mapping."""
    return tuple(
        sorted((key, _freeze(value)) for key, value in (overrides or {}).items())
    )


@functools.cache
def _shared_flash_generation(
    build_spec: Callable[..., ConfigSpec],
    args: tuple[object, ...],
    kwargs: tuple[tuple[str, object], ...],
    overrides: tuple[tuple[str, object], ...],
    excluded_search_choices: tuple[tuple[str, object], ...],
    environment: tuple[tuple[str, str], ...],
) -> ConfigGeneration:
    spec = build_spec(*args, **dict(kwargs))
    generation = spec.create_config_generation(overrides=dict(overrides))
    if excluded_search_choices:
        # The explicit resident-free reference: the pinned generation's
        # surface with the excluded choices removed from the search, under
        # the same overrides and pipeline-family projection.
        family = generation._flash_pipeline_family_override
        fields = dict(
            spec._flat_fields()
            if family is None
            else spec._flat_fields_with_flash_family(family)
        )
        for key, value in excluded_search_choices:
            fragment = fields[key]
            assert isinstance(fragment, EnumFragment)
            fields[key] = dataclasses.replace(
                fragment,
                search_choices=tuple(
                    choice for choice in fragment._active_choices() if choice != value
                ),
            )
        generation = ConfigGeneration(
            spec,
            overrides=generation._override_values,
            _flash_pipeline_family_override=family,
            _field_view=fields,
        )
    with memoized_flash_fragments():
        # Build the design without validating it: a deliberately incomplete
        # domain (an override that pins a child's parent away) is still a
        # valid instance to query for its uncovered telemetry.
        cute_generation_extension(
            generation
        ).flash_structural_coverage_uncovered_values()
    return generation


def shared_flash_generation(
    build_spec: Callable[..., ConfigSpec] = flash_config_spec,
    /,
    *args: object,
    overrides: Mapping[str, object] | None = None,
    excluded_search_choices: Mapping[str, object] | None = None,
    **kwargs: object,
) -> ConfigGeneration:
    """A warmed ``ConfigGeneration`` shared across read-only tests.

    ``build_spec(*args, **kwargs)`` builds the spec (``flash_config_spec`` by
    default; the arguments must be hashable), ``overrides`` pin config values,
    and ``excluded_search_choices`` maps an enum key to the one search choice
    an explicit reference domain leaves out. The structural-coverage design is
    built once per distinct key and reused: it depends only on these
    parameters and the ``HELION_`` environment, so tests that merely query the
    surface share one instance. Tests that mutate or patch the spec or the
    generation must build their own.
    """
    return _shared_flash_generation(
        build_spec,
        args,
        frozen_overrides(kwargs),
        frozen_overrides(overrides),
        frozen_overrides(excluded_search_choices),
        _flash_environment(),
    )


_BOUND_KERNELS: dict[tuple[object, ...], BoundKernel[Any]] = {}
_RENDERED_SOURCES: dict[tuple[object, ...], str] = {}


def bind_flash_kernel(
    kernel: Kernel[Any],
    arg_key: tuple[object, ...],
    make_args: Callable[[], tuple[object, ...]],
) -> BoundKernel[Any]:
    """Bind ``kernel`` on CPU once per ``arg_key`` under ``cpu_codegen``.

    ``arg_key`` must determine ``make_args()`` (shapes, dtypes, strides and
    any non-tensor arguments); the bound kernel is reused by every later
    render with the same key and ``HELION_`` environment, including after
    ``kernel.reset()``.
    """
    key = (kernel, *arg_key, _flash_environment())
    bound = _BOUND_KERNELS.get(key)
    if bound is None:
        with cpu_codegen():
            bound = _BOUND_KERNELS[key] = kernel.bind(make_args())
    return bound


def render_flash_config(
    kernel: Kernel[Any],
    arg_key: tuple[object, ...],
    make_args: Callable[[], tuple[object, ...]],
    config: Mapping[str, object] | Config,
) -> str:
    """The CPU render of ``kernel`` at ``config``, one per normalized config.

    ``config`` is the full config (the spec's default merged with a test's
    pins); it is normalized through the bound kernel and the source is cached
    by the normalized identity, so pins that normalize to the same config
    share one render.
    """
    bound = bind_flash_kernel(kernel, arg_key, make_args)
    if not isinstance(config, Config):
        config = Config(**cast("dict[str, Any]", dict(config)))
    with cpu_codegen():
        normalized = bound._normalized_config_copy(config)
        key = (kernel, *arg_key, normalized, _flash_environment())
        source = _RENDERED_SOURCES.get(key)
        if source is None:
            source = _RENDERED_SOURCES[key] = bound.to_code(normalized)
    return source


def compile_and_source(
    bound: BoundKernel[Any], config: ConfigLike
) -> tuple[Callable[..., Any], str]:
    """Compile ``config`` on ``bound`` and return the compiled kernel with the
    exact source the compile wrote (``BoundKernel.compile_config`` memoizes per
    normalized config, so repeated calls compile once per process)."""
    compiled = bound.compile_config(config)
    path = bound.get_cached_path(config)
    assert path is not None
    return compiled, Path(path).read_text()


def compiled_source(bound: BoundKernel[Any], config: ConfigLike) -> str:
    """Compile ``config`` on ``bound``, make it the bound config, and return its source.

    ``bound.to_code(config)`` followed by ``set_config`` renders the kernel
    twice, and a CuTe flash or collective render costs about as much as the
    compile itself once the DSL cache is warm. The compiled module's file
    holds the exact text code generation produced, so read that back instead
    (under ``HELION_PRINT_OUTPUT_CODE=1`` it also carries the appended repro
    caller). ``BoundKernel.compile_config`` memoizes per normalized config,
    so repeated calls for one config compile once per process.
    """
    bound.set_config(config)
    return compile_and_source(bound, config)[1]


def gated_attention_match(bound: BoundKernel[Any]) -> GatedAttentionMatch | None:
    """The config-independent gated-attention matcher
    (``cute_flash_gated.match_gated_attention``) on ``bound``'s device IR."""
    env, host_function = bound.env, bound.host_function
    assert env is not None and host_function is not None
    with env, host_function:
        return cute_flash_gated.match_gated_attention(host_function.device_ir)


def compile_and_run(
    kernel: Kernel[Any], args: tuple[object, ...], config: ConfigLike
) -> tuple[str, Any]:
    """Compile ``kernel`` at ``config`` for ``args``, run it once, return ``(code, result)``.

    A one-render replacement for ``code_and_output`` (see ``compiled_source``)
    that leaves the bound kernel's own config untouched. ``Kernel.bind``
    memoizes per argument specialization, so a reference kernel several tests
    compare against is bound and compiled once per process. The generated
    source goes to stderr when the launch fails.
    """
    bound = kernel.bind(args)
    compiled, code = compile_and_source(bound, config)
    try:
        result = compiled(*args)
        torch.cuda.synchronize()
    except Exception:
        sys.stderr.write(f"Failed to run kernel:\n{code}\n")
        raise
    return code, result


def seeds_per_codegen_branch(
    seeds: Sequence[Config],
    branch_key: Callable[[Config], Hashable],
    *,
    last_per: Callable[[Config], Hashable] | None = None,
) -> list[Config]:
    """The seeds worth rendering out of a seed list: one per codegen branch.

    Returns the first seed of every distinct ``branch_key`` plus the last seed
    (of each ``last_per`` group when given, otherwise of the whole list),
    without repeats. The seed axes outside the key (tile shapes, thread
    layouts, stage depths, swizzles) re-parameterize the same generated text,
    so rendering every seed only repeats these renders; assert the per-seed
    config properties on ``seeds`` directly instead.
    """
    branches: dict[Hashable, Config] = {}
    last: dict[Hashable, Config] = {}
    for seed in seeds:
        branches.setdefault(branch_key(seed), seed)
        last[None if last_per is None else last_per(seed)] = seed
    chosen = list(branches.values())
    chosen.extend(
        seed for seed in last.values() if not any(seed is kept for kept in chosen)
    )
    return chosen
