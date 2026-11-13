"""The Metal backend's autotune search surface: its backend-specific config
keys (``BackendSearchSurface.config_keys``), the fields and seeds that put
them in the search where a kernel's matmuls lower on fragments too, and
their normalization."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

from ...autotuner.config_fragment import EnumFragment
from ...autotuner.search_surface import BackendSearchSurface
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..backend import largest_divisor_at_most
from ._constants import MAX_THREADS_PER_THREADGROUP
from ._constants import SIMD_WIDTH
from .thread_counts import launch_threads

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...autotuner.config_fragment import ConfigSpecFragment
    from ...autotuner.config_spec import ConfigSpec

#: How a K-loop matmul is lowered: on MPP's ``matmul2d`` (the default; a
#: config without the key means it), or on 8x8 simdgroup-matrix fragments
#: built in registers (``fragment_mma``).
METAL_MATMUL = "metal_matmul"
METAL_MATMUL_CHOICES = ("mpp", "fragment")


@dataclasses.dataclass(frozen=True)
class MetalKernelFacts:
    """What ``MetalBackend.register_search_surface`` found in the kernel
    (``ConfigSpec.backend_facts``)."""

    #: Whether the kernel has K-loop matmuls and every one of them lowers on
    #: fragments (``fragment_mma``'s ``check_fragment_group``).
    fragment_matmuls: bool = False
    #: The tunable tiles of each device loop, the grid's first (empty when it
    #: has none), when the kernel's matmuls lower on MPP's cooperative ops or
    #: on fragments, whose launches take whole SIMD groups (:func:`fit_threads`),
    #: and its loops form one nest; else ``()``.
    thread_loops: tuple[tuple[int, ...], ...] = ()
    #: The product of each of those loops' literal-size tiles (a
    #: ``block_size=`` the config can't change), which have no count and run
    #: one thread per element.
    literal_threads: tuple[int, ...] = ()
    #: Those tiles a reduction contracts after their loop, which shrink to
    #: one thread (``blocks_reduced_after_loop``).
    reduced_after_loop: frozenset[int] = frozenset()


def kernel_facts(config_spec: ConfigSpec) -> MetalKernelFacts:
    """The spec's :class:`MetalKernelFacts`; empty ones when
    ``register_search_surface`` recorded none."""
    facts = config_spec.backend_facts
    if facts is None:
        return MetalKernelFacts()
    assert isinstance(facts, MetalKernelFacts)
    return facts


def metal_matmul(config: Mapping[str, object]) -> object:
    """The config's ``metal_matmul``, ``"mpp"`` when it has none."""
    return config.get(METAL_MATMUL, METAL_MATMUL_CHOICES[0])


def lowers_on_fragments(config_spec: ConfigSpec) -> bool:
    """Whether the kernel's K-loop matmuls all lower on fragments."""
    return kernel_facts(config_spec).fragment_matmuls


def fit_threads(config_spec: ConfigSpec, config: dict[str, object]) -> None:
    """Give a matmul config a threadgroup its matmuls run on: exactly one
    SIMD group for a fragment run, whole SIMD groups and at most
    ``MAX_THREADS_PER_THREADGROUP`` for MPP's cooperative ops
    (``mpp_matmul.check_mpp_launch``).

    ``num_warps`` alone doesn't give it: ``num_threads`` sets the
    threadgroup (``thread_counts.launch_threads``), so the search's
    ``[0, 0, 4]`` under one SIMD group launches 128 threads, ``[16, 32, 8]``
    2048 under any ``num_warps``, and ``[2, 2, 1]`` 4.  A config whose counts
    launch a threadgroup its matmuls take keeps them.  Another's are made
    explicit, each at most its block, and fitted to the budget,
    ``min(1024, 32 * num_warps)``: its explicit counts kept (the widest
    halved while they pass the budget), the auto ones filling what is left
    in block order, and the smallest count below its block grown while the
    threadgroup falls short.  All explicit, the counts launch as given.  A
    config with every count on auto, as the default config has, is left as
    it is.  Literal-size tiles (a fixed ``block_size=``) have no count and
    take one thread per element first; the counts fit what they leave.

    Tiles past ``MAX_FRAGMENTS`` or ``MAX_STEPS`` are left as they are:
    codegen refuses them, which costs a codegen rather than a compile, and
    keeps that M4 Pro speed cutoff out of the search.  So is a wide MPP tile
    at too few SIMD groups, which needs ``num_warps`` raised."""
    facts = kernel_facts(config_spec)
    spec = config_spec.num_threads
    blocks = config["block_sizes"]
    # Normalization leaves out counts that are all auto.
    threads = config.get("num_threads", [0] * len(spec))
    num_warps = config["num_warps"]
    assert isinstance(blocks, list) and isinstance(threads, list)
    assert isinstance(num_warps, int)
    if not any(t > 0 for t in threads):
        # Every count on auto, as in the default config: left as it is.
        return
    # A tile over a size-1 dimension runs on one thread.
    sizes = [
        1
        if item.size_hint == 1
        else int(config_spec.block_sizes.config_get(blocks, item.block_ids[0], 1))
        for item in spec
    ]
    budget = min(MAX_THREADS_PER_THREADGROUP, SIMD_WIDTH * num_warps)
    # Literal-size tiles take their threads first, one per element.
    literal = math.prod(facts.literal_threads)
    launch = literal * math.prod(
        launch_threads(
            [[spec.block_id_to_index(b) for b in loop] for loop in facts.thread_loops],
            sizes,
            threads,
            budget,
            fixed=facts.literal_threads,
            to_one={spec.block_id_to_index(b) for b in facts.reduced_after_loop},
        )
    )
    if metal_matmul(config) == "fragment":
        if launch == SIMD_WIDTH:
            return
    elif launch % SIMD_WIDTH == 0 and launch <= MAX_THREADS_PER_THREADGROUP:
        return
    if budget % literal:
        # No counts fit beside the literal tiles; codegen says why.
        return
    budget //= literal
    counts = [
        largest_divisor_at_most(b, t) if t > 0 else 0
        for t, b in zip(threads, sizes, strict=True)
    ]
    while math.prod(c for c in counts if c) > budget:
        widest = max(range(len(counts)), key=counts.__getitem__)
        counts[widest] = largest_divisor_at_most(sizes[widest], counts[widest] - 1)
    left = budget // math.prod(c for c in counts if c)
    for i, size in enumerate(sizes):
        if not counts[i]:
            counts[i] = largest_divisor_at_most(size, left)
            left //= counts[i]
    full: set[int] = set()
    while (total := math.prod(counts)) < budget:
        room = [i for i in range(len(counts)) if counts[i] < sizes[i] and i not in full]
        if not room:
            break
        smallest = min(room, key=counts.__getitem__)
        grown = largest_divisor_at_most(
            sizes[smallest], counts[smallest] * min(2, budget // total)
        )
        if grown == counts[smallest]:
            full.add(smallest)
        counts[smallest] = grown
    config["num_threads"] = counts


class MetalSearchSurface(BackendSearchSurface):
    config_keys = frozenset({METAL_MATMUL})

    def extra_fields(self, config_spec: ConfigSpec) -> dict[str, ConfigSpecFragment]:
        """``metal_matmul`` where both lowerings apply: a kernel whose K-loop
        matmuls all lower on fragments too."""
        if not lowers_on_fragments(config_spec):
            return {}
        return {METAL_MATMUL: EnumFragment(METAL_MATMUL_CHOICES)}

    def trim_reference_config(self, config_spec: ConfigSpec, config: Config) -> None:
        """The default config leaves ``metal_matmul`` out, as before it was
        searched: a config without it means MPP."""
        if metal_matmul(config.config) == METAL_MATMUL_CHOICES[0]:
            config.config.pop(METAL_MATMUL, None)

    def seed_configs(self, config_spec: ConfigSpec) -> list[Config]:
        """Both lowerings in the initial population: the default is MPP's, so
        seed a fragment config (one SIMD group per threadgroup, which the
        fragment run takes)."""
        if not lowers_on_fragments(config_spec):
            return []
        return [Config(metal_matmul="fragment", num_warps=1)]

    def normalize_knobs(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        if metal_matmul(config) not in METAL_MATMUL_CHOICES:
            if not fix_invalid:
                raise InvalidConfig(
                    f"{METAL_MATMUL} must be one of {METAL_MATMUL_CHOICES}, got "
                    f"{config[METAL_MATMUL]!r}"
                )
            del config[METAL_MATMUL]
        if metal_matmul(config) == "fragment" and (
            fix_invalid or "num_warps" not in config
        ):
            # A limit of the fragment lowering, not a choice: a run's
            # accumulator fragments live in one SIMD group's lanes, so it
            # takes one SIMD group per threadgroup, and the search's fragment
            # configs at other num_warps couldn't compile.  A config that
            # leaves num_warps out gets 1 rather than the default; one that
            # sets another value still fails, saying why.  Splitting K over
            # the SIMD groups of a threadgroup would lift the limit.
            config["num_warps"] = 1

    def normalize_finalize(
        self,
        config_spec: ConfigSpec,
        config: dict[str, object],
        *,
        fix_invalid: bool,
        provided_keys: set[str],
    ) -> None:
        if fix_invalid and kernel_facts(config_spec).thread_loops:
            fit_threads(config_spec, config)


METAL_SEARCH_SURFACE = MetalSearchSurface()
