"""The Metal backend's autotune search surface: its backend-specific config
keys (``BackendSearchSurface.config_keys``), the fields and seeds that put
them in the search where a kernel's matmuls have more than one lowering,
their normalization, and the fallback of a config that names no lowering
(:class:`MatmulFallback`)."""

from __future__ import annotations

import contextlib
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_spec import DEFAULT_NUM_WARPS
from ...autotuner.config_spec import RangeNumStagesSpec
from ...autotuner.search_surface import BackendSearchSurface
from ...exc import BackendUnsupported
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..backend import _largest_divisor_at_most
from ..emitter_protocols import RetryPolicy
from ._constants import MAX_THREADS_PER_THREADGROUP
from ._constants import SIMD_WIDTH
from .thread_counts import launch_threads

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

    from ...autotuner.config_fragment import ConfigSpecFragment
    from ...autotuner.config_spec import ConfigSpec
    from .backend import MetalBackend

#: How a kernel's matmuls are lowered: on MPP's ``matmul2d`` (the default; a
#: config without the key means it), on 8x8 simdgroup-matrix fragments built
#: in registers (``fragment_mma``), or on the scalar contraction, each
#: thread forming the products at its own M, K and N and K's reduction
#: summing them (``matmul_ops``).
METAL_MATMUL = "metal_matmul"
METAL_MATMUL_CHOICES = ("mpp", "fragment", "scalar")


#: The most entries of a ``constant`` table a fragment element reads a value
#: computed from a small bit field from (``tabulate``), 0 for none: the
#: default.  A table replaces the field's decode with a load, which pays for
#: a 4-bit field's decode but not always for a byte's.
METAL_TABULATE = "metal_tabulate_max_entries"
METAL_TABULATE_CHOICES = (0, 16, 256)


@dataclasses.dataclass(frozen=True)
class MetalKernelFacts:
    """What ``MetalBackend.register_search_surface`` found in the kernel
    (``ConfigSpec.backend_facts``)."""

    #: Whether the kernel's matmuls lower on fragments (``fragment_mma``'s
    #: ``check_fragment_group``: K-loop accumulators, whole-K matmuls and
    #: per-K-tile products): all of its aten matmuls, and its ``hl.dot``
    #: calls when the fragment rules take every one of them
    #: (``mpp_matmul._takes_dots``); else the dots keep the scalar
    #: contraction.
    fragment_matmuls: bool = False
    #: Whether the kernel has matmuls, which the scalar contraction lowers
    #: under ``metal_matmul="scalar"``.
    scalar_matmuls: bool = False
    #: The sizes of the tables those matmuls' fragment elements would read
    #: (``tabulate.table_sizes``).
    table_sizes: frozenset[int] = frozenset()
    #: Whether SIMD groups can split those matmuls' K loops
    #: (``operand_elements.split_refusal``): a fragment config's ``num_warps``.
    fragment_splits: bool = False
    #: Whether, besides, no free ``hl.arange`` axis takes threads from the
    #: ``num_warps`` budget and ``thread_loops`` gives the launch, so that
    #: normalization sees an MPP config's launch as codegen does and, where
    #: its counts launch one SIMD group, ``num_warps`` changes nothing but
    #: the SIMD groups splitting its register runs' K loops
    #: (``mpp_registers.register_split``, which codegen runs only here); a
    #: reduction, which also would draw on it, refuses the split.
    register_splits: bool = False
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
    #: The K-loop MPP matmuls that stage one operand per K tile (a value the
    #: Scalar layout computes, a dequantized weight) beside a device view:
    #: ``(tall, narrow, k)`` blocks, the view's side (whose threads share
    #: the staged tile), the staged operand's other side, and K
    #: (``staged_matmul_heuristic``).
    staged_matmuls: tuple[tuple[int, int, int], ...] = ()
    #: The K-loop matmuls the fragment lowering takes: ``(m, n, k)`` blocks
    #: of each (``fragment_matmul_heuristic``).
    fragment_blocks: tuple[tuple[int, int, int], ...] = ()


def kernel_facts(config_spec: ConfigSpec) -> MetalKernelFacts:
    """The spec's :class:`MetalKernelFacts`; empty ones when
    ``register_search_surface`` recorded none."""
    facts = config_spec.backend_facts
    if facts is None:
        return MetalKernelFacts()
    assert isinstance(facts, MetalKernelFacts)
    return facts


def metal_tabulate_max_entries(config: Mapping[str, object]) -> int:
    """The config's ``metal_tabulate_max_entries``, 0 when it has none."""
    value = config.get(METAL_TABULATE, METAL_TABULATE_CHOICES[0])
    assert isinstance(value, int)
    return value


def metal_matmul(config: Mapping[str, object]) -> object:
    """The config's ``metal_matmul``, ``"mpp"`` when it has none."""
    return config.get(METAL_MATMUL, METAL_MATMUL_CHOICES[0])


def lowers_on_fragments(config_spec: ConfigSpec) -> bool:
    """Whether the kernel's matmuls lower on fragments
    (``MetalKernelFacts.fragment_matmuls``)."""
    return kernel_facts(config_spec).fragment_matmuls


def matmul_choices(config_spec: ConfigSpec) -> tuple[str, ...]:
    """The ``metal_matmul`` values that lower the kernel's matmuls: MPP's,
    fragments' where they apply (:func:`lowers_on_fragments`) and the scalar
    contraction's where the kernel has matmuls."""
    facts = kernel_facts(config_spec)
    return tuple(
        choice
        for choice, applies in zip(
            METAL_MATMUL_CHOICES,
            (True, facts.fragment_matmuls, facts.scalar_matmuls),
            strict=True,
        )
        if applies
    )


class MatmulFallback(RetryPolicy):
    """A config that names no ``metal_matmul`` (the default config, or one
    written before the key was searched) takes the first lowering that
    compiles it, in :func:`matmul_choices`' order: MPP's, fragments' on one
    SIMD group (a fragment config's ``num_warps`` splits its K loops), then
    the scalar contraction's.  ``generate_ast`` generates the kernel again
    under the next one, from the config as given, when a pass whose
    matmuls a cooperative lowering planned raises ``BackendUnsupported``;
    the last one's error propagates, caused by the first's.  A config that
    names a lowering keeps it and fails with its error, as the search's
    drawn configs do: the search draws every lowering itself.  Its default
    config, which leaves the key out, falls back like any other."""

    signals = (BackendUnsupported,)

    def __init__(self) -> None:
        self._first: BaseException | None = None
        self._config: Config | None = None

    @contextlib.contextmanager
    def scope(self) -> Iterator[None]:
        # Each pass plans afresh; a pass that fails before planning leaves
        # no plan and no refusal.
        backend = _metal_backend()
        backend.mpp_plan, backend.mpp_plan_refused = None, False
        yield

    def repair(
        self, signal: BaseException, config: Config, *, caller_graphs: bool
    ) -> Config | None:
        from ..compile_environment import CompileEnvironment

        backend = _metal_backend()
        if caller_graphs or (self._first is None and METAL_MATMUL in config.config):
            return None
        plan, refused = backend.mpp_plan, backend.mpp_plan_refused
        backend.mpp_plan, backend.mpp_plan_refused = None, False
        choices = matmul_choices(CompileEnvironment.current().config_spec)
        current = metal_matmul(config)
        # A pass that failed before planning, or whose matmuls all took the
        # scalar contraction, failed for a reason no other lowering changes.
        planned = refused or (plan is not None and bool(plan.by_matmul))
        if self._first is None:
            if not planned:
                return None
            self._first, self._config = signal, config
        index = choices.index(current) if current in choices else len(choices)
        rest = choices[index + 1 :] if planned else ()
        if not rest:
            if signal is not self._first:
                signal.__cause__ = self._first
            return None
        # From the config as given: a fragment pass's num_warps=1 is its own.
        assert self._config is not None
        fallback = {**self._config.config, METAL_MATMUL: rest[0]}
        if rest[0] == "fragment":
            fallback["num_warps"] = 1
        return Config.from_dict(fallback)


def _metal_backend() -> MetalBackend:
    from ..compile_environment import CompileEnvironment
    from .backend import MetalBackend

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, MetalBackend)
    return backend


def fragment_tiles(config_spec: ConfigSpec, config: Mapping[str, object]) -> bool:
    """Whether a config's tiles run on one SIMD group's 32 threads: a
    fragment config of a kernel whose matmuls lower on fragments, whose
    ``num_warps`` SIMD groups each run the tiles, splitting its K loops
    (:func:`fragment_split`)."""
    return metal_matmul(config) == "fragment" and lowers_on_fragments(config_spec)


def fragment_split(config_spec: ConfigSpec, config: Mapping[str, object]) -> int:
    """The SIMD groups that split each K loop of a kernel's fragment runs
    (``fragment_mma``): a fragment config's ``num_warps`` where the tiles
    take one SIMD group (:func:`fragment_tiles`), else 1.  Each runs the
    tiles on every ``num_warps``-th K tile."""
    if not fragment_tiles(config_spec, config):
        return 1
    return int(cast("int", config.get("num_warps", 1)))


def tile_warps(config_spec: ConfigSpec, config: Mapping[str, object]) -> int:
    """The SIMD groups whose threads a launch lays the tiles out over:
    ``num_warps``, or one where the tiles take one SIMD group
    (:func:`fragment_tiles`)."""
    if fragment_tiles(config_spec, config):
        return 1
    return int(cast("int", config.get("num_warps", DEFAULT_NUM_WARPS)))


def _table_choices(config_spec: ConfigSpec) -> tuple[int, ...]:
    """The ``metal_tabulate_max_entries`` that generate different code for
    the kernel: 0, and each cap that admits a table a smaller one doesn't."""
    sizes = kernel_facts(config_spec).table_sizes
    choices = [0]
    for cap in METAL_TABULATE_CHOICES[1:]:
        if any(choices[-1] < size <= cap for size in sizes):
            choices.append(cap)
    return tuple(choices)


#: The ``num_warps`` of the split fragment seeds: SIMD groups splitting a K
#: loop pay off where few threadgroups run, as a few-row matmul with few
#: output columns has; the search moves it from there.
SPLIT_SEED = 4

#: The ``range_num_stages`` a loop over fragment runs takes: 0 and 1 load
#: each K tile's operands in its iteration, 2 one tile ahead
#: (``fragment_mma.prefetch_loads``).
FRAGMENT_STAGES = (0, 1, 2)


class FragmentNumStagesSpec(RangeNumStagesSpec):
    """``range_num_stages`` of a K loop over fragment runs, with Triton's
    meaning: 2 stages issue the next tile's loads before this tile's work.
    The search tries 0 and 2 (1 is 0's code).  Registered after
    ``CompileEnvironment._pin_range_num_stages_for_aliasing``, it is never
    pinned: a fragment run never loads a tensor the kernel writes from a view
    (the planner stages such a view, and ``check_fragment_group`` refuses a
    staged operand's loads of one), so no load it moves can alias a store."""

    # An enum where Triton's slot is an integer range: 1 would repeat 0.
    # pyrefly: ignore [bad-override]
    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        return EnumFragment((0, 2))

    def _normalize(self, name: str, value: object) -> int:
        value = super()._normalize(name, value)
        if value not in FRAGMENT_STAGES:
            raise InvalidConfig(
                f"{name} must be one of {FRAGMENT_STAGES} on a Metal fragment "
                f"K loop, got {value!r}"
            )
        return value


def fragment_stages(
    config_spec: ConfigSpec, config: Mapping[str, object], block: int
) -> int:
    """The config's ``range_num_stages`` for the loop over ``block``."""
    values = config.get("range_num_stages", [])
    assert isinstance(values, list)
    stages = config_spec.range_num_stages.config_get(values, block, 0)
    assert isinstance(stages, int)
    return stages


def _launch(
    config_spec: ConfigSpec, config: Mapping[str, object], budget: int
) -> tuple[list[int], int, int]:
    """A matmul config's tile sizes per count, the product of its
    literal-size tiles (a fixed ``block_size=``, one thread per element)
    and the threads its counts launch under ``budget``
    (``thread_counts.launch_threads``)."""
    facts = kernel_facts(config_spec)
    spec = config_spec.num_threads
    blocks = config["block_sizes"]
    threads = config.get("num_threads", [0] * len(spec))
    assert isinstance(blocks, list) and isinstance(threads, list)
    # A tile over a size-1 dimension runs on one thread.
    sizes = [
        1
        if item.size_hint == 1
        else int(config_spec.block_sizes.config_get(blocks, item.block_ids[0], 1))
        for item in spec
    ]
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
    return sizes, literal, launch


def mpp_one_simdgroup(config_spec: ConfigSpec, config: Mapping[str, object]) -> bool:
    """Whether an MPP config's counts, all explicit, launch one SIMD group:
    a launch ``num_warps`` doesn't change, whose ``num_warps`` SIMD groups
    split the register runs' K loops (``mpp_registers.register_split``)."""
    threads = config.get("num_threads")
    if (
        metal_matmul(config) != "mpp"
        or not isinstance(threads, list)
        or len(threads) != len(config_spec.num_threads)
        or not all(isinstance(t, int) and t > 0 for t in threads)
    ):
        return False
    return _launch(config_spec, config, MAX_THREADS_PER_THREADGROUP)[2] == SIMD_WIDTH


def fit_threads(config_spec: ConfigSpec, config: dict[str, object]) -> None:
    """Give a matmul config a threadgroup its matmuls run on: tiles over
    exactly one SIMD group for a fragment run, which each of the
    ``num_warps`` SIMD groups splitting its K loop runs, whole SIMD groups
    and at most ``MAX_THREADS_PER_THREADGROUP`` for MPP's cooperative ops
    (``mpp_matmul.check_mpp_launch``).

    ``num_warps`` alone doesn't give it: ``num_threads`` sets the
    threadgroup (``thread_counts.launch_threads``), so the search's
    ``[0, 0, 4]`` under one SIMD group launches 128 threads, ``[16, 32, 8]``
    2048 under any ``num_warps``, and ``[2, 2, 1]`` 4.  A config whose counts
    launch a threadgroup its matmuls take keeps them.  Another's are made
    explicit, each at most its block, and fitted to the budget,
    ``min(1024, 32 * num_warps)`` (32 for a fragment run): its explicit
    counts kept (the widest halved while they pass the budget), the auto
    ones filling what is left in block order, and the smallest count below
    its block grown while the threadgroup falls short.  All explicit, the
    counts launch as given.  A config with every count on auto, as the
    default config has, is left as it is.  Literal-size tiles (a fixed
    ``block_size=``) have no count and take one thread per element first;
    the counts fit what they leave.

    K tiles past ``MAX_STEPS`` are left as they are: codegen refuses them,
    which costs a codegen rather than a compile.  So is a wide MPP tile at
    too few SIMD groups, which needs ``num_warps`` raised."""
    if metal_matmul(config) == "scalar":
        # The scalar contraction runs on any threadgroup.
        return
    spec = config_spec.num_threads
    # Normalization leaves out counts that are all auto.
    threads = config.get("num_threads", [0] * len(spec))
    assert isinstance(threads, list)
    if not any(t > 0 for t in threads):
        # Every count on auto, as in the default config: left as it is.
        return
    budget = min(
        MAX_THREADS_PER_THREADGROUP, SIMD_WIDTH * tile_warps(config_spec, config)
    )
    sizes, literal, launch = _launch(config_spec, config, budget)
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
        _largest_divisor_at_most(b, t) if t > 0 else 0
        for t, b in zip(threads, sizes, strict=True)
    ]
    while math.prod(c for c in counts if c) > budget:
        widest = max(range(len(counts)), key=counts.__getitem__)
        counts[widest] = _largest_divisor_at_most(sizes[widest], counts[widest] - 1)
    left = budget // math.prod(c for c in counts if c)
    for i, size in enumerate(sizes):
        if not counts[i]:
            counts[i] = _largest_divisor_at_most(size, left)
            left //= counts[i]
    full: set[int] = set()
    while (total := math.prod(counts)) < budget:
        room = [i for i in range(len(counts)) if counts[i] < sizes[i] and i not in full]
        if not room:
            break
        smallest = min(room, key=counts.__getitem__)
        grown = _largest_divisor_at_most(
            sizes[smallest], counts[smallest] * min(2, budget // total)
        )
        if grown == counts[smallest]:
            full.add(smallest)
        counts[smallest] = grown
    config["num_threads"] = counts


class MetalSearchSurface(BackendSearchSurface):
    config_keys = frozenset({METAL_MATMUL, METAL_TABULATE})

    def extra_fields(self, config_spec: ConfigSpec) -> dict[str, ConfigSpecFragment]:
        """``metal_matmul`` where more than one lowering applies
        (:func:`matmul_choices`); ``metal_tabulate_max_entries`` where the
        kernel's fragment elements would read a table."""
        choices = matmul_choices(config_spec)
        if len(choices) == 1:
            return {}
        fields: dict[str, ConfigSpecFragment] = {METAL_MATMUL: EnumFragment(choices)}
        if (
            lowers_on_fragments(config_spec)
            and len(tables := _table_choices(config_spec)) > 1
        ):
            fields[METAL_TABULATE] = EnumFragment(tables)
        return fields

    def trim_reference_config(self, config_spec: ConfigSpec, config: Config) -> None:
        """The default config leaves ``metal_matmul``,
        ``metal_tabulate_max_entries`` and unstaged ``range_num_stages`` out,
        as before they were searched: a config without them means MPP, no
        tables and no prefetch."""
        if metal_matmul(config.config) == METAL_MATMUL_CHOICES[0]:
            config.config.pop(METAL_MATMUL, None)
        if metal_tabulate_max_entries(config.config) == 0:
            config.config.pop(METAL_TABULATE, None)
        stages = config.config.get("range_num_stages", [])
        assert isinstance(stages, list)
        if not any(stages):
            config.config.pop("range_num_stages", None)

    def seed_configs(self, config_spec: ConfigSpec) -> list[Config]:
        """Fragment configs in the initial population beside MPP's default,
        one per table cap that changes the code (the key left out at 0), each
        with and without loading its K loops' operands a tile ahead where it
        can (``range_num_stages`` slots), on one SIMD group per threadgroup
        and, where its K loops split, on :data:`SPLIT_SEED` SIMD groups
        splitting them.  The scalar contraction needs no seed: the search
        draws it from ``metal_matmul`` as it draws the others."""
        if not lowers_on_fragments(config_spec):
            return []
        seeds = []
        slots = len(config_spec.range_num_stages)
        splits = (1, SPLIT_SEED) if kernel_facts(config_spec).fragment_splits else (1,)
        for num_warps in splits:
            for cap in _table_choices(config_spec):
                for stages in (0, 2) if slots else (0,):
                    seed = Config(metal_matmul="fragment", num_warps=num_warps)
                    if cap:
                        seed.config[METAL_TABULATE] = cap
                    if stages:
                        seed.config["range_num_stages"] = [stages] * slots
                    seeds.append(seed)
        return seeds

    def normalize_knobs(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        for key, choices in (
            (METAL_MATMUL, METAL_MATMUL_CHOICES),
            (METAL_TABULATE, METAL_TABULATE_CHOICES),
        ):
            if key not in config:
                continue
            value = config[key]
            # By type too: 16.0 == 16, and False == 0.
            if value in choices and type(value) is type(choices[0]):
                continue
            if not fix_invalid:
                raise InvalidConfig(
                    f"{key} must be one of {choices}, got {config[key]!r}"
                )
            del config[key]
        if metal_matmul(config) != "fragment":
            # Tables are read by fragment elements only.
            config.pop(METAL_TABULATE, None)
        if metal_matmul(config) == "fragment" and (
            "num_warps" not in config
            or (fix_invalid and not kernel_facts(config_spec).fragment_splits)
        ):
            # A fragment config's num_warps is the SIMD groups that split
            # its K loops (``fragment_split``).  One that leaves it out gets
            # 1, as fragment configs ran before K loops split, rather than
            # the default.  Where the K loops can't split
            # (``operand_elements.split_refusal``) the search's fragment configs
            # take 1, and a config setting another value fails, saying why.
            config["num_warps"] = 1

    def normalize_before_sequences(
        self, config_spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        """Metal reads ``range_num_stages`` on fragment runs only, one entry
        per K loop that can load a tile ahead.  Before it took the key, a
        Metal config's ``range_num_stages`` (a Triton config's per-loop
        stages, pasted or cached) was dropped as a key of another backend; a
        config that doesn't lower on fragments still drops it, so it keeps
        loading whatever its length and values.  A fragment config, new with
        the key, must give one entry per such loop: the search's repairs
        drop a wrong length, and a user's config is refused."""
        stages = config.get("range_num_stages")
        if stages is None:
            return
        if metal_matmul(config) != "fragment":
            del config["range_num_stages"]
            return
        slots = len(config_spec.range_num_stages)
        if isinstance(stages, list) and len(stages) == slots:
            return
        if not fix_invalid:
            raise InvalidConfig(
                f"range_num_stages must have one entry per Metal fragment K loop "
                f"that can load a tile ahead ({slots}), got {stages!r}"
            )
        del config["range_num_stages"]

    def normalize_finalize(
        self,
        config_spec: ConfigSpec,
        config: dict[str, object],
        *,
        fix_invalid: bool,
        provided_keys: set[str],
    ) -> None:
        if not config_spec.range_num_stages:
            # A kernel without fragment K loops keeps its configs as they
            # were before Metal took the key: the sequence pass gives it an
            # empty range_num_stages, which this drops.
            config.pop("range_num_stages", None)
        if fix_invalid and kernel_facts(config_spec).thread_loops:
            fit_threads(config_spec, config)
        if (
            "num_warps" not in provided_keys
            and kernel_facts(config_spec).register_splits
            and mpp_one_simdgroup(config_spec, config)
        ):
            # An MPP config whose counts launch one SIMD group splits its
            # register runs' K loops over its num_warps SIMD groups
            # (``mpp_registers.register_split``), where the kernel lets them
            # split and nothing else of the launch draws on num_warps
            # (``register_splits``).  One that leaves num_warps out gets 1,
            # as such configs ran before, rather than the default.  Last, so
            # the counts read are the ones codegen reads: flat, whole, and
            # past the clamp of bounded blocks and fit_threads.
            config["num_warps"] = 1


METAL_SEARCH_SURFACE = MetalSearchSurface()
