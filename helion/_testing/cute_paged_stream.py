"""Shared fixtures for the paired attention schedule's page-stream knobs.

The lead of the loader's tail release, the depth of the raw V ring and the
order of a unit's pages change how one CTA streams its pages, not what it
computes or how the units are divided. Their CPU tests bind the partition
fixture's kernel on a modelled device, and their GPU tests compare a config
with the knob against the config without it.
"""

from __future__ import annotations

import ast
import functools
import importlib.abc
import importlib.util
import inspect
from pathlib import Path
import re
import types
from typing import TYPE_CHECKING
from typing import cast
import zlib

import pytest
import torch

from . import DEVICE
from . import cpu_render_target
from . import is_cuda
from .cute_capture_replay import observed_wait_modes
from .cute_codegen import cpu_bind
from .cute_paged_partition import PAGE
from .cute_paged_partition import PAGES
from .cute_paged_partition import partition_attention_args
from .cute_paged_partition import partition_attention_kernel
from .cute_paged_partition import partition_config
from .cute_paged_partition import randomize_partition_inputs
from .cute_paged_partition import set_request_lengths
from .poisoned_outputs import poison_
from .poisoned_outputs import poisoned_launch
from .poisoned_outputs import written_bits
from helion._compiler.cute import target_facts
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_shared_memory import (
    MMA_DAG_VALUE_RING_TILES_KEY,
)
from helion._compiler.cute.paged_attention_shared_memory import ValueRingTiles
from helion._compiler.cute.paged_attention_shared_memory import value_ring_reason
from helion.runtime.cute.capture_pdl import WaitMode

if TYPE_CHECKING:
    from collections.abc import Callable

    import helion
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

# The modelled device of the CPU tests: a B200.
SMS = 148
PAIRED = "attention_paired"
# The guarded dependent-launch policy of a proved paired kernel, whose
# release is the loader's tail on a grid of one wave.
CAPTURE_POLICY = {
    "cute_pdl": True,
    "cute_pdl_capture_overlap": True,
    "cute_pdl_release": "loader_tail",
}
# No key window: the kernel's mask is causal only.
NO_WINDOW = -1
# SMs the GPU tests' grid of one unit per CTA leaves idle: one wave, on
# which a ``loader_tail`` release is the loader's own.
IDLE_SMS = 20
# CTAs a persistent launch of the one-wave fixture gets in the GPU tests of
# the persistent partitions' stream options: its units (the device's SMs
# less ``IDLE_SMS``) outnumber them, so every CTA streams more than one
# unit and the balanced partition splits units between CTAs.
PERSISTENT_CTAS = 64
# Compiled pairs of the GPU tests, by their arguments: each kernel is
# compiled once per process.
_PAIRS: dict[tuple[object, ...], tuple[object, ...]] = {}


@functools.cache
def _source_function_codes(module_name: str) -> dict[str, list[types.CodeType]]:
    """The code of a module's top-level functions as its source compiles,
    by name: the import system's own code object of the module."""
    spec = importlib.util.find_spec(module_name)
    assert spec is not None
    code = cast("importlib.abc.InspectLoader", spec.loader).get_code(module_name)
    assert code is not None
    functions: dict[str, list[types.CodeType]] = {}
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            functions.setdefault(constant.co_name, []).append(constant)
    return functions


def jit_host_body(function: object) -> Callable[..., object]:
    """The Python body of a module-level ``cute.jit`` function, for a host
    simulation.

    The DSL replaces a jit function's code with its rewritten form the first
    time a kernel traces it in a process, so ``__wrapped__`` runs plain
    Python only until some test of the process has compiled a kernel that
    calls the function. This runs the body the module's source compiles to,
    which is the same in every process and at every point of one: nothing
    the DSL keeps or renames is read. A function the module's source does
    not define at its top level, exactly once, is a ``KeyError`` or a
    ``ValueError``.
    """
    plain = cast("types.FunctionType", function.__wrapped__)  # type: ignore[attr-defined]
    (code,) = _source_function_codes(plain.__module__)[plain.__name__]
    return types.FunctionType(
        code, plain.__globals__, plain.__name__, plain.__defaults__, plain.__closure__
    )


def source_function(name: str, module: types.ModuleType) -> ast.FunctionDef:
    """The definition of function ``name`` in the source of ``module``, as
    its syntax tree: the one function of that name at any depth. A CPU test
    reads from it what a device template's code says, statement by
    statement."""
    (found,) = [
        node
        for node in ast.walk(ast.parse(Path(inspect.getfile(module)).read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return found


def source_calls(node: ast.AST, name: str) -> list[ast.Call]:
    """The calls of ``name`` in the syntax tree ``node``, by the last part
    of the called name: ``prims.barrier_cta_sync(...)`` is a call of
    ``barrier_cta_sync``."""
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func).rpartition(".")[2] == name
    ]


@functools.cache
def stream_bound(
    units: int = 128,
    *,
    sms: int = SMS,
    input_banks: int = 1,
    window: int | None = None,
    fast_math: bool = True,
    smem_capacity_bytes: int | None = None,
    rows: int = 8,
    table_pages: int = PAGES,
    pool_pages: int = 13,
    reference_scores: bool = False,
) -> BoundKernel[object]:
    """The partition fixture's kernel over ``units`` head pairs, bound without
    a GPU on a device of ``sms`` SMs.

    ``window`` None keeps the fixture's window over the whole page table
    (a windowed kernel); ``NO_WINDOW`` is a kernel without one. More than
    one input bank is the benchmark that times a search's candidates back to
    back in one captured graph. ``smem_capacity_bytes`` models a device with
    another shared-memory capacity per CTA than the B200's. ``rows`` are the
    query rows of an item: 8, 32 or 40 query columns. ``table_pages`` is the
    page table's extent and ``pool_pages`` the pool's. ``reference_scores``
    is the fixture's recurrence over scores relative to the running maximum,
    the one bind that is offered the early probability schedules.
    """
    args = partition_attention_args(
        units=units,
        rows=rows,
        reference_scores=reference_scores,
        window=table_pages * PAGE - 1 if window is None else window,
        table_pages=table_pages,
        pool_pages=pool_pages,
    )
    target = (
        cpu_render_target(fake_device_sm_count=sms, num_sm=sms)
        if smem_capacity_bytes is None
        else cpu_render_target(
            fake_device_sm_count=sms,
            num_sm=sms,
            smem_capacity_bytes=smem_capacity_bytes,
        )
    )
    with target:
        return cpu_bind(
            cast(
                "Kernel[object]",
                partition_attention_kernel(
                    fast_math=fast_math, input_banks=input_banks
                ),
            ),
            args,
            num_sm=sms,
        )


def stream_config(rows: int = 8, **overrides: object) -> helion.Config:
    """The paired schedule with V two tiles behind K, under its default
    policies (the guarded capture overlap, released from the loader's tail
    on one wave) unless ``overrides`` name ``cute_pdl``; ``rows`` are the
    bound kernel's query rows."""
    knobs: dict[str, object] = {
        "block_sizes": [1 << (rows - 1).bit_length()],
        "cute_mma_dag_v_lag": 2,
    }
    knobs.update(overrides)
    config = partition_config(default_policies="cute_pdl" not in knobs)
    config.config.update(knobs)
    return config


def stream_render(
    bound: BoundKernel[object], config: helion.Config, *, sms: int = SMS
) -> str:
    with cpu_render_target(fake_device_sm_count=sms, num_sm=sms):
        return bound.to_code(config)


def body_keywords(source: str) -> dict[str, str]:
    """The keyword arguments of the rendered kernel's call of the paired
    body, each as source text."""
    (call,) = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).rpartition(".")[2] == "paged_attention_paired_body"
    ]
    return {
        cast("str", keyword.arg): ast.unparse(keyword.value)
        for keyword in call.keywords
    }


def stream_seeds(
    bound: BoundKernel[object], spec: ConfigSpec | None = None, *, sms: int = SMS
) -> list[helion.Config]:
    """The compiler seeds of ``bound``, or of ``spec`` on its facts."""
    spec = bound.config_spec if spec is None else spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=sms), bound.env, bound.host_function:
        planner = MmaDagSeedPlanner(facts, spec, occupancy_key="cute_min_blocks_per_mp")
        return planner.configs(planner.ranked())


def seeds_ahead_of_the_partition_look(
    seeds: list[helion.Config],
) -> list[helion.Config]:
    """The seeds ahead of those whose spare warp looks for a unit's lowest
    part, which a windowed kernel on a grid of more units than SMs has
    behind every other seed where its search samples the boundary prefetch:
    the range-ahead seeds once more with that look
    (``test_cute_paged_attention_partition_look`` pins them)."""
    looked = [seed for seed in seeds if "cute_mma_dag_partition_look" in seed.config]
    ahead = seeds[: len(seeds) - len(looked)]
    assert seeds[len(ahead) :] == looked
    return ahead


# What a softmax tile loop that the assembler cannot prove warp-uniform
# carries on its tile path: a yield, convergence regions around its waits,
# warp synchronizations and divergent branches.
_LOOP_FORMS = ("TRYWAIT", "YIELD", "BSSY", "BSYNC", "WARPSYNC", "BRA.DIV")
# The try-waits of the loop that stand inside a convergence region.
LOOP_WAITS_IN_REGIONS = "TRYWAIT in a region"
_SASS_LABEL = re.compile(r"^\s*(\.L_x_\d+):")
_SASS_INSTRUCTION = re.compile(r"/\*[0-9a-f]{4,6}\*/\s+(.*);")
_SASS_BRANCH = re.compile(r"\bBRA(?:\.\w+)*\s+(?:!?U?P\d+,\s*)?`\((\.L_x_\d+)\)")


def _sass_form(form: str, instruction: str) -> bool:
    return bool(re.search(r"(^|[\s.])" + re.escape(form) + r"\b", instruction))


def softmax_loop_forms(sass: str) -> dict[str, int]:
    """Counts, in the innermost loop around the softmax's maximum vote of a
    disassembled paired kernel (``nvdisasm -c``; a kernel has one such
    loop), of its mbarrier try-waits and of what a loop that is not
    warp-uniform adds to them: ``YIELD``, ``BSSY`` / ``BSYNC`` regions,
    ``WARPSYNC`` and ``BRA.DIV``; and, as ``LOOP_WAITS_IN_REGIONS``, of the
    try-waits that stand between a ``BSSY`` and its ``BSYNC``.

    The loop is the shortest span from a label to a branch back to it that
    holds a vote, ``BAR.RED.OR``. The recurrence over reference scores
    votes twice a tile (whether a row's maximum changed and, behind such
    an event, whether a row's reference is invalid) where the per-key one
    votes once: the loop around either vote is the same loop."""
    rows: list[tuple[str, str]] = []
    for line in sass.splitlines():
        label = _SASS_LABEL.match(line)
        if label:
            rows.append(("label", label.group(1)))
            continue
        instruction = _SASS_INSTRUCTION.search(line)
        if instruction:
            rows.append(("instruction", instruction.group(1).strip()))
    labels = {text: index for index, (kind, text) in enumerate(rows) if kind == "label"}
    votes = [
        index
        for index, (kind, text) in enumerate(rows)
        if kind == "instruction" and "BAR.RED.OR" in text
    ]
    assert votes
    loops = set()
    for vote in votes:
        spans = []
        for index, (kind, text) in enumerate(rows):
            branch = _SASS_BRANCH.search(text) if kind == "instruction" else None
            if branch and labels.get(branch.group(1), index) < vote < index:
                spans.append(
                    (index - labels[branch.group(1)], labels[branch.group(1)], index)
                )
        loops.add(min(spans))
    ((_length, top, back),) = loops
    body = [text for kind, text in rows[top : back + 1] if kind == "instruction"]
    forms = {form: sum(_sass_form(form, text) for text in body) for form in _LOOP_FORMS}
    # A region is the instructions from its ``BSSY`` to its ``BSYNC``, which
    # the disassembly lists in that order, an inner region inside its outer.
    depth = 0
    enclosed = 0
    for text in body:
        if _sass_form("BSSY", text):
            depth += 1
        elif _sass_form("BSYNC", text):
            depth -= 1
            assert depth >= 0
        elif _sass_form("TRYWAIT", text) and depth > 0:
            enclosed += 1
    assert depth == 0
    forms[LOOP_WAITS_IN_REGIONS] = enclosed
    return forms


def seeds_ahead_of_the_range_ahead(
    seeds: list[helion.Config],
) -> list[helion.Config]:
    """The seeds ahead of those that name a distance of the range-ahead
    prefetch, which a windowed kernel on a grid of more units than SMs has
    behind every other seed but the spare warp's look's where its search
    samples the boundary prefetches
    (``test_cute_paged_attention_range_ahead`` pins them)."""
    seeds = seeds_ahead_of_the_partition_look(seeds)
    served = [
        seed for seed in seeds if "cute_mma_dag_boundary_prefetch_ahead" in seed.config
    ]
    ahead = seeds[: len(seeds) - len(served)]
    assert seeds[len(ahead) :] == served
    return ahead


def seeds_ahead_of_the_cold_prefetch(
    seeds: list[helion.Config],
) -> list[helion.Config]:
    """The seeds ahead of those that name the sibling prefetch's depth in a
    cold launch, which a windowed kernel on a grid of fewer units than SMs
    has behind every other seed but the range-ahead ones where its search
    samples that prefetch (``test_cute_paged_attention_prefetch_cold`` pins
    them)."""
    seeds = seeds_ahead_of_the_range_ahead(seeds)
    cold = [
        seed for seed in seeds if "cute_mma_dag_boundary_prefetch_cold" in seed.config
    ]
    ahead = seeds[: len(seeds) - len(cold)]
    assert seeds[len(ahead) :] == cold
    return ahead


def seeds_ahead_of_the_store_hint(
    seeds: list[helion.Config],
) -> list[helion.Config]:
    """The seeds ahead of those that store their outputs under the L2
    evict-last hint, which a kernel with a sliding window has behind every
    other seed but the cold prefetch's and the range-ahead ones
    (``test_cute_paged_attention_store_policy`` pins them)."""
    seeds = seeds_ahead_of_the_cold_prefetch(seeds)
    hinted = [
        seed
        for seed in seeds
        if seed.config.get("cute_store_policy") == "l2_evict_last"
    ]
    ahead = seeds[: len(seeds) - len(hinted)]
    assert seeds[len(ahead) :] == hinted
    return ahead


def seeds_ahead_of_the_reversed_order(
    seeds: list[helion.Config],
) -> list[helion.Config]:
    """The seeds ahead of those in reversed page order, which a kernel that
    is offered that order has behind every other seed but the store hint's
    (``test_cute_paged_attention_reversed_order`` pins them): what the
    planner seeds for the other knobs, in its order."""
    seeds = seeds_ahead_of_the_store_hint(seeds)
    walked = [
        seed
        for seed in seeds
        if seed.config.get("cute_mma_dag_page_order") == "reversed"
    ]
    ahead = seeds[: len(seeds) - len(walked)]
    assert seeds[len(ahead) :] == walked
    return ahead


def stream_step(
    spec: ConfigSpec, parent: helion.Config, **moves: object
) -> tuple[helion.Config, helion.Config]:
    """The configs one requested step from ``parent`` gives a search: with
    the prerequisites the backend adds, and as requested."""
    generation = spec.create_config_generation()
    base, _config = generation.canonicalize_flat(generation.flatten(parent))
    slots = {key: entry[0][0] for key, entry in generation._key_to_flat_indices.items()}
    requested = list(base)
    for key, value in moves.items():
        requested[slots[key]] = value
    neighbor = generation.conditional_neighbor(base, requested)
    return generation.unflatten(list(neighbor)), generation.unflatten(requested)


def transferred_seeds(
    spec: ConfigSpec, overrides: dict[str, object] | None
) -> tuple[list[helion.Config], list[str]]:
    """The compiler seeds a search under ``overrides`` starts from, and the
    reasons of those it rejected."""
    failures: list[str] = []
    generation = spec.create_config_generation(overrides=overrides)
    pairs = generation.seed_flat_config_pairs(failures.append)
    return [config for _flat, config in pairs], failures


def seed_transfers(
    spec: ConfigSpec, overrides: dict[str, object] | None
) -> list[tuple[helion.Config, helion.Config | None]]:
    """Each compiler seed of ``spec`` beside the config a search under
    ``overrides`` starts from in its place, ``None`` for a seed the search
    rejects: ``transferred_seeds`` seed by seed, which leaves out a seed
    that became the config of one ahead of it as well."""
    generation = spec.create_config_generation(overrides=overrides)
    moves: list[tuple[helion.Config, helion.Config | None]] = []
    for seed in spec.compiler_seed_configs:
        pair = generation.transfer_seed(seed)
        moves.append((seed, None if pair is None else pair[1]))
    return moves


def launched_bits(kernel: object, args: tuple[object, ...]) -> list[torch.Tensor]:
    """The bits of one eager call's outputs, every element stored by it."""
    return poisoned_launch(lambda: cast("Callable[..., object]", kernel)(*args))


def replayed_bits(result: object) -> list[torch.Tensor]:
    """The bits of a captured call's outputs after a replay."""
    return [written_bits(value) for value in cast("tuple[torch.Tensor, ...]", result)]


def poison_results(results: list[object]) -> None:
    """Overwrite the outputs of captured calls ahead of a replay."""
    for result in results:
        poison_(*cast("tuple[torch.Tensor, ...]", result))


def assert_same_bits(
    actual: list[torch.Tensor], expected: list[torch.Tensor], what: object
) -> None:
    for value, reference in zip(actual, expected, strict=True):
        assert torch.equal(value, reference), what


def one_wave_units() -> int:
    """Units of a grid of one wave on the test device, which a paired
    attention CTA needs to be an SM100 part."""
    if not is_cuda():
        pytest.skip("the paired attention schedule requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    if sms <= 2 * IDLE_SMS:
        pytest.skip("the grid leaves twenty SMs idle")
    return sms - IDLE_SMS


def device_shared_memory() -> int:
    """Shared memory the test device grants one CTA, in bytes: what a bind
    on it records as its capacity."""
    return target_facts.per_cta_smem_capacity_bytes(torch.device(DEVICE))


def pair_seed(key: tuple[object, ...]) -> int:
    """The seed of a compiled pair's random inputs: a function of the
    arguments the pair is cached under (strings, numbers and tuples of
    them), so a pair has the same inputs in every process and in whatever
    order a worker's tests compile their pairs. The bounds of the page
    orders' rounding are checked on those inputs."""
    return zlib.crc32(repr(key).encode())


def stream_pair(
    candidate: dict[str, object],
    *,
    schedule: dict[str, object] | None = None,
    window: int | None = None,
    reverse_pages: bool = False,
    rows: int = 8,
    reserved_sms: int = 0,
    table_pages: int = PAGES,
    pool_pages: int = 13,
    bounds: bool = False,
    reference_scores: bool = False,
) -> tuple[tuple[object, ...], object, object, str, str]:
    """A one-wave grid's arguments on the test device, its kernel under the
    paired schedule's default policies (the guarded capture overlap,
    released from the loader's tail) with the knobs of ``schedule``, the
    same kernel with ``candidate``'s knobs as well, and the sources of both.

    ``window`` None keeps the fixture's window over the whole page table,
    ``reserved_sms`` shrinks the grid of a persistent partition, and
    ``rows`` are the query rows of an item. ``table_pages`` is the page
    table's extent and ``pool_pages`` the pool's; ``bounds`` gives the
    kernel each request's page range in a tensor instead of deriving it
    from the length, and ``reference_scores`` is the fixture's recurrence
    over scores relative to the running maximum. Both kernels are warmed:
    the capture guard tracks only launches of a warmed kernel.
    """
    key = (
        tuple(sorted(candidate.items())),
        tuple(sorted((schedule or {}).items())),
        window,
        reverse_pages,
        rows,
        reserved_sms,
        table_pages,
        pool_pages,
        bounds,
        reference_scores,
    )
    if key not in _PAIRS:
        units = one_wave_units()
        values = list(
            partition_attention_args(
                units=units,
                rows=rows,
                reference_scores=reference_scores,
                reverse_pages=reverse_pages,
                window=table_pages * PAGE - 1 if window is None else window,
                table_pages=table_pages,
                pool_pages=pool_pages,
                device=DEVICE,
            )
        )
        if bounds:
            values[10] = torch.zeros(units, 2, dtype=torch.int32, device=DEVICE)
        args = tuple(values)
        randomize_partition_inputs(args, pair_seed(key))
        bound = partition_attention_kernel(
            reverse_pages=reverse_pages, reserved_sms=reserved_sms
        ).bind(args)
        knobs: dict[str, object] = {
            "block_sizes": [1 << (rows - 1).bit_length()],
            "cute_mma_dag_v_lag": 2,
            **(schedule or {}),
        }
        configs = []
        for extra in ({}, candidate):
            config = partition_config(default_policies=True)
            config.config.update({**knobs, **extra})
            configs.append(config)
        sources = [bound.to_code(config) for config in configs]
        kernels = [bound.compile_config(config) for config in configs]
        set_request_lengths(args, [PAGE] * units)
        for kernel in kernels:
            kernel(*args)
        torch.cuda.synchronize()
        _PAIRS[key] = (args, *kernels, *sources)
    return cast("tuple[tuple[object, ...], object, object, str, str]", _PAIRS[key])


def persistent_pair(
    candidate: dict[str, object],
    ctas: int,
    rendered: dict[str, str],
    *,
    window: int = PAGES * PAGE - 1,
    rows: int = 8,
    **schedule: object,
) -> tuple[tuple[object, ...], object, object]:
    """The one-wave fixture's arguments, its kernel on a persistent
    partition of ``ctas`` CTAs under ``schedule``, and the same kernel with
    ``candidate``'s options.

    ``rendered`` are the keywords the candidate's call of the paired body
    has for those options, each with its source text, none of which the
    kernel without them has. A candidate that counts its raw V ring in
    tiles is skipped on a device whose CTA does not hold the count."""
    units = one_wave_units()
    tiles = candidate.get(MMA_DAG_VALUE_RING_TILES_KEY)
    if tiles:
        reason = value_ring_reason(
            ValueRingTiles(cast("int", tiles)),
            device_shared_memory(),
            8 if rows <= 8 else 32 if rows <= 32 else 40,
            query_stages=2,
            units=units,
            balanced=schedule.get("cute_mma_dag_partition") == "balanced",
        )
        if reason is not None:
            pytest.skip(reason)
    if not ctas < units:
        pytest.skip("the grid needs more units than CTAs")
    args, off, on, plain, source = stream_pair(
        candidate,
        schedule=schedule,
        window=window,
        rows=rows,
        reserved_sms=units + IDLE_SMS - ctas,
    )
    plain_keywords, keywords = body_keywords(plain), body_keywords(source)
    for name, text in rendered.items():
        assert name not in plain_keywords, name
        assert keywords[name] == text, name
    return args, off, on


def check_persistent_bits(
    candidate: dict[str, object],
    ctas: int,
    rendered: dict[str, str],
    *,
    window: int = PAGES * PAGE - 1,
    rows: int = 8,
    calls: int = 3,
    **schedule: object,
) -> None:
    """``persistent_pair`` over streams of every length the table has, eager
    and captured: the kernel with the options stores the bits of the kernel
    without them."""
    args, off, on = persistent_pair(
        candidate, ctas, rendered, window=window, rows=rows, **schedule
    )
    units = cast("torch.Tensor", args[6]).numel()
    check_same_bits(
        args,
        off,
        on,
        stream_patterns(units, window),
        calls=calls,
        what=(candidate, ctas, rows, schedule),
    )


def stream_patterns(
    units: int, window: int, table_pages: int = PAGES
) -> dict[str, list[int]]:
    """Request lengths for a page table of ``table_pages`` pages: streams of
    every length from none to the whole table, in several mixes."""
    generator = torch.Generator().manual_seed(units)
    limit = table_pages * PAGE
    ragged = torch.randint(0, limit + 1, (units,), generator=generator).tolist()
    return {
        "ragged": ragged,
        "even empty": [
            0 if index % 2 == 0 else ragged[index] for index in range(units)
        ],
        "all empty": [0] * units,
        # At most three pages: fewer than every lead of the tail release,
        # the default's four among them.
        "short": torch.randint(0, 3 * PAGE, (units,), generator=generator).tolist(),
        "full": [limit] * units,
        # Lengths at and next to page boundaries, in every page of the table.
        "page edges": [
            (index % table_pages + 1) * PAGE - index % 3 for index in range(units)
        ],
        # A window's first page moves with the length: a clipped first tile.
        "window edges": [
            min(limit, window + 5 + index * 37 % max(limit - window, 1))
            for index in range(units)
        ],
    }


def check_same_bits(
    args: tuple[object, ...],
    reference: object,
    candidate: object,
    patterns: dict[str, list[int]],
    *,
    calls: int = 3,
    overlapped: bool = True,
    what: object = None,
) -> None:
    """Every pattern eager, then as ``calls`` captured back-to-back launches
    replayed over the patterns: ``candidate`` stores the bits ``reference``
    stores. ``overlapped``: the captured launches carry the guarded capture
    policy, and each call but the first overlaps the one ahead."""
    expected = {}
    for name, lengths in patterns.items():
        set_request_lengths(args, lengths)
        expected[name] = launched_bits(reference, args)
        assert_same_bits(
            launched_bits(candidate, args), expected[name], (what, name, "eager")
        )
    graph = torch.cuda.CUDAGraph()
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        captured = [
            cast("Callable[..., object]", candidate)(*args) for _ in range(calls)
        ]
    if overlapped:
        assert modes == [WaitMode.EARLY] + [WaitMode.BEFORE_WRITES] * (calls - 1), modes
    for name, lengths in patterns.items():
        set_request_lengths(args, lengths)
        poison_results(captured)
        graph.replay()
        torch.cuda.synchronize()
        for index, result in enumerate(captured):
            assert_same_bits(replayed_bits(result), expected[name], (what, name, index))
