"""Code shape of the ``cute_pdl`` knob (programmatic dependent launch).

Knob on: the module tags the kernel with ``_helion_cute_use_pdl = True`` for
the launcher and the kernel executes ``cute.arch.griddepcontrol_wait()``
ahead of its first memory access; nothing else changes.  Passes that run
after the wait is spliced in may hoist register allocations above it, so the
invariant is "no load, store, atomic or bulk copy ahead of the wait, and no
release of the dependents", not "the wait is the first statement".  Knob off
(the default): neither line appears.  The knob-on kernel also releases its
own dependents (``cute_pdl_release``): right after the wait (``entry``, so
the next kernel of a sequence launches early too) or after the body's last
statement (``exit``); ``none`` leaves the implicit trigger at grid
completion; ``last_wave`` guards the entry release to the CTAs of the
launch's last wave (plain entry for a one-wave grid); ``auto`` (the default,
and the only searched value) resolves at codegen to ``entry`` while the
launch grid is known to fit in one wave of resident CTAs and to
``last_wave`` past it or when the host sizes the grid at runtime, counting
the CTAs the launcher receives (every root of a multi-root kernel, the
parent grid of an L2-grouped one, the persistent grid, the cluster size)
against the SM count times the CTAs an SM holds (from the launch block, or
``cute_min_blocks_per_mp`` when set).  The kernels that already
take part in dependent launch (the fission producer that releases its
dependents at entry, the tcgen05 consumer with its role-local wait) are
covered in ``test_cute_materialized_pdl.py`` and ``test_cute_tcgen05_pdl.py``;
GPU semantics live in ``test_cute_pdl_cuda.py``.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

from examples.aot_example import rms_norm_batched
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test._cute_vloop_sink_kernels import col_reduce_max_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_from8_static
from test._cute_vloop_sink_kernels import col_reduce_sum_lower_triangle_static
from test._cute_vloop_sink_kernels import col_reduce_sum_pair_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_static
from test._cute_vloop_sink_kernels import col_weighted_mean_static
from test.test_cute_dynamic_vector_reduction import _bind_rms_norm
from test.test_cute_dynamic_vector_reduction import _dynamic_kernel
from test.test_cute_dynamic_vector_reduction import _rms_norm_specialized_n
from test.test_cute_dynamic_vector_reduction import _rolled_config

import helion
from helion import exc
from helion._compiler.cute.pdl_release import resident_ctas_per_sm
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def scale_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * w[tile_n][None, :]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def two_roots(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two top-level loops: one launch whose grid is the sum of both."""
    m, n = x.size()
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    for tile_m, tile_n in hl.tile([m, n]):
        out_x[tile_m, tile_n] = x[tile_m, tile_n] * 2
    for tile_m, tile_n in hl.tile([m, n]):
        out_y[tile_m, tile_n] = y[tile_m, tile_n] + 1
    return out_x, out_y


_COLUMN_SUM = {
    "block_sizes": [1024, 16],
    "num_threads": [128, 4],
    "cute_vector_widths": [1, 4],
    "cute_lane_layouts": ["strided", "blocked"],
    "cute_vloop_sink": True,
    "cute_lane_unroll": 8,
}

_WAIT = "cute.arch.griddepcontrol_wait()"
_RELEASE = "cute.arch.griddepcontrol_launch_dependents()"
_ATTRIBUTE = "._helion_cute_use_pdl = True"
# Attribute names of calls that touch memory or take part in dependent launch.
_MEMORY_TOKENS = (
    "load",
    "store",
    "copy",
    "atomic",
    "cp_async",
    "tma",
    "mbarrier",
    "fence",
    "griddepcontrol",
)


def _launch(code: str) -> tuple[str, int]:
    """The launcher call's grid expression and its CTA's thread count."""
    (call,) = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    (block,) = [keyword.value for keyword in call.keywords if keyword.arg == "block"]
    assert isinstance(block, ast.Tuple)
    threads = 1
    for element in block.elts:
        assert isinstance(element, ast.Constant)
        threads *= int(element.value)
    return ast.unparse(call.args[1]), threads


def _kernel_def(code: str) -> ast.FunctionDef:
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
    ]
    return kernel


def _on_and_off(bound: Any, config: dict[str, object]) -> tuple[str, str]:
    spec = bound.config_spec
    base = {key: value for key, value in config.items() if key != "cute_pdl"}
    on = bound.to_code(
        spec.normalized_config(helion.Config.from_dict(base | {"cute_pdl": True}))
    )
    off = bound.to_code(spec.normalized_config(helion.Config.from_dict(base)))
    return on, off


def _codes(kernel: helion.Kernel, args: tuple, **config: object) -> tuple[str, str]:
    return _on_and_off(_cpu_bind(kernel, args), config)


def _statements_ahead_of_the_wait(kernel: ast.FunctionDef) -> list[str]:
    """The wait is a top-level statement of the kernel body and nothing ahead
    of it touches memory, at any nesting depth: no tensor indexing and no
    call that loads, stores, copies, syncs a barrier or speaks
    ``griddepcontrol``.  Returns what does precede it."""
    waits = [
        index
        for index, statement in enumerate(kernel.body)
        if ast.unparse(statement) == _WAIT
    ]
    assert len(waits) == 1, ast.unparse(kernel)[:600]
    ahead = kernel.body[: waits[0]]
    for statement in ahead:
        for node in ast.walk(statement):
            assert not isinstance(node, ast.Subscript), ast.unparse(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not any(token in node.func.attr for token in _MEMORY_TOKENS), (
                    ast.unparse(statement)
                )
    return [ast.unparse(statement) for statement in ahead]


def _assert_only_the_dependent_launch_differs(on: str, off: str) -> list[str]:
    kernel = _kernel_def(on)
    ahead = _statements_ahead_of_the_wait(kernel)
    assert on.count(_WAIT) == 1
    assert on.count(_ATTRIBUTE) == 1
    assert f"{kernel.name}{_ATTRIBUTE}" in on
    # The default placement (``auto``) releases the dependents right after
    # the wait while the grid fits in one wave (``_implicit_pdl_release``);
    # a grid that needs more waves, or one sized at runtime, releases from
    # the CTAs of its last wave (``_last_wave_guard``).
    assert on.count(_RELEASE) == 1
    _last_wave_guard(on)
    assert _WAIT not in off and _ATTRIBUTE not in off and _RELEASE not in off
    assert _without_dependent_launch(on) == ast.unparse(ast.parse(off))
    return ahead


def _without_dependent_launch(code: str) -> str:
    """``code`` without the launch attribute, the wait and the release (the
    plain one or the index and guard of ``last_wave``), unparsed (comments
    dropped) for comparison with the knob-off code."""
    module = ast.parse(code)
    name = _kernel_def(code).name
    module.body = [
        node
        for node in module.body
        if not (
            isinstance(node, ast.Assign) and ast.unparse(node) == f"{name}{_ATTRIBUTE}"
        )
    ]
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    statements = [ast.unparse(s) for s in kernel.body]
    wait = statements.index(_WAIT)
    end = wait + 2 if statements[wait + 1] == _RELEASE else wait + 3
    del kernel.body[wait:end]
    return ast.unparse(module)


def test_the_column_sum_waits_before_its_first_global_access(cpu_only: None) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    on, off = _codes(col_reduce_sum_static, (x,), **_COLUMN_SUM)
    assert _assert_only_the_dependent_launch_differs(on, off) == []
    assert on.count(_RELEASE) == 1  # 64 CTAs: one wave
    assert "_vsink_vec" in on
    # The attribute is set on the kernel object the host code launches.
    kernel = _kernel_def(on)
    assert f"_launcher({kernel.name}," in on


def test_dynamic_grids_and_pointwise_kernels_follow_the_grid_rule(
    cpu_only: None,
) -> None:
    x = torch.empty(1000, 1008, dtype=torch.bfloat16)
    on, off = _codes(col_reduce_sum_dynamic, (x,), **_COLUMN_SUM)
    assert _assert_only_the_dependent_launch_differs(on, off) == []
    # A grid the host sizes at runtime releases from its last wave, sized
    # against the launched grid (512-thread CTAs: 4 per SM, 592 slots).
    assert _last_wave_guard(on) == f"if pdl_cta >= {_GRID_CTAS} - 592:"
    w = torch.empty(1008, dtype=torch.bfloat16)
    pointwise = {"block_sizes": [16, 64], "num_threads": [4, 64]}
    on, off = _codes(scale_rows, (x, w), **pointwise)
    assert _assert_only_the_dependent_launch_differs(on, off) == []
    # 63 x 16 = 1008 static CTAs of 256 threads: 8 per SM, one wave of the
    # 148 SMs' 1184 slots, released at plain entry.
    assert _launch(on) == ("(1008,)", 256)
    assert _last_wave_guard(on) is None
    # At the occupancy the config asks ptxas for (4 CTAs per SM, 592
    # slots) the same grid runs in two waves: the last 592 CTAs release.
    on, off = _codes(scale_rows, (x, w), **pointwise, cute_min_blocks_per_mp=4)
    assert _assert_only_the_dependent_launch_differs(on, off) == []
    assert _last_wave_guard(on) == "if pdl_cta >= 416:"


@pytest.mark.parametrize(
    "kernel,shapes",
    [
        (col_reduce_max_dynamic, [(1000, 1008)]),
        (col_reduce_sum_from8_static, [(1024, 1024)]),
        (col_reduce_sum_lower_triangle_static, [(1024, 1024)]),
        (col_reduce_sum_pair_dynamic, [(1000, 1008), (1000, 1008)]),
        (col_weighted_mean_static, [(1024, 1024), (1024,)]),
    ],
)
def test_column_reductions_take_the_knob_under_their_default_config(
    cpu_only: None, kernel: helion.Kernel, shapes: list[tuple[int, ...]]
) -> None:
    args = tuple(torch.empty(*shape, dtype=torch.bfloat16) for shape in shapes)
    bound = _cpu_bind(kernel, args)
    on, off = _on_and_off(bound, dict(bound.config_spec.default_config().config))
    assert _assert_only_the_dependent_launch_differs(on, off) == []


@pytest.mark.parametrize("static_shapes", [False, True])
def test_hoisted_register_allocations_may_precede_the_wait(
    cpu_only: None, static_shapes: bool
) -> None:
    # The two-pass load fuser runs after the wait is spliced in and hoists
    # its register cache to the top of the kernel.  That is no memory
    # access, so it may sit ahead of the wait; the first load still follows.
    fn = rms_norm_batched.fn if static_shapes else _rms_norm_specialized_n
    x = torch.empty((13, 4096), dtype=torch.bfloat16)
    args = (x, 1e-5)
    bound = _cpu_bind(_dynamic_kernel(fn, static_shapes=static_shapes), args)
    config = _rolled_config(bound, rows=4, threads=64, vec=8, chunk=1024)
    on, off = _on_and_off(bound, dict(config.config))
    assert _assert_only_the_dependent_launch_differs(on, off) == [
        "_fuse_cache_0 = cute.make_rmem_tensor(64, cutlass.Uint16)"
    ]


def test_the_knob_is_a_searchable_boolean(cpu_only: None) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))
    spec = bound.config_spec
    assert spec.default_config().config.get("cute_pdl", False) is False
    with pytest.raises(exc.InvalidConfig, match="cute_pdl must be a boolean"):
        spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl="yes"))
    assert (
        spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl=False)).config[
            "cute_pdl"
        ]
        is False
    )
    # The search space carries the knob as a boolean fragment and searches
    # both values for this kernel (no GEMM, sm_100 target).
    from helion.autotuner.config_generation import ConfigGeneration

    generation = ConfigGeneration(spec)
    assert "cute_pdl" in {key for key, _count, _seq in spec.flat_key_layout()}
    assert spec._flat_fields()["cute_pdl"].search_choices == (False, True)
    values = {
        generation.unflatten(generation.flatten(config)).config.get("cute_pdl")
        for config in (
            helion.Config(**_COLUMN_SUM, cute_pdl=True),
            helion.Config(**_COLUMN_SUM, cute_pdl=False),
        )
    }
    assert values == {True, False}


def test_pre_hopper_targets_neither_search_nor_apply_the_knob(
    cpu_only: None,
) -> None:
    # ``griddepcontrol`` needs sm_90: below it the fragment offers only the
    # off value and an explicit ``cute_pdl=True`` generates the knob-off code.
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))
    spec = bound.config_spec
    with patch.object(spec, "target_device_capability", (8, 0)):
        assert spec._flat_fields()["cute_pdl"].search_choices == (False,)
        on, off = _on_and_off(bound, _COLUMN_SUM)
    assert _WAIT not in on and _ATTRIBUTE not in on
    assert on == off


def _release_code(bound: Any, config: dict[str, object], placement: str) -> str:
    spec = bound.config_spec
    base = {
        key: value
        for key, value in config.items()
        if key not in ("cute_pdl", "cute_pdl_release")
    }
    return bound.to_code(
        spec.normalized_config(
            helion.Config.from_dict(
                base | {"cute_pdl": True, "cute_pdl_release": placement}
            )
        )
    )


def test_the_release_placements_add_one_line_in_the_named_spot(
    cpu_only: None,
) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))
    on, _off = _on_and_off(bound, _COLUMN_SUM)
    # ``entry`` is the knob-on code of the tests above; ``none`` is that code
    # without the release.
    assert _release_code(bound, _COLUMN_SUM, "entry") == on
    none = _release_code(bound, _COLUMN_SUM, "none")
    assert _RELEASE not in none
    assert none.splitlines() == [
        line for line in on.splitlines() if _RELEASE not in line
    ]
    for placement in ("exit", "entry"):
        code = _release_code(bound, _COLUMN_SUM, placement)
        assert code.count(_RELEASE) == 1, placement
        stripped = [line for line in code.splitlines() if _RELEASE not in line]
        assert stripped == none.splitlines(), placement
        statements = [ast.unparse(s) for s in _kernel_def(code).body]
        wait = statements.index(_WAIT)
        if placement == "entry":
            # Right after the wait: the dependents may launch before this
            # kernel's first memory access, their own wait orders them.
            assert statements[wait + 1] == _RELEASE
        else:
            # After the body's last statement, with the body between.
            assert statements[-1] == _RELEASE
            assert len(statements) - 1 > wait + 1


def test_the_release_knob_rides_on_cute_pdl(cpu_only: None) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))
    spec = bound.config_spec
    assert "cute_pdl_release" not in spec.default_config().config
    # Without the knob the placement has nothing to act on: the default one
    # (``auto``) is dropped, a non-default one stays (inert, so a flat
    # configuration round-trips through the search) and either way the code
    # is the knob-off code.
    plain = bound.to_code(spec.normalized_config(helion.Config(**_COLUMN_SUM)))
    off = spec.normalized_config(
        helion.Config(**_COLUMN_SUM, cute_pdl=False, cute_pdl_release="auto")
    )
    assert "cute_pdl_release" not in off.config
    assert bound.to_code(off) == plain
    for placement in ("entry", "none", "exit", "last_wave"):
        off_placed = spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=False, cute_pdl_release=placement)
        )
        assert off_placed.config["cute_pdl_release"] == placement
        assert bound.to_code(off_placed) == plain
    on = spec.normalized_config(
        helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_pdl_release="none")
    )
    assert on.config["cute_pdl_release"] == "none"
    # Knob on without a placement: the default (``auto``) is implicit and
    # resolves by the grid; 64 CTAs fit in one wave, so it is ``entry``
    # here, and an explicit ``auto`` renders the same.
    implicit = spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl=True))
    assert "cute_pdl_release" not in implicit.config
    assert bound.to_code(implicit) == _release_code(bound, _COLUMN_SUM, "entry")
    assert _release_code(bound, _COLUMN_SUM, "auto") == bound.to_code(implicit)
    with pytest.raises(exc.InvalidConfig, match="cute_pdl_release must be one of"):
        spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_pdl_release="late")
        )
    # ``auto`` is the only searched value: the placement follows the grid,
    # and a kernel whose calls are separated by a plain launch cannot observe
    # its own placement in isolation (a searched placement is a coin flip
    # there, paid by the kernel's dependent in the real call sequence).  The
    # explicit values stay valid for pins and seeds.
    assert spec._flat_fields()["cute_pdl_release"].choices == (
        "auto",
        "entry",
        "none",
        "exit",
        "last_wave",
    )
    assert spec._flat_fields()["cute_pdl_release"].search_choices == ("auto",)
    # The nested-row seeds turn the knob on and leave the placement to its
    # grid-aware default.
    from helion._compiler.autotuner_heuristics.cute import _dependent_launch

    assert _dependent_launch(spec) == {"cute_pdl": True}
    # Pre-Hopper targets search neither the knob nor the placement.
    below = _cpu_bind(col_reduce_sum_static, (x,)).config_spec
    with patch.object(below, "target_device_capability", (8, 0)):
        assert below._flat_fields()["cute_pdl_release"].search_choices == ("auto",)
        assert _dependent_launch(below) == {}


def test_the_flat_round_trip_keeps_the_grid_rule(cpu_only: None) -> None:
    # The search canonicalizes every compiler seed and pinned finalist
    # through flatten -> unflatten, which fills every field of the flat
    # configuration.  A config with the knob on and no placement (the seeds,
    # every stored record) must render the same code afterwards: the flat
    # default is ``auto`` and the grid rule is applied at codegen, so a
    # multi-wave kernel is not pinned with the entry release it never had.
    from helion.autotuner.config_generation import ConfigGeneration

    x = torch.empty(1000, 1008, dtype=torch.bfloat16)
    w = torch.empty(1008, dtype=torch.bfloat16)
    bound = _cpu_bind(scale_rows, (x, w))
    spec = bound.config_spec
    generation = ConfigGeneration(spec)
    pointwise = {"block_sizes": [16, 64], "num_threads": [4, 64], "cute_pdl": True}
    for hint, guard in ((0, None), (4, "if pdl_cta >= 416:")):
        config = dict(pointwise)
        if hint:
            config["cute_min_blocks_per_mp"] = hint
        seed = spec.normalized_config(helion.Config(**config))
        assert "cute_pdl_release" not in seed.config
        code = bound.to_code(seed)
        assert _last_wave_guard(code) == guard
        _flat, canonical = generation.canonicalize_flat(generation.flatten(seed))
        assert canonical.config["cute_pdl_release"] == "auto"
        assert bound.to_code(canonical) == code
        # An explicit placement survives the round trip as given.
        for placement in ("entry", "none", "exit", "last_wave"):
            placed = spec.normalized_config(
                helion.Config(**config, cute_pdl_release=placement)
            )
            _flat, canonical = generation.canonicalize_flat(generation.flatten(placed))
            assert canonical.config["cute_pdl_release"] == placement
            assert bound.to_code(canonical) == bound.to_code(placed)


def test_the_implicit_release_follows_the_grid(cpu_only: None) -> None:
    # 1024 x 1024 under blocks [1024, 16]: a static grid of 64 CTAs of 512
    # threads, four of which fit an SM's 2048 thread slots.  With the
    # placement unnamed the kernel releases at entry while the grid fits in
    # one wave of the machine (SM count times the CTAs an SM holds), and
    # from the CTAs of its last wave once it needs more; an explicit
    # placement is taken as given.
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))
    spec = bound.config_spec
    implicit = spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl=True))
    entry = _release_code(bound, _COLUMN_SUM, "entry")
    none = _release_code(bound, _COLUMN_SUM, "none")
    assert _launch(entry) == ("(64,)", 512)
    with patch.object(spec, "num_sm", 16):  # 64 slots
        assert bound.to_code(implicit) == entry
    with patch.object(spec, "num_sm", 15):  # 60 slots
        code = bound.to_code(implicit)
        assert code == _release_code(bound, _COLUMN_SUM, "last_wave")
        assert _last_wave_guard(code) == "if pdl_cta >= 4:"
        assert _release_code(bound, _COLUMN_SUM, "entry") == entry
        assert _release_code(bound, _COLUMN_SUM, "none") == none
        # ``cute_min_blocks_per_mp`` names the occupancy the config asks
        # ptxas for; it cannot raise the estimate above the thread limit.
        eight_per_sm = spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_min_blocks_per_mp=8)
        )
        assert _last_wave_guard(bound.to_code(eight_per_sm)) == "if pdl_cta >= 4:"
    with patch.object(spec, "num_sm", 32):  # 128 slots by threads, 32 at 1 CTA/SM
        assert bound.to_code(implicit) == entry
        one_per_sm = spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_min_blocks_per_mp=1)
        )
        assert _last_wave_guard(bound.to_code(one_per_sm)) == "if pdl_cta >= 32:"
    # An unknown SM count (CPU targets) keeps the default.
    with patch.object(spec, "num_sm", 1):
        assert bound.to_code(implicit) == entry


def test_the_resident_capacity_follows_the_launch_block() -> None:
    # The CTAs an SM holds: the thread limit (2048 threads, at most 32 CTAs)
    # unless the config asked ptxas for an occupancy, which is then the
    # estimate, bounded by the thread limit; without a launch shape only the
    # hint, or one CTA, is known.
    assert resident_ctas_per_sm(None, 0) == 1
    assert resident_ctas_per_sm(None, 4) == 4
    assert resident_ctas_per_sm(256, 0) == 8
    assert resident_ctas_per_sm(1024, 0) == 2
    assert resident_ctas_per_sm(32, 0) == 32
    assert resident_ctas_per_sm(512, 8) == 4
    assert resident_ctas_per_sm(512, 2) == 2
    assert resident_ctas_per_sm(256, 0, max_threads_per_sm=1536) == 6


def _implicit_code(bound: Any, **config: object) -> str:
    return bound.to_code(
        bound.config_spec.normalized_config(helion.Config(**config, cute_pdl=True))
    )


def test_the_implicit_release_counts_the_launch_grid(cpu_only: None) -> None:
    # The rule sizes the grid the launcher receives, not the program ids of
    # one strategy: 148 SMs, 256-thread CTAs, 8 per SM, 1184 slots.
    x = torch.empty(1000, 1008, dtype=torch.bfloat16)
    w = torch.empty(1008, dtype=torch.bfloat16)
    pointwise = {"block_sizes": [16, 64], "num_threads": [4, 64]}
    # Two top-level loops launch the sum of their grids: 2 x 1008 CTAs need
    # two waves; 2 x 256 fit one.
    bound = _cpu_bind(two_roots, (x, x.clone()))
    code = _implicit_code(
        bound, block_sizes=[16, 64, 16, 64], num_threads=[4, 64, 4, 64]
    )
    assert _launch(code) == ("(2016,)", 256)
    assert code.count(_WAIT) == 1 and code.count(_RELEASE) == 1
    assert _last_wave_guard(code) == "if pdl_cta >= 832:"
    code = _implicit_code(
        bound, block_sizes=[64, 64, 64, 64], num_threads=[4, 64, 4, 64]
    )
    assert _launch(code) == ("(512,)", 256)
    assert _last_wave_guard(code) is None
    # L2 grouping reorders the parent grid without resizing it: 1008 CTAs
    # fit at 8 per SM, not at the 4 the hint asks for.
    bound = _cpu_bind(scale_rows, (x, w))
    code = _implicit_code(bound, **pointwise, l2_groupings=[4])
    assert _launch(code) == ("(1008,)", 256)
    assert _last_wave_guard(code) is None
    code = _implicit_code(
        bound, **pointwise, l2_groupings=[4], cute_min_blocks_per_mp=4
    )
    assert _last_wave_guard(code) == "if pdl_cta >= 416:"
    # A persistent launch is ``_NUM_SM * num_sm_multiplier`` CTAs however
    # many virtual tiles it walks: one wave at a multiplier the occupancy
    # covers, more past it.
    code = _implicit_code(bound, **pointwise, pid_type="persistent_blocked")
    assert _launch(code) == ("(_NUM_SM,)", 256)
    assert _last_wave_guard(code) is None
    code = _implicit_code(
        bound, **pointwise, pid_type="persistent_interleaved", num_sm_multiplier=8
    )
    assert _launch(code) == ("(_NUM_SM * 8,)", 256)
    assert _last_wave_guard(code) is None
    code = _implicit_code(
        bound, **pointwise, pid_type="persistent_blocked", num_sm_multiplier=16
    )
    assert _launch(code) == ("(_NUM_SM * 16,)", 256)
    assert _last_wave_guard(code) == "if pdl_cta >= 1184:"
    # A cluster launch multiplies the grid by the cluster size: 4 rows on a
    # (1, 2, 1) cluster are 8 CTAs.
    bound, _arguments = _bind_rms_norm(
        torch.empty((4, 8192), dtype=torch.bfloat16), static_shapes=True
    )
    spec = bound.config_spec
    rolled = _rolled_config(bound, threads=128, vec=8, chunk=2048, reload="gmem")
    clustered = spec.normalized_config(
        helion.Config.from_dict(
            {
                **rolled.config,
                "cute_cluster_n": 2,
                "cute_pdl": True,
                "cute_min_blocks_per_mp": 1,
            }
        )
    )
    for num_sm, guard in ((8, None), (7, "if pdl_cta >= 1:")):
        with patch.object(spec, "num_sm", num_sm):
            code = bound.to_code(clustered)
        assert "_helion_cute_cluster_shape = (1, 2, 1)" in code
        assert _launch(code) == ("(*(4,), 2)", 128)
        assert _last_wave_guard(code) == guard, num_sm


_CTA_INDEX = (
    "pdl_cta = cutlass.Int32(cute.arch.block_idx()[0]) + "
    "cutlass.Int32(cute.arch.grid_dim()[0]) * "
    "(cutlass.Int32(cute.arch.block_idx()[1]) + "
    "cutlass.Int32(cute.arch.grid_dim()[1]) * cutlass.Int32(cute.arch.block_idx()[2]))"
)
_GRID_CTAS = (
    "cutlass.Int32(cute.arch.grid_dim()[0]) * cutlass.Int32(cute.arch.grid_dim()[1])"
    " * cutlass.Int32(cute.arch.grid_dim()[2])"
)


def _last_wave_guard(code: str) -> str | None:
    """The condition of the guard ``last_wave`` adds after the wait when the
    grid needs more than one wave (preceded by the CTA's dispatch-order
    index; the release is the guard's only statement, the else branch is
    empty); ``None`` when the kernel releases at plain entry (the release is
    the statement right after the wait)."""
    statements = [ast.unparse(s) for s in _kernel_def(code).body]
    wait = statements.index(_WAIT)
    if statements[wait + 1] == _RELEASE:
        return None
    assert statements[wait + 1] == _CTA_INDEX
    condition, body, orelse, nothing = statements[wait + 2].split("\n")
    assert (body, orelse, nothing) == (f"    {_RELEASE}", "else:", "    pass")
    return condition


def _without_last_wave(code: str) -> str:
    """``code`` with the index and the guard removed, unparsed (comments
    dropped) for comparison with the ``none`` placement's code."""
    module = ast.parse(code)
    kernel = _kernel_def(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == kernel.name
    ]
    wait = [ast.unparse(s) for s in kernel.body].index(_WAIT)
    del kernel.body[wait + 1 : wait + 3]
    return ast.unparse(module)


def test_the_last_wave_release_is_plain_entry_for_a_one_wave_grid(
    cpu_only: None,
) -> None:
    # 64 CTAs of 512 threads on 148 SMs (592 slots): one wave, so the last
    # wave is the whole grid and the placement renders the entry code.
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))
    assert _release_code(bound, _COLUMN_SUM, "last_wave") == _release_code(
        bound, _COLUMN_SUM, "entry"
    )


def test_the_last_wave_release_guards_the_entry_release_past_one_wave(
    cpu_only: None,
) -> None:
    # 64 CTAs against 60 slots (15 SMs x 4 CTAs of 512 threads): the CTAs
    # whose dispatch-order index is >= 64 - 60 release at entry, the first
    # four count as released when they exit, so the dependents launch as
    # CTA 4, the first of the last wave, starts.  Nothing else changes: the
    # code without the index and the guarded release is the ``none`` code.
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))
    spec = bound.config_spec
    with patch.object(spec, "num_sm", 15):
        code = _release_code(bound, _COLUMN_SUM, "last_wave")
        none = _release_code(bound, _COLUMN_SUM, "none")
    assert _launch(code) == ("(64,)", 512)
    assert code.count(_RELEASE) == 1
    assert _last_wave_guard(code) == "if pdl_cta >= 4:"
    assert _without_last_wave(code) == ast.unparse(ast.parse(none))
    # ``auto`` resolves to the same placement past one wave.
    with patch.object(spec, "num_sm", 15):
        implicit = bound.to_code(
            spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl=True))
        )
    assert implicit == code


def test_the_last_wave_release_sizes_a_dynamic_grid_at_runtime(
    cpu_only: None,
) -> None:
    # The grid of a dynamic column count is the host's: the guard compares
    # the CTA's index against ``grid_dim``'s product less the resident
    # capacity (148 SMs x 4 CTAs of 512 threads = 592), a compile-time
    # constant.
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_dynamic, (x,))
    code = _release_code(bound, _COLUMN_SUM, "last_wave")
    assert _launch(code)[0] == "((n + _BLOCK_SIZE_1 - 1) // _BLOCK_SIZE_1,)"
    assert _last_wave_guard(code) == f"if pdl_cta >= {_GRID_CTAS} - 592:"
    none = _release_code(bound, _COLUMN_SUM, "none")
    assert _without_last_wave(code) == ast.unparse(ast.parse(none))
    # ``cute_min_blocks_per_mp`` names the occupancy the config asked for and
    # is the capacity estimate then (148 x 1).
    spec = bound.config_spec
    one_per_sm = bound.to_code(
        spec.normalized_config(
            helion.Config(
                **_COLUMN_SUM,
                cute_pdl=True,
                cute_pdl_release="last_wave",
                cute_min_blocks_per_mp=1,
            )
        )
    )
    assert _last_wave_guard(one_per_sm) == f"if pdl_cta >= {_GRID_CTAS} - 148:"


def _last_wave_code(bound: Any, **config: object) -> str:
    return bound.to_code(
        bound.config_spec.normalized_config(
            helion.Config(**config, cute_pdl=True, cute_pdl_release="last_wave")
        )
    )


def test_the_last_wave_release_counts_the_launch_grid(cpu_only: None) -> None:
    # The same launch count as the grid rule (148 SMs, 256-thread CTAs, 8
    # per SM, 1184 slots): two roots of 1008 CTAs each are 2016 CTAs, the
    # last 1184 of which release; 2 x 256 fit one wave and release at plain
    # entry.
    x = torch.empty(1000, 1008, dtype=torch.bfloat16)
    w = torch.empty(1008, dtype=torch.bfloat16)
    bound = _cpu_bind(two_roots, (x, x.clone()))
    code = _last_wave_code(
        bound, block_sizes=[16, 64, 16, 64], num_threads=[4, 64, 4, 64]
    )
    assert _launch(code) == ("(2016,)", 256)
    assert _last_wave_guard(code) == "if pdl_cta >= 832:"
    code = _last_wave_code(
        bound, block_sizes=[64, 64, 64, 64], num_threads=[4, 64, 4, 64]
    )
    assert _launch(code) == ("(512,)", 256)
    assert _last_wave_guard(code) is None
    # A persistent launch is ``_NUM_SM * num_sm_multiplier`` CTAs: one wave
    # at a multiplier the occupancy covers (plain entry), guarded past it
    # (16 x 148 = 2368 CTAs, the last 1184 release).
    bound = _cpu_bind(scale_rows, (x, w))
    pointwise = {"block_sizes": [16, 64], "num_threads": [4, 64]}
    code = _last_wave_code(bound, **pointwise, pid_type="persistent_blocked")
    assert _launch(code) == ("(_NUM_SM,)", 256)
    assert _last_wave_guard(code) is None
    code = _last_wave_code(
        bound, **pointwise, pid_type="persistent_blocked", num_sm_multiplier=16
    )
    assert _launch(code) == ("(_NUM_SM * 16,)", 256)
    assert _last_wave_guard(code) == "if pdl_cta >= 1184:"
    # An unknown SM count (CPU targets) has no capacity to guard against:
    # plain entry, like ``auto``.
    spec = bound.config_spec
    with patch.object(spec, "num_sm", 1):
        code = _last_wave_code(
            bound, **pointwise, pid_type="persistent_blocked", num_sm_multiplier=16
        )
    assert _last_wave_guard(code) is None
