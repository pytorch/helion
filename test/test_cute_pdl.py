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
the GPU semantics are checked at the end of this module.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

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
from helion._compiler.cute.host_zero_fill_launches import MAX_FILL_NUMEL
from helion._compiler.cute.pdl_release import ZERO_FILL_KERNEL_PREFIX
from helion._compiler.cute.pdl_release import resident_ctas_per_sm
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import onlyBackends
from helion._testing import patch_target_facts
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_pdl import CONTROL_DELAY as _CONTROL_DELAY
from helion._testing.cute_pdl import PRODUCER_BLOCK as _BLOCK
from helion._testing.cute_pdl import PRODUCER_GRID as _PRODUCER_GRID
from helion._testing.cute_pdl import WAIT_CALL as _WAIT
from helion._testing.cute_pdl import attribute_only_control as _attribute_only_control
from helion._testing.cute_pdl import kernel_definition as _kernel_def
from helion._testing.cute_pdl import produce as _produce
from helion._testing.cute_pdl import triton_kernels as _tk
from helion.autotuner.benchmarking import _make_cudagraph_replay
import helion.language as hl

pytestmark = pytest.mark.cpu_only

rms_norm_batched = import_example("aot_example").rms_norm_batched

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
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
    """The kernel's launcher call's grid expression and its CTA's thread
    count (not the launch of a zero-fill kernel, ``_fill_kernels``)."""
    (call,) = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
        and not (
            isinstance(node.args[0], ast.Name)
            and node.args[0].id.startswith(ZERO_FILL_KERNEL_PREFIX)
        )
    ]
    (block,) = [keyword.value for keyword in call.keywords if keyword.arg == "block"]
    assert isinstance(block, ast.Tuple)
    threads = 1
    for element in block.elts:
        assert isinstance(element, ast.Constant)
        threads *= int(element.value)
    return ast.unparse(call.args[1]), threads


def _fill_kernels(code: str) -> list[ast.FunctionDef]:
    """The zero-fill kernels a ``cute_pdl`` kernel's host zero fills launch
    as (``host_zero_fill_launches``): one per surviving ``torch.zeros``."""
    return [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith(ZERO_FILL_KERNEL_PREFIX)
    ]


def _on_and_off(bound: Any, config: dict[str, object]) -> tuple[str, str]:
    spec = bound.config_spec
    base = {key: value for key, value in config.items() if key != "cute_pdl"}
    on = bound.to_code(
        spec.normalized_config(helion.Config.from_dict(base | {"cute_pdl": True}))
    )
    off = bound.to_code(spec.normalized_config(helion.Config.from_dict(base)))
    return on, off


def _codes(kernel: helion.Kernel, args: tuple, **config: object) -> tuple[str, str]:
    return _on_and_off(cpu_bind(kernel, args), config)


def _release_block(kernel: ast.FunctionDef, start: int) -> int:
    """The number of statements of the entry release at ``start``: the plain
    release, or ``last_wave``'s CTA index and guard; 0 when none is there."""
    statements = [ast.unparse(s) for s in kernel.body[start : start + 2]]
    if statements[:1] == [_RELEASE]:
        return 1
    if len(statements) == 2 and statements[0] == _CTA_INDEX:
        return 2
    return 0


def _late_wait_names(code: str) -> set[str]:
    """The host names of the zero fills launched as fill kernels: behind
    them a kernel's wait moves down to the first touch of a filled tensor
    (``passes.pdl.place_late_wait``)."""
    fills = {fill.name for fill in _fill_kernels(code)}
    names: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        # ``_launcher(fill, (grid,), out.view(-1), block=...)``
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and len(node.value.args) >= 3
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id in fills
        ):
            continue
        view = node.value.args[2]
        if (
            isinstance(view, ast.Call)
            and isinstance(view.func, ast.Attribute)
            and isinstance(view.func.value, ast.Name)
        ):
            names.add(view.func.value.id)
    return names


def _statements_ahead_of_the_wait(
    kernel: ast.FunctionDef, filled: set[str] = frozenset()
) -> list[str]:
    """The wait is a top-level statement of the kernel body and nothing ahead
    of it touches memory, at any nesting depth: no tensor indexing and no
    call that loads, stores, copies, syncs a barrier or speaks
    ``griddepcontrol``.  Returns what does precede it.

    A kernel whose zero fills launch as fill kernels (``filled`` names them)
    may wait late, below a memory-touching loop, right before the first
    touch of a filled tensor; its entry release then leads the body.  Such
    a kernel's statements ahead of the wait may touch memory but never a
    filled tensor; the release block is left out of the returned list."""
    waits = [
        index
        for index, statement in enumerate(kernel.body)
        if ast.unparse(statement) == _WAIT
    ]
    assert len(waits) == 1, ast.unparse(kernel)[:600]
    ahead = kernel.body[: waits[0]]
    if waits[0] > 0 and _release_block(kernel, 0):
        # The late form: the entry release leads, a memory-touching loop
        # runs ahead of the wait and no statement ahead touches a filled
        # tensor.
        ahead = ahead[_release_block(kernel, 0) :]
        for statement in ahead:
            assert not any(
                isinstance(node, ast.Name) and node.id in filled
                for node in ast.walk(statement)
            ), ast.unparse(statement)
        assert any(isinstance(statement, (ast.For, ast.While)) for statement in ahead)
        return [ast.unparse(statement) for statement in ahead]
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
    ahead = _statements_ahead_of_the_wait(kernel, _late_wait_names(on))
    main = ast.unparse(kernel)
    assert main.count(_WAIT) == 1
    attributes = [
        ast.unparse(node)
        for node in ast.parse(on).body
        if isinstance(node, ast.Assign) and ast.unparse(node).endswith(_ATTRIBUTE)
    ]
    assert attributes.count(f"{kernel.name}{_ATTRIBUTE}") == 1
    # The default placement (``auto``) releases the dependents right after
    # the wait while the grid fits in one wave (``_implicit_pdl_release``);
    # a grid that needs more waves, or one sized at runtime, releases from
    # the CTAs of its last wave (``_last_wave_guard``).
    assert main.count(_RELEASE) == 1
    _last_wave_guard(on)
    # A host zero fill the kernel keeps launches as a fill kernel of the
    # module, itself waiting and releasing at entry, ahead of the kernel.
    for fill in _fill_kernels(on):
        assert [ast.unparse(s) for s in fill.body[:2]] == [_WAIT, _RELEASE]
        assert attributes.count(f"{fill.name}{_ATTRIBUTE}") == 1
    assert _WAIT not in off and _ATTRIBUTE not in off and _RELEASE not in off
    assert ZERO_FILL_KERNEL_PREFIX not in off
    assert _without_dependent_launch(on) == ast.unparse(ast.parse(off))
    return ahead


_ZEROS_OF_EMPTY = {
    "empty": "zeros",
    "empty_like": "zeros_like",
    "new_empty": "new_zeros",
}


def _is_fill_launch(statement: ast.stmt, fills: set[str]) -> bool:
    """Whether ``statement`` launches one of the ``fills`` kernels: the plain
    launch, or, for a tensor the host sizes at runtime, the launch behind the
    host bound on the element count with the memset (``zero_()``) past it
    (the fill kernel counts in Int32: ``MAX_FILL_NUMEL``)."""
    if isinstance(statement, ast.If):
        if len(statement.body) != 1 or not _is_fill_launch(statement.body[0], fills):
            return False
        assert ast.unparse(statement.test).endswith(f".numel() < {MAX_FILL_NUMEL}")
        assert [ast.unparse(s) for s in statement.orelse] == [
            f"{ast.unparse(statement.test).split('.numel()')[0]}.zero_()"
        ]
        return True
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.args[0], ast.Name)
        and statement.value.args[0].id in fills
    )


def _without_dependent_launch(code: str) -> str:
    """``code`` without the launch attribute, the wait and the release (the
    plain one or the index and guard of ``last_wave``), and without the
    zero-fill kernels (their definitions, attributes and launches; the
    ``empty`` each launch follows reads ``zeros`` again), unparsed (comments
    dropped) for comparison with the knob-off code."""
    module = ast.parse(code)
    name = _kernel_def(code).name
    fills = {fill.name for fill in _fill_kernels(code)}
    attributes = {f"{owner}{_ATTRIBUTE}" for owner in (name, *fills)}
    module.body = [
        node
        for node in module.body
        if not (isinstance(node, ast.Assign) and ast.unparse(node) in attributes)
        and not (isinstance(node, ast.FunctionDef) and node.name in fills)
    ]
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    statements = [ast.unparse(s) for s in kernel.body]
    wait = statements.index(_WAIT)
    if wait > 0 and _release_block(kernel, 0):
        # The late wait: the release block leads the body, the wait follows
        # the loop ahead of the first touch of a filled tensor.
        del kernel.body[wait]
        del kernel.body[: _release_block(kernel, 0)]
    else:
        del kernel.body[wait : wait + 1 + _release_block(kernel, wait + 1)]
    if fills:
        (host,) = [
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name != name
        ]
        body: list[ast.stmt] = []
        for statement in host.body:
            if not _is_fill_launch(statement, fills):
                body.append(statement)
                continue
            allocation = body[-1]
            assert isinstance(allocation, ast.Assign)
            factory = allocation.value
            assert isinstance(factory, ast.Call) and isinstance(
                factory.func, ast.Attribute
            )
            factory.func.attr = _ZEROS_OF_EMPTY[factory.func.attr]
        host.body = body
    return ast.unparse(module)


def test_the_column_sum_waits_before_its_first_global_access(cpu_only: None) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    on, off = _codes(col_reduce_sum_static, (x,), **_COLUMN_SUM)
    ahead = _assert_only_the_dependent_launch_differs(on, off)
    # With a fill kernel ahead of it the kernel waits late (its row loop runs
    # ahead of the wait, which precedes the atomic into ``out``); without one
    # it waits first.
    assert (ahead != []) == bool(_fill_kernels(on))
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
    ahead = _assert_only_the_dependent_launch_differs(on, off)
    assert (ahead != []) == bool(_fill_kernels(on))
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
    bound = cpu_bind(kernel, args)
    on, off = _on_and_off(bound, dict(bound.config_spec.default_config().config))
    ahead = _assert_only_the_dependent_launch_differs(on, off)
    # A zero-filled accumulator launches as a fill kernel and the wait moves
    # below the reduction loop, right before the atomic into the filled
    # tensor; a kernel without one waits first.
    assert (ahead != []) == bool(_fill_kernels(on))


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
    bound = cpu_bind(_dynamic_kernel(fn, static_shapes=static_shapes), args)
    config = _rolled_config(bound, rows=4, threads=64, vec=8, chunk=1024)
    on, off = _on_and_off(bound, dict(config.config))
    assert _assert_only_the_dependent_launch_differs(on, off) == [
        "_fuse_cache_0 = cute.make_rmem_tensor(64, cutlass.Uint16)"
    ]


def test_the_knob_is_a_searchable_boolean(cpu_only: None) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = cpu_bind(col_reduce_sum_static, (x,))
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
    bound = cpu_bind(col_reduce_sum_static, (x,))
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
    bound = cpu_bind(col_reduce_sum_static, (x,))
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
    bound = cpu_bind(col_reduce_sum_static, (x,))
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
        "loader_tail",
    )
    assert spec._flat_fields()["cute_pdl_release"].search_choices == ("auto",)
    # The nested-row seeds turn the knob on, leave the placement to its
    # grid-aware default and spread the CTAs to one wave's density.
    from helion._compiler.cute.autotune.heuristics.cute import _dependent_launch

    assert _dependent_launch(spec) == {"cute_pdl": True, "cute_pdl_spread": True}
    # Pre-Hopper targets search neither the knob nor the placement.
    below = cpu_bind(col_reduce_sum_static, (x,)).config_spec
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
    bound = cpu_bind(scale_rows, (x, w))
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
    bound = cpu_bind(col_reduce_sum_static, (x,))
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
    bound = cpu_bind(two_roots, (x, x.clone()))
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
    bound = cpu_bind(scale_rows, (x, w))
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
    kernel = _kernel_def(code)
    statements = [ast.unparse(s) for s in kernel.body]
    wait = statements.index(_WAIT)
    # The release block follows the wait, or leads the body ahead of a late
    # wait (a kernel whose zero fills launch as fill kernels).
    start = 0 if wait > 0 and _release_block(kernel, 0) else wait + 1
    if statements[start] == _RELEASE:
        return None
    assert statements[start] == _CTA_INDEX
    condition, body, orelse, nothing = statements[start + 1].split("\n")
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
    bound = cpu_bind(col_reduce_sum_static, (x,))
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
    bound = cpu_bind(col_reduce_sum_static, (x,))
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
    bound = cpu_bind(col_reduce_sum_dynamic, (x,))
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
    bound = cpu_bind(two_roots, (x, x.clone()))
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
    bound = cpu_bind(scale_rows, (x, w))
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


_PAD = "pdl_spread_pad = cute.arch.alloc_smem(cutlass.Uint8, {}, alignment=128)"
_KEEP = "_helion_pdl_spread_keep(pdl_spread_pad.toint())"
_KEEP_IMPORT = (
    "from helion._compiler.cute._pdl_spread_runtime import "
    "keep_smem_live as _helion_pdl_spread_keep"
)


def _without_spread_pad(code: str) -> str:
    """``code`` without the pad of ``cute_pdl_spread`` and its import."""
    module = ast.parse(code)
    module.body = [
        node
        for node in module.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "helion._compiler.cute._pdl_spread_runtime"
        )
    ]
    kernel = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
    )
    kernel.body = [
        statement
        for statement in kernel.body
        if "pdl_spread_pad" not in ast.unparse(statement)
    ]
    return ast.unparse(module)


def test_the_spread_pad_sizes_the_kernel_to_one_wave() -> None:
    from helion._compiler.cute.pdl_release import DEFAULT_SMEM_PER_SM
    from helion._compiler.cute.pdl_release import SMEM_RESERVED_PER_CTA
    from helion._compiler.cute.pdl_release import spread_pad_bytes

    # 128 CTAs on 148 SMs run one per SM: the pad raises a 25600-byte body to
    # the smallest footprint at which two CTAs, each with the system's
    # reserved slice, no longer fit an SM while one still does.
    pad = spread_pad_bytes(128, 148, 25600, 25600)
    assert pad is not None
    footprint = 25600 + pad
    assert footprint == 115713
    assert 2 * (footprint + SMEM_RESERVED_PER_CTA) > DEFAULT_SMEM_PER_SM
    assert footprint + SMEM_RESERVED_PER_CTA <= DEFAULT_SMEM_PER_SM
    # Two per SM (296 CTAs): three no longer fit, two still do.
    pad = spread_pad_bytes(296, 148, 0, 0)
    assert pad is not None
    assert 3 * (pad + SMEM_RESERVED_PER_CTA) > DEFAULT_SMEM_PER_SM
    assert 2 * (pad + SMEM_RESERVED_PER_CTA) <= DEFAULT_SMEM_PER_SM
    # A body the compiler may keep anywhere between 25600 and 65536 bytes of:
    # at one CTA per SM the pad tops up the least it keeps, so the kernel
    # reaches the footprint whatever is dropped (more costs no CTA up to the
    # per-block maximum, which the most must stay within); at two per SM so
    # large a pad would leave room for one CTA when everything is kept, and
    # it tops up the most instead.
    assert spread_pad_bytes(128, 148, 25600, 65536) == 115713 - 25600
    assert spread_pad_bytes(128, 148, 0, 131072) is None
    assert spread_pad_bytes(296, 148, 0, 25600) == 76801 - 25600
    # A body whose own allocation already holds the CTAs to that density
    # needs no pad (the 164 KB output kernel of the linear attention
    # forward), nor does a kernel dense enough that the footprint drops
    # below what it allocates.
    assert spread_pad_bytes(128, 148, 167936, 167936) is None
    assert spread_pad_bytes(148 * 16, 148, 16384, 16384) is None
    # No grid, no SMs, no ordered bounds, or a footprint past the per-block
    # maximum: no pad.
    assert spread_pad_bytes(0, 148, 0, 0) is None
    assert spread_pad_bytes(128, 0, 0, 0) is None
    assert spread_pad_bytes(128, 148, 1024, 0) is None
    assert spread_pad_bytes(128, 148, 0, 0, max_smem_per_block=65536) is None
    assert spread_pad_bytes(128, 148, 0, 0, smem_per_sm=2000) is None
    # Every resident CTA keeps one 1 KB unit of slack beyond its footprint
    # (rounded up to the 128-byte allocation unit), so the alignment padding
    # of the body's allocations or a helper's own allocation cannot cost a
    # resident CTA: 14 per SM still fit with that slack on a 228 KiB SM, 15
    # do not, and the pad is declined there rather than sized to the edge.
    from helion._compiler.cute.pdl_release import SMEM_ALLOCATION_UNIT
    from helion._compiler.cute.pdl_release import SPREAD_SLACK_PER_CTA

    for cap in range(1, 15):
        pad = spread_pad_bytes(148 * cap, 148, 0, 0)
        assert pad is not None, cap
        allocated = -(-pad // SMEM_ALLOCATION_UNIT) * SMEM_ALLOCATION_UNIT
        per_cta = allocated + SMEM_RESERVED_PER_CTA
        assert (cap + 1) * per_cta > DEFAULT_SMEM_PER_SM
        assert cap * (per_cta + SPREAD_SLACK_PER_CTA) <= DEFAULT_SMEM_PER_SM
    assert spread_pad_bytes(148 * 15, 148, 0, 0) is None
    assert spread_pad_bytes(148 * 32, 148, 0, 0) is None


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_max_plus_sum(x: torch.Tensor) -> torch.Tensor:
    """Two block reduces per row block (``torch.amax`` and ``torch.sum``)."""
    n, v = x.size()
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        row_max = hl.full([tile_n], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_n], dtype=torch.float32)
        for tile_v in hl.tile(v):
            vals = x[tile_n, tile_v].to(torch.float32)
            row_max = torch.maximum(row_max, torch.amax(vals, dim=-1))
            row_sum = row_sum + torch.sum(vals, dim=-1)
        out[tile_n] = row_max + row_sum
    return out


def test_the_spread_pad_counts_the_reduce_buffers_of_every_lane(
    cpu_only: None,
) -> None:
    # One CTA of 256 x 4 threads over 4096 rows: the 4 threads of a row
    # combine its 1024 columns with a shared-memory reduce, and each of the
    # V unrolled row lanes (``cute_vector_widths[0]``) traces its own copy
    # of both reduces, a 5120-byte buffer each (the interleaved block form,
    # ``pre=256, group_span=1024``).
    x = torch.empty(4096, 1024, dtype=torch.bfloat16)
    bound = cpu_bind(row_max_plus_sum, (x,))
    spec = bound.config_spec

    def render(width: int, spread: bool) -> str:
        return bound.to_code(
            spec.normalized_config(
                helion.Config(
                    block_sizes=[4096, 1024],
                    num_threads=[256, 4],
                    cute_vector_widths=[width, 1],
                    cute_pdl=True,
                    cute_pdl_spread=spread,
                )
            )
        )

    # Four lanes hold eight buffers: the pad raises their 40960 bytes to the
    # one-CTA-per-SM footprint.
    code = render(4, True)
    assert code.count("_cute_grouped_reduce_shared_two_stage(") == 2
    assert _PAD.format(115713 - 8 * 5120) in code
    # Sixteen lanes hold 32 buffers, 163840 bytes, which already keep one
    # CTA per SM: no pad.  The pad sized against the body's literal
    # allocations alone (115713 bytes) took this kernel to 279553 bytes,
    # past the 232448-byte per-block maximum, and libNVVM failed.
    assert render(16, True) == render(16, False)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def two_row_sums(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two roots, each a block reduce of its rows."""
    m, n = x.size()
    out_x = torch.empty([m], dtype=torch.float32, device=x.device)
    out_y = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        out_x[tile_m] = torch.sum(x[tile_m, :].to(torch.float32), dim=-1)
    for tile_m in hl.tile(m):
        out_y[tile_m] = torch.sum(y[tile_m, :].to(torch.float32), dim=-1)
    return out_x, out_y


def test_the_spread_pad_counts_the_buffers_of_every_root(cpu_only: None) -> None:
    # 2048 CTAs of 1024 threads on 148 SMs (fourteen per SM): a 14541-byte
    # footprint.  The launch dispatches the two roots on the block index
    # (``if pid_shared < 1024``), so the DSL traces both and the pad leaves
    # room for both roots' 132-byte reduce buffers (33 fp32 slots of the
    # two-stage form over 32 warps).
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = cpu_bind(two_row_sums, (x, x.clone()))
    spec = bound.config_spec
    base = {
        key: value
        for key, value in spec.default_config().config.items()
        if key != "cute_pdl"
    }
    code = bound.to_code(
        spec.normalized_config(
            helion.Config(**base, cute_pdl=True, cute_pdl_spread=True)
        )
    )
    assert code.count("_cute_grouped_reduce_shared_two_stage(") == 2
    assert _PAD.format(14541 - 2 * 132) in code


def test_the_spread_knob_pads_the_kernel_after_its_release(cpu_only: None) -> None:
    # 64 CTAs of 512 threads on 148 SMs: one per SM.  The body's only shared
    # memory is its column reduce's buffer (the fragment two-stage over 16
    # warps, 4 x 4 slots of partials each plus the results: 1088 bytes), so
    # the pad raises it to the 115713-byte footprint, placed right after the
    # wait and the release and kept live by the helper the module imports;
    # everything else is the knob-on code.
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = cpu_bind(col_reduce_sum_static, (x,))
    spec = bound.config_spec
    on = bound.to_code(
        spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl=True))
    )
    spread = bound.to_code(
        spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_pdl_spread=True)
        )
    )
    assert _KEEP_IMPORT in spread and _KEEP_IMPORT not in on
    statements = [ast.unparse(statement) for statement in _kernel_def(spread).body]
    index = statements.index(_WAIT)
    assert statements[index + 1 : index + 4] == [
        _RELEASE,
        _PAD.format(115713 - 1088),
        _KEEP,
    ]
    assert _without_spread_pad(spread) == ast.unparse(ast.parse(on))
    from helion._compiler.cute.pdl_release import spread_rendered

    assert spread_rendered(spread) and not spread_rendered(on)
    # The final selection ranks the padded kernel ahead of the released one.
    from helion._compiler.cute.backend import CuteBackend

    cfg = spec.normalized_config(
        helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_pdl_spread=True)
    )
    assert CuteBackend.finalist_preference(cfg, lambda: spread) == 0
    assert (
        CuteBackend.finalist_preference(
            spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl=True)),
            lambda: on,
        )
        == 1
    )
    # Past one wave the pad follows the ``last_wave`` guard: 15 SMs hold 64
    # CTAs at five per SM, so six must not fit.
    with patch.object(spec, "num_sm", 15):
        code = bound.to_code(cfg)
        statements = [ast.unparse(statement) for statement in _kernel_def(code).body]
        index = statements.index(_WAIT)
        assert _last_wave_guard(code) == "if pdl_cta >= 4:"
        assert statements[index + 3 : index + 5] == [
            _PAD.format(37889 - 1088),
            _KEEP,
        ]
        assert spread_rendered(code)
    # An unknown SM count (CPU targets) or a device whose shared memory
    # cannot hold the footprint leaves the body alone.
    with patch.object(spec, "num_sm", 1):
        assert bound.to_code(cfg) == on
    with patch("helion._compiler.cute.passes.pdl._smem_per_sm", return_value=2000):
        assert bound.to_code(cfg) == on
    # Nor does a block maximum below the footprint (a modelled 99 KiB part):
    # the pad reads the maximum the per-block check enforces.
    with patch_target_facts(smem_capacity_bytes=101376):
        assert bound.to_code(cfg) == on


def test_the_spread_pad_follows_the_launch_grid(cpu_only: None) -> None:
    # 1008 CTAs of 256 threads on 148 SMs: seven per SM, so the footprint is
    # the smallest at which eight no longer fit.
    x = torch.empty(1000, 1008, dtype=torch.bfloat16)
    w = torch.empty(1008, dtype=torch.bfloat16)
    pointwise = {"block_sizes": [16, 64], "num_threads": [4, 64]}
    bound = cpu_bind(scale_rows, (x, w))
    spec = bound.config_spec
    code = bound.to_code(
        spec.normalized_config(
            helion.Config(**pointwise, cute_pdl=True, cute_pdl_spread=True)
        )
    )
    assert _launch(code) == ("(1008,)", 256)
    assert _PAD.format(28161) in code and _KEEP in code
    # A grid the host sizes at runtime gets no pad: its density is unknown.
    bound = cpu_bind(col_reduce_sum_dynamic, (x,))
    spec = bound.config_spec
    on = bound.to_code(
        spec.normalized_config(helion.Config(**_COLUMN_SUM, cute_pdl=True))
    )
    spread = bound.to_code(
        spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_pdl_spread=True)
        )
    )
    assert spread == on


def test_the_spread_knob_rides_on_cute_pdl(cpu_only: None) -> None:
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    bound = cpu_bind(col_reduce_sum_static, (x,))
    spec = bound.config_spec
    assert "cute_pdl_spread" not in spec.default_config().config
    plain = bound.to_code(spec.normalized_config(helion.Config(**_COLUMN_SUM)))
    # Knob off: the default is dropped, a set value stays (inert, so a flat
    # configuration round-trips) and the code is the knob-off code.
    off = spec.normalized_config(
        helion.Config(**_COLUMN_SUM, cute_pdl=False, cute_pdl_spread=False)
    )
    assert "cute_pdl_spread" not in off.config
    assert bound.to_code(off) == plain
    off_spread = spec.normalized_config(
        helion.Config(**_COLUMN_SUM, cute_pdl=False, cute_pdl_spread=True)
    )
    assert off_spread.config["cute_pdl_spread"] is True
    assert bound.to_code(off_spread) == plain
    with pytest.raises(exc.InvalidConfig, match="cute_pdl_spread must be a boolean"):
        spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_pdl_spread="yes")
        )
    # Searched with the knob where ``griddepcontrol`` exists, and kept by the
    # flat round trip the search canonicalizes every seed through.
    assert spec._flat_fields()["cute_pdl_spread"].search_choices == (False, True)
    from helion.autotuner.config_generation import ConfigGeneration

    generation = ConfigGeneration(spec)
    for value in (False, True):
        config = spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_pdl_spread=value)
        )
        round_trip = generation.unflatten(generation.flatten(config))
        assert round_trip.config.get("cute_pdl_spread", False) is value
        assert bound.to_code(round_trip) == bound.to_code(config)
    with patch.object(spec, "target_device_capability", (8, 0)):
        assert spec._flat_fields()["cute_pdl_spread"].search_choices == (False,)


# GPU semantics of ``cute_pdl`` (programmatic dependent launch).
#
# A programmatic dependent launch starts as soon as the kernel ahead of it in
# the stream releases its dependents (``griddepcontrol.launch_dependents``), so
# a producer that releases them at entry and writes late exposes every read a
# dependent issues ahead of its ``griddepcontrol.wait``.  Such a producer (a
# Triton kernel) runs ahead of the knob-on kernels here, which must match the
# knob-off kernels bit for bit, in the stream and inside a CUDA graph, where
# the launch attribute becomes a programmatic edge.  A control that launches
# the knob-on module without its wait has to read stale data: that proves the
# producer really overlaps the launch and the wait is load-bearing.  The
# pure-launch chain at the end shows why a kernel that releases its dependents
# at entry has to stay an ordinary launch (the materialized-fission producer,
# which ``_stage_config`` keeps away from the knob).  The release placements
# (``cute_pdl_release``) are told apart by a dependent that does not wait: it
# reaches the kernel's input while the kernel still reads it only when the
# kernel released at entry.


_COLUMN = {
    "block_sizes": [1024, 16],
    "num_threads": [128, 4],
    "cute_vector_widths": [1, 4],
    "cute_lane_layouts": ["strided", "blocked"],
    "cute_vloop_sink": True,
    "cute_lane_unroll": 8,
}


def _build(
    kernel: helion.Kernel, x: torch.Tensor, **config: object
) -> tuple[helion.runtime.kernel.BoundKernel, Callable[..., torch.Tensor], str]:
    bound = kernel.bind((x,))
    normalized = bound.config_spec.normalized_config(helion.Config(**config))
    return bound, bound.compile_config(normalized), bound.to_code(normalized)


def _stale_runs(
    fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    sources: list[torch.Tensor],
    references: list[torch.Tensor],
    iterations: int,
    *,
    delay: int = 64,
) -> int:
    """Runs producer -> ``fn`` back to back and counts runs whose output
    differs (bitwise) from the reference of the source just produced.

    ``delay`` is the producer's spin count ahead of every store: the
    zero-stale assertions hold at any delay, while the positive control (a
    programmatic launch without the wait) needs a producer that outlives the
    consumer's host launch latency, or the race it demonstrates may never
    be lost."""
    stale = torch.zeros((), dtype=torch.int64, device=DEVICE)
    for i in range(iterations):
        j = i & 1
        _produce(sources[j], x, delay=delay)
        stale += (fn(x) != references[j]).any()
    torch.cuda.synchronize()
    return int(stale)


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestCutePdl(TestCase):
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    def test_explicit_api_chain_eager_and_graph_replay(self) -> None:
        @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
        def entry_release(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            hl.pdl_launch_dependents()
            hl.pdl_wait()
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
        def exit_release(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            hl.pdl_wait()
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            hl.pdl_launch_dependents()
            return out

        x = torch.empty(1 << 18, dtype=torch.float32, device=DEVICE)
        sources = [torch.full_like(x, value) for value in (3, 17)]
        references = [source + 1 for source in sources]
        bound, entry, code = _build(
            entry_release, x, block_sizes=[256], num_threads=128
        )
        _bound, finish, _code = _build(
            exit_release,
            x,
            block_sizes=[256],
            num_threads=128,
            cute_pdl=True,
            cute_pdl_release="entry",
        )
        self.assertEqual(
            _stale_runs(entry, x, sources, references, 8, delay=_CONTROL_DELAY), 0
        )
        standalone_code = bound.to_code(
            helion.Config(block_sizes=[256], num_threads=128),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
        self.assertIn("use_pdl=True", standalone_code)
        standalone = PyCodeCache.load(standalone_code).entry_release
        self.assertEqual(
            _stale_runs(standalone, x, sources, references, 8, delay=_CONTROL_DELAY),
            0,
        )
        # The same programmatic launch without its wait must see stale input:
        # this proves the delayed producer exercises the dependency edge.
        control = _attribute_only_control(bound, code)
        self.assertGreater(
            _stale_runs(control, x, sources, references, 8, delay=_CONTROL_DELAY), 0
        )
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            _produce(sources[0], x, delay=_CONTROL_DELAY)
            finish(standalone(x))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _produce(sources[0], x, delay=_CONTROL_DELAY)
            out = finish(standalone(x))
        for value in (5, 29, -11):
            sources[0].fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out, torch.full_like(out, value + 2)))

    def test_knob_on_matches_knob_off_behind_a_producer_that_releases_at_entry(
        self,
    ) -> None:
        iterations = 100
        # The max kernel writes into ``torch.empty``: nothing sits between
        # the producer and the reduction, so a missing wait reads stale
        # data (the control proves it).  The sum zero-fills its output first;
        # that fill node is what the knob's launch overlaps.
        for kernel, shape, control in (
            (col_reduce_max_dynamic, (2000, 1008), True),
            (col_reduce_sum_static, (1024, 1024), False),
        ):
            x = torch.empty(*shape, dtype=torch.bfloat16, device=DEVICE)
            sources = [
                torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE),
                torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE) * 7,
            ]
            bound, off, off_code = _build(kernel, x, **_COLUMN)
            self.assertNotIn(_WAIT, off_code)
            self.assertNotIn(_ATTRIBUTE, off_code)
            references = []
            for source in sources:
                _produce(source, x)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(x, source))
                references.append(off(x).clone())
            self.assertEqual(_stale_runs(off, x, sources, references, iterations), 0)
            # The knob with each placement of its own release: ``entry``
            # lets the next kernel (here the producer of the next source)
            # launch while this one runs; every placement must still read
            # the source just produced.
            for release in (
                {},
                {"cute_pdl_release": "none"},
                {"cute_pdl_release": "exit"},
                {"cute_pdl_release": "entry"},
                {"cute_pdl_release": "last_wave"},
                {"cute_pdl_spread": True},
            ):
                _bound, on, on_code = _build(
                    kernel, x, **_COLUMN, cute_pdl=True, **release
                )
                self.assertEqual(on_code.count(_WAIT), 1)
                self.assertEqual(on_code.count(_ATTRIBUTE), 1)
                self.assertEqual(
                    on_code.count(_RELEASE),
                    int(release.get("cute_pdl_release", "entry") != "none"),
                )
                # ``cute_pdl_spread`` pads the static grid's kernel to one
                # wave's density (one CTA of 512 threads per SM at 64 CTAs);
                # the pad is a live allocation the kernel never reads.
                self.assertEqual(
                    on_code.count("_helion_pdl_spread_keep(pdl_spread_pad.toint())"),
                    int(bool(release.get("cute_pdl_spread")) and not control),
                )
                self.assertEqual(_stale_runs(on, x, sources, references, iterations), 0)
                # One graph holding producer -> kernel -> producer -> kernel:
                # the launch attribute becomes a programmatic edge inside the
                # graph.
                side = torch.cuda.Stream()
                with torch.cuda.stream(side):
                    _produce(sources[0], x)
                    on(x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _produce(sources[0], x)
                    out0 = on(x)
                    _produce(sources[1], x)
                    out1 = on(x)
                for _ in range(iterations // 4):
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(out0, references[0]))
                    self.assertTrue(torch.equal(out1, references[1]))
            _bound, on, on_code = _build(kernel, x, **_COLUMN, cute_pdl=True)
            if control:
                attribute_only = _attribute_only_control(bound, on_code)
                self.assertGreater(
                    _stale_runs(
                        attribute_only,
                        x,
                        sources,
                        references,
                        iterations,
                        delay=_CONTROL_DELAY,
                    ),
                    0,
                )

    def test_only_the_entry_release_lets_a_dependent_run_during_the_kernel(
        self,
    ) -> None:
        # One CTA sums 2^18 rows of zeros serially (hundreds of microseconds)
        # leaving room for a dependent kernel. A dependent launched behind
        # it with ``launch_pdl`` and no wait of its own writes ones over the
        # last rows the moment it is released: at ``entry`` that is while the
        # sum still has those rows ahead of it, so the sum reads them (the
        # output is no longer zero); released after the body (``exit``) or
        # at grid completion (``none``) the sum has finished reading first.
        rows, cols, tail = 1 << 18, 16, 4096
        x = torch.zeros(rows, cols, dtype=torch.bfloat16, device=DEVICE)
        bound = col_reduce_sum_static.bind((x,))
        iterations = 8

        def dirty_runs(placement: str) -> int:
            config = bound.config_spec.normalized_config(
                helion.Config(**_COLUMN, cute_pdl=True, cute_pdl_release=placement)
            )
            code = bound.to_code(config)
            self.assertEqual(code.count(_RELEASE), int(placement != "none"))
            fn = bound.compile_config(config)

            def run() -> torch.Tensor:
                x.zero_()
                out = fn(x)
                _tk()._poison_tail_without_wait[(tail * cols // _BLOCK,)](
                    x.view(-1),
                    (rows - tail) * cols,
                    rows * cols,
                    BLOCK=_BLOCK,
                    launch_pdl=True,
                )
                return out

            # Even warm eager calls can be separated by CPU scheduling long
            # enough for the reduction to finish before the dependent launches.
            # Capture both launches so the GPU observes the release ordering
            # without an intervening host launch gap.
            replay = _make_cudagraph_replay(run)
            return sum(int(bool((replay() != 0).any())) for _ in range(iterations))

        self.assertEqual(dirty_runs("none"), 0)
        self.assertEqual(dirty_runs("exit"), 0)
        self.assertEqual(dirty_runs("entry"), iterations)
        # One CTA is one wave: ``last_wave`` is the entry release.
        self.assertEqual(dirty_runs("last_wave"), iterations)

    def test_graph_replay_sees_the_fill_before_the_kernel(self) -> None:
        x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEVICE)
        bound = col_reduce_sum_static.bind((x,))
        config = bound.config_spec.normalized_config(
            helion.Config(**_COLUMN, cute_pdl=True)
        )
        fn = bound.compile_config(config)
        # ``col_reduce_sum`` zero-fills its output in a kernel that precedes
        # the reduction inside the captured graph; the reduction's launch
        # overlaps that fill and its wait orders the two.
        replay = _make_cudagraph_replay(lambda: fn(x))
        expected = x.float().sum(0).to(torch.bfloat16)
        for _ in range(3):
            torch.testing.assert_close(replay(), expected, atol=2e-2, rtol=2e-2)

    def test_a_kernel_that_releases_its_dependents_at_entry_must_be_an_ordinary_launch(
        self,
    ) -> None:
        # Producer (releases at entry, writes x late) -> middle kernel ->
        # copy of x launched as a dependent with no wait of its own, the
        # shape of the tcgen05 consumer's ordinary-operand loads ahead of
        # its wait.
        n = 512 * 1024
        x = torch.empty(n, dtype=torch.bfloat16, device=DEVICE)
        out = torch.empty_like(x)
        flag = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        sources = [
            torch.randn(n, dtype=torch.bfloat16, device=DEVICE),
            torch.randn(n, dtype=torch.bfloat16, device=DEVICE),
        ]
        iterations = 50

        def stale_runs(*, middle_is_dependent: bool, release_first: bool) -> int:
            stale = torch.zeros((), dtype=torch.int64, device=DEVICE)
            for i in range(iterations):
                j = i & 1
                _produce(sources[j], x, delay=256)
                _tk()._middle[(_PRODUCER_GRID,)](
                    flag, RELEASE_FIRST=release_first, launch_pdl=middle_is_dependent
                )
                _tk()._copy_without_wait[(n // _BLOCK,)](
                    x, out, n, BLOCK=_BLOCK, launch_pdl=True
                )
                stale += (out != sources[j]).any()
            torch.cuda.synchronize()
            return int(stale)

        # The fission producer's shape: an ordinary launch that releases at
        # entry.  It starts after the producer completes, so its dependents
        # cannot start earlier either.
        self.assertEqual(stale_runs(middle_is_dependent=False, release_first=True), 0)
        # The shape ``cute_pdl`` gave it before ``_stage_config`` kept the
        # knob away: launched as a dependent it starts while the producer
        # still runs, releases at once, and the copy reads x unwritten.
        self.assertGreater(stale_runs(middle_is_dependent=True, release_first=True), 0)
        # A dependent launch that waits before it releases is safe.
        self.assertEqual(stale_runs(middle_is_dependent=True, release_first=False), 0)
