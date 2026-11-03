"""The persistent grouped GEMM example reaches the tcgen05 grouped family.

``examples.grouped_gemm.grouped_gemm_jagged_persistent`` hand-rolls its tile
scheduler: ``hl.grid`` workers, an ``hl.grid`` over the groups and one over
each group's tiles, and row/column indices built with ``hl.arange`` from
``register_block_size`` blocks under data-dependent branches.  No CuTe MMA
family lowers that structure directly (the collective marker needs grid
tiles, the tcgen05 planner ``hl.tile`` tiles); the segmented-matmul
normalization (``cute_segmented_matmul_tiling``, proposed automatically under
autotuning) recognises the schedule as the per-group tile loop it spells out
and emits the segment tile space ``grouped_gemm_jagged`` gets, which the
tcgen05 grouped scheduler takes over.  The refusal cases pin the proof: a
schedule that skips a tile, a wider row mask, a changed store value or a
dropped column mask keep the original program, while the column-fastest
linearization, a commuted tile index and a constant worker count are the
same schedule.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import textwrap
from typing import TYPE_CHECKING

from benchmarks.cute.matmul_sweep import _group_rows
import pytest
import torch

from test._cute_grouped_gemm_split_sizes import _selected_config
from test.cute_population_contracts import tcgen05_mma_support
from test.test_cute_shared_rhs_grouped import _plans

import helion
from helion._compiler.cute.cute_mma import analyze_tcgen05_grouped_worklist
from helion._testing import cpu_render_target
from helion._testing import import_example

_examples_grouped_gemm = import_example("grouped_gemm")
grouped_gemm_jagged = _examples_grouped_gemm.grouped_gemm_jagged
grouped_gemm_jagged_persistent = _examples_grouped_gemm.grouped_gemm_jagged_persistent

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from pathlib import Path

    from helion.runtime.kernel import BoundKernel


# (total_M, N, K, groups): the sweep's ``groupedp`` cell and the example's
# own ``main()`` shapes.
SWEEP_CELL = (8192, 4096, 4096, 8)
EXAMPLE = (640, 128, 256, 4)


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with cpu_render_target(mma_support=tcgen05_mma_support()):
        yield
    torch.set_num_threads(previous)


def _args(shape: tuple[int, int, int, int]) -> tuple[torch.Tensor, ...]:
    total_m, n, k, groups = shape
    offsets = [0]
    for rows in _group_rows(total_m, groups, seed=groups):
        offsets.append(offsets[-1] + rows)
    assert offsets[-1] == total_m
    return (
        torch.empty(total_m, k, dtype=torch.bfloat16),
        torch.empty(k, n, dtype=torch.bfloat16),
        torch.tensor(offsets, dtype=torch.int32),
    )


def _bind(
    fn: Callable[..., object], args: tuple[torch.Tensor, ...], **settings: object
) -> BoundKernel:
    """Bind on the modelled B200 (the module fixture holds the target)."""
    kernel = helion.kernel(
        fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        **settings,
    )
    return kernel._bind_isolated(args)


def _root_loop(bound: BoundKernel) -> ast.For:
    host = bound.host_function
    assert host is not None
    return next(statement for statement in host.body if isinstance(statement, ast.For))


def test_persistent_example_binds_on_the_modelled_b200() -> None:
    # The body calls ``helion.runtime.get_num_sm`` on the host; the modelled
    # target answers it with a plain probe, not a MagicMock the type
    # propagator mistakes for a device-only API function.
    args = _args(EXAMPLE)
    bound = _bind(grouped_gemm_jagged_persistent.fn, args)
    code = bound.to_code(bound.config_spec.default_config())
    assert "helion.runtime.get_num_sm(" in code
    loop = _root_loop(bound)
    assert isinstance(loop.target, ast.Name) and loop.target.id == "worker_id"


@pytest.mark.parametrize("shape", [SWEEP_CELL, EXAMPLE], ids=["sweep_g8", "example_g4"])
def test_persistent_example_reaches_the_grouped_tcgen05_family(
    shape: tuple[int, int, int, int],
) -> None:
    args = _args(shape)
    bound = _bind(
        grouped_gemm_jagged_persistent.fn, args, cute_segmented_matmul_tiling=True
    )
    config = _selected_config(64)
    persistent = bound.to_code(config)
    jagged = _bind(grouped_gemm_jagged.fn, args, cute_segmented_matmul_tiling=True)
    reference = jagged.to_code(config)
    spec = bound.config_spec
    assert spec.tcgen05.search_enabled
    assert bound.host_function is not None
    (fact,) = spec.matmul_facts
    analysis = analyze_tcgen05_grouped_worklist(
        bound.env, bound.host_function.device_ir, fact
    )
    assert analysis is not None
    assert analysis.seed_facts.groups_hint == shape[3]
    assert analysis.seed_facts.packed_m_hint == shape[0]
    assert any(
        seed.config.get("tcgen05_grouped_mode") == "worklist_nm"
        for seed in spec.compiler_seed_configs
    )
    # The schedule and its two registered tiles are gone from the program.
    loop = _root_loop(bound)
    assert isinstance(loop.target, ast.Tuple)
    source = "\n".join(ast.unparse(statement) for statement in bound.host_function.body)
    assert "register_block_size" not in source
    assert "arange" not in source
    # The same tcgen05 grouped program as the per-group tile loop's.
    assert "cute.gemm(" in persistent
    assert _plans(persistent) == _plans(reference)


_SOURCE = textwrap.dedent(inspect.getsource(grouped_gemm_jagged_persistent.fn))
_SOURCE = _SOURCE[_SOURCE.index("def ") :]

# (replacements, rewritten): the schedule variants of the example's source.
_VARIANTS = {
    "column_fastest": (
        (
            ("tile_in_group % num_m_tiles", "tile_in_group // num_n_tiles"),
            ("tile_in_group // num_m_tiles", "tile_in_group % num_n_tiles"),
        ),
        True,
    ),
    "commuted_tile_index": (
        (
            (
                "local_tile * num_workers + worker_id",
                "worker_id + num_workers * local_tile",
            ),
        ),
        True,
    ),
    "constant_worker_count": (
        (("helion.runtime.get_num_sm(A_packed.device)", "8"),),
        True,
    ),
    "skips_a_tile": (
        (
            (
                "local_tile * num_workers + worker_id",
                "local_tile * num_workers + worker_id + 1",
            ),
        ),
        False,
    ),
    "wider_row_mask": ((("row_idx < group_end", "row_idx < group_end + 1"),), False),
    "scaled_store": ((("acc.to(out.dtype)", "(acc * 2).to(out.dtype)"),), False),
    "row_mask_only_store": (
        (("extra_mask=valid_2d", "extra_mask=rows_valid[:, None]"),),
        False,
    ),
}


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_schedule_variants(variant: str, tmp_path: Path) -> None:
    replacements, rewritten = _VARIANTS[variant]
    source = _SOURCE
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    # Kernels are parsed from their source file.
    path = tmp_path / f"{variant}.py"
    path.write_text(
        "import helion\nimport helion.language as hl\nimport torch\n\n" + source
    )
    spec = importlib.util.spec_from_file_location(f"_persistent_{variant}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = _args(EXAMPLE)
    bound = _bind(
        module.grouped_gemm_jagged_persistent,
        args,
        cute_segmented_matmul_tiling=True,
    )
    loop = _root_loop(bound)
    assert isinstance(loop.target, ast.Tuple) == rewritten
    assert bound.config_spec.tcgen05.search_enabled == rewritten
