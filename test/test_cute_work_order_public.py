from __future__ import annotations

import ast
import difflib

import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_work_order_scalar import _ragged_sum
import helion


def _policies(bound, selected):
    spec = bound.config_spec
    assert len(spec.cute_work_order_candidates) == 1
    chosen = spec.cute_work_order_candidates[0] if selected else None
    return [
        "longest_first" if axis == chosen else "identity"
        for axis in spec.cute_work_order_axes
    ]


def _has_shared_permutation(source):
    tree = ast.parse(source)
    calls = [
        ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]
    assert "cute.arch.shuffle_sync" in calls and "cute.arch.warp_reduction_max" in calls
    assert "cutlass.Uint64" in calls


def _assert_original_body_exact(before, after):
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    changes = [
        entry
        for entry in difflib.SequenceMatcher(a=old, b=new).get_opcodes()
        if entry[0] != "equal"
    ]
    assert len(changes) == 1
    tag, i, j, start, end = changes[0]
    assert tag == "insert" and i == j
    inserted = "".join(new[start:end])
    assert inserted.lstrip().startswith("work_candidate = ")
    assert new[end - 1].rstrip().endswith(" = work_selected")
    assert "".join(new[:start] + new[end:]) == before
    _has_shared_permutation(after)


def test_normal_ordinary_work_order_and_identity(tmp_path):
    args = (
        torch.empty((5, 64)),
        torch.arange(5, dtype=torch.int64),
        torch.arange(5, dtype=torch.int64) + 32,
        torch.empty(5),
    )
    sources = []
    for selected in (None, False, True):
        with _cpu_codegen():
            bound = _ragged_sum._bind_isolated(args)
            config = helion.Config(block_sizes=[32], num_warps=4, pid_type="flat")
            if selected is not None:
                config = helion.Config.from_dict(
                    {**config, "cute_grid_work_order": _policies(bound, selected)}
                )
            with bound.env.use_runtime_arg_values(_runtime_values(_ragged_sum, args)):
                sources.append(bound.to_code(config))
    assert sources[0] == sources[1]
    _has_shared_permutation(sources[2])
    _assert_original_body_exact(sources[0], sources[2])
    for name, source in zip(("absent", "identity", "selected"), sources, strict=True):
        (tmp_path / (name + ".py")).write_text(source)


def test_normal_fused_work_order_and_identity(tmp_path):
    kernel, args = _kda_fixture()
    sources = []
    for selected in (None, False, True):
        with _cpu_codegen():
            bound = kernel._bind_isolated(args)
            config = helion.Config(
                block_sizes=[128],
                num_warps=16,
                num_stages=2,
                cute_chained_mma_schedule="tcgen05_tmem",
                cute_chained_group_contractions=True,
                cute_chained_scratch_layout="xor",
                cute_chained_pointwise_vectorize=True,
                cute_chained_scan_schedule="warp",
                cute_chained_pointwise_cache_bytes=4096,
                cute_chained_pointwise_unroll=8,
                cute_chained_warp_mma_rows=32,
            )
            if selected is not None:
                config = helion.Config.from_dict(
                    {**config, "cute_grid_work_order": _policies(bound, selected)}
                )
            with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
                sources.append(bound.to_code(config))
    assert sources[0] == sources[1]
    _has_shared_permutation(sources[2])
    _assert_original_body_exact(sources[0], sources[2])
    for name, source in zip(("absent", "identity", "selected"), sources, strict=True):
        (tmp_path / (name + ".py")).write_text(source)
