from __future__ import annotations

import ast
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test.test_cute_resident_pipeline_depth import _run_pipeline
from test.test_cute_resident_reductions import _rewrite
from test.test_cute_resident_reductions import _source
from test.test_cute_resident_serial_row_seeds import _bind
from test.test_cute_resident_serial_row_seeds import _cpu_only  # noqa: F401

from helion._compiler.cute.resident_reductions import _local_tree_update
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig

KEY = "cute_reduction_local_tree"


class _Term:
    def __init__(self, expression):
        self.expression = expression

    def __add__(self, other):
        return _Term((self.expression, other.expression))

    def __radd__(self, other):
        assert other == 0
        return self


@pytest.mark.parametrize("extent", (1, 2, 4, 8, 16, 32, 64))
def test_complete_balanced_tree_has_no_uninitialized_reads(extent):
    body = _local_tree_update("lane", extent, "value", "fragment", "result")
    code = compile(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
        "<tree>",
        "exec",
    )
    namespace = {
        "fragment": [None] * (extent.bit_length() - 1),
        "cutlass": SimpleNamespace(const_expr=bool, Float32=int),
    }
    expected = [_Term(i) for i in range(extent)]
    for lane in range(extent):
        namespace.update(lane=lane, value=_Term(lane))
        exec(code, namespace)
        assert ("result" in namespace) == (lane == extent - 1)
    while len(expected) > 1:
        expected = [
            left + right
            for left, right in zip(expected[::2], expected[1::2], strict=True)
        ]
    assert namespace["result"].expression == expected[0].expression


@pytest.mark.parametrize("extent", (1, 2, 4, 8, 16, 32, 64))
def test_tree_keeps_positive_zero_sum_identity(extent):
    body = _local_tree_update("lane", extent, "value", "fragment", "result")
    code = compile(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
        "<zero tree>",
        "exec",
    )
    namespace = {
        "fragment": np.full(extent.bit_length() - 1, np.nan, dtype=np.float32),
        "cutlass": SimpleNamespace(const_expr=bool, Float32=np.float32),
    }
    for lane in range(extent):
        namespace.update(lane=lane, value=np.float32(-0.0))
        exec(code, namespace)
    assert namespace["result"] == 0
    assert not np.signbit(namespace["result"])


@pytest.mark.parametrize(
    "width,rows,group_rows,depth",
    ((1, 0, 1, 2), (2, 7, 2, 2), (4, 13, 4, 4), (8, 17, 2, 4)),
)
def test_pipeline_tree_preserves_carries_masks_and_copy_lifetimes(
    width, rows, group_rows, depth
):
    source = _rewrite(
        _source(rows=rows, width=width),
        width=width,
        group_rows=group_rows,
        pipelined=True,
        pipeline_depth=depth,
        local_tree=True,
    )
    assert "resident_tree" in source
    with np.errstate(over="raise", invalid="raise"):
        _run_pipeline(source, rows, depth, group_rows)


@skipUnlessBackends(["cute"])
def test_default_off_and_searchable_seed_siblings():
    bound, args = _bind(256, 1024, kind="layer")
    seeds = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if not seed.config.get("cute_reduction_row_schedule")
    ]
    trees = [seed for seed in seeds if seed.config.get(KEY)]
    controls = [seed for seed in seeds if not seed.config.get(KEY)]
    assert trees and seeds == controls + trees
    control = deepcopy(
        next(
            seed
            for seed in controls
            if seed.config.get("cute_reduction_schedule") == "pipelined"
        )
    )
    tree = deepcopy(control)
    tree.config[KEY] = True
    assert tree in trees
    bound.config_spec.normalize(tree)
    assert tree.config[KEY] is True
    old = bound.to_code(control)
    off = deepcopy(control)
    off.config[KEY] = False
    assert bound.to_code(off) == old
    assert "resident_tree" in bound.to_code(tree)
    assert "resident_tree" not in old
    for value in (0, 1, None, "true"):
        invalid = deepcopy(control)
        invalid.config[KEY] = value
        with pytest.raises(InvalidConfig, match=KEY):
            bound.config_spec.normalize(invalid)
    invalid = deepcopy(tree)
    invalid.config["cute_reduction_schedule"] = "scalar"
    with pytest.raises(InvalidConfig, match=KEY):
        bound.config_spec.normalize(invalid)
    bound.config_spec.normalize(invalid, _fix_invalid=True)
    assert KEY not in invalid.config


@pytest.mark.parametrize("kind,columns", (("layer", 8192), ("rms", 4096)))
@skipUnlessBackends(["cute"])
def test_true_seeds_survive_the_actual_full_population(kind, columns):
    bound, args = _bind(256, columns, kind=kind)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        pairs = generation.seed_flat_config_pairs()
        true = [(flat, config) for flat, config in pairs if config.config.get(KEY)]
        assert true
        population = generation.random_population_flat(100)
        assert population[0] == generation.default_flat()
        assert population[1 : len(pairs) + 1] == [flat for flat, config in pairs]
        for flat, config in true:
            assert generation.unflatten(flat) == config
            assert flat in population
