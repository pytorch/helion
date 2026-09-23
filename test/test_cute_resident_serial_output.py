from __future__ import annotations

import ast
from copy import deepcopy
import itertools
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test.test_cute_resident_pipeline_depth import _run_pipeline
from test.test_cute_resident_reductions import _namespace
from test.test_cute_resident_reductions import _rewrite
from test.test_cute_resident_reductions import _source
from test.test_cute_resident_serial_row_seeds import _bind
from test.test_cute_resident_serial_row_seeds import _cpu_only  # noqa: F401

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.autotuner_heuristics.cute_resident_reductions import (
    CuteResidentReductionHeuristic,
)
from helion._compiler.cute.resident_reductions import _packed_terminal_body
from helion._compiler.cute.resident_reductions import _terminal_output_product
from helion._testing import skipUnlessBackends
from helion.autotuner.compiler_coverage import append_compiler_coverage
from helion.autotuner.config_generation import ConfigGeneration

ROW = "cute_reduction_row_schedule"
PACK = "cute_reduction_pack_output"


def _carrier(bound):
    assert bound.host_function is not None
    carrier = CuteResidentReductionHeuristic.serial_output_carrier(
        bound.env, bound.host_function.device_ir
    )
    assert carrier is not None
    return carrier


@pytest.mark.parametrize("rows", (0, 1, 2, 3, 4, 7, 17))
@pytest.mark.parametrize("group", (1, 2, 3, 4))
def test_serial_consumption_preserves_carries_masks_and_copy_age(rows, group):
    source = _rewrite(
        _source(rows=rows, width=2),
        width=2,
        group_rows=group,
        pipelined=True,
        row_schedule="serial",
    )
    assert "_cute_resident_sums_disjoint" in source

    def namespace():
        result = _namespace()
        cast("SimpleNamespace", result["cutlass"]).utils = SimpleNamespace(
            SmemAllocator=lambda: SimpleNamespace(
                allocate_tensor=lambda dtype, layout: np.full(
                    layout, np.nan, dtype=dtype
                )
            )
        )
        result["_cute_resident_sums_disjoint"] = lambda values, threads, scratch, slot: (
            values.copy()
        )
        return result

    with patch("test.test_cute_resident_pipeline_depth._namespace", namespace):
        _run_pipeline(source, rows, 2, group)


def test_deferred_requires_cross_warp_publication():
    # The one-thread mathematical fixture has no full-CTA publication in its
    # sum helper. It must not acquire the deferred shared-ring retirement.
    source = _source(rows=7)
    rewritten = _rewrite(
        source, group_rows=3, pipelined=True, row_schedule="serial_deferred"
    )
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(rewritten))


@pytest.mark.parametrize("kind,columns", (("layer", 8192), ("rms", 4096)))
@skipUnlessBackends(["cute"])
def test_full_population_reaches_coupled_carrier_and_preserves_old_seed_prefix(
    kind, columns
):
    bound, args = _bind(256, columns, kind=kind, dtype=torch.bfloat16)
    assert bound.host_function is not None
    spec = bound.config_spec
    with bound.env:
        old_default = deepcopy(spec.compiler_default_config)
        with patch.object(
            CuteResidentReductionHeuristic, "serial_output_carrier", return_value=None
        ):
            old = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        assert spec.compiler_seed_configs[: len(old)] == old
        assert spec.compiler_default_config == old_default
        appended = spec.compiler_seed_configs[len(old) :]
        assert [
            (config[ROW], config.config.get(PACK, False)) for config in appended
        ] == [("serial", False), ("serial_deferred", False), ("serial_deferred", True)]
        generation = ConfigGeneration(spec)
        base = generation.random_population_flat(100)
        population, outcomes = append_compiler_coverage(
            base, generation, cached_configs=list, pin=lambda _: None
        )
        assert population[:100] == base
        for config in appended:
            flat, normalized = generation.strict_config_pair(config)
            assert flat in population
            assert generation.unflatten(flat) == normalized
        assert all(
            outcome.outcome in ("added", "already_present") for outcome in outcomes
        )
    for config in appended:
        if kind == "rms" and config.config.get(PACK):
            # RMS ends in a subtraction. Packing an earlier product would
            # expand this terminal-only transformation's arithmetic scope.
            with pytest.raises(helion.exc.BackendUnsupported):
                bound.to_code(config)
        else:
            assert "_cute_resident_sums_disjoint" in bound.to_code(config)
    if kind == "layer":
        assert "mul_packed_f32x2" in bound.to_code(appended[-1])


@skipUnlessBackends(["cute"])
def test_default_modes_are_omitted_and_do_not_change_generated_source():
    bound, _ = _bind(128, 4096, kind="layer")
    carrier = _carrier(bound)
    control = deepcopy(carrier)
    control.config["cute_reduction_group_rows"] = 2
    source = bound.to_code(control)
    explicit = helion.Config.from_dict(control.config | {ROW: "batched", PACK: False})
    normalized = bound.config_spec.normalized_config(explicit)
    assert ROW not in normalized and PACK not in normalized
    assert bound.to_code(normalized) == source
    for key, bad in (
        (ROW, None),
        (ROW, True),
        (ROW, "unknown"),
        (PACK, 0),
        (PACK, 1),
        (PACK, "true"),
    ):
        with pytest.raises(helion.exc.InvalidConfig):
            bound.config_spec.normalized_config(
                helion.Config.from_dict(control.config | {key: bad})
            )
    for replacement in (
        {"cute_reduction_schedule": "resident"},
        {"cute_reduction_pipeline_depth": 4},
    ):
        invalid = helion.Config.from_dict(
            carrier.config | {ROW: "serial_deferred"} | replacement
        )
        with pytest.raises(helion.exc.InvalidConfig, match=ROW):
            bound.config_spec.normalize(invalid)
    original = bound.config_spec.target_device_capability
    try:
        bound.config_spec.target_device_capability = (9, 0)
        with pytest.raises(helion.exc.InvalidConfig, match=PACK):
            bound.config_spec.normalized_config(
                helion.Config.from_dict(carrier.config | {PACK: True})
            )
    finally:
        bound.config_spec.target_device_capability = original


def test_terminal_matcher_rejects_changed_math_extra_uses_and_narrow_products():
    source = "result = left * right\ncast = cutlass.BFloat16(result)\n(out.iterator + column).store(cast)"
    types = dict.fromkeys(("left", "right", "result"), "cutlass.Float32")
    assert _terminal_output_product(ast.parse(source).body, types, {}) == "result"
    for wrong in (
        source.replace("left * right", "left + right"),
        source.replace("cast =", "extra = result + left\ncast ="),
        source.replace("cast = cutlass.BFloat16(result)", "cast = result + right"),
    ):
        assert _terminal_output_product(ast.parse(wrong).body, types, {}) is None
    assert (
        _terminal_output_product(
            ast.parse(source).body, types | {"left": "cutlass.BFloat16"}, {}
        )
        is None
    )


class _Term:
    def __init__(self, tree):
        self.tree = tree

    def __add__(self, other):
        return _Term(("add", self.tree, other.tree))

    def __sub__(self, other):
        return _Term(("sub", self.tree, other.tree))

    def __mul__(self, other):
        return _Term(("mul", self.tree, other.tree))


@pytest.mark.parametrize("width", (2, 4, 8))
def test_packing_keeps_each_scalar_dag_and_ordered_output_lane(width):
    body = ast.parse(
        "lane = chunk * WIDTH + component\na = x[lane] * coefficient\nb = a + average\nc = dy[lane] - b\nresult = c * scale\ncast = cutlass.BFloat16(result)\nout[component] = cutlass.BFloat16(cast)"
    ).body
    serial = ast.For(
        target=ast.Name("component", ast.Store()),
        iter=ast.Call(ast.Name("range", ast.Load()), [ast.Constant(width)], []),
        body=deepcopy(body),
        orelse=[],
    )
    count = itertools.count()
    packed = _packed_terminal_body(
        body,
        "component",
        width,
        "result",
        lambda name: f"{name}_{next(count)}",
        "unused_coordinate",
    )
    calls = []

    def mul(left, right, *, rnd, ftz):
        assert rnd == "rn" and ftz is False
        calls.append((left, right))
        return left[0] * right[0], left[1] * right[1]

    values = {
        "chunk": 1,
        "WIDTH": width,
        "x": [_Term(("x", i)) for i in range(2 * width)],
        "dy": [_Term(("dy", i)) for i in range(2 * width)],
        "coefficient": _Term("coefficient"),
        "average": _Term("average"),
        "scale": _Term("scale"),
        "cutlass": SimpleNamespace(BFloat16=lambda value: _Term(("bf16", value.tree))),
        "cute": SimpleNamespace(arch=SimpleNamespace(mul_packed_f32x2=mul)),
    }

    outputs = []
    alternatives: tuple[list[ast.stmt], list[ast.stmt]] = ([serial], packed)
    for statements in alternatives:
        out: list[_Term | None] = [None] * width
        namespace: dict[str, object] = values | {"out": out}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                "<terminal output>",
                "exec",
            ),
            namespace,
        )
        assert all(value is not None for value in out)
        outputs.append([cast("_Term", value).tree for value in out])
    assert outputs[0] == outputs[1]
    assert len(calls) == width // 2
