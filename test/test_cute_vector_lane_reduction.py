from __future__ import annotations

import ast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_block_reduce_cta import _execute
from test.test_cute_block_reduce_cta import _input
from test.test_cute_block_reduce_cta import _tile_reduce

import helion
from helion._compiler import tile_strategy as lanes


def _code(operation, threads, width, layout, *, tail=5):
    rows, columns, block = 3, threads * width * 2 + tail, threads * width * 2
    x = _input(rows, columns, operation)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_tile_reduce, (x, operation))
        config = bound.config_spec.default_config()
        config.config.update(
            block_sizes=[2, block],
            num_threads=[2, threads],
            cute_vector_widths=[width, width],
            cute_lane_layouts=[layout, layout],
            loop_orders=[[1, 0]],
        )
        code = bound.to_code(config)
    return code, x, config, block


@pytest.mark.parametrize("operation", ["max", "min", "sum", "prod"])
@pytest.mark.parametrize("threads,width", [(2, 8), (16, 4), (32, 2), (64, 2)])
@pytest.mark.parametrize("layout", ["blocked", "strided"])
def test_vector_slots_belong_to_complete_reduction(operation, threads, width, layout):
    code, x, _, block = _code(operation, threads, width, layout)
    actual, writes = _execute(code, x, operation)
    expected = torch.stack(
        [
            getattr(
                x[:, start : start + block],
                {"max": "amax", "min": "amin"}.get(operation, operation),
            )(-1)
            for start in range(0, x.shape[1], block)
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sorted(offset for _, _, offset in writes) == list(range(expected.numel()))


def _loop(prefix="base = outer * 8", suffix="", *, owned=True):
    marker = lanes._lane_reduce_marker_expr(
        "value", "max", "cutlass.Float32(float('-inf'))", 2, owner_lane="outer"
    )
    loop = ast.parse(
        "for outer in range(2):\n"
        + "".join(f"    {line}\n" for line in prefix.splitlines())
        + "    for vector in cutlass.range_constexpr(8):\n"
        + "        index = base + vector\n"
        + "        value = (x.iterator + index).load()\n"
        + f"        reduced = {marker}\n"
        + "        out.store(reduced)\n"
        + "".join(f"    {line}\n" for line in suffix.splitlines())
    ).body[0]
    setattr(loop, lanes.HELION_LANE_LOOP_VAR_ATTR, "outer")
    vector = next(node for node in loop.body if isinstance(node, ast.For))
    if owned:
        vector._helion_vector_reduction_owner = "outer"
    return loop, vector


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ("base = values.load()", ""),
        ("base = outer * 8", "flush.store(base)"),
        ("base = base + 8", ""),
        ("base = later\nlater = outer * 8", ""),
        ("base = value + outer * 8", ""),
        ("base = shared[0]", ""),
    ],
)
def test_vector_ownership_declines_unproved_preparation(prefix, suffix):
    loop, vector = _loop(prefix, suffix)
    with pytest.raises(helion.exc.BackendUnsupported, match="scalar ownership proof"):
        lanes._flatten_vector_reduction_lane(loop, vector, "outer")


def test_independent_serial_loop_is_not_flattened():
    loop, _ = _loop(owned=False)
    output = lanes.interchange_lane_outside_serial_reductions([loop])
    text = ast.unparse(ast.Module(output, []))
    assert "// 8" not in text
    assert "% 8" not in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("threads,width", [(2, 8), (32, 4), (64, 2)])
def test_vector_lane_reduction_native(threads, width):
    for operation in ("max", "sum"):
        _, x, config, block = _code(operation, threads, width, "strided")
        x = x.cuda()
        actual = _tile_reduce.bind((x, operation)).compile_config(config)(x, operation)
        expected = torch.stack(
            [
                getattr(
                    x[:, start : start + block], "amax" if operation == "max" else "sum"
                )(-1)
                for start in range(0, x.shape[1], block)
            ],
            dim=1,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutation", ["zero", "orelse", "target", "nested", "owner", "coordinate_write"]
)
def test_vector_ownership_rejects_incomplete_domains(mutation):
    loop, vector = _loop()
    if mutation == "zero":
        vector.iter.args[0] = ast.Constant(0)
    elif mutation == "orelse":
        vector.orelse = [ast.Pass()]
    elif mutation == "target":
        vector.target = ast.Tuple(
            [ast.Name("a", ast.Store()), ast.Name("b", ast.Store())], ast.Store()
        )
    elif mutation == "nested":
        vector.body.insert(
            0, ast.parse("for item in range(2):\n    value = item").body[0]
        )
    elif mutation == "owner":
        marker = lanes._find_lane_reduce_call(vector.body[2])
        assert marker is not None
        marker.args[9] = ast.Constant("different_owner")
    else:
        vector.body.insert(0, ast.parse("outer = 0").body[0])
    with pytest.raises(helion.exc.BackendUnsupported, match="scalar ownership proof"):
        lanes._flatten_vector_reduction_lane(loop, vector, "outer")


def test_single_outer_vector_retains_every_coordinate():
    loop, vector = _loop()
    loop.iter.args[0] = ast.Constant(1)
    flat = lanes._flatten_vector_reduction_lane(loop, vector, "outer")
    assert ast.literal_eval(flat.iter.args[0]) == 8
    assert "outer // 8" in ast.unparse(flat)
    assert "outer % 8" in ast.unparse(flat)
