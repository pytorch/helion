from __future__ import annotations

import ast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_atomic_ops import _simulate_register_load_program

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _readonly_count(x, limits, capacity: hl.constexpr, remap: hl.constexpr = False):
    out = torch.empty(x.size(0), device=x.device, dtype=torch.int32)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(capacity)
        loaded = hl.load(x, [row, lane], extra_mask=lane < x.size(1))
        valid = lane < x.size(1)
        keys = loaded * 2 + 1
        iteration = hl.full([], 0, dtype=torch.int32)
        count = hl.full([], 0, dtype=torch.int32)
        history = hl.full([], 1, dtype=torch.int32)
        limit = limits[row]
        if limit > 0:
            while iteration < limit:
                previous = count
                values = keys
                if remap:
                    values = torch.gather(keys, 0, ((lane + 1) % capacity).long())
                count = count + torch.sum(
                    valid & (values >= iteration), dtype=torch.int32
                )
                history = history + previous
                iteration = iteration + 1
            selected = torch.where(valid, keys, 0)
        else:
            selected = torch.where(valid, -keys, 0)
        prefix = hl.cumsum(selected, dim=0)
        out[row] = count + history + prefix.sum(dtype=torch.int32)
    return out


def _codegen(x, limits, threads, mode, *, remap=False):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _readonly_count,
            (x, limits, 1 << (x.size(1) - 1).bit_length(), remap),
        )
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_threads=threads,
            cute_fragment_reduction=mode,
            cute_fragment_bounded_gather=remap,
            cute_fragment_scan="cooperative",
            cute_fragment_warp_scan=True,
        )
        return bound.to_code(config)


@pytest.mark.parametrize("width,threads", [(17, 32), (65, 128), (257, 128)])
def test_cta_reductions_preserve_padded_loads_readonly_control_and_carries(
    width, threads
):
    raw = (torch.arange(3 * width).reshape(3, width) % 11 - 4).to(torch.int32)
    x = raw
    limits = torch.tensor([0, 1, 4], dtype=torch.int32)
    expected = []
    for row, limit in zip(raw, limits, strict=True):
        row = row * 2 + 1
        if int(limit) > 0:
            value, history = 0, 1
            for index in range(int(limit)):
                history += value
                value += int((row >= index).sum())
            value += history
            padded = torch.zeros(1 << (width - 1).bit_length(), dtype=torch.int32)
            padded[:width] = row
            value += int(padded.cumsum(0).sum())
        else:
            padded = torch.zeros(1 << (width - 1).bit_length(), dtype=torch.int32)
            padded[:width] = row
            value = 1 - int(padded.cumsum(0).sum())
        expected.append(value)
    expected = torch.tensor(expected, dtype=torch.int32)
    before = x.clone(), limits.clone()
    for mode in ("cta", "cta_unroll"):
        source = _codegen(x, limits, threads, mode)
        module = ast.parse(source)
        loops = [node for node in ast.walk(module) if isinstance(node, ast.While)]
        assert len(loops) == 1
        while_text = ast.unparse(loops[0])
        assert "fragment_invariant" not in while_text
        for order in (list(range(threads)), list(reversed(range(threads)))):
            out = torch.full_like(limits, -99)
            _simulate_register_load_program(
                source,
                x,
                threads,
                host_tensors={"limits": limits, "out": out},
                lane_order=order,
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert torch.equal(x, before[0])
        assert torch.equal(limits, before[1])


@pytest.mark.parametrize("mode", ["cta", "cta_unroll"])
@pytest.mark.parametrize("kind", ["sum", "min", "max", "prod"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float64])
def test_cta_reductions_preserve_empty_stripes_and_typed_combines(mode, kind, dtype):
    x = ((torch.arange(3 * 128).reshape(3, 128) % 4) - 2).to(dtype)
    if dtype == torch.float64:
        x[1, 5] = float("nan")
    values = (x + 1).reshape(3, 4, 32)
    if kind == "sum":
        expected = values.sum(-1, dtype=dtype)
    elif kind == "min":
        expected = values.amin(-1)
    elif kind == "max":
        expected = values.amax(-1)
    else:
        expected = values.prod(-1, dtype=dtype)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_factored_cta_reduction, (x, kind))
        config = bound.config_spec.default_config()
        config.config.update(cute_fragment_threads=128, cute_fragment_reduction=mode)
        source = bound.to_code(config)
    for order in (list(range(128)), list(reversed(range(128)))):
        out = torch.full_like(expected, -99)
        _simulate_register_load_program(
            source, x, 128, host_tensors={"out": out}, lane_order=order
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _factored_cta_reduction(x, kind: hl.constexpr):
    columns = hl.specialize(x.size(1))
    out = torch.empty((x.size(0), 4), device=x.device, dtype=x.dtype)
    for row in hl.grid(x.size(0)):
        values = hl.load(x, [row, hl.arange(columns)]) + 1
        factored = values.reshape(4, columns // 4)
        if kind == "sum":
            reduced = factored.sum(-1, dtype=x.dtype)
        elif kind == "min":
            reduced = factored.amin(-1)
        elif kind == "max":
            reduced = factored.amax(-1)
        else:
            reduced = factored.prod(-1, dtype=x.dtype)
        out[row, :] = reduced
    return out
