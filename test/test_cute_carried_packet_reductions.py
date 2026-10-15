from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_bounded_cache_codegen import _cpu_target
from test.test_cute_full_reduction_loop_encoding import rms_norm

import helion
from helion._compiler.cute.vector_reduction_packets import _bounded_count
from helion._compiler.cute.vector_reduction_packets import optimize_vector_reductions
from helion.autotuner.config_generation import ConfigGeneration


def _source(
    width: int = 4, trips: int = 8, packets: int = 1, *, padded: bool = False
) -> str:
    extent = width * trips * packets
    valid = extent - width if padded else extent
    stores = "\n".join(
        f"        cache[offset + packet * {width} + {i}] = xs[{i}]"
        for i in range(width)
    )
    load = "cutlass.Uint32(xs[lane]).bitcast(cutlass.Float32)"
    if padded:
        load += " if mask else cutlass.Float32(0)"
    return f"""cache = cute.make_rmem_tensor({extent}, cutlass.Uint32)
acc = cutlass.Float32(0)
for offset in range(cutlass.Int32(0), cutlass.Int32({extent}), cutlass.Int32({width * packets})):
    for packet in range({packets}):
        base = offset + packet * {width}
        mask = base < {valid}
        xs = _cute_load_l2_evict_last(x.iterator + base if base < {valid} else x.iterator + 0, ir.VectorType.get([{width}], cutlass.Uint32.mlir_type))
{stores}
        for lane in cutlass.range_constexpr({width}):
            value = {load}
            product = value * value
            acc = acc + cutlass.Float32(product)
result = acc
"""


def _transform(source: str, enabled: bool = True) -> ast.Module:
    tree = ast.parse(source)
    body = optimize_vector_reductions(
        tree.body,
        {"EXTENT": 32},
        thread_block_dims=(128, 1, 1),
        independent_accumulators=enabled,
        replicated_single_use=False,
    )
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


class _Word(np.uint32):
    mlir_type = np.uint32

    def bitcast(self, dtype: object) -> np.float32:
        assert dtype is np.float32
        return np.asarray(self, dtype=np.uint32).view(np.float32)[()]


def _execute(
    tree: ast.Module, values: np.ndarray
) -> tuple[float, list[tuple[object, ...]], np.ndarray]:
    events: list[tuple[object, ...]] = []
    storage = np.full(values.size, 0xDEADBEEF, dtype=np.uint32)

    class Cache:
        def __setitem__(self, index: int, value: object) -> None:
            storage[index] = value
            events.append(("store", int(index), int(storage[index])))

    class Pointer:
        def __init__(self, index: int = 0) -> None:
            self.index = index

        def __add__(self, delta: int) -> Pointer:
            return Pointer(self.index + int(delta))

    def load(pointer: Pointer, typ: tuple[int, object]) -> np.ndarray:
        width, dtype = typ
        assert dtype is np.uint32
        events.append(("load", pointer.index, width))
        return values.view(np.uint32)[pointer.index : pointer.index + width]

    namespace = {
        "x": SimpleNamespace(iterator=Pointer()),
        "_cute_load_l2_evict_last": load,
        "cute": SimpleNamespace(make_rmem_tensor=lambda n, dtype: Cache()),
        "cutlass": SimpleNamespace(
            Float32=np.float32, Uint32=_Word, Int32=int, range_constexpr=range
        ),
        "ir": SimpleNamespace(
            VectorType=SimpleNamespace(get=lambda shape, dtype: (shape[0], dtype))
        ),
    }
    with np.errstate(over="ignore", invalid="ignore"):
        exec(compile(tree, "<carried-packet-test>", "exec"), namespace)
    return namespace["result"], events, storage


@pytest.mark.parametrize("width,trips,packets", [(2, 8, 2), (4, 8, 1), (8, 1, 2)])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("population", ["random", "magnitude", "nonfinite"])
def test_carried_packets_preserve_memory_order_and_explicit_tree(
    width: int, trips: int, packets: int, padded: bool, population: str
) -> None:
    source = _source(width, trips, packets, padded=padded)
    transformed = _transform(source)
    assert "_helion_packet_sum" in ast.unparse(transformed)
    values = (
        np.random.default_rng(19)
        .normal(size=width * trips * packets)
        .astype(np.float32)
    )
    if population == "magnitude":
        values[::3] *= np.float32(2**20)
        values[1::3] *= np.float32(2**-20)
        values[-1] = -0.0
    elif population == "nonfinite":
        values[:3] = [np.nan, np.inf, -np.inf]
    actual, log, cache = _execute(transformed, values)
    _, original_log, original_cache = _execute(ast.parse(source), values)
    assert log == original_log
    np.testing.assert_array_equal(cache, original_cache)
    partials = np.zeros(width, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for packet in range(trips * packets):
            for lane in range(width):
                value = (
                    np.float32(0)
                    if padded and packet == trips * packets - 1
                    else values[packet * width + lane]
                )
                partials[lane] = np.float32(partials[lane] + np.float32(value * value))
        while len(partials) > 1:
            half = len(partials) // 2
            partials = np.add(partials[:half], partials[half:], dtype=np.float32)
    if np.isnan(partials[0]):
        assert np.isnan(actual)
    else:
        assert np.float32(actual).tobytes() == partials[0].tobytes()
    if padded:
        # Invalid packet redirects to valid input, caches real bits, then the
        # original typed-zero conditional excludes them from the reduction.
        np.testing.assert_array_equal(cache[-width:], values[:width].view(np.uint32))


@pytest.mark.parametrize(
    "before,after",
    [
        ("acc = cutlass.Float32(0)", "acc = cutlass.Float32(-0.0)"),
        ("cutlass.Int32(32)", "cutlass.Int32(unknown)"),
        ("range_constexpr(4)", "range_constexpr(3)"),
        ("result = acc", "result = lane"),
        ("result = acc", "result = product"),
        ("acc = cutlass.Float32(0)", "alias = cache\nacc = cutlass.Float32(0)"),
        ("base = offset + packet * 4", "base = base + 1"),
        ("base = offset + packet * 4", "base = acc + packet * 4"),
        ("        for lane", "        observe(cache)\n        for lane"),
        ("        for lane", "        acc = 1\n        for lane"),
        ("cache[offset + packet * 4 + 0]", "cache[product]"),
        ("        for lane", "        xs = opaque\n        for lane"),
        ("        for lane", "        mask = opaque\n        for lane"),
        ("product = value * value", "product = product + value"),
        ("product = value * value", "product = opaque(value)"),
        ("product = value * value", "product = acc * value"),
        ("product = value * value", "product = cache[lane] * value"),
        (
            "acc = acc + cutlass.Float32(product)",
            "acc = acc * cutlass.Float32(product)",
        ),
        (
            "acc = acc + cutlass.Float32(product)",
            "acc = acc + cutlass.Float16(product)",
        ),
        ("else cutlass.Float32(0)", "else cutlass.Float32(-0.0)"),
        ("else cutlass.Float32(0)", "else cutlass.Float16(0)"),
        ("else cutlass.Float32(0)", "else external.load()"),
        ("        for lane", "        cute.arch.sync_threads()\n        for lane"),
        (
            "        for lane",
            "        if flag:\n            continue\n        for lane",
        ),
        ("cutlass.Uint32.mlir_type", "cutlass.Float32.mlir_type"),
        ("VectorType.get([4]", "VectorType.get([4.0]"),
    ],
)
def test_uncertain_carried_packets_decline(before: str, after: str) -> None:
    source = _source(padded=True).replace(before, after)
    assert source != _source(padded=True)
    assert ast.dump(_transform(source)) == ast.dump(ast.parse(source))


@pytest.mark.parametrize(
    "header,expected",
    [
        ("range(cutlass.Int32(0), cutlass.Int32(32), cutlass.Int32(4))", 8),
        ("range(cutlass.Int64(0), cutlass.Int64(32), cutlass.Int64(4))", 8),
        ("range(2147483646, 2147483647, 2)", None),
        ("range(cutlass.Int32(4294967296), 8, 1)", None),
        ("range(0, 1267650600228229401496703205376, 1)", None),
        ("range(-4, 4)", None),
        ("range(0, 4, -1)", None),
        ("range(0, 4, 0)", None),
        ("range(0, unknown, 4)", None),
    ],
)
def test_bounded_ranges_reject_wrapping_or_unknown(
    header: str, expected: int | None
) -> None:
    loop = ast.parse(f"for i in {header}:\n    pass").body[0]
    assert isinstance(loop, ast.For)
    assert _bounded_count(loop, {}) == expected


def test_shadowed_constant_does_not_bound_wrapper() -> None:
    source = "EXTENT = 64\n" + _source().replace(
        "cutlass.Int32(32)", "cutlass.Int32(EXTENT)"
    )
    assert ast.dump(_transform(source)) == ast.dump(ast.parse(source))


def test_disabled_path_and_wrapper_headers_are_unchanged() -> None:
    source = _source()
    assert ast.dump(_transform(source, enabled=False)) == ast.dump(ast.parse(source))
    original = ast.parse(source).body[2]
    transformed = next(n for n in _transform(source).body if isinstance(n, ast.For))
    assert isinstance(original, ast.For)
    assert ast.dump(original.iter) == ast.dump(transformed.iter)
    assert isinstance(original.body[-1], ast.For) and isinstance(
        transformed.body[-1], ast.For
    )
    assert ast.dump(original.body[-1].iter) == ast.dump(transformed.body[-1].iter)
    assert [ast.dump(n) for n in original.body[-1].body[:-1]] == [
        ast.dump(n)
        for n in transformed.body[-1].body[: len(original.body[-1].body) - 1]
    ]


def test_signed_contributions_and_cancellation_use_only_new_sum_tree() -> None:
    source = _source().replace("product = value * value", "product = value")
    values = np.array(
        [1e20, 1, -1e20, -0.0, -1e20, -1, 1e20, 0.0] * 4, dtype=np.float32
    )
    actual, log, cache = _execute(_transform(source), values)
    _, old_log, old_cache = _execute(ast.parse(source), values)
    assert log == old_log
    np.testing.assert_array_equal(cache, old_cache)
    sums = np.zeros(4, dtype=np.float32)
    for packet in values.reshape(8, 4):
        sums = np.add(sums, packet, dtype=np.float32)
    expected = np.float32(np.float32(sums[0] + sums[2]) + np.float32(sums[1] + sums[3]))
    assert np.float32(actual).tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "rows,width,dtype,threads,chunk,packet",
    [
        (1024, 4096, torch.float32, 128, 512, 4),
        (16, 7168, torch.bfloat16, 1024, 8192, 8),
        (16, 7168, torch.float16, 1024, 8192, 8),
        (16, 7168, torch.float32, 1024, 8192, 4),
    ],
)
def test_actual_carried_and_padded_codegen(
    rows: int, width: int, dtype: torch.dtype, threads: int, chunk: int, packet: int
) -> None:
    initialized = torch.cuda.is_initialized()
    with (
        _cpu_target(),
        _forbid_native_compile() as native,
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 3)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
    ):
        bound = helion.kernel(
            rms_norm, backend="cute", static_shapes=True
        )._bind_isolated(
            (
                torch.empty((rows, width), dtype=dtype),
                torch.empty(width, dtype=dtype),
                1e-6,
            )
        )
        generation = ConfigGeneration(bound.config_spec)
        codes = []
        for independent in (False, True):
            config = helion.Config(
                block_sizes=[1],
                num_threads=[1, threads],
                reduction_loops=[chunk],
                cute_vector_widths=[packet, 1],
                cute_lane_layouts=["strided", "blocked"],
                cute_reduction_reloads=["register"],
                cute_independent_reduction=independent,
                cute_replicated_reduction=False,
                cute_pdl=True,
            )
            with bound.env:
                flat, effective = generation.strict_config_pair(config)
                assert generation.strict_unflatten(flat) == effective
            codes.append(ast.parse(bound.to_code(effective)))
        before, after = codes
        sums = [
            n
            for n in ast.walk(after)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id.startswith("_helion_packet_sum_")
            and isinstance(n.value, ast.Call)
        ]
        assert len(sums) == packet
        # Raw cache stores and complete global-load expressions (including
        # redirected padded packet predicates) stay at their original addresses.

        def memory(tree: ast.AST) -> list[str]:
            return [
                ast.dump(n)
                for n in ast.walk(tree)
                if (
                    isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Subscript)
                )
                or (isinstance(n, ast.Call) and ast.unparse(n.func) == "cute.arch.load")
            ]

        assert memory(before) and memory(before) == memory(after)
        old_headers = [
            ast.dump(n.iter) for n in ast.walk(before) if isinstance(n, ast.For)
        ]
        new_headers = [
            ast.dump(n.iter) for n in ast.walk(after) if isinstance(n, ast.For)
        ]
        assert len(old_headers) == len(new_headers) + 1
        for n in native:
            n.assert_not_called()
    assert torch.cuda.is_initialized() == initialized


def test_shadowed_range_is_not_an_exact_trip_count() -> None:
    source = "range = custom_range\n" + _source()
    assert ast.dump(_transform(source)) == ast.dump(ast.parse(source))


@pytest.mark.parametrize("where", ["outer", "packet"])
def test_prior_lane_binding_read_in_prefix_declines(where: str) -> None:
    source = "lane = 1\n" + _source()
    if where == "outer":
        source = source.replace(
            "    for packet", "    prior_lane = lane\n    for packet"
        )
    else:
        source = source.replace(
            "base = offset + packet * 4", "base = offset + packet * 4 + lane * 0"
        )
    # The original traversal overwrites lane. Expanding it without assigning
    # lane could change this next-packet live-in, even with a dominating initial
    # binding. Whole-body read counts must reject it before scalar admission.
    assert ast.dump(_transform(source)) == ast.dump(ast.parse(source))
