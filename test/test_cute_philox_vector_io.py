from __future__ import annotations

import ast
import itertools
import operator
from types import SimpleNamespace

import pytest
import torch

from test.test_cute_affine_vector_io import _Float32
from test.test_cute_affine_vector_io import _Pointer
from test.test_cute_affine_vector_io import _Tensor
from test.test_cute_affine_vector_io import _Uint32
from test.test_cute_affine_vector_io import _vector_load
from test.test_cute_affine_vector_io import _vector_store
from test.test_cute_philox_stream import _f32
from test.test_cute_philox_stream import _interpret
from test.test_cute_philox_stream import _oracle
from test.test_cute_philox_stream import _program
from test.test_cute_philox_stream import _signed
from test.test_cute_philox_stream import cpu_only  # noqa: F401

import helion
from helion._compiler.cute.affine_vector_io import _Vectorizer
from helion._compiler.cute.philox_packets import lower_philox_packets
from helion._compiler.cute.philox_stream import PACKET_ASM
from helion._compiler.cute.philox_stream import SCALAR_ASM
from helion._compiler.cute.philox_stream import SCALAR_CONSTRAINTS
from helion._testing import skipUnlessBackends


def _source(width=4, predicate="index < length", extra=""):
    return f"""def kernel(src, out, seed_tensor, start, length, p, scale, other):
    for vec_lane_0 in cutlass.range_constexpr({width}):
        index = cutlass.Int32(start) + cutlass.Int32(vec_lane_0)
        mask = {predicate}
        value = (src.iterator + cutlass.Int32(index) * cutlass.Int32(src.layout.stride[0])).load() if mask else cutlass.Float32(0)
        offset = cutlass.Int64(index)
        random = _cute_inline_asm_elementwise((seed_tensor[0], offset), asm={SCALAR_ASM!r}, constraints={SCALAR_CONSTRAINTS!r}, dtype=cutlass.Float32, is_pure=True)
        {extra or "unused = 0"}
        keep = operator.gt(random, p)
        result = cutlass.Float32(value * scale) if keep else cutlass.Float32(0)
        if mask:
            (out.iterator + cutlass.Int32(index) * cutlass.Int32(out.layout.stride[0])).store(result)
"""


def _rewrite(source, *, aligned=True, disjoint=True, stride=1):
    tree = ast.parse(source)
    changed = []
    names = itertools.count()

    def vectorize(loop, read, known):
        rewriter = _Vectorizer(
            [loop],
            strides={("src", 0): stride, ("out", 0): stride},
            alignments={"src": 16 if aligned else 1, "out": 16},
            dtypes={"src": "cutlass.Float32", "out": "cutlass.Float32"},
            disjoint={frozenset(("src", "out"))} if disjoint else set(),
            constexpr={},
            integer_parameters=known,
            register_accesses=frozenset({ast.dump(read, include_attributes=False)}),
        )
        result = rewriter.body([loop])
        changed.append(rewriter.changed)
        return result

    tree.body[0].body = lower_philox_packets(
        tree.body[0].body,
        seed_names={"seed_tensor"},
        integer_names={"start", "length"},
        new_name=lambda prefix: f"{prefix}_{next(names)}",
        vectorize_packet=vectorize,
    )
    return ast.fix_missing_locations(tree), sum(changed)


def _run(tree, *, start=0, length=4, seed=123, p=0.25, scale=4 / 3, stride=1):
    stats = dict.fromkeys(
        (
            "scalar_loads",
            "scalar_stores",
            "vector_loads",
            "vector_stores",
            "packet",
            "scalar",
        ),
        0,
    )
    data = [float(index - 11) for index in range(64)]
    output = [-123.0] * 64

    def asm(args, **kwargs):
        packet = kwargs["asm"] == PACKET_ASM
        stats["packet" if packet else "scalar"] += 1
        values = _interpret(_program(kwargs["asm"]), *args, 4 if packet else 1)
        return values if packet else values[0]

    namespace = {
        "operator": operator,
        "cutlass": SimpleNamespace(
            Int32=lambda value: _signed(value, 32),
            Int64=lambda value: _signed(value, 64),
            Float32=_Float32,
            Uint32=_Uint32,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(arch=SimpleNamespace(load=_vector_load)),
        "ir": SimpleNamespace(
            VectorType=SimpleNamespace(get=lambda shape, dtype: (shape, dtype))
        ),
        "_cute_store_u32_vec": _vector_store,
        "_cute_inline_asm_elementwise": asm,
    }
    _Uint32.mlir_type = "i32"
    exec(compile(tree, "<philox-vector-proof>", "exec"), namespace)
    src = _Tensor(
        iterator=_Pointer(data, stats), layout=SimpleNamespace(stride=(stride,))
    )
    out = _Tensor(
        iterator=_Pointer(output, stats), layout=SimpleNamespace(stride=(stride,))
    )
    namespace["kernel"](src, out, [seed], start, length, p, scale, [2])
    # The imported pointer fixture stores Python floats. Model the actual
    # FP32 output storage on both paths; this corpus never rereads the output.
    return [_f32(value) for value in output], stats


@pytest.mark.parametrize("width", (4, 8))
def test_packet_vector_memory_and_exact_new_stream_with_tail_and_probability(width):
    source = _source(width)
    tree, changed = _rewrite(source)
    assert changed == 1
    for start in (0, 1, 4):
        for length in (0, 1, 3, 4, 5, 7, 8, 13):
            for seed in (123, -(2**63)):
                for p, scale in ((0.25, 4 / 3), (0.0, 1.0), (0.75, 4.0)):
                    args = {
                        "start": start,
                        "length": length,
                        "seed": seed,
                        "p": p,
                        "scale": scale,
                    }
                    expected, _ = _run(ast.parse(source), **args)
                    actual, stats = _run(tree, **args)
                    assert actual == expected
                    for i in range(start, min(start + width, length)):
                        assert actual[i] == (
                            _f32((i - 11) * scale) if _oracle(seed, i) > p else 0.0
                        )
                    vectors = (width // 4) * int(
                        start % width == 0 and start + width <= length
                    )
                    assert stats["vector_loads"] == stats["vector_stores"] == vectors
                    assert stats["packet"] == (width // 4 if start % 4 == 0 else 0)


@pytest.mark.parametrize(
    "predicate", ("index < length and index != 2", "index < length and index % 4 != 1")
)
def test_packet_hole_masks_do_not_speculate_vector_memory(predicate):
    source = _source(predicate=predicate)
    tree, changed = _rewrite(source)
    assert changed == 1
    expected, _ = _run(ast.parse(source))
    actual, stats = _run(tree)
    assert actual == expected
    assert stats["packet"] == 1 and stats["vector_loads"] == stats["vector_stores"] == 0


@pytest.mark.parametrize(
    "options", ({"aligned": False}, {"disjoint": False}, {"stride": 2})
)
def test_unproved_memory_keeps_packet_but_scalar_io(options):
    source = _source()
    tree, changed = _rewrite(source, **options)
    assert changed == 0
    args = {"stride": options.get("stride", 1)}
    expected, _ = _run(ast.parse(source), **args)
    actual, stats = _run(tree, **args)
    assert actual == expected
    assert stats["packet"] == 1 and stats["scalar_loads"] == stats["scalar_stores"] == 4


@pytest.mark.parametrize(
    "extra",
    (
        "unknown = other[0]",
        "unknown = observer(value)",
        "unknown = _cute_inline_asm_elementwise((value,), asm='opaque', constraints='=f,f', dtype=cutlass.Float32, is_pure=True)",
    ),
)
def test_packet_tuple_grant_does_not_allow_other_reads_or_calls(extra):
    tree, changed = _rewrite(_source(extra=extra))
    assert "philox_packet" in ast.unparse(tree)
    assert changed == 0 and "cute.arch.load" not in ast.unparse(tree)


@pytest.mark.parametrize(
    "geometry", ((1024, 128, 4), (512, 128, 4), (1024, 256, 4), (1024, 128, 8))
)
@pytest.mark.usefixtures("cpu_only")
@skipUnlessBackends(["cute"])
def test_original_dynamic_cpu_emission_composes_packet_and_vector_io(geometry):
    from examples.low_mem_dropout import low_mem_dropout

    block, threads, width = geometry
    bound = helion.kernel(
        low_mem_dropout.fn,
        backend="cute",
        static_shapes=False,
        cute_rng_stream="philox4",
    )._bind_isolated((0.25, torch.empty(1025), -(2**63)))
    config = helion.Config(
        block_sizes=[block],
        num_threads=[threads],
        cute_vector_widths=[width],
        cute_lane_layouts=["blocked"],
        cute_rng_packet=True,
    )
    source = bound.to_code(config)
    assert "cute.arch.load(" in source and "_cute_store_u32_vec(" in source
    assert "philox_packet_" in source and "_cute_stage_explicit_seed" in source
    assert SCALAR_ASM in [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    expected_config = dict(config.config)
    bound.config_spec.normalize(config)
    assert config.config == expected_config | {"cute_proven_bounds": False}
