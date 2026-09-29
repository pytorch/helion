from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute.vector_reduction_packets import optimize_vector_reductions
from helion.language.memory_ops import _cute_unroll_vec_load_expr


def _transform(source: str, *, unroll_packets: bool = False) -> ast.Module:
    tree = ast.parse(source)
    result = ast.Module(
        body=optimize_vector_reductions(
            tree.body,
            {"N": 4096, "BS": 4096},
            thread_block_dims=(256, 1, 1),
            independent_accumulators=True,
            replicated_single_use=True,
            unroll_packets=unroll_packets,
        ),
        type_ignores=[],
    )
    return ast.fix_missing_locations(result)


def _sum_source(width: int = 4, packets: int = 4) -> str:
    return f"""acc = cutlass.Float32(0)
for packet in range({packets}):
    xs = _cute_load_l1_l2_evict_last_8b(x[packet], vector_type)
    ys = _cute_load_l1_l2_evict_last_8b(y[packet], vector_type)
    for lane in cutlass.range_constexpr({width}):
        a = cutlass.Float16(xs[lane])
        b = cutlass.Float16(ys[lane])
        product = a * b
        acc = acc + cutlass.Float32(product)
"""


@pytest.mark.parametrize("width", (2, 4, 8))
@pytest.mark.parametrize("packets", (1, 2, 4, 8))
def test_independent_sums_preserve_product_rounding(width: int, packets: int) -> None:
    tree = _transform(_sum_source(width, packets))
    assert "_helion_packet_sum" in ast.unparse(tree)
    rng = np.random.default_rng(731)
    x = rng.normal(size=(packets, width)).astype(np.float16)
    y = rng.normal(size=(packets, width)).astype(np.float16)
    namespace = {
        "x": x,
        "y": y,
        "vector_type": None,
        "_cute_load_l1_l2_evict_last_8b": lambda ptr, dtype: ptr,
        "cutlass": SimpleNamespace(
            Float16=np.float16, Float32=np.float32, range_constexpr=range
        ),
    }
    exec(compile(tree, "<independent sums>", "exec"), namespace)
    partials = [np.float32(0)] * width
    for packet in range(packets):
        for lane in range(width):
            partials[lane] = np.float32(
                partials[lane]
                + np.float32(np.float16(x[packet, lane] * y[packet, lane]))
            )
    while len(partials) > 1:
        half = len(partials) // 2
        partials = [np.float32(partials[i] + partials[i + half]) for i in range(half)]
    assert namespace["acc"].tobytes() == partials[0].tobytes()


@pytest.mark.parametrize(
    ("before", "after"),
    (
        ("Float32(0)", "Float32(1)"),
        ("Float32(0)", "Float32(-0.0)"),
        ("range(4)", "range(n)"),
        ("range_constexpr(4)", "range_constexpr(3)"),
        ("a * b", "random_value()"),
        ("a * b", "product + a * b"),
        ("a * b", "opaque.bitcast(cutlass.Float16)"),
        ("a * b", "opaque.value"),
        ("a * b", "external[lane]"),
        ("a = cutlass.Float16(xs[lane])", "a = b"),
        ("product = a * b", "lane = a * b"),
        ("a * b", "a * b + acc"),
        ("acc + cutlass.Float32(product)", "acc * cutlass.Float32(product)"),
        ("for packet in", "for acc in"),
        ("for lane in", "for acc in"),
        ("x[packet]", "x[packet] + product"),
    ),
)
def test_uncertain_or_loop_carried_expressions_stay_unchanged(
    before: str, after: str
) -> None:
    source = _sum_source().replace(before, after)
    assert ast.dump(_transform(source)) == ast.dump(ast.parse(source))


def test_inner_index_cannot_escape_through_an_ancestor_loop() -> None:
    source = (
        "for tile in range(1):\n"
        + "\n".join("    " + line for line in _sum_source().splitlines())
        + "\nresult = lane\n"
    )
    assert ast.dump(_transform(source)) == ast.dump(ast.parse(source))


def test_vector_provenance_uses_the_last_dominating_assignment() -> None:
    source = _sum_source().replace(
        "    for lane in", "    xs = opaque\n    for lane in"
    )
    assert ast.dump(_transform(source)) == ast.dump(ast.parse(source))


def _cta_source() -> str:
    return """lane = cutlass.Int32(cute.arch.thread_idx()[0])
lane_in_group = lane % 256
lane_mod_pre = lane_in_group % 1
result = _cute_grouped_reduce_shared_two_stage(value, 'sum', cutlass.Float32(0), lane, lane_in_group, lane_mod_pre, pre=1, group_span=256, group_count=1)
"""


def test_single_use_reduction_and_extended_ast() -> None:
    source = _cta_source()
    output = ast.unparse(_transform(source))
    assert output.count("cute.arch.sync_threads()") == 1
    assert "threads_in_group=8" in output
    body = [statement_from_string(line) for line in source.splitlines()]
    transformed = optimize_vector_reductions(
        body,
        {},
        thread_block_dims=(256, 1, 1),
        independent_accumulators=True,
        replicated_single_use=True,
    )
    assert ast.dump(ast.Module(body=transformed, type_ignores=[])) == ast.dump(
        _transform(source)
    )


@pytest.mark.parametrize(
    "scope", ("for tile in range(2):", "if guard:", "while guard:")
)
def test_replicated_reduction_requires_converged_single_use_scratch(scope: str) -> None:
    source = (
        scope + "\n" + "\n".join("    " + line for line in _cta_source().splitlines())
    )
    assert ast.dump(_transform(source)) == ast.dump(ast.parse(source))


@pytest.mark.parametrize(
    ("before", "after"),
    (
        ("cute.arch.thread_idx()[0]", "cute.arch.thread_idx()[0] ^ 1"),
        ("lane % 256", "lane % 128"),
        ("lane_mod_pre, pre=1", "other, pre=1"),
        ("group_count=1", "group_count=2"),
        ("Float32(0)", "Float16(0)"),
    ),
)
def test_logical_lane_or_unsupported_group_cannot_use_physical_subgroups(
    before: str, after: str
) -> None:
    source = _cta_source().replace(before, after)
    assert ast.dump(_transform(source)) == ast.dump(ast.parse(source))


def test_vector_packet_unrolling_excludes_protocols() -> None:
    source = """for packet in range(4):
    xs = _cute_load_l1_l2_evict_last_8b(ptr, vector_type)
    values = []
    for lane in cutlass.range_constexpr(4):
        values.append(cutlass.Float16(xs[lane]))
    _cute_store_u16_vec(out, values)
"""
    assert "for packet in cutlass.range_constexpr(4)" in ast.unparse(
        _transform(source, unroll_packets=True)
    )
    protocol = source.replace(
        "    values = []", "    cute.arch.sync_threads()\n    values = []"
    )
    assert ast.dump(_transform(protocol, unroll_packets=True)) == ast.dump(
        ast.parse(protocol)
    )


def test_eight_byte_hints_preserve_width_and_other_load_policies() -> None:
    for dtype, width in ((torch.float16, 4), (torch.bfloat16, 4), (torch.float32, 2)):
        for policy in ("first", "last"):
            code = _cute_unroll_vec_load_expr(
                "ptr", dtype, width, f"__l1_l2_{policy}__"
            )
            assert f"_cute_load_l1_l2_evict_{policy}_8b" in code
            assert f"[{width}]" in code
        assert "cute.arch.load" in _cute_unroll_vec_load_expr(
            "ptr", dtype, width, "__l2_last__"
        )


def test_cache_helpers_build_matching_eight_and_sixteen_byte_ir() -> None:
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func

    from helion._compiler.cute import l2_policy

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            for width in (4, 8):
                for policy in ("first", "last"):
                    vector_type = ir.VectorType.get([width], cutlass.Uint16.mlir_type)
                    function = func.FuncOp(
                        f"load_{width}_{policy}",
                        ir.FunctionType.get([cutlass.Uint64.mlir_type], [vector_type]),
                    )
                    block = function.add_entry_block()
                    with ir.InsertionPoint(block):

                        class Address:
                            def __init__(self, value):
                                self.value = value

                            def toint(self, **kwargs):
                                return self

                            def ir_value(self, **kwargs):
                                return self.value

                        helpers = {
                            (4, "first"): l2_policy.load_v8b_l1_l2_evict_first,
                            (4, "last"): l2_policy.load_v8b_l1_l2_evict_last,
                            (8, "first"): l2_policy.load_v16b_l1_l2_evict_first,
                            (8, "last"): l2_policy.load_v16b_l1_l2_evict_last,
                        }
                        value = helpers[width, policy](
                            Address(block.arguments[0]), vector_type
                        )
                        assert value.type == vector_type
                        func.ReturnOp([value])
        module.operation.verify()
        text = str(module)
        for words in (2, 4):
            for policy in ("first", "last"):
                assert (
                    f"ld.global.L1::evict_{policy}.L2::cache_hint.v{words}.b32" in text
                )
        assert text.count("llvm.inline_asm has_side_effects") == 4
