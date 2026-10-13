from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest

from helion._compiler.cute import chained_fill_ownership as module
from helion.autotuner import config_spec

Read, choose = module.Read, module.choose_striped


def fixture(rows=128, columns=16, threads=128, vector=8, source_stride=None):
    read = Read(
        "shared",
        (f"i // {columns}", f"i % {columns}"),
        (columns if source_stride is None else source_stride, 1),
        4,
        True,
    )
    return {
        "lines": [
            f"x = shared[i // {columns}, i % {columns}]",
            "y = cutlass.BFloat16(x)",
        ],
        "value": "y",
        "bounds": [],
        "reads": [read],
        "index": "i",
        "origins": (),
        "shape": (columns, rows),
        "stride": (1, columns + 8),
        "threads": threads,
        "vector": vector,
    }


@pytest.mark.parametrize(
    "rows,columns,threads,vector",
    [(64, 16, 64, 4), (128, 16, 128, 8), (128, 32, 256, 8), (64, 32, 128, 2)],
)
def test_dense_read_reaches_ideal_with_varied_shape_and_team(
    rows, columns, threads, vector
):
    assert choose(**fixture(rows, columns, threads, vector))


def test_padded_result_improvement_is_not_ideal_and_remains_original():
    assert not choose(**fixture(source_stride=20))


@pytest.mark.parametrize(
    "lines",
    [
        ["x = shared[i // 16, i % 16]", "y = unknown(x)"],
        ["x = shared[i // 16, i % 16]", "x = x + 1", "y = cutlass.BFloat16(x)"],
        ["x = shared[i // 16, i % 16]", "shared[0, 0] = x", "y = cutlass.BFloat16(x)"],
        ["x = shared[i // 16, i % 16]", "y = cutlass.BFloat16(x) + chain_thread"],
        ["x = shared[i // 16, i % 16]", "y = cute.arch.shuffle_sync(x, 0)"],
        ["x = shared[i // 16, i % 16]", "y = cutlass.Int64(x)"],
        [
            "x = shared[i // 16, i % 16]",
            "y = cutlass.BFloat16(x) if x > 0 else cutlass.BFloat16(0)",
        ],
        [
            "x = shared[i // 16, i % 16]",
            "y = cutlass.BFloat16(x) if chain_thread < 4 else cutlass.BFloat16(0)",
        ],
        ["x = shared[i // 16, i % 16]", "y = cutlass.BFloat16(x + captured_register)"],
        ["for j in range(2):\n    x = shared[j, i]", "y = cutlass.BFloat16(x)"],
        ["x = shared[i // 16, i % 16]", "y = cutlass.BFloat16(x)", "escape(y)"],
    ],
)
def test_effect_rebinding_fragment_and_predicate_declines(lines):
    case = fixture()
    case["lines"] = lines
    assert not choose(**case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("threads", 0),
        ("threads", 33),
        ("shape", (15, 128)),
        ("shape", (False, 128)),
        ("stride", (1, 1)),
        ("vector", 16),
    ],
)
def test_invalid_or_partial_workgroups_decline(field, value):
    case = fixture()
    case[field] = value
    assert not choose(**case)


@pytest.mark.parametrize("strides", [(0, 1), (-16, 1), (16, -1)])
def test_unknown_zero_or_negative_physical_stride_declines(strides):
    case = fixture()
    case["reads"] = [replace(case["reads"][0], strides=strides)]
    assert not choose(**case)


def test_memory_derived_address_declines():
    case = fixture()
    case["lines"].insert(0, "j = shared[0, i % 16]")
    case["reads"] = [replace(case["reads"][0], indices=("j", "i % 16"))]
    assert not choose(**case)


def test_uniform_origin_is_preserved_but_mixed_modulo_declines():
    case = fixture()
    case["origins"] = ("origin",)
    case["reads"] = [replace(case["reads"][0], indices=("origin + i // 16", "i % 16"))]
    assert choose(**case)
    case["reads"] = [
        replace(case["reads"][0], indices=("(origin + i) // 16", "i % 16"))
    ]
    assert not choose(**case)


def test_extra_read_regression_declines():
    case = fixture()
    case["reads"].append(Read("second", ("i // 16", "i % 16"), (1, 128), 4, True))
    assert not choose(**case)


def test_actual_source_selects_exactly_the_three_measured_fills():
    source = GENERATED_FILLS
    tree = ast.parse(source)
    selected = []
    for stage, count, rows, kind in (
        (0, 2, 128, "carry"),
        (1, 1, 64, "residual"),
        (2, 2, 128, "carry"),
        (3, 1, 64, "result"),
        (4, 1, 64, "result"),
        (5, 1, 64, "result"),
    ):
        loop = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id == f"chain_{stage}_load_step"
            and ast.unparse(n.iter) == f"cutlass.range({count}, unroll=1)"
        )
        body = loop.body[0].body[1].body
        lines = list(map(ast.unparse, body[:-1]))
        value = ast.unparse(body[-1].value.args[0])
        index = f"chain_{stage}_load"
        reads = []
        for statement in body:
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id
                    in ("chain_loop_carry_0", "chain_0_c", "chain_1_c")
                ):
                    stride = 16 if node.value.id == "chain_loop_carry_0" else 20
                    reads.append(
                        Read(
                            node.value.id,
                            tuple(map(ast.unparse, node.slice.elts)),
                            (stride, 1),
                            4,
                            True,
                        )
                    )
        if kind == "residual":
            reads.extend(
                [
                    Read(
                        "input_tensor_3",
                        (
                            "chain_origin_0",
                            "chain_origin_1",
                            "chain_loop_index",
                            f"{index} // 16",
                        ),
                        (65536, 2048, 64, 1),
                        2,
                        False,
                    ),
                    Read(
                        "input_tensor_4",
                        (
                            "chain_origin_0",
                            "chain_value_111",
                            "chain_origin_1",
                            f"chain_origin_2 + {index} % 16",
                        ),
                        (8388608, 4096, 128, 1),
                        2,
                        False,
                    ),
                ]
            )
        if choose(
            lines=lines,
            value=value,
            bounds=[],
            reads=reads,
            index=index,
            origins=(
                "chain_origin_0",
                "chain_origin_1",
                "chain_origin_2",
                "chain_loop_index",
            ),
            shape=(16, rows),
            stride=(1, 24),
            threads=128,
            vector=8,
        ):
            selected.append(stage)
    assert selected == [0, 1, 2]


def test_semantic_eligibility_precedes_cost_even_with_ideal_read():
    case = fixture()
    case["lines"].insert(0, "synchronize()")
    assert not choose(**case)


@pytest.mark.parametrize(
    "definitions",
    [
        ["a = a + 1"],
        ["a = b + 1", "b = a + 1"],
        ["a = b < 1", "b = a"],
    ],
)
def test_predicate_definition_graph_is_acyclic_by_ordered_admission(definitions):
    case = fixture()
    case["lines"] = [*definitions, *case["lines"]]
    assert not choose(**case)


@pytest.mark.parametrize(
    "mutation",
    [None, "alias", "blocked", "xor", "workspace", "fragment", "dtype", "rebound"],
)
def test_actual_adapter_retains_storage_and_alternate_owner_gates(
    monkeypatch, mutation
):
    from types import SimpleNamespace

    import torch

    from helion._compiler.cute import chained_fill_ownership
    from helion._compiler.cute import chained_loop
    from helion._compiler.cute import chained_matmul

    monkeypatch.setattr(
        chained_loop, "loop_storage_is_proven", lambda loop: mutation != "alias"
    )
    monkeypatch.setattr(
        chained_matmul, "_shape", lambda node: tuple(node.meta["val"].shape)
    )

    class Node:
        meta: ClassVar[dict[str, object]] = {
            "val": torch.empty((128, 16), dtype=torch.float32)
        }

    node = Node()
    loop = SimpleNamespace(boundaries=lambda: {node: "shared"})
    plan = SimpleNamespace(
        strategy="warp",
        loop=loop,
        preparation_pipeline=None,
        pointwise_cache=None,
        loop_workspace=None,
        warp_workspace=None,
        root_vector_cache=None,
        row_registers=None,
        row_parallel=None,
        mma_frontier=False,
        completed_image_bytes=0,
        dots=(),
        threads=128,
    )
    case = fixture()
    case["lines"] = [
        line.replace("[i // 16, i % 16]", "[chain_0_load // 16, chain_0_load % 16]")
        for line in case["lines"]
    ]
    expression = SimpleNamespace(
        boundaries={node: "shared"},
        scan_inputs=[],
        staged_inputs=[],
        fragment_images={},
        coordinate_normalization=None,
        global_accesses=[],
        lines=case["lines"],
        origins={},
    )
    layout = SimpleNamespace(
        dtype=torch.bfloat16, shape=case["shape"], stride=case["stride"]
    )
    config = {"cute_chained_fill_ownership": "auto"}
    if mutation == "blocked":
        config.clear()
    if mutation == "xor":
        config["cute_chained_scratch_layout"] = "xor"
    if mutation == "workspace":
        plan.loop_workspace = object()
    if mutation == "fragment":
        expression.fragment_images = {node: object()}
    if mutation == "dtype":
        layout.dtype = torch.float32
    if mutation == "rebound":
        expression.lines = [*expression.lines, "shared = x"]
    cg = SimpleNamespace(
        device_function=SimpleNamespace(config=SimpleNamespace(config=config))
    )
    assert chained_fill_ownership.select_fill(
        cg, plan, expression, layout, 0, 8, case["value"], [], 8
    ) == (mutation is None)


@pytest.mark.parametrize(
    "policy,backend,loop,schedule,expected",
    [
        ("blocked", "cute", True, "cp_async", "removed"),
        ("auto", "cute", True, "cp_async", "kept"),
        (True, "cute", True, "cp_async", "reject"),
        ("striped", "cute", True, "cp_async", "reject"),
        ("auto", "triton", True, "cp_async", "reject"),
        ("auto", "cute", False, "cp_async", "reject"),
        ("auto", "cute", True, "tcgen05_tmem", "kept"),
    ],
)
@pytest.mark.parametrize("preparation", [False, True])
def test_actual_normalizer_default_and_invalid_policy(
    policy, backend, loop, schedule, expected, preparation
):
    from types import SimpleNamespace

    tree = ast.parse(Path(config_spec.__file__).read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_normalize_cute_chained_fill_ownership"
    )

    class InvalidConfig(Exception):
        pass

    ns = {
        "CUTE_CHAINED_FILL_OWNERSHIP_KEY": "cute_chained_fill_ownership",
        "CUTE_CHAINED_MMA_SCHEDULE_KEY": "cute_chained_mma_schedule",
        "CUTE_CHAINED_PREPARATION_PIPELINE_KEY": "cute_chained_preparation_pipeline",
        "InvalidConfig": InvalidConfig,
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), "<actual-normalizer>", "exec"
        ),
        ns,
    )
    owner = SimpleNamespace(backend_name=backend, cute_chained_loop_search_enabled=loop)
    config = {
        "cute_chained_fill_ownership": policy,
        "cute_chained_mma_schedule": schedule,
        "cute_chained_preparation_pipeline": preparation,
    }
    if expected == "reject":
        with pytest.raises(InvalidConfig):
            ns[method.name](owner, config)
    else:
        ns[method.name](owner, config)
        assert ("cute_chained_fill_ownership" in config) == (expected == "kept")


# Exact AST-unparsed excerpts from the observed source39025e1a.
GENERATED_FILLS = "for chain_0_load_step in cutlass.range(2, unroll=1):\n    for chain_0_load_vec in cutlass.range_constexpr(8):\n        chain_0_load = chain_thread * 8 + chain_0_load_step * 1024 + chain_0_load_vec\n        if chain_0_load < 2048:\n            v_2 = chain_loop_carry_0[chain_0_load // 16, chain_0_load % 16] if 0 <= chain_0_load // 16 < 128 and 0 <= chain_0_load % 16 < 16 else cutlass.Float32(0)\n            v_3 = cutlass.BFloat16(v_2)\n            chain_value_109 = v_3\n            chain_0_b[chain_0_load % 16, chain_0_load // 16] = cutlass.BFloat16(chain_value_109)\nfor chain_1_load_step in cutlass.range(1, unroll=1):\n    for chain_1_load_vec in cutlass.range_constexpr(8):\n        chain_1_load = chain_thread * 8 + chain_1_load_step * 1024 + chain_1_load_vec\n        if chain_1_load < 1024:\n            v_18 = 0 + chain_1_load // 16 * 1\n            v_19 = chain_loop_index * 64\n            v_20 = cutlass.Int32(v_19)\n            v_21 = v_18 + v_20\n            chain_value_111 = v_21\n            v_22 = 2048\n            v_23 = operator.lt(chain_value_111, v_22)\n            chain_value_112 = v_23\n            chain_value_113 = (input_tensor_3.iterator + cutlass.Int32(chain_origin_0 // 1) * cutlass.Int32(input_tensor_3.layout.stride[0]) + cutlass.Int32(chain_origin_1 // 1) * cutlass.Int32(input_tensor_3.layout.stride[1]) + cutlass.Int32(chain_loop_index) * cutlass.Int32(input_tensor_3.layout.stride[2]) + cutlass.Int32(chain_1_load // 16) * cutlass.Int32(input_tensor_3.layout.stride[3])).load() if 0 <= chain_origin_0 // 1 < 1 and (0 <= chain_origin_1 // 1 < 32 and (0 <= chain_loop_index < 32 and 0 <= chain_1_load // 16 < 64)) else cutlass.BFloat16(0)\n            v_24 = cutlass.Float32(chain_value_113)\n            chain_value_114 = v_24\n            chain_value_115 = (input_tensor_4.iterator + cutlass.Int32(chain_origin_0 // 1) * cutlass.Int32(input_tensor_4.layout.stride[0]) + cutlass.Int32(chain_value_111) * cutlass.Int32(input_tensor_4.layout.stride[1]) + cutlass.Int32(chain_origin_1 // 1) * cutlass.Int32(input_tensor_4.layout.stride[2]) + cutlass.Int32(chain_origin_2 + chain_1_load % 16) * cutlass.Int32(input_tensor_4.layout.stride[3])).load() if 0 <= chain_origin_0 // 1 < 1 and 0 <= chain_value_111 < 2048 and (0 <= chain_origin_1 // 1 < 32) and (0 <= chain_origin_2 + chain_1_load % 16 < 128) and chain_value_112 else cutlass.BFloat16(0)\n            v_25 = cutlass.Float32(chain_value_115)\n            chain_value_116 = v_25\n            v_26 = chain_0_c[chain_1_load // 16, chain_1_load % 16] if 0 <= chain_1_load // 16 < 64 and 0 <= chain_1_load % 16 < 16 else cutlass.Float32(0)\n            v_27 = chain_value_116 - v_26\n            chain_value_117 = v_27\n            v_28 = chain_value_114 * chain_value_117\n            chain_value_118 = v_28\n            chain_value_119 = chain_value_118 if chain_value_112 else cutlass.Float32(0.0)\n            v_29 = cutlass.Float16(chain_value_119)\n            chain_value_120 = v_29\n            chain_1_b[chain_1_load % 16, chain_1_load // 16] = cutlass.Float16(chain_value_120)\nfor chain_2_load_step in cutlass.range(2, unroll=1):\n    for chain_2_load_vec in cutlass.range_constexpr(8):\n        chain_2_load = chain_thread * 8 + chain_2_load_step * 1024 + chain_2_load_vec\n        if chain_2_load < 2048:\n            v_45 = chain_loop_carry_0[chain_2_load // 16, chain_2_load % 16] if 0 <= chain_2_load // 16 < 128 and 0 <= chain_2_load % 16 < 16 else cutlass.Float32(0)\n            v_46 = cutlass.BFloat16(v_45)\n            chain_value_122 = v_46\n            chain_2_b[chain_2_load % 16, chain_2_load // 16] = cutlass.BFloat16(chain_value_122)\nfor chain_3_load_step in cutlass.range(1, unroll=1):\n    for chain_3_load_vec in cutlass.range_constexpr(8):\n        chain_3_load = chain_thread * 8 + chain_3_load_step * 1024 + chain_3_load_vec\n        if chain_3_load < 1024:\n            v_51 = chain_1_c[chain_3_load // 16, chain_3_load % 16] if 0 <= chain_3_load // 16 < 64 and 0 <= chain_3_load % 16 < 16 else cutlass.Float32(0)\n            v_52 = cutlass.BFloat16(v_51)\n            chain_value_124 = v_52\n            chain_3_b[chain_3_load % 16, chain_3_load // 16] = cutlass.BFloat16(chain_value_124)\nfor chain_4_load_step in cutlass.range(1, unroll=1):\n    for chain_4_load_vec in cutlass.range_constexpr(8):\n        chain_4_load = chain_thread * 8 + chain_4_load_step * 1024 + chain_4_load_vec\n        if chain_4_load < 1024:\n            v_69 = chain_1_c[chain_4_load // 16, chain_4_load % 16] if 0 <= chain_4_load // 16 < 64 and 0 <= chain_4_load % 16 < 16 else cutlass.Float32(0)\n            v_70 = cutlass.BFloat16(v_69)\n            chain_value_126 = v_70\n            chain_4_b[chain_4_load % 16, chain_4_load // 16] = cutlass.BFloat16(chain_value_126)\nfor chain_5_load_step in cutlass.range(1, unroll=1):\n    for chain_5_load_vec in cutlass.range_constexpr(8):\n        chain_5_load = chain_thread * 8 + chain_5_load_step * 1024 + chain_5_load_vec\n        if chain_5_load < 1024:\n            v_93 = chain_1_c[chain_5_load // 16, chain_5_load % 16] if 0 <= chain_5_load // 16 < 64 and 0 <= chain_5_load % 16 < 16 else cutlass.Float32(0)\n            v_94 = cutlass.BFloat16(v_93)\n            chain_value_130 = v_94\n            v_95 = cutlass.Float32(chain_value_130)\n            chain_value_131 = v_95\n            v_96 = chain_1_c[chain_5_load // 16, chain_5_load % 16] if 0 <= chain_5_load // 16 < 64 and 0 <= chain_5_load % 16 < 16 else cutlass.Float32(0)\n            v_97 = v_96 - chain_value_131\n            chain_value_132 = v_97\n            v_98 = cutlass.BFloat16(chain_value_132)\n            chain_value_133 = v_98\n            chain_5_b[chain_5_load % 16, chain_5_load // 16] = cutlass.BFloat16(chain_value_133)"
