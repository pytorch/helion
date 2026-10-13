from __future__ import annotations

import ast
import itertools
from textwrap import indent
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_fuser_inherited_context import _config
from .test_cute_fuser_occurrences import _args
from .test_cute_fuser_occurrences import _state_reduction
from helion._compiler.cute.computed_sweep_cache import MAX_CACHE_ELEMENTS
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads


def _body(
    *,
    middle: str = "",
    first: str = "cutlass.Float32(raw) * cutlass.Float32(2)",
    second: str | None = None,
    before: str = "",
    load: str = "(x.iterator + synthetic_lane_7).load()",
    suffix: str = "",
    count: int = 8,
) -> list[ast.stmt]:
    return ast.parse(
        f"""
{before}
total = cutlass.Float32(0)
for synthetic_lane_7 in range({count}):
    raw = {load}
    current = {first}
    total = total + current
{middle}
for synthetic_lane_7 in range({count}):
    raw = {load}
    current = {first if second is None else second}
    (out.iterator + synthetic_lane_7).store(current + total)
{suffix}
"""
    ).body


def _run(body: list[ast.stmt], mode: str = "computed", **kwargs: Any) -> str:
    options: dict[str, Any] = {
        "tensor_dtypes": dict.fromkeys(("x", "out", "other"), "cutlass.Float32"),
        "reload_modes": {7: mode},
        "proven_disjoint_tensor_pairs": {
            frozenset(pair) for pair in itertools.combinations(("x", "out", "other"), 2)
        },
        **kwargs,
    }
    return ast.unparse(
        ast.Module(body=fuse_two_pass_loads(body, **options), type_ignores=[])
    )


def _declarations(source: str) -> list[ast.Assign]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
    ]


def test_complete_result_cached_and_original_reduction_store_order_retained() -> None:
    source = _run(_body())
    assert len(_declarations(source)) == 1
    assert source.count(".load()") == 1
    loops = [node for node in ast.parse(source).body if isinstance(node, ast.For)]
    assert (
        ast.unparse(loops[0].body[1])
        == "current = cutlass.Float32(raw) * cutlass.Float32(2)"
    )
    assert ast.unparse(loops[0].body[-1]) == "total = total + current"
    assert (
        ast.unparse(loops[1].body[-1])
        == "(out.iterator + synthetic_lane_7).store(current + total)"
    )
    assert "current = _computed_cache_" in ast.unparse(loops[1])


def test_actual_transformed_body_preserves_fp32_results_and_signed_zero() -> None:
    class Pointer:
        def __init__(self, data: np.ndarray, index: int = 0) -> None:
            self.data, self.index = data, index

        def __add__(self, offset: int) -> Pointer:
            return Pointer(self.data, self.index + offset)

        def load(self) -> np.float32:
            return self.data[self.index]

        def store(self, value: np.float32) -> None:
            self.data[self.index] = value

    results = []
    for mode in ("gmem", "computed"):
        data = np.asarray(
            [0.0, -0.0, 0.25, -0.25, 1.5, 1e-30, -2.5, 3.0], dtype=np.float32
        )
        out = np.full(8, np.nan, dtype=np.float32)
        namespace = {
            "x": SimpleNamespace(iterator=Pointer(data)),
            "out": SimpleNamespace(iterator=Pointer(out)),
            "cutlass": SimpleNamespace(Float32=np.float32),
            "cute": SimpleNamespace(
                make_rmem_tensor=lambda count, dtype: np.empty(count, dtype=dtype)
            ),
        }
        exec(_run(_body(), mode), namespace)
        results.append(out.view(np.uint32))
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.parametrize(
    "effect",
    [
        "(x.iterator + 0).store(cutlass.Float32(7))",
        "(unknown.iterator + 0).store(cutlass.Float32(7))",
        "cute.arch.atomic_add(x.iterator, cutlass.Float32(7))",
        "unknown_effect(x)",
        "cute.arch.barrier()",
    ],
)
def test_alias_and_unknown_effects_reject(effect: str) -> None:
    assert not _declarations(_run(_body(middle=effect)))


def test_disjoint_store_can_remain_between_complete_sweeps() -> None:
    effect = "(other.iterator + 0).store(cutlass.Float32(7))"
    assert _declarations(_run(_body(middle=effect)))
    assert not _declarations(
        _run(_body(middle=effect), proven_disjoint_tensor_pairs=set())
    )


@pytest.mark.parametrize(
    "replacement",
    [
        "cutlass.Float32(raw) * cutlass.Float32(3)",
        "cutlass.Float32(raw) + cutlass.Float32(2)",
        "cutlass.Float64(raw) * cutlass.Float64(2)",
        "cutlass.Float16(raw) * cutlass.Float16(2)",
        "opaque(raw)",
        "current + cutlass.Float32(raw)",
    ],
)
def test_exact_operator_dtype_and_unbound_carry_required(replacement: str) -> None:
    assert not _declarations(_run(_body(second=replacement)))


def test_changed_scalar_and_completed_induction_bindings_reject() -> None:
    source = _body(
        before="factor = cutlass.Float32(2)",
        first="cutlass.Float32(raw) * factor",
        middle="factor = cutlass.Float32(3)",
    )
    assert not _declarations(_run(source))
    source = _body(
        before="for j in range(2):\n    pass\nslot = j",
        load="(x.iterator + synthetic_lane_7 + slot).load()",
        middle="for j in range(3):\n    pass\nslot = j",
    )
    assert not _declarations(_run(source))


def test_mask_signature_and_raw_dtype_must_match() -> None:
    body = _body(
        load="(x.iterator + synthetic_lane_7).load() if synthetic_lane_7 < 7 else cutlass.Float32(0)"
    )
    second = body[-1]
    assert isinstance(second, ast.For)
    second.body[0] = ast.parse(
        "raw = (x.iterator + synthetic_lane_7).load() if synthetic_lane_7 < 6 else cutlass.Float32(0)"
    ).body[0]
    assert not _declarations(_run(body))
    bad = "cute.arch.load(x.iterator + synthetic_lane_7, cutlass.Int32) if synthetic_lane_7 < 7 else cutlass.Float32(0)"
    assert not _declarations(_run(_body(load=bad)))


@pytest.mark.parametrize(
    "load",
    [
        "(x.iterator + synthetic_lane_7).load(0)",
        "cute.arch.load(x.iterator + synthetic_lane_7, cutlass.Float32, 0)",
        "(x.iterator + synthetic_lane_7).load(volatile=True)",
        "(x.iterator + synthetic_lane_7).load() if synthetic_lane_7 < 7 else cutlass.Float32()",
    ],
)
def test_unknown_or_incomplete_scalar_load_signature_rejects(load: str) -> None:
    assert not _declarations(_run(_body(load=load)))


@pytest.mark.parametrize(
    "middle",
    [
        "result = cutlass.Float32(cute.arch.warp_reduction_sum(total, opaque=unknown_effect()))",
        "result = cutlass.Float32(cute.arch.warp_reduction_sum(total, **unknown))",
        "result = x[0]",
    ],
)
def test_unknown_intervening_reduction_argument_or_tensor_read_rejects(
    middle: str,
) -> None:
    assert not _declarations(_run(_body(middle=middle)))


def test_partial_population_and_range_rebinding_reject() -> None:
    body = _body()
    first = body[1]
    assert isinstance(first, ast.For)
    first.body[1] = ast.If(
        test=ast.parse("synthetic_lane_7 < 4", mode="eval").body,
        body=[first.body[1]],
        orelse=[],
    )
    assert not _declarations(_run(body))
    source = ast.unparse(
        ast.Module(body=_body(before="N = 8", middle="N = 7"), type_ignores=[])
    ).replace("range(8)", "range(N)")
    assert not _declarations(_run(ast.parse(source).body, constexpr_values={"N": 8}))


def test_live_fanout_and_escaping_values_are_not_pruned() -> None:
    body = _body(suffix="(other.iterator + 0).store(raw)")
    source = _run(body)
    assert len(_declarations(source)) == 1
    assert source.count(".load()") == 2
    assert "other.iterator + 0" in source


@pytest.mark.parametrize("container", ["for outer in range(2):", "if active:"])
def test_value_escaping_enclosing_scope_is_not_pruned(container: str) -> None:
    body = _body()
    second = body[-1]
    assert isinstance(second, ast.For)
    load, compute = second.body[:2]
    assert isinstance(load, ast.Assign) and isinstance(load.targets[0], ast.Name)
    assert isinstance(compute, ast.Assign)
    load.targets[0].id = "later_raw"
    compute.value = ast.parse(
        "cutlass.Float32(later_raw) * cutlass.Float32(2)", mode="eval"
    ).body
    nested = (
        container
        + "\n"
        + indent(ast.unparse(ast.Module(body=body, type_ignores=[])), "    ")
    )
    nested += "\n(other.iterator + 0).store(later_raw)\n"
    source = _run(ast.parse(nested).body)
    assert len(_declarations(source)) == 1
    assert "later_raw = (x.iterator + synthetic_lane_7).load()" in source


def test_cache_name_does_not_shadow_existing_binding() -> None:
    source = _run(_body(before="_computed_cache_0 = cutlass.Float32(9)"))
    assert "_computed_cache_1 = cute.make_rmem_tensor" in source
    assert "_computed_cache_0 = cutlass.Float32(9)" in source


def test_total_register_budget_and_different_sweep_domains() -> None:
    assert _declarations(_run(_body(count=MAX_CACHE_ELEMENTS)))
    assert not _declarations(_run(_body(count=MAX_CACHE_ELEMENTS + 1)))
    body = _body(count=40) + _body(count=40)
    count = 0
    for node in _declarations(_run(body)):
        assert isinstance(node.value, ast.Call)
        argument = node.value.args[0]
        assert isinstance(argument, ast.Constant) and isinstance(argument.value, int)
        count += argument.value
    assert count <= MAX_CACHE_ELEMENTS
    body = _body()
    second = body[-1]
    assert isinstance(second, ast.For)
    second.iter = ast.parse("range(7)", mode="eval").body
    assert not _declarations(_run(body))


@pytest.mark.parametrize("mode", ["auto", "register", "gmem"])
def test_old_policies_never_invoke_computed_pass(mode: str) -> None:
    with patch(
        "helion._compiler.cute.computed_sweep_cache.cache_computed_sweeps",
        side_effect=AssertionError("old mode"),
    ):
        source = _run(_body(), mode)
    assert "_computed_cache_" not in source


def test_shared_ast_occurrences_and_original_input_remain_independent() -> None:
    body = _body()
    first, second = body[1], body[-1]
    assert isinstance(first, ast.For) and isinstance(second, ast.For)
    second.body[:2] = first.body[:2]
    original = ast.unparse(ast.Module(body=body, type_ignores=[]))
    result = _run(body)
    assert "_computed_cache_" in result
    assert ast.unparse(ast.Module(body=body, type_ignores=[])) == original


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [17, 32])
def test_actual_compiler_caches_current_expression_and_preserves_old_modes(
    dtype: torch.dtype, width: int
) -> None:
    with _cpu_codegen():
        args = _args(dtype, width, 128)
        bound = _state_reduction._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_state_reduction, args)):
            computed = bound.to_code(_config("computed"))
            original = bound.to_code(_config("gmem"))
    assert len(_declarations(computed)) == 1
    assert "_computed_cache_" in computed
    assert computed.count("state.iterator") == 2
    assert original.count("state.iterator") == 3
    assert computed.count("cute.math.exp2(") == original.count("cute.math.exp2(") - 1
    assert computed.count("warp_reduction_sum(") == original.count(
        "warp_reduction_sum("
    )
