from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_host_paired_sum import _config
from test.test_cute_host_paired_sum import _host

import helion
from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.cute.host_single_sum import lower_host_single_sum
from helion._compiler.cute.host_single_sum import prove_host_single_sum
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _single(
    a: torch.Tensor, owner: torch.Tensor, enabled: hl.constexpr = True
) -> tuple[torch.Tensor, torch.Tensor | None]:
    partial = torch.empty_like(a)
    for rows, columns in hl.tile(a.shape):
        partial[rows, columns] = a[rows, columns] + 1
    if enabled:
        return a, partial.sum(0).to(owner.dtype)
    return a, None


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _literal(a: torch.Tensor) -> torch.Tensor:
    partial = torch.empty_like(a)
    for i, j in hl.tile(a.shape):
        partial[i, j] = a[i, j]
    return partial.sum(0).to(torch.float16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _one_tuple(a: torch.Tensor) -> tuple[torch.Tensor]:
    partial = torch.empty_like(a)
    for i, j in hl.tile(a.shape):
        partial[i, j] = a[i, j]
    return (partial.sum(0).to(torch.float32),)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pair_and_single(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    partial = torch.empty_like(a)
    for i, j in hl.tile(a.shape):
        partial[i, j] = a[i, j]
    left = partial.sum(0).to(torch.bfloat16)
    right = a.sum(0).to(torch.bfloat16)
    return left, right, partial.sum(0).to(torch.float16)


def _arguments(
    dtype: torch.dtype = torch.float16, enabled: bool = True
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    return torch.randn(17, 20), torch.empty(20, dtype=dtype), enabled


@pytest.fixture(scope="module")
def single_bound() -> BoundKernel:
    return _cpu_bind(_single, _arguments())


@pytest.mark.parametrize("layout", ("mapped", "narrow"))
def test_terminal_sum_preserves_complete_producer_and_original_return(
    single_bound: BoundKernel, layout: str
) -> None:
    host = single_bound.host_function
    assert host is not None
    with single_bound.env:
        proof = prove_host_single_sum(host)
    assert proof.site is not None and proof.site.prefix == ("a",)
    assert proof.site.value.dtype == torch.float16
    control = ast.parse(single_bound.to_code(_config(single_bound, "off")))
    fused = ast.parse(single_bound.to_code(_config(single_bound, layout)))
    assert [
        ast.dump(node)
        for node in control.body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    ] == [
        ast.dump(node)
        for node in fused.body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    ]
    before, after = _host(control), _host(fused)
    producer = next(
        i
        for i, node in enumerate(before.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_launcher"
    )
    assert ast.dump(ast.Module(before.body[: producer + 1], [])) == ast.dump(
        ast.Module(after.body[: producer + 1], [])
    )
    call = after.body[producer + 1].value
    assert ast.unparse(call.func) == "_cute_try_single_sum_cast"
    assert [ast.unparse(value) for value in call.args] == ["partial", "owner"]
    fallback = after.body[producer + 2]
    assert isinstance(fallback, ast.If)
    original = next(
        node
        for node in ast.walk(before)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and isinstance(node.value.elts[-1], ast.Call)
    )
    assert ast.dump(fallback.body[0]) == ast.dump(original)
    assert ast.unparse(fallback.orelse[0]) == "return (a, _helion_single_sum)"


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_typed_destination_dtypes(dtype: torch.dtype) -> None:
    bound = _cpu_bind(_single, _arguments(dtype))
    assert bound.config_spec.cute_host_paired_sum_available
    assert "_cute_try_single_sum_cast(" in bound.to_code(_config(bound, "mapped"))


@pytest.mark.parametrize("kernel", (_literal, _one_tuple))
def test_direct_and_singleton_tuple_return_keep_result_type(
    kernel: helion.Kernel,
) -> None:
    args = (torch.arange(17 * 20, dtype=torch.float32).view(17, 20),)
    bound = _cpu_bind(kernel, args)
    bound.set_config(_config(bound, "narrow"))
    host = bound._run
    assert host is not None
    calls = []

    def producer(kernel, grid, *values, **kwargs):
        assert len(values) == 2
        values[1].copy_(values[0])
        calls.append(id(kernel))

    # This is the real generated caller and its original CPU fallback.
    with patch.dict(host.__kwdefaults__, {"_launcher": producer}):
        result = bound(*args)
    expected = args[0].sum(0).to(torch.float16 if kernel is _literal else torch.float32)
    if kernel is _one_tuple:
        assert isinstance(result, tuple) and len(result) == 1
        result = result[0]
    assert torch.equal(result, expected)
    assert len(calls) == 1 and bound._run is host


def test_pair_and_terminal_sum_are_independent() -> None:
    bound = _cpu_bind(_pair_and_single, (torch.ones(17, 20),))
    source = bound.to_code(_config(bound, "narrow"))
    assert source.count("_cute_try_paired_sum_cast(") == 1
    assert source.count("_cute_try_single_sum_cast(") == 1


def _clone(node: ast.AST) -> ast.AST:
    if isinstance(node, ExtendedAST):
        return node.new(
            {
                field: [_clone(x) if isinstance(x, ast.AST) else x for x in value]
                if isinstance(value, list)
                else _clone(value)
                if isinstance(value, ast.AST)
                else value
                for field, value in ast.iter_fields(node)
            }
        )
    return deepcopy(node)


@pytest.mark.parametrize(
    "mutation",
    (
        "no_metadata",
        "axis",
        "keepdim",
        "dtype_call",
        "dtype_unknown",
        "prefix_call",
        "prefix_attribute",
        "prefix_index",
        "prefix_unknown",
        "suffix_call",
        "unknown_source",
        "unknown_owner",
        "dynamic_branch",
        "exception_handler",
        "delete",
        "global",
        "later_helper_store",
    ),
)
def test_unknown_types_effects_bindings_and_control_flow_decline(
    single_bound: BoundKernel, mutation: str
) -> None:
    host = single_bound.host_function
    assert host is not None
    with single_bound.env:
        body = [_clone(node) for node in host.body]
        branch = body[-2]
        assert isinstance(branch, ast.If)
        returned = branch.body[0]
        cast = returned.value.elts[-1]
        if mutation == "no_metadata":
            body = ast.parse(ast.unparse(ast.Module(body, []))).body
        elif mutation == "axis":
            cast.func.value.args[0].value = 1
        elif mutation == "keepdim":
            cast.func.value.keywords = [ast.keyword("keepdim", ast.Constant(True))]
        elif mutation == "dtype_call":
            cast.args[0] = ast.parse("owner.dtype()", mode="eval").body
        elif mutation == "dtype_unknown":
            cast.args[0] = ast.parse("owner.dtype", mode="eval").body
        elif mutation.startswith("prefix_"):
            expressions = {
                "prefix_call": "a.clone()",
                "prefix_attribute": "a.dtype",
                "prefix_index": "a[0]",
                "prefix_unknown": "missing",
            }
            returned.value.elts[0] = ast.parse(expressions[mutation], mode="eval").body
        elif mutation == "suffix_call":
            returned.value.elts.reverse()
        elif mutation == "unknown_source":
            cast.func.value.func.value.id = "unbound_partial"
        elif mutation == "unknown_owner":
            cast.args[0].value.id = "unbound_owner"
        elif mutation == "dynamic_branch":
            branch.test = ast.Name("unproved", ast.Load())
        elif mutation == "exception_handler":
            body = [
                ast.Try(body, [ast.ExceptHandler(None, None, [ast.Pass()])], [], [])
            ]
        elif mutation == "delete":
            body.insert(-2, ast.Delete([ast.Name("partial", ast.Del())]))
        elif mutation == "global":
            body.insert(0, ast.Global(["partial"]))
        elif mutation == "later_helper_store":
            body.append(ast.parse("_cute_try_single_sum_cast = None").body[0])
        with patch.object(host.definition, "body", body):
            assert prove_host_single_sum(host).site is None


def test_final_return_requires_full_ast_and_original_boolean(
    single_bound: BoundKernel,
) -> None:
    host = single_bound.host_function
    assert host is not None
    with single_bound.env:
        untyped = ast.parse(ast.unparse(ast.Module(host.body, []))).body
        assert lower_host_single_sum(host, untyped, "mapped") is untyped
        branch = host.body[-2]
        for expression in ("False", "1", "observe_condition()"):
            changed = branch.copy(test=ast.parse(expression, mode="eval").body)
            body = [*host.body[:-2], changed, host.body[-1]]
            assert lower_host_single_sum(host, body, "narrow") is body


@pytest.mark.parametrize(
    "enabled,dtype", ((False, torch.float16), (True, torch.float64))
)
def test_negative_admission_has_no_seeds_and_rejects_explicit_choice(
    enabled: bool, dtype: torch.dtype
) -> None:
    bound = _cpu_bind(_single, _arguments(dtype, enabled))
    assert not bound.config_spec.cute_host_paired_sum_available
    assert all(
        "cute_host_paired_sum" not in seed
        for seed in bound.config_spec.compiler_seed_configs
    )
    with pytest.raises(InvalidConfig, match="terminal sum/cast"):
        bound.to_code(_config(bound, "narrow"))
