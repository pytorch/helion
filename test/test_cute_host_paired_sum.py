from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute.host_paired_sum import PAIRED_SUM_KEY
from helion._compiler.cute.host_paired_sum import lower_host_sum_pairs
from helion._compiler.cute.host_paired_sum import prove_host_sum_pairs
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
def _paired(
    a: torch.Tensor,
    b: torch.Tensor,
    dtype_owner: torch.Tensor,
    enabled: hl.constexpr = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    left_partial = torch.empty_like(a)
    right_partial = torch.empty_like(b)
    for rows, columns in hl.tile(a.shape):
        left_partial[rows, columns] = a[rows, columns] + 1
        right_partial[rows, columns] = b[rows, columns] - 1
    left = left_partial.sum(0).to(dtype_owner.dtype)
    if enabled:
        right = right_partial.sum(0).to(dtype_owner.dtype)
        return left, right, left_partial, right_partial
    return left, None, left_partial, right_partial


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _literal(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    partial = torch.empty_like(a)
    for i, j in hl.tile(a.shape):
        partial[i, j] = a[i, j]
    left = partial.sum(0).to(torch.float32)
    right = b.sum(0).to(torch.float32)
    return left, right


def _args(
    dtype: torch.dtype = torch.float32, enabled: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    return (
        torch.randn((17, 20), dtype=dtype),
        torch.randn((17, 20), dtype=dtype),
        torch.empty(20, dtype=torch.bfloat16),
        enabled,
    )


@pytest.fixture(scope="module")
def pair_bound() -> BoundKernel:
    return _cpu_bind(_paired, _args())


def _config(bound: BoundKernel, layout: object) -> helion.Config:
    with bound.env:
        config = bound.env.config_spec.default_config()
    config.config[PAIRED_SUM_KEY] = layout
    return config


def _source(bound: BoundKernel, layout: str) -> ast.Module:
    return ast.parse(bound.to_code(_config(bound, layout)))


def _host(module: ast.Module) -> ast.FunctionDef:
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and not node.decorator_list
    )


@pytest.mark.parametrize("layout", ("mapped", "narrow"))
def test_original_typed_proof_and_final_host_pair_preserve_producer(
    pair_bound: BoundKernel, layout: str
) -> None:
    host = pair_bound.host_function
    assert host is not None
    with pair_bound.env:
        proof = prove_host_sum_pairs(host)
    assert len(proof.pairs) == 1
    control, fused = _source(pair_bound, "off"), _source(pair_bound, layout)

    def devices(module: ast.Module) -> list[str]:
        return [
            ast.dump(node)
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.decorator_list
        ]

    assert devices(control) == devices(fused)
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
    calls = [
        node
        for node in ast.walk(after)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_cute_try_paired_sum_cast"
    ]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[2]) == "dtype_owner"
    fallback = next(
        node
        for node in after.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    )
    assert ast.dump(fallback.body[0]) == ast.dump(before.body[producer + 1])
    assert ast.dump(fallback.body[1]) == ast.dump(before.body[producer + 2].body[0])


@pytest.mark.parametrize(
    "enabled,dtype", [(False, torch.float32), (True, torch.float16)]
)
def test_missing_or_wrong_typed_pair_declines(
    enabled: bool, dtype: torch.dtype
) -> None:
    bound = _cpu_bind(_paired, _args(dtype, enabled))
    assert not bound.env.config_spec.cute_host_paired_sum_available
    assert "_cute_try_paired_sum_cast" not in bound.to_code(_config(bound, "off"))
    with pytest.raises(InvalidConfig, match="typed independent pair"):
        bound.to_code(_config(bound, "narrow"))


def test_literal_dtype_and_argument_source_are_typed() -> None:
    bound = _cpu_bind(_literal, _args()[:2])
    assert bound.env.config_spec.cute_host_paired_sum_available
    assert "dtype_is_tensor=False" in bound.to_code(_config(bound, "mapped"))


@pytest.mark.parametrize(
    "mutation",
    [
        "no_metadata",
        "first_target_dependency",
        "same_target",
        "existing_target",
        "intervening_effect",
        "dynamic_branch",
        "effectful_true_branch",
        "axis",
        "keepdim",
        "dtype_effect",
        "method_spelling",
        "exception_handler",
        "global_output",
        "nonlocal_output",
    ],
)
def test_proof_declines_effects_dependencies_or_untyped_lookalikes(
    pair_bound: BoundKernel, mutation: str
) -> None:
    original = pair_bound.host_function
    assert original is not None
    with pair_bound.env:
        # Keep real compiler TypeInfo identities while independently replacing
        # AST structure. Deepcopy of symbolic fake tensors is not a proof.
        from helion._compiler import ast_extension

        def clone(node: ast.AST) -> ast.AST:
            if isinstance(node, ast_extension.ExtendedAST):
                return node.new(
                    {
                        field: [
                            clone(x) if isinstance(x, ast.AST) else x for x in value
                        ]
                        if isinstance(value, list)
                        else clone(value)
                        if isinstance(value, ast.AST)
                        else value
                        for field, value in ast.iter_fields(node)
                    }
                )
            return deepcopy(node)

        body = [clone(node) for node in original.body]
        first, branch = body[-3:-1]
        assert isinstance(first, ast.Assign) and isinstance(branch, ast.If)
        second = branch.body[0]
        assert isinstance(second, ast.Assign)
        if mutation == "no_metadata":
            body = ast.parse(ast.unparse(ast.Module(body, []))).body
        elif mutation == "first_target_dependency":
            second.value.func.value.func.value.id = first.targets[0].id
        elif mutation == "same_target":
            second.targets[0].id = first.targets[0].id
        elif mutation == "existing_target":
            body.insert(0, ast.parse("left = None").body[0])
        elif mutation == "intervening_effect":
            body.insert(-2, ast.parse("observe(left)").body[0])
        elif mutation == "dynamic_branch":
            branch.test = ast.Name("unproved_condition", ast.Load())
        elif mutation == "effectful_true_branch":
            old = branch.test
            branch.test = ast_extension.create(
                ast.Call, func=ast.Name("effect", ast.Load()), args=[], keywords=[]
            )
            branch.test._type_info = old._type_info
        elif mutation == "axis":
            first.value.func.value.args[0].value = 1
        elif mutation == "keepdim":
            first.value.func.value.keywords = [
                ast.keyword("keepdim", ast.Constant(True))
            ]
        elif mutation == "dtype_effect":
            first.value.args[0] = ast.Call(ast.Name("dtype", ast.Load()), [], [])
        elif mutation == "method_spelling":
            first.value.func.value.func.attr = "custom_sum"
        elif mutation == "exception_handler":
            body = [
                ast.Try(body, [ast.ExceptHandler(None, None, [ast.Pass()])], [], [])
            ]
        elif mutation == "global_output":
            body.insert(0, ast.Global([first.targets[0].id]))
        elif mutation == "nonlocal_output":
            body.insert(0, ast.Nonlocal([first.targets[0].id]))
        with patch.object(original.definition, "body", body):
            assert not prove_host_sum_pairs(original).pairs


def test_final_host_must_still_equal_the_proved_sites(pair_bound: BoundKernel) -> None:
    host = pair_bound.host_function
    assert host is not None
    # An untyped final lookalike cannot reuse an earlier typed proof.
    untyped = ast.parse(ast.unparse(ast.Module(host.body, []))).body
    with pair_bound.env:
        assert lower_host_sum_pairs(host, untyped, "mapped") is untyped


@pytest.mark.parametrize("changed_test", ("observe_condition()", "False", "1"))
def test_final_host_branch_must_preserve_the_proved_boolean(
    pair_bound: BoundKernel, changed_test: str
) -> None:
    host = pair_bound.host_function
    assert host is not None
    branch = host.body[-2]
    assert isinstance(branch, ast.If)
    # Preserve the real source location/type metadata while changing only the
    # final host's test. Source location alone cannot authorize a branch fold.
    changed = branch.copy(test=ast.parse(changed_test, mode="eval").body)
    assert changed.body is branch.body and changed.orelse is branch.orelse
    final = [*host.body[:-2], changed, host.body[-1]]
    with pair_bound.env:
        assert lower_host_sum_pairs(host, final, "narrow") is final


@pytest.mark.parametrize("bad", (True, 3, "automatic"))
def test_invalid_choice_rejected(pair_bound: BoundKernel, bad: object) -> None:
    with pytest.raises(InvalidConfig, match="paired host sum"):
        pair_bound.to_code(_config(pair_bound, bad))


def test_none_constructor_choice_is_the_implicit_off_default(
    pair_bound: BoundKernel,
) -> None:
    assert "cute_host_paired_sum" not in helion.Config(cute_host_paired_sum=None).config
    assert pair_bound.to_code(_config(pair_bound, None)) == pair_bound.to_code(
        _config(pair_bound, "off")
    )


def test_disabled_heuristics_keep_explicit_proof_but_no_seeds() -> None:
    kernel = helion.kernel(
        _paired.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    bound = _cpu_bind(kernel, _args())
    assert bound.env.config_spec.cute_host_paired_sum_available
    assert not bound.env.config_spec.compiler_seed_configs
    assert "_cute_try_paired_sum_cast" in bound.to_code(_config(bound, "narrow"))
