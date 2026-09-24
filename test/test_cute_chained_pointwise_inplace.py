from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_pointwise import _args
from .test_cute_chained_pointwise import _code
from .test_cute_chained_pointwise import _config
from .test_cute_chained_pointwise import _pointwise_dot
from .test_cute_chained_pointwise_fp32 import _mixed_args
from .test_cute_chained_scan_export import _args as scan_args
from .test_cute_chained_scan_export import _config as scan_config
from .test_cute_chained_scan_export import _cpu_codegen
from .test_cute_chained_scan_export import _scan_export
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute.chained_pointwise_inplace import PointwiseInplace
from helion._compiler.cute.chained_pointwise_inplace import sw128_ownership
from helion._compiler.cute.chained_tcgen05 import _VectorLeaf
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_pointwise_inplace_async"


def code(args: tuple, value: object = False, *, enabled: bool = True) -> str:
    config = helion.Config.from_dict(_config(enabled).config | {KEY: value})
    with patch("test.test_cute_chained_pointwise._config", return_value=config):
        return _code(args, enabled)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(
    "kind", ("dense", "transpose", "offset", "stride", "tail", "padded")
)
def test_source_default_identity_and_exact_math(dtype: torch.dtype, kind: str) -> None:
    args = _args("cpu", kind, dtype)
    old = _code(args)
    assert code(args) == old
    new = code(args, True)
    assert "_raw_copy =" in new
    a, b = ast.parse(old), ast.parse(new)
    assert ast.dump(a.body[-1]) == ast.dump(b.body[-1])

    def branches(tree: ast.AST) -> list[ast.If]:
        return [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and "pointwise_leaf_" in ast.unparse(n.test)
        ]

    before, after = branches(a), branches(b)
    assert len(before) == len(after)
    for first, second in zip(before, after, strict=True):
        assert ast.dump(first.test) == ast.dump(second.test)
        assert ast.dump(ast.Module(first.orelse, type_ignores=[])) == ast.dump(
            ast.Module(second.orelse, type_ignores=[])
        )
        old_loop = next(n for n in first.body if isinstance(n, ast.For))
        new_loop = next(n for n in second.body if isinstance(n, ast.For))
        assert ast.dump(old_loop.iter) == ast.dump(new_loop.iter)

        def math_nodes(loop):
            return [
                ast.dump(n)
                for n in loop.body
                if not (
                    isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == "cute.copy"
                )
            ]

        assert math_nodes(old_loop) == math_nodes(new_loop)
        if "_raw_copy" in ast.unparse(second):
            prefix = [
                ast.unparse(n) for n in second.body[: second.body.index(new_loop)]
            ]
            pos = next(
                i
                for i, text in enumerate(prefix)
                if text.startswith("cute.copy(") and "_raw_copy" in text
            )
            assert prefix[pos + 1 : pos + 4] == [
                "cute.arch.cp_async_commit_group()",
                "cute.arch.cp_async_wait_group(0)",
                "cute.arch.sync_threads()",
            ]
    assert [line for line in old.splitlines() if "alloc_smem(" in line] == [
        line for line in new.splitlines() if "alloc_smem(" in line
    ]


@pytest.mark.parametrize("value", (0, 1, "true", None, 2))
def test_invalid_bool_rejected(value: object) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        code(_args("cpu", "dense"), value)


def test_disabled_vector_path_canonicalizes() -> None:
    args = _args("cpu", "dense")
    assert code(args, True, enabled=False) == _code(args, False)


@pytest.mark.parametrize(
    "shape,inner,expected",
    [
        ((128, 128), 0, True),
        ((128, 64), 1, True),
        ((64, 128), 0, True),
        ((128, 32), 1, False),
        ((12, 64), 1, False),
        ((128, 96), 1, False),
        ((128, 2048), 1, False),
    ],
)
def test_sw128_layout_envelope(shape, inner, expected) -> None:
    assert sw128_ownership(shape, inner) is expected


def test_typed_selection_and_private_activation() -> None:
    def leaf(dtype: torch.dtype, stride: int = 128) -> _VectorLeaf:
        return _VectorLeaf(
            torch.fx.Graph().placeholder("input"),
            ("row", "col"),
            "input",
            "0",
            stride,
            (),
            dtype,
        )

    first = PointwiseInplace(True)
    leaves = [leaf(torch.float32), leaf(torch.bfloat16)]
    assert first.select(leaves, (128, 128), 1, torch.bfloat16) == 1
    first.validate()
    other = PointwiseInplace(True)
    assert other.select(leaves[:1], (128, 128), 1, torch.bfloat16) is None
    with pytest.raises(exc.BackendUnsupported, match="same-dtype"):
        other.validate()
    PointwiseInplace(False).validate()
    assert (
        PointwiseInplace(True).select(
            [leaf(torch.float16)], (128, 128), 1, torch.bfloat16
        )
        is None
    )
    assert (
        PointwiseInplace(True).select(
            [leaf(torch.bfloat16, 0)], (128, 128), 1, torch.bfloat16
        )
        is None
    )


def test_actual_initial100_and_filtered_seed_order() -> None:
    with patch_cute_mma_support():
        bound = _pointwise_dot._bind_isolated(_args("cpu", "dense"))
    spec = bound.config_spec
    assert bound.host_function is not None
    assert spec.cute_chained_pointwise_inplace_search_enabled
    with bound.env:
        new = CuteChainedMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        spec.cute_chained_pointwise_inplace_search_enabled = False
        try:
            old = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        finally:
            spec.cute_chained_pointwise_inplace_search_enabled = True
        assert new is not None and old is not None
        assert [seed for seed in new if not seed.config.get(KEY)] == old
        generation = ConfigGeneration(spec)
        population = [
            generation.unflatten(value)
            for value in generation.random_population_flat(100)
        ]
        candidates = [seed for seed in population if seed.config.get(KEY)]
        assert candidates
        assert not population[0].config.get(KEY)
        for candidate in candidates:
            assert (
                generation.unflatten(generation.flatten(candidate)).config[KEY] is True
            )
    with patch("test.test_cute_chained_pointwise._config", return_value=candidates[0]):
        assert "_raw_copy =" in _code(_args("cpu", "dense"))


def test_fp32_leaf_keeps_original_typed_load_path() -> None:
    args = _mixed_args("cpu", "dense")
    before, after = _code(args), code(args, True)
    assert "chain_0_a_raw_copy" in after
    assert "chain_0_b_raw_copy" not in after
    begin, end = "    chain_0_b_ptr =", "    cute.arch.cp_async_commit_group()"
    assert before.split(begin)[1].split(end)[0] == after.split(begin)[1].split(end)[0]


def test_no_dense_16bit_leaf_rejects_explicit_request() -> None:
    a, b, scale, bias, transpose = _mixed_args("cpu", "dense")
    a = torch.empty((*a.shape[:-1], a.shape[-1] * 2), dtype=a.dtype)[..., ::2]
    with pytest.raises(exc.InvalidConfig, match="computed same-dtype TCgen05 operands"):
        code((a, b, scale, bias, transpose), True)


@pytest.mark.parametrize("factor", (1, 2, 4, 8))
def test_scan_export_cache_and_unroll_interaction(factor: int) -> None:
    config = scan_config().config | {
        KEY: True,
        "cute_chained_pointwise_vectorize": True,
        "cute_chained_pointwise_unroll": factor,
        "cute_chained_pointwise_read_cache": True,
        "cute_chained_auxiliary_cache": True,
        "cute_chained_c_smem_padding": 4,
    }
    with _cpu_codegen():
        bound = _scan_export._bind_isolated((*scan_args(), "normal"))
        old = bound.to_code(helion.Config.from_dict(config | {KEY: False}))
        new = bound.to_code(helion.Config.from_dict(config))
    assert "_raw_copy =" in new and "_read_cache_" in new
    marker = "    if chain_origin_1 == 0 and chain_origin_2 == 0"
    assert marker in new and new.split(marker)[1] == old.split(marker)[1]


def test_direct_family_does_not_advertise_inplace() -> None:
    from .test_cute_chained_tcgen05_config import _config_chain
    from .test_cute_chained_tcgen05_config import _inputs

    with patch_cute_mma_support():
        bound = _config_chain._bind_isolated((*_inputs(), None, False))
    assert not bound.config_spec.cute_chained_pointwise_inplace_search_enabled
    assert KEY not in bound.config_spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="same-dtype TCgen05"):
        bound.config_spec.normalized_config(
            helion.Config.from_dict(_config().config | {KEY: True})
        )
