from __future__ import annotations

import ast
from collections import Counter
import hashlib
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _root_shared_gram(x, weight, mask_kind: hl.constexpr):
    rows_count, reduction = x.shape
    columns_count = weight.shape[1]
    output = torch.empty(
        (rows_count, columns_count), dtype=torch.float32, device=x.device
    )
    for rows, columns in hl.tile([rows_count, columns_count], block_size=[128, 128]):
        kk = hl.arange(reduction)
        if mask_kind == 1:
            raw = hl.load(x, [rows, kk], extra_mask=(rows.index % 2 == 0)[:, None])
        elif mask_kind == 2:
            raw = hl.load(x, [rows, kk], extra_mask=(kk % 2 == 0)[None, :])
        else:
            raw = x[rows, kk]
        intermediate = torch.exp(raw.to(torch.float32) * 0.125)
        rounded = (intermediate * 0.0625).to(x.dtype)
        gram = hl.dot(rounded, rounded.T, out_dtype=torch.float32)
        output[rows, columns] = hl.dot(
            gram.to(x.dtype),
            weight[rows.index - rows.begin, columns],
            out_dtype=torch.float32,
        )
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loop_shared_siblings(x, initial, mask_kind: hl.constexpr, steps: int):
    steps = hl.specialize(steps)
    _, rows_count, reduction = x.shape
    history = torch.empty(
        (steps, rows_count, 128), dtype=torch.float32, device=x.device
    )
    final = torch.empty_like(initial)
    for rows, columns in hl.tile([rows_count, 128], block_size=[128, 128]):
        state = initial[rows, columns]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(reduction)
            if mask_kind == 1:
                raw = hl.load(
                    x,
                    [step.id, rows, kk],
                    extra_mask=(rows.index % 2 == 0)[:, None],
                )
            elif mask_kind == 2:
                raw = hl.load(x, [step.id, rows, kk], extra_mask=(kk % 2 == 0)[None, :])
            else:
                raw = x[step.id, rows, kk]
            intermediate = torch.exp(raw.to(torch.float32) * 0.125)
            common = (intermediate * 0.0625).to(x.dtype)
            first_rhs = (intermediate + 0.5).to(x.dtype).T
            second_rhs = (intermediate * 0.5).to(x.dtype).T
            first = hl.dot(common, first_rhs, out_dtype=torch.float32)
            state = hl.dot(common, second_rhs, acc=state, out_dtype=torch.float32)
            history[step.id, rows, columns] = first
        final[rows, columns] = state
    return history, final


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _root_distinct_leaves(x, other, weight):
    rows_count, reduction = x.shape
    columns_count = weight.shape[1]
    output = torch.empty(
        (rows_count, columns_count), dtype=torch.float32, device=x.device
    )
    for rows, columns in hl.tile([rows_count, columns_count], block_size=[128, 128]):
        kk, nn = hl.arange(reduction), hl.arange(other.shape[0])
        left = torch.exp(x[rows, kk].float() * 0.125).to(x.dtype)
        right = torch.exp(other[nn, kk].float() * 0.125).to(x.dtype).T
        first = hl.dot(left, right, out_dtype=torch.float32)
        output[rows, columns] = hl.dot(
            first.to(x.dtype), weight[nn, columns], out_dtype=torch.float32
        )
    return output


def _config(family: str, enabled: bool | None) -> helion.Config:
    values = {
        "num_warps": 4,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_pointwise_vectorize": True,
    }
    if family == "loop":
        values["cute_chained_group_contractions"] = True
    if enabled is not None:
        values["cute_chained_vector_group"] = enabled
    return helion.Config.from_dict(values)


def _args(
    family: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    *,
    rows: int | None = None,
    mask_kind: int = 0,
    steps: int = 3,
) -> tuple:
    generator = torch.Generator(device=device).manual_seed(38701)
    if rows is None:
        rows = 256 if family == "root" else 259
    if family == "root":
        return (
            torch.randn((rows, 32), dtype=dtype, device=device, generator=generator)
            * 0.125,
            torch.randn((128, 128), dtype=dtype, device=device, generator=generator)
            * 0.125,
            mask_kind,
        )
    assert family == "loop"
    return (
        torch.randn(
            (max(steps, 1), rows, 32),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * 0.125,
        torch.randn(
            (rows, 128), dtype=torch.float32, device=device, generator=generator
        )
        * 0.125,
        mask_kind,
        steps,
    )


def _kernel(family: str) -> Kernel:
    return _root_shared_gram if family == "root" else _loop_shared_siblings


def _source(family: str, args: tuple, enabled: bool | None) -> str:
    with _cpu_codegen():
        return _kernel(family)._bind_isolated(args).to_code(_config(family, enabled))


def _calls(source: str) -> Counter[str]:
    return Counter(
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    )


@pytest.mark.parametrize("family", ["root", "loop"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mask_kind", [0, 1])
def test_public_vector_group_shares_fp32_dag_and_host_vector_cpu(
    family: str, dtype: torch.dtype, mask_kind: int
) -> None:
    args = _args(family, dtype=dtype, mask_kind=mask_kind)
    baseline = _source(family, args, False)
    grouped = _source(family, args, True)
    assert "chain_0_vector_group_leaf_0_vectorized" in grouped
    assert "chain_0_vector_group_leaf_1_vectorized" not in grouped
    assert _calls(grouped)["cute.math.exp2"] == 1
    # Root's older routes can spell both fast and fallback expressions.
    assert _calls(baseline)["cute.math.exp2"] >= (2 if family == "root" else 3)
    assert "cutlass.Float32" in grouped
    assert (
        "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    ) in grouped
    assert "_last_address ==" in grouped and "_address % 16 == 0" in grouped
    assert "chain_0_vector_group_output_1_copy" in grouped
    if family == "loop":
        assert "chain_1_mma" not in grouped
        assert "chain_0_vector_group_output_2_copy" in grouped
        assert "cute.domain_offset((128, 0), chain_0_b)" in grouped
    else:
        assert "chain_1_mma" in grouped
        assert "chain_0_vector_group_output_2_copy" not in grouped
    assert (
        _calls(grouped)["cute.arch.fence_view_async_shared"]
        == _calls(baseline)["cute.arch.fence_view_async_shared"]
    )
    assert (
        _calls(grouped)["cute.arch.sync_threads"]
        == _calls(baseline)["cute.arch.sync_threads"]
    )
    if mask_kind:
        assert "% 2" in grouped or "& 1" in grouped


@pytest.mark.parametrize("family", ["root", "loop"])
def test_public_vector_group_varying_mask_is_ineffective_cpu(family: str) -> None:
    args = _args(family, mask_kind=2)
    ordinary = _config(family, False)
    ordinary.config["cute_chained_pointwise_vectorize"] = False
    with _cpu_codegen():
        _kernel(family)._bind_isolated(args).to_code(ordinary)
    with pytest.raises(helion.exc.BackendUnsupported, match="vector grouping"):
        _source(family, args, True)


@pytest.mark.parametrize("width", [64, 128])
def test_distinct_original_leaves_or_rectangular_geometry_reject_cpu(
    width: int,
) -> None:
    args = (
        torch.empty((256, 32), dtype=torch.bfloat16),
        torch.empty((width, 32), dtype=torch.bfloat16),
        torch.empty((width, 128), dtype=torch.bfloat16),
    )
    with _cpu_codegen():
        _root_distinct_leaves._bind_isolated(args).to_code(_config("root", False))
        with pytest.raises(helion.exc.BackendUnsupported, match="vector grouping"):
            _root_distinct_leaves._bind_isolated(args).to_code(_config("root", True))


@pytest.mark.parametrize("family", ["root", "loop"])
def test_public_flag_true_cannot_silently_fall_back_cpu(family: str) -> None:
    from helion._compiler.cute import chained_vector_group

    with (
        patch.object(chained_vector_group, "emit_vector_group", return_value=None),
        pytest.raises(helion.exc.BackendUnsupported, match="vector grouping"),
    ):
        _source(family, _args(family), True)


@pytest.mark.parametrize("family", ["root", "loop"])
def test_integrated_outputs_share_original_leaf_at_exact_coordinates_cpu(
    family: str,
) -> None:
    from helion._compiler.cute import chained_matmul as chain
    from helion._compiler.cute import chained_vector_group

    original = chained_vector_group.emit_vector_group
    checked = []

    def observe(*args: Any, **kwargs: Any) -> list[str] | None:
        lines = original(*args, **kwargs)
        if lines is not None:
            cg, plan, boundaries, outputs = args[:4]
            keys = []
            for output in outputs:
                expression = chain._Expression(cg, plan, boundaries)
                expression.coordinate_names.update(("review_row", "review_column"))
                expression.value(
                    output.node, output.coordinates("review_row", "review_column")
                )
                keys.append(
                    {(leaf, coords) for leaf, coords, *_ in expression.loaded_inputs}
                )
            assert len(keys[0]) == 1 and all(key == keys[0] for key in keys)
            if family == "root":
                assert outputs[0].node in chain._ancestors(outputs[1].node)
            else:
                assert len(outputs) == 3
                assert not any(
                    outputs[left].node in chain._ancestors(outputs[right].node)
                    for left, right in ((0, 1), (0, 2), (1, 2), (2, 1))
                )
            checked.append(len(outputs))
        return lines

    with patch.object(chained_vector_group, "emit_vector_group", side_effect=observe):
        _source(family, _args(family), True)
    assert checked == [2 if family == "root" else 3]


@pytest.mark.parametrize("family", ["root", "loop"])
def test_same_tag_does_not_hide_mismatched_operand_domain_cpu(family: str) -> None:
    from helion._compiler.cute import chained_matmul as chain

    original = chain._operand_domain

    def different_domain(*args: Any, **kwargs: Any) -> list[str]:
        domain = original(*args, **kwargs)
        _cg, node, coords, _plan = args
        if node.target in (torch.ops.aten.permute.default, torch.ops.aten.t.default):
            return [*domain, f"({coords[1]}) < 64"]
        return domain

    with (
        patch.object(chain, "_operand_domain", side_effect=different_domain),
        pytest.raises(helion.exc.BackendUnsupported, match="vector grouping"),
    ):
        _source(family, _args(family), True)


def test_root_group_retains_requested_unroll_two_cpu() -> None:
    config = _config("root", True)
    config.config["cute_chained_pointwise_unroll"] = 2
    with _cpu_codegen():
        source = _root_shared_gram._bind_isolated(_args("root")).to_code(config)
    assert "chain_0_vector_group_output_1_copy" in source
    assert "for chain_0_vector_group_step in cutlass.range(4, unroll=2)" in source


# Populated only from the untouched lMqLVZgp compiler in a fresh CPU process.
_FROZEN_DISABLED_SHA256: dict[tuple[str, torch.dtype], str] = {
    (
        "root",
        torch.bfloat16,
    ): "d703e43400ac176d5f209834bd761b454f12e96686b7330f2ccc5cafcd67aeff",
    (
        "root",
        torch.float16,
    ): "359448e3c50aac05ee32501a9425fdfad55423777b63fc0db055eb35ac1c752d",
    (
        "loop",
        torch.bfloat16,
    ): "43c5136fa37553dfe184a9cfa6e877bf28fec4df4f246e0cc9cba525c59aa4d4",
    (
        "loop",
        torch.float16,
    ): "cb825e00cee608a3dd88ad4fa16cb52d751efea1652defebb2f456022df4b283",
}


@pytest.mark.parametrize("family", ["root", "loop"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_disabled_vector_group_preserves_frozen_source_cpu(
    family: str, dtype: torch.dtype
) -> None:
    args = _args(family, dtype=dtype)
    absent, disabled = (_source(family, args, enabled) for enabled in (None, False))
    assert absent == disabled
    assert (
        hashlib.sha256(disabled.encode()).hexdigest()
        == _FROZEN_DISABLED_SHA256[family, dtype]
    )


def _reference(
    family: str, args: tuple
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    x, second, mask_kind = args[:3]

    def operands(raw: torch.Tensor, begin: int) -> tuple[torch.Tensor, ...]:
        value = raw.float()
        if mask_kind == 1:
            mask = (torch.arange(raw.shape[0], device=x.device) + begin) % 2 == 0
            value = torch.where(mask[:, None], value, 0.0)
        intermediate = torch.exp(value * 0.125)
        forms = (intermediate * 0.0625, intermediate + 0.5, intermediate * 0.5)
        result = []
        for form in forms:
            padded = torch.zeros((128, 32), dtype=x.dtype, device=x.device)
            padded[: raw.shape[0]] = form.to(x.dtype)
            result.append(padded.float())
        return tuple(result)

    if family == "root":
        output = torch.empty((x.shape[0], 128), dtype=torch.float32, device=x.device)
        for begin in range(0, x.shape[0], 128):
            left = operands(x[begin : begin + 128], begin)[0]
            gram = (left @ left.T).to(x.dtype).float()
            output[begin : begin + 128] = (gram @ second.float())[
                : min(128, x.shape[0] - begin)
            ]
        return output
    steps = args[3]
    state = second.clone()
    history = torch.empty((steps, *state.shape), dtype=torch.float32, device=x.device)
    for step in range(steps):
        for begin in range(0, x.shape[1], 128):
            left, first, last = operands(x[step, begin : begin + 128], begin)
            update = left @ last.T
            state[begin : begin + 128] += update[: min(128, x.shape[1] - begin)]
            history[step, begin : begin + 128] = (left @ first.T)[
                : min(128, x.shape[1] - begin)
            ]
    return history, state


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "family,steps", [("root", 1), ("loop", 0), ("loop", 1), ("loop", 5)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mask_kind", [0, 1])
def test_vector_group_gpu_exact_replay_mutation_and_immutable_inputs(
    family: str, steps: int, dtype: torch.dtype, mask_kind: int
) -> None:
    args = _args(family, DEVICE, dtype, steps=steps, mask_kind=mask_kind)
    originals = tuple(
        value.clone() for value in args if isinstance(value, torch.Tensor)
    )
    baseline = (
        _kernel(family)._bind_isolated(args).compile_config(_config(family, False))
    )
    bound = _kernel(family)._bind_isolated(args)
    source = bound.to_code(_config(family, True))
    assert "chain_0_vector_group_output_1_copy" in source
    compiled = bound.compile_config(_config(family, True))
    expected, actual = baseline(*args), compiled(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, _reference(family, args), rtol=2e-3, atol=2e-3)
    for _ in range(3):
        torch.empty((131072,), dtype=torch.float32, device=DEVICE).fill_(float("nan"))
        torch.testing.assert_close(compiled(*args), expected, rtol=0, atol=0)
    changed = (args[0].clone() + 0.25, *args[1:])
    changed_saved = changed[0].clone()
    changed_expected = baseline(*changed)
    changed_actual = compiled(*changed)
    torch.testing.assert_close(changed_actual, changed_expected, rtol=0, atol=0)
    if family == "root" or steps:
        original_final = actual if family == "root" else actual[1]
        changed_final = changed_actual if family == "root" else changed_actual[1]
        assert not torch.equal(original_final, changed_final)
    torch.testing.assert_close(changed[0], changed_saved, rtol=0, atol=0)
    for value, original in zip(
        (value for value in args if isinstance(value, torch.Tensor)),
        originals,
        strict=True,
    ):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
