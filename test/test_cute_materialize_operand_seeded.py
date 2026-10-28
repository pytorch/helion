"""CPU admission and semantics of operand materialization for seeded and
transposed GEMM programs (``examples.se_block`` backward)."""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING

from examples.se_block import se_block_bwd_dw
from examples.se_block import se_block_bwd_dx
import pytest
import torch

from test.test_cute_materialize_operand import _bind
from test.test_cute_materialize_operand import _computed_rhs
from test.test_cute_materialize_operand import _cpu_b200 as _cpu_b200
from test.test_cute_materialize_operand import _reference

import helion
from helion._compiler.cute.materialize_operand import _additive_seed
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _scaled_seed(grad_out: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The seed needs the accumulator's value: not an additive rebinding."""
    m, n = grad_out.size()
    out = torch.empty([m, n], dtype=grad_out.dtype, device=grad_out.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc = acc * 2.0 + grad_out[tile_m, tile_n]
        for tile_k in hl.tile(n):
            operand = (grad_out[tile_m, tile_k].to(torch.float32) * 0.5).to(
                grad_out.dtype
            )
            acc = torch.addmm(acc, operand, w[tile_n, tile_k].T)
        out[tile_m, tile_n] = acc.to(grad_out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _subtracted_from_seed(grad_out: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``acc = e - acc`` negates the sum once moved after the loop: refused."""
    m, n = grad_out.size()
    out = torch.empty([m, n], dtype=grad_out.dtype, device=grad_out.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc = grad_out[tile_m, tile_n] - acc
        for tile_k in hl.tile(n):
            operand = (grad_out[tile_m, tile_k].to(torch.float32) * 0.5).to(
                grad_out.dtype
            )
            acc = torch.addmm(acc, operand, w[tile_n, tile_k].T)
        out[tile_m, tile_n] = acc.to(grad_out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _assigned_seeds(grad_out: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Two ``Assign`` seeds (``acc + e`` and ``e + acc``) are both additive."""
    m, n = grad_out.size()
    out = torch.empty([m, n], dtype=grad_out.dtype, device=grad_out.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc = acc + grad_out[tile_m, tile_n]
        acc = grad_out[tile_m, tile_n].to(torch.float32) * 0.25 + acc
        for tile_k in hl.tile(n):
            operand = (grad_out[tile_m, tile_k].to(torch.float32) * 0.5).to(
                grad_out.dtype
            )
            acc = torch.addmm(acc, operand, w[tile_n, tile_k].T)
        out[tile_m, tile_n] = acc.to(grad_out.dtype)
    return out


def _se_args(m: int, n: int, *, dyadic: bool = False) -> dict[str, torch.Tensor]:
    """fp16 inputs; ``dyadic`` draws values whose fp16 products and fp32 sums
    are exact, so two accumulation orders give the same bits."""
    if dyadic:
        generator = torch.Generator().manual_seed(1904)
        grad_out = torch.randint(-2, 3, (m, n), generator=generator) * 0.5
        x = torch.randint(-2, 3, (m, n), generator=generator) * 0.5
        w = torch.randint(-2, 3, (n, n), generator=generator) * 0.5
        s = torch.randint(0, 5, (m, n), generator=generator) * 0.25
        return {
            name: value.to(torch.float16)
            for name, value in (("grad_out", grad_out), ("x", x), ("w", w), ("s", s))
        }
    return {
        "grad_out": torch.empty((m, n), dtype=torch.float16),
        "x": torch.empty((m, n), dtype=torch.float16),
        "w": torch.empty((n, n), dtype=torch.float16),
        "s": torch.empty((m, n), dtype=torch.float16),
    }


_dx_args = operator.itemgetter("grad_out", "x", "w", "s")
_dw_args = operator.itemgetter("grad_out", "x", "s")


def _bind_se(
    kernel: Kernel[Any], args: tuple[torch.Tensor, ...], *, static_shapes: bool = True
) -> BoundKernel[Any]:
    return _bind(kernel, args, static_shapes=static_shapes)


def _roots(bound: BoundKernel[Any]) -> tuple[str, str]:
    plan = bound.env.cute_fission_plan
    assert plan is not None and plan.replacement_body is not None
    roots = [
        statement
        for statement in plan.replacement_body
        if isinstance(statement, ast.For)
    ]
    assert len(roots) == 2
    producer, consumer = (ast.unparse(root) for root in roots)
    return producer, consumer


def test_augassign_seed_is_materialized_with_the_seed_in_the_epilogue() -> None:
    """``acc += 2 * grad_out[m, n] * s[m, n]`` before the K loop: the LHS recipe
    is produced into an ``[M, K]`` tensor, the consumer contracts it against
    the transposed ``w`` tile and adds the seed after the contraction."""
    bound = _bind_se(se_block_bwd_dx, _dx_args(_se_args(1024, 512)))
    plan = bound.env.cute_fission_plan
    assert plan is not None and plan.region_count == 2
    assert plan.operand_interleave_factor == 1
    assert plan.pointwise_region_indices == (0,)
    assert bound.host_function is not None
    assert len(bound.host_function.device_ir.root_ids) == 2
    assert bound.config_spec.cute_tcgen05_search_enabled
    producer, consumer = _roots(bound)
    assert producer.startswith(
        "for _helion_materialize_m, _helion_materialize_k in hl.tile([m, n]):"
    )
    assert (
        "_helion_materialized_operand[_helion_materialize_m, _helion_materialize_k]"
        " = grad_to_w" in producer
    )
    assert "grad_out[_helion_materialize_m, _helion_materialize_k]" in producer
    consumer_lines = [line.strip() for line in consumer.splitlines()]
    assert consumer_lines[1] == "acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)"
    assert consumer_lines[2] == (
        "for _helion_materialize_logical_k in hl.tile(w.size(1)):"
    )
    assert consumer_lines[3] == (
        "acc = torch.addmm(acc, _helion_materialized_operand[tile_m, "
        "_helion_materialize_logical_k], w[tile_n, _helion_materialize_logical_k].T)"
    )
    assert consumer_lines[4] == (
        "acc = acc + 2.0 * grad_out[tile_m, tile_n] * s[tile_m, tile_n]"
    )
    assert consumer_lines[5] == "grad_x[tile_m, tile_n] = acc.to(x.dtype)"
    assert len(consumer_lines) == 6
    allocation = next(
        ast.unparse(statement)
        for statement in plan.replacement_body
        if isinstance(statement, ast.Assign)
        and ast.unparse(statement.targets[0]) == "_helion_materialized_operand"
    )
    assert allocation == (
        "_helion_materialized_operand = torch.empty((m, w.size(1)), "
        "dtype=torch.float16, device=w.device)"
    )


def test_transposed_direct_operand_is_materialized_on_the_rhs() -> None:
    """``x[k, n1].T @ recipe[k, n2]``: the direct A operand is a transposed input
    tile, so K is its leading position; the consumer keeps the ``.T`` read."""
    bound = _bind_se(se_block_bwd_dw, _dw_args(_se_args(1024, 512)))
    plan = bound.env.cute_fission_plan
    assert plan is not None and plan.region_count == 2
    assert bound.config_spec.cute_tcgen05_search_enabled
    producer, consumer = _roots(bound)
    assert producer.startswith(
        "for _helion_materialize_k, _helion_materialize_n in hl.tile([m, n]):"
    )
    assert (
        "_helion_materialized_operand[_helion_materialize_k, _helion_materialize_n]"
        " = grad_to_w" in producer
    )
    consumer_lines = [line.strip() for line in consumer.splitlines()]
    assert consumer_lines[2] == (
        "for _helion_materialize_logical_k in hl.tile(x.size(0)):"
    )
    assert consumer_lines[3] == (
        "acc_w = torch.addmm(acc_w, x[_helion_materialize_logical_k, tile_n1].T, "
        "_helion_materialized_operand[_helion_materialize_logical_k, tile_n2])"
    )
    assert consumer_lines[4] == "grad_w[tile_n1, tile_n2] = acc_w.to(x.dtype)"
    assert len(consumer_lines) == 5
    allocation = next(
        ast.unparse(statement)
        for statement in plan.replacement_body
        if isinstance(statement, ast.Assign)
        and ast.unparse(statement.targets[0]) == "_helion_materialized_operand"
    )
    assert allocation == (
        "_helion_materialized_operand = torch.empty((x.size(0), n), "
        "dtype=torch.float16, device=x.device)"
    )


@pytest.mark.parametrize(
    ("m", "n", "dx_expected", "dw_expected"),
    [
        # dx reuses its [M, K] recipe across N = 128 (< 2 x 256 columns), dw
        # its [K, N] recipe across N1 = 128 (< 2 x 128 rows): both fused.
        (256, 128, False, False),
        # N = 256 is one 256-column tile for dx; two 128-row tiles for dw.
        (512, 256, False, True),
        (1024, 512, True, True),
        (4096, 1024, True, True),
    ],
)
def test_recompute_gate_keeps_small_reused_axes_fused(
    m: int, n: int, dx_expected: bool, dw_expected: bool
) -> None:
    values = _se_args(m, n)
    dx = _bind_se(se_block_bwd_dx, _dx_args(values))
    dw = _bind_se(se_block_bwd_dw, _dw_args(values))
    assert (dx.env.cute_fission_plan is not None) is dx_expected
    assert (dw.env.cute_fission_plan is not None) is dw_expected
    for bound in (dx, dw):
        assert bound.host_function is not None
        assert len(bound.host_function.device_ir.root_ids) == (
            2 if bound.env.cute_fission_plan is not None else 1
        )


def test_one_source_same_width_recipe_keeps_its_fused_launch_when_static() -> None:
    """A static program's fused derived-operand pipeline reads one bf16 source
    per bf16 operand element (traffic 1x): the producer launch cannot pay for
    itself whatever the recompute factor.  Dynamic programs have no fused
    tcgen05 competitor and materialize as before."""
    args = (
        torch.empty((512, 1024), dtype=torch.bfloat16),
        torch.empty((1024, 512), dtype=torch.bfloat16),
    )
    assert _bind(_computed_rhs, args, static_shapes=True).env.cute_fission_plan is None
    assert _bind(_computed_rhs, args).env.cute_fission_plan is not None


@pytest.mark.parametrize("kernel", (_scaled_seed, _subtracted_from_seed))
def test_non_additive_seeds_keep_the_original_program(kernel: Kernel[Any]) -> None:
    values = _se_args(64, 32)
    bound = _bind(kernel, (values["grad_out"], values["w"]))
    assert bound.env.cute_fission_plan is None
    assert bound.host_function is not None
    assert len(bound.host_function.device_ir.root_ids) == 1


def test_assigned_seeds_move_to_the_epilogue_in_order() -> None:
    values = _se_args(64, 32)
    bound = _bind(_assigned_seeds, (values["grad_out"], values["w"]))
    assert bound.env.cute_fission_plan is not None
    _, consumer = _roots(bound)
    lines = [line.strip() for line in consumer.splitlines()]
    assert lines[2].startswith("for _helion_materialize_logical_k in")
    assert lines[4] == "acc = acc + grad_out[tile_m, tile_n]"
    assert lines[5] == "acc = grad_out[tile_m, tile_n].to(torch.float32) * 0.25 + acc"
    assert lines[6].startswith("out[tile_m, tile_n] = ")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("acc += e", True),
        ("acc -= e", True),
        ("acc = acc + e", True),
        ("acc = e + acc", True),
        ("acc = acc - e", True),
        ("acc = e - acc", False),
        ("acc *= 2", False),
        ("acc = acc * 2 + e", False),
        ("acc += acc", False),
        ("acc += tile_k", False),
        ("other = acc + e", False),
        ("acc = e", False),
    ],
)
def test_additive_seed_rule(source: str, expected: bool) -> None:
    statement = ast.parse(source).body[0]
    assert _additive_seed(statement, "acc", "tile_k") is expected


@pytest.mark.parametrize(
    ("kernel", "pick"),
    [(se_block_bwd_dx, _dx_args), (se_block_bwd_dw, _dw_args)],
    ids=["dx", "dw"],
)
def test_cpu_reference_matches_the_original_program_bitwise(
    kernel: Kernel[Any], pick: Any
) -> None:
    """Exact inputs: every fp16 product and fp32 partial sum is representable,
    so moving the seed after the contraction cannot change a bit."""
    values = _se_args(24, 40, dyadic=True)
    args = pick(values)
    bound = _bind_se(kernel, args, static_shapes=False)
    assert bound.env.cute_fission_plan is not None
    actual = _reference(bound, args, transformed=True)
    expected = _reference(bound, args, transformed=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    original = _bind_se(kernel, args, static_shapes=False)
    with torch.no_grad():
        grad_out, x, s = values["grad_out"], values["x"], values["s"]
        derived = (
            2.0 * grad_out.float() * x.float() * s.float() * (1.0 - s.float())
        ).to(torch.float16)
        if kernel is se_block_bwd_dx:
            formula = 2.0 * grad_out.float() * s.float() + derived.float() @ (
                values["w"].float().T
            )
        else:
            formula = x.float().T @ derived.float()
    assert original.kernel.fn is kernel.fn
    torch.testing.assert_close(actual, formula.to(torch.float16), rtol=0, atol=0)
