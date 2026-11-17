"""Dead-code elimination keeps asm not declared pure that nothing reads."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(),
    skipUnlessBackends(["triton", "cute"]),
]

# Stores its second operand through its first; the result is a dummy.
_STORE_ASM = "st.global.f32 [$1], $2; mov.u32 $0, 0;"


@helion.kernel(static_shapes=True, autotune_effort="none")
def _store_then_double(x: torch.Tensor, addrs: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.inline_asm_elementwise(
            _STORE_ASM,
            "=r,l,f",
            [addrs[tile], x[tile]],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        out[tile] = x[tile] * 2
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _store_row_then_sum(x: torch.Tensor, addrs: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile in hl.tile(m):
        row = x[tile, :]
        hl.inline_asm_elementwise(
            _STORE_ASM,
            "=r,l,f",
            [addrs[tile, :], row],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        out[tile] = row.sum(-1)
    return out


def _asm_lines(code: str) -> list[str]:
    # The generated code quotes each statement's source in comments.
    return [
        line
        for line in code.splitlines()
        if _STORE_ASM.split(";")[0] in line and not line.lstrip().startswith("#")
    ]


@skipIfRefEager("ref-eager mode generates no code")
@pytest.mark.parametrize(
    ("kernel", "shape", "reduction_loops"),
    [
        (_store_then_double, (64,), None),
        (_store_row_then_sum, (8, 256), [None]),
        # The roll's loops copy and clean up the row's graph.
        (_store_row_then_sum, (8, 256), [32]),
    ],
    ids=["pointwise", "row", "rolled-row"],
)
def test_asm_not_declared_pure_stays_without_readers(
    kernel: object, shape: tuple[int, ...], reduction_loops: list[int] | None
) -> None:
    """Its result is a dummy, which nothing reads; the asm stores, so the
    kernel still runs it."""
    args = (torch.empty(shape), torch.empty(shape, dtype=torch.int64))
    bound = cpu_bind(kernel, args)
    config = bound.config_spec.default_config()
    if reduction_loops is not None:
        config = helion.Config(**{**config.config, "reduction_loops": reduction_loops})
    assert _asm_lines(bound.to_code(config))


# Two dummy results.
_STORE_PAIR_ASM = "st.global.f32 [$2], $3; mov.u32 $0, 0; mov.u32 $1, 0;"


@helion.kernel(static_shapes=True, autotune_effort="none")
def _store_pair_then_double(
    x: torch.Tensor, addrs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    first = torch.empty(x.size(0), dtype=torch.int32, device=x.device)
    second = torch.empty(x.size(0), dtype=torch.int32, device=x.device)
    for tile in hl.tile(x.size(0)):
        hl.inline_asm_elementwise(
            _STORE_PAIR_ASM,
            "=r,=r,l,f",
            [addrs[tile], x[tile]],
            dtype=(torch.int32, torch.int32),
            is_pure=False,
            pack=1,
        )
        out[tile] = x[tile] * 2
        read_first, read_second = hl.inline_asm_elementwise(
            _STORE_PAIR_ASM,
            "=r,=r,l,f",
            [addrs[tile], out[tile]],
            dtype=(torch.int32, torch.int32),
            is_pure=False,
            pack=1,
        )
        first[tile] = read_first
        second[tile] = read_second
    return out, first, second


@skipIfRefEager("ref-eager mode generates no code")
def test_asm_with_several_results_runs_once() -> None:
    """An asm with two results stays when nothing reads them, and runs once
    when both are read (each result reads one call)."""
    bound = cpu_bind(
        _store_pair_then_double, (torch.empty(64), torch.empty(64, dtype=torch.int64))
    )
    code = bound.to_code(bound.config_spec.default_config())
    calls = [
        line
        for line in code.splitlines()
        if "st.global.f32 [$2]" in line and not line.lstrip().startswith("#")
    ]
    # One call per asm, each result indexing the call's value.
    assert len(calls) == 2, code
    assert not any(")[" in line for line in calls), code
