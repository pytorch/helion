"""CPU data/layout and bounded numerical references for fragment storage tests."""

from __future__ import annotations

from itertools import starmap
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

from .cute_paged_conditional_semantics import written_conditional_attention
import helion.language as hl


def storage_view(value: torch.Tensor) -> torch.Tensor:
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def move_backings(args: tuple[object, ...], device: torch.device) -> tuple[object, ...]:
    copies: dict[int, torch.Tensor] = {}
    result: list[object] = []
    for value in args:
        if not isinstance(value, torch.Tensor):
            result.append(value)
            continue
        identity = value.untyped_storage()._cdata
        if identity not in copies:
            copies[identity] = storage_view(value).to(device, copy=True)
        result.append(
            torch.empty(0, dtype=value.dtype, device=device).set_(
                copies[identity].untyped_storage(),
                value.storage_offset(),
                value.shape,
                value.stride(),
            )
        )
    return tuple(result)


def copy_backings(target: tuple[object, ...], source: tuple[object, ...]) -> None:
    done: set[int] = set()
    for dst, src in zip(target, source, strict=True):
        if not isinstance(dst, torch.Tensor):
            assert dst == src
            continue
        assert isinstance(src, torch.Tensor)
        assert (dst.dtype, dst.shape, dst.stride(), dst.storage_offset()) == (
            src.dtype,
            src.shape,
            src.stride(),
            src.storage_offset(),
        )
        identity = dst.untyped_storage()._cdata
        if identity not in done:
            storage_view(dst).copy_(storage_view(src))
            done.add(identity)


def backing_snapshots(args: tuple[object, ...]) -> list[torch.Tensor]:
    seen: set[int] = set()
    result = []
    for value in args:
        if (
            isinstance(value, torch.Tensor)
            and value.untyped_storage()._cdata not in seen
        ):
            seen.add(value.untyped_storage()._cdata)
            result.append(storage_view(value).cpu().clone())
    return result


def raw_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return (
        lhs.dtype == rhs.dtype
        and lhs.shape == rhs.shape
        and torch.equal(
            lhs.contiguous().view(torch.uint8), rhs.contiguous().view(torch.uint8)
        )
    )


def assert_attention_result(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> None:
    assert len(actual) == len(expected) == 2
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        assert value.shape == reference.shape and value.dtype == reference.dtype
        for classification in (torch.isnan, torch.isposinf, torch.isneginf):
            assert torch.equal(classification(value), classification(reference))
        finite = torch.isfinite(reference)
        # This bounded authored-source CPU oracle is not a bitwise GPU GEMM,
        # exp2, reciprocal or log2 reference. Check every finite coordinate.
        torch.testing.assert_close(
            value[finite],
            reference[finite],
            rtol=0.02 if index == 0 else 0.002,
            atol=0.01 if index == 0 else 0.002,
        )


def _fma(lhs: float, rhs: float, addend: float) -> float:
    # _cpu_fma supplies FP32 operands. Their product is exact in FP64,
    # including the full finite FP32 exponent range, so this has precisely
    # the previous math.fma binary64 result before the existing FP32 cast.
    # Invalid infinity operations produce NaN, as the old ValueError path did.
    return lhs * rhs + addend


def _cpu_fma(
    asm: str,
    constraints: str,
    args: list[torch.Tensor],
    dtype: torch.dtype,
    is_pure: bool,
    pack: int,
) -> torch.Tensor:
    assert asm == "fma.rn.f32 $0, $1, $2, $3;"
    assert (
        constraints == "=f,f,f,f" and dtype == torch.float32 and is_pure and pack == 1
    )
    assert all(value.dtype == torch.float32 for value in args)
    a, b, c = torch.broadcast_tensors(*args)
    values = starmap(
        _fma,
        zip(
            a.flatten().tolist(),
            b.flatten().tolist(),
            c.flatten().tolist(),
            strict=True,
        ),
    )
    return torch.tensor(list(values), dtype=torch.float32).reshape(a.shape)


def attention_cpu(args: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    assert all(
        not isinstance(value, torch.Tensor) or value.device.type == "cpu"
        for value in args
    )

    def dot(
        a: torch.Tensor, b: torch.Tensor, acc: torch.Tensor | None = None
    ) -> torch.Tensor:
        product = a.float() @ b.float()
        return product if acc is None else product + acc

    with patch.multiple(
        hl,
        specialize=lambda value: value,
        tile=lambda count, block_size: [
            SimpleNamespace(id=torch.tensor(i, dtype=torch.int32)) for i in range(count)
        ],
        arange=lambda count: torch.arange(count, dtype=torch.int32),
        grid=lambda count: range(int(count)),
        zeros=torch.zeros,
        full=torch.full,
        dot=dot,
        inline_asm_elementwise=_cpu_fma,
    ):
        return cast(
            "tuple[torch.Tensor, torch.Tensor]", written_conditional_attention.fn(*args)
        )
