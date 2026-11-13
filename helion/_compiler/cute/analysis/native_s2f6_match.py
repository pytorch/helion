"""Recognize the integer BF16 lattice used by the authored S2F6 decoder.

This semantic leaf proves the decomposed operators, integer widths, constants
and shared dataflow. A value-level emitter must preserve broadcasting and the
scale-255 result bits 0x7fc0, and check hardware eligibility. An emitter that
changes loads must also prove raw/scale addressing and grouping. Recognition
alone does not admit an attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING

import torch

from ....language._tracing_ops import _new_var
from .match_common import node_tensor

if TYPE_CHECKING:
    from collections.abc import Callable

_aten = torch.ops.aten
_convert = torch.ops.prims.convert_element_type.default


@dataclass(frozen=True)
class NativeS2F6Decode:
    raw: torch.fx.Node
    scale: torch.fx.Node
    raw_int32: torch.fx.Node
    scale_int32: torch.fx.Node
    value: torch.fx.Node
    # Matched nodes and observed resolver/wrapper endpoints. An opaque caller
    # resolver may traverse additional internal nodes, which it must prove safe.
    nodes: frozenset[torch.fx.Node]


@dataclass(frozen=True, eq=False)
class _Input:
    dtype: torch.dtype


@dataclass(frozen=True, eq=False)
class _Constant:
    value: int
    dtype: torch.dtype
    full: bool = False


@dataclass(frozen=True, eq=False)
class _Op:
    target: object
    args: tuple[object, ...]
    dtype: torch.dtype


def _recipe() -> tuple[_Input, _Input, _Op, _Op, _Op]:
    raw = _Input(torch.int8)
    scale = _Input(torch.uint8)

    def op(target: object, *args: object, dtype: torch.dtype = torch.int32) -> _Op:
        return _Op(target, args, dtype)

    signed = op(_convert, raw, torch.int32)
    exponent = op(_convert, scale, torch.int32)
    magnitude = op(_aten.abs.default, signed)
    top: _Constant | _Op = _Constant(0, torch.int32, full=True)
    for bit in range(1, 8):
        predicate = op(_aten.ge.Scalar, magnitude, 2**bit, dtype=torch.bool)
        top = op(_aten.where.self, predicate, _Constant(bit, torch.int32), top)
    biased = op(_aten.sub.Tensor, op(_aten.add.Tensor, top, exponent), 6)
    mantissa = op(
        _aten.sub.Tensor,
        op(_aten.__lshift__.Tensor, magnitude, op(_aten.sub.Tensor, 7, top)),
        128,
    )
    normal = op(_aten.add.Tensor, op(_aten.__lshift__.Scalar, biased, 7), mantissa)
    subnormal = op(
        _aten.__lshift__.Tensor, magnitude, op(_aten.clamp_max.default, exponent, 6)
    )
    bits = op(
        _aten.where.self,
        op(_aten.le.Scalar, biased, 0, dtype=torch.bool),
        subnormal,
        normal,
    )
    bits = op(_aten.clamp_max.default, op(_aten.clamp_min.default, bits, 0), 0x7F7F)
    bits = op(
        _aten.where.self,
        op(_aten.eq.Scalar, magnitude, 0, dtype=torch.bool),
        _Constant(0, torch.int32),
        bits,
    )
    sign = op(
        _aten.where.self,
        op(_aten.lt.Scalar, signed, 0, dtype=torch.bool),
        _Constant(0x8000, torch.int64),
        _Constant(0, torch.int64),
        dtype=torch.int64,
    )
    bits = op(_aten.bitwise_or.Tensor, bits, sign, dtype=torch.int64)
    bits = op(
        _aten.where.self,
        op(_aten.eq.Scalar, exponent, 255, dtype=torch.bool),
        _Constant(0x7FC0, torch.int64),
        bits,
        dtype=torch.int64,
    )
    result = op(
        _aten.view.dtype,
        op(_convert, bits, torch.int16, dtype=torch.int16),
        torch.bfloat16,
        dtype=torch.bfloat16,
    )
    return raw, scale, signed, exponent, result


_RAW, _SCALE, _SIGNED, _EXPONENT, _RESULT = _recipe()


def match_native_s2f6_decode(
    value: torch.fx.Node,
    *,
    resolve: Callable[[torch.fx.Node], torch.fx.Node] | None = None,
) -> NativeS2F6Decode | None:
    """Match a BF16 bit reinterpretation, optionally resolving lifted operands.

    ``resolve`` is the caller's existing cross-graph operand resolver. Without
    it, lifted constants/operations fail; typed raw/scale placeholders may match.
    Callers must prove any semantics removed by their resolver. Value-level
    rewrites may reuse the matched casts directly; load-transforming callers
    must additionally check both inputs against their expected accesses. No
    graph is mutated. Aliases and tracing renames may wrap operands; arithmetic
    is not simplified.
    """
    matched: dict[object, torch.fx.Node] = {}
    nodes: set[torch.fx.Node] = set()

    def unwrap(node: object) -> object:
        seen: set[torch.fx.Node] = set()
        while isinstance(node, torch.fx.Node) and node not in seen:
            seen.add(node)
            if resolve is not None:
                resolved = resolve(node)
                if resolved is not node:
                    nodes.add(node)
                    node = resolved
                    continue
            if (
                node.op != "call_function"
                or node.target not in (_aten.alias.default, _new_var)
                or len(node.args) != 1
                or node.kwargs
            ):
                break
            tensor = node_tensor(node)
            source = node_tensor(node.args[0])
            if tensor is None or source is None or tensor.dtype != source.dtype:
                break
            nodes.add(node)
            node = node.args[0]
        return node

    def match(pattern: object, candidate: object) -> bool:
        if not isinstance(pattern, (_Input, _Constant, _Op)):
            # In particular, bool/float constants do not stand in for integers.
            return type(candidate) is type(pattern) and candidate == pattern
        candidate = unwrap(candidate)
        tensor = node_tensor(candidate)
        if (
            not isinstance(candidate, torch.fx.Node)
            or tensor is None
            or tensor.dtype != pattern.dtype
        ):
            return False
        if pattern in matched:
            return matched[pattern] is candidate
        matched[pattern] = candidate
        nodes.add(candidate)
        if isinstance(pattern, _Input):
            return True
        if candidate.op != "call_function":
            return False
        if isinstance(pattern, _Constant):
            expected = (
                _aten.full.default if pattern.full else _aten.scalar_tensor.default
            )
            index = 1 if pattern.full else 0
            if candidate.target is not expected or len(candidate.args) != index + 1:
                return False
            literal = candidate.args[index]
            if type(literal) is not int:
                return False
            return (
                literal == pattern.value
                and candidate.kwargs.get("dtype") is pattern.dtype
                and candidate.kwargs.get("layout", torch.strided) is torch.strided
                and candidate.kwargs.get("pin_memory", False) is False
                and not candidate.kwargs.keys()
                - {"dtype", "layout", "device", "pin_memory"}
                and (pattern.full or tensor.ndim == 0)
            )
        return (
            candidate.target is pattern.target
            and len(candidate.args) == len(pattern.args)
            and not candidate.kwargs
            and all(starmap(match, zip(pattern.args, candidate.args, strict=True)))
        )

    if not match(_RESULT, value):
        return None
    return NativeS2F6Decode(
        matched[_RAW],
        matched[_SCALE],
        matched[_SIGNED],
        matched[_EXPONENT],
        value,
        frozenset(nodes),
    )
