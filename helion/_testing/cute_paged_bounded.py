"""Explicit direct-score partial statistics and a bounded two-part merge source."""

from __future__ import annotations

import ast
import copy
import functools
import math
from typing import cast

import torch

from .cute_folded_attention import folded_attention_args
from .cute_folded_attention import folded_paged_attention
from .cute_source import transform_kernel
import helion
import helion.language as hl


class _BoundedPartialSource(ast.NodeTransformer):
    """Keep the shared address/typed-mask source and replace its written math."""

    def __init__(self, *, alias_statistics: bool = False) -> None:
        self.changed: dict[str, int] = {}
        self.alias_statistics = alias_statistics

    def record(self, name: str) -> None:
        self.changed[name] = self.changed.get(name, 0) + 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        position = (
            next(i for i, arg in enumerate(node.args.args) if arg.arg == "out") + 1
        )
        node.args.args[position:position] = [
            ast.arg("segment_begin", ast.parse("torch.Tensor", mode="eval").body),
            ast.arg("segment_end", ast.parse("torch.Tensor", mode="eval").body),
        ]
        node.returns = ast.parse(
            "tuple[torch.Tensor, torch.Tensor, torch.Tensor]", mode="eval"
        ).body
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        target = node.targets[0]
        if isinstance(target, ast.Name):
            name = target.id
            if name == "window_left":
                self.record(name)
                return ast.If(
                    ast.Compare(
                        ast.Name("window_left", ast.Load()),
                        [ast.IsNot()],
                        [ast.Constant(None)],
                    ),
                    [node],
                    [],
                )
            if name == "scale":
                node.value = ast.BinOp(
                    node.value, ast.Mult(), ast.Constant(math.log2(math.e) * 64)
                )
            elif name in ("key", "value"):
                exponent = -6 if name == "key" else -22
                node.value = ast.BinOp(
                    node.value, ast.Mult(), ast.Constant(2.0**exponent)
                )
            elif name in ("alpha", "probability"):
                assert isinstance(node.value, ast.Call)
                assert isinstance(node.value.func, ast.Attribute)
                assert node.value.func.attr == "exp"
                node.value.func.attr = "exp2"
                node.value = ast.Call(
                    ast.Attribute(
                        ast.Name("torch", ast.Load()), "clamp_max", ast.Load()
                    ),
                    [node.value, ast.Constant(1.0)],
                    [],
                )
            elif name in ("begin", "end"):
                operation = "maximum" if name == "begin" else "minimum"
                bound = "segment_begin" if name == "begin" else "segment_end"
                node.value = ast.Call(
                    ast.Attribute(ast.Name("torch", ast.Load()), operation, ast.Load()),
                    [node.value, ast.parse(f"{bound}[request]", mode="eval").body],
                    [],
                )
            elif name == "lse":
                maximum = copy.deepcopy(node)
                maximum.targets = [ast.Name("maximum_output", ast.Store())]
                denominator = copy.deepcopy(node)
                denominator.targets = [ast.Name("denominator_output", ast.Store())]
                if self.alias_statistics:
                    denominator.value = ast.Name("maximum_output", ast.Load())
                self.record(name)
                return [maximum, denominator]
            elif name == "lse_storage":
                self.record(name)
                return ast.parse(
                    "maximum_storage = maximum_output.view(-1)\n"
                    "denominator_storage = denominator_output.view(-1)"
                ).body
            else:
                return node
            self.record(name)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            if target.value.id == "output_storage":
                self.record("raw_output")
                node.value = ast.Name("accumulator", ast.Load())
            elif target.value.id == "lse_storage":
                self.record("statistics")
                maximum = copy.deepcopy(node)
                cast("ast.Subscript", maximum.targets[0]).value = ast.Name(
                    "maximum_storage", ast.Load()
                )
                maximum.value = ast.Name("maximum", ast.Load())
                denominator = copy.deepcopy(node)
                cast("ast.Subscript", denominator.targets[0]).value = ast.Name(
                    "denominator_storage", ast.Load()
                )
                denominator.value = ast.Name("denominator", ast.Load())
                return (
                    [denominator, maximum]
                    if self.alias_statistics
                    else [maximum, denominator]
                )
        return node

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        if isinstance(node.test, ast.Name) and node.test.id == "return_lse":
            return node.body
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        return ast.Return(
            ast.Tuple(
                [
                    ast.Name(name, ast.Load())
                    for name in ("out", "maximum_output", "denominator_output")
                ],
                ast.Load(),
            )
        )


@functools.cache
def bounded_partial_kernel(*, alias_statistics: bool = False) -> helion.Kernel:
    transform = _BoundedPartialSource(alias_statistics=alias_statistics)
    suffix = "_aliased" if alias_statistics else ""
    kernel = transform_kernel(
        folded_paged_attention,
        transform,
        name=f"bounded_direct_attention_partial{suffix}",
        filename=f"<helion_bounded_direct_attention_partial_v1{suffix}>",
    )
    assert transform.changed == {
        "window_left": 1,
        "lse": 1,
        "lse_storage": 1,
        "scale": 1,
        "begin": 1,
        "end": 1,
        "key": 1,
        "value": 1,
        "alpha": 1,
        "probability": 2,
        "raw_output": 1,
        "statistics": 1,
    }
    return kernel


def bounded_partial_fixture(
    query_len: int,
    device: torch.device | str = "cpu",
    *,
    alias_statistics: bool = False,
    reverse_pages: bool = True,
) -> tuple[helion.Kernel, tuple[object, ...]]:
    args = list(folded_attention_args(query_len, device=device))
    assert isinstance(args[0], torch.Tensor)
    assert isinstance(args[5], torch.Tensor)
    assert isinstance(args[10], torch.Tensor)
    args[9] = True
    args[10] = torch.empty(args[10].shape, dtype=torch.float32, device=device)
    batch = args[0].shape[0]
    begin = torch.zeros(batch, dtype=torch.int32, device=device)
    end = torch.full((batch,), args[5].shape[1], dtype=torch.int32, device=device)
    return bounded_partial_kernel(alias_statistics=alias_statistics), (
        *args,
        begin,
        end,
        False,
        reverse_pages,
    )


def bounded_merge_args(device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    """Mass, empty, nonfinite-product, finite-clamp, and invalid-maximum rows."""
    maximum0 = torch.tensor(
        [2**28, -math.inf, 0, 0, 0, 0, math.inf], device=device
    ).reshape(7, 1, 1)
    maximum1 = torch.tensor(
        [2**28, -math.inf, -math.inf, 0, 0, 0, 0], device=device
    ).reshape(7, 1, 1)
    denominator0 = torch.tensor(
        [1, 0, 1, 1, 1, 1, 1], dtype=torch.float32, device=device
    ).reshape(7, 1, 1)
    denominator1 = torch.tensor(
        [3, 0, 0, 1, 1, 1, 1], dtype=torch.float32, device=device
    ).reshape(7, 1, 1)
    raw0 = (
        torch.tensor(
            [64 * 2.0**-22, 0, 1, math.inf, 2.0**110, -(2.0**110), 1], device=device
        )
        .reshape(7, 1, 1, 1)
        .expand(7, 1, 1, 128)
        .contiguous()
    )
    raw1 = (
        torch.tensor([0, 0, math.nan, 0, 0, 0, 0], device=device)
        .reshape(7, 1, 1, 1)
        .expand_as(raw0)
        .contiguous()
    )
    return raw0, maximum0, denominator0, raw1, maximum1, denominator1


@helion.kernel(backend="cute", static_shapes=True)
def bounded_partial_merge(
    raw0: torch.Tensor,
    maximum0: torch.Tensor,
    denominator0: torch.Tensor,
    raw1: torch.Tensor,
    maximum1: torch.Tensor,
    denominator1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge raw scaled FP32 statistics before the one final BF16 rounding.

    The finite-only clamp is an explicit source operation. Inf/NaN paths and
    zero times nonfinite products remain evaluated by the written arithmetic.
    """
    dim = raw0.shape[-1]
    rows = maximum0.numel()
    first = raw0.reshape(rows, dim)
    second = raw1.reshape(rows, dim)
    m0 = maximum0.reshape(rows)
    m1 = maximum1.reshape(rows)
    l0 = denominator0.reshape(rows)
    l1 = denominator1.reshape(rows)
    output = torch.empty_like(raw0, dtype=torch.bfloat16)
    output_rows = output.reshape(rows, dim)
    lse = torch.empty_like(maximum0)
    lse_rows = lse.reshape(rows)
    for row in hl.tile(rows):
        maximum = torch.maximum(m0[row], m1[row])
        safe_maximum = torch.where(maximum == -float("inf"), 0.0, maximum)
        a0 = torch.clamp_max(torch.exp2(m0[row] - safe_maximum), 1.0)
        a1 = torch.clamp_max(torch.exp2(m1[row] - safe_maximum), 1.0)
        total = l0[row] * a0 + l1[row] * a1
        safe_total = torch.where(total == 0, 1.0, total)
        numerator = first[row, :] * a0[:, None] + second[row, :] * a1[:, None]
        mean = numerator / safe_total[:, None]
        clipped = torch.clamp(mean, -254.0 * 2.0**98, 254.0 * 2.0**98)
        finite_mean = torch.where(torch.isfinite(mean), clipped, mean)
        output_rows[row, :] = finite_mean * 2.0**22
        lse_rows[row] = maximum * math.log(2) + torch.log(total)
    return output, lse
