"""Authored complete-head lazy rescaling over the existing postdecode recurrence."""

from __future__ import annotations

import ast
import copy
import functools
from typing import TYPE_CHECKING
from typing import cast

import torch

import helion
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_paged_rowwise_lazy import paged_query_first_direct_attention
from helion._testing.cute_paged_rowwise_native import check_paged_recurrence_native
from helion._testing.cute_paged_rowwise_native import configure_rowwise_case
from helion._testing.cute_paged_rowwise_native import rowwise_config
from helion._testing.cute_paged_rowwise_native import rowwise_reference
from helion._testing.cute_source import transform_kernel
import helion.language as hl

if TYPE_CHECKING:
    from pathlib import Path

    from helion.runtime.kernel import Kernel


class _HeadwideSource(ast.NodeTransformer):
    def __init__(self) -> None:
        self.tiles = 0
        self.votes = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if ast.unparse(node.func) == "hl.tile":
            assert ast.unparse(node.args[0]) == "[items, rows]"
            (keyword,) = node.keywords
            assert keyword.arg == "block_size"
            assert ast.unparse(keyword.value) == "[1, None]"
            keyword.value = ast.parse(
                "[1, helion.next_power_of_2(rows)]", mode="eval"
            ).body
            self.tiles += 1
        elif (
            ast.unparse(node.func) == "torch.where"
            and ast.unparse(node.args[0]) == "needs_rescale"
        ):
            assert [ast.unparse(arg) for arg in node.args] == [
                "needs_rescale",
                "proposed_maximum",
                "maximum",
            ]
            # max over exact Boolean 0/1 values is a full-head any. The
            # ordinary max reduction masks padded rows with negative infinity.
            node.args[0] = ast.parse(
                "torch.amax(needs_rescale.float(), dim=0) > 0.0", mode="eval"
            ).body
            self.votes += 1
        return node


@functools.cache
def headwide_kernel() -> Kernel:
    transform = _HeadwideSource()
    source = transform_kernel(
        paged_query_first_direct_attention,
        transform,
        name="paged_headwide_rescale",
        filename="<paged_headwide_rescale>",
    )
    assert (transform.tiles, transform.votes) == (1, 1)
    return helion.kernel(
        source.fn, settings=copy.deepcopy(paged_query_first_direct_attention.settings)
    )


def headwide_config(query_len: int, *, scoped: bool = False) -> helion.Config:
    config = rowwise_config(query_len)
    # The complete logical head is fixed in source, so this tile has no search slot.
    config.config["block_sizes"] = []
    if scoped:
        config.config.update(
            cute_mma_dag_pass_warps="attention_scoped",
            cute_min_blocks_per_mp=2,
            cute_mma_dag_probability_schedule="serial",
            cute_mma_dag_query_tma=False,
            cute_mma_dag_v_lag=0,
            cute_mma_dag_value_metadata_skip=False,
        )
    return config


def configure_headwide_case(args: tuple[object, ...], case: str) -> None:
    configure_rowwise_case(args, case)
    if case == "changed_q":
        cast("torch.Tensor", args[6]).copy_(
            cast("torch.Tensor", args[6]).new_tensor([128, 0])
        )
    elif case == "mixed_rows":
        # Only the first head has changing page maxima; the second head must
        # retain its independent vote and state despite the shared CTA.
        k = cast("torch.Tensor", args[1])
        k[:, :, 1, 0] = 1


def check_headwide_native(
    query_len: int, scoped: bool, device: torch.device, tmp_path: Path
) -> None:
    args_list = list(folded_attention_args(query_len, device=device))
    args_list[7] = 2047
    args = tuple(args_list)
    configure_headwide_case(args, "original")
    check_paged_recurrence_native(
        args,
        headwide_kernel(),
        headwide_config(query_len, scoped=scoped),
        functools.partial(rowwise_reference, headwide=True),
        tmp_path / f"headwide-ql{query_len}-{'scoped' if scoped else 'paired'}",
        (
            "headwide_rescale=True",
            "rowwise_rescale_threshold=2.0",
            "paged_attention_partial_body("
            if scoped
            else "paged_attention_paired_body(",
        ),
        configure=configure_headwide_case,
    )


@helion.kernel(backend="cute", static_shapes=True)
def fixed_root_matmul(
    a: torch.Tensor, b: torch.Tensor, block: hl.constexpr
) -> torch.Tensor:
    output = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=torch.float32)
    for m, n in hl.tile([a.size(0), b.size(1)], block_size=[block, 16]):
        output[m, n] = hl.dot(a[m, :], b[:, n])
    return output
