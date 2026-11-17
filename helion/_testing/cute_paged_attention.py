"""Small paged online-attention source for semantic recognition tests."""

from __future__ import annotations

import ast
import copy
import functools
import inspect
from typing import TYPE_CHECKING

import torch

from .cute_source import function_from_ast
import helion
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.cute.cute_mma_dag import MmaDagProgram
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True)
def paged_online_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    pages: torch.Tensor,
    lengths: torch.Tensor,
    query_len: int,
    group: int,
    window: int,
    bounds: torch.Tensor | None = None,
    per_key_denominator: hl.constexpr = False,  # pyrefly: ignore [bad-function-definition]
    return_lse: hl.constexpr = True,  # pyrefly: ignore [bad-function-definition]
    reference_scores: hl.constexpr = False,  # pyrefly: ignore [bad-function-definition]
    reverse_pages: hl.constexpr = False,  # pyrefly: ignore [bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    query_len = hl.specialize(query_len)
    group = hl.specialize(group)
    window = hl.specialize(window)
    items, rows, dim = q.shape
    _, subpages, width, heads, _ = k.shape
    scale_group = dim // k_scale.shape[-1]
    page_size = subpages * width
    out = torch.empty((items, rows, dim), dtype=q.dtype, device=q.device)
    lse = (
        torch.empty((items, rows), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    for item_tile, row in hl.tile([items, rows], block_size=[1, None]):
        item = item_tile.id
        request = item // heads
        head = item % heads
        query_position = lengths[request] - query_len + row.index // group
        length = lengths[request]
        begin = (
            torch.clamp(length - query_len - window, min=0)
            if window >= 0
            else length * 0
        )
        if bounds is not None:
            first_page = bounds[request, 0]
            last_page = bounds[request, 1]
        else:
            first_page = begin // page_size
            last_page = (length + page_size - 1) // page_size
        query = q[item, row, :]
        maximum = hl.full([row], -float("inf"), torch.float32)
        denominator = hl.full(
            [row, width] if per_key_denominator else [row], 0.0, torch.float32
        )
        accumulator = hl.full([row, dim], 0.0, torch.float32)
        for iteration in hl.grid(first_page * subpages, last_page * subpages):
            tile = (
                (first_page + last_page) * subpages - 1 - iteration
                if reverse_pages
                else iteration
            )
            physical_page = pages[request, tile // subpages]
            subpage = tile % subpages
            key = (
                k[physical_page, subpage, :, head, :].to(torch.float32)
                * torch.exp2(
                    k_scale[
                        physical_page, subpage, :, head, hl.arange(dim) // scale_group
                    ].to(torch.float32)
                    - 127
                )
            ).to(q.dtype)
            value = (
                v[physical_page, subpage, :, head, :].to(torch.float32)
                * torch.exp2(
                    v_scale[
                        physical_page, subpage, :, head, hl.arange(dim) // scale_group
                    ].to(torch.float32)
                    - 127
                )
            ).to(q.dtype)
            scores = hl.dot(query, key.T) * dim**-0.5
            reference = maximum
            if reference_scores:
                reference = torch.where(
                    (maximum >= 0.0) & (maximum < float("inf")), maximum, 0.0
                )
                scores = scores - reference[:, None]
            key_position = hl.arange(width) + tile * width
            visible = key_position[None, :] <= query_position[:, None]
            if window >= 0:
                visible = visible & (
                    key_position[None, :] >= query_position[:, None] - window
                )
            scores = torch.where(visible, scores, -float("inf"))
            if reference_scores:
                next_maximum = torch.maximum(
                    maximum, torch.amax(scores, dim=1) + reference
                )
            else:
                next_maximum = torch.maximum(maximum, torch.amax(scores, dim=1))
            safe_maximum = torch.where(next_maximum == -float("inf"), 0.0, next_maximum)
            alpha = torch.exp(maximum - safe_maximum)
            if reference_scores:
                probability = torch.exp(scores + (reference - safe_maximum)[:, None])
            else:
                probability = torch.exp(scores - safe_maximum[:, None])
            if per_key_denominator:
                denominator = denominator * alpha[:, None] + probability
            else:
                denominator = denominator * alpha + torch.sum(probability, dim=1)
            accumulator = hl.dot(
                probability.to(q.dtype), value, acc=accumulator * alpha[:, None]
            )
            maximum = next_maximum
        if per_key_denominator:
            final_denominator = torch.sum(denominator, dim=1)
        else:
            final_denominator = denominator
        safe_denominator = torch.where(final_denominator == 0, 1.0, final_denominator)
        out[item, row, :] = (accumulator / safe_denominator[:, None]).to(q.dtype)
        if lse is not None:
            lse[item, row] = maximum + torch.log(final_denominator)
    return out, lse


def copy_paged_attention_program(program: MmaDagProgram) -> MmaDagProgram:
    """Clone the root/body FX graphs without copying fake tensor storage."""
    assert program.loop_node is not None
    memo: dict[int, object] = {
        id(access.fake): access.fake for access in program.accesses
    }
    for graph in (program.loop_node.graph, program.dots[0].node.graph):
        copied = torch.fx.Graph()
        nodes: dict[torch.fx.Node, torch.fx.Node] = {}
        for node in graph.nodes:
            nodes[node] = copied.node_copy(node, nodes.__getitem__)
            memo[id(node)] = nodes[node]
        memo[id(graph)] = copied
    return copy.deepcopy(program, memo)


@functools.cache
def ordered_paged_online_attention() -> Kernel[
    tuple[torch.Tensor, torch.Tensor | None]
]:
    """Prepend a materialized query copy to the shared attention source.

    Building this CPU fixture from the original function keeps its semantic
    recurrence identical without maintaining a second copy of the algorithm.
    """
    function = ast.parse(inspect.getsource(paged_online_attention.fn)).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.decorator_list = []
    function.name = "ordered_paged_attention"
    # Explicit constexpr arguments avoid a host specialize call, which the
    # ordered planner intentionally excludes from its shape-only prelude.
    scalars = {"query_len", "group", "window"}
    for argument in function.args.args:
        if argument.arg in scalars:
            argument.annotation = ast.Attribute(
                ast.Name("hl", ast.Load()), "constexpr", ast.Load()
            )
    original_statement_count = len(function.body)
    function.body = [
        statement
        for statement in function.body
        if not (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in scalars
                for target in statement.targets
            )
        )
    ]
    assert original_statement_count - len(function.body) == len(scalars)
    # The ordered prelude requires unique assignment targets, including
    # discarded shape fields that the shared function names "_".
    for index, node in enumerate(ast.walk(function)):
        if (
            isinstance(node, ast.Name)
            and node.id == "_"
            and isinstance(node.ctx, ast.Store)
        ):
            node.id = f"unused_shape_{index}"
    first_root = next(
        i for i, node in enumerate(function.body) if isinstance(node, ast.For)
    )
    copy_statements = ast.parse(
        "copied_q = torch.empty_like(q)\n"
        "for copy_item in hl.tile(items, block_size=1):\n"
        "    copied_q[copy_item, :, :] = q[copy_item, :, :]\n"
        "hl.barrier()\n"
    ).body
    renamed_queries = 0
    for node in ast.walk(function.body[first_root]):
        if isinstance(node, ast.Name) and node.id == "q":
            node.id = "copied_q"
            renamed_queries += 1
    assert renamed_queries > 0
    function.body[first_root:first_root] = copy_statements
    return helion.kernel(
        function_from_ast(
            function, globals(), "<helion_testing_ordered_paged_attention>"
        ),
        backend="cute",
        static_shapes=True,
    )
