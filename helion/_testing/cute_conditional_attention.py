"""CPU-only projection of the first phase of authored conditional attention.

This fixture isolates branch/phi and recurrence admission from the separate
two-root dependency proof. It preserves the original host prelude and complete
first device root. Only the second root is removed and the return names the
first phase's partial storage. Unwritten whole-owner slots make this a compiler
fixture, not a numerical kernel or an alternative public attention API.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import inspect
import json
import sys
from typing import TYPE_CHECKING

import torch

from .cute_paged_conditional_semantics import written_conditional_attention
from .cute_source import function_from_ast
import helion

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from pathlib import Path
    from types import FrameType

    from _typeshed import TraceFunction

    from helion._compiler.device_ir import GraphInfo
    from helion._compiler.generate_ast import GenerateAST


@contextmanager
def observe_fragment_discovery(
    directory: Path,
    on_codegen_graphs: Callable[[GenerateAST, list[GraphInfo]], None] | None = None,
) -> Iterator[list[dict[str, object]]]:
    """Save the real discovery inputs and return sites without replacing it."""
    from helion._compiler.cute.computed_fragment import codegen_computed_fragment_root
    from helion._compiler.cute.computed_fragment import computed_fragment_supported

    records: list[dict[str, object]] = []
    code = computed_fragment_supported.__code__
    previous = sys.gettrace()
    assert previous is None

    def observe(frame: FrameType, event: str, argument: object) -> TraceFunction | None:
        if frame.f_code is not code:
            return None
        if event == "call":
            parent = frame.f_back
            final_codegen = (
                on_codegen_graphs is not None
                and parent is not None
                and parent.f_code is codegen_computed_fragment_root.__code__
            )
            destination = directory / "codegen" if final_codegen else directory
            destination.mkdir(exist_ok=True)
            metadata = []
            for info in frame.f_locals["graphs"]:
                name = f"graph-{info.graph_id}-{type(info).__name__}.txt"
                (destination / name).write_text(str(info.graph))
                metadata.append(
                    {
                        "id": info.graph_id,
                        "type": type(info).__name__,
                        "fields": {
                            key: repr(value) for key, value in info.kwargs().items()
                        },
                        "nodes": [
                            {
                                "name": node.name,
                                "target": str(node.target),
                                "lowering": type(node.meta.get("lowering")).__name__,
                                "val": repr(node.meta.get("val")),
                            }
                            for node in info.graph.nodes
                        ],
                    }
                )
            (destination / "graph-metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            if final_codegen:
                assert on_codegen_graphs is not None and parent is not None
                on_codegen_graphs(parent.f_locals["cg"], frame.f_locals["graphs"])
        elif event == "return":
            records.append({"line": frame.f_lineno, "result": argument})
        return observe

    sys.settrace(observe)
    try:
        yield records
    finally:
        sys.settrace(previous)


def _prefix_function() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    function = ast.parse(inspect.getsource(written_conditional_attention.fn)).body[0]
    assert isinstance(function, ast.FunctionDef)
    roots = [node for node in function.body if isinstance(node, ast.For)]
    assert len(roots) == 2
    first_key = ast.dump(roots[0], include_attributes=False)
    function.decorator_list = []
    function.name = "conditional_online_attention_prefix"
    function.body.remove(roots[1])
    assert isinstance(function.body[-1], ast.Return)
    function.body[-1].value = ast.Tuple(
        elts=[
            ast.Name(id="partial_o", ctx=ast.Load()),
            ast.Name(id="partial_ml", ctx=ast.Load()),
        ],
        ctx=ast.Load(),
    )
    assert (
        ast.dump(
            next(node for node in function.body if isinstance(node, ast.For)),
            include_attributes=False,
        )
        == first_key
    )
    ast.fix_missing_locations(function)
    return function_from_ast(
        function,
        written_conditional_attention.fn.__globals__,
        "<conditional-online-attention-prefix>",
        copy_namespace=False,
    )


conditional_online_attention_prefix = helion.kernel(
    _prefix_function(), backend="cute", static_shapes=True, fast_math=True
)


def conditional_attention_args() -> tuple[object, ...]:
    """The existing three-request bound fixture, with original padded aliases."""
    batch = 3
    generator = torch.Generator().manual_seed(461)
    storage = torch.randn(batch * 18432 + 8, generator=generator).bfloat16()
    q = storage.as_strided((batch, 4, 16, 128), (18432, 4608, 128, 1), 8)
    packed = []
    for _ in range(2):
        raw = torch.randint(-8, 8, (7 * 33792,), generator=generator, dtype=torch.int8)
        data = raw.as_strided((7, 128, 2, 128), (33792, 256, 128, 1))
        scale = raw.view(torch.uint8).as_strided(
            (7, 128, 2, 4), (33792, 4, 512, 1), 32768
        )
        scale.fill_(127)
        packed.append((data, scale))
    table = (torch.arange(batch * 32).reshape(batch, 32) % 7).to(torch.int32)
    lengths = torch.tensor([0, 2051, 128], dtype=torch.int32)
    output = torch.empty((batch, 4, 16, 128), dtype=torch.bfloat16)
    lse = torch.empty((batch, 4, 16), dtype=torch.float32)
    return (
        q,
        packed[0][0],
        packed[1][0],
        packed[0][1],
        packed[1][1],
        table,
        lengths,
        output,
        lse,
        2,
        18,
        2047,
    )
