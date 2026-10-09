"""Reuse invariant Inductor handlers within one CuTe register program."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import sympy
from torch._inductor.codegen.simd import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.virtualized import V

from ..inductor_lowering import FakeGraphLowering
from ..inductor_lowering import GenerateASTFromInductor
from ..inductor_lowering import _patched_inductor_config

if TYPE_CHECKING:
    import ast
    from collections.abc import Iterator

    from torch.fx import Graph
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from ..inductor_lowering import PointwiseLowering


@dataclass
class _ScalarHandlers:
    graph: FakeGraphLowering
    kernel: TritonKernel


class ScalarLoweringContextCache:
    """Keep graph/kernel setup local to a program and its individual branches.

    Scalar operations still receive a fresh ops handler with their own argument
    ASTs and current FX node. Dtype conversions and the ordinary lowering are
    unchanged; only the empty graph and kernel handler constructors are reused.
    """

    def __init__(self, cg: GenerateAST) -> None:
        self.cg = cg
        self._handlers: dict[Graph, _ScalarHandlers] = {}

    @contextmanager
    def install(
        self,
        node: Node,
        arguments: dict[str, ast.AST],
        *,
        lowering: PointwiseLowering | None = None,
    ) -> Iterator[None]:
        with _patched_inductor_config():
            handlers = self._handlers.get(node.graph)
            graph = FakeGraphLowering() if handlers is None else handlers.graph
            load_check = None
            if lowering is not None:
                indices = [
                    sympy.Symbol(f"i{n}")
                    for n in range(len(lowering.buffer.data.ranges))
                ]
                load_check = partial(
                    lowering._check_position_preserving_load, node, indices
                )
            with (
                V.set_graph_handler(graph),
                V.set_ops_handler(
                    GenerateASTFromInductor(self.cg, arguments, load_check)
                ),
            ):
                if handlers is None:
                    handlers = _ScalarHandlers(
                        graph,
                        TritonKernel({}, features=SIMDKernelFeatures([], sympy.S.One)),
                    )
                    self._handlers[node.graph] = handlers
                with V.set_kernel_handler(handlers.kernel):
                    yield
