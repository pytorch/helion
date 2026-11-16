"""Scalar pointwise lowering shared by the CuTe register, fragment and epilogue emitters.

Every register-resident emitter re-enters Inductor's pointwise codegen one
scalar at a time, supplying its own operand ASTs: ``scalar_lowering_context``
builds the ``LoweringContext`` those calls read and ``lower_pointwise_scalar``
runs one node through it, returning the lowering's raw result for the caller
to validate and cast.  ``ScalarLoweringContextCache`` reuses the invariant
Inductor graph / kernel handlers within one register program.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from typing import cast

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
    from collections.abc import Mapping

    from torch.fx import Graph
    from torch.fx import Node
    from torch.fx.node import Argument

    from ..aten_lowering import LoweringContext
    from ..generate_ast import GenerateAST
    from ..helper_function import CodegenInterface
    from ..inductor_lowering import PointwiseLowering


def scalar_lowering_context(
    cg: CodegenInterface, env: Mapping[Node, object]
) -> LoweringContext:
    """A ``LoweringContext`` over *cg* whose fx environment is *env*.

    The scalar emitters lower one node with explicitly supplied operand
    ASTs instead of the device function's environment; this is the context
    Inductor's ``codegen_from_input_asts`` / ``codegen`` read.
    """
    from ..aten_lowering import LoweringContext

    ctx = LoweringContext.__new__(LoweringContext)
    ctx.cg = cg
    ctx.env = cast("dict[Node, Argument]", env)
    return ctx


def lower_pointwise_scalar(
    cg: GenerateAST,
    node: Node,
    lowering: PointwiseLowering,
    input_asts: list[ast.AST],
    *,
    env: Mapping[Node, object],
    statements: list[ast.AST],
) -> object:
    """Run Inductor's pointwise codegen for *node* over *input_asts*.

    The statements the lowering emits are collected into *statements*.  The
    RAW result is returned: each caller decides what a non-AST result means
    (a decline, an unsupported error) and whether to cast it to the node's
    dtype.
    """
    ctx = scalar_lowering_context(cg, env)
    with cg.set_statements(statements), V.set_current_node(node):
        return lowering.codegen_from_input_asts(ctx, node, input_asts)


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
