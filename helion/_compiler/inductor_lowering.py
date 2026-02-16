from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
from operator import getitem
from typing import TYPE_CHECKING
from typing import ContextManager
from typing import NamedTuple
from typing import cast

import sympy
import torch
from torch._dynamo.convert_frame import compile_lock
from torch._inductor import config as inductor_config
from torch._inductor import ir
from torch._inductor.codegen.simd import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import InputBuffer
from torch._inductor.ir import Pointwise
from torch._inductor.ir import Reduction
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TensorBox
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.utils import triton_type
from torch._inductor.virtualized import OpsValue
from torch._inductor.virtualized import V
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.fx.experimental import proxy_tensor
from torch.fx.experimental.sym_node import SymNode
from torch.fx.interpreter import Interpreter
from torch.fx.node import Node
from torch.fx.node import map_arg

from .. import exc
from ..exc import InductorLoweringError
from ..language._decorators import APIFunc
from ..language._decorators import is_api_func
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .aten_lowering import Lowering
from .aten_lowering import LoweringContext
from .aten_lowering import aten_lowering_dispatch
from .compile_environment import CompileEnvironment
from .compile_environment import FixedBlockSizeSource
from .device_function import VarInfo
from .device_function import contains_only_block_size_symbols
from .dtype_utils import cast_ast
from .node_masking import inductor_masked_value
from .node_masking import mask_node_inputs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils._ordered_set import OrderedSet

    from .. import Config
    from .backend import InductorOpOverrides
    from .device_function import DeviceFunction
    from .generate_ast import GenerateAST
    from .helper_function import CodegenInterface
    from .tile_dispatch import TileStrategyDispatch

INDUCTOR_PATCH: dict[str, object] = {
    # Allow implicit upcasts to FP32 for elementwise math correctness
    "triton.codegen_upcast_to_fp32": True,
    # Ensure Inductor preserves reductions (even tiny ones) as Reduction IR
    # so we can attach ReductionLowering instead of seeing pointwise fusions.
    "split_reductions": False,
    "unroll_reductions_threshold": 1,
}


def prepare_graph_lowerings(graph: torch.fx.Graph) -> None:
    with compile_lock:
        graph_lowering = GraphLowering(
            _LazyGraphModule({}, graph),
            shape_env=CompileEnvironment.current().shape_env,
        )

        with V.set_graph_handler(graph_lowering):
            for node in graph.nodes:
                assert node.op in {
                    "call_function",
                    "placeholder",
                    "output",
                }, node.op
                if node.op == "call_function":
                    with node.meta["location"]:
                        prepare_node_lowering(graph_lowering, node)


def prepare_node_lowering(
    graph_lowering: GraphLowering,
    node: Node,
) -> None:
    if is_api_func(api := node.target):
        APIFuncLowering.normalize_args_kwargs(api, node)
        node.meta["lowering"] = APIFuncLowering(api)
        return

    if node.target in aten_lowering_dispatch:
        node.meta["lowering"] = aten_lowering_dispatch[node.target](node)
        return

    if isinstance(
        val := node.meta["val"], (torch.SymInt, torch.SymFloat, torch.SymBool)
    ):
        node.meta["lowering"] = SympyExprLowering(val._sympy_())
        return

    # Track arguments to reuse names for duplicates
    arg_to_name: dict[Node, str] = {}

    def convert_arg(arg: Node) -> TensorBox:
        example = arg.meta["val"]

        # Reuse existing name for duplicate arguments
        if arg in arg_to_name:
            name = arg_to_name[arg]
        else:
            name = f"{node.name}_input{len(input_names)}"
            arg_to_name[arg] = name
            input_names.append(name)

        if isinstance(example, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            dtype = {
                torch.SymInt: torch.int64,
                torch.SymFloat: torch.float32,
                torch.SymBool: torch.bool,
            }[type(example)]
            result = TensorBox.create(
                InputBuffer(
                    name=name,
                    layout=FixedLayout(
                        CompileEnvironment.current().device,
                        dtype,
                        [],
                        [],
                    ),
                )
            )
        else:
            assert isinstance(example, torch.Tensor), (
                f"Expected Tensor, got {type(example)}: {node.target}"
            )
            result = TensorBox.create(
                InputBuffer(
                    name=name,
                    layout=FixedLayout(
                        example.device,
                        example.dtype,
                        [*map(_unpack_symint, example.size())],
                        [*map(_unpack_symint, example.stride())],
                    ),
                )
            )
        assert isinstance(result, TensorBox)
        return result

    prior_buffers = len(graph_lowering.buffers)
    input_names: list[str] = []
    with inductor_config.patch(INDUCTOR_PATCH):
        with node.meta["location"], graph_lowering.set_current_node(node):
            try:
                result = graph_lowering.call_function(
                    # pyrefly: ignore [bad-argument-type]
                    node.target,
                    # pyrefly: ignore [bad-argument-type]
                    *map_arg((node.args, node.kwargs), convert_arg),
                )
            # pyrefly: ignore [implicit-import]
            except torch._inductor.exc.LoweringException as e:
                # Wrap in Helion exception to get location automatically
                raise InductorLoweringError(str(e)) from e
        if not isinstance(result, tuple):
            result = (result,)
        buffer_name_to_output_index = {}
        for i, r in enumerate(result):
            r.realize()
            if not isinstance(r, TensorBox) or not isinstance(r.data, StorageBox):
                raise InductorLoweringError(
                    f"Lowering {node.target} returned {type(r)}, expected TensorBox(StorageBox(...)): {r}"
                )
            if not isinstance(buffer := r.data.data, ComputedBuffer):
                raise InductorLoweringError(
                    f"Lowering {node.target} returned buffer type {type(buffer)}, expected ComputedBuffer: {buffer}"
                )
            buffer_name_to_output_index[buffer.get_name()] = i

    new_buffers = graph_lowering.buffers[prior_buffers:]
    assert buffer in new_buffers
    nodes = []
    extra_input_names = []
    new_node: torch.fx.Node

    # Explicitly track the mapping from node to Inductor buffer name.
    # First, map the original input nodes to their names.
    node_to_buf_name_mapping: dict[torch.fx.Node, str] = dict(
        zip(node._input_nodes, input_names, strict=True)
    )

    for i, buffer in enumerate(new_buffers):
        if not isinstance(buffer, ComputedBuffer) or not isinstance(
            buffer.data, (Pointwise, Reduction)
        ):
            raise InductorLoweringError(
                f"Lowering {node.target} returned buffer type {type(buffer)}, expected ComputedBuffer(Pointwise|Reduction): {buffer}"
            )
        if i == len(new_buffers) - 1:
            new_node = node
            if nodes:
                new_node.kwargs = {**new_node.kwargs, "_extra_args": [*nodes]}
        else:
            new_node = create_extra_node(node, buffer, [*node._input_nodes, *nodes])

        # Store output index if this buffer corresponds to an output
        if buffer.get_name() in buffer_name_to_output_index:
            new_node.meta["output_index"] = buffer_name_to_output_index[
                buffer.get_name()
            ]

        lowering_cls = (
            PointwiseLowering
            if isinstance(buffer.data, Pointwise)
            else ReductionLowering
        )
        buffer.freeze_layout()

        current_input_nodes = new_node._input_nodes
        current_input_names = []
        for inp_node in current_input_nodes:
            current_input_names.append(node_to_buf_name_mapping[inp_node])

        used_input_names = strip_unused_inputs(
            new_node,
            buffer.get_read_names(),
            dict(zip(current_input_nodes, current_input_names, strict=True)),
        )
        new_node.meta["lowering"] = lowering = lowering_cls(buffer, used_input_names)
        new_node.meta["orig_node"] = node
        if isinstance(lowering, ReductionLowering):
            lowering.add_input_mask(new_node)
        nodes.append(new_node)
        extra_input_names.append(buffer.get_name())

        # Add this node to our mapping for future nodes to reference
        node_to_buf_name_mapping[new_node] = buffer.get_name()

    # After all nodes are created, build the output_nodes mapping for multi-output operations
    if len(result) > 1 and nodes:
        last_node = nodes[-1]  # The last node is the main node
        output_nodes = {}
        extra_deps = []
        for n in nodes:
            if "output_index" in n.meta:
                output_nodes[n.meta["output_index"]] = n.name
                if n is not last_node and n not in last_node._input_nodes:
                    extra_deps.append(n)
        last_node.meta["output_nodes"] = output_nodes
        if extra_deps:
            # Need to ensure that the last node depends on all output nodes to prevent DCE issues
            last_node.kwargs = {**last_node.kwargs, "_extra_deps": extra_deps}


def strip_unused_inputs(
    node: torch.fx.Node,
    used_input_names: OrderedSet[str],
    input_names: dict[torch.fx.Node, str],
) -> list[str]:
    """
    Remove unused inputs from the node.  Inplace updates node.args and
    node.kwargs to replace unused inputs with None.

    Args:
        node: Node to mutate args of
        used_input_names: Set of input names that are used in the node's lowering.
        input_names: Mapping of node inputs to their names.

    Returns:
        list[str]: List of names that were used in the lowering.
    """

    def mask_unused_inputs(n: torch.fx.Node) -> torch.fx.Node | None:
        if (name := input_names[n]) in used_input_names and name not in seen_names:
            seen_names.setdefault(name)
            return n
        return None

    assert len(input_names) == len(node._input_nodes)
    seen_names: dict[str, None] = {}
    node.args = map_arg(node.args, mask_unused_inputs)
    node.kwargs = map_arg(node.kwargs, mask_unused_inputs)
    assert len(seen_names) == len(used_input_names)
    return [*seen_names]


def create_extra_node(
    original_node: torch.fx.Node,
    buffer: ComputedBuffer,
    input_nodes: list[torch.fx.Node],
) -> torch.fx.Node:
    """When inductor lowerings produce multiple buffers,
    we add extra nodes to maintain a 1:1 mapping between fx nodes and buffers."""
    from ..language._tracing_ops import _inductor_lowering_extra

    graph = original_node.graph
    with graph.inserting_before(original_node):
        node = graph.create_node(
            "call_function",
            _inductor_lowering_extra,
            (input_nodes,),
            {},
            name=f"{original_node.name}_extra",
        )
    with proxy_tensor.disable_proxy_modes_tracing():
        node.meta["val"] = torch.empty(
            # pyrefly: ignore [no-matching-overload]
            [*map(to_symint, buffer.get_size())],
            dtype=buffer.get_dtype(),
            device=buffer.get_device(),
        )
    for key in ("stack_trace", "original_aten", "location"):
        node.meta[key] = original_node.meta.get(key, None)
    return node


def to_symint(x: object) -> torch.SymInt | int:
    if isinstance(x, (int, sympy.Integer)):
        return int(x)
    assert isinstance(x, sympy.Expr)
    return torch.SymInt(
        SymNode(x, CompileEnvironment.current().shape_env, int, hint=None)
    )


def _unpack_symint(x: torch.SymInt | int) -> sympy.Expr:
    if isinstance(x, torch.SymInt):
        return x._sympy_()
    if isinstance(x, int):
        # type: ignore [bad-return]
        return sympy.sympify(x)
    raise TypeError(f"Expected SymInt or int, got {type(x)}")


@dataclasses.dataclass
class InductorLowering(Lowering):
    buffer: ComputedBuffer
    input_names: list[str]

    def input_asts(self, ctx: LoweringContext, node: torch.fx.Node) -> list[ast.AST]:
        def visit(n: torch.fx.Node) -> None:
            ast_val = cast("ast.AST", ctx.env[n])
            if isinstance(fake_val := n.meta["val"], torch.Tensor):
                # Don't expand scalars (0-D tensors) - let Triton handle broadcasting naturally
                # Expanding scalars with [None, None] creates incorrect broadcast shapes
                if fake_val.ndim < ndim and fake_val.ndim > 0:
                    # Broadcast to force ranks to match (but only for non-scalar tensors)
                    expand = ["None"] * (ndim - fake_val.ndim) + [":"] * fake_val.ndim
                    ast_val = expr_from_string(
                        "{tensor}[" + ", ".join(expand) + "]", tensor=ast_val
                    )
            if (
                isinstance(ast_val, ast.Name)
                and ast_val.id in device_function._constexpr_args
            ):
                # introduce a copy so triton doesn't complain about `id.to(...)` calls
                assert isinstance(ast_val, ExtendedAST)
                with ast_val:
                    copy_var = device_function.new_var(f"{ast_val.id}_", dce=True)
                    ctx.cg.add_statement(
                        statement_from_string(f"{copy_var} = {ast_val.id}")
                    )
                    input_asts.append(expr_from_string(f"{copy_var}"))
            else:
                input_asts.append(ast_val)

        device_function: DeviceFunction = ctx.cg.device_function
        ndim: int = max([x.ndim for x in self.input_fake_tensors(node)] or (0,))
        input_asts: list[ast.AST] = []
        # _extra_deps should not be included in the inductor node inputs
        map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
        assert len(input_asts) == len(self.input_names)
        return input_asts

    @staticmethod
    def input_fake_tensors(node: torch.fx.Node) -> list[torch.Tensor]:
        def visit(n: torch.fx.Node) -> torch.fx.Node:
            if isinstance(val := n.meta["val"], torch.Tensor):
                result.append(val)
            return n

        result: list[torch.Tensor] = []
        map_arg((node.args, node.kwargs), visit)
        return result

    def codegen(self, ctx: LoweringContext, node: torch.fx.Node) -> object:
        raise NotImplementedError(
            f"codegen not implemented for {type(self).__name__}: {self.buffer}"
        )

    def install_kernel_handlers(
        self, ctx: LoweringContext, node: torch.fx.Node
    ) -> ContextManager[None]:
        return install_inductor_kernel_handlers(
            ctx.cg,
            dict(zip(self.input_names, self.input_asts(ctx, node), strict=True)),
        )


@contextlib.contextmanager
def install_inductor_kernel_handlers(
    cg: CodegenInterface, args: dict[str, ast.AST]
) -> Iterator[None]:
    with (
        inductor_config.patch(INDUCTOR_PATCH),
        V.set_graph_handler(FakeGraphLowering()),
        V.set_ops_handler(
            GenerateASTFromInductor(
                cg,
                args,
            )
        ),
        V.set_kernel_handler(
            TritonKernel({}, features=SIMDKernelFeatures([], sympy.S.One))
        ),
    ):
        yield


@functools.cache
def dummy_gm() -> torch.fx.GraphModule:
    return torch.fx.symbolic_trace(lambda: None)


class FakeGraphLowering(GraphLowering):
    def __init__(self) -> None:
        env = CompileEnvironment.current()
        super().__init__(dummy_gm(), shape_env=env.shape_env)
        # Ensure Inductor helpers see a valid current device
        self.current_device = env.device


class PointwiseLowering(InductorLowering):
    def codegen(self, ctx: LoweringContext, node: torch.fx.Node) -> object:
        # Validate broadcasting of tile block dimensions to catch shape mismatches
        self._check_block_broadcast_compatibility(node)
        with self.install_kernel_handlers(ctx, node):
            indices = [
                sympy.Symbol(f"i{n}") for n in range(len(self.buffer.data.ranges))
            ]
            output_name = _unpack_opsvalue(self.buffer.data.inner_fn(indices))
            result = expr_from_string(output_name)

        return self._reshape_for_size1_reduction(ctx, node, result)

    def _reshape_for_size1_reduction(
        self, ctx: LoweringContext, node: torch.fx.Node, result: ast.AST
    ) -> ast.AST:
        # When Inductor converts a size-1 reduction to a Pointwise op, the
        # buffer has fewer ranges than the inputs.  This happens when the
        # literal 1 comes from ops like unsqueeze or keepdim=True (e.g.,
        # val.unsqueeze(0).sum(0) where val is [D] — the unsqueeze creates
        # [1, D], Inductor sees sum over literal-1 dim, converts to Pointwise
        # with ranges [D], but the inner_fn still produces a 2-D value).
        # Reshape the result to match the expected output shape.
        output_val = node.meta.get("val")
        max_input_ndim = max(
            (inp.ndim for inp in self.input_fake_tensors(node)), default=0
        )
        if max_input_ndim > len(self.buffer.data.ranges) and isinstance(
            output_val, torch.Tensor
        ):
            shape_str = ctx.cg.device_function.tile_strategy.shape_str(
                [*output_val.size()]
            )
            result = expr_from_string(
                f"tl.reshape({{result}}, {shape_str})", result=result
            )
        return result

    def get_masked_value(self, node: torch.fx.Node) -> float | bool | None:
        return inductor_masked_value(self, node)

    def _check_block_broadcast_compatibility(self, node: torch.fx.Node) -> None:
        """Detect invalid broadcasting between tile-related dimensions in pointwise ops.

        This guards against patterns like subtracting a reduced tensor without
        keepdim from a 2D tile, which would otherwise silently broadcast along
        the wrong axis (e.g., [M, N] - [M] -> [M, N] by aligning on N).

        We right-align shapes and then, per-dimension, verify that there aren't
        two distinct non-1 symbolic sizes that are not known-equal. This is more
        robust than relying solely on block-id provenance and works even if
        upstream rewrites introduced fresh symbolic expressions.
        """
        env = CompileEnvironment.current()
        inputs = self.input_fake_tensors(node)
        if len(inputs) < 2:
            return

        # Right-align shapes for broadcasting comparison
        shapes: list[list[int | torch.SymInt]] = [[*t.size()] for t in inputs]
        max_rank = max((len(s) for s in shapes), default=0)
        for i, s in enumerate(shapes):
            pad = max_rank - len(s)
            if pad > 0:
                shapes[i] = [1] * pad + s

        def is_one(x: int | torch.SymInt) -> bool:
            if isinstance(x, int):
                return x == 1
            if isinstance(x, torch.SymInt):
                expr = x._sympy_()
                if isinstance(expr, sympy.Integer):
                    return int(expr) == 1
                # Treat tiles with a fixed block size of 1 as broadcastable-1
                block_id = env.get_block_id(x)
                if block_id is not None:
                    bs = env.block_sizes[block_id]
                    if isinstance(bs.block_size_source, FixedBlockSizeSource):
                        val = bs.block_size_source.value
                        if isinstance(val, int):
                            return val == 1
                        if isinstance(val, torch.SymInt):
                            vexpr = val._sympy_()
                            return isinstance(vexpr, sympy.Integer) and int(vexpr) == 1
                return False
            return False

        # Check each dimension independently
        for dim in range(max_rank):
            # First, see if multiple distinct block-ids appear in this dim
            block_ids: set[int] = set()
            for s in shapes:
                size_i = s[dim]
                if is_one(size_i):
                    continue
                block_id = env.get_block_id(size_i)
                if block_id is not None:
                    block_ids.add(block_id)
            if len(block_ids) >= 2:
                raise exc.ShapeMismatch(
                    str(shapes[0]),
                    ", ".join(map(str, shapes[1:])),
                )

            # Otherwise, fall back to strict symbolic inequality among non-1 sizes
            exprs: set[object] = set()
            for s in shapes:
                size_i = s[dim]
                if is_one(size_i):
                    continue
                if isinstance(size_i, torch.SymInt):
                    exprs.add(size_i._sympy_())
                else:
                    exprs.add(size_i)
            if len(exprs) >= 2:
                raise exc.ShapeMismatch(
                    str(shapes[0]),
                    ", ".join(map(str, shapes[1:])),
                )


@dataclasses.dataclass
class ReductionLowering(InductorLowering):
    def __init__(
        self,
        buffer: ComputedBuffer,
        input_names: list[str],
    ) -> None:
        super().__init__(buffer, input_names)
        reduction = self.buffer.data
        assert isinstance(reduction, Reduction)
        reduction_ranges = reduction.reduction_ranges
        if len(reduction_ranges) != 1:
            # TODO(jansel): can this happen?
            raise NotImplementedError("multiple reduction dimensions")
        # In Inductor IR, reduction_ranges holds sizes, not loop vars.
        # Support both symbolic and constant sizes by allocating/looking up
        # a matching reduction dimension in the current environment.
        reduction_size = reduction_ranges[0]

        env = CompileEnvironment.current()
        if isinstance(reduction_size, sympy.Symbol):
            block_index: int | None = env.get_block_id(reduction_size)
        elif isinstance(reduction_size, (int, sympy.Integer)):
            # Allocate or find a reduction dimension matching this size.
            # Convert to a SymInt when needed.
            size_symint_or_int = to_symint(reduction_size)
            block_index = env.allocate_reduction_dimension(size_symint_or_int).block_id
        elif isinstance(reduction_size, sympy.Expr):
            # Handle symbolic expressions (including those with only block size symbols)
            if contains_only_block_size_symbols(reduction_size):
                size_symint = to_symint(reduction_size)
                block_index = env.allocate_reduction_dimension(size_symint).block_id
            else:
                raise exc.ReductionOnNonTile(reduction_size)
        else:
            raise exc.ReductionOnNonTile(reduction_size)
        assert block_index is not None
        self.block_index: int = block_index

    @property
    def reduction_type(self) -> str:
        reduction = self.buffer.data
        assert isinstance(reduction, Reduction)
        return reduction.reduction_type

    def add_input_mask(self, node: torch.fx.Node) -> None:
        """Modify the node to apply masking for the reduction if needed."""
        reduction_type = self.reduction_type
        input_dtype = None
        for inp in node.all_input_nodes:
            if isinstance(inp.meta["val"], torch.Tensor):
                input_dtype = inp.meta["val"].dtype
                break
        assert input_dtype is not None
        default = ir.Reduction.default_accumulator(reduction_type, input_dtype)
        assert isinstance(default, (float, int, bool))
        mask_node_inputs(node, default)

    def codegen(self, ctx: LoweringContext, node: torch.fx.Node) -> object:
        reduction = self.buffer.data
        assert isinstance(reduction, Reduction)
        indices = [sympy.Symbol(f"i{n}") for n in range(len(reduction.ranges))]
        reduction_indices = [
            sympy.Symbol(f"i{n}")
            for n in range(len(indices), len(indices) + len(reduction.reduction_ranges))
        ]
        with self.install_kernel_handlers(ctx, node):
            # codegen the pointwise part before reduction
            output_name = _unpack_opsvalue(
                self.buffer.data.inner_fn(indices, reduction_indices)
            )

        from .generate_ast import GenerateAST

        if not isinstance(ctx.cg, GenerateAST):
            raise exc.NotAllowedInHelperFunction

        state = CodegenState(
            ctx.cg,
            fx_node=node,
        )
        if CompileEnvironment.current().block_sizes[self.block_index].reduction:
            strategy = ctx.cg.device_function.tile_strategy.get_reduction_strategy(
                self.block_index
            )
        else:
            from .reduction_strategy import BlockReductionStrategy

            strategy = BlockReductionStrategy(state, self.block_index)

        inputs = self.input_fake_tensors(node)

        if len(inputs) == 1:
            repr_input = inputs[0]
        elif node.meta["orig_node"].target == torch.ops.aten.var_mean.correction:
            assert len(inputs) == 2
            # `inputs[0]` is the original input tensor to var_mean
            repr_input = inputs[0]
        else:
            # TODO(jansel): combine multiple inputs into a single fake value
            raise NotImplementedError("reductions with >1 input")

        dims = self._get_reduction_dims(node.meta["orig_node"], repr_input)
        if len(dims) != 1:
            # TODO(jansel): support multiple reduction dims
            raise exc.MultipleReductionDims

        result_ast = strategy.codegen_reduction(
            state,
            output_name,
            reduction.reduction_type,
            dims[0],
            repr_input,
            node.meta["val"],
        )
        # For looped reductions, the actual value is assigned after the loop in
        # the strategy's outer_suffix. Casting at this point would reference the
        # result before it is defined. The strategy is responsible for casting
        # to the final dtype in that case.
        from .reduction_strategy import (
            LoopedReductionStrategy,
        )  # local import to avoid cycles

        if isinstance(strategy, LoopedReductionStrategy):
            # Mark this node as having a delayed result so downstream codegen can
            # avoid emitting an early assignment or dtype assert.
            node.meta["delayed_result"] = True
            return result_ast

        # Non-looped reductions compute the value inline; cast now to ensure the
        # result dtype matches torch.* semantics reflected in meta["val"].dtype.
        desired_dtype = node.meta["val"].dtype
        return cast_ast(result_ast, desired_dtype)

    def get_masked_value(self, node: torch.fx.Node) -> float | bool | None:
        # reduction types that preserve zeroness
        if self.reduction_type in {"sum", "prod", "min", "max"}:
            value = inductor_masked_value(self, node)
            if value == 0:
                return value
        return None

    @staticmethod
    def _get_reduction_dims(node: torch.fx.Node, fake_input: torch.Tensor) -> list[int]:
        if fake_input.ndim == 1:
            return [0]

        dims = node.kwargs.get("dim", node.kwargs.get("dims"))
        if dims is None:
            schema = node.meta["original_aten"]._schema
            assert isinstance(schema, torch._C.FunctionSchema)
            for index, arg in enumerate(schema.arguments):
                if arg.name in {"dim", "dims"}:
                    dims = (
                        node.args[index]
                        if index < len(node.args)
                        else arg.default_value
                    )
                    break
            if dims is None:
                dims = [*range(fake_input.ndim)]

        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        result = []
        for dim in dims:
            if not isinstance(dim, (int, sympy.Integer)):
                raise exc.InvalidReductionDim(dim)
            dim = int(dim)
            if dim < 0:
                dim = fake_input.ndim + dim
            if not (0 <= dim < fake_input.ndim):
                raise exc.ReductionDimInvalidForShape(dim, fake_input.shape)
            result.append(dim)
        return result


class APIFuncLowering(Lowering):
    def __init__(self, api_func: object) -> None:
        super().__init__()
        assert is_api_func(api_func)
        self.api_func: APIFunc = api_func

    def codegen(self, ctx: LoweringContext, node: torch.fx.Node) -> object:
        assert not node.kwargs
        ast_args = [*map_arg(node.args, lambda arg: ctx.env[arg])]
        proxy_args = [*map_arg(node.args, lambda arg: arg.meta["val"])]

        env = CompileEnvironment.current()
        codegen_fn = self.api_func._codegen.get(env.backend_name)
        if codegen_fn is None:
            codegen_fn = self.api_func._codegen.get("common")
        if codegen_fn is None:
            raise exc.BackendImplementationMissing(
                env.backend_name,
                f"codegen for API function {self.api_func.__qualname__}",
            )
        from .generate_ast import GenerateAST

        if not isinstance(ctx.cg, GenerateAST):
            raise exc.NotAllowedInHelperFunction

        return codegen_fn(
            CodegenState(
                ctx.cg,
                fx_node=node,
                # pyrefly: ignore [bad-argument-type]
                proxy_args=proxy_args,
                # pyrefly: ignore [bad-argument-type]
                ast_args=ast_args,
            ),
        )

    @staticmethod
    def normalize_args_kwargs(
        api_func: APIFunc,
        node: torch.fx.Node,
    ) -> None:
        bound = api_func._signature.bind(*node.args, **node.kwargs)
        bound.apply_defaults()
        node.args = (*bound.arguments.values(),)
        node.kwargs = {}

    def get_masked_value(self, node: torch.fx.Node) -> float | bool | None:
        if self.api_func._get_masked_value is not None:
            return self.api_func._get_masked_value(node)
        return None


@dataclasses.dataclass
class SympyExprLowering(Lowering):
    expr: sympy.Expr

    def codegen(self, ctx: LoweringContext, node: torch.fx.Node) -> object:
        return expr_from_string(ctx.cg.device_function.user_sympy_expr(self.expr))

    def get_masked_value(self, node: torch.fx.Node) -> float | bool | None:
        if isinstance(self.expr, sympy.Integer):
            return int(self.expr)
        if isinstance(self.expr, sympy.Float):
            return float(self.expr)
        return None


class GenerateASTFromInductor(DefaultHandler):
    def __init__(
        self, cg: CodegenInterface, input_name_lookup: dict[str, ast.AST]
    ) -> None:
        super().__init__()
        self.parent_handler: InductorOpOverrides = (
            CompileEnvironment.current().backend.inductor_op_overrides()
        )
        self.cg = cg
        self.input_name_lookup = input_name_lookup

    def _cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        backend = CompileEnvironment.current().backend
        return backend.cast_ast(x, target_dtype)

    def _to_ast(self, x: object) -> ast.AST:
        if isinstance(x, ast.AST):
            return x
        return expr_from_string(_unpack_opsvalue(x))

    def _lift(self, expr: ast.AST) -> str:
        return self.cg.lift(expr).id

    def _expected_tensor_dtype(self) -> torch.dtype | None:
        """Best-effort retrieval of the current FX node's tensor dtype."""
        current_node = V.current_node
        if current_node is None:
            return None
        val = current_node.meta.get("val")
        if isinstance(val, torch.Tensor):
            return val.dtype
        return None

    def _create_cast_expr(self, x: object, target_dtype: torch.dtype) -> ast.AST:
        """Create a backend cast expression from AST or string input.

        Args:
            x: Input value (AST node or string/OpsValue)
            target_dtype: Target dtype

        Returns:
            AST expression for the cast operation
        """
        x_ast = self._to_ast(x)
        return self._cast_ast(x_ast, target_dtype)

    def _maybe_cast_to_expected_dtype(self, expr: ast.AST) -> ast.AST:
        """Cast expression to expected dtype if needed.

        Args:
            expr: Input expression to potentially cast

        Returns:
            Original or casted expression
        """
        expected_dtype = self._expected_tensor_dtype()
        if expected_dtype is None:
            return expr
        return self._create_cast_expr(expr, expected_dtype)

    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> str:
        result_str = _unpack_opsvalue(
            getattr(self.parent_handler, name)(*args, **kwargs)
        )
        return self._lift(expr_from_string(result_str))

    def to_dtype(
        self,
        x: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        """Emit explicit backend cast to enforce final dtype conversion.

        We avoid delegating to the parent handler to prevent reliance on global
        device context during compute-type selection, and to guarantee a visible
        cast in generated code that matches PyTorch's dtype semantics.
        """
        cast_expr = self._create_cast_expr(x, dtype)
        return self._lift(cast_expr)

    def sigmoid(self, x: object) -> str:  # type: ignore[override]
        if CompileEnvironment.current().backend.name != "triton":
            return self._default("sigmoid", (x,), {})

        # Triton sigmoid expects fp32/fp64 inputs; enforce fp32 compute, then cast back.
        inner_name = self._lift(self._create_cast_expr(x, torch.float32))
        result = expr_from_string(
            _unpack_opsvalue(self.parent_handler.sigmoid(inner_name))
        )

        expected_dtype = self._expected_tensor_dtype()
        if expected_dtype is not None and expected_dtype != torch.float32:
            result = self._maybe_cast_to_expected_dtype(result)
        return self._lift(result)

    def mul(self, a: object, b: object) -> str:  # type: ignore[override]
        def has_scalar_operand() -> bool:
            current_node = V.current_node
            if current_node is None:
                return False
            return any(isinstance(arg, (int, float, bool)) for arg in current_node.args)

        result_str = _unpack_opsvalue(self.parent_handler.mul(a, b))
        result_expr = expr_from_string(result_str)

        # Only cast if we have a scalar operand and expected dtype is not float32.
        # This is to handle cases like `x_bf16 * 0.1` where Triton would promote the result to float32,
        # deviating from PyTorch semantics.
        expected_dtype = self._expected_tensor_dtype()
        if (
            has_scalar_operand()
            and expected_dtype is not None
            and expected_dtype != torch.float32
        ):
            result_expr = self._maybe_cast_to_expected_dtype(result_expr)

        return self._lift(result_expr)

    def load(self, name: str, index: sympy.Expr) -> str:
        # TODO(jansel): assert the index is correct
        return self.cg.lift(self.input_name_lookup[name]).id

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> str:
        name = self.cg.lift(
            expr_from_string(self.cg.device_function.user_sympy_expr(expr))
        ).id

        # If the lifted symbol refers to a `tl.constexpr` kernel
        # argument (for example a tile/block size constant such as
        # `_BLOCK_SIZE_1`) the resulting Triton value is not a tensor
        # and therefore does not expose a `.to` method.
        if name in self.cg.device_function._constexpr_args:
            return name

        return self._lift(self._create_cast_expr(expr_from_string(name), dtype))


def _unpack_opsvalue(value: object) -> str:
    if isinstance(value, OpsValue):
        return str(value)
    assert isinstance(value, str)
    return value


class GraphInterpreter(LoweringContext, Interpreter):
    def __init__(self, graph: torch.fx.Graph, cg: CodegenInterface) -> None:
        super().__init__(_LazyGraphModule({}, graph), garbage_collect_values=False)
        self.cg = cg
        self.env = self.env

    def to_ast(self, value: object) -> ast.AST:
        """
        Convert a value to an AST expression.
        """
        if isinstance(value, torch.fx.Node):
            result = self.env[value]
            assert isinstance(result, ast.AST)
            return result
        if isinstance(value, (int, float, bool)):
            return create(ast.Constant, value=value)
        if isinstance(value, ast.AST):
            return value
        raise TypeError(f"Unsupported value type for AST conversion: {type(value)}")

    def _create_named_result(self, node: Node, result: ast.expr) -> str:
        """Create a named variable for a node result, handling block-size-only expressions as constexpr."""
        val = node.meta.get("val")

        # Check if we should create a constexpr for block-size-only expressions used in tl.arange
        if (
            isinstance(val, torch.SymInt)
            and contains_only_block_size_symbols(val._sympy_())
            and any(
                user.op == "call_function"
                and user.target == torch.ops.prims.iota.default
                for user in node.users
            )
        ):
            # This expression is used in tl.arange, make it a constexpr
            name = self.cg.device_function.new_var(node.name)
            self.cg.device_function.constexpr_arg(name, val._sympy_())
            return name

        # If the lowering produced a named value that is already defined elsewhere
        # (e.g., looped reduction assigned in an outer suffix), avoid emitting a
        # premature assignment that could reference it before definition.
        delayed_result = bool(node.meta.get("delayed_result", False))
        if isinstance(result, ast.Name):
            name = result.id
        else:
            # Regular variable assignment
            name = self.cg.device_function.new_var(node.name)
            self.cg.add_statement(
                statement_from_string(f"{name} = {{result}}", result=result)
            )
        # Optionally enforce and assert dtype after each device node
        settings = CompileEnvironment.current().settings
        if (
            settings.debug_dtype_asserts
            and isinstance(val, torch.Tensor)
            and not delayed_result
        ):
            # Skip pure view ops; their dtype matches their input, which we've likely asserted already
            if node.op == "call_function" and node.target in (
                torch.ops.aten.unsqueeze.default,
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten.expand.default,
                torch.ops.aten.permute.default,
            ):
                return name
            expected_dtype = val.dtype
            # First, enforce the expected dtype to mirror PyTorch semantics
            self.cg.add_statement(
                statement_from_string(
                    f"{name} = tl.cast({name}, {triton_type(expected_dtype)})"
                )
            )
            self.cg.add_statement(
                statement_from_string(
                    f"tl.static_assert({name}.dtype == {triton_type(expected_dtype)})"
                )
            )
        return name

    def _collect_multi_outputs(
        self, node: Node, last_node_result: object
    ) -> tuple[object, ...]:
        """
        Collect outputs for multi-output operations using metadata.
        """
        # Check if this operation has multiple outputs using the new metadata
        assert "output_nodes" in node.meta
        output_nodes = node.meta["output_nodes"]
        outputs: list[object | None] = [None] * len(output_nodes)
        all_nodes = {
            n.name: n
            # pyrefly: ignore [missing-attribute]
            for n in self.module.graph.nodes
        }

        for idx, node_name in output_nodes.items():
            if node_name == node.name:
                # This is the last node
                outputs[idx] = last_node_result
            else:
                # This is an extra node - get its result from env
                if node_name in all_nodes:
                    extra_node = all_nodes[node_name]
                    if extra_node in self.env:
                        outputs[idx] = self.env[extra_node]

        # Ensure all outputs are found and are ast.Name nodes
        final_outputs = []
        for i, result in enumerate(outputs):
            assert result is not None
            if not isinstance(result, ast.Name):
                var_name = self.cg.device_function.new_var(f"{node.name}_output{i}")
                assert isinstance(result, ast.AST)
                self.cg.add_statement(
                    statement_from_string(f"{var_name} = {{result}}", result=result)
                )
                result = create(ast.Name, id=var_name, ctx=ast.Load())
            final_outputs.append(result)

        return tuple(final_outputs)

    def run_node(self, n: Node) -> object:
        if n.op == "call_function":
            with self._set_current_node(n), n.meta["location"], V.set_current_node(n):
                try:
                    lowering: Lowering = n.meta["lowering"]
                    result = lowering.codegen(self, n)
                    n.meta["codegen"] = result

                    # Generic handling for operations with multiple outputs
                    if n.kwargs.get("_extra_args"):
                        # Check if this node has getitem users, indicating multiple outputs
                        getitem_users = [
                            user for user in n.users if user.target == getitem
                        ]
                        if len(getitem_users) > 0:
                            return self._collect_multi_outputs(n, result)

                    if result is None:
                        return None
                    if not isinstance(result, ast.AST):
                        return result
                    assert isinstance(result, ast.expr)
                    if len(n.users) > 0:
                        if not isinstance(result, (ast.Name, ast.Constant)):
                            name = self._create_named_result(n, result)
                            result = create(ast.Name, id=name, ctx=ast.Load())
                        if (
                            isinstance(val := n.meta["val"], torch.SymInt)
                            and len((expr := val._sympy_()).free_symbols) > 0
                        ):
                            # Keep track of what variable symints are stored in to support DeviceFunction.sympy_expr()
                            expr = CompileEnvironment.current().shape_env.simplify(expr)
                            if isinstance(result, ast.Name):
                                self.cg.device_function.expr_to_var_info[expr] = (
                                    VarInfo(result.id, n)
                                )
                            else:
                                assert isinstance(result, ast.Constant)
                                self.cg.device_function.expr_to_var_info[expr] = (
                                    VarInfo(repr(result.value), n)
                                )
                        return result
                    if not isinstance(result, (ast.Name, ast.Constant)):
                        self.cg.add_statement(create(ast.Expr, value=result))
                    return None
                except exc.Base:
                    raise
                except Exception as e:
                    raise InductorLoweringError(
                        f"Error in codegen for node {n.name} ({n.target}): {e}"
                    ) from e
        return super().run_node(n)


def codegen_call_with_graph(
    cg: GenerateAST, graph: torch.fx.Graph, args: list[ast.AST]
) -> list[object]:
    with compile_lock:
        new_args = []
        placeholders = graph.find_nodes(op="placeholder")
        for arg, placeholder in zip(args, placeholders, strict=True):
            if all(
                user.target == torch.ops.aten.sym_size.int for user in placeholder.users
            ):
                # TODO(jansel): we should remove these sym_size-only args from the graph
                new_args.append(arg)
            elif isinstance(arg, ast.Name):
                # We need to copy the inputs to a loop so that phi nodes are handled properly.
                # Phi nodes will merge variable names from outside the loop, but the old value
                # of those variables could have usages.
                copy_name = cg.device_function.new_var(arg.id + "_copy")
                cg.add_statement(
                    statement_from_string(f"{copy_name} = {{arg}}", arg=arg)
                )
                new_args.append(expr_from_string(copy_name))
            else:
                new_args.append(cg.lift(arg))
        return GraphInterpreter(graph, cg).run(*new_args)


class CodegenState(NamedTuple):
    codegen: GenerateAST
    fx_node: torch.fx.Node | None
    proxy_args: list[object] = dataclasses.field(default_factory=list)
    ast_args: list[object] = dataclasses.field(default_factory=list)

    def proxy_arg(self, i: int) -> object:
        return self.proxy_args[i]

    def ast_arg(self, i: int) -> ast.AST:
        rv = self.ast_args[i]
        if isinstance(rv, int | float | bool | None):
            rv = ast.Constant(value=rv)
        assert isinstance(rv, ast.AST), "TODO: convert nested/defaults"
        return rv

    @property
    def fake_value(self) -> object:
        assert self.fx_node is not None
        return self.fx_node.meta["val"]

    @property
    def device_function(self) -> DeviceFunction:
        return self.codegen.device_function

    @property
    def tile_strategy(self) -> TileStrategyDispatch:
        return self.codegen.device_function.tile_strategy

    @property
    def config(self) -> Config:
        return self.codegen.device_function.config

    def add_statement(self, statement: ast.AST | str) -> None:
        return self.codegen.add_statement(statement)

    def sympy_expr(self, expr: sympy.Expr) -> str:
        return self.codegen.device_function.sympy_expr(expr)
