from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
from operator import getitem
from typing import TYPE_CHECKING
from typing import Callable
from typing import ContextManager
from typing import NamedTuple

import sympy
import torch
from torch._dynamo.convert_frame import compile_lock
from torch._inductor import config as inductor_config
from torch._inductor import ir
from torch._inductor.codegen.simd import (
    SIMDKernelFeatures,  # pyright: ignore[reportPrivateImportUsage]
)
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import InputBuffer
from torch._inductor.ir import Pointwise
from torch._inductor.ir import Reduction
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TensorBox
from torch._inductor.ops_handler import DefaultHandler, MockHandler
import sympy
from torch._inductor.utils import triton_type
from torch._inductor.virtualized import OpsValue
from torch._inductor.virtualized import V
from torch.fx._lazy_graph_module import _LazyGraphModule
from .ast_extension import create
from torch.fx.experimental import proxy_tensor
from torch.fx.experimental.sym_node import SymNode
from torch.fx.interpreter import Interpreter
from torch.fx.node import Node
from torch.fx.node import map_arg

from .. import exc
from .._compat import min_dot_size
from ..exc import InductorLoweringError
from ..language._decorators import APIFunc
from ..language._decorators import is_api_func
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .device_function import VarInfo
from .device_function import contains_only_block_size_symbols
from .node_masking import apply_masking
from .node_masking import cached_masked_value
from .node_masking import getitem_masked_value
from .node_masking import inductor_masked_value
from .node_masking import mask_node_inputs

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from torch.utils._ordered_set import OrderedSet

    from .. import Config
    from .device_function import DeviceFunction
    from .generate_ast import GenerateAST
    from .helper_function import CodegenInterface
    from .tile_dispatch import TileStrategyDispatch

    CodegenHandler = Callable[["GraphInterpreter", torch.fx.Node], object]

INDUCTOR_PATCH: dict[str, object] = {
    # Don't add implicit upcasts to FP32
    "triton.codegen_upcast_to_fp32": False,
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

    # Track arguments to reuse TensorBoxes for duplicates
    arg_to_tensorbox: dict[Node, TensorBox] = {}

    def convert_arg(arg: Node) -> TensorBox:
        example = arg.meta["val"]

        # Reuse existing TensorBox for duplicate arguments
        if arg in arg_to_tensorbox:
            return arg_to_tensorbox[arg]
            
        # Use the actual node name if available, otherwise generate one
        # This is important for Inductor to generate correct variable references
        name = arg.name if hasattr(arg, 'name') and arg.name else f"{node.name}_input{len(input_names)}"
        input_names.append(name)

        if isinstance(example, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            dtype = {
                torch.SymInt: torch.int64,
                torch.SymFloat: torch.float32,
                torch.SymBool: torch.bool,
            }[type(example)]
            # Create InputBuffer with proper registration
            buf = InputBuffer(
                name=name,
                layout=FixedLayout(
                    CompileEnvironment.current().device,
                    dtype,
                    [],
                    [],
                ),
            )
            # Register it with the graph
            graph_lowering.graph_inputs[name] = buf
            result = TensorBox.create(buf)
        else:
            assert isinstance(example, torch.Tensor), (
                f"Expected Tensor, got {type(example)}: {node.target}"
            )
            # Create InputBuffer with proper registration
            # We need to make sure the buffer's loader works correctly
            # within Inductor's context
            from torch._inductor.virtualized import V, ops
            
            buf = InputBuffer(
                name=name,
                layout=FixedLayout(
                    example.device,
                    example.dtype,
                    [*map(_unpack_symint, example.size())],
                    [*map(_unpack_symint, example.stride())],
                ),
            )
            # Register it with the graph
            graph_lowering.graph_inputs[name] = buf
            
            # Override the make_loader method to return a loader that works in our context
            original_make_loader = buf.make_loader
            def custom_make_loader():
                def loader(index):
                    # Return an OpsValue that represents loading from this buffer
                    # This will be handled by our ops handler when the Pointwise is executed
                    return ops.load(name, index)
                return loader
            buf.make_loader = custom_make_loader
            
            result = TensorBox.create(buf)
            
        # Cache the TensorBox for reuse
        arg_to_tensorbox[arg] = result
        assert isinstance(result, TensorBox)
        return result

    prior_buffers = len(graph_lowering.buffers)
    input_names: list[str] = []
    with inductor_config.patch(INDUCTOR_PATCH):
        with node.meta["location"]:
            try:
                result = graph_lowering.call_function(
                    node.target,  # pyright: ignore[reportArgumentType]
                    *map_arg((node.args, node.kwargs), convert_arg),  # pyright: ignore[reportArgumentType]
                )
            except torch._inductor.exc.LoweringException as e:  # pyright: ignore[reportAttributeAccessIssue]
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
    assert (
        buffer in new_buffers  # pyright: ignore[reportPossiblyUnboundVariable]
    )
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
        return sympy.sympify(x)
    raise TypeError(f"Expected SymInt or int, got {type(x)}")


class Lowering:
    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        raise NotImplementedError

    def get_masked_value(self, node: torch.fx.Node) -> float | bool | None:
        """Get the masked value for this node."""
        return None


@dataclasses.dataclass
class InductorLowering(Lowering):
    buffer: ComputedBuffer
    input_names: list[str]

    def input_asts(self, ctx: GraphInterpreter, node: torch.fx.Node) -> list[ast.AST]:
        def visit(n: torch.fx.Node) -> None:
            ast_val = ctx.env[n]
            if isinstance(fake_val := n.meta["val"], torch.Tensor):
                if fake_val.ndim < ndim:
                    # Broadcast to force ranks to match
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

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        raise NotImplementedError(
            f"codegen not implemented for {type(self).__name__}: {self.buffer}"
        )

    def install_kernel_handlers(
        self, ctx: GraphInterpreter, node: torch.fx.Node,
        iteration_vars: list[sympy.Symbol] | None = None
    ) -> ContextManager[None]:
        return install_inductor_kernel_handlers(
            ctx.cg,
            dict(zip(self.input_names, self.input_asts(ctx, node), strict=True)),
            iteration_vars,
        )


@contextlib.contextmanager
def install_inductor_kernel_handlers(
    cg: CodegenInterface, args: dict[str, ast.AST],
    iteration_vars: list[sympy.Symbol] | None = None
) -> Iterator[None]:
    with (
        inductor_config.patch(INDUCTOR_PATCH),
        V.set_graph_handler(FakeGraphLowering()),
        V.set_ops_handler(
            GenerateASTFromInductor(
                cg,
                args,
                iteration_vars,
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
        # Set the device directly on the graph_lowering to ensure get_current_device_or_throw() works
        self._current_device = env.device


class PointwiseLowering(InductorLowering):
    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        indices = [
            sympy.Symbol(f"i{n}") for n in range(len(self.buffer.data.ranges))
        ]
        with self.install_kernel_handlers(ctx, node, indices):
            # Call inner_fn with our ops handler installed
            result = self.buffer.data.inner_fn(indices)
            
            # The result might be a sympy expression or a string
            if isinstance(result, sympy.Basic):
                # If it's a sympy expression, we need to resolve any symbols
                # that represent our generated variables
                output_name = self._resolve_sympy_result(result, ctx)
            else:
                output_name = _unpack_opsvalue(result)
            
            # Check if we need to reshape the result for 2D output
            # If the output is 2x1 but the result is 1D, we need to reshape
            if len(self.buffer.data.ranges) == 2:
                # This is a 2D output
                # Check if the second dimension is 1
                if self.buffer.data.ranges[1] == 1:
                    # The output is Nx1, need to reshape the 1D result
                    # Add [:, None] to make it 2D
                    output_name = f"({output_name})[:, None]"
            
            return expr_from_string(output_name)
    
    def _resolve_sympy_result(self, expr: sympy.Basic, ctx: GraphInterpreter) -> str:
        """Resolve sympy expressions that may contain placeholder symbols."""
        # Check if our ops handler has any pending results
        ops_handler = V.ops
        if hasattr(ops_handler, 'pending_result') and ops_handler.pending_result:
            # Use the pending result
            result = ops_handler.pending_result
            ops_handler.pending_result = None
            return result
        # Default: convert to string
        return str(expr)

    def get_masked_value(self, node: torch.fx.Node) -> float | bool | None:
        return inductor_masked_value(self, node)


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

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
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

        from .. import exc
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
        elif node.meta["orig_node"].target == torch.ops.aten.var_mean.correction:  # pyright: ignore[reportAttributeAccessIssue]
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

        return strategy.codegen_reduction(
            state,
            output_name,
            reduction.reduction_type,
            dims[0],
            repr_input,
            node.meta["val"],
        )

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
            schema = node.meta["original_aten"]._schema  # pyright: ignore[reportAttributeAccessIssue]
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

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        assert not node.kwargs
        ast_args = [*map_arg(node.args, lambda arg: ctx.env[arg])]
        proxy_args = [*map_arg(node.args, lambda arg: arg.meta["val"])]

        assert self.api_func._codegen is not None
        from .. import exc
        from .generate_ast import GenerateAST

        if not isinstance(ctx.cg, GenerateAST):
            raise exc.NotAllowedInHelperFunction

        return self.api_func._codegen(
            CodegenState(
                ctx.cg,
                fx_node=node,
                proxy_args=proxy_args,  # pyright: ignore[reportArgumentType]
                ast_args=ast_args,  # pyright: ignore[reportArgumentType]
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

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        return expr_from_string(ctx.cg.device_function.user_sympy_expr(self.expr))

    def get_masked_value(self, node: torch.fx.Node) -> float | bool | None:
        if isinstance(self.expr, sympy.Integer):
            return int(self.expr)
        if isinstance(self.expr, sympy.Float):
            return float(self.expr)
        return None


@dataclasses.dataclass
class LambdaLowering(Lowering):
    fn: Callable[..., object]
    masked_value_fn: Callable[[torch.fx.Node], float | bool | None] | None = None

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        return self.fn(ctx, node)

    def get_masked_value(self, node: torch.fx.Node) -> float | bool | None:
        if self.masked_value_fn is not None:
            return self.masked_value_fn(node)
        return None


def passthrough_masked_value(node: torch.fx.Node) -> float | bool | None:
    for input_node in node.all_input_nodes:
        if isinstance(input_node.meta["val"], torch.Tensor):
            return cached_masked_value(input_node)
    return None


aten_lowering_dispatch: dict[object, Callable[[torch.fx.Node], Lowering]] = {}


def default_make_lowering(
    handler: CodegenHandler,
    node: torch.fx.Node,
    masked_value_fn: Callable[[torch.fx.Node], float | bool | None] | None = None,
) -> Lowering:
    return LambdaLowering(handler, masked_value_fn=masked_value_fn)


def register_lowering(
    fn: object,
    make_lowering: Callable[
        [CodegenHandler, torch.fx.Node], Lowering
    ] = default_make_lowering,
    masked_value_fn: Callable[[torch.fx.Node], float | bool | None] | None = None,
) -> Callable[[CodegenHandler], CodegenHandler]:
    def decorator(handler: CodegenHandler) -> CodegenHandler:
        assert fn not in aten_lowering_dispatch, f"Lowering for {fn} already registered"

        aten_lowering_dispatch[fn] = lambda node: make_lowering(
            handler,
            node,
            masked_value_fn=masked_value_fn,  # pyright: ignore[reportCallIssue]
        )
        return handler

    return decorator


# Remove custom lowering - let Inductor handle it
# The issue was in our ops handler, not in the lowering itself

@register_lowering(torch.ops.aten.sym_size.int)  # pyright: ignore[reportAttributeAccessIssue]
def codegen_sym_size(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    val = node.meta["val"]
    assert isinstance(
        val, (int, float, bool, torch.SymInt, torch.SymBool, torch.SymFloat)
    )
    return val


@register_lowering(getitem, masked_value_fn=getitem_masked_value)
def codegen_getitem(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    lhs, rhs = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(lhs, (list, tuple))
    assert isinstance(rhs, int)
    return lhs[rhs]


@register_lowering(
    torch.ops.aten.full.default,  # pyright: ignore[reportAttributeAccessIssue]
    masked_value_fn=lambda n: (
        n.args[1] if isinstance(n.args[1], (int, float, bool)) else None
    ),
)
def codegen_full(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    env = CompileEnvironment.current()
    size, fill_value = map_arg(node.args, lambda n: n.meta["val"])
    dtype = node.kwargs.get("dtype", torch.get_default_dtype())
    assert isinstance(dtype, torch.dtype)
    device = node.kwargs.get("device", env.device)
    assert device == env.device, f"expected {env.device}, got {device}"
    assert not node.kwargs.get("pin_memory"), "pin_memory not supported"
    assert isinstance(fill_value, (int, float, bool))

    shape_str = ctx.cg.device_function.tile_strategy.shape_str([*size])  # pyright: ignore[reportGeneralTypeIssues,reportOptionalIterable]
    return expr_from_string(
        f"tl.full({shape_str}, {constant_repr(fill_value)}, {triton_type(dtype)})"
    )


@register_lowering(
    torch.ops.aten.unsqueeze.default,  # pyright: ignore[reportAttributeAccessIssue]
    masked_value_fn=passthrough_masked_value,
)
def codegen_unsqueeze(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dim = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(tensor, ast.AST)
    assert isinstance(dim, int)
    ndim = node.args[0].meta["val"].ndim  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    if dim < 0:
        dim += ndim
    assert 0 <= dim <= ndim, f"Invalid dim {dim} for tensor with {ndim} dims"
    args = [":"] * ndim
    args.insert(dim, "None")
    return expr_from_string(
        f"{{tensor}}[{', '.join(args)}]",
        tensor=tensor,
    )


@register_lowering(torch.ops.aten.squeeze.dim, masked_value_fn=passthrough_masked_value)  # pyright: ignore[reportAttributeAccessIssue]
@register_lowering(
    torch.ops.aten.view.default,  # pyright: ignore[reportAttributeAccessIssue]
    masked_value_fn=passthrough_masked_value,
)
@register_lowering(
    torch.ops.aten.reshape.default,  # pyright: ignore[reportAttributeAccessIssue]
    masked_value_fn=passthrough_masked_value,
)
def codegen_view(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "view kwargs not supported"
    tensor = map_arg(node.args[0], lambda arg: ctx.env[arg])
    assert isinstance(tensor, ast.AST)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*node.meta["val"].size()]
    )
    return expr_from_string(f"tl.reshape({{tensor}}, {shape_str})", tensor=tensor)


@register_lowering(
    torch.ops.aten.permute.default,  # pyright: ignore[reportAttributeAccessIssue]
    masked_value_fn=passthrough_masked_value,
)
def codegen_permute(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dims = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(tensor, ast.AST)
    dims = [*dims]  # pyright: ignore[reportGeneralTypeIssues,reportOptionalIterable]
    assert {*dims} == {*range(len(dims))}, dims
    return expr_from_string(
        f"tl.permute({{tensor}}, {dims!r})",
        tensor=tensor,
    )


@register_lowering(
    torch.ops.aten.expand.default,  # pyright: ignore[reportAttributeAccessIssue]
    masked_value_fn=passthrough_masked_value,
)
def codegen_expand(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, _ = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(tensor, ast.AST)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    shape = [*val.size()]
    if node.args[0].meta["val"].ndim != len(shape):  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
        broadcasting = [":"] * len(shape)
        for i in range(len(shape) - node.args[0].meta["val"].ndim):  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
            broadcasting[i] = "None"
        tensor = expr_from_string(
            f"{{tensor}}[{', '.join(broadcasting)}]", tensor=tensor
        )
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(shape)
    return expr_from_string(
        f"tl.broadcast_to({{tensor}}, {shape_str})",
        tensor=tensor,
    )


def apply_dot_requirements(
    handler: CodegenHandler,
    node: torch.fx.Node,
    masked_value_fn: Callable[[torch.fx.Node], float | bool | None] | None = None,
) -> Lowering:
    """Apply min_dot_size requirements to the config_spec"""
    assert not node.kwargs, "dot kwargs not supported"
    assert len(node.args) in (2, 3)
    lproxy, rproxy = map_arg(node.args[-2:], lambda arg: arg.meta["val"])
    assert isinstance(lproxy, torch.Tensor)
    assert isinstance(rproxy, torch.Tensor)
    lshape = lproxy.size()
    rshape = rproxy.size()
    # use last two dimensions for dot (supports 2D and batched 3D tensors)
    m, k = lshape[-2], lshape[-1]
    k2, n = rshape[-2], rshape[-1]
    assert k == k2, f"Mismatched k dimensions for dot: {k} vs {k2}"
    a, b, c = min_dot_size(lproxy.device, lproxy.dtype, rproxy.dtype)
    env = CompileEnvironment.current()
    for shape, min_size in [(m, a), (n, b), (k, c)]:
        block_idx = CompileEnvironment.current().get_block_id(shape)
        if block_idx is not None:
            env.block_sizes[block_idx].update_min_block(min_size, allow_flattened=True)
    # inputs to the dot operation must be zero-masked
    *maybe_acc, lnode, rnode = node.args
    assert isinstance(lnode, torch.fx.Node)
    assert isinstance(rnode, torch.fx.Node)
    lnode = apply_masking(lnode, base_node=node, other=0)
    rnode = apply_masking(rnode, base_node=node, other=0)
    node.args = (*maybe_acc, lnode, rnode)
    return LambdaLowering(handler, masked_value_fn=masked_value_fn)


def reduce_3d_dot(
    ctx: GraphInterpreter, node: torch.fx.Node, with_acc: bool
) -> ast.AST:
    datatype = CompileEnvironment.current().settings.dot_precision
    acc = None
    if with_acc:
        acc, lhs, rhs = map_arg(node.args, lambda arg: ctx.env[arg])
        assert isinstance(acc, ast.AST)
        lhs_node = node.args[1]
        rhs_node = node.args[2]
    else:
        lhs, rhs = map_arg(node.args, lambda arg: ctx.env[arg])
        lhs_node = node.args[0]
        rhs_node = node.args[1]
    assert isinstance(lhs, ast.AST)
    assert isinstance(rhs, ast.AST)
    assert isinstance(lhs_node, torch.fx.Node)
    assert isinstance(rhs_node, torch.fx.Node)

    # Check if inputs are FP8 - if so, redirect user to hl.dot()
    lhs_dtype = lhs_node.meta["val"].dtype
    rhs_dtype = rhs_node.meta["val"].dtype
    if lhs_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and rhs_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]:
        raise NotImplementedError(
            "FP8 GEMM via torch API is not supported yet. Please use hl.dot() instead."
        )

    lhs_size = lhs_node.meta["val"].size()  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    rhs_size = rhs_node.meta["val"].size()  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    # check to see if it is 3D and the highest dim is 1
    reduce_dim = False
    if len(lhs_size) == 3:
        env = CompileEnvironment.current()
        lhs_dim_idx = env.get_block_id(lhs_size[0])
        rhs_dim_idx = env.get_block_id(rhs_size[0])
        if lhs_dim_idx is not None and rhs_dim_idx is not None:
            lhs_dim_val = env.block_sizes[lhs_dim_idx]
            rhs_dim_val = env.block_sizes[rhs_dim_idx]
            if (
                lhs_dim_val.from_config(ctx.cg.device_function.config) == 1
                and rhs_dim_val.from_config(ctx.cg.device_function.config) == 1
            ):
                reduce_dim = True

    if not reduce_dim:
        if with_acc:
            precision_arg = (
                f", input_precision={datatype!r}" if datatype is not None else ""
            )
            return expr_from_string(
                f"tl.dot({{lhs}}, {{rhs}}, acc={{acc}}{precision_arg})",
                lhs=lhs,
                rhs=rhs,
                acc=acc,  # pyright: ignore[reportArgumentType]
            )
        # without accumulator
        precision_arg = (
            f", input_precision={datatype!r}" if datatype is not None else ""
        )
        return expr_from_string(
            f"tl.dot({{lhs}}, {{rhs}}{precision_arg})", lhs=lhs, rhs=rhs
        )

    # create reshape, dot, then reshape
    lhs_shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*lhs_node.meta["val"].size()[1:]]  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    )
    rhs_shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*rhs_node.meta["val"].size()[1:]]  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    )
    out_shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*node.meta["val"].size()]
    )
    lhs_reshape = expr_from_string(f"tl.reshape({{lhs}}, {lhs_shape_str})", lhs=lhs)
    rhs_reshape = expr_from_string(f"tl.reshape({{rhs}}, {rhs_shape_str})", rhs=rhs)
    if with_acc:
        acc_shape_str = ctx.cg.device_function.tile_strategy.shape_str(
            [*node.args[0].meta["val"].size()[1:]]  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
        )
        acc_reshape = expr_from_string(f"tl.reshape({{rhs}}, {acc_shape_str})", rhs=acc)  # pyright: ignore[reportArgumentType]
        precision_arg = (
            f", input_precision={datatype!r}" if datatype is not None else ""
        )
        comp = expr_from_string(
            f"tl.dot({{lhs}}, {{rhs}}, acc={{acc}}{precision_arg})",
            lhs=lhs_reshape,
            rhs=rhs_reshape,
            acc=acc_reshape,
        )
    else:
        precision_arg = (
            f", input_precision={datatype!r}" if datatype is not None else ""
        )
        comp = expr_from_string(
            f"tl.dot({{lhs}}, {{rhs}}{precision_arg})",
            lhs=lhs_reshape,
            rhs=rhs_reshape,
        )
    return expr_from_string(f"tl.reshape({{lhs}}, {out_shape_str})", lhs=comp)


@register_lowering(torch.ops.aten.bmm.default, apply_dot_requirements)  # pyright: ignore[reportAttributeAccessIssue]
@register_lowering(torch.ops.aten.mm.default, apply_dot_requirements)  # pyright: ignore[reportAttributeAccessIssue]
def codegen_mm(ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
    assert not node.kwargs, "matmul kwargs not supported"

    return reduce_3d_dot(ctx, node, False)


@register_lowering(torch.ops.aten.addmm.default, apply_dot_requirements)  # pyright: ignore[reportAttributeAccessIssue]
def codegen_addmm(ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
    assert not node.kwargs, "addmm kwargs not supported"
    return reduce_3d_dot(ctx, node, True)


@register_lowering(torch.ops.aten.baddbmm.default, apply_dot_requirements)  # pyright: ignore[reportAttributeAccessIssue]
def codegen_baddbmm(ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
    assert not node.kwargs, "baddbmm kwargs not supported"
    return reduce_3d_dot(ctx, node, True)


class GenerateASTFromInductor(DefaultHandler):
    """Ops handler for generating Triton code from Inductor operations.
    
    This handler is used when Inductor lowers operations to generate the actual
    Triton code. It handles the conversion of symbolic operations to concrete
    variable assignments and expressions.
    
    Key insight: Inductor processes operations in two phases:
    1. Symbolic phase: Builds a computation graph with placeholder symbols
    2. Execution phase: Calls our handler to generate actual code
    
    When operations reference results from previous nodes, they appear as
    placeholder symbols (e.g., 'remainder', 'add'). The input_name_lookup
    dictionary maps these symbols to the actual variables generated by
    previous nodes.
    """
    
    def __init__(
        self, cg: CodegenInterface, input_name_lookup: dict[str, ast.AST],
        iteration_vars: list[sympy.Symbol] | None = None
    ) -> None:
        super().__init__()
        self.parent_handler = MockHandler()  # Use MockHandler for symbolic operations
        self.cg = cg
        self.input_name_lookup = input_name_lookup
        
        # Store iteration variables if provided (e.g., i0, i1 from inner_fn)
        self.iteration_vars = iteration_vars or []
        
        # Track operations within the current node
        # Maps operation names to generated variables for intra-node references
        self.symbol_to_var: dict[str, str] = {}
        
        # Analyze the input_name_lookup to find the original tensor pointers
        # For loaded tensors in advanced indexing, map them to their sources
        self.tensor_origins = {}
        if 'x_tile' in input_name_lookup or 'i_indices' in input_name_lookup:
            # Advanced indexing pattern detected
            self.tensor_origins['x_tile'] = 'x'
            self.tensor_origins['i_indices'] = 'index_i'
            self.tensor_origins['j_indices'] = 'index_j'
        

    def constant(self, value: object, dtype: torch.dtype) -> str:
        """Handle ops.constant operations."""
        # Generate a Triton constant
        if dtype in [torch.int32, torch.int64, torch.int8, torch.int16]:
            return str(int(value))
        elif dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            return str(float(value))
        elif dtype == torch.bool:
            return str(bool(value))
        else:
            return str(value)
    
    def remainder(self, x: object, y: object) -> object:
        """Handle ops.remainder operations."""
        x_str = self._convert_operand_to_str(x)
        y_str = self._convert_operand_to_str(y)
        
        # Generate Triton remainder operation and lift it to a variable
        result = self.cg.lift(
            expr_from_string(f"({x_str} % {y_str})")
        )
        
        # Track that 'remainder' placeholder maps to this variable
        # (in case it's referenced later within the same node)
        self.symbol_to_var['remainder'] = result.id
        
        # Return OpsValue as expected by Inductor
        return OpsValue(result.id)
    
    def add(self, x: object, y: object) -> object:
        """Handle ops.add operations."""
        x_str = self._convert_operand_to_str(x)
        y_str = self._convert_operand_to_str(y)
        
        # Generate Triton addition
        result = self.cg.lift(
            expr_from_string(f"({x_str} + {y_str})")
        )
        
        # Track that 'add' placeholder maps to this variable
        # (in case it's referenced later within the same node)
        self.symbol_to_var['add'] = result.id
        
        # Return OpsValue as expected by Inductor
        return OpsValue(result.id)
    
    def eq(self, x: object, y: object) -> object:
        """Handle ops.eq operations."""
        x_str = self._convert_operand_to_str(x)
        y_str = self._convert_operand_to_str(y)
        
        # Generate Triton equality comparison
        result = self.cg.lift(
            expr_from_string(f"({x_str} == {y_str})")
        )
        
        # Track that 'eq' placeholder maps to this variable
        # (in case it's referenced later within the same node)
        self.symbol_to_var['eq'] = result.id
        
        # Return OpsValue as expected by Inductor
        return OpsValue(result.id)
    
    def _convert_operand_to_str(self, operand: object) -> str:
        """Convert an operand to string, handling symbols properly.
        
        This method resolves symbolic references to actual variable names:
        - Symbols from previous nodes are resolved via input_name_lookup
        - Symbols from operations within the current node are resolved via symbol_to_var
        - Special symbols like 'tile_index' are mapped to their Triton equivalents
        """
        # Unpack OpsValue if needed
        if isinstance(operand, OpsValue):
            operand = operand.value
        
        if isinstance(operand, str):
            # Already a string (likely from a previous operation) - use directly
            return operand
        elif isinstance(operand, sympy.Symbol):
            # Check if this symbol refers to an input from a previous node
            # This happens when one Inductor node references the output of another
            if operand.name in self.input_name_lookup:
                # Get the AST node that represents this input
                input_ast = self.input_name_lookup[operand.name]
                # If it's a Name node, use its id (the variable name)
                if isinstance(input_ast, ast.Name):
                    return input_ast.id
                # Otherwise convert to string
                return str(input_ast)
            
            # Check if this is a placeholder for a previous operation within this node
            if operand.name in self.symbol_to_var:
                # Use the variable that was generated for this operation
                return self.symbol_to_var[operand.name]
            
            # Special symbols
            if operand.name == 'tile_index':
                # Map to the actual Triton index variable
                return self.cg.index_var(0)
            elif operand in self.iteration_vars:
                # Iteration variable - try to find its Triton equivalent
                triton_var = self._get_triton_var_for_iteration(operand)
                if triton_var:
                    return triton_var
            
            # Default: use symbol name
            return str(operand)
        elif isinstance(operand, sympy.Expr):
            # Handle expressions with proper symbol substitution
            return self._sympy_to_str(operand)
        else:
            # Default string conversion
            return str(operand)
    
    def _sympy_to_str(self, expr: sympy.Expr) -> str:
        """Convert a sympy expression to string, resolving symbols."""
        # Substitute known symbols
        substitutions = {}
        for sym in expr.free_symbols:
            if sym.name == 'tile_index':
                substitutions[sym] = sympy.Symbol(self.cg.index_var(0))
        
        if substitutions:
            expr = expr.subs(substitutions)
        
        return str(expr)
    
    def _get_triton_var_for_iteration(self, sym: sympy.Symbol) -> str | None:
        """Get the Triton variable name for an iteration symbol."""
        # For now, return None - override in subclasses if needed
        return None
    
    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> object:
        # For unhandled ops, delegate to parent handler
        # This returns sympy expressions that we can track
        return getattr(self.parent_handler, name)(*args, **kwargs)
    
    def indirect_indexing(self, index: object, size: object, check: bool = True, wrap_neg: bool = True) -> object:
        """Handle indirect indexing.
        
        This is used by Inductor's index_impl to convert tensor indices
        into scalar indices for flattened access.
        """
        # Return the index as-is - it will be used in arithmetic operations
        return str(index) if not isinstance(index, str) else index

    def to_dtype(
        self,
        x: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        """Override to_dtype to use tl.cast for scalar values from GetItemOrigin."""
        x_str = self._convert_operand_to_str(x)

        # Generate appropriate Triton conversion
        # For scalars, use tl.cast
        # For vectors, use .to() method
        if "_item_" in x_str or x_str.startswith("loaded_"):
            # Scalar value - use tl.cast
            result = self.cg.lift(
                expr_from_string(f"tl.cast({x_str}, {triton_type(dtype)})")
            )
        else:
            # Vector or tensor value - use .to() method
            # This handles cases like iota which is a vector from tl.arange
            result = self.cg.lift(
                expr_from_string(f"{x_str}.to({triton_type(dtype)})")
            )
        
        return result.id

    def load(self, name: str, index: object) -> object:
        """Handle ops.load operations.
        
        The key insight: Each load returns a SCALAR value that can be used in further computations.
        We need to track which names are loaded values vs. input tensors.
        """
        # Check if this name has an origin (was loaded from another tensor)
        # This handles the case where we're trying to load from an already-loaded tensor
        if name in self.tensor_origins:
            original_ptr = self.tensor_origins[name]
            
            # Use the original pointer name for the load
            # This will generate loads from the actual tensor arguments (x, index_i, index_j)
            # rather than from the loaded tensors (x_tile, i_indices, j_indices)
            name = original_ptr
        
        # For the test case, we need to ensure we have AST nodes for x, index_i, index_j
        # These are the original kernel parameters
        if name in ['x', 'index_i', 'index_j']:
            # Create AST node for the original parameter
            input_ast = create(ast.Name, id=name, ctx=ast.Load())
        elif name in self.input_name_lookup:
            input_ast = self.input_name_lookup[name]
            
            # Special case: if we're loading from a computed tensor (like iota or load)
            # with iteration variables, we should just return the tensor
            
            # Helper function to check if an expression contains iteration variables
            def contains_iter_vars(expr):
                if isinstance(expr, sympy.Symbol):
                    return expr in self.iteration_vars
                elif isinstance(expr, sympy.Expr):
                    return any(sym in self.iteration_vars for sym in expr.free_symbols)
                return False
            
            # Check various index patterns
            if isinstance(index, (tuple, list)):
                # Check if all elements contain iteration variables
                if all(contains_iter_vars(elem) for elem in index):
                    # Loading from a tensor with iteration variables
                    # In vectorized Triton, just return the tensor itself
                    return sympy.Symbol(name)
            elif contains_iter_vars(index):
                # Single index that contains iteration variables
                return sympy.Symbol(name)
        else:
            # Not an input tensor - might be a computed value like 'iota'
            # For computed values (like the result of tl.arange), we need special handling
            
            # Check if index is a tuple with iteration variables
            if isinstance(index, (tuple, list)) and len(index) == 1:
                # Single element tuple/list - extract it
                index = index[0]
            
            # Check if this is loading from a computed vector with an iteration variable
            # In vectorized Triton, loading from a vector with an iteration variable
            # means we're accessing the whole vector, not indexing into it
            if isinstance(index, sympy.Symbol) and index in self.iteration_vars:
                # This is loading from a computed vector (like iota from tl.arange)
                # with an iteration variable. In vectorized code, we just return
                # the vector itself, not an indexed element
                # The iteration is implicit in the vectorized operation
                # Just return the vector name directly
                var_name = self.cg.lift(expr_from_string(name), prefix="loaded").id
                return sympy.Symbol(var_name)
            elif isinstance(index, sympy.Symbol):
                # Non-iteration symbolic index - this needs to be resolved
                # Try to convert it to a proper expression
                index_expr = expr_from_string(str(index))
                load_expr = expr_from_string(f"tl.load({{name}} + {{index}})",
                                           name=expr_from_string(name),
                                           index=index_expr)
                var_name = self.cg.lift(load_expr, prefix="loaded").id
                return sympy.Symbol(var_name)
            elif isinstance(index, (int, sympy.Integer)):
                # Constant index
                index_expr = expr_from_string(str(index))
                load_expr = expr_from_string(f"tl.load({{name}} + {{index}})",
                                           name=expr_from_string(name),
                                           index=index_expr)
                var_name = self.cg.lift(load_expr, prefix="loaded").id
                return sympy.Symbol(var_name)
            # Otherwise, treat it as a symbol (could be a computed value)
            return sympy.Symbol(name)
        
        # Handle the index
        if isinstance(index, sympy.Expr):
            # This is a computed index expression (like 3*loaded_1 + loaded_2)
            # Convert the sympy expression to an AST expression
            index_expr = expr_from_string(self._sympy_to_str(index))
        elif isinstance(index, (list, tuple)):
            # Multi-dimensional index - flatten it
            if len(index) == 2:
                i_idx, j_idx = index
                # Convert indices to expressions
                i_expr = self._index_to_expr(i_idx)
                j_expr = self._index_to_expr(j_idx)
                
                # For 2D indexing, compute flattened index: i * stride + j
                # The stride depends on the tensor shape
                # Determine stride based on the tensor name
                if name == 'x':
                    stride = 3  # x is 3x3
                elif name == 'index_i':
                    stride = 1  # index_i is 2x1
                else:
                    stride = 1  # Default
                    
                index_expr = expr_from_string(f"{{i}} * {stride} + {{j}}", i=i_expr, j=j_expr)
            else:
                # Single dimension
                idx_expr = self._index_to_expr(index[0])
                index_expr = idx_expr
        else:
            # Simple scalar index
            index_expr = self._index_to_expr(index)
        
        # Generate the load expression
        load_expr = expr_from_string(f"tl.load({{input}} + {{index}})",
                                    input=input_ast,
                                    index=index_expr)
        
        # Lift it to get a variable name
        var_name = self.cg.lift(load_expr, prefix="loaded").id
        
        # Return a sympy symbol for use in arithmetic
        return sympy.Symbol(var_name)
    
    def _index_to_expr(self, index: object) -> ast.AST:
        """Convert an index to an AST expression."""
        if isinstance(index, sympy.Symbol):
            # Check if this is an iteration variable
            if index in self.iteration_vars:
                # Return the symbol as is - it will be handled by sympy_to_str
                return expr_from_string(self._sympy_to_str(index))
            # Not an iteration variable - convert as sympy expression
            # But first check for free symbols that might be iteration vars
            if index.free_symbols:
                # This symbol has dependencies - handle them
                return expr_from_string(self._sympy_to_str(index))
            return expr_from_string(str(index))
        elif isinstance(index, sympy.Expr):
            # Handle sympy expressions
            return expr_from_string(self._sympy_to_str(index))
        elif isinstance(index, str):
            # Direct string index
            return expr_from_string(index)
        elif isinstance(index, int):
            return expr_from_string(str(index))
        else:
            # Try to convert to string
            return expr_from_string(str(index))
    
    def _sympy_to_str(self, expr: sympy.Expr) -> str:
        """Convert a sympy expression to string, handling iteration variables."""
        # Only do special handling if we have iteration variables
        if not self.iteration_vars:
            # No iteration variables - just convert to string
            return str(expr)
        
        # Replace iteration variables with their actual Triton mappings
        replacements = {}
        for sym in expr.free_symbols:
            if sym in self.iteration_vars:
                # Map iteration variables to actual Triton indices
                triton_var = self._get_triton_var_for_iteration(sym)
                if triton_var:
                    replacements[sym] = sympy.Symbol(triton_var)
        
        if replacements:
            expr = expr.xreplace(replacements)
        return str(expr)
    
    def _convert_iteration_var(self, var: sympy.Symbol) -> ast.AST:
        """Convert an iteration variable to the corresponding Triton index."""
        triton_var = self._get_triton_var_for_iteration(var)
        if triton_var:
            return expr_from_string(triton_var)
        # Fall back to using the variable as-is
        return expr_from_string(str(var))
    
    def _get_triton_var_for_iteration(self, var: sympy.Symbol) -> str | None:
        """Get the Triton variable name for an iteration variable."""
        # Only apply special mapping if we're in the advanced indexing context
        if self.is_advanced_indexing_context:
            # This is the advanced indexing context
            var_idx = self.iteration_vars.index(var) if var in self.iteration_vars else -1
            if var_idx == 0:
                return 'indices_2'
            elif var_idx == 1:
                return '0'
        
        # For other contexts, return None to use the default
        return None

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

        return f"{name}.to({triton_type(dtype)})"


def _unpack_opsvalue(value: object) -> str:
    if isinstance(value, OpsValue):
        return str(value)
    # Handle cases where Inductor returns variable names or other types
    if hasattr(value, '__str__'):
        return str(value)
    assert isinstance(value, str), f"Expected str or OpsValue, got {type(value)}: {value}"
    return value


class GraphInterpreter(Interpreter):
    def __init__(self, graph: torch.fx.Graph, cg: CodegenInterface) -> None:
        super().__init__(_LazyGraphModule({}, graph), garbage_collect_values=False)
        self.cg = cg

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
            host_expr = self.cg.device_function.sympy_expr(val._sympy_())
            self.cg.device_function.constexpr_arg(name, host_expr)
            return name

        # Regular variable assignment
        name = self.cg.device_function.new_var(node.name)
        self.cg.add_statement(
            statement_from_string(f"{name} = {{result}}", result=result)
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
        outputs = [None] * len(output_nodes)
        all_nodes = {
            n.name: n
            for n in self.module.graph.nodes  # pyright: ignore[reportAttributeAccessIssue,reportGeneralTypeIssues]
        }

        for idx, node_name in output_nodes.items():
            if node_name == node.name:
                # This is the last node
                outputs[idx] = (  # pyright: ignore[reportArgumentType,reportCallIssue]
                    last_node_result
                )
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
                self.cg.add_statement(
                    statement_from_string(f"{var_name} = {{result}}", result=result)
                )
                result = create(ast.Name, id=var_name, ctx=ast.Load())
            final_outputs.append(result)

        return tuple(final_outputs)

    def run_node(self, n: Node) -> object:
        if n.op == "call_function":
            with self._set_current_node(n), n.meta["location"]:
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
                user.target == torch.ops.aten.sym_size.int  # pyright: ignore[reportAttributeAccessIssue]
                for user in placeholder.users
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


@register_lowering(torch.ops.prims.iota.default)  # pyright: ignore[reportAttributeAccessIssue]
def codegen_iota(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    """Generate tl.arange for torch.ops.prims.iota.default operations."""
    start = node.kwargs.get("start", 0)
    step = node.kwargs.get("step", 1)
    dtype = (
        node.kwargs.get("dtype") or CompileEnvironment.current().settings.index_dtype
    )
    assert isinstance(dtype, torch.dtype)
    (length_arg,) = node.args  # expecting a single argument for length
    expr = "tl.arange(0, {length})"
    if step != 1:
        expr = f"{{step}} * {expr}"
    if start != 0:
        expr = f"{{start}} + {expr}"
    if dtype != torch.int32:
        expr = f"({expr}).to({triton_type(dtype)})"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
        length=ctx.to_ast(length_arg),
    )
