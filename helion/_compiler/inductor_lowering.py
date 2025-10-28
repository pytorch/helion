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
from triton import next_power_of_2

from .. import exc
from ..exc import InductorLoweringError
from ..language._decorators import APIFunc
from ..language._decorators import is_api_func
from ..language.matmul_ops import enforce_dot_requirements
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .compile_environment import FixedBlockSizeSource
from .device_function import SymbolArgument
from .device_function import VarInfo
from .device_function import contains_only_block_size_symbols
from .dtype_utils import cast_ast
from .matmul_utils import emit_tl_dot_with_padding
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
        self, ctx: GraphInterpreter, node: torch.fx.Node
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
    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        # Validate broadcasting of tile block dimensions to catch shape mismatches
        self._check_block_broadcast_compatibility(node)
        with self.install_kernel_handlers(ctx, node):
            indices = [
                sympy.Symbol(f"i{n}") for n in range(len(self.buffer.data.ranges))
            ]
            output_name = _unpack_opsvalue(self.buffer.data.inner_fn(indices))
            return expr_from_string(output_name)

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
        desired_dtype = node.meta["val"].dtype  # pyright: ignore[reportAttributeAccessIssue]
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
    size = map_arg(node.args[0], lambda n: n.meta["val"])
    dtype = node.kwargs.get("dtype", torch.get_default_dtype())
    assert isinstance(dtype, torch.dtype)
    device = node.kwargs.get("device", env.device)
    assert device == env.device, f"expected {env.device}, got {device}"
    assert not node.kwargs.get("pin_memory"), "pin_memory not supported"
    value_ast = map_arg(node.args[1], lambda arg: ctx.env[arg])
    if isinstance(value_ast, (int, float, bool)):
        value_ast = expr_from_string(constant_repr(value_ast))
    assert isinstance(value_ast, ast.AST), value_ast
    shape_str = ctx.cg.device_function.tile_strategy.shape_str([*size])  # pyright: ignore[reportGeneralTypeIssues,reportOptionalIterable]
    return expr_from_string(
        f"tl.full({shape_str}, {{value}}, {triton_type(dtype)})",
        value=value_ast,
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
    torch.ops.aten.stack.default,  # pyright: ignore[reportAttributeAccessIssue]
    masked_value_fn=passthrough_masked_value,
)
def codegen_stack(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)

    assert isinstance(tensors, (list, tuple))
    tensor_asts = [ctx.env[t] for t in tensors]  # pyright: ignore[reportArgumentType]
    n = len(tensor_asts)

    if n == 0:
        raise ValueError("Cannot stack empty tensor list")

    # Round up to power of 2 for efficient masking
    padded_size = 1 << (n - 1).bit_length()

    # Create index array [0, 1, 2, 3, ...] for tensor selection
    idx = ctx.cg.device_function.new_var("stack_idx")
    ctx.cg.add_statement(statement_from_string(f"{idx} = tl.arange(0, {padded_size})"))

    # Broadcast index to target dimension shape
    # e.g., dim=0: [:, None, None], dim=1: [None, :, None], dim=2: [None, None, :]
    bidx = ctx.cg.device_function.new_var("broadcast_idx")
    assert isinstance(dim, int)
    pattern = "[" + ", ".join(["None"] * dim + [":"] + ["None"] * max(0, 2 - dim)) + "]"
    ctx.cg.add_statement(statement_from_string(f"{bidx} = {idx}{pattern}"))

    # Expand each input tensor along the stack dimension
    expanded = [ctx.cg.device_function.new_var(f"expanded_{i}") for i in range(n)]
    for var, tensor in zip(expanded, tensor_asts, strict=False):
        ctx.cg.add_statement(
            statement_from_string(f"{var} = tl.expand_dims({{t}}, {dim})", t=tensor)
        )

    # Initialize result with zeros
    result = ctx.cg.device_function.new_var("stacked_result")
    ctx.cg.add_statement(
        statement_from_string(f"{result} = tl.zeros_like({expanded[0]})")
    )

    # Select each tensor using masks
    for i in range(n):
        mask = ctx.cg.device_function.new_var(f"mask_{i}")
        ctx.cg.add_statement(statement_from_string(f"{mask} = {bidx} == {i}"))
        ctx.cg.add_statement(
            statement_from_string(
                f"{result} = tl.where({mask}, {expanded[i]}, {result})"
            )
        )

    return expr_from_string(result)


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
    # Update config spec min sizes for M, N, K
    enforce_dot_requirements(lproxy, rproxy)
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
    acc = None
    acc_node: torch.fx.Node | None = None
    if with_acc:
        acc, lhs, rhs = map_arg(node.args, lambda arg: ctx.env[arg])
        assert isinstance(acc, ast.AST)
        assert isinstance(node.args[0], torch.fx.Node)
        acc_node = node.args[0]
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
    acc_dtype_meta: torch.dtype | None = None
    if with_acc:
        assert acc_node is not None
        assert isinstance(acc_node, torch.fx.Node)
        acc_dtype_meta = acc_node.meta["val"].dtype
    if lhs_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and rhs_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]:
        raise NotImplementedError(
            "FP8 GEMM via torch API is not supported yet. Please use hl.dot() instead."
        )

    lhs_shape = list(lhs_node.meta["val"].size())  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    rhs_shape = list(rhs_node.meta["val"].size())  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    acc_shape = (
        list(acc_node.meta["val"].size())
        if (with_acc and acc_node is not None)
        else None
    )  # pyright: ignore[reportOptionalMemberAccess]

    # Extract expected output dtype from FX node to match PyTorch eager mode behavior
    out_dtype: torch.dtype | None = None
    if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
        out_dtype = node.meta["val"].dtype

    return emit_tl_dot_with_padding(
        lhs,
        rhs,
        acc if with_acc else None,
        lhs_dtype,
        rhs_dtype,
        acc_dtype=acc_dtype_meta if with_acc else None,
        out_dtype=out_dtype,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        acc_shape=acc_shape,
    )


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
    def __init__(
        self, cg: CodegenInterface, input_name_lookup: dict[str, ast.AST]
    ) -> None:
        super().__init__()
        self.parent_handler = TritonOverrides()
        self.cg = cg
        self.input_name_lookup = input_name_lookup

    def _expected_tensor_dtype(self) -> torch.dtype | None:
        """Best-effort retrieval of the current FX node's tensor dtype."""
        current_node = V.current_node
        if current_node is None:
            return None
        val = current_node.meta.get("val")
        if isinstance(val, torch.Tensor):
            return val.dtype
        return None

    def _create_cast_expr(self, x: object, target_dtype_str: str) -> ast.AST:
        """Create a tl.cast expression from AST or string input.

        Args:
            x: Input value (AST node or string/OpsValue)
            target_dtype_str: Target Triton dtype as string (e.g., "tl.float32")

        Returns:
            AST expression for the cast operation
        """
        if isinstance(x, ast.AST):
            return expr_from_string(f"tl.cast({{x}}, {target_dtype_str})", x=x)
        base = _unpack_opsvalue(x)
        return expr_from_string(f"tl.cast({base}, {target_dtype_str})")

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
        return self._create_cast_expr(expr, triton_type(expected_dtype))

    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> str:
        result_str = _unpack_opsvalue(
            getattr(self.parent_handler, name)(*args, **kwargs)
        )

        return self.cg.lift(expr_from_string(result_str)).id

    def to_dtype(
        self,
        x: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        """Emit explicit tl.cast to enforce final dtype conversion.

        We avoid delegating to the parent handler to prevent reliance on global
        device context during compute-type selection, and to guarantee a visible
        cast in generated code that matches PyTorch's dtype semantics.
        """
        cast_expr = self._create_cast_expr(x, triton_type(dtype))
        return self.cg.lift(cast_expr).id

    def _is_scalar_like_str(self, x_str: str) -> bool:
        """Best-effort detection for scalar-origin expressions.

        Today we rely on GetItem-origin naming containing "_item_"; centralize
        this heuristic so future improvements can be made in one place.
        """
        return "_item_" in x_str

    # Ensure non-linear elementwise ops receive fp32 inputs for Triton
    def sigmoid(self, x: object) -> str:  # type: ignore[override]
        # Build tl.sigmoid(tl.cast(x, tl.float32)) and lift
        inner = self._create_cast_expr(x, "tl.float32")
        result = expr_from_string("tl.sigmoid({x})", x=inner)

        # Only cast if expected dtype is not float32
        expected_dtype = self._expected_tensor_dtype()
        if expected_dtype is not None and expected_dtype != torch.float32:
            result = self._maybe_cast_to_expected_dtype(result)

        return self.cg.lift(result).id

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

        return self.cg.lift(result_expr).id

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

        return f"{name}.to({triton_type(dtype)})"


def _unpack_opsvalue(value: object) -> str:
    if isinstance(value, OpsValue):
        return str(value)
    assert isinstance(value, str)
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
                torch.ops.aten.unsqueeze.default,  # pyright: ignore[reportAttributeAccessIssue]
                torch.ops.aten.view.default,  # pyright: ignore[reportAttributeAccessIssue]
                torch.ops.aten.reshape.default,  # pyright: ignore[reportAttributeAccessIssue]
                torch.ops.aten.expand.default,  # pyright: ignore[reportAttributeAccessIssue]
                torch.ops.aten.permute.default,  # pyright: ignore[reportAttributeAccessIssue]
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
    """Generate tl.arange for torch.ops.prims.iota.default operations with automatic power-of-2 padding."""
    start = node.kwargs.get("start", 0)
    step = node.kwargs.get("step", 1)
    dtype = (
        node.kwargs.get("dtype") or CompileEnvironment.current().settings.index_dtype
    )
    assert isinstance(dtype, torch.dtype)
    (length_arg,) = node.args  # expecting a single argument for length

    # Pad static non-power-of-2 lengths to next power of 2
    length_expr = "{length}"
    if isinstance(length_arg, int) and length_arg != next_power_of_2(length_arg):
        length_expr = str(next_power_of_2(length_arg))

    expr = f"tl.arange(0, {length_expr})"
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


def _codegen_rng_op(
    ctx: GraphInterpreter,
    node: torch.fx.Node,
    rng_function: str,
) -> object:
    """Common codegen implementation for all RNG operations.

    Args:
        ctx: The graph interpreter context
        node: The FX node for this operation
        rng_function: Either "rand" or "randn"
    """
    assert rng_function in ["rand", "randn"]

    # Get unique seed index for this RNG operation
    device_fn = ctx.cg.device_function
    seed_index = device_fn.allocate_rng_seed()

    # Get dimensionality and dtype
    assert hasattr(node, "meta") and "val" in node.meta
    ndim = node.meta["val"].ndim
    dtype = node.kwargs.get("dtype", None)

    # Get the dimension variable names from the device function's symbol arguments
    device_fn = ctx.cg.device_function
    symbol_args = [
        arg for arg in device_fn.arguments if isinstance(arg, SymbolArgument)
    ]

    # Extract dimension names - they should be the last ndim symbol arguments
    dim_names = []
    assert len(symbol_args) >= ndim, "Not enough symbol arguments for dimensions"
    dim_names = [arg.name for arg in symbol_args[-ndim:]]

    offset_parts = []

    for i in range(ndim):
        # Create the index variable with proper broadcasting
        index_expr = f"indices_{i}"

        # Add broadcasting slices for this dimension
        # For 1D tensors, this will just be indices_0 with no slicing
        slice_parts = []
        for j in range(ndim):
            if j < i:
                slice_parts.append("None")
            elif j == i:
                slice_parts.append(":")
            else:
                slice_parts.append("None")

        # Create the broadcasted index expression
        if ndim == 1:
            # For 1D, no broadcasting needed
            broadcasted_index = index_expr
        else:
            broadcasted_index = f"{index_expr}[{', '.join(slice_parts)}]"

        # Calculate stride (product of dimensions after this one)
        if i < ndim - 1:
            # Use the actual dimension variable names
            stride_parts = dim_names[i + 1 :]
            stride_expr = " * ".join(stride_parts)
            offset_parts.append(f"{broadcasted_index} * {stride_expr}")
        else:
            # Last dimension has no stride multiplication
            offset_parts.append(broadcasted_index)

    offset_expr = expr_from_string(" + ".join(offset_parts))

    # Load seed from buffer using the kernel parameter name
    assert device_fn.rng_seed_buffer_param_name is not None
    seed_expr = expr_from_string(
        "tl.load({buffer} + {index})",
        buffer=expr_from_string(device_fn.rng_seed_buffer_param_name),
        index=create(ast.Constant, value=seed_index),
    )

    # Generate the RNG call
    # Note: tl.rand() and tl.randn() always return float32
    rng_expr = expr_from_string(
        f"tl.{rng_function}({{seed}}, {{offset}})", seed=seed_expr, offset=offset_expr
    )

    # Cast to target dtype only if explicitly specified
    if dtype is not None:
        assert isinstance(dtype, torch.dtype)
        rng_expr = expr_from_string(f"{{val}}.to({triton_type(dtype)})", val=rng_expr)

    return rng_expr


@register_lowering(torch.ops.aten.rand.default)  # pyright: ignore[reportAttributeAccessIssue]
def codegen_rand(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    return _codegen_rng_op(ctx, node, "rand")


@register_lowering(torch.ops.aten.randn.default)  # pyright: ignore[reportAttributeAccessIssue]
def codegen_randn(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    return _codegen_rng_op(ctx, node, "randn")
