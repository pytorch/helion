from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import dataclasses
import functools
import logging
import math
import operator
import re
import textwrap
import threading
from typing import TYPE_CHECKING
from typing import Callable
from typing import Iterator
from typing import NamedTuple
from typing import Protocol
from typing import cast
from unittest.mock import patch

import torch
from torch._dynamo.convert_frame import compile_lock
from torch._inductor.decomposition import select_decomp_table
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.fx.experimental import proxy_tensor
from torch.fx.traceback import preserve_node_meta
from torch.utils import _pytree as pytree

from .. import Config
from .. import exc
from .. import language as hl
from ..autotuner.config_spec import FULL_EXTENT_CATEGORIES
from ..autotuner.config_spec import SIZED_REDUCTION_CATEGORIES
from ..autotuner.config_spec import CoResidencyGroup
from ..autotuner.config_spec import CuteVectorWidthSpec
from ..autotuner.config_spec import MatmulWithReductionEpilogueFact
from ..autotuner.config_spec import ReductionCategory
from ..autotuner.config_spec import ReductionDescriptor
from ..autotuner.config_spec import ReductionKernelFact
from ..autotuner.config_spec import ReductionLoopSpec
from ..language import _tracing_ops
from ..language._decorators import args_to_proxies
from ..language._decorators import get_device_func_replacement
from ..language._tracing_ops import _new_var
from ..language.tile_proxy import Tile
from ..language.tile_proxy import _CheckForIndexCalls
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import NodeVisitor
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_read_writes import ReadWrites
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .inductor_lowering import APIFuncLowering
from .inductor_lowering import CodegenState
from .inductor_lowering import codegen_call_with_graph
from .inductor_lowering import prepare_graph_lowerings
from .matmul_utils import tensor_matmul_replacement
from .matmul_utils import torch_matmul_replacement
from .node_masking import defer_pallas_load_masks
from .node_masking import remove_unnecessary_masking
from .roll_reduction import ReductionRoller
from .source_location import current_location
from .type_info import CallableType
from .type_info import DictType
from .type_info import GridIndexType
from .type_info import IterType
from .type_info import JaggedTileIndexType
from .type_info import LiteralType
from .type_info import NumericType
from .type_info import SequenceType
from .type_info import StackTensorType
from .type_info import TensorType
from .type_info import TileIndexType
from .type_info import TypeInfo
from .type_info import _eval_binary
from .type_info import _eval_compare
from .type_info import _eval_unary

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..autotuner.config_spec import AccumulatorFact
    from ..autotuner.config_spec import ConfigSpec
    from ..autotuner.config_spec import MemoryOpFact
    from .cute.layout import CuTeGridExecutionPlan
    from .device_ir_analysis import DeviceIRAnalysis
    from .tile_dependency import TaskFamily
    from .tile_dependency import TileDependencyGraph

    class _TLS(Protocol):
        device_irs: list[DeviceIR]


tls: _TLS = cast("_TLS", threading.local())

log = logging.getLogger(__name__)


def _lerp_scalar_decomp(
    start: torch.Tensor, end: torch.Tensor, weight: float
) -> torch.Tensor:
    # PyTorch nightly's inductor _lerp_scalar decomposition branches on
    # `weight >= 0.5` for numerical stability.  Helion traces scalar kernel
    # args as unbacked symfloats, so that comparison raises
    # GuardOnDataDependentSymNode.  Use the simple algebraic form instead.
    return start + weight * (end - start)


def _get_custom_decomp_table() -> dict[torch._ops.OpOverload, Callable[..., object]]:
    from ..language._gelu_tanh_approx import install_gelu_decomp

    decomp_table = select_decomp_table().copy()
    # Normally, aten.stack is decomposed to aten.unsqueeze + aten.cat, but it's difficult to
    # figure out the right Triton implementation for aten.cat. As a workaround, we disable
    # the decomp for aten.stack and implement aten.stack in Triton (codegen_stack) instead.
    decomp_table.pop(torch.ops.aten.stack.default, None)
    # Override lerp.Scalar to avoid data-dependent guard on the weight parameter.
    decomp_table[torch.ops.aten.lerp.Scalar] = _lerp_scalar_decomp
    # Map F.gelu(x, approximate="tanh") to a single _gelu_tanh_approx FX
    # node so the cute epilogue chain analyzer can fuse it; the inductor
    # default decomp expands the polynomial form and breaks the chain.
    install_gelu_decomp(decomp_table)
    return decomp_table


def _make_fx(fn: Callable[..., object], *args: object) -> torch.fx.Graph:
    """
    We monkey patch get_proxy_slot to support Tensor/SymInt/SymFloat/SymBool in the
    graph without any origin for them.  We instead insert _host_tensor(), _get_symnode()
    in the graph to originate them.
    """

    def _get_proxy_slot(
        obj: object,
        tracer: proxy_tensor.PythonKeyTracer,
        default: object = proxy_tensor.no_default,
        transform: Callable[[object], object] = lambda x: x,
    ) -> object:
        if isinstance(obj, torch.Tensor) and not isinstance(obj, Tile):
            tracker = tracer.tensor_tracker
            if obj not in tracker:
                host_function = HostFunction.current()
                origin = host_function.tensor_to_origin.get(obj)
                if origin is not None:
                    assert origin.is_host()
                    # pyrefly: ignore [unsupported-operation]
                    tracker[obj] = proxy = tracer.create_proxy(
                        "call_function",
                        _tracing_ops._host_tensor,
                        (origin.host_str(),),
                        {},
                        name=origin.suggest_var_name(),
                    )
                    proxy.node.meta["val"] = obj
                    proxy.node.meta["lowering"] = APIFuncLowering(
                        _tracing_ops._host_tensor
                    )
                elif obj.numel() == 1 and not isinstance(
                    obj, torch._subclasses.FakeTensor
                ):
                    # Handle constant scalar tensors created inside the kernel
                    # (e.g., torch.tensor(val, dtype=...))
                    # These are real tensors (not FakeTensors) that contain constant values
                    from torch.utils._python_dispatch import _disable_current_modes

                    # Need to exit dispatch modes temporarily to access the real tensor value
                    with _disable_current_modes():
                        value = obj.detach().cpu().item()
                    # pyrefly: ignore [unsupported-operation]
                    tracker[obj] = proxy = tracer.create_proxy(
                        "call_function",
                        _tracing_ops._constant_tensor,
                        (value, obj.dtype),
                        {},
                        name="constant",
                    )
                    proxy.node.meta["val"] = obj
                    proxy.node.meta["lowering"] = APIFuncLowering(
                        _tracing_ops._constant_tensor
                    )
                else:
                    raise KeyError(
                        f"Tensor {obj} not found in tensor_to_origin and is not a scalar constant"
                    )
            return transform(tracker[obj])
        if isinstance(obj, proxy_tensor.py_sym_types):
            tracker = tracer.symnode_tracker
            if obj not in tracker:
                debug_name = CompileEnvironment.current().sympy_debug(obj._sympy_())
                # pyrefly: ignore [unsupported-operation]
                tracker[obj] = proxy = tracer.create_proxy(
                    "call_function",
                    _tracing_ops._get_symnode,
                    (debug_name,),
                    {},
                    name=debug_name if debug_name.isidentifier() else "symnode",
                )
                proxy.node.meta["val"] = obj
                proxy.node.meta["lowering"] = APIFuncLowering(_tracing_ops._get_symnode)
                # pyrefly: ignore [missing-attribute]
                proxy.force = lambda: proxy
            return transform(tracker[obj])
        return get_proxy_slot(obj, tracer, default, transform)

    get_proxy_slot: Callable[..., object] = proxy_tensor.get_proxy_slot

    with (
        preserve_node_meta(),
        patch.object(proxy_tensor, "get_proxy_slot", _get_proxy_slot),
        patch.object(
            torch.fx.proxy,
            "_COPY_META_FIELDS",
            [*torch.fx.proxy._COPY_META_FIELDS, "location"],
        ),
        patch.object(torch, "matmul", torch_matmul_replacement),
        patch.object(
            torch.Tensor,
            "matmul",
            tensor_matmul_replacement,
        ),
    ):
        current_location().set_fx_location()
        return proxy_tensor.make_fx(fn, decomposition_table=_get_custom_decomp_table())(
            *args
        ).graph


@dataclasses.dataclass
class GraphInfo:
    graph_id: int
    graph: torch.fx.Graph

    @property
    def name(self) -> str:
        raise NotImplementedError

    def kwargs(self) -> dict[str, object]:
        """Return a dictionary of keyword needed to copy this graph."""
        return {}

    def __str__(self) -> str:
        output = (
            _LazyGraphModule({}, self.graph).print_readable(print_output=False).strip()
        )
        return textwrap.dedent(
            re.sub(
                r"forward\(self,? ?([^)]*)\)",
                rf"{self.name}(\1)",
                # remove `class <lambda>():` from the output
                re.sub("^[^\n]+\n", "", output),
            )
        )

    def copy(self) -> GraphInfo:
        """Deep-copy the graph using node_copy, preserving metadata."""
        new_graph = torch.fx.Graph()
        node_map: dict[torch.fx.Node, torch.fx.Node] = {}
        for node in self.graph.nodes:
            new_node = new_graph.node_copy(node, lambda n: node_map[n])
            node_map[node] = new_node
        return type(self)(graph_id=self.graph_id, graph=new_graph, **self.kwargs())

    def codegen(self, state: CodegenState) -> list[object]:
        raise NotImplementedError


@dataclasses.dataclass
class RootGraphInfo(GraphInfo):
    phase_index: int = 0
    cute_grid_execution_plans: tuple[CuTeGridExecutionPlan, ...] = ()

    def kwargs(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(type(self))
            if field.name not in {"graph_id", "graph"}
        }

    @property
    def name(self) -> str:
        return f"root_graph_{self.graph_id}"


@dataclasses.dataclass
class NodeArgsGraphInfo(GraphInfo):
    """Common base class for graphs that have arguments from another graph."""

    node_args: list[torch.fx.Node]

    def placeholder_to_outer_arg(self, node: torch.fx.Node) -> torch.fx.Node:
        assert node.op == "placeholder"
        for placeholder, outer_node in zip(
            node.graph.find_nodes(op="placeholder"),
            self.node_args,
            strict=True,
        ):
            if placeholder is node:
                return outer_node
        raise KeyError("Placeholder not found in node_args")

    def kwargs(self) -> dict[str, object]:
        # TODO(jansel): do we need to map these to the new graph in the case of a copy?
        return {
            "node_args": [*self.node_args],
        }


@dataclasses.dataclass
class ForLoopGraphInfo(NodeArgsGraphInfo):
    block_ids: list[int]
    # Host AST read/write names for this device loop body (siblings only; see
    # ``_ReadWriteVisitor.visit_For`` in ast_read_writes.py).  Used to insert
    # ``tl.debug_barrier()`` between loops when there is a global RAW dep.
    host_loop_reads: frozenset[str] = dataclasses.field(default_factory=frozenset)
    host_loop_writes: frozenset[str] = dataclasses.field(default_factory=frozenset)
    # Precomputed by GenerateAST._compute_inter_loop_barriers: True iff a
    # tl.debug_barrier() must be emitted immediately before this for-loop's
    # outer prefix to make global writes from the previous sibling for-loop
    # in this scope visible.  Not copied across graph copies; recomputed.
    needs_barrier_before: bool = False

    @property
    def name(self) -> str:
        return f"for_loop_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            **super().kwargs(),
            "block_ids": [*self.block_ids],
            "host_loop_reads": self.host_loop_reads,
            "host_loop_writes": self.host_loop_writes,
            # ``needs_barrier_before`` is excluded -- recomputed by GenerateAST
            # per codegen run.
        }

    def codegen(self, state: CodegenState) -> list[object]:
        args = state.ast_args[3]
        assert isinstance(args, list)
        assert all(isinstance(x, ast.AST) for x in args)
        device_loop = state.device_function.tile_strategy.codegen_device_loop(
            state, self.block_ids
        )
        if state.fx_node is not None:
            from .tile_dependency import TILE_DEPENDENCY_SITE_ID_ATTR
            from .tile_dependency import TILE_DEPENDENCY_SITE_IDS_META

            site_ids = dict(state.fx_node.meta.get(TILE_DEPENDENCY_SITE_IDS_META, ()))
            if (site_id := site_ids.get(0)) is not None:
                setattr(device_loop.for_node, TILE_DEPENDENCY_SITE_ID_ATTR, site_id)
        # Make the active graph reachable by the strategy so it can pick
        # different lane-loop shapes for the reduce vs consume sweeps.
        # pyrefly: ignore [missing-attribute]
        state.codegen._cute_active_graph_info = self
        try:
            with state.codegen.add_device_loop(
                device_loop,
                needs_barrier_before=self.needs_barrier_before,
            ):
                return codegen_call_with_graph(
                    state.codegen,
                    self.graph,
                    args,
                )
        finally:
            # pyrefly: ignore [missing-attribute]
            state.codegen._cute_active_graph_info = None


def control_flow_parent_entries(
    graphs: Sequence[GraphInfo],
) -> dict[int, tuple[torch.fx.Node, int]]:
    """Map child graph IDs to their parent call and captured-argument slot."""
    result: dict[int, tuple[torch.fx.Node, int]] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.op != "call_function" or not node.args:
                continue
            if _tracing_ops.is_for_loop_target(node.target) and isinstance(
                node.args[0], int
            ):
                result[node.args[0]] = (node, 3)
            elif node.target is _tracing_ops._if and len(node.args) >= 5:
                if isinstance(node.args[1], int):
                    result[node.args[1]] = (node, 3)
                if isinstance(node.args[2], int):
                    result[node.args[2]] = (node, 4)
    return result


def device_loop_bounds(
    info: GraphInfo,
    parents: Mapping[int, torch.fx.Node],
    block_id: int,
) -> tuple[object, object] | None:
    """Return one device loop's traced begin/end values for ``block_id``."""
    parent = parents.get(info.graph_id)
    if (
        not isinstance(info, ForLoopGraphInfo)
        or parent is None
        or block_id not in info.block_ids
    ):
        return None
    begins, ends = parent.args[1:3]
    if not isinstance(begins, (list, tuple)) or not isinstance(ends, (list, tuple)):
        return None
    position = info.block_ids.index(block_id)

    def value(item: object) -> object:
        return item.meta.get("val") if isinstance(item, torch.fx.Node) else item

    return value(begins[position]), value(ends[position])


class ReductionLoopGraphInfo(ForLoopGraphInfo):
    @property
    def name(self) -> str:
        return f"reduction_loop_{self.graph_id}"


@dataclasses.dataclass
class IfGraphInfo(NodeArgsGraphInfo):
    predicate_is_tensor: bool = False
    else_branch: ElseGraphInfo | None = None

    if_arg_names: list[str] | None = None
    else_arg_names: list[str] | None = None

    # list of outputs of the branches,
    # [(if_out_0, else_out_0), (if_out_1, else_out_1), ...]
    # where each output is represented either as an index into the graph output,
    # or as a name of a non-local variable that is written to
    branches_outputs: list[tuple[int | str, ...]] | None = None

    @property
    def name(self) -> str:
        return f"if_graph_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            **super().kwargs(),
            "predicate_is_tensor": self.predicate_is_tensor,
            "else_branch": self.else_branch,
            "if_arg_names": self.if_arg_names,
            "else_arg_names": self.else_arg_names,
            "branches_outputs": self.branches_outputs,
        }

    def get_branches_return_names(
        self, state: CodegenState, if_outputs: list[object], else_outputs: list[object]
    ) -> tuple[list[str], list[str]]:
        if_args = state.ast_args[3]
        assert isinstance(if_args, list)
        assert all(isinstance(x, ast.AST) for x in if_args)
        else_args = state.ast_args[4]
        assert isinstance(else_args, list)
        assert all(isinstance(x, ast.AST) for x in else_args)

        assert self.if_arg_names is not None
        assert self.else_arg_names is not None
        assert self.branches_outputs is not None

        arg_node_name_to_ast_name = {
            self.if_arg_names[i]: if_args[i].id for i in range(len(if_args))
        } | {self.else_arg_names[i]: else_args[i].id for i in range(len(else_args))}

        if_return_names = [
            cast("ast.Name", if_outputs[o]).id
            if isinstance(o, int)
            else arg_node_name_to_ast_name[o]
            for (o, _) in self.branches_outputs
        ]
        else_return_names = [
            cast("ast.Name", else_outputs[o]).id
            if isinstance(o, int)
            else arg_node_name_to_ast_name[o]
            for (_, o) in self.branches_outputs
        ]
        return if_return_names, else_return_names

    def codegen(self, state: CodegenState) -> list[object]:
        from .generate_ast import GenerateAST

        if_args = state.ast_args[3]
        assert isinstance(if_args, list)
        assert all(isinstance(x, ast.AST) for x in if_args)
        else_args = state.ast_args[4]
        assert isinstance(else_args, list)
        assert all(isinstance(x, ast.AST) for x in else_args)

        assert isinstance(state.codegen, GenerateAST)

        test = state.ast_arg(0)
        # A branch condition that depends only on block sizes is a compile-time
        # constant for this config, even though it stayed symbolic during the
        # single frontend pass.  evaluate_constexpr_condition resolves it to a
        # concrete bool, or None if it is a genuine runtime condition.  When it is
        # constant, emit it as a literal ``True``/``False`` so Triton drops the
        # dead branch entirely and only the live branch is compiled; when it is
        # None, leave it as a runtime condition.
        constexpr_test = state.device_function.evaluate_constexpr_condition(
            state.proxy_arg(0)
        )
        if constexpr_test is not None:
            test = expr_from_string(repr(constexpr_test))

        body_stmts: list[ast.AST] = []
        orelse_stmts: list[ast.AST] = []
        if_ast_node = create(ast.If, test=test, body=body_stmts, orelse=orelse_stmts)
        state.add_statement(if_ast_node)

        with state.codegen.set_statements(body_stmts):
            if_outputs = codegen_call_with_graph(state.codegen, self.graph, if_args)

        else_outputs = []
        if self.else_branch is not None:
            else_graph = state.get_graph(self.else_branch)
            assert isinstance(else_graph, ElseGraphInfo)
            with state.codegen.set_statements(orelse_stmts):
                else_outputs = codegen_call_with_graph(
                    state.codegen, else_graph.graph, else_args
                )

        if len(body_stmts) == 0:
            body_stmts.append(ast.Pass())
        if len(orelse_stmts) == 0:
            orelse_stmts.append(ast.Pass())

        graph_info = state.get_graph(state.proxy_arg(1))
        assert isinstance(graph_info, IfGraphInfo)

        if_return_names, else_return_names = graph_info.get_branches_return_names(
            state, if_outputs, else_outputs
        )

        return cast(
            "list[object]",
            [expr_from_string(n) for n in if_return_names]
            + [expr_from_string(n) for n in else_return_names],
        )


@dataclasses.dataclass
class ElseGraphInfo(NodeArgsGraphInfo):
    @property
    def name(self) -> str:
        return f"else_graph_{self.graph_id}"

    def codegen(self, state: CodegenState) -> list[object]:
        raise exc.InternalError(
            RuntimeError("ElseGraphInfo should not be codegenned directly")
        )


@dataclasses.dataclass
class WhileConditionGraphInfo(NodeArgsGraphInfo):
    @property
    def name(self) -> str:
        return f"while_condition_{self.graph_id}"

    def codegen(self, state: CodegenState) -> list[object]:
        raise exc.InternalError(
            RuntimeError("WhileConditionGraphInfo should not be codegenned directly")
        )


@dataclasses.dataclass
class WhileLoopGraphInfo(NodeArgsGraphInfo):
    cond_graph_id: int

    @property
    def name(self) -> str:
        return f"while_loop_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            **super().kwargs(),
            "cond_graph_id": self.cond_graph_id,
        }

    def codegen(self, state: CodegenState) -> list[object]:
        cond_info = state.get_graph(self.cond_graph_id)

        args = state.ast_args[2]
        assert isinstance(args, list)
        assert all(isinstance(x, ast.AST) for x in args)

        def emit_condition(
            target_statements: list[ast.AST],
            cond_args: list[ast.AST] | None = None,
        ) -> ast.expr:
            with state.codegen.set_statements(target_statements):
                cond_outputs = codegen_call_with_graph(
                    state.codegen,
                    cond_info.graph,
                    # pyrefly: ignore [bad-argument-type]
                    cond_args or args,
                    copy_named_args=False,
                )
            if len(cond_outputs) != 1:
                raise exc.InternalError(
                    RuntimeError("While loop condition must produce a single value")
                )
            cond_output = cond_outputs[0]
            if isinstance(cond_output, ast.expr):
                return cond_output
            if isinstance(cond_output, ast.AST):
                return cast("ast.expr", cond_output)
            if isinstance(cond_output, (bool, int, float)):
                return cast("ast.expr", expr_from_string(repr(cond_output)))
            raise exc.InternalError(
                RuntimeError(
                    f"While loop condition produced unsupported value: {cond_output!r}"
                )
            )

        condition_statements: list[ast.AST] = []
        cond_expr = emit_condition(condition_statements)
        cond_var = state.device_function.new_var("while_cond")
        for stmt in condition_statements:
            state.codegen.add_statement(stmt)
        state.codegen.add_statement(
            create(
                ast.Assign,
                targets=[create(ast.Name, id=cond_var, ctx=ast.Store())],
                value=cond_expr,
            )
        )

        body_statements: list[ast.AST] = []
        with state.codegen.set_statements(body_statements):
            outputs = codegen_call_with_graph(
                state.codegen,
                self.graph,
                args,
                copy_named_args=False,
            )
        loop_condition_update: list[ast.AST] = []
        cond_expr_loop = emit_condition(loop_condition_update)
        body_statements.extend(loop_condition_update)
        body_statements.append(
            create(
                ast.Assign,
                targets=[create(ast.Name, id=cond_var, ctx=ast.Store())],
                value=cond_expr_loop,
            )
        )

        state.codegen.add_statement(
            create(
                ast.While,
                test=create(ast.Name, id=cond_var, ctx=ast.Load()),
                body=body_statements,
                orelse=[],
            )
        )
        return outputs


class RolledReductionInfo(NamedTuple):
    rolled_block_ids: list[int]
    original_graph_id: int
    used_rdim: bool
    can_be_rolled_by_caller: bool


@dataclasses.dataclass
class KernelPhase:
    roots: list[int]  # store root indices
    root_nodes: list[ast.For]


def _tensor_to_inter_loop_rw_name(host: HostFunction, t: torch.Tensor) -> str | None:
    o = host.tensor_to_origin.get(t)
    if o is None:
        return None
    return o.root_rw_name()


def _fx_trace_tensor_arg_rw_names(
    host: HostFunction, arg: object, seen: set[int] | None = None
) -> list[str]:
    """Map a load/store tensor FX arg to the list of host variable names it
    aliases.  Returns an empty list when the arg cannot be resolved to any
    host name (e.g. a purely device-internal temporary)."""
    from ..language import _tracing_ops

    if seen is None:
        seen = set()
    if isinstance(arg, tuple):
        out: list[str] = []
        for a in arg:
            out.extend(_fx_trace_tensor_arg_rw_names(host, a, seen))
        return out
    if not isinstance(arg, torch.fx.Node):
        return []
    nid = id(arg)
    if nid in seen:
        return []
    seen.add(nid)
    val = arg.meta.get("val")
    if isinstance(val, torch.Tensor):
        n = _tensor_to_inter_loop_rw_name(host, val)
        if n is not None:
            return [n]
    if arg.op == "call_function" and arg.target is _tracing_ops._host_tensor:
        val = arg.meta.get("val")
        if isinstance(val, torch.Tensor):
            n = _tensor_to_inter_loop_rw_name(host, val)
            if n is not None:
                return [n]
        return []
    out2: list[str] = []
    for a in arg.args:
        out2.extend(_fx_trace_tensor_arg_rw_names(host, a, seen))
    return out2


def _reduction_fx_inter_loop_rw_names(
    graph: torch.fx.Graph,
    host: HostFunction,
) -> tuple[frozenset[str], frozenset[str]]:
    """Infer host buffer names read/written in a rolled reduction FX subgraph.

    Resolves every hl.load / hl.store / atomic_* tensor arg back to host-named buffers.
    Args not resolving to a host name are device-internal temporaries (no cross-wavefront
    coherence) and are excluded.
    """
    from ..language import atomic_add
    from ..language import atomic_and
    from ..language import atomic_cas
    from ..language import atomic_max
    from ..language import atomic_min
    from ..language import atomic_or
    from ..language import atomic_xchg
    from ..language import atomic_xor
    from ..language import memory_ops

    atomic_funcs = frozenset(
        {
            atomic_add,
            atomic_and,
            atomic_cas,
            atomic_max,
            atomic_min,
            atomic_or,
            atomic_xchg,
            atomic_xor,
        }
    )
    reads: set[str] = set()
    writes: set[str] = set()

    for node in graph.find_nodes(
        op="call_function", target=memory_ops.load, sort=False
    ):
        reads.update(_fx_trace_tensor_arg_rw_names(host, node.args[0]))

    for node in graph.find_nodes(
        op="call_function", target=memory_ops.store, sort=False
    ):
        writes.update(_fx_trace_tensor_arg_rw_names(host, node.args[0]))

    for atomic_target in atomic_funcs:
        for node in graph.find_nodes(
            op="call_function", target=atomic_target, sort=False
        ):
            nms = _fx_trace_tensor_arg_rw_names(host, node.args[0])
            reads.update(nms)
            writes.update(nms)

    return frozenset(reads), frozenset(writes)


class DeviceIR:
    def __init__(self) -> None:
        super().__init__()
        self.graphs: list[GraphInfo] = []
        self.root_ids: list[int] = []
        self.rolled_reductions: list[RolledReductionInfo] = []
        self.phases: list[KernelPhase] = []
        self.implicit_dependency_starts: frozenset[int] = frozenset()
        self.tile_dependency_graph: TileDependencyGraph | None = None
        self.task_families: list[TaskFamily] = []
        self.grid_block_ids: list[list[int]] = []
        self.noncanonical_task_origin_block_ids: set[int] = set()
        # Owning HostFunction (captured in ``lower_to_device_ir``).
        self.host_function: HostFunction | None = None

    def __str__(self) -> str:
        return "\n\n".join(map(str, self.graphs))

    def debug_str(self) -> str:
        result = str(self)
        # Normalize indentation to 4 spaces to handle both PyTorch 2.9 and nightly formatting
        return re.sub(r" *(# File:\s+).*/([^/:]+:\d+)", r"    \1.../\2", result)

    def semantic_debug_str(self) -> str:
        """Render graph operations without layout-dependent FX metadata."""
        return "\n\n".join(
            f"{type(graph_info).__name__} {graph_info.name}\n"
            f"{graph_info.graph.python_code('self').src}"
            for graph_info in self.graphs
        )

    def add_graph(
        self,
        graph: torch.fx.Graph,
        graph_info_cls: type[GraphInfo] = GraphInfo,
        **kwargs: object,
    ) -> int:
        graph.eliminate_dead_code()
        graph_id = len(self.graphs)
        self.graphs.append(graph_info_cls(graph_id=graph_id, graph=graph, **kwargs))
        return graph_id

    def add_reduction_loop_graph(
        self,
        graph: torch.fx.Graph,
        block_index: int,
        node_args: list[torch.fx.Node],
    ) -> int:
        reads, writes = _reduction_fx_inter_loop_rw_names(graph, HostFunction.current())
        return self.add_graph(
            graph,
            graph_info_cls=ReductionLoopGraphInfo,
            block_ids=[block_index],
            node_args=node_args,
            host_loop_reads=reads,
            host_loop_writes=writes,
        )

    def add_root_graph(self, graph: torch.fx.Graph) -> None:
        self.root_ids.append(self.add_graph(graph, graph_info_cls=RootGraphInfo))

    def phase_for_root(self, root_id: int) -> int:
        graph_info = self.graphs[self.root_ids[root_id]]
        assert isinstance(graph_info, RootGraphInfo)
        return graph_info.phase_index

    @staticmethod
    def branch_paths_mutually_exclusive(
        path_a: list[tuple[int, int]] | None,
        path_b: list[tuple[int, int]] | None,
    ) -> bool:
        """True when two control-flow branch paths can never both execute.

        Paths are mutually exclusive when some dynamic ``_if`` appears in both
        with different branch sides (one took the ``if`` body, the other the
        ``else`` body). Each entry is ``(if_node_key, side)``.
        """
        if path_a is None or path_b is None:
            return False
        sides_b = dict(path_b)
        return any(
            node_id in sides_b and sides_b[node_id] != side for node_id, side in path_a
        )

    def reduction_block_id_branch_paths(
        self,
    ) -> dict[int, list[list[tuple[int, int]]]]:
        """Map each reduction block id to the control-flow branch path(s) it runs in.

        Walks the graph tree from each root, tracking ``(if_graph_id, side)``
        decisions (side 0 = ``if`` body, 1 = ``else`` body) made by dynamic
        ``_if`` nodes. For every reduction node it records the branch path of
        the enclosing graph, keyed by the reduction's ``block_index``.

        Two reductions whose paths diverge at a common ``_if`` (one took the
        ``if`` body, the other the ``else`` body) are mutually exclusive in time
        and may therefore share a CUDA thread axis. Returns ``{}`` when no
        reduction lives under a dynamic branch (the common, non-branching case),
        so callers leave the default thread-axis assignment untouched.
        """
        from .inductor_lowering import ReductionLowering

        result: dict[int, list[list[tuple[int, int]]]] = {}

        def walk(graph_id: int, path: list[tuple[int, int]]) -> None:
            if not 0 <= graph_id < len(self.graphs):
                return
            graph = self.graphs[graph_id].graph
            for node in graph.nodes:
                if node.op != "call_function":
                    continue
                lowering = node.meta.get("lowering")
                if isinstance(lowering, ReductionLowering) and isinstance(
                    lowering.block_index, int
                ):
                    result.setdefault(lowering.block_index, [])
                    if path not in result[lowering.block_index]:
                        result[lowering.block_index].append(list(path))
                if node.target is _tracing_ops._if and len(node.args) >= 3:
                    _, if_graph_id, else_graph_id, *_rest = node.args
                    if_node_key = id(node)
                    if isinstance(if_graph_id, int):
                        walk(if_graph_id, [*path, (if_node_key, 0)])
                    if isinstance(else_graph_id, int):
                        walk(else_graph_id, [*path, (if_node_key, 1)])
                elif (
                    _tracing_ops.is_for_loop_target(node.target)
                    and node.args
                    and isinstance(node.args[0], int)
                ):
                    # For/reduction loops do not introduce mutual exclusivity;
                    # descend without extending the path.
                    walk(node.args[0], path)

        for root_id in self.root_ids:
            walk(root_id, [])
        return result

    def register_rollable_reductions(self) -> None:
        """Analyze graphs for rollable reductions and register ReductionLoopSpec entries.

        This is analysis-only: it runs the roller to determine which graphs can
        be rolled, records lightweight RolledReductionInfo entries, and registers
        config_spec entries for the autotuner. Subgraphs created by the roller
        (e.g. ReductionLoopGraphInfo) are kept so the analysis layer can account
        for their loads/stores in the indexing config.
        """
        env = CompileEnvironment.current()
        rdims = [bs for bs in env.block_sizes if bs.reduction]
        # Eager tile-slot registration (see _register_cute_tile_vec_slots).
        # A present reduction must keep its slot at index 0, so reduction
        # kernels register their tile slots after the reduction-loop pass below.
        if env.backend_name == "cute" and not rdims:
            self._register_cute_tile_vec_slots(env)
        if not rdims:
            return
        num_original_graphs = len(self.graphs)

        # First pass: run roller analysis for all reduction dims and
        # record which original graphs use each rdim.
        rdim_results = []
        for rdim in rdims:
            graph_to_info: dict[int, RolledReductionInfo] = {}
            allow_loop = False

            # Check if any graph contains matmul or dev_prts stacking with
            # rdim, or reduces over rdim along a non-tile (hl.arange) axis
            # that cannot be sliced inside the reduction loop.
            can_roll_graphs = True
            for graph_info in self.graphs[:num_original_graphs]:
                roller = ReductionRoller(self, rdim, {})
                if (
                    roller.has_matmul_with_rdim(graph_info.graph)
                    or roller.has_stack_tensor_with_rdim(graph_info.graph)
                    or roller.has_unrollable_reduction(graph_info.graph)
                ):
                    can_roll_graphs = False
                    break

            if not can_roll_graphs:
                rdim_results.append((rdim, False, set()))
                continue

            used_graphs: set[int] = set()
            all_graphs_processed = True
            for graph_id in range(num_original_graphs):
                graph_info = self.graphs[graph_id]
                assert graph_id == graph_info.graph_id
                roller = ReductionRoller(self, rdim, graph_to_info)
                try:
                    roller.process(graph_info.graph)
                except NotImplementedError:
                    all_graphs_processed = False
                    break
                reduction_info = RolledReductionInfo(
                    rolled_block_ids=[rdim.block_id],
                    original_graph_id=graph_id,
                    used_rdim=len(roller.graphs_added) > 0,
                    can_be_rolled_by_caller=roller.outer_count == 0
                    and len(roller.graphs_added) == 1,
                )
                allow_loop = allow_loop or reduction_info.used_rdim
                if reduction_info.used_rdim:
                    used_graphs.add(graph_id)
                self.rolled_reductions.append(reduction_info)
                graph_to_info[graph_id] = reduction_info
            if not all_graphs_processed:
                allow_loop = False
            rdim_results.append((rdim, allow_loop, used_graphs))

        # Second pass: register reduction loop specs, ensuring that each
        # original graph is only rolled for one reduction dim at a time.
        graphs_with_rolled_rdim: set[int] = set()
        for rdim, allow_loop, used_graphs in rdim_results:
            if not allow_loop:
                continue
            if used_graphs & graphs_with_rolled_rdim:
                continue
            if env.backend_name != "pallas":
                env.config_spec.reduction_loops.append(
                    ReductionLoopSpec(
                        block_id=rdim.block_id,
                        size_hint=rdim.size_hint(),
                    )
                )
                if env.backend_name == "cute":
                    env.config_spec.cute_vector_widths.append(
                        CuteVectorWidthSpec(
                            block_id=rdim.block_id,
                            size_hint=rdim.size_hint(),
                        )
                    )
            graphs_with_rolled_rdim |= used_graphs

        # Track which rdims appear as the reduction axis of an indexed
        # reduction (argmin/argmax). On CuTe these can only be combined
        # via cute.arch.warp_reduction (32 threads max), so the autotuner
        # must keep their persistent thread count and looped chunk size
        # within a single warp.
        if env.backend_name == "cute":
            indexed_blocks: set[int] = set()
            indexed_targets = {
                torch.ops.aten.argmin.default,
                torch.ops.aten.argmax.default,
            }
            for graph_info in self.graphs[:num_original_graphs]:
                for node in graph_info.graph.nodes:
                    if getattr(node, "target", None) not in indexed_targets:
                        continue
                    args = node.args or ()
                    if not args:
                        continue
                    val = getattr(args[0], "meta", {}).get("val")
                    if val is None:
                        continue
                    dim_arg = args[1] if len(args) >= 2 else -1
                    dim_indices = (
                        [int(cast("int", d)) for d in dim_arg]
                        if isinstance(dim_arg, list)
                        else [int(cast("int", dim_arg))]
                    )
                    for dim_idx in dim_indices:
                        if dim_idx < 0:
                            dim_idx += val.ndim
                        if 0 <= dim_idx < val.ndim:
                            reduce_dim = val.size(dim_idx)
                            block_id = env.resolve_block_id(reduce_dim)
                            if block_id is not None:
                                indexed_blocks.add(block_id)
            env.config_spec.cute_indexed_reduction_block_ids = indexed_blocks

        # Reduction-dim slots are registered above; add the remaining
        # non-reduction tile slots after them (keeps the rdim slot at index 0).
        if env.backend_name == "cute":
            self._register_cute_tile_vec_slots(env)

    def _register_cute_tile_vec_slots(self, env: CompileEnvironment) -> None:
        """Eagerly register cute_vector_widths slots for non-reduction tile blocks.

        Eager (not lazy in codegen) keeps the config-spec width fixed before the
        autotuner snapshots it; callers register reduction-dim slots first so
        those keep index 0.
        """
        already_registered = set(env.config_spec.cute_vector_widths.valid_block_ids())
        for tile_bs in env.block_sizes:
            if tile_bs.reduction or tile_bs.block_id in already_registered:
                continue
            # Non-static size (jagged / AutoSize) never forms a lane loop.
            if not isinstance(tile_bs.size, (int, torch.SymInt)):
                continue
            try:
                size_hint_val = int(tile_bs.size_hint())
            except (TypeError, ValueError, AttributeError, AssertionError):
                continue
            env.config_spec.cute_vector_widths.append(
                CuteVectorWidthSpec(
                    block_id=tile_bs.block_id,
                    size_hint=size_hint_val,
                )
            )

    def _categorize_reduction(
        self, block_id: int, grid_ids: set[int]
    ) -> ReductionCategory:
        """Classify one reduction occurrence into the §2.1 taxonomy from FAITHFUL structure:
        ``(block_size_source kind, on the grid?, cdiv==1?)``. The single categorization rule
        the kernel-fact builder keys every reduction descriptor on.

        - no static extent -> DECLINED (jagged).
        - on the grid + fully resident (cdiv==1) -> FULL_GRID; on the grid + partial -> GRID_TILE.
        - a tunable ``block_sizes`` entry, not on the grid -> USER_TILE.
        - otherwise (rolled ``reduction_loops`` OR materialized full-width) -> FULL_SLICE.
        """
        env = CompileEnvironment.current()
        info = env.block_sizes[block_id]
        if not isinstance(info.size, (int, torch.SymInt)):
            return ReductionCategory.DECLINED
        if block_id in grid_ids:
            if self._is_fully_resident_grid_axis(block_id):
                return ReductionCategory.FULL_GRID
            return ReductionCategory.GRID_TILE
        if block_id in env.config_spec.block_sizes.valid_block_ids():
            return ReductionCategory.USER_TILE
        # FULL_SLICE is reached by elimination, so guard the structural invariant it relies on:
        # a genuine full-slice reduction is one the compiler allocated a reduction dimension for,
        # either ROLLED into reduction_loops (``ReductionLoopBlockSizeSource``) or MATERIALIZED
        # full-width after the roller declined; a fully-resident specialized axis reduced over in
        # one program is a ``FixedBlockSizeSource`` (rms_norm_per_block_quant's group_size=128 in a
        # sequential graph -- on the grid only via a shared hl.tile, so absent from grid_ids here).
        # ANY OTHER source landing here (e.g. a plain ``LoopSpecBlockSizeSource`` that should have
        # been USER_TILE) means the kernel fell through for the WRONG reason and may be mis-sized.
        # Debug-level (this fires on real corpus cells -- rms_norm_per_block_quant -- so it is not
        # an anomaly, just an inspection hook), never crash.
        from .compile_environment import FixedBlockSizeSource as _Fixed
        from .compile_environment import ReductionLoopBlockSizeSource as _RedLoop

        if not isinstance(info.block_size_source, (_RedLoop, _Fixed)):
            log.debug(
                "reduction block_id %s classified FULL_SLICE by fallthrough but carries "
                "%s, not ReductionLoopBlockSizeSource or FixedBlockSizeSource -- the full-slice "
                "sizing assumption (a compiler-allocated reduction dimension) may not hold",
                block_id,
                type(info.block_size_source).__name__,
            )
        return ReductionCategory.FULL_SLICE

    def build_reduction_kernel_fact(
        self,
        memory_op_facts: list[MemoryOpFact],
        accumulator_facts: list[AccumulatorFact],
        analysis: DeviceIRAnalysis,
    ) -> None:
        """Build the categorizing ``ReductionKernelFact`` — the list of reduction descriptors +
        their ``graph_id`` co-residency groups + the non-reduction loops + the parallel grid axes.
        The reduction seed + the Stage-2 allocator consume this fact directly.
        """
        env = CompileEnvironment.current()
        spec = env.config_spec
        grid_ids = {b for bids in self.grid_block_ids for b in bids}

        occurrences = analysis.original_reductions()
        # carried-2D tiles: count of accumulators whose last dim is the rdim (per block_id,
        # kernel-wide -- an accumulator is carried across the whole inner loop, so it is not
        # graph-scoped). A count (not a bool) is load-bearing: the carried byte cap divides the
        # budget by it, so the descriptor must carry the multiplicity.
        carried_2d_by_bid: dict[int, int] = {}
        for a in accumulator_facts:
            if len(a.dim_block_ids) >= 2 and a.dim_block_ids[-1] is not None:
                carried_2d_by_bid[a.dim_block_ids[-1]] = (
                    carried_2d_by_bid.get(a.dim_block_ids[-1], 0) + 1
                )

        descriptors: list[ReductionDescriptor] = []
        for gid, bid in occurrences:
            category = self._categorize_reduction(bid, grid_ids)
            info = env.block_sizes[bid]
            size_hint = (
                info.size_hint() if isinstance(info.size, (int, torch.SymInt)) else 0
            )
            per = self._per_reduction_memory_fields(bid, gid, memory_op_facts)
            descriptors.append(
                ReductionDescriptor(
                    category=category,
                    block_id=bid,
                    graph_id=gid,
                    size_hint=size_hint,
                    itemsize=analysis.reduction_input_itemsize(bid),
                    input_load_itemsize=per["input_load_itemsize"],
                    carried_2d_count=carried_2d_by_bid.get(bid, 0),
                    row_reread=per["row_reread"],
                    reread_eviction_index=per["reread_eviction_index"],
                    num_load=per["num_load"],
                )
            )

        # Co-residency groups = original graph_id equivalence classes. The materialized-feature
        # footprint a group byte-caps against is not stored here — the Stage-2 allocator derives it
        # at the comparison site (each group can materialize different feature axes, so a kernel-wide
        # value would be wrong for a multi-group kernel).
        groups_by_gid: dict[int, list[int]] = {}
        for idx, d in enumerate(descriptors):
            groups_by_gid.setdefault(d.graph_id, []).append(idx)
        # The per-group resident tile set: each group's peak live tiles, attributed home +
        # driven-loop-bodies, max'd across If/Else. Consumed by the Stage-2 footprint (which sums
        # ∏(dims) per actual tile).
        #
        # Preserve the legacy best-effort fallback: incomplete graph metadata
        # yields an extent-based footprint rather than failing fact construction.
        # The focused reduction corpus keeps this liveness-derived field pinned.
        try:
            group_live = analysis.group_live_tiles(sorted(groups_by_gid))
        except (KeyError, AttributeError, TypeError):
            group_live = {}
        coresidency_groups = tuple(
            CoResidencyGroup(
                graph_id=gid,
                descriptor_indices=tuple(idxs),
                live_tiles=tuple(group_live.get(gid, [])),
            )
            for gid, idxs in sorted(groups_by_gid.items())
        )

        # non-reduction loops + parallel grid axes. The non-reduction loops are the union over the
        # sized reductions' apply/normalize candidates; the grid axes are grid block_ids with no
        # reduction sized over them.
        sized_bids = {
            d.block_id for d in descriptors if d.category in SIZED_REDUCTION_CATEGORIES
        }
        non_reduction_loops: set[int] = set()
        for bid in sized_bids:
            non_reduction_loops.update(
                self._non_reduction_loop_candidates(bid, grid_ids)
            )
        non_reduction_loops -= sized_bids
        grid_axis_block_ids = tuple(sorted(grid_ids - sized_bids))

        spec.reduction_kernel_fact = ReductionKernelFact(
            reductions=tuple(descriptors),
            coresidency_groups=coresidency_groups,
            non_reduction_loop_block_ids=tuple(sorted(non_reduction_loops)),
            grid_axis_block_ids=grid_axis_block_ids,
        )

    def _per_reduction_memory_fields(
        self,
        red_block_id: int,
        graph_id: int,
        memory_op_facts: list[MemoryOpFact],
    ) -> dict:
        """The memory-op-derived per-reduction fields, computed exactly as
        ``_assemble_reduction_fact`` does (so a descriptor is field-equal to the legacy fact),
        but SCOPED to this reduction's original graph for ``num_load`` (a co-resident pass loads
        in its own graph). ``row_reread`` / ``input_load_itemsize`` are axis-keyed and
        graph-agnostic (a re-read is global to the axis), matching the legacy computation.
        """
        num_load = sum(
            1 for f in memory_op_facts if f.kind == "load" and f.graph_id == graph_id
        )
        row_reread = False
        reread_eviction_index: int | None = None
        for f in memory_op_facts:
            if f.kind != "load":
                continue
            cnt = next((c for ax, c in f.reductions_fed if ax == red_block_id), 0)
            if cnt >= 2 or (cnt >= 1 and f.stores_fed):
                row_reread = True
                reread_eviction_index = f.eviction_index
                break
        fed_sizes = [
            f.dtype.itemsize
            for f in memory_op_facts
            if f.kind == "load"
            and f.dtype is not None
            and any(ax == red_block_id for ax, _ in f.reductions_fed)
        ]
        if fed_sizes:
            input_load_itemsize = min(fed_sizes)
        else:
            row_sizes = [
                f.dtype.itemsize
                for f in memory_op_facts
                if f.kind == "load"
                and f.dtype is not None
                and (
                    (
                        f.subscript_block_ids
                        and f.subscript_block_ids[-1] == red_block_id
                    )
                    or (f.indexed_block_ids and f.indexed_block_ids[-1] == red_block_id)
                )
            ]
            input_load_itemsize = min(row_sizes) if row_sizes else 0
        return {
            "num_load": num_load,
            "row_reread": row_reread,
            "reread_eviction_index": reread_eviction_index,
            "input_load_itemsize": input_load_itemsize,
        }

    def build_kernel_matmul_fact(self, analysis: DeviceIRAnalysis) -> None:
        """Install the contraction fact derived by the analysis layer."""
        env = CompileEnvironment.current()
        env.config_spec.kernel_matmul_fact = analysis.kernel_matmul_fact(
            env,
            env.config_spec,
        )

    def build_matmul_reduction_epilogue_facts(self) -> None:
        """Phase 4: compose a ``MatmulWithReductionEpilogueFact`` for a fused matmul +
        reduction-over-output-axis epilogue. Fires iff exactly one ``MatmulFact`` AND exactly
        one SIZED full-extent reduction descriptor in a singleton co-residency group (the
        epilogue reduction over the specialized output N); holds the matmul fact plus the
        N-extent the seed keys on. Pure-matmul kernels have no epilogue reduction and
        pure-reduction kernels no MatmulFact, so the composed fact fires ONLY on the fused
        family.

        CONSTRAINT (PROMPT §2.9 — must NOT inherit the relaxed reduction gate): the epilogue
        seed is tuned for ONE specific reduction shape (a single full-extent reduction over the
        specialized output N, no other reduction co-resident). Expressed in the Stage-1
        vocabulary: exactly ONE sized reduction, FULL_SLICE/FULL_GRID, in a singleton
        co-residency group. A MULTI-reduction matmul kernel must NOT compose this fact (its
        bare ``reduction.size_hint`` would no longer be unambiguously the epilogue's N).
        """
        env = CompileEnvironment.current()
        spec = env.config_spec

        if len(spec.matmul_facts) != 1:
            return
        # §2.9 guard (Stage-1 vocabulary): the kernel's SIZED reductions must be exactly ONE
        # full-extent reduction in a singleton co-residency group — a single epilogue reduction
        # over the specialized output N, no other reduction co-resident. A MULTI-reduction matmul
        # kernel must NOT compose this fact (its epilogue N would be ambiguous). This is the sole
        # reduction gate (the legacy ``len(reduction_facts)==1`` check it replaced was strictly
        # looser — this also rejects a co-resident second sized reduction).
        kf = spec.reduction_kernel_fact
        if kf is None:
            return
        sized = [d for d in kf.reductions if d.category in SIZED_REDUCTION_CATEGORIES]
        if len(sized) != 1 or sized[0].category not in FULL_EXTENT_CATEGORIES:
            return
        group = next(
            (
                g
                for g in kf.coresidency_groups
                if any(
                    kf.reductions[i].block_id == sized[0].block_id
                    for i in g.descriptor_indices
                )
            ),
            None,
        )
        if group is not None and len(group.descriptor_indices) != 1:
            return
        matmul = spec.matmul_facts[0]
        # N-extent = the epilogue reduction's extent (the specialized, hl.specialize'd output N).
        spec.matmul_reduction_epilogue_facts.append(
            MatmulWithReductionEpilogueFact(
                matmul=matmul,
                n_extent=sized[0].size_hint,
                m_block_id=matmul.m_block_id,
                k_block_id=matmul.k_block_id,
            )
        )

    def build_pointwise_facts(self, analysis: DeviceIRAnalysis) -> None:
        """Install the pure-elementwise fact derived by the analysis layer."""
        spec = CompileEnvironment.current().config_spec
        fact = analysis.pointwise_fact(spec)
        if fact is not None:
            spec.pointwise_facts.append(fact)

    def build_accumulator_facts(
        self,
        analysis: DeviceIRAnalysis,
    ) -> list[AccumulatorFact]:
        """Return loop-carried accumulator facts derived by the analysis layer."""
        return analysis.accumulator_facts(CompileEnvironment.current())

    def _is_fully_resident_grid_axis(self, block_id: int) -> bool:
        """True iff a grid axis is tiled at its FULL extent (``block_size == extent`` =>
        ``cdiv(extent, block) == 1`` => the whole axis is resident in each program). Such an axis
        is on ``grid_block_ids`` only because it shares a combined ``hl.tile`` (per_token_group's
        ``hl.specialize``'d ``group_size``); a reduction over it reduces the WHOLE extent in one
        program, so its ``size_hint`` is a faithful per-program reduction extent and it is worth
        seeding as a TILED reduction.

        This is a PERF/SIZING eligibility gate, NOT a correctness guard: the seed only proposes a
        starting config that the autotuner accuracy-gates, so it can neither make a correct kernel
        wrong nor fix a wrong one. Its job is (a) to ADMIT a fully-resident grid reduction so it
        FIRES and gets a sized seed (per_token_group; dropping this check un-fires it -> starved
        default), and (b) to DECLINE a PARTIAL grid axis (``cdiv > 1``) whose whole-axis
        ``size_hint`` is NOT what one program reduces -- feeding that lie to the footprint sizer
        would mis-size. (A kernel that reduces over a partial grid axis WITHOUT a cross-CTA combine
        is already miscompiled at every config including the default -- that is an upstream lowering
        bug, unrelated to and unaffected by this seed-time classification.)

        Reads the FIXED block size from ``env.block_sizes`` (keyed by all block_ids incl. grid);
        a non-static extent or non-fixed source (a tunable grid tile) is never fully resident.
        """
        from .compile_environment import FixedBlockSizeSource

        env = CompileEnvironment.current()
        info = env.block_sizes[block_id]
        if not isinstance(info.size, (int, torch.SymInt)):
            return False
        source = info.block_size_source
        if not isinstance(source, FixedBlockSizeSource):
            return False
        block = source.value
        if not isinstance(block, (int, torch.SymInt)):
            return False
        return bool(env.known_equal(block, info.size))

    def _non_reduction_loop_candidates(
        self, red_block_id: int, grid_ids: set[int]
    ) -> tuple[int, ...]:
        """Identify non-reduction loop tiles for ``red_block_id`` -- non-grid
        ``block_sizes`` loops that are NOT the reduction axis. Shared by the standard
        and user-tiled fact builders.

        Returns ``qualifying`` (block_sizes order): candidate loops spanning the
        reduction extent with a resolvable static size -- the seed widens these to
        ``next_pow2``. A non-qualifying candidate (extent unresolvable or != the
        reduction extent) is left out and floored by the caller.
        """
        try:
            red_info = CompileEnvironment.current().block_sizes[red_block_id]
        except (IndexError, KeyError):
            return ()
        if not isinstance(red_info.size, (int, torch.SymInt)):
            return ()
        red_size_hint = red_info.size_hint()

        env = CompileEnvironment.current()
        qualifying: list[int] = []
        for bid in env.config_spec.block_sizes.valid_block_ids():
            if bid in grid_ids or bid == red_block_id:
                continue
            try:
                info = env.block_sizes[bid]
            except (IndexError, KeyError):
                continue
            if not isinstance(info.size, (int, torch.SymInt)) or (
                info.size_hint() != red_size_hint
            ):
                continue
            qualifying.append(bid)
        return tuple(qualifying)

    def build_codegen_graphs(self, config: Config) -> list[GraphInfo]:
        """Build and return graph copies with reduction rolling and epilogue subtiling applied.

        Creates a temporary DeviceIR with copied graphs, applies reduction
        rolling and epilogue subtiling based on the config, and returns the
        resulting graphs. The original graphs are never modified.
        """

        temp = copy.copy(self)
        temp.graphs = [g.copy() for g in self.graphs]
        temp._apply_rolling(config)
        temp._apply_epilogue_subtiling(config)
        if CompileEnvironment.current().backend_name == "metal":
            from .metal.mpp_graph_transform import rewrite_mpp_graphs

            rewrite_mpp_graphs(temp)
        return temp.graphs

    def _apply_rolling(self, config: Config) -> None:
        """Apply reduction rolling on the graph copies."""
        env = CompileEnvironment.current()
        reduction_loops = config.reduction_loops

        enabled_reduction_blocks = [
            spec.block_id
            for spec in env.config_spec.reduction_loops
            if env.config_spec.reduction_loops.config_get(
                reduction_loops, spec.block_id, None
            )
            is not None
        ]
        if not enabled_reduction_blocks:
            return

        rdims_by_block = {bs.block_id: bs for bs in env.block_sizes if bs.reduction}
        num_original_graphs = len(self.graphs)

        for block_id in enabled_reduction_blocks:
            rdim = rdims_by_block.get(block_id)
            if rdim is None:
                continue

            # Build graph_to_info from rolled_reductions for this block_id
            graph_to_info: dict[int, RolledReductionInfo] = {}
            for info in self.rolled_reductions:
                if info.rolled_block_ids == [block_id]:
                    graph_to_info[info.original_graph_id] = info

            for graph_id in range(num_original_graphs):
                info = graph_to_info.get(graph_id)
                if info is None or not info.used_rdim:
                    continue
                graph_info = self.graphs[graph_id]
                roller = ReductionRoller(self, rdim, graph_to_info)
                new_graph = roller.process(graph_info.graph)
                new_graph_id = self.add_graph(
                    new_graph, type(graph_info), **graph_info.kwargs()
                )
                # Replace only the graph payload to preserve root metadata
                # (e.g., phase_index used for barrier phase splitting).
                graph_info.graph = self.graphs[new_graph_id].graph

    def _apply_epilogue_subtiling(self, config: Config) -> None:
        """Apply epilogue subtiling on the graph copies if enabled."""
        split_factor = config.epilogue_subtile
        if not split_factor:
            return

        from ..language import memory_ops
        from ..language.atomic_ops import ATOMIC_OPS
        from .epilogue_subtiling import apply_epilogue_subtiling

        env = CompileEnvironment.current()
        configured_block_sizes = {
            info.block_id: info.from_config_assert(config)
            for info in env.block_sizes
            if not info.reduction
        }
        descriptor_output_nodes_by_graph: dict[int, set[torch.fx.Node]] = {}
        memory_op_index = 0
        atomic_op_index = 0
        for graph_info in self.graphs:
            descriptor_output_nodes: set[torch.fx.Node] = set()
            for node in graph_info.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target is memory_ops.load:
                    memory_op_index += 1
                elif node.target is memory_ops.store:
                    if _indexing_uses_tensor_descriptor(
                        config.indexing,
                        memory_op_index,
                    ):
                        descriptor_output_nodes.add(node)
                    memory_op_index += 1
                elif node.target in ATOMIC_OPS:
                    if _indexing_uses_tensor_descriptor(
                        config.atomic_indexing,
                        atomic_op_index,
                    ):
                        descriptor_output_nodes.add(node)
                    atomic_op_index += 1
            if descriptor_output_nodes:
                descriptor_output_nodes_by_graph[graph_info.graph_id] = (
                    descriptor_output_nodes
                )

        for graph_info in self.graphs:
            # Epilogue output ops can live in nested/reduction/control-flow graphs,
            # not just roots.  The indexing configs are global across codegen_graphs:
            # this mirrors the existing load/store/atomic counters used when
            # registering indexing tunables and tensor-descriptor layout guards.
            apply_epilogue_subtiling(
                graph_info.graph,
                split_factor,
                configured_block_sizes,
                descriptor_output_nodes_by_graph.get(graph_info.graph_id, set()),
            )

    def __enter__(self) -> None:
        try:
            tls.device_irs.append(self)
        except AttributeError:
            tls.device_irs = [self]

    def __exit__(self, *args: object) -> None:
        tls.device_irs.pop()

    @staticmethod
    def current() -> DeviceIR:
        return tls.device_irs[-1]


def _is_static_slice_bound(bound: object) -> bool:
    """A compile-time-known slice bound: a plain int or a backed SymInt."""
    from .compile_environment import _has_unbacked

    if isinstance(bound, int):
        return True
    if isinstance(bound, torch.SymInt):
        return not _has_unbacked(bound._sympy_())
    return False


class WalkDeviceAST(NodeVisitor):
    def __init__(self, device_ir: DeviceIR) -> None:
        super().__init__()
        self.device_ir = device_ir
        self.scope: dict[str, object] = {}

    def generic_visit(self, node: ast.AST) -> None:
        raise exc.StatementNotSupported(type(node).__name__)

    def _assign(self, target: ast.AST, value: object) -> None:
        if isinstance(target, ast.Name):
            if isinstance(value, torch.Tensor):
                # rename the node to match the variable name
                mode = proxy_tensor.get_proxy_mode()
                assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
                tracer = mode.tracer
                slot = proxy_tensor.get_proxy_slot(value, tracer, default=None)
                if isinstance(slot, proxy_tensor._ProxyTensor):
                    node = slot.proxy.node
                    if target.id not in node.name:
                        node.name = node.graph._graph_namespace.create_name(
                            target.id, None
                        )
            self.scope[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, n in enumerate(target.elts):
                if isinstance(n, ast.Starred):
                    raise exc.StarredArgsNotSupportedOnDevice

                # pyrefly: ignore [bad-index]
                self._assign(n, value[i])
        elif isinstance(target, ast.Subscript):
            dst = self.visit(target.value)
            assert isinstance(value, torch.Tensor)
            assert isinstance(dst, torch.Tensor)
            hl.store(
                dst,
                self._subscript_slice_proxy(target.slice),
                value,
            )
        else:
            raise NotImplementedError(
                f"Unsupported target type {type(target).__name__}"
            )

    def _body(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            self.visit(stmt)

    def _static_scope(self) -> dict[str, object]:
        return {k: v for k, v in self.scope.items() if not self.should_become_arg(v)}

    def _lift_inputs(self, names: Iterable[str]) -> LiftTensorArgs:
        return LiftTensorArgs(
            {
                name: self.scope[name]
                for name in names
                if name in self.scope and self.should_become_arg(self.scope[name])
            }
        )

    def _collect_outputs(
        self,
        subgraph_scope: dict[str, object],
        writes: dict[str, int],
        include_new: bool = False,
    ) -> LiftTensorArgs:
        return LiftTensorArgs(
            {
                k: v
                for k, v in subgraph_scope.items()
                if k in writes
                and (include_new or k in self.scope)
                and self.scope.get(k) is not v
            }
        )

    @staticmethod
    def _rw_names(rw: ReadWrites) -> tuple[str, ...]:
        ordered = dict.fromkeys([*rw.reads.keys(), *rw.writes.keys()])
        return tuple(ordered)

    def _trace_graph(
        self,
        inputs: LiftTensorArgs,
        build_fn: Callable[[WalkDeviceAST], tuple[object, LiftTensorArgs]],
        *,
        graph_info_cls: type[NodeArgsGraphInfo],
        copy_tensor_args: bool = True,
        **graph_kwargs: object,
    ) -> tuple[int, LiftTensorArgs]:
        outputs_holder: LiftTensorArgs | None = None

        def runner(*args: object) -> object:
            nonlocal outputs_holder
            subgraph_walker = WalkDeviceAST(self.device_ir)
            subgraph_walker.scope.update(self._static_scope())
            subgraph_walker.scope.update(
                inputs.replace_tensor_args(args, copy_tensors=copy_tensor_args)
            )
            result, outputs_holder = build_fn(subgraph_walker)
            return result

        with self.disable_tracing() as tracer:
            graph = proxy_tensor.make_fx(
                runner, decomposition_table=_get_custom_decomp_table()
            )(*inputs.get_tensor_args()).graph
            graph_id = self.device_ir.add_graph(
                graph,
                graph_info_cls=graph_info_cls,
                node_args=inputs.get_node_args(tracer),
                **graph_kwargs,
            )
        assert outputs_holder is not None
        return graph_id, outputs_holder

    def visit_Pass(self, node: ast.Pass) -> None:
        return None

    def visit_BinOp(self, node: ast.BinOp) -> object:
        left = self.visit(node.left)
        right = self.visit(node.right)
        # Special handling for Tile + offset: expand to tile.index + offset
        # and mark with metadata for indexing strategies to recognize
        if (
            isinstance(node.op, ast.Add)
            and isinstance(left, Tile)
            and isinstance(right, (int, torch.SymInt))
        ):
            # Implicitly expand to tile.index + offset
            left = hl.tile_index(left)
        return _eval_binary(node.op, left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> object:
        return _eval_unary(node.op, self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> object:
        lhs = self.visit(node.left)
        results = []
        for op, rhs in zip(node.ops, node.comparators, strict=True):
            rhs = self.visit(rhs)
            results.append(result := _eval_compare(op, lhs, rhs))
            if not isinstance(result, _tracing_ops._symbolic_types) and not result:
                break
            lhs = rhs
        return functools.reduce(_tracing_ops._and, results)

    def visit_BoolOp(self, node: ast.BoolOp) -> object:
        if isinstance(node.op, ast.And):
            combine_op = _tracing_ops._and
            early_exit = operator.not_
        else:
            assert isinstance(node.op, ast.Or)
            combine_op = _tracing_ops._or
            early_exit = operator.truth
        results = []
        for value in node.values:
            results.append(result := self.visit(value))
            if not isinstance(result, _tracing_ops._symbolic_types) and early_exit(
                result
            ):
                break
        return functools.reduce(combine_op, results)

    @staticmethod
    @contextlib.contextmanager
    def disable_tracing() -> Iterator[proxy_tensor.PythonKeyTracer]:
        mode = proxy_tensor.get_proxy_mode()
        assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
        tracer = mode.tracer
        assert isinstance(tracer, proxy_tensor.PythonKeyTracer)
        with proxy_tensor.disable_proxy_modes_tracing():
            yield tracer

    @staticmethod
    def should_become_arg(value: object) -> bool:
        from ..language.distributed_ops import AsyncCopyDescriptor

        # Descriptor operations in a dynamic branch close over the descriptor
        # created by the surrounding graph. The lowering resolves that handle by
        # compiler ID, so it is a compile-time value rather than a branch input.
        if isinstance(value, AsyncCopyDescriptor):
            return False
        if isinstance(value, (Tile, int, float, bool, type(None), torch.SymInt)):
            return False
        if isinstance(value, torch.Tensor):
            if (
                origin := HostFunction.current().tensor_to_origin.get(value)
            ) is not None:
                return origin.is_device()
        return True

    def _extract_tile_range(
        self, for_node: ast.For, *, supports_step: bool
    ) -> tuple[object, object, object | None]:
        call_node = for_node.iter
        assert isinstance(call_node, ast.Call)
        func_node = call_node.func
        assert isinstance(func_node, ExtendedAST)
        func_type = func_node._type_info
        assert isinstance(func_type, CallableType)
        assert func_type.value in (hl.jagged_tile, hl.tile, hl.grid, builtins.range)
        args = call_node.args
        assert len(args) >= 1
        if len(args) == 1:
            begin = None
            end = self.visit(args[0])
            step = (
                next(
                    (
                        self.visit(keyword.value)
                        for keyword in call_node.keywords
                        if keyword.arg == "step"
                    ),
                    None,
                )
                if supports_step
                else None
            )
        else:
            begin = self.visit(args[0])
            end = self.visit(args[1])
            step = (
                self.visit(args[2])
                if supports_step and len(args) >= 3
                else next(
                    (
                        self.visit(keyword.value)
                        for keyword in call_node.keywords
                        if keyword.arg == "step"
                    ),
                    None,
                )
                if supports_step
                else None
            )
        return begin, end, step

    def _handle_sequence_unrolling(
        self,
        sequence_iter: ast.AST,
        target: ast.AST,
        element_processor: Callable[[], object | None],
        preserve_scope: bool = False,
    ) -> list[object]:
        """Common logic for unrolling sequences in both loops and comprehensions."""
        # Get the sequence of values to iterate over
        sequence_value = self.visit(sequence_iter)
        assert isinstance(sequence_value, (tuple, list)), (
            f"Expected tuple or list, got {type(sequence_value)}"
        )

        results = []
        for element_value in sequence_value:
            if preserve_scope:
                # For loops: don't create new scope, allow state to persist
                self._assign(target, element_value)
                result = element_processor()
                if result is not None:
                    results.append(result)
            else:
                # For comprehensions: create isolated scope for each iteration
                old_scope = self.scope.copy()
                try:
                    self._assign(target, element_value)
                    result = element_processor()
                    if result is not None:
                        results.append(result)
                finally:
                    self.scope = old_scope

        return results

    def _handle_tuple_unrolling(
        self,
        node: ast.For,
    ) -> None:
        """Handle unrolling of loops that iterate over tuples of tensors."""

        def execute_body() -> None:
            self._body(node.body)
            return None  # No result to collect for loops

        self._handle_sequence_unrolling(
            node.iter, node.target, execute_body, preserve_scope=True
        )

    def visit_For(self, node: ast.For) -> None:
        assert isinstance(node, ExtendedAST)
        assert not node.orelse
        assert isinstance(node.iter, ExtendedAST)
        iter_type = node.iter._type_info

        # Check if we're iterating directly over a sequence (tuple unrolling)
        if isinstance(iter_type, SequenceType):
            self._handle_tuple_unrolling(node)
            return

        # Special handling for variables that might contain sequences from list comprehensions
        if isinstance(node.iter, ast.Name) and node.iter.id in self.scope:
            scope_value = self.scope[node.iter.id]
            if isinstance(scope_value, (tuple, list)):
                # This is a sequence in the scope, we should try to unroll it
                # even if the type info doesn't indicate it's a SequenceType
                self._handle_tuple_unrolling(node)
                return

        if not isinstance(iter_type, IterType):
            raise exc.InvalidDeviceForLoop(iter_type)
        inner_type: TypeInfo = iter_type.inner
        if node._loop_type == LoopType.GRID:
            self._assign(node.target, inner_type.proxy())
            self._body(node.body)
        elif node._loop_type == LoopType.DEVICE:
            rw: ReadWrites = ReadWrites.from_ast(node)
            inputs = self._lift_inputs(self._rw_names(rw))
            supports_step = False
            if isinstance(inner_type, SequenceType):
                supports_step = all(
                    isinstance(value, GridIndexType) for value in inner_type.unpack()
                )
            else:
                supports_step = isinstance(inner_type, GridIndexType)
            begin, end, step = self._extract_tile_range(
                node, supports_step=supports_step
            )
            if isinstance(inner_type, SequenceType):
                iter_vars = inner_type.unpack()
                if begin is None:
                    begin = [0] * len(iter_vars)
                if step is None:
                    step = [None] * len(iter_vars)
            else:
                if isinstance(inner_type, JaggedTileIndexType):
                    # hl.jagged_tile takes an N-D parent tensor, not a scalar bound.
                    assert isinstance(end, torch.Tensor)
                    jagged_parent = end

                    # The first lifted loop input must be the jagged parent tensor.
                    # _setup_mask uses that parent tensor to recover each lane's true end.
                    assert inputs.flat_values[0] is jagged_parent

                    # Flatten so the global max becomes a single-axis reduction —
                    # Inductor only supports one reduction dim per buffer.
                    end = torch.amax(jagged_parent.reshape(-1))

                iter_vars = [inner_type]
                begin = [0] if begin is None else [begin]
                end = [end]
                step = [step]
            assert all(isinstance(x, (TileIndexType, GridIndexType)) for x in iter_vars)

            def build_subgraph(
                subgraph_walker: WalkDeviceAST,
            ) -> tuple[list[object], LiftTensorArgs]:
                subgraph_walker._assign(node.target, inner_type.proxy())
                subgraph_walker._body(node.body)
                loop_outputs = self._collect_outputs(subgraph_walker.scope, rw.writes)
                return loop_outputs.get_tensor_args(), loop_outputs

            block_ids: list[int] = []
            for var in iter_vars:
                assert isinstance(var, (TileIndexType, GridIndexType))
                block_ids.append(var.block_id)
            begin_values = begin if isinstance(begin, (list, tuple)) else [begin]
            step_values = (
                step if isinstance(step, (list, tuple)) else [step] * len(block_ids)
            )
            for block_id, begin_value, step_value in zip(
                block_ids, begin_values, step_values, strict=True
            ):
                if (
                    not isinstance(begin_value, int)
                    or begin_value != 0
                    or step_value not in (None, 1)
                ):
                    self.device_ir.noncanonical_task_origin_block_ids.add(block_id)

            host_reads, host_writes = rw.read_and_write_name_frozensets()
            graph_idx, outputs = self._trace_graph(
                inputs,
                build_subgraph,
                graph_info_cls=ForLoopGraphInfo,
                block_ids=block_ids,
                host_loop_reads=host_reads,
                host_loop_writes=host_writes,
            )
            step_list = step if isinstance(step, list) else None
            if step_list is None or all(s is None for s in step_list):
                args = (
                    graph_idx,
                    begin,
                    end,
                    inputs.get_tensor_args(),
                )
                loop_target = _tracing_ops._for_loop
            else:
                args = (
                    graph_idx,
                    begin,
                    end,
                    inputs.get_tensor_args(),
                    step_list,
                )
                loop_target = _tracing_ops._for_loop_step
            mode = proxy_tensor.get_proxy_mode()
            assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
            tracer = mode.tracer
            proxy_out = tracer.create_proxy(
                "call_function",
                loop_target,
                # pyrefly: ignore [bad-argument-type]
                *args_to_proxies(tracer, args),
            )
            proxy_tensor.track_tensor_tree(
                outputs.get_tensor_args(),
                proxy_out,
                constant=None,
                tracer=tracer,
            )
            for name, value in outputs.unflatten().items():
                if isinstance(value, Tile):
                    continue
                if name in self.scope:
                    try:
                        self.scope[name] = _tracing_ops._phi(self.scope[name], value)
                    except Exception as e:
                        raise exc.CantCombineTypesInControlFlow(
                            name, self.scope[name], value
                        ) from e
                else:
                    self.scope[name] = value
        else:
            raise AssertionError(f"Unexpected loop type {node._loop_type}")

    def visit_While(self, node: ast.While) -> None:
        if node.orelse:
            raise exc.StatementNotSupported("while ... else ...")

        test_rw = ReadWrites.from_ast(node.test)
        body_rw = ReadWrites.from_list(node.body)
        names = tuple(
            dict.fromkeys((*self._rw_names(test_rw), *self._rw_names(body_rw)))
        )

        inputs = self._lift_inputs(names)

        def build_condition(
            subgraph_walker: WalkDeviceAST,
        ) -> tuple[list[object], LiftTensorArgs]:
            result = subgraph_walker.visit(node.test)
            return [result], LiftTensorArgs({})

        cond_graph_id, _ = self._trace_graph(
            inputs,
            build_condition,
            graph_info_cls=WhileConditionGraphInfo,
            copy_tensor_args=False,
        )

        def build_body(
            subgraph_walker: WalkDeviceAST,
        ) -> tuple[list[object], LiftTensorArgs]:
            subgraph_walker._body(node.body)
            loop_outputs = self._collect_outputs(subgraph_walker.scope, body_rw.writes)
            return loop_outputs.get_tensor_args(), loop_outputs

        body_graph_id, outputs = self._trace_graph(
            inputs,
            build_body,
            graph_info_cls=WhileLoopGraphInfo,
            cond_graph_id=cond_graph_id,
            copy_tensor_args=False,
        )

        args = (
            cond_graph_id,
            body_graph_id,
            inputs.get_tensor_args(),
            None,
        )
        mode = proxy_tensor.get_proxy_mode()
        assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
        tracer = mode.tracer
        proxy_out = tracer.create_proxy(
            "call_function",
            _tracing_ops._while_loop,
            # pyrefly: ignore [bad-argument-type]
            *args_to_proxies(tracer, args),
        )
        proxy_tensor.track_tensor_tree(
            outputs.get_tensor_args(),
            proxy_out,
            constant=None,
            tracer=tracer,
        )

        for name, value in outputs.unflatten().items():
            if isinstance(value, Tile):
                continue
            if name in self.scope:
                try:
                    self.scope[name] = _tracing_ops._phi(self.scope[name], value)
                except Exception as e:
                    raise exc.CantCombineTypesInControlFlow(
                        name, self.scope[name], value
                    ) from e
            else:
                self.scope[name] = value

    def visit_If(self, node: ast.If) -> object:
        test_proxy = self.visit(node.test)
        if not isinstance(test_proxy, _tracing_ops._symbolic_types):
            body = node.body if test_proxy else node.orelse
            if body:
                self._body(body)
            return
        self._create_if_subgraph(test_proxy, node.body, node.orelse)

    def _create_if_subgraph(
        self,
        test_proxy: object,
        body: list[ast.stmt],
        orelse: list[ast.stmt],
    ) -> int:
        # Track whether the predicate is a tensor with numel > 1
        predicate_is_tensor = (
            isinstance(test_proxy, torch.Tensor) and math.prod(test_proxy.shape) > 1
        )

        if_branch_rw: ReadWrites = ReadWrites.from_list(body)
        else_branch_rw: ReadWrites = ReadWrites.from_list(orelse)

        if_branch_inputs = self._lift_inputs(self._rw_names(if_branch_rw))
        else_branch_inputs = self._lift_inputs(self._rw_names(else_branch_rw))

        def build_body(
            subgraph_walker: WalkDeviceAST,
            stmts: list[ast.stmt],
            rw: ReadWrites,
        ) -> tuple[list[object], LiftTensorArgs]:
            subgraph_walker._body(stmts)
            outputs_local = self._collect_outputs(
                subgraph_walker.scope, rw.writes, include_new=True
            )
            return outputs_local.get_tensor_args(), outputs_local

        else_graph_idx, else_outputs = self._trace_graph(
            else_branch_inputs,
            functools.partial(build_body, stmts=orelse, rw=else_branch_rw),
            graph_info_cls=ElseGraphInfo,
        )

        if_graph_idx, if_outputs = self._trace_graph(
            if_branch_inputs,
            functools.partial(build_body, stmts=body, rw=if_branch_rw),
            graph_info_cls=IfGraphInfo,
            predicate_is_tensor=predicate_is_tensor,
            else_branch=else_graph_idx,
        )
        if_graph = cast("IfGraphInfo", self.device_ir.graphs[if_graph_idx])

        def get_arg_values_and_names(
            inputs: LiftTensorArgs,
        ) -> tuple[list[object], list[str]]:
            input_tensor_arg_values = inputs.get_tensor_args()

            def is_tensor_arg_value(v: object) -> bool:
                return any(v is t for t in input_tensor_arg_values)

            input_tensor_node_names = [
                k for k, v in inputs.values.items() if is_tensor_arg_value(v)
            ]

            return input_tensor_arg_values, input_tensor_node_names

        if_arg_values, if_graph.if_arg_names = get_arg_values_and_names(
            if_branch_inputs
        )
        else_arg_values, if_graph.else_arg_names = get_arg_values_and_names(
            else_branch_inputs
        )

        args = (
            test_proxy,
            if_graph_idx,
            else_graph_idx,
            if_arg_values,
            else_arg_values,
        )
        mode = proxy_tensor.get_proxy_mode()
        assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
        tracer = mode.tracer
        proxy_out = tracer.create_proxy(
            "call_function",
            _tracing_ops._if,
            # pyrefly: ignore [bad-argument-type]
            *args_to_proxies(tracer, args),
        )

        if_output_values = if_outputs.values
        else_output_values = else_outputs.values

        common_output_names = [n for n in if_output_values if n in else_output_values]
        if_nonlocal_outputs_names = [
            name
            for name in if_output_values
            if name not in common_output_names and name in self.scope
        ]
        else_nonlocal_output_names = [
            name
            for name in else_output_values
            if name not in common_output_names and name in self.scope
        ]

        if_common_outputs = [if_output_values[name] for name in common_output_names]
        if_nonlocal_outputs = [
            if_output_values[name] for name in if_nonlocal_outputs_names
        ]
        if_unmodified_nonlocal_outputs = [
            self.scope[name] for name in else_nonlocal_output_names
        ]
        else_common_outputs = [else_output_values[name] for name in common_output_names]
        else_unmodified_nonlocal_outputs = [
            self.scope[name] for name in if_nonlocal_outputs_names
        ]
        else_nonlocal_outputs = [
            else_output_values[name] for name in else_nonlocal_output_names
        ]
        proxy_tensor.track_tensor_tree(
            if_common_outputs
            + if_nonlocal_outputs
            + if_unmodified_nonlocal_outputs
            + else_common_outputs
            + else_unmodified_nonlocal_outputs
            + else_nonlocal_outputs,
            proxy_out,
            constant=None,
            tracer=tracer,
        )

        # branches_outputs:  [(if_out_0, else_out_0), (if_out_1, else_out_1), ...]
        # where each output is either an index if the graph's output values,
        # or a name of a nonlocal variable which the opposite branch writes to.
        # Ordering: common -> if-only-nonlocal -> else-only-nonlocal,
        # (i.e. same as ordering of values in track_tensor_tree above)
        if_graph.branches_outputs = []

        def get_output_idx(name: str, output_values: dict[str, object]) -> int:
            return next(i for i, n in enumerate(output_values) if n == name)

        for name in common_output_names:
            if_value = if_output_values[name]
            else_value = else_output_values[name]
            self.scope[name] = _tracing_ops._phi(if_value, else_value)
            if_output_index = get_output_idx(name, if_output_values)
            else_output_index = get_output_idx(name, else_output_values)
            if_graph.branches_outputs.append((if_output_index, else_output_index))

        for name in if_nonlocal_outputs_names:
            self.scope[name] = _tracing_ops._phi(
                self.scope[name], if_output_values[name]
            )
            if_output_index = get_output_idx(name, if_output_values)
            if_graph.branches_outputs.append((if_output_index, name))

        for name in else_nonlocal_output_names:
            self.scope[name] = _tracing_ops._phi(
                self.scope[name], else_output_values[name]
            )
            else_output_index = get_output_idx(name, else_output_values)
            if_graph.branches_outputs.append((name, else_output_index))

        return if_graph_idx

    def visit_Name(self, node: ast.Name) -> object:
        if node.id in self.scope:
            return self.scope[node.id]
        assert isinstance(node, ExtendedAST)
        type_info = node._type_info
        assert type_info is not None
        if type_info.origin.is_device():
            raise exc.CrossRootDeviceValue(node.id)
        try:
            return type_info.proxy()
        except NotImplementedError:
            raise exc.CantReadOnDevice(type_info) from None

    def _subscript_slice_proxy(self, slice_node: ast.AST) -> list[object]:
        assert isinstance(slice_node, ExtendedAST)
        result = self.visit(slice_node)
        if isinstance(result, (list, tuple)):
            return [*result]
        return [result]

    def visit_Tuple(self, node: ast.Tuple) -> tuple[object, ...]:
        return tuple([self.visit(x) for x in node.elts])

    def visit_List(self, node: ast.List) -> list[object]:
        return [self.visit(x) for x in node.elts]

    def _visit_comprehension(
        self, node: ast.ListComp | ast.GeneratorExp, name: str
    ) -> tuple[object, ...]:
        """Handle list comprehension or generator expression unrolling."""
        assert isinstance(node, ExtendedAST)

        # Only handle simple cases with single generator and no if conditions
        if len(node.generators) != 1 or node.generators[0].ifs:
            raise exc.StatementNotSupported(f"Complex {name}s are not supported")

        generator = node.generators[0]
        assert isinstance(generator.iter, ExtendedAST)
        iter_type = generator.iter._type_info

        # Check if we're iterating over a sequence (similar to tuple unrolling)
        if isinstance(iter_type, SequenceType):
            return self._handle_comprehension_unrolling(node.elt, generator)

        # For non-sequence iterables, we could extend this later
        raise exc.StatementNotSupported(
            f"{name.capitalize()}s over non-sequence types are not supported"
        )

    def visit_ListComp(self, node: ast.ListComp) -> tuple[object, ...]:
        return self._visit_comprehension(node, "list comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> tuple[object, ...]:
        return self._visit_comprehension(node, "generator expression")

    def _handle_comprehension_unrolling(
        self, elt: ast.expr, generator: ast.comprehension
    ) -> tuple[object, ...]:
        """Handle unrolling of comprehensions (list comp or generator exp) over sequences."""

        def evaluate_expression() -> object:
            # Evaluate the comprehension expression
            result = self.visit(elt)
            # If the result is a SymInt that can be evaluated to a concrete value, do so
            if isinstance(result, torch.SymInt):
                try:
                    return int(result)
                except (ValueError, TypeError):
                    return result
            return result

        results = self._handle_sequence_unrolling(
            generator.iter, generator.target, evaluate_expression, preserve_scope=False
        )
        # Return as tuple to match the expected type for tuple unrolling
        return tuple(results)

    def visit_DictComp(self, node: ast.DictComp) -> dict[object, object]:
        """Handle dict comprehension unrolling."""
        assert isinstance(node, ExtendedAST)

        if len(node.generators) != 1 or node.generators[0].ifs:
            raise exc.StatementNotSupported(
                "Complex dict comprehensions are not supported"
            )

        generator = node.generators[0]
        assert isinstance(generator.iter, ExtendedAST)
        iter_type = generator.iter._type_info

        if not isinstance(iter_type, SequenceType):
            raise exc.StatementNotSupported(
                "Dict comprehensions over non-sequence types are not supported"
            )

        result: dict[object, object] = {}

        def evaluate_key_value() -> None:
            key = self.visit(node.key)
            value = self.visit(node.value)
            result[key] = value

        self._handle_sequence_unrolling(
            generator.iter, generator.target, evaluate_key_value, preserve_scope=False
        )
        return result

    def visit_Dict(self, node: ast.Dict) -> dict[object, object]:
        keys = [self.visit(key) if key is not None else None for key in node.keys]
        values = [self.visit(value) for value in node.values]
        return dict(zip(keys, values, strict=False))

    def visit_Slice(self, node: ast.Slice) -> slice | torch.Tensor:
        if node.lower is None:
            lower = None
        else:
            lower = self.visit(node.lower)
        if node.upper is None:
            upper = None
        else:
            upper = self.visit(node.upper)
        if node.step is None:
            step = None
        else:
            step = self.visit(node.step)

        # A fully-bounded unit-step slice becomes hl.arange so FX can trace
        # dynamic bounds.  Static bounds don't need that and stay a real slice.
        if step not in (None, 1) or lower is None or upper is None:
            return slice(lower, upper, step)

        if not (_is_static_slice_bound(lower) and _is_static_slice_bound(upper)):
            # pyrefly: ignore [bad-argument-type]
            return hl.arange(lower, upper)

        return slice(lower, upper, step)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise exc.AssignmentMultipleTargets
        (target,) = node.targets
        if isinstance(target, ast.Name):
            # TODO(jansel): should assert that name is only used on device
            value = self.visit(node.value)
            # For simple variable assignments like `a = b`, we need to create a new
            # variable to avoid phi node issues when the source variable gets mutated
            if isinstance(node.value, ast.Name) and (
                isinstance(value, torch.Tensor) and not isinstance(value, Tile)
            ):
                value = _new_var(value)
            self._assign(target, value)
            return None
        if isinstance(target, ast.Tuple):
            # Handle tuple unpacking
            value = self.visit(node.value)
            if not isinstance(value, tuple):
                raise exc.InvalidAssignment
            if len(target.elts) != len(value):
                raise exc.InvalidAssignment
            for t, v in zip(target.elts, value, strict=True):
                if isinstance(t, ast.Name):
                    self._assign(t, v)
                elif isinstance(t, ast.Subscript):
                    # Handle subscript targets in tuple unpacking (e.g., a[i], b[j] = tuple)
                    self._assign_subscript(t, v)
                else:
                    raise exc.InvalidAssignment
            return None
        if not isinstance(target, ast.Subscript):
            raise exc.InvalidAssignment
        assert isinstance(target, ExtendedAST)
        assert isinstance(target.value, ExtendedAST)
        assert target.value._type_info is not None
        # Handle list element assignment (e.g., cached[i] = tensor in static_range)
        if isinstance(target.value._type_info, SequenceType):
            index_value = self.visit(target.slice)
            if not isinstance(index_value, int):
                raise exc.InvalidSequenceSubscription(target.slice)
            val = self.visit(node.value)
            base_list = self.visit(target.value)
            assert isinstance(base_list, list)
            base_list[index_value] = val
            return None
        assert isinstance(node.value, ExtendedAST)
        rhs_type = node.value._type_info
        lhs_type = target._type_info
        if not isinstance(lhs_type, TensorType) or not isinstance(
            rhs_type, (TensorType, NumericType, LiteralType)
        ):
            raise exc.NonTensorSubscriptAssign(lhs_type, rhs_type)
        target_origin = target.value._type_info.origin
        if not target_origin.is_host() and not isinstance(
            target.value._type_info, StackTensorType
        ):
            # Get the variable name for the error message
            var_name = (
                target.value.id
                if isinstance(target.value, ast.Name)
                else str(target.value)
            )
            raise exc.DeviceTensorSubscriptAssignmentNotAllowed(var_name)
        val = self.visit(node.value)
        if (
            isinstance(rhs_type, TensorType)
            and rhs_type.origin.is_host()
            and not isinstance(node.value, ast.Subscript)
        ):
            full_slices: list[object] = [slice(None)] * rhs_type.fake_value.ndim
            # pyrefly: ignore [bad-argument-type]
            val = hl.load(val, full_slices)
        self._assign_subscript(target, val)

    def _assign_subscript(self, target: ast.Subscript, val: object) -> None:
        """Helper method to assign a value to a subscript target."""
        assert isinstance(target, ExtendedAST)
        lhs_type = target._type_info

        # Validate that we're assigning to a tensor subscript
        from .type_info import TensorType

        if not isinstance(lhs_type, TensorType):
            raise exc.NonTensorSubscriptAssign(lhs_type, type(val))

        assert isinstance(target.value, ExtendedAST)
        assert target.value._type_info is not None
        target_origin = target.value._type_info.origin
        assert target_origin.is_host() or isinstance(
            target.value._type_info, StackTensorType
        )

        return hl.store(
            # pyrefly: ignore [bad-argument-type]
            self.visit(target.value),
            self._subscript_slice_proxy(target.slice),
            # pyrefly: ignore [bad-argument-type]
            val,
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(
                create(
                    ast.Assign,
                    targets=[node.target],
                    value=node.value,
                )
            )

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        assert isinstance(node.target, ExtendedAST)
        self._assign(
            node.target,
            _eval_binary(node.op, self.visit(node.target), self.visit(node.value)),
        )

    def visit_Subscript(self, node: ast.Subscript) -> object:
        value = node.value
        assert isinstance(value, ExtendedAST)
        type_info = value._type_info
        if isinstance(type_info, SequenceType):
            index_value = self.visit(node.slice)
            if isinstance(index_value, int):
                # pyrefly: ignore [bad-index]
                return self.visit(value)[index_value]
            raise exc.InvalidSequenceSubscription(node.slice)
        # Check StackTensorType before DictType since StackTensorType inherits from DictType
        if isinstance(type_info, StackTensorType):
            # pyrefly: ignore [bad-argument-type]
            return hl.load(self.visit(value), self._subscript_slice_proxy(node.slice))
        if isinstance(type_info, DictType):
            key_value = self.visit(node.slice)
            if isinstance(key_value, (str, int)):
                # pyrefly: ignore [bad-index]
                return self.visit(value)[key_value]
            raise exc.TypeInferenceError(
                f"Dict subscript must be a literal str or int, got {type(key_value).__name__}"
            )
        if type_info is not None and type_info.origin.is_host():
            # pyrefly: ignore [bad-argument-type]
            return hl.load(self.visit(value), self._subscript_slice_proxy(node.slice))
        # pyrefly: ignore [bad-argument-type]
        return hl.subscript(self.visit(value), self._subscript_slice_proxy(node.slice))

    def visit_Call(self, node: ast.Call) -> object:
        args = []
        kwargs = {}
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                # pyrefly: ignore [bad-argument-type]
                args.extend(self.visit(arg.value))
            else:
                args.append(self.visit(arg))
        for kwarg in node.keywords:
            if kwarg.arg is None:
                # pyrefly: ignore [no-matching-overload]
                kwargs.update(self.visit(kwarg.value))
            else:
                kwargs[kwarg.arg] = self.visit(kwarg.value)

        if isinstance(
            (
                # pyrefly: ignore [missing-attribute]
                func_type_info := node.func._type_info
            ),
            CallableType,
        ) and (replacement := get_device_func_replacement(func_type_info.value)):
            func = replacement
        else:
            func = self.visit(node.func)

        # pyrefly: ignore [bad-argument-type]
        return _CheckForIndexCalls.retry_call(func, args, kwargs)

    def visit_Attribute(self, node: ast.Attribute) -> object:
        return getattr(self.visit(node.value), node.attr)

    def visit_Expr(self, node: ast.Expr) -> object:
        return self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> object:
        return node.value


class LiftTensorArgs:
    values: dict[str, object]
    flat_values: list[object]
    spec: pytree.TreeSpec
    tensor_indices: list[int]

    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.flat_values, self.spec = pytree.tree_flatten(values)
        self.tensor_indices = [
            i
            for i, v in enumerate(self.flat_values)
            if isinstance(v, torch.Tensor) and not isinstance(v, Tile)
        ]

    def unflatten(self) -> dict[str, object]:
        return pytree.tree_unflatten(self.flat_values, self.spec)

    def replace_tensor_args(
        self, args: Sequence[object], *, copy_tensors: bool = True
    ) -> dict[str, object]:
        flat_values = [*self.flat_values]
        assert len(self.tensor_indices) == len(args)
        for i, v in zip(self.tensor_indices, args, strict=False):
            flat_values[i] = _new_var(v) if copy_tensors else v
        return pytree.tree_unflatten(flat_values, self.spec)

    def get_tensor_args(self) -> list[object]:
        return [self.flat_values[i] for i in self.tensor_indices]

    def get_node_args(
        self, tracer: proxy_tensor.PythonKeyTracer
    ) -> list[torch.fx.Node]:
        proxy_args = args_to_proxies(tracer, self.get_tensor_args())[0]
        result = []
        for proxy in proxy_args:
            assert isinstance(proxy, torch.fx.Proxy)
            result.append(proxy.node)
        return result


def _literal_sequence_is(
    expr: ast.expr,
    *,
    rank: int,
    value: int,
) -> bool:
    if isinstance(expr, ast.Constant) and expr.value == value:
        return True
    if not isinstance(expr, (ast.List, ast.Tuple)) or len(expr.elts) != rank:
        return False
    return all(
        isinstance(item, ast.Constant) and item.value == value for item in expr.elts
    )


def _root_loop_has_canonical_task_origin(
    iter_expr: ast.expr,
    rank: int,
    *,
    supports_step: bool,
) -> bool:
    """Return whether local PIDs map to zero-based, unit-stride coordinates."""
    if not isinstance(iter_expr, ast.Call) or not iter_expr.args:
        return False
    if len(iter_expr.args) == 1 or (
        len(iter_expr.args) >= 2
        and isinstance(iter_expr.args[1], ast.Constant)
        and iter_expr.args[1].value is None
    ):
        zero_start = True
    else:
        zero_start = _literal_sequence_is(iter_expr.args[0], rank=rank, value=0)
    if not zero_start:
        return False
    if not supports_step:
        return True

    step = (
        iter_expr.args[2]
        if len(iter_expr.args) >= 3
        else next(
            (keyword.value for keyword in iter_expr.keywords if keyword.arg == "step"),
            None,
        )
    )
    return step is None or _literal_sequence_is(step, rank=rank, value=1)


class WalkHostAST(NodeVisitor):
    def __init__(self, device_ir: DeviceIR) -> None:
        super().__init__()
        self.device_ir = device_ir
        self.root_index = 0
        self.current_phase_roots: list[int] = []
        self.phases: list[KernelPhase] = []
        self.root_nodes: list[ast.For] = []

    def _flush_current_phase(self) -> None:
        assert self.current_phase_roots
        self.phases.append(
            KernelPhase(
                roots=self.current_phase_roots,
                root_nodes=[self.root_nodes[r] for r in self.current_phase_roots],
            )
        )
        self.current_phase_roots = []

    def visit_For(self, node: ast.For) -> None:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            self.device_ir.add_root_graph(
                _make_fx(lambda: WalkDeviceAST(self.device_ir).visit(node))
            )
            # pyrefly: ignore [missing-attribute]
            iter_type = node.iter._type_info
            assert isinstance(iter_type, IterType)
            inner = iter_type.inner
            if isinstance(inner, SequenceType):
                # pyrefly: ignore [missing-attribute]
                block_ids = [x.block_id for x in inner.unpack()]
            else:
                # pyrefly: ignore [missing-attribute]
                block_ids = [inner.block_id]
            self.device_ir.grid_block_ids.append(block_ids)
            supports_step = (
                all(isinstance(value, GridIndexType) for value in inner.unpack())
                if isinstance(inner, SequenceType)
                else isinstance(inner, GridIndexType)
            )
            canonical_origin = _root_loop_has_canonical_task_origin(
                node.iter,
                len(block_ids),
                supports_step=supports_step,
            )
            if not canonical_origin:
                self.device_ir.noncanonical_task_origin_block_ids.update(block_ids)
            from .tile_dependency import TaskAxis
            from .tile_dependency import TaskFamily

            env = CompileEnvironment.current()
            self.device_ir.task_families.append(
                TaskFamily(
                    axes=tuple(
                        TaskAxis(
                            block_id=block_id,
                            extent=env.block_sizes[block_id].numel,
                            canonical_origin=canonical_origin,
                        )
                        for block_id in block_ids
                    ),
                )
            )
            # store root index (position) not graph id
            self.root_nodes.append(node)
            self.current_phase_roots.append(len(self.device_ir.root_ids) - 1)
            self.root_index += 1
        else:
            self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        # Record barrier placement between top-level loops.
        from .type_info import BarrierResultType

        assert isinstance(node, ExtendedAST)
        assert isinstance(node.value, ExtendedAST)
        is_barrier = isinstance(node.value._type_info, BarrierResultType)

        if is_barrier:
            if self.root_index == 0 or not self.current_phase_roots:
                raise exc.BarrierOnlyAllowedAtTopLevel
            self._flush_current_phase()
            return
        self.generic_visit(node)

    def flush_phases(self) -> None:
        if self.current_phase_roots:
            self._flush_current_phase()


def _indexing_uses_tensor_descriptor(
    indexing_config: object,
    op_index: int,
) -> bool:
    if isinstance(indexing_config, str):
        return indexing_config == "tensor_descriptor"
    if isinstance(indexing_config, (list, tuple)):
        return (
            op_index < len(indexing_config)
            and indexing_config[op_index] == "tensor_descriptor"
        )
    return False


def _count_device_atomics(device_ir: DeviceIR) -> int:
    """Count the number of atomic operations in device code for autotuning."""
    from ..language import atomic_ops

    atomic_count = 0
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op == "call_function" and node.target in vars(atomic_ops).values():
                atomic_count += 1
    return atomic_count


def _register_load_store_tunables(
    total_load_count: int,
    loads_without_eviction_policy: int,
    loads_without_cache_modifier: int,
    stores_without_cache_modifier: int,
    store_indices: list[int],
) -> None:
    """Register list-based tunables for device loads and stores.

    Args:
        total_load_count: Total number of loads (for indexing tunable)
        loads_without_eviction_policy: Number of loads that need eviction policy tuning
        loads_without_cache_modifier: Number of loads that need cache modifier tuning
        stores_without_cache_modifier: Number of stores that need cache modifier tuning
        store_indices: Positions of store ops in the combined indexing list
    """
    store_count = len(store_indices)
    env = CompileEnvironment.current()
    env.config_spec.store_indices = store_indices
    if total_load_count == 0 and store_count == 0:
        return

    from ..autotuner.config_fragment import EnumFragment
    from ..autotuner.config_fragment import ListOf
    from ..autotuner.config_spec import get_valid_eviction_policies
    from ..autotuner.config_spec import get_valid_load_cache_modifiers
    from ..autotuner.config_spec import get_valid_store_cache_modifiers

    # Register eviction policies only for loads without explicit eviction_policy
    if loads_without_eviction_policy > 0:
        env.config_spec.load_eviction_policies = ListOf(
            EnumFragment(choices=get_valid_eviction_policies(env.backend_name)),
            length=loads_without_eviction_policy,
        )

    # Register cache modifiers only for loads and only when the backend has
    # a non-trivial search space.
    load_cache_modifier_choices = get_valid_load_cache_modifiers(env.backend_name)
    if loads_without_cache_modifier > 0 and len(load_cache_modifier_choices) > 1:
        env.config_spec.load_cache_modifiers = ListOf(
            EnumFragment(choices=load_cache_modifier_choices),
            length=loads_without_cache_modifier,
        )

    # Register cache modifiers for stores when the backend has a non-trivial
    # search space.
    store_cache_modifier_choices = get_valid_store_cache_modifiers(env.backend_name)
    if stores_without_cache_modifier > 0 and len(store_cache_modifier_choices) > 1:
        env.config_spec.store_cache_modifiers = ListOf(
            EnumFragment(choices=store_cache_modifier_choices),
            length=stores_without_cache_modifier,
        )

    # Indexing applies to ALL loads and stores
    total_count = total_load_count + store_count
    if total_count > 0:
        env.config_spec.indexing = ListOf(
            EnumFragment(choices=env.config_spec.valid_indexing_types()),
            length=total_count,
        )


def _register_atomic_tunables(atomic_count: int) -> None:
    """Register atomic_indexing tunable for all atomic operations."""
    if atomic_count == 0:
        return

    from ..autotuner.config_fragment import EnumFragment
    from ..autotuner.config_fragment import ListOf

    env = CompileEnvironment.current()
    env.config_spec.atomic_indexing = ListOf(
        EnumFragment(choices=env.config_spec.valid_atomic_indexing_types()),
        length=atomic_count,
    )


def _register_tensor_descriptor_layout_guards(device_ir: DeviceIR) -> None:
    env = CompileEnvironment.current()
    if env.settings.static_shapes:
        return

    from .._compat import supports_tensor_descriptor
    from ..language import atomic_ops
    from ..language import memory_ops

    if not supports_tensor_descriptor():
        return

    atomic_targets = tuple(getattr(atomic_ops, name) for name in atomic_ops.__all__)

    def tensor_arg_value(arg: object) -> object:
        if isinstance(arg, torch.fx.Node):
            return arg.meta.get("val")
        return arg

    memory_op_index = 0
    atomic_op_index = 0
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in (memory_ops.load, memory_ops.store):
                tensor = tensor_arg_value(node.args[0])
                if isinstance(tensor, torch.Tensor) and 2 <= tensor.ndim <= 5:
                    env.register_tensor_descriptor_layout_guard(
                        tensor, memory_op_index=memory_op_index
                    )
                memory_op_index += 1
                continue
            if node.target in atomic_targets:
                tensor = tensor_arg_value(node.args[0])
                if isinstance(tensor, torch.Tensor) and 2 <= tensor.ndim <= 5:
                    env.register_tensor_descriptor_layout_guard(
                        tensor, atomic_op_index=atomic_op_index
                    )
                atomic_op_index += 1


def _register_cute_lane_vector_width_specs(config_spec: ConfigSpec) -> None:
    """Pre-register CuTe vector-width slots for every lane-loop-eligible block.

    On the CuTe backend a tile block whose configured ``block_size`` exceeds
    its ``num_threads`` is traversed by a synthetic lane loop, and the tile
    strategy lazily registers a ``cute_vector_widths`` slot for it during
    codegen. That lazy append happens *per config* and is config dependent
    (whether a lane loop appears depends on the config's block sizes and
    num_threads), so the shared ``config_spec`` would grow after
    ``ConfigGeneration`` has already captured its ``flat_spec`` — leaving the
    flat layout stale and causing ``unflatten`` to over-walk the spec.

    Registering the slots once here (config independently) keeps the lazy
    append idempotent: any block that *could* be lane-looped under some config
    already owns its slot, so the per-config append is a no-op and the spec
    stays a fixed size for the lifetime of autotuning. Blocks that never end up
    lane-looped simply keep ``V=1`` (the slot's default), which the tile
    strategy ignores.
    """
    from ..autotuner.config_spec import CuteVectorWidthSpec

    num_thread_block_ids = set(config_spec.num_threads.valid_block_ids())
    existing = set(config_spec.cute_vector_widths.valid_block_ids())
    for spec in config_spec.block_sizes:
        block_id = spec.block_id
        if block_id not in num_thread_block_ids or block_id in existing:
            continue
        # ``max_size`` bounds the largest block_size the autotuner may pick;
        # only blocks that can exceed a single thread can ever be lane-looped.
        if spec.max_size <= 1:
            continue
        config_spec.cute_vector_widths.append(
            CuteVectorWidthSpec(block_id=block_id, size_hint=spec.size_hint)
        )
        existing.add(block_id)


def _root_phases(phases: list[KernelPhase], root_count: int) -> tuple[int, ...]:
    result = [-1] * root_count
    for phase_index, phase in enumerate(phases):
        for root in phase.roots:
            result[root] = phase_index
    assert all(phase >= 0 for phase in result)
    return tuple(result)


def _install_dependency_phases(
    device_ir: DeviceIR,
    root_nodes: list[ast.For],
    dependency_graph: TileDependencyGraph,
    source_phase_starts: frozenset[int],
) -> None:
    if dependency_graph.edges and source_phase_starts:
        raise exc.CrossLoopSchedulingError(
            "mixing explicit hl.barrier() phases with implicit "
            "tile-dependency stages is not supported yet"
        )

    predecessors_by_consumer: list[list[int]] = [[] for _ in device_ir.task_families]
    for edge in dependency_graph.edges:
        predecessors_by_consumer[edge.consumer_root].append(edge.producer_root)
    stage_by_root: list[int] = []
    implicit_starts: set[int] = set()
    stage = 0
    for root in range(len(device_ir.task_families)):
        if root in source_phase_starts:
            stage += 1
        if any(
            stage_by_root[producer] == stage
            for producer in predecessors_by_consumer[root]
        ):
            implicit_starts.add(root)
            stage += 1
        stage_by_root.append(stage)

    device_ir.implicit_dependency_starts = frozenset(implicit_starts)
    stage_count = stage_by_root[-1] + 1 if stage_by_root else 0
    roots_by_stage = [[] for _ in range(stage_count)]
    for root, root_stage in enumerate(stage_by_root):
        roots_by_stage[root_stage].append(root)
    device_ir.phases = [
        KernelPhase(
            roots=roots,
            root_nodes=[root_nodes[root] for root in roots],
        )
        for roots in roots_by_stage
    ]
    for phase_index, phase in enumerate(device_ir.phases):
        for root in phase.roots:
            graph_info = device_ir.graphs[device_ir.root_ids[root]]
            assert isinstance(graph_info, RootGraphInfo)
            graph_info.phase_index = phase_index


def lower_to_device_ir(func: HostFunction) -> DeviceIR:
    device_ir = DeviceIR()
    device_ir.host_function = func
    with func, device_ir, compile_lock:
        visitor = WalkHostAST(device_ir)
        for stmt in func.body:
            visitor.visit(stmt)
        visitor.flush_phases()
        device_ir.phases = visitor.phases
        source_root_phases = _root_phases(device_ir.phases, len(device_ir.root_ids))
        source_phase_starts = frozenset(
            phase.roots[0] for phase in device_ir.phases[1:]
        )
        for phase_idx, phase in enumerate(device_ir.phases):
            for ridx in phase.roots:
                graph_info = device_ir.graphs[device_ir.root_ids[ridx]]
                assert isinstance(graph_info, RootGraphInfo)
                graph_info.phase_index = phase_idx
        # If there are no top-level device loops, we cannot generate a valid kernel.
        # Raise a friendly error instead of emitting an empty Triton function body.
        if len(device_ir.root_ids) == 0:
            raise exc.NoDeviceLoopsInKernel
        from ..language.random_ops import rewrite_implicit_random_ops

        for graph in device_ir.graphs:
            rewrite_implicit_random_ops(graph.graph)
        if CompileEnvironment.current().backend.name == "cute":
            promotions = collect_cute_half_atomic_output_promotions(device_ir.graphs)
            if promotions:
                host_fn = HostFunction.current()
                rewrite_cute_half_atomic_output_allocations(host_fn, promotions)
                promote_cute_root_graph_host_tensors(device_ir.graphs, promotions)
        for graph in device_ir.graphs:
            prepare_graph_lowerings(graph.graph)
        defer_load_masks = CompileEnvironment.current().backend.name == "pallas"
        for graph in device_ir.graphs:
            validate_host_tensor_usage(graph.graph)
            add_tile_with_offset_metadata(graph)
            remove_unnecessary_tile_index(graph.graph)
            if defer_load_masks:
                defer_pallas_load_masks(graph.graph)
            remove_unnecessary_masking(graph.graph)

        # TODO(hinriksnaer): extract into a separate step? everything below
        # is post-processing computed from the completed DeviceIR.
        from .epilogue_subtiling import has_epilogue_subtiling_candidate

        has_epilogue_subtile_candidate = False
        for graph_info in device_ir.graphs:
            if has_epilogue_subtiling_candidate(graph_info.graph):
                has_epilogue_subtile_candidate = True
                break
        config_spec = CompileEnvironment.current().config_spec
        config_spec.epilogue_subtile_candidate_enabled = has_epilogue_subtile_candidate
        config_spec.epilogue_subtile_k_hint = 0
        config_spec.epilogue_subtile_autotune_choices = None

        # Phase 1: roll + register the reduction_loops spec. The ReductionKernelFact is built
        # in Phase 3 (after analysis derives memory_op_facts) so it can read the enriched
        # load/store facts.
        device_ir.register_rollable_reductions()
        if CompileEnvironment.current().backend.name == "cute":
            _register_cute_lane_vector_width_specs(config_spec)
            # Enable the flash-attention autotune surface when
            # the dense flash dataflow is detected, analogous to how a matmul
            # detection sets ``cute_tcgen05_search_enabled``. Default-off
            # otherwise so the flash knobs never widen the search surface for
            # ordinary cute kernels.
            from .backend import detect_flash_search_surface

            flash_shape = detect_flash_search_surface(device_ir)
            if flash_shape is not None:
                config_spec.enable_cute_flash_search(
                    head_dim=flash_shape.head_dim,
                    num_kv=flash_shape.num_kv,
                    num_bh=flash_shape.num_bh,
                    tensor_4d_heads=flash_shape.tensor_4d_heads,
                    dtype=flash_shape.io_dtype,
                    block_size_targets=flash_shape.block_size_targets,
                    is_causal=flash_shape.is_causal,
                    has_kv_tile_pruning=flash_shape.has_kv_tile_pruning,
                    requires_ws_overlap=flash_shape.requires_ws_overlap,
                    small_biased_candidate=flash_shape.small_biased_candidate,
                    standard_dense_output=flash_shape.standard_dense_output,
                    standard_causal_output=flash_shape.standard_causal_output,
                    output_requires_tma=flash_shape.output_requires_tma,
                    supports_tensor_4d_tma=flash_shape.supports_tensor_4d_tma,
                )
            else:
                from ..language.matmul_ops import enable_cute_tcgen05_search
                from ..language.matmul_ops import plan_cute_tcgen05_search
                from .cute.cute_mma import analyze_cute_mma_node
                from .cute.cute_mma import (
                    tcgen05_fragment_epilogue_source_tiles_reachable,
                )

                # The same structural analyzer gates tcgen05 search and codegen
                # for every matrix rank. This prevents transformed loads from
                # receiving a tcgen05-only search space that codegen later rejects.
                root_grid_block_ids = {
                    tuple(block_ids) for block_ids in device_ir.grid_block_ids
                }
                mma_candidates = []
                for graph_info in device_ir.graphs:
                    for node in graph_info.graph.nodes:
                        candidate = analyze_cute_mma_node(node, device_ir=device_ir)
                        if (
                            candidate is None
                            or candidate.requires_accumulator_seed
                            or candidate.operands.output_block_ids
                            not in root_grid_block_ids
                        ):
                            continue
                        lhs = candidate.lhs.meta.get("val")
                        rhs = candidate.rhs.meta.get("val")
                        if not (
                            isinstance(lhs, torch.Tensor)
                            and isinstance(rhs, torch.Tensor)
                        ):
                            continue
                        mma_candidates.append((candidate, lhs, rhs, node))
                supports_small_n_role_local_tma = bool(mma_candidates) and all(
                    candidate.supports_small_n_role_local_tma
                    for candidate, _lhs, _rhs, _node in mma_candidates
                )
                supports_small_n_scalar_fallback = bool(mma_candidates) and all(
                    candidate.supports_small_n_scalar_fallback
                    for candidate, _lhs, _rhs, _node in mma_candidates
                )
                search_candidates = []
                for candidate, lhs, rhs, _node in mma_candidates:
                    search_plan = plan_cute_tcgen05_search(
                        lhs,
                        rhs,
                        has_leading_passthrough=(
                            candidate.operands.has_leading_passthrough
                        ),
                        supports_small_n_role_local_tma=(
                            supports_small_n_role_local_tma
                        ),
                        supports_small_n_scalar_fallback=(
                            supports_small_n_scalar_fallback
                        ),
                    )
                    if (
                        search_plan is None
                        or not candidate.supports_tcgen05_search_plan(search_plan)
                    ):
                        continue
                    if not tcgen05_fragment_epilogue_source_tiles_reachable(
                        candidate, search_plan, config_spec
                    ):
                        continue
                    search_candidates.append((candidate, lhs, search_plan))
                analysis_keys = {
                    (
                        candidate.operands.output_block_ids,
                        candidate.operands.k_block_id,
                        candidate.operands.lhs.source_fake.dtype,
                    )
                    for candidate, _lhs, _plan in search_candidates
                }
                if len(analysis_keys) == 1:
                    candidate, lhs, search_plan = search_candidates[0]
                    enable_cute_tcgen05_search(
                        lhs,
                        plan=search_plan,
                        m_block_id=candidate.operands.m_block_id,
                        n_block_id=candidate.operands.n_block_id,
                        k_block_id=candidate.operands.k_block_id,
                        input_dtype=candidate.operands.lhs.source_fake.dtype,
                        has_leading_passthrough=(
                            candidate.operands.has_leading_passthrough
                        ),
                        explicit_epi_tile_compatible=all(
                            item.explicit_epi_tile_compatible
                            for item, _lhs, _plan in search_candidates
                        ),
                    )
        config_spec.raise_grid_block_minimums()
        if len(device_ir.root_ids) > 1:
            # xyz is not supported with shared program IDs. Non-tcgen05
            # persistent kernels are allowed; tcgen05 persistent has a
            # single-root scheduler/grid contract today.
            config_spec.disallow_pid_type(
                "xyz",
                reason="multiple root loops share program IDs; xyz grids are not "
                "supported with shared program IDs",
            )
            if config_spec.cute_tcgen05_search_enabled:
                # The tcgen05 persistent launch grid is derived from a single
                # root's PID space today. Keep persistent pid types out of
                # multi-root autotune until the scheduler/grid spans all cases.
                non_persistent_pid_types = tuple(
                    pid_type
                    for pid_type in config_spec.allowed_pid_types
                    if pid_type not in ("persistent_blocked", "persistent_interleaved")
                )
                if not non_persistent_pid_types:
                    raise exc.InvalidConfig(
                        "CuTe tcgen05 multi-root kernels do not support "
                        "persistent pid types yet, and no non-persistent "
                        "pid type is available. Disable forced/distributed "
                        "persistent-only mode or use a single root loop."
                    )
                for pid_type in config_spec.allowed_pid_types:
                    if pid_type not in non_persistent_pid_types:
                        config_spec.disallowed_pid_type_reasons.setdefault(
                            pid_type,
                            "CuTe tcgen05 multi-root kernels do not support "
                            "persistent pid types yet (single-root launch grid)",
                        )
                config_spec.allowed_pid_types = non_persistent_pid_types

        # Build structural and liveness observations once, after reduction rolling has
        # finalized the graph set. Specialized fact passes consume this shared analysis.
        from .device_ir_analysis import DeviceIRAnalysis

        env = CompileEnvironment.current()
        analysis = DeviceIRAnalysis.build(device_ir, env)
        config_spec.kernel_grid_fact = analysis.kernel_grid_fact

        # Collect per-load/store metadata so heuristics can map each Config.indexing
        # slot to its graph op.
        memory_op_facts = analysis.memory_op_facts(env, func)
        tile_accesses = analysis.tile_accesses(device_ir, env, func)
        config_spec.memory_op_facts = memory_op_facts
        from .tile_dependency import build_tile_dependency_graph

        if len(device_ir.task_families) > 1:
            device_ir.tile_dependency_graph = build_tile_dependency_graph(
                tile_accesses,
                device_ir=device_ir,
                root_phases=source_root_phases,
            )
            _install_dependency_phases(
                device_ir,
                visitor.root_nodes,
                device_ir.tile_dependency_graph,
                source_phase_starts,
            )
            if device_ir.implicit_dependency_starts:
                if (
                    CompileEnvironment.current().backend_name != "triton"
                    or CompileEnvironment.current().device.type != "cuda"
                    or torch.version.hip is not None
                ):
                    edge = next(
                        edge
                        for edge in device_ir.tile_dependency_graph.edges
                        if edge.consumer_root in device_ir.implicit_dependency_starts
                    )
                    names = sorted(edge.tensor_names)
                    raise exc.LoopDependencyError(
                        names[0] if names else "tensor allocation"
                    )
                reason = (
                    "tile dependencies require a persistent blocked kernel for "
                    "tile-dependency scheduling"
                )
                CompileEnvironment.current().require_persistent_blocked(reason)
                config_spec.enable_cross_loop_schedule()
        if config_spec.supports_config_key("pallas_load_buffer_count"):
            config_spec.pallas_load_buffer_count.length = len(
                LiftTensorArgs(dict(func.params.arguments)).get_tensor_args()
            )
        if config_spec.supports_config_key("pallas_indirect_access_mode"):
            from .pallas.tensorcore_plan import indirect_access_modes

            config_spec.pallas_indirect_access_modes = indirect_access_modes(device_ir)
        load_count = sum(f.kind == "load" for f in memory_op_facts)
        _register_load_store_tunables(
            load_count,
            sum(f.eviction_index is not None for f in memory_op_facts),
            # cache_modifier is tuned for every load (no per-load override)
            load_count,
            sum(f.kind == "store" for f in memory_op_facts),
            [f.indexing_index for f in memory_op_facts if f.kind == "store"],
        )
        _register_atomic_tunables(_count_device_atomics(device_ir))
        _register_tensor_descriptor_layout_guards(device_ir)

        # Accumulator facts are a standalone, reduction-agnostic fact layer (any heuristic
        # may read them), so build them independently of the reduction facts. Must run
        # after the rolling (it walks the rolled loop subgraphs).
        config_spec.accumulator_facts = device_ir.build_accumulator_facts(analysis)
        # Phase 3 (PROMPT §2.1/§2.2): build the categorizing ReductionKernelFact — the first-class
        # reduction-descriptor list + co-residency groups the reduction seed + allocator consume.
        device_ir.build_reduction_kernel_fact(
            memory_op_facts,
            config_spec.accumulator_facts,
            analysis,
        )
        # Phase 3b: compose the whole-kernel contraction fact for ANY kernel with a matmul,
        # so both matmul front ends read one description of the workload (axis roles, knob
        # competition, per-dot placement/work, peak liveness).
        device_ir.build_kernel_matmul_fact(analysis)
        # Phase 4: compose a matmul + reduction-over-output epilogue fact when a matmul AND a
        # register-resident epilogue reduction co-occur (matmul_rms_norm etc.).
        device_ir.build_matmul_reduction_epilogue_facts()
        # Phase 5: record a PointwiseElementwiseFact for a PURE elementwise kernel (no
        # reduction/matmul/accumulator fact) so the pointwise seed can size a BW-saturating
        # tile instead of the starved block_size=32 default.
        device_ir.build_pointwise_facts(analysis)

        return device_ir


@dataclasses.dataclass
class HelperFunctionGraphInfo(NodeArgsGraphInfo):
    """Graph info for helper functions in higher-order operations like associative_scan."""

    _param_names: list[str] = dataclasses.field(default_factory=list)
    original_function_name: str | None = dataclasses.field(default=None)

    @property
    def name(self) -> str:
        # This property should only be used during registration, not for final names
        # Final names are generated in codegen using the namespace below
        if self.original_function_name:
            return f"{self.original_function_name}_{self.graph_id}"
        return f"helper_function_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            **super().kwargs(),
            "_param_names": [*self._param_names],
            "original_function_name": self.original_function_name,
        }

    def find_input_nodes(self) -> list[torch.fx.Node]:
        """Find all placeholder nodes (inputs) in the graph."""
        return self.graph.find_nodes(op="placeholder")

    def codegen(self, state: CodegenState) -> list[object]:
        from .helper_function import codegen_helper_function_graph_info

        return codegen_helper_function_graph_info(self, state)


def validate_host_tensor_usage(graph: torch.fx.Graph) -> None:
    """
    Validate that scalar _host_tensor ops only flow into allowed operations.
    This replaces the AST visitor context detection with cleaner FX graph validation.
    Only checks 0-dimensional tensors (scalars), not regular tensors.
    Uses decorator metadata to determine which operations allow host tensors.
    """
    from ..language._decorators import is_api_func
    from ..language._tracing_ops import _host_tensor

    for node in graph.find_nodes(op="call_function", target=_host_tensor):
        scalar_tensor_name = node.args[0]
        assert isinstance(scalar_tensor_name, str), scalar_tensor_name

        # Check all users of this scalar _host_tensor node
        for user in node.users:
            if user.op == "call_function":
                # Check if this operation allows host tensors via decorator metadata
                if not (
                    is_api_func(user.target)
                    and getattr(user.target, "_allow_host_tensor", False)
                ):
                    op_name = getattr(user.target, "__name__", str(user.target))
                    raise exc.HostTensorDirectUsage(scalar_tensor_name, op_name)


def add_tile_with_offset_metadata(graph_info: GraphInfo) -> None:
    """
    Recognize tile.index + offset patterns and add metadata to enable tensor descriptor indexing.

    This pass identifies FX nodes that represent `tile.index + offset` (where offset is an
    integer or SymInt), and adds the `tile_with_offset` metadata to those nodes so that
    indexing strategies can generate efficient code (e.g., tensor descriptors) for them.
    """
    graph = graph_info.graph
    env = CompileEnvironment.current()
    add_targets = (operator.add, torch.ops.aten.add.Tensor)
    offset_types = (int, torch.SymInt)
    for node in graph.nodes:
        if (
            node.op != "call_function"
            or node.target not in add_targets
            or node.kwargs
            or len(node.args) != 2
        ):
            continue

        block_id: int | None = None
        total_offset: int | torch.SymInt = 0
        valid = True

        for arg in node.args:
            tile_offset_value: int | torch.SymInt | None = None
            arg_block_id: int | None = None

            if isinstance(arg, torch.fx.Node):
                meta_tile = arg.meta.get("tile_with_offset")
                if meta_tile is not None:
                    arg_block_id = meta_tile.get("block_id")
                    if arg_block_id is None:
                        valid = False
                        break
                    tile_offset_value = meta_tile.get("offset", 0)
                elif (
                    arg.op == "call_function"
                    and arg.target == hl.tile_index
                    and arg.args
                    and isinstance(arg.args[0], torch.fx.Node)
                ):
                    tile_val = arg.args[0].meta.get("val")
                    if isinstance(tile_val, torch.SymInt):
                        arg_block_id = env.get_block_id(tile_val)
                        if arg_block_id is None:
                            valid = False
                            break
                        tile_offset_value = 0
                else:
                    val = arg.meta.get("val")
                    if isinstance(val, offset_types):
                        total_offset = total_offset + val
                        continue

                if arg_block_id is not None:
                    if block_id is not None:
                        valid = False
                        break
                    if tile_offset_value is None:
                        tile_offset_value = 0
                    block_id = arg_block_id
                    total_offset = total_offset + tile_offset_value
                    continue

                val = arg.meta.get("val")
                if isinstance(val, offset_types):
                    total_offset = total_offset + val
                    continue

                valid = False
                break

            if isinstance(arg, offset_types):
                total_offset = total_offset + arg
                continue
            valid = False
            break

        if not valid or block_id is None:
            continue

        node.meta["tile_with_offset"] = {
            "block_id": block_id,
            "offset": total_offset,
        }


def remove_unnecessary_tile_index(graph: torch.fx.Graph) -> None:
    """
    Remove unnecessary tile_index nodes from the graph.
    Passing a tile directly results block_ptrs being supported.
    """
    for node in graph.find_nodes(op="call_function", target=hl.tile_index):
        for user in [*node.users]:
            if user.op == "call_function" and user.target in (hl.load, hl.store):
                new_args = [*user.args]
                assert isinstance(new_args[1], (list, tuple))
                new_args[1] = [(node.args[0] if x is node else x) for x in new_args[1]]
                user.args = tuple(new_args)
        if len(node.users) == 0:
            graph.erase_node(node)


def collect_cute_half_atomic_output_promotions(
    graph_infos: list[GraphInfo],
) -> dict[str, torch.dtype]:
    from ..language import atomic_add
    from ..language._tracing_ops import _host_tensor
    from .variable_origin import ArgumentOrigin

    promotions: dict[str, torch.dtype] = {}
    host_fn = HostFunction.current()
    host_tensor_nodes: dict[str, list[torch.fx.Node]] = {}

    for graph_info in graph_infos:
        for node in graph_info.graph.nodes:
            if node.op == "call_function" and node.target is _host_tensor:
                target_name = node.args[0]
                if isinstance(target_name, str):
                    host_tensor_nodes.setdefault(target_name, []).append(node)

    def is_promotable_target(node: torch.fx.Node) -> bool:
        target_val = node.meta.get("val")
        if (
            not isinstance(target_val, torch.Tensor)
            or target_val.dtype != torch.float16
        ):
            return False
        origin = host_fn.tensor_to_origin.get(target_val)
        if origin is None or isinstance(origin, ArgumentOrigin):
            return False
        if not node.users:
            return False
        for user in node.users:
            if user.op != "call_function" or user.target is not atomic_add:
                return False
            if len(user.args) < 3 or user.args[0] is not node or len(user.users) != 0:
                return False
            value_node = user.args[2]
            if not isinstance(value_node, torch.fx.Node):
                return False
            value_val = value_node.meta.get("val")
            if not isinstance(value_val, torch.Tensor) or value_val.dtype not in (
                torch.float16,
                torch.float32,
            ):
                return False
        return True

    for target_name, nodes in host_tensor_nodes.items():
        if all(is_promotable_target(node) for node in nodes):
            promotions[target_name] = torch.float16

    return promotions


def rewrite_cute_half_atomic_output_allocations(
    host_fn: HostFunction,
    promotions: dict[str, torch.dtype],
) -> None:
    torch_factory_names = {
        "empty",
        "empty_like",
        "full",
        "full_like",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    }

    def dtype_expr(dtype: torch.dtype) -> ast.expr:
        expr = expr_from_string(f"torch.{str(dtype).split('.', 1)[1]}")
        assert isinstance(expr, ast.expr)
        return expr

    def is_torch_factory_call(call: ast.Call) -> bool:
        return (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in torch_factory_names
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "torch"
        )

    def rewrite_allocation_dtype(call: ast.Call) -> None:
        dtype = dtype_expr(torch.float32)
        for kwarg in call.keywords:
            if kwarg.arg == "dtype":
                kwarg.value = dtype
                return
        if is_torch_factory_call(call):
            call.keywords.append(create(ast.keyword, arg="dtype", value=dtype))

    def rewrite_return_expr(expr: ast.expr) -> ast.expr:
        if isinstance(expr, ast.Name) and expr.id in promotions:
            cast_expr = expr_from_string(
                "{value}.to({dtype})",
                value=expr,
                dtype=dtype_expr(promotions[expr.id]),
            )
            assert isinstance(cast_expr, ast.expr)
            return cast_expr
        if isinstance(expr, ast.Tuple):
            return create(
                ast.Tuple,
                elts=[rewrite_return_expr(elt) for elt in expr.elts],
                ctx=expr.ctx,
            )
        if isinstance(expr, ast.List):
            return create(
                ast.List,
                elts=[rewrite_return_expr(elt) for elt in expr.elts],
                ctx=expr.ctx,
            )
        return expr

    for stmt in ast.walk(ast.Module(body=host_fn.body, type_ignores=[])):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in promotions
            and isinstance(stmt.value, ast.Call)
        ):
            rewrite_allocation_dtype(stmt.value)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            stmt.value = rewrite_return_expr(stmt.value)


def promote_cute_root_graph_host_tensors(
    graph_infos: list[GraphInfo],
    promotions: dict[str, torch.dtype],
) -> None:
    from ..language._tracing_ops import _host_tensor

    host_fn = HostFunction.current()
    for graph_info in graph_infos:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not _host_tensor:
                continue
            target_name = node.args[0]
            if not isinstance(target_name, str) or target_name not in promotions:
                continue
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                promoted_value = value.to(dtype=torch.float32)
                if origin := host_fn.tensor_to_origin.get(value):
                    host_fn.tensor_to_origin[promoted_value] = origin
                node.meta["val"] = promoted_value
