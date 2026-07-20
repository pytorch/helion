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
from .loop_dependency_checker import LoopDependencyChecker
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
    from collections.abc import Sequence

    from ..autotuner.config_spec import AccumulatorFact
    from ..autotuner.config_spec import ConfigSpec
    from ..autotuner.config_spec import MemoryOpFact
    from .cute.layout import CuTeGridExecutionPlan

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
        # Make the active graph reachable by the strategy so it can pick
        # different lane-loop shapes for the reduce vs consume sweeps.
        # pyrefly: ignore [missing-attribute]
        state.codegen._cute_active_graph_info = self
        try:
            with state.codegen.add_device_loop(
                state.device_function.tile_strategy.codegen_device_loop(
                    state, self.block_ids
                ),
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
    loop_dependency_checker: LoopDependencyChecker = dataclasses.field(
        default_factory=LoopDependencyChecker
    )


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


def _classify_load_dataflow(
    load_node: torch.fx.Node, redset: set[int], env: CompileEnvironment
) -> tuple[set[int], tuple[tuple[int | None, ...], ...]]:
    """Trace a load's value forward over ``node.users``, cutting at BOTH reductions and
    stores, and record the sinks it reaches:

    - ``reductions_fed``: the ``redset`` reduction ids the value flows into (recorded,
      not traversed through — we want where the row goes, not the reduced result).
    - ``stores_fed``: the stores the value reaches WITHOUT passing through a reduction,
      each keyed by the store's full subscript-axis tuple (``store[tile_m, tile_n]`` ->
      ``(id_m, id_n)``). A non-empty set means the row is live across the reduction.
    """
    from ..language.memory_ops import store as _store_op

    reductions_fed: set[int] = set()
    stores_fed: set[tuple[int | None, ...]] = set()
    seen: set[int] = set()
    stack = list(load_node.users)
    while stack:
        u = stack.pop()
        if id(u) in redset:
            reductions_fed.add(id(u))
            continue  # do not traverse THROUGH the reduction
        if id(u) in seen:
            continue
        seen.add(id(u))
        if u.op == "call_function" and u.target is _store_op:
            stores_fed.add(_store_axis_key(env, u))
            continue  # do not traverse THROUGH the store
        stack.extend(u.users)
    # deterministic order (None sorts low); only non-emptiness is consumed today
    stores_sorted = tuple(
        sorted(stores_fed, key=lambda t: tuple(-1 if b is None else b for b in t))
    )
    return reductions_fed, stores_sorted


def _store_axis_key(
    env: CompileEnvironment, store_node: torch.fx.Node
) -> tuple[int | None, ...]:
    """Full subscript-axis key of a store: the block-id each non-bare-int subscript indexes
    (the faithful subscript provenance, falling back to the shape-resolved id for a plain-slice
    subscript), one entry per dim. E.g. ``store[tile_m, tile_n]`` -> ``(id_m, id_n)``;
    ``out[tile_m, :]`` -> ``(id_m, <shape-resolved id>)``."""
    fake = _accessed_tensor_fake(store_node)
    index_list = store_node.args[1] if len(store_node.args) >= 2 else None
    if fake is None or not isinstance(index_list, (list, tuple)):
        return ()
    key: list[int | None] = []
    for pos, sub in enumerate(index_list):
        if isinstance(sub, int) or pos >= fake.ndim:
            continue
        bid = _subscript_block_id(env, sub)
        if bid is None:
            bid = env.resolve_block_id(fake.shape[pos])
        key.append(bid)
    return tuple(key)


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
        self.grid_block_ids: list[list[int]] = []
        # Owning HostFunction (captured in ``lower_to_device_ir``).
        self.host_function: HostFunction | None = None

    def __str__(self) -> str:
        return "\n\n".join(map(str, self.graphs))

    def debug_str(self) -> str:
        result = str(self)
        # Normalize indentation to 4 spaces to handle both PyTorch 2.9 and nightly formatting
        return re.sub(r" *(# File:\s+).*/([^/:]+:\d+)", r"    \1.../\2", result)

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
        config_spec entries for the autotuner.  Sub-graphs created by the roller
        (e.g. ReductionLoopGraphInfo) are kept so that _collect_memory_op_facts
        can account for their loads/stores in the indexing config.
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

    def _original_graph_reductions(self) -> list[tuple[int, int]]:
        """Every reduction OCCURRENCE on the ORIGINAL (pre-roll) device graphs as
        ``(graph_id, block_id)`` pairs, de-duplicated within a graph (one descriptor per
        distinct rdim per original graph).

        Rolled ``ReductionLoopGraphInfo`` subgraphs are EXCLUDED: they are copies the roller
        creates for one config, so their ``graph_id`` is contingent on the autotuner flipping a
        ``reduction_loops`` knob. The original graphs carry the faithful co-residency structure
        (PROMPT §2.2 / §2.7): two distinct rdims sharing an original graph were fused by the
        compiler -> co-resident; the same rdim in two original graphs is two sequential passes
        (jsd's two V-reductions). Verified against the IR (``_lab/redesign/ir_*.json``): rolled
        subgraphs are exactly ``ReductionLoopGraphInfo``.
        """
        from .inductor_lowering import ReductionLowering

        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, int]] = []
        for gi in self.graphs:
            if isinstance(gi, ReductionLoopGraphInfo):
                continue
            for node in gi.graph.nodes:
                lowering = node.meta.get("lowering")
                if not isinstance(lowering, ReductionLowering):
                    continue
                bid = getattr(lowering, "block_index", None)
                if bid is None:
                    continue
                key = (gi.graph_id, bid)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

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

    def _group_live_tiles(
        self, group_gids: list[int], env: CompileEnvironment
    ) -> dict[int, list[tuple[int | None, ...]]]:
        """Attribute the resident tile set to each co-residency group. Returns ``{group_gid:
        [dim_block_ids, ...]}`` — the peak live set (``_graph_peak_live_tiles``) of the group's home
        graph, combined with the for-loop bodies the group drives. Four attribution rules:

        1. **Descend driven for-loop bodies to the ancestor group.** The heavy tiles often live in a
           for-loop body the group's home graph drives (via a ``loop->child`` edge), not in the thin
           home graph itself. A loop body is owned by the group whose reduction loops over its axis
           (the loop's ``block_ids`` intersect the group's reduction axes).

        2. **Skip ``ReductionLoopGraphInfo`` copies.** The roller's per-config duplicates carry a
           strict subset of the original graph's pre-roll body, so the original graph the group is
           keyed on already holds the peak.

        3. **Do not cross ``_if`` edges.** If/Else branch subtrees are their own groups or mutually
           exclusive with a sibling; a parent group never absorbs a branch's tiles. Branches combine
           via rule 4's max, not by summing into an ancestor.

        4. **Combine a group's owned graphs by max-by-profile, never sum.** Co-resident reductions
           share one original graph by construction, so their simultaneous residency is already
           inside that graph's peak snapshot. A group's other owned graphs are only ever nested
           (home + driven body) or If/Else siblings (max picks the populated branch), never
           simultaneously-additively resident, so max is both faithful and conservative.
        """
        from .inductor_lowering import ReductionLowering

        def reds_in(gid: int) -> set[int]:
            out: set[int] = set()
            for node in self.graphs[gid].graph.nodes:
                low = node.meta.get("lowering")
                if isinstance(low, ReductionLowering) and isinstance(
                    getattr(low, "block_index", None), int
                ):
                    out.add(low.block_index)
            return out

        # A group is keyed on its home (original) graph_id; its reduction axes come from that graph.
        group_axes = {gid: reds_in(gid) for gid in group_gids}

        # Peak live-tile snapshot per graph (skip rolled copies — rule 2).
        peak_of: dict[int, list[tuple[int | None, ...]]] = {}
        for gi in self.graphs:
            if isinstance(gi, ReductionLoopGraphInfo):
                continue
            peak_of[gi.graph_id] = _graph_peak_live_tiles(gi.graph, env)

        # For-loop child edges (loop->body), keyed by the loop's block_ids (rule 1). ``_if`` edges are
        # deliberately NOT followed (rule 3) — see the helper.
        child_loops = self._forloop_child_edges()

        def max_by_profile(
            a: list[tuple[int | None, ...]], b: list[tuple[int | None, ...]]
        ) -> list[tuple[int | None, ...]]:
            mr = max(
                (_tile_rank(t) for t in a + b),
                default=0,
            )
            ka = _tile_set_rank_profile(a, mr)
            kb = _tile_set_rank_profile(b, mr)
            return a if ka >= kb else b

        group_key_set = set(group_gids)
        result: dict[int, list[tuple[int | None, ...]]] = {}
        for gid in group_gids:
            axes = group_axes[gid]
            tiles = list(peak_of.get(gid, []))
            # Descend, transitively, the for-loop bodies this group drives (rule 1): a body whose
            # loop axis is one of the group's reduction axes. Combine each by max-by-profile. Stop
            # at any body that is itself another group's home (its own sequential group) and never
            # cross ``_if`` (branch subtrees are handled by their own group keys — rule 3). A BFS so
            # a home -> loop -> loop chain is fully covered, not just one level. ``not axes`` cannot
            # happen for a real group key (its home graph holds >=1 reduction lowering).
            seen_bodies: set[int] = {gid}
            frontier = [gid]
            while frontier:
                cur = frontier.pop()
                for body_gid, blk in child_loops.get(cur, []):
                    if body_gid in seen_bodies or body_gid in group_key_set:
                        continue
                    if not axes or (blk & axes):
                        seen_bodies.add(body_gid)
                        tiles = max_by_profile(tiles, peak_of.get(body_gid, []))
                        frontier.append(body_gid)
            result[gid] = tiles
        return result

    def _forloop_child_edges(self) -> dict[int, list[tuple[int, frozenset[int]]]]:
        """The for-loop child-edge map for ``_group_live_tiles`` rule 1: for each non-rolled graph,
        the ``(body_graph_id, frozenset(body_block_ids))`` of every for-loop it drives. Keyed by the
        parent graph_id; ``child_loops[gid] = [(body_gid, block_ids), ...]``.

        A pure node scan over ``self.graphs`` (the SAME scan the CF walker uses): follows only
        ``is_for_loop_target`` call_functions whose first arg is the body graph id. ``_if`` edges are
        deliberately NOT collected (rule 3 — branch subtrees are their own groups). Rolled
        (``ReductionLoopGraphInfo``) parents and bodies are skipped (rule 2)."""
        n = len(self.graphs)
        child_loops: dict[int, list[tuple[int, frozenset[int]]]] = {}
        for gi in self.graphs:
            if isinstance(gi, ReductionLoopGraphInfo):
                continue
            edges: list[tuple[int, frozenset[int]]] = []
            for node in gi.graph.nodes:
                if node.op != "call_function":
                    continue
                if (
                    _tracing_ops.is_for_loop_target(node.target)
                    and node.args
                    and isinstance(node.args[0], int)
                ):
                    body_gid = node.args[0]
                    if 0 <= body_gid < n and not isinstance(
                        self.graphs[body_gid], ReductionLoopGraphInfo
                    ):
                        blk = frozenset(
                            getattr(self.graphs[body_gid], "block_ids", []) or []
                        )
                        edges.append((body_gid, blk))
            if edges:
                child_loops[gi.graph_id] = edges
        return child_loops

    def build_reduction_kernel_fact(
        self,
        memory_op_facts: list[MemoryOpFact],
        accumulator_facts: list[AccumulatorFact],
        liveness_by_axis: dict[int, int] | None = None,
    ) -> None:
        """Build the categorizing ``ReductionKernelFact`` — the list of reduction descriptors +
        their ``graph_id`` co-residency groups + the non-reduction loops + the parallel grid axes.
        The reduction seed + the Stage-2 allocator consume this fact directly.
        """
        env = CompileEnvironment.current()
        spec = env.config_spec
        liveness_by_axis = liveness_by_axis or {}
        grid_ids = {b for bids in self.grid_block_ids for b in bids}

        occurrences = self._original_graph_reductions()
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
            per = self._per_reduction_memory_fields(
                bid, gid, memory_op_facts, liveness_by_axis
            )
            descriptors.append(
                ReductionDescriptor(
                    category=category,
                    block_id=bid,
                    graph_id=gid,
                    size_hint=size_hint,
                    itemsize=self._reduction_input_itemsize(bid),
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
        # The try/except is NECESSARY: ``_group_live_tiles`` walks the graphs through ``env`` (block
        # id resolution in ``_graph_peak_live_tiles``), which a bare-spec unit test builds the fact
        # WITHOUT -> the walk raises before producing tiles. The fallback degrades to empty tiles
        # (the Stage-2 allocator then falls back to its extent-based footprint), so a missing env
        # yields a valid-but-coarser fact rather than a crash. Trade-off to be aware of: the broad
        # catch also swallows a genuine graph-structure bug INSIDE the walk as a silent empty-tiles
        # perf regression (not a crash) — acceptable because the walk is best-effort attribution, but
        # if this ever needs tightening, catch ``NoCurrentEnvironment`` specifically (the real no-env
        # trigger) rather than the broad trio.
        try:
            group_live = self._group_live_tiles(sorted(groups_by_gid), env)
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
        liveness_by_axis: dict[int, int],
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

    def build_pointwise_facts(self) -> None:
        """Phase 5: record one ``PointwiseElementwiseFact`` for a PURE elementwise kernel.
        Disjointness rule: if any reduction / matmul / accumulator fact fired, the kernel belongs
        to that family and gets NO pointwise fact (the fact's meaning IS their absence). Derived
        from the walker ``memory_op_facts`` + block-size specs, plus one graph walk for the widest
        compute dtype and the SFU op count. Runs last, after the other facts exist."""
        env = CompileEnvironment.current()
        spec = env.config_spec
        from ..autotuner.config_spec import SIZED_REDUCTION_CATEGORIES
        from ..autotuner.config_spec import PointwiseElementwiseFact
        from ..language.atomic_ops import ATOMIC_OPS
        from ..language.inline_asm_ops import inline_asm_elementwise
        from ..language.inline_triton_ops import inline_triton
        from ..language.inline_triton_ops import triton_kernel
        from ..language.reduce_ops import _reduce
        from ..language.scan_ops import _associative_scan

        # Disjointness: a kernel with any SIZED reduction (the Stage-1 kernel fact's
        # FULL_SLICE/FULL_GRID/USER_TILE descriptors — the faithful successor to a non-empty
        # ``reduction_facts`` list), any matmul, or any loop-carried accumulator belongs to that
        # family and gets NO pointwise fact.
        kf = spec.reduction_kernel_fact
        has_sized_reduction = kf is not None and any(
            d.category in SIZED_REDUCTION_CATEGORIES for d in kf.reductions
        )
        if has_sized_reduction or spec.matmul_facts or spec.accumulator_facts:
            return
        if not spec.memory_op_facts or not spec.block_sizes:
            return
        # Ops whose result is NOT an embarrassingly-parallel function of the tile, so the pointwise
        # seed's core assumption (any tiling is equally correct) does not hold:
        #  - a scan/reduce HOP carries a cross-element dependency along the tiled axis (shrinking it
        #    splits the op);
        #  - an opaque triton/asm body cannot be traced to prove tile-independence;
        #  - an atomic is cross-program communication, not the bandwidth-bound elementwise workload
        #    this seed models and was tuned on (and order-sensitive ones like cas/xchg additionally
        #    shift with the tile). Exclude the whole atomic family rather than draw a fragile
        #    commutativity line.
        # None of these produce a reduction/matmul/accumulator fact, so without this guard they would
        # fall through to a promoted large tile and silently change results. Exclude them like the
        # other non-pointwise families above. (``_associative_scan`` is the single HOP that
        # associative_scan/cumsum/cumprod all lower to.)
        _unsafe_pointwise_ops = frozenset(
            {
                _associative_scan,
                _reduce,
                inline_triton,
                triton_kernel,
                inline_asm_elementwise,
                *ATOMIC_OPS,
            }
        )
        # The static problem element count = product of the tiled block dims' size_hints (the
        # per-dim extents the seed needs for the tile distribution are read live off block_sizes).
        total_numel = 1
        for block_size in spec.block_sizes:
            total_numel *= block_size.size_hint
        # Guard a 0-extent (empty) problem: a 0 hint would make the slab fold below divide by zero.
        total_numel = max(1, total_numel)
        # Widest COMPUTE dtype (register-resident width): a memory op knows only its STORAGE dtype,
        # but the math promotes to fp32, which sets register pressure. Max itemsize over every
        # tensor-valued node in one graph walk (a hl.split val is a tuple of tensors, so recurse).

        def _val_itemsizes(val: object) -> list[int]:
            if isinstance(val, torch.Tensor):
                return [val.dtype.itemsize]
            if isinstance(val, (list, tuple)):
                return [s for v in val for s in _val_itemsizes(v)]
            return []

        # SFU (transcendental) op count — the num_warps signal, counted in the same graph walk. SFU
        # ops run on a distinct, low-throughput unit (~4 SFUs vs ~128 FP32 lanes/SM), so a
        # transcendental-heavy tile is latency-bound and wants more warps; an all-FMA tile of the same
        # op count is not. That is why SFU count, not total op count, is the key.
        _SFU_OPS = frozenset(
            {
                "sin",
                "cos",
                "tan",
                "tanh",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "atanh",
                "asinh",
                "acosh",
                "exp",
                "exp2",
                "expm1",
                "log",
                "log2",
                "log10",
                "log1p",
                "sqrt",
                "rsqrt",
                "sigmoid",
                "erf",
                "erfc",
                "pow",
                "reciprocal",
            }
        )
        compute_itemsize = 1
        sfu_ops = 0
        for graph_info in self.graphs:
            for node in graph_info.graph.nodes:
                for size in _val_itemsizes(node.meta.get("val")):
                    compute_itemsize = max(compute_itemsize, size)
                if node.op == "call_function":
                    if node.target in _unsafe_pointwise_ops:
                        # Not tile-independent (see the set's comment above). Record no fact so the
                        # seed does not fire and the base default (bs=32) is kept — the same no-fire
                        # path the reduction/matmul/accumulator families take at the disjointness
                        # gate above.
                        return
                    base = getattr(node.target, "__name__", str(node.target)).split(
                        "."
                    )[0]
                    if base in _SFU_OPS:
                        sfu_ops += 1

        # slab_numel = sum over FULL-EXTENT load/store ops (accessed_numel >= total_numel) of the
        # untiled elements each drags per tiled element, accessed_numel // total_numel. Flat kernel:
        # every quotient is 1 (slab_numel = op count); rope: heads*head_dim. The seed scales it by
        # storage_itemsize (max storage width) for bandwidth and compute_itemsize (fp32) for registers.
        # A BROADCAST operand (bias[N], [M,1], stride-0) has accessed_numel < total_numel → excluded.
        # contig = TILED block-ids that are the stride-1 axis of some full-extent op (from each op's
        # subscript_strides, no graph walk): a coalesced tile must give width to a stride-1 axis. Row-
        # major → the last block dim (byte-identical); transposed/strided → a different dim (or >1, a
        # load-vs-store conflict) that the seed roots the wide tile on instead of pinning to width 1.
        tiled_ids = {bs.block_id for bs in spec.block_sizes}
        contig: set[int] = set()
        slab_numel = 0
        storage_itemsize = 1
        for memfact in spec.memory_op_facts:
            if memfact.dtype is None or memfact.accessed_numel < total_numel:
                continue
            slab_numel += memfact.accessed_numel // total_numel
            storage_itemsize = max(storage_itemsize, memfact.dtype.itemsize)
            for bid, stride in zip(
                memfact.subscript_block_ids, memfact.subscript_strides, strict=True
            ):
                if bid is not None and stride == 1 and bid in tiled_ids:
                    contig.add(bid)
        spec.pointwise_facts.append(
            PointwiseElementwiseFact(
                total_numel=total_numel,
                slab_numel=slab_numel,
                storage_itemsize=storage_itemsize,
                compute_itemsize=compute_itemsize,
                contig_block_ids=tuple(sorted(contig)),
                sfu_ops=sfu_ops,
            )
        )

    def build_accumulator_facts(self) -> list[AccumulatorFact]:
        """One ``AccumulatorFact`` per loop-carried tensor accumulator in any loop —
        reduction-AGNOSTIC, so it is built independently of (and before) the reduction
        facts: a standalone fact layer any heuristic can read off
        ``ConfigSpec.accumulator_facts``. Run unconditionally (matmul/elementwise kernels
        simply get a list with no rdim-shaped tile). ``ReductionLoopGraphInfo`` is a
        ``ForLoopGraphInfo`` subclass, so this also reaches standard's rolled loops (whose
        ``[M_BLOCK]`` scalar carry yields ``num_carried_2d_tiles == 0`` naturally). Must
        run after the reduction rolling so the rolled subgraphs exist.
        """
        from ..autotuner.config_spec import AccumulatorFact

        facts: list[AccumulatorFact] = []
        for gi in self.graphs:
            if not isinstance(gi, ForLoopGraphInfo):
                continue
            for outer_node in gi.node_args:
                val = getattr(outer_node, "meta", {}).get("val")
                if isinstance(val, torch.Tensor):
                    facts.append(
                        AccumulatorFact(
                            dim_block_ids=tuple(
                                self._resolve_accumulator_dim_block_id(s)
                                for s in val.shape
                            ),
                            itemsize=val.element_size(),
                        )
                    )
        return facts

    def _resolve_accumulator_dim_block_id(self, size: object) -> int | None:
        """Resolve an accumulator dim to its block id, recovering the non-pow2 padded
        case ``resolve_block_id`` misses. A grad-parameter buffer is padded to the
        next power of two, so its extent matches neither the block's registered size
        nor any block-size origin and ``resolve_block_id`` returns ``None``. Fall back
        to matching the padded extent against a reduction block whose
        ``next_power_of_2(size_hint)`` equals it -- but ONLY when UNIQUE. Two axes
        padding to the same extent are indistinguishable, so DECLINE (return ``None``)
        rather than mis-assign an identity the ``num_carried_2d_tiles`` and
        ``per_feature_accumulator`` consumers would read.
        """
        from .._utils import next_power_of_2

        env = CompileEnvironment.current()
        block_id = env.resolve_block_id(size)
        if block_id is not None:
            return block_id
        extent = env.size_hint(cast("int | torch.SymInt", size))
        matches = [
            info.block_id
            for info in env.block_sizes
            if info.reduction and next_power_of_2(info.size_hint()) == extent
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            log.warning(
                "accumulator dim (padded extent %s) matches %d reduction axes %s; cannot "
                "differentiate by extent -- left unresolved (per_feature_accumulator may miss it)",
                extent,
                len(matches),
                matches,
            )
        return None

    def _reduction_input_itemsize(self, red_block_id: int) -> int:
        """Element size (bytes) of the tensor reduced over ``red_block_id`` — read from the
        reduction node's INPUT (not its ``meta['val']`` output, which may differ in dtype,
        e.g. argmax). Helion fp32-promotes the norm/softmax family, so this is 4 at both
        bf16 and fp32 there. All reductions over one rdim share an input element size (last
        wins). The byte caps key on ``size_hint * itemsize``.
        """
        from .inductor_lowering import ReductionLowering

        itemsize = 0
        for graph_info in self.graphs:
            for node in graph_info.graph.nodes:
                lowering = node.meta.get("lowering")
                if (
                    isinstance(lowering, ReductionLowering)
                    and getattr(lowering, "block_index", None) == red_block_id
                ):
                    for inp in node.all_input_nodes:
                        in_val = inp.meta.get("val")
                        if isinstance(in_val, torch.Tensor):
                            itemsize = in_val.element_size()
                            break
        return itemsize

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
        assert type_info is not None and type_info.origin.is_host()
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

        # Convert slice to hl.arange when step is None or 1 and we have both bounds
        # This allows FX tracing to handle slice operations with dynamic bounds
        if lower is not None and upper is not None and (step is None or step == 1):
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


class WalkHostAST(NodeVisitor):
    def __init__(self, device_ir: DeviceIR) -> None:
        super().__init__()
        self.device_ir = device_ir
        self.root_index = 0
        self.current_phase_roots: list[int] = []
        self.phases: list[KernelPhase] = []
        self.root_nodes: list[ast.For] = []

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
            self.phases.append(
                KernelPhase(
                    roots=self.current_phase_roots,
                    root_nodes=[self.root_nodes[r] for r in self.current_phase_roots],
                )
            )
            self.current_phase_roots = []
            return
        self.generic_visit(node)

    def flush_phases(self) -> None:
        if self.current_phase_roots:
            self.phases.append(
                KernelPhase(
                    roots=self.current_phase_roots,
                    root_nodes=[self.root_nodes[r] for r in self.current_phase_roots],
                )
            )
            self.current_phase_roots = []


# Matmul/dot FX targets mapped to (lhs_arg_index, rhs_arg_index). Used to tag
# which load nodes feed a matmul (and as which operand) at the recognition site,
# instead of scanning every load's downstream users.
def _matmul_operand_positions() -> dict[object, tuple[int, int]]:
    from ..language import matmul_ops

    return {
        # mat1=0, mat1_scale=1, mat1_format=2, mat2=3 -> lhs/rhs matrices are 0/3
        matmul_ops.dot_scaled: (0, 3),
        matmul_ops.dot: (0, 1),
        torch.ops.aten.mm.default: (0, 1),
        torch.ops.aten.bmm.default: (0, 1),
        torch.ops.aten.addmm.default: (1, 2),
        torch.ops.aten.baddbmm.default: (1, 2),
    }


def _trace_back_to_load(arg: object, load_op: object) -> torch.fx.Node | None:
    """Follow a matmul operand back through pass-through ops to its load node.

    Only follows single-input pass-through ops (cast / transpose / view / unary
    elementwise), so an operand that is a genuine computation of two loads (e.g.
    ``a[...] + bias[...]``) is left untagged rather than mis-attributed to its
    first input. Returns the producing ``hl.load`` node, or ``None``.
    """
    cur = arg
    for _ in range(8):
        if not isinstance(cur, torch.fx.Node):
            return None
        if cur.target is load_op:
            return cur
        tensor_inputs = [
            a
            for a in cur.args
            if isinstance(a, torch.fx.Node)
            and isinstance(a.meta.get("val"), torch.Tensor)
        ]
        if len(tensor_inputs) != 1:
            return None
        cur = tensor_inputs[0]
    return None


def _load_needs_eviction_tunable(node: torch.fx.Node) -> bool:
    """A load gets an eviction-policy slot only when the user did not pass one."""
    eviction_policy_arg = node.kwargs.get("eviction_policy")
    if eviction_policy_arg is None and len(node.args) >= 4:
        eviction_policy_arg = node.args[3]
    return eviction_policy_arg is None


def _accessed_tensor_fake(node: torch.fx.Node) -> torch.Tensor | None:
    """Fake tensor of the buffer a load/store accesses (``args[0]``)."""
    arg = node.args[0] if node.args else None
    if isinstance(arg, torch.fx.Node):
        val = arg.meta.get("val")
        if isinstance(val, torch.Tensor):
            return val
    return None


def _subscript_block_id(env: CompileEnvironment, sub: object) -> int | None:
    """Block-id a subscript expression INDEXES, from the index node's own provenance (tile/offset
    meta, else its block-var SymInt). Reduction-AGNOSTIC (resolves a tile var regardless of the
    block's ``reduction`` flag, unlike a shape-size resolve). ``None`` for a plain slice / scalar."""
    if not isinstance(sub, torch.fx.Node):
        return None
    two = sub.meta.get("tile_with_offset")
    if isinstance(two, dict) and two.get("block_id") is not None:
        return two["block_id"]
    val = sub.meta.get("val")
    if val is not None:
        return env.resolve_block_id(val)
    return None


def _tile_rank(dims: tuple[int | None, ...]) -> int:
    """The number of BLOCK dims a tile spans (``None`` static/broadcast dims do not count) — the
    block-size-free proxy for tile bytes used by the lexicographic peak selector."""
    return sum(1 for d in dims if d is not None)


def _tile_set_rank_profile(
    tiles: list[tuple[int | None, ...]], max_rank: int
) -> tuple[int, ...]:
    """A tile set's RANK PROFILE as a lexicographic key over ABSOLUTE ranks, HIGHEST rank first:
    index 0 is the count of rank-``max_rank`` tiles, ..., last is the count of rank-1 tiles. More
    top-rank tiles wins; equal-top-rank falls through to the next rank down. Fixed length so the
    same rank aligns when comparing two tile sets (a per-set-length key would compare one set's
    rank-2 count against another's rank-1 count and mis-rank a scalar swarm as the winner)."""
    by_rank: dict[int, int] = {}
    for t in tiles:
        r = _tile_rank(t)
        if r:
            by_rank[r] = by_rank.get(r, 0) + 1
    return tuple(by_rank.get(k, 0) for k in range(max_rank, 0, -1))


def _graph_peak_live_by_axis(
    graph: torch.fx.Graph, env: CompileEnvironment
) -> dict[int, int]:
    """Peak count of simultaneously-live tensor values whose shape spans each
    block-id axis, over ONE graph -- a liveness sweep in FX topo order.
    Consumer-AGNOSTIC per-axis provenance keyed by block_id; a consumer reads its
    own axis slice.

    A value is live from definition through its last use in the graph. At each
    step the sweep counts live values whose ``meta['val'].shape`` includes an axis
    and tracks the per-axis peak. A CONSERVATIVE over-count of register pressure
    (no remat / reuse beyond last-use modeled), so a heavy body is never
    under-counted into an unsafe persistent spill.
    """
    nodes = list(graph.nodes)
    last_use: dict[torch.fx.Node, int] = {}
    for i, node in enumerate(nodes):
        for inp in node.all_input_nodes:
            last_use[inp] = i
    axes_of: dict[torch.fx.Node, frozenset[int]] = {}
    for node in nodes:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            axes = {
                bid for s in val.shape if (bid := env.resolve_block_id(s)) is not None
            }
            if axes:
                axes_of[node] = frozenset(axes)
    peak: dict[int, int] = {}
    live: set[torch.fx.Node] = set()
    for i, node in enumerate(nodes):
        if node in axes_of:
            live.add(node)
        per_axis: dict[int, int] = {}
        for v in live:
            for a in axes_of[v]:
                per_axis[a] = per_axis.get(a, 0) + 1
        for a, c in per_axis.items():
            if c > peak.get(a, 0):
                peak[a] = c
        live = {v for v in live if last_use.get(v, -1) > i}
    return peak


def _graph_peak_live_tiles(
    graph: torch.fx.Graph, env: CompileEnvironment
) -> list[tuple[int | None, ...]]:
    """The per-tile shapes of the simultaneously-live tensors at the graph's peak-live step.

    Each live tile is recorded as its ``dim_block_ids`` tuple (one entry per shape dim: the block
    id it spans, or ``None`` for a non-block/constant dim), the same representation as
    ``AccumulatorFact.dim_block_ids``, so a footprint can sum ``∏(dims)`` per actual tile shape
    rather than assuming every live tile is full-rank over all axes.

    Which step is "peak"? Register/SRAM pressure is maximized at the step holding the most bytes,
    but tile bytes depend on the not-yet-chosen block sizes. Rank is the block-size-free proxy for
    "big": a rank-2 ``[m, r]`` tile is ``m_block × r_block`` elements vs ``m_block`` for a rank-1
    ``[m]``, so one higher-rank tile outweighs any number of lower-rank tiles. We pick the step
    whose live set is lexicographically greatest by rank profile — the tuple ``(#rank-D tiles,
    #rank-(D-1) tiles, ..., #rank-1 tiles)`` compared most-dims-first — so a step with one ``[m, r]``
    beats a step with many scalar ``[m]`` carries. Maximizing the top-rank count first captures the
    most heavy R-scaling tiles (the term that prevents spills); lower-rank ties break to more
    next-heaviest, then to the first such step.

    Returns the actual co-resident live set at that one real step (never a per-shape union across
    steps, which would stitch together shapes that never coexist). The "1 higher-rank beats any
    number of lower-rank" tie-rule is not byte-exact in a pathological extreme (hundreds of rank-n
    tiles vs one rank-(n+1)), but a higher-rank tile contains a lower-rank one as a factor and the
    miss errs toward a slightly larger R (bounded by the persistence/chunk math) — a stated
    heuristic, not a random choice.
    """
    nodes = list(graph.nodes)
    last_use: dict[torch.fx.Node, int] = {}
    for i, node in enumerate(nodes):
        for inp in node.all_input_nodes:
            last_use[inp] = i
    shape_of: dict[torch.fx.Node, tuple[int | None, ...]] = {}
    for node in nodes:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor) and val.shape:
            dims = tuple(env.resolve_block_id(s) for s in val.shape)
            # only tiles that span at least one block axis are register-resident reduction tiles
            if any(d is not None for d in dims):
                shape_of[node] = dims

    # Graph-wide max rank -> a FIXED-length, absolute-rank-indexed profile so the lexicographic
    # comparison aligns the same rank across steps (a per-step-length key would compare a step's
    # rank-2 count against another's rank-1 count and mis-rank the scalar swarm as the winner).
    max_rank = max((_tile_rank(d) for d in shape_of.values()), default=0)

    best_key: tuple[int, ...] = ()
    best_tiles: list[tuple[int | None, ...]] = []
    live: set[torch.fx.Node] = set()
    for i, node in enumerate(nodes):
        if node in shape_of:
            live.add(node)
        cur = [shape_of[v] for v in live]
        key = _tile_set_rank_profile(cur, max_rank)
        if key > best_key:
            best_key = key
            best_tiles = cur
        live = {v for v in live if last_use.get(v, -1) > i}
    return best_tiles


def _collect_memory_op_facts(
    device_ir: DeviceIR,
) -> tuple[list[MemoryOpFact], dict[int, int]]:
    """Walk every device graph once and record per-load/store metadata.

    Produces one ``MemoryOpFact`` per load/store in the order used to size
    ``Config.indexing``, so ``memory_op_facts[i]`` describes ``config.indexing[i]``.
    Single source of truth for load/store counts, eviction slots, and
    ``store_indices`` (all derived from the result).

    Per-op enrichment fields (``reductions_fed``/``stores_fed``/
    ``indexed_block_ids``/``inner_extent``/``graph_id``) are computed here so the
    reduction-fact builders need no bespoke walk. ``reductions_fed`` runs
    ``_classify_load_dataflow`` once over ALL reduction axes, grouped by axis.

    Returns ``(memory_op_facts, liveness_by_axis)`` -- the second is the per-axis
    peak simultaneously-live rdim-shaped tile count (max over graphs), computed in
    this SAME pass so the kernel-fact builder can read a per-axis slice.
    """
    from ..autotuner.config_spec import MemoryOpFact
    from ..language import memory_ops
    from .inductor_lowering import ReductionLowering

    load_op = memory_ops.load
    store_op = memory_ops.store
    operand_positions = _matmul_operand_positions()

    env = CompileEnvironment.current()
    host = HostFunction.current()
    # Matmul operands always precede their matmul in graph order, so `operands` is
    # complete by the time we apply it to the (operand-less) facts below.
    operands: dict[torch.fx.Node, str] = {}
    records: list[tuple[torch.fx.Node, MemoryOpFact]] = []
    memory_op_index = 0
    eviction_index = 0
    liveness_by_axis: dict[int, int] = {}

    for graph_info in device_ir.graphs:
        graph = graph_info.graph
        # Reduction-body liveness (per-axis peak live tiles), merged max over graphs — part of
        # this single collect pass, not a second traversal.
        for axis, peak in _graph_peak_live_by_axis(graph, env).items():
            if peak > liveness_by_axis.get(axis, 0):
                liveness_by_axis[axis] = peak
        # Axis (block_index) of every ReductionLowering node in this graph — the cut set
        # for the dataflow classification, so a fed reduction maps to its axis.
        red_axis_by_id: dict[int, int] = {
            id(node): node.meta["lowering"].block_index
            for node in graph.nodes
            if isinstance(node.meta.get("lowering"), ReductionLowering)
            and isinstance(getattr(node.meta["lowering"], "block_index", None), int)
        }
        redset = set(red_axis_by_id)
        for node in graph.nodes:
            if node.op != "call_function":
                continue

            positions = operand_positions.get(node.target)
            if positions is not None:
                for arg_index, operand in (
                    (positions[0], "lhs"),
                    (positions[1], "rhs"),
                ):
                    if arg_index < len(node.args):
                        load = _trace_back_to_load(node.args[arg_index], load_op)
                        if load is not None:
                            operands.setdefault(load, operand)
                continue

            is_load = node.target is load_op
            if not (is_load or node.target is store_op):
                continue

            this_eviction_index: int | None = None
            if is_load and _load_needs_eviction_tunable(node):
                this_eviction_index = eviction_index
                eviction_index += 1

            fake = _accessed_tensor_fake(node)
            origin = host.tensor_to_origin.get(fake) if fake is not None else None

            # reductions_fed (loads): per-axis count of the reductions this load's value
            # flows into; stores_fed: the stores it reaches without passing a reduction,
            # keyed by full axis tuple. Both reduction-AGNOSTIC (all axes at once).
            reductions_fed: tuple[tuple[int, int], ...] = ()
            stores_fed: tuple[tuple[int | None, ...], ...] = ()
            if is_load:
                feeds, stores_fed = _classify_load_dataflow(node, redset, env)
                per_axis: dict[int, int] = {}
                for fed_id in feeds:
                    axis = red_axis_by_id[fed_id]
                    per_axis[axis] = per_axis.get(axis, 0) + 1
                reductions_fed = tuple(sorted(per_axis.items()))

            # indexed_block_ids: shape-resolved block-id per non-bare-int subscript (None
            # where unresolvable; the fallback). subscript_block_ids: the index-subscript
            # block-id (the faithful axis key). inner_extent: inner-dim extent.
            indexed_block_ids: tuple[int | None, ...] = ()
            subscript_block_ids: tuple[int | None, ...] = ()
            subscript_strides: tuple[int, ...] = ()
            inner_extent: int | None = None
            accessed_numel = 0
            if fake is not None:
                # Distinct HBM elements the op touches = product of size-hinted shape dims over
                # NON-broadcast dims (stride != 0). A stride-0 dim (an .expand()/broadcast — full
                # SIZE but one underlying element, e.g. bias[N].expand_as(x) or a broadcast_tensors
                # operand) contributes factor 1, so an expanded broadcast is NOT mistaken for
                # full-extent traffic. size_hint (not int()) avoids specializing a SymInt dim. A
                # broadcast operand thus has a strictly smaller numel than the problem numel — the
                # faithful full-extent signal at any rank/stride (see MemoryOpFact.accessed_numel).
                accessed_numel = 1
                fake_strides = fake.stride()
                for dim_index, dim_size in enumerate(fake.shape):
                    if fake_strides[dim_index] != 0:
                        accessed_numel *= env.size_hint(dim_size)
                index_list = node.args[1] if len(node.args) >= 2 else None
                if isinstance(index_list, (list, tuple)):
                    # same positions for both tuples so [-1] aligns (drop bare-int / OOB)
                    positions = [
                        pos
                        for pos, sub in enumerate(index_list)
                        if not isinstance(sub, int) and pos < fake.ndim
                    ]
                    indexed_block_ids = tuple(
                        env.resolve_block_id(fake.shape[pos]) for pos in positions
                    )
                    subscript_block_ids = tuple(
                        _subscript_block_id(env, index_list[pos]) for pos in positions
                    )
                    # Element stride of the accessed tensor along each subscripted axis, aligned
                    # 1:1 with subscript_block_ids (same `positions`). Raw provenance from the fake
                    # tensor's .stride(); a stride-1 position is the contiguous/coalescing axis (the
                    # last subscript for a row-major tensor, a DIFFERENT one for a transposed view).
                    # size_hint (not int()) avoids specializing a SymInt stride to a constant: a
                    # contiguous axis is a concrete 1, and a symbolic (outer) stride hints > 1 anyway,
                    # so the stride==1 contiguity test is unaffected while dynamic shapes stay dynamic.
                    subscript_strides = tuple(
                        env.size_hint(fake_strides[pos]) for pos in positions
                    )
                if fake.ndim >= 2:
                    # size_hint (not int()): int() would guard a SymInt inner dim to a
                    # constant, over-specializing dynamic-shape kernels. inner_extent is a
                    # heuristic signal, so a hint is sufficient and must not specialize.
                    inner_extent = env.size_hint(fake.shape[-1])

            records.append(
                (
                    node,
                    MemoryOpFact(
                        indexing_index=memory_op_index,
                        kind="load" if is_load else "store",
                        eviction_index=this_eviction_index,
                        tensor_name=origin.root_rw_name() if origin else None,
                        dtype=fake.dtype if fake is not None else None,
                        ndim=fake.ndim if fake is not None else 0,
                        num_reuses=len(node.users) if is_load else 0,
                        matmul_operand=None,
                        graph_id=graph_info.graph_id,
                        reductions_fed=reductions_fed,
                        stores_fed=stores_fed,
                        indexed_block_ids=indexed_block_ids,
                        inner_extent=inner_extent,
                        subscript_block_ids=subscript_block_ids,
                        subscript_strides=subscript_strides,
                        accessed_numel=accessed_numel,
                    ),
                )
            )
            memory_op_index += 1

    facts = [fact._replace(matmul_operand=operands.get(node)) for node, fact in records]
    return facts, liveness_by_axis


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


def lower_to_device_ir(func: HostFunction) -> DeviceIR:
    device_ir = DeviceIR()
    device_ir.host_function = func
    with func, device_ir, compile_lock:
        visitor = WalkHostAST(device_ir)
        for stmt in func.body:
            visitor.visit(stmt)
        visitor.flush_phases()
        device_ir.phases = visitor.phases
        # Run dependency checks once, per phase, so codegen does not redo it per-config.
        for phase in device_ir.phases:
            checker = phase.loop_dependency_checker
            for loop_node in phase.root_nodes:
                checker.register_loop(loop_node)
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
        # in Phase 3 (build_reduction_kernel_fact, after _collect_memory_op_facts) so it can
        # read the enriched memory_op_facts.
        device_ir.register_rollable_reductions()
        if CompileEnvironment.current().backend.name == "cute":
            _register_cute_lane_vector_width_specs(config_spec)
            # Enable the flash-attention autotune surface (Tasks #25 + #28) when
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
                    block_size_targets=flash_shape.block_size_targets,
                    is_causal=flash_shape.is_causal,
                    has_kv_tile_pruning=flash_shape.has_kv_tile_pruning,
                    requires_ws_overlap=flash_shape.requires_ws_overlap,
                    small_biased_candidate=flash_shape.small_biased_candidate,
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

        # Collect per-load/store metadata once so heuristics can map each Config.indexing slot to its
        # graph op; the same pass returns the reduction-body liveness (per-axis peak live tiles).
        memory_op_facts, liveness_by_axis = _collect_memory_op_facts(device_ir)
        config_spec.memory_op_facts = memory_op_facts
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
        config_spec.accumulator_facts = device_ir.build_accumulator_facts()
        # Phase 3 (PROMPT §2.1/§2.2): build the categorizing ReductionKernelFact — the first-class
        # reduction-descriptor list + co-residency groups the reduction seed + allocator consume.
        # liveness_by_axis supplies each descriptor's per-axis liveness slice.
        device_ir.build_reduction_kernel_fact(
            memory_op_facts, config_spec.accumulator_facts, liveness_by_axis
        )
        # Phase 4: compose a matmul + reduction-over-output epilogue fact when a matmul AND a
        # register-resident epilogue reduction co-occur (matmul_rms_norm etc.).
        device_ir.build_matmul_reduction_epilogue_facts()
        # Phase 5: record a PointwiseElementwiseFact for a PURE elementwise kernel (no
        # reduction/matmul/accumulator fact) so the pointwise seed can size a BW-saturating
        # tile instead of the starved block_size=32 default.
        device_ir.build_pointwise_facts()

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
