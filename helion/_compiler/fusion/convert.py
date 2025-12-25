"""Fusion context and specifications for Helion.

This module handles:
- Fusion context and specifications (EpilogueSpec, PrologueSpec, FusionContext)
- Utility functions for AST transformation
"""
from __future__ import annotations

import ast
import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator

import sympy
import torch

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode


# --- Specifications ---
@dataclass
class EpilogueSpec:
    """Epilogue fusion spec: inline (N=1) or deferred (N>1)."""
    epilogue_nodes: list["BaseSchedulerNode"]
    accumulator_name: str | set[str]
    @property
    def accumulator_names(self) -> set[str]:
        return self.accumulator_name if isinstance(self.accumulator_name, set) else {self.accumulator_name}
    @property
    def is_deferred(self) -> bool: return len(self.accumulator_names) > 1

@dataclass
class PrologueSpec:
    """Prologue fusion spec."""
    prologue_nodes: list["BaseSchedulerNode"]
    input_name: str


# --- Epilogue Tracing Context (for pure tracing approach) ---

@dataclass
class EpilogueOp:
    """Represents a single operation in an epilogue chain.

    Attributes:
        op_fn: The torch operation function (e.g., torch.add, torch.relu)
        is_binary: True if this op takes two inputs (acc + ext_buf or acc + constant), False for unary
        ext_buf: The external buffer for binary ops, None for unary or constant
        ext_buf_name: Name of the external buffer for tracing
        dim_idx: Dimension index for slicing the external buffer
        constant_val: Scalar constant value for binary ops with constants (e.g., x * 2.0)
        dtype_val: dtype for to_dtype operations (e.g., torch.float32)
    """
    op_fn: "Callable[..., torch.Tensor]"
    is_binary: bool
    ext_buf: "torch.Tensor | None"
    ext_buf_name: str
    dim_idx: int
    constant_val: "float | int | None" = None
    dtype_val: "torch.dtype | None" = None


@dataclass
class PrologueInfo:
    """Prologue ops to apply after load (e.g., relu on input before matmul)."""
    ops: list[EpilogueOp]


@dataclass
class EpilogueTracingState:
    """State for epilogue/prologue application during tracing."""
    store_counter: list[int] = field(default_factory=lambda: [0])
    store_map: dict[int, str] = field(default_factory=dict)
    stored_values: dict[str, "torch.Tensor"] = field(default_factory=dict)
    stored_proxies: dict[str, Any] = field(default_factory=dict)
    multi_output_op_fn: "Callable | None" = None
    multi_output_acc_names: list[str] = field(default_factory=list)
    multi_output_epilogue_buffer: "torch.Tensor | None" = None
    epilogue_callables: dict[str, "Callable"] = field(default_factory=dict)
    prologue_info: dict[int, "PrologueInfo"] = field(default_factory=dict)

    def get_and_increment_store_idx(self) -> int:
        idx = self.store_counter[0]; self.store_counter[0] += 1; return idx

    def get_epilogue_callable(self, store_idx: int) -> "Callable | None":
        buf = self.store_map.get(store_idx)
        return self.epilogue_callables.get(buf) if buf else None

    def is_last_store(self, store_idx: int) -> bool:
        return store_idx == max(self.store_map.keys()) if self.store_map else False

    def get_prologue_info_for_tensor(self, tensor_id: int) -> "PrologueInfo | None":
        return self.prologue_info.get(tensor_id)


_epilogue_tracing_ctx: ContextVar[EpilogueTracingState | None] = ContextVar(
    "_epilogue_tracing_ctx", default=None
)


@contextlib.contextmanager
def epilogue_tracing_context(
    store_map: dict[int, str], multi_output_op_fn: "Callable | None" = None,
    multi_output_acc_names: list[str] | None = None, multi_output_epilogue_buffer: "torch.Tensor | None" = None,
    prologue_info: dict[int, "PrologueInfo"] | None = None, epilogue_callables: dict[str, "Callable"] | None = None,
) -> Iterator[EpilogueTracingState]:
    """Context for applying epilogues/prologues during tracing."""
    state = EpilogueTracingState(
        store_map=store_map, multi_output_op_fn=multi_output_op_fn, multi_output_acc_names=multi_output_acc_names or [],
        multi_output_epilogue_buffer=multi_output_epilogue_buffer, prologue_info=prologue_info or {},
        epilogue_callables=epilogue_callables or {},
    )
    token = _epilogue_tracing_ctx.set(state)
    try: yield state
    finally: _epilogue_tracing_ctx.reset(token)

def get_epilogue_tracing_state() -> EpilogueTracingState | None:
    return _epilogue_tracing_ctx.get()


# --- Fusion Error ---

class FusionError(Exception):
    """Raised when fusion cannot be applied."""
    pass


# --- FX to PyTorch Callable Conversion ---

# Helper functions for comparison ops that handle both tensor and scalar arguments
def _make_comparison_op(tensor_op: Callable[..., Any], scalar_op: Callable[..., Any]) -> Callable[..., Any]:
    """Create a comparison function that dispatches to Tensor or Scalar variant."""
    def compare(a: Any, b: Any) -> Any:
        if isinstance(b, (int, float)):
            return scalar_op(a, b)
        return tensor_op(a, b)
    return compare


def _where_op(condition: Any, self_val: Any, other_val: Any) -> Any:
    """Where op that handles scalar arguments."""
    return torch.where(condition, self_val, other_val)


def _get_aten_op_overload(packet: Any) -> Callable[..., Any] | None:
    """Get the best overload from an OpOverloadPacket.

    Returns .Tensor if available (for binary ops), otherwise .default.
    These overloads have _overloadpacket which is required for prologue fusion.
    """
    if hasattr(packet, 'Tensor'):
        return packet.Tensor
    if hasattr(packet, 'default'):
        return packet.default
    return None


# Ops where inductor name differs from aten name
_OPS_NAME_MAPPING: dict[str, str] = {
    "truediv": "div",
    "floordiv": "floor_divide",
    "mod": "remainder",
    "and_": "bitwise_and",
    "or_": "bitwise_or",
    "xor": "bitwise_xor",
}

# Ops that need specific overloads (not .Tensor or .default)
_OPS_SPECIFIC_OVERLOAD: dict[str, Callable[..., Any]] = {
    "pow": torch.ops.aten.pow.Tensor_Tensor,
}

# Ops that need custom handling (lambdas or special functions)
_OPS_CUSTOM: dict[str, Callable[..., Any]] = {
    "lshift": lambda a, b: a << b,
    "rshift": lambda a, b: a >> b,
    "constant": lambda v, dtype=None: v,
    "where": _where_op,
    "to_dtype": lambda x, dtype: x.to(dtype),
    "index_expr": lambda x: x,
}

# Comparison ops need both Tensor and Scalar variants
_OPS_COMPARISON: dict[str, Callable[..., Any]] = {
    "eq": _make_comparison_op(torch.ops.aten.eq.Tensor, torch.ops.aten.eq.Scalar),
    "ne": _make_comparison_op(torch.ops.aten.ne.Tensor, torch.ops.aten.ne.Scalar),
    "lt": _make_comparison_op(torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Scalar),
    "le": _make_comparison_op(torch.ops.aten.le.Tensor, torch.ops.aten.le.Scalar),
    "gt": _make_comparison_op(torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Scalar),
    "ge": _make_comparison_op(torch.ops.aten.ge.Tensor, torch.ops.aten.ge.Scalar),
}


def get_torch_op(op_name: str) -> Callable[..., Any]:
    """Get the PyTorch equivalent of an Inductor op.

    Priority order:
    1. Custom ops (lambdas, special functions)
    2. Specific overloads (e.g., pow.Tensor_Tensor)
    3. Comparison ops (need Tensor/Scalar dispatch)
    4. Name mapping + aten lookup (truediv -> div, etc.)
    5. Direct aten lookup with automatic overload selection
    6. Fallback to torch module functions
    """
    # 1. Custom ops
    if op_name in _OPS_CUSTOM:
        return _OPS_CUSTOM[op_name]

    # 2. Specific overloads
    if op_name in _OPS_SPECIFIC_OVERLOAD:
        return _OPS_SPECIFIC_OVERLOAD[op_name]

    # 3. Comparison ops
    if op_name in _OPS_COMPARISON:
        return _OPS_COMPARISON[op_name]

    # 4. Name mapping
    aten_name = _OPS_NAME_MAPPING.get(op_name, op_name)

    # 5. Direct aten lookup with overload selection
    if hasattr(torch.ops.aten, aten_name):
        packet = getattr(torch.ops.aten, aten_name)
        overload = _get_aten_op_overload(packet)
        if overload is not None:
            return overload
        # If no standard overload, return the packet (it may be callable)
        return packet

    # 6. Fall back to torch module functions
    if hasattr(torch, op_name):
        return getattr(torch, op_name)

    raise ValueError(f"Unknown op: {op_name}")


def _resolve_args(node_args: tuple | list, values: dict[str, Any], default: Any) -> list[Any]:
    """Resolve FX node args to actual values from values dict, or use default."""
    return [values.get(a.name, default) if isinstance(a, torch.fx.Node) else a for a in node_args]


def _extract_buffer_info(
    graph: "torch.fx.Graph", accumulator_names: set[str],
    indexing_exprs: dict[str, "sympy.Expr"] | None = None, iter_vars: list["sympy.Symbol"] | None = None,
) -> tuple[list[str], dict[str, list[int]], dict[str, str]]:
    """Extract external buffer names, dimension indices, and node-to-buffer mapping from FX graph."""
    external_buffers: list[str] = []
    external_buffer_dims: dict[str, list[int]] = {}
    node_to_buffer: dict[str, str] = {}

    for node in graph.find_nodes(op="call_method", target="load"):
        buf_name = node.args[1] if len(node.args) > 1 else None
        if not buf_name: continue
        node_to_buffer[node.name] = buf_name
        if buf_name in accumulator_names or buf_name in external_buffers: continue
        external_buffers.append(buf_name)
        # Extract dimension indices from indexing expression
        if indexing_exprs is not None and iter_vars is not None and len(node.args) > 2:
            idx_node = node.args[2]
            if hasattr(idx_node, 'op') and idx_node.op == 'call_module' and idx_node.target == 'get_index' and idx_node.args:
                idx_expr = indexing_exprs.get(idx_node.args[0])
                if idx_expr is not None and hasattr(idx_expr, 'free_symbols'):
                    dims_used = [i for i, iv in enumerate(iter_vars) if iv in idx_expr.free_symbols]
                    if dims_used: external_buffer_dims[buf_name] = dims_used
    return external_buffers, external_buffer_dims, node_to_buffer


def fx_to_pytorch_epilogue(
    spec: "EpilogueSpec", accumulator_name: str | set[str],
) -> tuple[None, list[str], dict[str, list[int]], list[str], Callable | None, list[Callable]]:
    """Extract buffer info and ops from Inductor FX graph (epilogue nodes).

    Note: The actual epilogue callable is created by create_helion_traceable_epilogue
    during re-tracing. This function only extracts the metadata needed for that.

    Returns:
        Tuple of (None, external_buffers, external_buffer_dims, accumulator_names_list, epilogue_op_fn, epilogue_ops)
        The first element is None (previously was make_fn which is no longer needed).
    """
    from torch._inductor.ir import ComputedBuffer, Pointwise
    from torch._inductor.loop_body import LoopBody
    from torch._inductor import dependencies

    # Normalize accumulator_name to a set
    if isinstance(accumulator_name, str):
        accumulator_names = {accumulator_name}
    else:
        accumulator_names = accumulator_name

    if not spec.epilogue_nodes:
        # Identity epilogue
        return None, [], {}, sorted(accumulator_names), None, []

    # Get the last epilogue node (the one that produces final output)
    last_node = spec.epilogue_nodes[-1].node
    if not (isinstance(last_node, ComputedBuffer) and isinstance(last_node.data, Pointwise)):
        raise FusionError(f"Expected Pointwise epilogue, got {type(last_node)}")

    pw = last_node.data
    (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(pw.ranges, (), prefix="i")

    try:
        lb = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, reduce_vars)
    except Exception as e:
        raise FusionError(f"Failed to create LoopBody: {e}") from e

    # Get indexing expressions from LoopBody
    indexing_exprs = getattr(lb, 'indexing_exprs', {})

    # Extract external buffer info using helper
    external_buffers, external_buffer_dims, _ = _extract_buffer_info(
        lb.root_block.graph, accumulator_names, indexing_exprs, iter_vars
    )

    # Extract epilogue operations - use loop to handle ValueError from get_torch_op
    epilogue_ops: list[Callable] = []
    for node in lb.root_block.graph.nodes:
        if node.op == "call_method" and node.target not in ("load", "store", "constant"):
            try:
                epilogue_ops.append(get_torch_op(node.target))
            except ValueError:
                pass  # Unknown op, skip
    epilogue_op_fn = epilogue_ops[-1] if epilogue_ops else None

    # Determine accumulator order for multi-output epilogues
    accumulator_names_list = sorted(accumulator_names)

    return None, external_buffers, external_buffer_dims, accumulator_names_list, epilogue_op_fn, epilogue_ops


def create_helion_traceable_epilogue(
    spec: "EpilogueSpec",
    accumulator_name: str | set[str],
    ext_buf_fakes: dict[str, torch.Tensor],
    external_buffer_dims: dict[str, list[int]],
) -> Callable[[torch.Tensor, list], torch.Tensor]:
    """Create epilogue callable with hl.load for external buffers (traced in proxy mode)."""
    from torch._inductor.ir import ComputedBuffer, Pointwise
    from torch._inductor.loop_body import LoopBody
    from torch._inductor import dependencies


    # Normalize accumulator_name to a set
    if isinstance(accumulator_name, str):
        accumulator_names = {accumulator_name}
    else:
        accumulator_names = accumulator_name

    if not spec.epilogue_nodes:
        return lambda value, subscript: value  # Identity epilogue

    # Get the last epilogue node
    last_node = spec.epilogue_nodes[-1].node
    if not (isinstance(last_node, ComputedBuffer) and isinstance(last_node.data, Pointwise)):
        raise FusionError(f"Expected Pointwise epilogue, got {type(last_node)}")

    pw = last_node.data
    (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(pw.ranges, (), prefix="i")

    try:
        lb = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, reduce_vars)
    except Exception as e:
        raise FusionError(f"Failed to create LoopBody: {e}") from e

    # Extract external buffer names using helper (dimension info not needed here)
    external_buffers, _, _ = _extract_buffer_info(lb.root_block.graph, accumulator_names)

    def helion_traceable_epilogue(value: torch.Tensor, subscript: list) -> torch.Tensor:
        """Apply epilogue with hl.load for external buffers (traced via ProxyTorchDispatchMode)."""
        from ...language import load as hl_load
        from ..host_function import HostFunction
        from ..variable_origin import ArgumentOrigin

        # Register external buffer fake tensors with HostFunction for proxy creation
        host_fn = HostFunction.current()
        for buf_name, ext_buf in ext_buf_fakes.items():
            if ext_buf not in host_fn.tensor_to_origin:
                host_fn.tensor_to_origin[ext_buf] = ArgumentOrigin(f"_ext_{buf_name}")
        values: dict[str, Any] = {}

        # Process FX graph nodes
        for node in lb.root_block.graph.nodes:
            if node.op == "placeholder":
                continue
            elif node.op == "output":
                if node.args and node.args[0]:
                    out = node.args[0]
                    if isinstance(out, (list, tuple)) and out:
                        out = out[0]
                    if isinstance(out, torch.fx.Node):
                        return values.get(out.name, value)
                return value
            elif node.op == "call_method":
                target = node.target
                node_args = node.args[1:] if len(node.args) > 1 else node.args

                if target == "load":
                    buf_name = node_args[0] if node_args else None
                    if buf_name in accumulator_names:
                        values[node.name] = value
                    elif buf_name in ext_buf_fakes:
                        ext_buf = ext_buf_fakes[buf_name]
                        dim_indices = external_buffer_dims.get(buf_name, [len(subscript) - 1])
                        buf_ndim = len(ext_buf.shape) if hasattr(ext_buf, 'shape') else len(dim_indices)
                        tile_indices = []

                        if buf_ndim == len(dim_indices):
                            for dim_idx in dim_indices:
                                if dim_idx < len(subscript): tile_indices.append(subscript[dim_idx])
                        else:
                            # Buffer has broadcast dims (size 1) - map dim_indices accordingly
                            dim_idx_iter = iter(dim_indices)
                            current_dim_idx = next(dim_idx_iter, None)
                            for buf_dim in range(buf_ndim):
                                buf_size = ext_buf.shape[buf_dim] if hasattr(ext_buf, 'shape') else 1
                                if current_dim_idx is not None and buf_dim < len(dim_indices):
                                    tile_indices.append(subscript[dim_indices[buf_dim]] if dim_indices[buf_dim] < len(subscript) else slice(None))
                                elif buf_size == 1:
                                    tile_indices.append(0)  # Broadcast dim
                                else:
                                    tile_indices.append(slice(None))

                        if tile_indices:
                            loaded = hl_load(ext_buf, tile_indices)
                            # Unsqueeze to match output dimensionality for broadcasting
                            num_output_dims = len(subscript)
                            if len(dim_indices) < num_output_dims:
                                dim_indices_set = set(dim_indices)
                                for dim in range(num_output_dims):
                                    if dim not in dim_indices_set:
                                        loaded = loaded.unsqueeze(sum(1 for d in range(dim) if d in dim_indices_set))
                            values[node.name] = loaded
                        else:
                            values[node.name] = ext_buf
                    else:
                        raise FusionError(f"Unknown buffer in load: {buf_name}")
                elif target == "store":
                    if len(node_args) > 2:
                        val = node_args[2]
                        if isinstance(val, torch.fx.Node):
                            values[node.name] = values.get(val.name, value)
                        else:
                            values[node.name] = val
                else:
                    # Apply operation using torch ops (these are traced automatically)
                    try:
                        op_fn = get_torch_op(target)
                        result = op_fn(*_resolve_args(node_args, values, value))
                        values[node.name] = result
                    except Exception as e:
                        raise FusionError(f"Failed to apply op {target}: {e}") from e
            elif node.op == "call_module" and node.target == "get_index":
                values[node.name] = 0  # Placeholder

        return value

    return helion_traceable_epilogue


def extract_prologue_ops(spec: "PrologueSpec", ext_buf_fakes: dict[str, torch.Tensor] | None = None) -> tuple[list["EpilogueOp"], list[str]]:
    """Extract prologue operations as EpilogueOp list (for tracing approach)."""
    from torch._inductor.ir import ComputedBuffer, Pointwise
    from torch._inductor.loop_body import LoopBody
    from torch._inductor import dependencies


    if not spec.prologue_nodes:
        return [], []

    # Get the last prologue node
    last_node = spec.prologue_nodes[-1].node
    if not (isinstance(last_node, ComputedBuffer) and isinstance(last_node.data, Pointwise)):
        return [], []

    pw = last_node.data
    try:
        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(pw.ranges, (), prefix="i")
        lb = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, reduce_vars)
    except Exception as e:
        return [], []

    # First load is input being transformed; subsequent loads are external buffers
    external_buffers: list[str] = []
    load_to_buffer: dict[str, str] = {}
    load_to_dim_idx: dict[str, int] = {}
    input_buf_name: str | None = None
    input_load_node_name: str | None = None

    for node in lb.root_block.graph.nodes:
        if node.op == "call_method" and node.target == "load":
            args = node.args[1:] if len(node.args) > 1 else node.args
            if args and isinstance(args[0], str):
                buf_name = args[0]
                if input_buf_name is None:
                    input_buf_name, input_load_node_name = buf_name, node.name
                else:
                    external_buffers.append(buf_name)
                    load_to_buffer[node.name] = buf_name
                    dim_idx = -1
                    # Try index expression from submodule
                    for arg in node.args:
                        if isinstance(arg, torch.fx.Node) and arg.op == "call_module":
                            if hasattr(lb, 'submodules') and arg.target in lb.submodules:
                                submod = lb.submodules[arg.target]
                                if hasattr(submod, 'index') and isinstance(submod.index, sympy.Expr):
                                    for var_idx, var in enumerate(iter_vars):
                                        if var in submod.index.free_symbols:
                                            dim_idx = var_idx; break
                                    if dim_idx >= 0: break

                    # Fallback: size comparison
                    if dim_idx < 0:
                        try:
                            from torch._inductor.virtualized import V
                            buf = V.graph.get_buffer(buf_name)
                            if buf is None and hasattr(V.graph, 'graph_inputs_original'):
                                buf = V.graph.graph_inputs_original.get(buf_name)
                            if buf is not None and len(buf.get_size()) == 1:
                                buf_dim_size = buf.get_size()[0]
                                matches = [i for i, v in enumerate(iter_vars) if v in var_ranges and (buf_dim_size == var_ranges[v] or str(buf_dim_size) == str(var_ranges[v]))]
                                dim_idx = matches[-1] if matches else -1
                        except Exception: pass

                    load_to_dim_idx[node.name] = dim_idx if dim_idx >= 0 else 0
    # Extract operations from FX graph
    prologue_ops: list["EpilogueOp"] = []
    for node in lb.root_block.graph.nodes:
        if node.op == "call_method" and node.target not in ("load", "store"):
            try:
                op_fn = get_torch_op(node.target)
                node_args = node.args[1:] if len(node.args) > 1 else node.args
                if node.target == "constant": continue  # Skip constant nodes

                # Find constant value or external buffer in args
                constant_val, ext_buf_name, ext_buf_load_node = None, None, None
                for arg in node_args:
                    if isinstance(arg, torch.fx.Node):
                        if arg.op == "call_method" and arg.target == "constant" and len(arg.args) >= 2:
                            constant_val = arg.args[1]; break
                        if arg.name in load_to_buffer:
                            ext_buf_name, ext_buf_load_node = load_to_buffer[arg.name], arg.name; break

                num_node_args = sum(1 for arg in node_args if isinstance(arg, torch.fx.Node))
                if constant_val is not None:
                    prologue_ops.append(EpilogueOp(op_fn=op_fn, is_binary=True, ext_buf=None, ext_buf_name="", dim_idx=-1, constant_val=constant_val))
                elif ext_buf_name is not None:
                    ext_buf = ext_buf_fakes.get(ext_buf_name) if ext_buf_fakes else None
                    prologue_ops.append(EpilogueOp(op_fn=op_fn, is_binary=True, ext_buf=ext_buf, ext_buf_name=ext_buf_name, dim_idx=load_to_dim_idx.get(ext_buf_load_node, 0)))
                elif num_node_args <= 1:
                    prologue_ops.append(EpilogueOp(op_fn=op_fn, is_binary=False, ext_buf=None, ext_buf_name="", dim_idx=-1))
            except ValueError: pass
    return prologue_ops, external_buffers


# --- AST Transform Utilities ---

def inject_closure_params(root: ast.Module, closures: dict[str, str], triton_fn: str, host_fn: str) -> None:
    """Inject closure parameters into kernel AST."""
    if not closures: return
    params = list(closures.values())
    tensor_ann = ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr="Tensor", ctx=ast.Load())

    def _insert(args, new_args, make):
        existing = {getattr(a, "arg", None) or getattr(a, "id", None) for a in args}
        to_add = [make(p) for p in new_args if p not in existing]
        if to_add:
            idx = next((i for i, a in enumerate(args) if (getattr(a, "arg", "") or getattr(a, "id", "")).startswith("_BLOCK_SIZE")), len(args))
            args[idx:idx] = to_add

    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef) and node.name == triton_fn:
            _insert(node.args.args, params, lambda p: ast.arg(arg=p))
        elif isinstance(node, ast.FunctionDef) and node.name == host_fn:
            _insert(node.args.args, params, lambda p: ast.arg(arg=p, annotation=tensor_ann))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_launcher":
            if len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == triton_fn:
                _insert(node.args, params, lambda p: ast.Name(id=p, ctx=ast.Load()))


def rename_function(root: ast.Module, old_name: str, new_name: str) -> None:
    """Rename a function in the AST."""
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef) and node.name == old_name:
            node.name = new_name
            break

