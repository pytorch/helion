"""FX-to-AST conversion for Helion xlogue fusion.

This module handles:
- Fusion specifications (EpilogueSpec, PrologueSpec)
- OpsHandler-based conversion for epilogue/prologue fusion using Inductor's TritonOverrides
- Apply functions that transform values using fused operations
"""

from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import sympy
import torch
import torch.fx
from torch._inductor import config as inductor_config
from torch._inductor.codegen.simd import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.graph import GraphLowering
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.virtualized import OpsValue
from torch._inductor.virtualized import V
from torch.fx._lazy_graph_module import _LazyGraphModule

from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode

    from ..compile_environment import CompileEnvironment


# Inductor config patch for xlogue fusion
INDUCTOR_PATCH: dict[str, object] = {
    "triton.codegen_upcast_to_fp32": True,
    "split_reductions": False,
    "unroll_reductions_threshold": 1,
}


# --- Specifications ---
@dataclass
class EpilogueSpec:
    """Epilogue fusion spec: inline (N=1) or deferred (N>1)."""

    epilogue_nodes: list[BaseSchedulerNode]
    accumulator_name: str | set[str]

    @property
    def accumulator_names(self) -> set[str]:
        return (
            self.accumulator_name
            if isinstance(self.accumulator_name, set)
            else {self.accumulator_name}
        )

    @property
    def is_deferred(self) -> bool:
        return len(self.accumulator_names) > 1


@dataclass
class PrologueSpec:
    """Prologue fusion spec."""

    prologue_nodes: list[BaseSchedulerNode]
    input_name: str


# --- Helper functions ---


def _unpack_opsvalue(value: object) -> str:
    """Unpack OpsValue to string."""
    if isinstance(value, OpsValue):
        return str(value)
    assert isinstance(value, str)
    return value


def _dummy_gm() -> torch.fx.GraphModule:
    """Create a dummy GraphModule for FakeGraphLowering."""
    return torch.fx.symbolic_trace(lambda: None)


class _FakeGraphLowering(GraphLowering):
    """Minimal GraphLowering for xlogue fusion context."""

    def __init__(self) -> None:
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        super().__init__(_LazyGraphModule({}, _dummy_gm().graph), shape_env=env.shape_env)
        self.current_device = env.device


# --- OpsHandler-based Xlogue Fusion ---


class XlogueFusionOpsHandler(DefaultHandler):
    """OpsHandler for xlogue fusion that generates Triton code strings.

    This handler wraps TritonOverrides to generate Triton code for fusion ops,
    giving access to all 135+ ops supported by Inductor.
    """

    def __init__(
        self,
        accumulator_map: dict[str, str],  # buffer_name -> variable_name string
        subscript_names: list[str],  # ["indices_0", "indices_1"]
        register_closure_fn: Callable[[str], str],
    ) -> None:
        super().__init__()
        self.parent_handler = TritonOverrides()
        self.accumulator_map = accumulator_map
        self.subscript_names = subscript_names
        self.register_closure = register_closure_fn
        self.ndim = len(subscript_names)
        self._index_map = {f"i{i}": name for i, name in enumerate(subscript_names)}

    def _default(self, name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> str:
        """Delegate to TritonOverrides for most ops, wrapping result in parens for precedence."""
        result = _unpack_opsvalue(getattr(self.parent_handler, name)(*args, **kwargs))
        return f"({result})"

    def load(self, name: str, index: sympy.Expr) -> str:
        """Handle load ops - return accumulator or generate tl.load."""
        if name in self.accumulator_map:
            return self.accumulator_map[name]
        ptr_name = self.register_closure(name)
        index_str = self._sympy_to_string(index)
        return f"tl.load({ptr_name} + {index_str})"

    def store(self, name: str, index: sympy.Expr, value: str, mode: object = None) -> str:
        """Handle store ops - just return the value being stored."""
        return value

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> str:
        """Convert sympy index expression to Triton code."""
        return self._sympy_to_string(expr)

    def constant(self, value: object, dtype: torch.dtype) -> str:
        """Handle constants."""
        return f"({repr(value)})"

    def to_dtype(
        self,
        x: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        """Emit explicit tl.cast for dtype conversion."""
        x_str = _unpack_opsvalue(x) if isinstance(x, OpsValue) else str(x)
        return f"(tl.cast({x_str}, {_triton_type(dtype)}))"

    def sigmoid(self, x: object) -> str:
        """Handle sigmoid - cast to fp32, apply sigmoid."""
        x_str = _unpack_opsvalue(x) if isinstance(x, OpsValue) else str(x)
        return f"(tl.sigmoid(tl.cast({x_str}, tl.float32)))"

    def _sympy_to_string(self, expr: sympy.Expr) -> str:
        """Convert sympy expression with index variable mapping and broadcasting."""
        result = str(expr)
        for i, name in enumerate(self.subscript_names):
            old = f"i{i}"
            if self.ndim > 1:
                # Apply broadcasting: indices_0[:, None] or indices_1[None, :]
                slices = [":," if j == i else "None," for j in range(self.ndim)]
                new = f"{name}[{(''.join(slices)).rstrip(',')}]"
            else:
                new = name
            result = re.sub(rf"\b{old}\b", new, result)
        return result


@contextlib.contextmanager
def _install_xlogue_handlers(handler: XlogueFusionOpsHandler):
    """Install handlers for xlogue fusion code generation.

    All three handlers are required:
    - V.set_graph_handler - for sizevars and shape_env
    - V.set_ops_handler - the main ops handler
    - V.set_kernel_handler - TritonOverrides methods call V.kernel
    """
    with (
        inductor_config.patch(INDUCTOR_PATCH),
        V.set_graph_handler(_FakeGraphLowering()),
        V.set_ops_handler(handler),
        V.set_kernel_handler(TritonKernel({}, features=SIMDKernelFeatures([], sympy.S.One))),
    ):
        yield


def _convert_fusion_nodes_v2(
    nodes: list,
    accumulator_map: dict[str, str],  # buffer_name -> variable_name string
    subscript_names: list[str],
    register_closure_fn: Callable[[str], str],
    node_type: str,
) -> ast.expr:
    """Convert fusion nodes using OpsHandler pattern.

    This replaces the old XlogueFusionFXConverter approach by directly
    invoking the Pointwise inner_fn with our OpsHandler installed.
    """
    from torch._inductor.ir import ComputedBuffer
    from torch._inductor.ir import Pointwise

    last_node = nodes[-1].node
    if not (isinstance(last_node, ComputedBuffer) and isinstance(last_node.data, Pointwise)):
        raise XlogueFusionError(f"Expected Pointwise {node_type}, got {type(last_node)}")

    pw = last_node.data
    handler = XlogueFusionOpsHandler(accumulator_map, subscript_names, register_closure_fn)

    with _install_xlogue_handlers(handler):
        indices = [sympy.Symbol(f"i{n}") for n in range(len(pw.ranges))]
        result_str = _unpack_opsvalue(pw.inner_fn(indices))

    return expr_from_string(result_str)


def _triton_type(dtype: torch.dtype) -> str:
    """Convert torch dtype to triton type string."""
    return {
        torch.float16: "tl.float16",
        torch.bfloat16: "tl.bfloat16",
        torch.float32: "tl.float32",
        torch.float64: "tl.float64",
        torch.int8: "tl.int8",
        torch.int16: "tl.int16",
        torch.int32: "tl.int32",
        torch.int64: "tl.int64",
        torch.uint8: "tl.uint8",
        torch.bool: "tl.int1",
    }.get(dtype, "tl.float32")


# --- Fusion Error ---


class XlogueFusionError(Exception):
    """Raised when xlogue fusion cannot be applied."""


# --- Apply Functions ---


def apply_epilogue(
    accumulator_values: dict[str, ast.expr | str],
    subscript_names: list[str],
    spec: EpilogueSpec,
    register_closure_fn: Callable[[str], str],
) -> ast.expr:
    def to_expr(v: ast.expr | str) -> ast.expr:
        return v if isinstance(v, ast.expr) else ast.parse(v, mode="eval").body

    def to_str(v: ast.expr | str) -> str:
        return ast.unparse(v) if isinstance(v, ast.expr) else v

    if not spec.epilogue_nodes:
        return to_expr(next(iter(accumulator_values.values())))

    # Convert AST values to strings for the new OpsHandler-based approach
    acc_map = {n: to_str(v) for n, v in accumulator_values.items()}
    return _convert_fusion_nodes_v2(
        spec.epilogue_nodes, acc_map, subscript_names, register_closure_fn, "epilogue"
    )


def apply_prologue(
    input_value: ast.expr,
    subscript_names: list[str],
    spec: PrologueSpec,
    register_closure_fn: Callable[[str], str],
) -> ast.expr:
    if not spec.prologue_nodes:
        return input_value
    # Convert AST value to string for the new OpsHandler-based approach
    input_str = ast.unparse(input_value)
    return _convert_fusion_nodes_v2(
        spec.prologue_nodes,
        {spec.input_name: input_str},
        subscript_names,
        register_closure_fn,
        "prologue",
    )


def generate_epilogue_store(
    ctx: CompileEnvironment, spec: EpilogueSpec, subscript_names: list[str]
) -> ast.expr | None:
    """Generate a store statement for an epilogue that produces a new output."""
    from torch._inductor.virtualized import V

    if not spec.epilogue_nodes:
        return None
    acc_values = {
        n: v for n in spec.accumulator_names if (v := ctx.xlogue_fusion_get_stored_value(n))
    }
    if not acc_values:
        return None

    epilogue_output = spec.epilogue_nodes[-1].node
    epilogue_buf_name = getattr(epilogue_output, "get_name", lambda: None)()
    if not epilogue_buf_name:
        return None

    def register(n: str) -> str:
        return ctx.xlogue_fusion_register_closure(n, epilogue=True)

    result_ast = apply_epilogue(
        acc_values,  # pyrefly: ignore
        subscript_names,
        spec,
        register,
    )
    closure_name = register(epilogue_buf_name)

    # Get strides from epilogue output or template buffer
    strides = None
    with contextlib.suppress(Exception):
        strides = (
            list(epilogue_output.get_stride())
            if hasattr(epilogue_output, "get_stride")
            else None
        )
    if not strides or len(strides) != len(subscript_names):
        for buf_name in ctx._xlogue_fusion_store_map.values():
            try:
                buf = V.graph.get_buffer(buf_name)
                if (
                    buf
                    and hasattr(buf, "get_stride")
                    and len(s := list(buf.get_stride())) == len(subscript_names)
                ):
                    strides = s
                    break
            except Exception:
                continue
    if not strides or len(strides) != len(subscript_names):
        raise XlogueFusionError(
            f"Cannot determine strides: subscript_names={subscript_names}, strides={strides}"
        )

    ndim = len(subscript_names)

    def _fmt(i: int, name: str, stride: int | sympy.Expr) -> str:
        if ndim == 1:
            return f"{name} * {stride}"
        return (
            f"{name}[:, None] * {stride}"
            if i == 0
            else f"{name}[None, :] * {stride}"
            if ndim == 2
            else f"{name} * {stride}"
        )

    offset_expr = (
        " + ".join(
            _fmt(i, n, s)
            for i, (n, s) in enumerate(zip(subscript_names, strides, strict=True))
        )
        or "0"
    )
    return expr_from_string(
        f"tl.store({closure_name} + {{offset}}, {{value}})",
        offset=ast.parse(offset_expr, mode="eval").body,
        value=ast.parse(ast.unparse(result_ast), mode="eval").body,
    )


# --- AST Transform Utilities ---


def inject_closure_params(
    root: ast.Module, closures: dict[str, str], triton_fn: str, host_fn: str
) -> None:
    """Inject closure parameters into kernel AST."""
    if not closures:
        return
    params = list(closures.values())
    tensor_ann = ast.Attribute(
        value=ast.Name(id="torch", ctx=ast.Load()), attr="Tensor", ctx=ast.Load()
    )

    def _insert(
        args: list[ast.AST], new_args: list[str], make: Callable[[str], ast.AST]
    ) -> None:
        existing = {getattr(a, "arg", None) or getattr(a, "id", None) for a in args}
        to_add = [make(p) for p in new_args if p not in existing]
        if to_add:
            idx = next(
                (
                    i
                    for i, a in enumerate(args)
                    if (getattr(a, "arg", "") or getattr(a, "id", "")).startswith(
                        "_BLOCK_SIZE"
                    )
                ),
                len(args),
            )
            args[idx:idx] = to_add

    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef) and node.name == triton_fn:
            _insert(
                node.args.args,  # pyrefly: ignore [bad-argument-type]
                params,
                lambda p: ast.arg(arg=p),
            )
        elif isinstance(node, ast.FunctionDef) and node.name == host_fn:
            _insert(
                node.args.args,  # pyrefly: ignore [bad-argument-type]
                params,
                lambda p: ast.arg(arg=p, annotation=tensor_ann),
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_launcher"
        ):
            if (
                len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == triton_fn
            ):
                _insert(
                    node.args,  # pyrefly: ignore [bad-argument-type]
                    params,
                    lambda p: ast.Name(id=p, ctx=ast.Load()),
                )


def rename_function(root: ast.Module, old_name: str, new_name: str) -> None:
    """Rename a function in the AST."""
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef) and node.name == old_name:
            node.name = new_name
            break
