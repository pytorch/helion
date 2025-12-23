"""FX-to-AST conversion and fusion context for Helion.

This module handles:
- Fusion context and specifications (EpilogueSpec, PrologueSpec, FusionContext)
- FX graph to Python AST conversion for epilogue/prologue fusion
- Apply functions that transform values using fused operations
"""
from __future__ import annotations

import ast
import contextlib
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator

import sympy
import torch.fx

from ..ast_extension import expr_from_string

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


# --- Fusion Context ---

@dataclass
class FusionContext:
    """Holds fusion state during code generation."""
    epilogues: dict[str, list[EpilogueSpec]]
    prologues: dict[str, list[PrologueSpec]]
    store_map: dict[int, str] = field(default_factory=dict)
    epilogue_closures: dict[str, str] = field(default_factory=dict)
    prologue_closures: dict[str, str] = field(default_factory=dict)
    stored_values: dict[str, "ast.AST"] = field(default_factory=dict)
    stored_subscript_names: dict[str, list[str]] = field(default_factory=dict)
    multi_output_epilogues: list[EpilogueSpec] = field(default_factory=list)

    def register_closure(self, buffer_name: str, *, epilogue: bool = True) -> str:
        closures = self.epilogue_closures if epilogue else self.prologue_closures
        if buffer_name not in closures:
            closures[buffer_name] = f"{'epilogue' if epilogue else 'prologue'}_closure_{len(closures)}"
        return closures[buffer_name]

    def get_epilogue_specs(self, name: str) -> list[EpilogueSpec]: return self.epilogues.get(name, [])
    def get_prologue_specs(self, name: str) -> list[PrologueSpec]: return self.prologues.get(name, [])
    def get_epilogue_for_store(self, idx: int) -> list[EpilogueSpec]:
        return self.get_epilogue_specs(self.store_map[idx]) if idx in self.store_map else []
    def get_all_epilogue_specs(self) -> list[EpilogueSpec]:
        return [s for specs in self.epilogues.values() for s in specs]
    def record_stored_value(self, name: str, value: "ast.AST", subscripts: list[str]) -> None:
        self.stored_values[name] = value; self.stored_subscript_names[name] = subscripts
    def get_stored_value(self, name: str) -> "ast.AST | None": return self.stored_values.get(name)
    def is_last_store(self, idx: int) -> bool: return bool(self.store_map) and idx == max(self.store_map.keys())
    def get_multi_output_epilogues(self) -> list[EpilogueSpec]: return self.multi_output_epilogues
    @property
    def is_multi_output(self) -> bool: return bool(self.store_map)
    @property
    def all_closures(self) -> dict[str, str]: return {**self.epilogue_closures, **self.prologue_closures}


_current_ctx: ContextVar[FusionContext | None] = ContextVar("_fusion_ctx", default=None)


@contextlib.contextmanager
def fusion_context(epilogues: dict[str, list[EpilogueSpec]], prologues: dict[str, list[PrologueSpec]],
                   store_index_to_buffer: dict[int, str] | None = None) -> Iterator[FusionContext]:
    ctx = FusionContext(epilogues=epilogues, prologues=prologues, store_map=store_index_to_buffer or {})
    token = _current_ctx.set(ctx)
    try: yield ctx
    finally: _current_ctx.reset(token)


def get_current_context() -> FusionContext | None:
    return _current_ctx.get()


# --- Operator Tables (op_name -> (ast_class, is_comparison)) ---
_BINARY_OPS = {
    "add": (ast.Add, False), "sub": (ast.Sub, False), "mul": (ast.Mult, False),
    "truediv": (ast.Div, False), "floordiv": (ast.FloorDiv, False), "mod": (ast.Mod, False),
    "pow": (ast.Pow, False), "and_": (ast.BitAnd, False), "or_": (ast.BitOr, False),
    "xor": (ast.BitXor, False), "lshift": (ast.LShift, False), "rshift": (ast.RShift, False),
    "eq": (ast.Eq, True), "ne": (ast.NotEq, True), "lt": (ast.Lt, True),
    "le": (ast.LtE, True), "gt": (ast.Gt, True), "ge": (ast.GtE, True),
}
# Unary functions: op_name -> (module, func_name)
_UNARY_FUNCS = {
    "exp": ("tl", "exp"), "exp2": ("tl", "exp2"), "log": ("tl", "log"), "log2": ("tl", "log2"),
    "sqrt": ("tl", "sqrt"), "rsqrt": ("tl", "rsqrt"), "abs": ("tl", "abs"), "neg": ("tl", "-"),
    "floor": ("tl", "floor"), "ceil": ("tl", "ceil"), "sin": ("tl", "sin"), "cos": ("tl", "cos"),
    "tan": ("libdevice", "tan"), "asin": ("libdevice", "asin"), "acos": ("libdevice", "acos"),
    "atan": ("libdevice", "atan"), "sinh": ("libdevice", "sinh"), "cosh": ("libdevice", "cosh"),
    "tanh": ("libdevice", "tanh"), "asinh": ("libdevice", "asinh"), "acosh": ("libdevice", "acosh"),
    "atanh": ("libdevice", "atanh"), "erf": ("libdevice", "erf"), "erfc": ("libdevice", "erfc"),
    "lgamma": ("libdevice", "lgamma"), "expm1": ("libdevice", "expm1"), "log1p": ("libdevice", "log1p"),
    "trunc": ("libdevice", "trunc"), "round": ("libdevice", "round"), "cbrt": ("libdevice", "cbrt"),
    "j0": ("libdevice", "j0"), "j1": ("libdevice", "j1"), "y0": ("libdevice", "y0"), "y1": ("libdevice", "y1"),
    "isinf": ("libdevice", "isinf"), "isnan": ("libdevice", "isnan"), "isfinite": ("libdevice", "isfinite"),
}
# Binary functions: op_name -> (module, func_name)
_BINARY_FUNCS = {
    "maximum": ("tl", "maximum"), "minimum": ("tl", "minimum"), "atan2": ("tl_math", "atan2"),
    "hypot": ("libdevice", "hypot"), "copysign": ("libdevice", "copysign"), "fmod": ("libdevice", "fmod"),
    "remainder": ("libdevice", "remainder"), "nextafter": ("libdevice", "nextafter"), "pow": ("libdevice", "pow"),
}


# --- FX Converter ---

class FusionFXConverter:
    """Convert Inductor FX graph to Python AST for fusion."""
    def __init__(self, accumulator_map: dict[str, ast.AST], subscript_names: list[str], register_closure_fn: Callable[[str], str]):
        self.accumulator_map, self.subscript_names, self.register_closure = accumulator_map, subscript_names, register_closure_fn
        self.node_to_ast: dict[torch.fx.Node, ast.AST] = {}
        self.ndim, self._index_map = len(subscript_names), {f"i{i}": name for i, name in enumerate(subscript_names)}

    def convert(self, fx_graph: torch.fx.Graph, indexing_exprs: dict[str, sympy.Expr]) -> ast.AST:
        self._indexing_exprs = indexing_exprs
        for node in fx_graph.nodes:
            if (result := self._convert_node(node)): self.node_to_ast[node] = result
        output_node = next(n for n in fx_graph.nodes if n.op == "output")
        if output_node.args and output_node.args[0]:
            rv = output_node.args[0]
            return self._get_ast(rv[0] if isinstance(rv, (list, tuple)) and rv else rv)
        raise ValueError("FX graph has no output value")

    def _convert_node(self, node: torch.fx.Node) -> ast.AST | None:
        if node.op in ("placeholder", "output"): return None
        if node.op == "call_method": return self._convert_op(node.target, node.args)
        if node.op == "call_module" and node.target == "get_index":
            return self._convert_index_expr(self._indexing_exprs[node.args[0]])
        raise ValueError(f"Unsupported node: {node.op}/{getattr(node, 'target', '')}")

    def _convert_op(self, target: str, args: tuple) -> ast.AST:
        a = args[1:] if len(args) > 1 else args
        if target == "load":
            if a[0] in self.accumulator_map: return self.accumulator_map[a[0]]
            return expr_from_string(f"tl.load({self.register_closure(a[0])} + {{idx}})", idx=self._get_ast(a[1]))
        if target == "store": return self._get_ast(a[2])
        if target in _BINARY_OPS:
            op, is_cmp = _BINARY_OPS[target]
            l, r = self._get_ast(a[0]), self._get_ast(a[1])
            return ast.Compare(left=l, ops=[op()], comparators=[r]) if is_cmp else ast.BinOp(left=l, op=op(), right=r)
        if target in _UNARY_FUNCS:
            mod, fn = _UNARY_FUNCS[target]
            return ast.UnaryOp(op=ast.USub(), operand=self._get_ast(a[0])) if fn == "-" else self._make_call(mod, fn, [self._get_ast(a[0])])
        if target in _BINARY_FUNCS:
            mod, fn = _BINARY_FUNCS[target]
            return self._make_call(mod, fn, [self._get_ast(a[0]), self._get_ast(a[1])])
        if target == "where": return expr_from_string("tl.where({c}, {t}, {f})", c=self._get_ast(a[0]), t=self._get_ast(a[1]), f=self._get_ast(a[2]))
        if target == "to_dtype": return expr_from_string(f"{{v}}.to(tl.{_torch_dtype_to_triton(a[1])})", v=self._get_ast(a[0]))
        if target == "constant": return ast.Constant(value=a[0])
        if target == "index_expr": return self._get_ast(a[0])
        if target == "sigmoid": return expr_from_string("tl.sigmoid({x})", x=expr_from_string("{x}.to(tl.float32)", x=self._get_ast(a[0])))
        if target == "relu": return expr_from_string("tl.maximum({x}, 0)", x=self._get_ast(a[0]))
        if target == "reciprocal": return ast.BinOp(left=ast.Constant(value=1.0), op=ast.Div(), right=self._get_ast(a[0]))
        raise ValueError(f"Unsupported op: {target}")

    def _convert_index_expr(self, expr: sympy.Expr) -> ast.AST:
        return sympy_to_ast(expr, self._index_map, self.ndim)

    def _get_ast(self, v: Any) -> ast.AST:
        if isinstance(v, torch.fx.Node):
            if v in self.node_to_ast: return self.node_to_ast[v]
            result = self._convert_node(v)
            if result: self.node_to_ast[v] = result; return result
            raise ValueError(f"Cannot get AST for node: {v}")
        if isinstance(v, (bool, int, float, str)): return ast.Constant(value=v)
        if isinstance(v, sympy.Expr): return sympy_to_ast(v, self._index_map)
        raise ValueError(f"Unknown value type: {type(v)}")

    def _make_call(self, mod: str, fn: str, args: list[ast.AST]) -> ast.AST:
        if len(args) == 1: return expr_from_string(f"{mod}.{fn}({{a}})", a=args[0])
        if len(args) == 2: return expr_from_string(f"{mod}.{fn}({{a}}, {{b}})", a=args[0], b=args[1])
        if len(args) == 3: return expr_from_string(f"{mod}.{fn}({{a}}, {{b}}, {{c}})", a=args[0], b=args[1], c=args[2])
        raise ValueError(f"_make_call doesn't support {len(args)} args")


# --- Sympy to AST Conversion ---

def _broadcast_slice(base: ast.AST, dim_idx: int, ndim: int) -> ast.AST:
    if ndim == 1: return base
    slices = [ast.Slice() if i == dim_idx else ast.Constant(value=None) for i in range(ndim)]
    return ast.Subscript(value=base, slice=ast.Tuple(elts=slices, ctx=ast.Load()), ctx=ast.Load())


def sympy_to_ast(expr: sympy.Expr, index_map: dict[str, str], ndim: int = 1) -> ast.AST:
    """Convert sympy expression to Python AST with broadcasting."""
    conv = lambda e: sympy_to_ast(e, index_map, ndim)
    if isinstance(expr, sympy.Symbol):
        name = str(expr)
        if name not in index_map: return ast.Name(id=name, ctx=ast.Load())
        base = ast.Name(id=index_map[name], ctx=ast.Load())
        return _broadcast_slice(base, int(name[1:]), ndim) if ndim > 1 and name.startswith('i') and name[1:].isdigit() and int(name[1:]) < ndim else base
    if isinstance(expr, sympy.Integer): return ast.Constant(value=int(expr))
    if isinstance(expr, sympy.Float): return ast.Constant(value=float(expr))
    if isinstance(expr, sympy.Rational): return ast.BinOp(left=ast.Constant(value=int(expr.p)), op=ast.Div(), right=ast.Constant(value=int(expr.q)))

    def _fold(args, op): r = conv(args[0]); [r := ast.BinOp(left=r, op=op(), right=conv(a)) for a in args[1:]]; return r

    if isinstance(expr, sympy.Add):
        args, result = list(expr.args), conv(expr.args[0])
        for a in args[1:]: result = ast.BinOp(left=result, op=ast.Sub() if _is_negative(a) else ast.Add(), right=conv(-a if _is_negative(a) else a))
        return result
    if isinstance(expr, sympy.Mul):
        args = list(expr.args); neg = sum(1 for a in args if a == sympy.Integer(-1)) % 2 == 1
        args = [a for a in args if a not in (sympy.Integer(-1), sympy.Integer(1))]
        if not args: return ast.Constant(value=-1 if neg else 1)
        result = _fold(args, ast.Mult); return ast.UnaryOp(op=ast.USub(), operand=result) if neg else result
    if isinstance(expr, sympy.Pow): return ast.BinOp(left=conv(expr.base), op=ast.Pow(), right=conv(expr.exp))
    if isinstance(expr, sympy.Mod): return ast.BinOp(left=conv(expr.args[0]), op=ast.Mod(), right=conv(expr.args[1]))
    if isinstance(expr, sympy.floor): return expr_from_string("tl.floor({x})", x=conv(expr.args[0]))
    if isinstance(expr, sympy.ceiling): return expr_from_string("tl.ceil({x})", x=conv(expr.args[0]))
    if isinstance(expr, (sympy.Min, sympy.Max)):
        fn, args = "minimum" if isinstance(expr, sympy.Min) else "maximum", [conv(a) for a in expr.args]
        result = args[0]; [result := expr_from_string(f"tl.{fn}({{a}}, {{b}})", a=result, b=a) for a in args[1:]]; return result
    if hasattr(sympy, 'FloorDiv') and isinstance(expr, sympy.FloorDiv):
        return ast.BinOp(left=conv(expr.args[0]), op=ast.FloorDiv(), right=conv(expr.args[1]))
    expr_str = str(expr)
    for old, new in index_map.items(): expr_str = re.sub(rf'\b{old}\b', new, expr_str)
    try: return ast.parse(expr_str, mode="eval").body
    except SyntaxError: raise ValueError(f"Cannot convert sympy expression to AST: {expr}")


def _is_negative(expr: sympy.Expr) -> bool:
    if isinstance(expr, sympy.Mul):
        return bool(expr.args) and expr.args[0] == sympy.Integer(-1)
    return isinstance(expr, sympy.Integer) and int(expr) < 0


def _torch_dtype_to_triton(dtype) -> str:
    import torch
    return {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32",
            torch.float64: "float64", torch.int8: "int8", torch.int16: "int16", torch.int32: "int32",
            torch.int64: "int64", torch.uint8: "uint8", torch.bool: "int1"}.get(dtype, "float32")


# --- Fusion Error ---

class FusionError(Exception):
    """Raised when fusion cannot be applied."""
    pass


# --- Apply Functions ---

def _convert_fusion_nodes(nodes: list, accumulator_map: dict[str, ast.AST], subscript_names: list[str],
                          register_closure_fn: Callable[[str], str], node_type: str) -> ast.AST:
    from torch._inductor.ir import ComputedBuffer, Pointwise
    from torch._inductor.loop_body import LoopBody; from torch._inductor import dependencies
    last_node = nodes[-1].node
    if not (isinstance(last_node, ComputedBuffer) and isinstance(last_node.data, Pointwise)):
        raise FusionError(f"Expected Pointwise {node_type}, got {type(last_node)}")
    pw = last_node.data
    (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(pw.ranges, (), prefix="i")
    try:
        lb = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, reduce_vars)
        return FusionFXConverter(accumulator_map, subscript_names, register_closure_fn).convert(lb.root_block.graph, lb.indexing_exprs)
    except Exception as e: raise FusionError(f"FX conversion failed: {e}") from e


def apply_epilogue(accumulator_values: dict[str, ast.AST | str], subscript_names: list[str],
                   spec: EpilogueSpec, register_closure_fn: Callable[[str], str]) -> ast.AST:
    if not spec.epilogue_nodes:
        v = next(iter(accumulator_values.values()))
        return v if isinstance(v, ast.AST) else ast.parse(v, mode="eval").body
    acc_map = {n: (v if isinstance(v, ast.AST) else ast.parse(v, mode="eval").body) for n, v in accumulator_values.items()}
    return _convert_fusion_nodes(spec.epilogue_nodes, acc_map, subscript_names, register_closure_fn, "epilogue")


def apply_prologue(input_value: ast.AST, subscript_names: list[str], spec: PrologueSpec,
                   register_closure_fn: Callable[[str], str]) -> ast.AST:
    if not spec.prologue_nodes: return input_value
    return _convert_fusion_nodes(spec.prologue_nodes, {spec.input_name: input_value}, subscript_names, register_closure_fn, "prologue")


def generate_epilogue_store(ctx: FusionContext, spec: EpilogueSpec, subscript_names: list[str]) -> ast.AST | None:
    """Generate a store statement for an epilogue that produces a new output."""
    from torch._inductor.virtualized import V
    if not spec.epilogue_nodes: return None
    acc_values = {n: v for n in spec.accumulator_names if (v := ctx.get_stored_value(n))}
    if not acc_values: return None

    epilogue_output = spec.epilogue_nodes[-1].node
    epilogue_buf_name = getattr(epilogue_output, "get_name", lambda: None)()
    if not epilogue_buf_name: return None

    register = lambda n: ctx.register_closure(n, epilogue=True)
    result_ast = apply_epilogue(acc_values, subscript_names, spec, register)
    closure_name = register(epilogue_buf_name)

    # Get strides from epilogue output or template buffer
    strides = None
    try: strides = list(epilogue_output.get_stride()) if hasattr(epilogue_output, "get_stride") else None
    except Exception: pass
    if not strides or len(strides) != len(subscript_names):
        for buf_name in ctx.store_map.values():
            try:
                buf = V.graph.get_buffer(buf_name)
                if buf and hasattr(buf, "get_stride") and len(s := list(buf.get_stride())) == len(subscript_names):
                    strides = s; break
            except Exception: continue
    if not strides or len(strides) != len(subscript_names):
        raise FusionError(f"Cannot determine strides: subscript_names={subscript_names}, strides={strides}")

    ndim = len(subscript_names)
    def _fmt(i, name, stride):
        if ndim == 1: return f"{name} * {stride}"
        return f"{name}[:, None] * {stride}" if i == 0 else f"{name}[None, :] * {stride}" if ndim == 2 else f"{name} * {stride}"
    offset_expr = " + ".join(_fmt(i, n, s) for i, (n, s) in enumerate(zip(subscript_names, strides))) or "0"
    return expr_from_string(f"tl.store({closure_name} + {{offset}}, {{value}})",
                           offset=ast.parse(offset_expr, mode="eval").body, value=ast.parse(ast.unparse(result_ast), mode="eval").body)


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
