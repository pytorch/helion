"""Ops handler for fusion code generation.

FusionOpsHandler generates Triton code strings for fusion operations,
converting Inductor IR ops to Triton expressions.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Callable

import sympy
import torch
from torch._inductor.codegen.triton import TritonOverrides


# Triton dtype mapping
_TRITON_DTYPE_MAP = {
    torch.float32: "tl.float32",
    torch.float16: "tl.float16",
    torch.bfloat16: "tl.bfloat16",
    torch.float64: "tl.float64",
    torch.int8: "tl.int8",
    torch.int16: "tl.int16",
    torch.int32: "tl.int32",
    torch.int64: "tl.int64",
    torch.uint8: "tl.uint8",
    torch.uint16: "tl.uint16",
    torch.uint32: "tl.uint32",
    torch.uint64: "tl.uint64",
    torch.bool: "tl.int1",
}


def _paren_wrap(op: str):
    """Wrap a parent op result with parentheses for precedence safety."""
    parent_fn = getattr(TritonOverrides, op)
    return staticmethod(lambda *args: f"({parent_fn(*args)})")


def _tl_fp32(name: str):
    """Create op that upcasts to fp32 before calling tl.{name}."""
    return staticmethod(lambda x: f"(tl.{name}(({x}).to(tl.float32)))")


class FusionOpsHandler(TritonOverrides):
    """String-based ops handler for fusion code generation.

    Inherits most ops from TritonOverrides. Only overrides ops that need
    special handling for Helion's indexing or that access V.kernel/V.graph.

    This handler converts Inductor IR operations into Triton code strings
    that can be injected into the fused kernel.
    """

    def __init__(
        self,
        accumulator_names: set[str],
        output_var: str,
        subscript_names: list[str],
        register_closure_fn: Callable[[str], str],
    ) -> None:
        """Initialize the ops handler.

        Args:
            accumulator_names: Names of accumulator buffers (kernel outputs)
            output_var: Placeholder variable name for the output
            subscript_names: Names of index variables (e.g., ["idx0", "idx1"])
            register_closure_fn: Function to register external buffer closures
        """
        self.accumulator_names = accumulator_names
        self.output_var = output_var
        self.subscript_names = subscript_names
        self.register_closure = register_closure_fn
        self.final_value = output_var

        # Build index translation map
        self._index_map = {
            f"{p}{i}": name
            for i, name in enumerate(subscript_names)
            for p in ("i", "x")
        }

    def _translate_index(self, index: sympy.Expr) -> str:
        """Translate sympy index expression to Triton indexing."""
        index_str = str(index)
        dims_used = [
            i
            for i, name in enumerate(self.subscript_names)
            if f"i{i}" in index_str or f"x{i}" in index_str
        ]

        if len(dims_used) > 1 and len(self.subscript_names) > 1:
            # Multi-dimensional indexing with broadcasting
            for i, name in enumerate(self.subscript_names):
                for p in ("i", "x"):
                    var = f"{p}{i}"
                    if var in index_str:
                        slices = [
                            ":" if j == i else "None"
                            for j in range(len(self.subscript_names))
                        ]
                        index_str = re.sub(
                            rf"\b{var}\b", f"{name}[{', '.join(slices)}]", index_str
                        )
        else:
            # Simple 1D indexing
            for var, name in self._index_map.items():
                index_str = re.sub(rf"\b{var}\b", name, index_str)

        return index_str

    def _broadcast_slice(self, expr: str, translated_idx: str) -> str:
        """Add broadcasting slices if needed."""
        dims_used = [
            i
            for i, name in enumerate(self.subscript_names)
            if re.search(rf"\b{re.escape(name)}\b", translated_idx)
        ]
        if len(dims_used) < len(self.subscript_names):
            slices = [
                ":" if i in dims_used else "None"
                for i in range(len(self.subscript_names))
            ]
            return f"{expr}[{', '.join(slices)}]"
        return expr

    # --- Core ops ---

    def load(self, name: str, index: sympy.Expr) -> str:
        """Load from a buffer."""
        if name in self.accumulator_names:
            return self.output_var
        closure = self.register_closure(name)
        idx = self._translate_index(index)
        return self._broadcast_slice(f"tl.load({closure} + {idx})", idx)

    def store(self, name: str, index: Any, value: str, mode: Any = None) -> None:
        """Store to a buffer (captures final value)."""
        self.final_value = value

    def store_reduction(self, name: str, index: Any, value: str) -> None:
        """Store reduction result (captures final value)."""
        self.final_value = value

    @staticmethod
    def constant(value: Any, dtype: torch.dtype) -> str:
        """Generate a constant."""
        return repr(value)

    def index_expr(self, expr: sympy.Expr, dtype: Any) -> str:
        """Generate an index expression."""
        return self._translate_index(expr)

    @staticmethod
    def to_dtype(
        x: str,
        dtype: torch.dtype,
        src_dtype: Any = None,
        use_compute_types: bool = True,
    ) -> str:
        """Generate a dtype cast."""
        if dtype == torch.bool:
            return f"({x} != 0)"
        triton_dtype = _TRITON_DTYPE_MAP.get(dtype, "tl.float32")
        return f"({x}).to({triton_dtype})"

    @staticmethod
    def indirect_indexing(*args: Any, **kwargs: Any) -> str:
        """Indirect indexing is not supported in fusion."""
        raise ValueError(
            "Indirect indexing not supported in epilogue/prologue fusion. "
            "Fusion will be disabled for this kernel."
        )

    # --- Math ops that need special handling ---

    @staticmethod
    def relu(x: str) -> str:
        return f"(tl.maximum(0, {x}))"

    @staticmethod
    def maximum(a: str, b: str) -> str:
        return f"(tl.maximum({a}, {b}))"

    @staticmethod
    def minimum(a: str, b: str) -> str:
        return f"(tl.minimum({a}, {b}))"

    @staticmethod
    def tanh(x: str) -> str:
        return f"(libdevice.tanh({x}))"

    @staticmethod
    def reciprocal(x: str) -> str:
        return f"(1.0 / ({x}))"

    # --- Ops that require fp32 (Triton doesn't support fp16/bf16) ---
    sigmoid = _tl_fp32("sigmoid")
    exp = _tl_fp32("exp")
    log = _tl_fp32("log")
    sqrt = _tl_fp32("sqrt")
    sin = _tl_fp32("sin")
    cos = _tl_fp32("cos")

    # --- Binary ops - wrap parent with parentheses ---

    add = _paren_wrap("add")
    sub = _paren_wrap("sub")
    mul = _paren_wrap("mul")
    truediv = _paren_wrap("truediv")
    floordiv = _paren_wrap("floordiv")
    mod = _paren_wrap("mod")
    neg = _paren_wrap("neg")

    # --- Defensive __getattr__ (safety net for missed ops) ---

    def __getattr__(self, name: str):
        """Catch any unimplemented op access.

        This is a safety net - if can_fuse's op whitelist is correct,
        this should never be called. But if an op slips through,
        we want a clear error instead of a cryptic V.kernel crash.

        Note: This is only called for attributes not found via normal lookup.
        So it won't interfere with methods defined above or inherited safely.
        """
        # Check if it exists in parent class (TritonOverrides)
        if hasattr(TritonOverrides, name):
            raise ValueError(
                f"Op '{name}' is not supported in prologue/epilogue fusion. "
                f"This op may access V.kernel which is not available during fusion. "
                f"Please add '{name}' to SUPPORTED_FUSION_OPS in can_fuse.py if it's safe, "
                f"or override it in FusionOpsHandler."
            )
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # --- Result conversion ---

    def to_ast(self) -> ast.AST:
        """Parse final value to AST for injection."""
        return ast.parse(self.final_value, mode="eval").body
