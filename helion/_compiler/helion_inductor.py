"""Helion integration with PyTorch Inductor for torch.compile support."""
from __future__ import annotations

import ast as python_ast
import contextlib
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Sequence, Union

import sympy
import torch

from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    IRNode,
    Layout,
    MultiOutput,
    MultiOutputLayout,
    Pointwise,
    ReinterpretView,
    StorageBox,
    TensorBox,
    TemplateBuffer,
    TritonTemplateBuffer,
)
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.lowering import register_lowering
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode

# --- Context variables for fusion ---
_epilogue_context: ContextVar[dict[str, list["EpilogueSpec"]] | None] = ContextVar(
    "_epilogue_context", default=None
)
_epilogue_closures: ContextVar[dict[str, str] | None] = ContextVar(
    "_epilogue_closures", default=None
)
_store_index_to_buffer: ContextVar[dict[int, str] | None] = ContextVar(
    "_store_index_to_buffer", default=None
)
_prologue_context: ContextVar[dict[str, list["PrologueSpec"]] | None] = ContextVar(
    "_prologue_context", default=None
)
_prologue_closures: ContextVar[dict[str, str] | None] = ContextVar(
    "_prologue_closures", default=None
)


def _register_closure(
    closures_var: ContextVar[dict[str, str] | None], prefix: str, buffer_name: str
) -> str:
    """Register an external buffer as a closure and return its parameter name."""
    closures = closures_var.get()
    if closures is None:
        raise RuntimeError(f"register_{prefix}_closure called outside of {prefix}_fusion_context")
    if buffer_name not in closures:
        closures[buffer_name] = f"{prefix}_closure_{len(closures)}"
    return closures[buffer_name]


def register_epilogue_closure(buffer_name: str) -> str:
    """Register an external buffer as an epilogue closure."""
    return _register_closure(_epilogue_closures, "epilogue", buffer_name)


def register_prologue_closure(buffer_name: str) -> str:
    """Register an external buffer as a prologue closure."""
    return _register_closure(_prologue_closures, "prologue", buffer_name)


@contextlib.contextmanager
def epilogue_fusion_context(
    epilogues: dict[str, list["EpilogueSpec"]],
    store_index_to_buffer: dict[int, str] | None = None,
) -> Iterator[dict[str, str]]:
    """Context manager for epilogue fusion during code generation."""
    token = _epilogue_context.set(epilogues)
    closures: dict[str, str] = {}
    closures_token = _epilogue_closures.set(closures)
    store_token = _store_index_to_buffer.set(store_index_to_buffer or {})
    try:
        yield closures
    finally:
        _epilogue_context.reset(token)
        _epilogue_closures.reset(closures_token)
        _store_index_to_buffer.reset(store_token)


@contextlib.contextmanager
def prologue_fusion_context(
    prologues: dict[str, list["PrologueSpec"]],
) -> Iterator[dict[str, str]]:
    """Context manager for prologue fusion during code generation."""
    token = _prologue_context.set(prologues)
    closures: dict[str, str] = {}
    closures_token = _prologue_closures.set(closures)
    try:
        yield closures
    finally:
        _prologue_context.reset(token)
        _prologue_closures.reset(closures_token)


def get_epilogue_closures() -> dict[str, str]:
    return _epilogue_closures.get() or {}


def get_prologue_closures() -> dict[str, str]:
    return _prologue_closures.get() or {}


def get_epilogue_specs(tensor_name: str) -> list["EpilogueSpec"]:
    context = _epilogue_context.get()
    return context.get(tensor_name, []) if context else []


def get_prologue_specs(input_name: str) -> list["PrologueSpec"]:
    context = _prologue_context.get()
    return context.get(input_name, []) if context else []


def get_epilogue_specs_for_store(store_index: int) -> list["EpilogueSpec"]:
    """Get epilogue specs for a specific store index (multi-output templates)."""
    store_map = _store_index_to_buffer.get()
    if store_map is None or store_index not in store_map:
        return []
    return get_epilogue_specs(store_map[store_index])


def is_multi_output_epilogue_context() -> bool:
    store_map = _store_index_to_buffer.get()
    return bool(store_map)


def get_any_epilogue_specs() -> list["EpilogueSpec"]:
    """Get all epilogue specs regardless of tensor name."""
    context = _epilogue_context.get()
    if not context:
        return []
    return [spec for specs in context.values() for spec in specs]


@dataclass
class EpilogueSpec:
    """Specification for epilogue fusion."""
    epilogue_nodes: list["BaseSchedulerNode"]
    accumulator_name: str | set[str]

    @property
    def accumulator_names(self) -> set[str]:
        if isinstance(self.accumulator_name, set):
            return self.accumulator_name
        return {self.accumulator_name}


@dataclass
class PrologueSpec:
    """Specification for prologue fusion."""
    prologue_nodes: list["BaseSchedulerNode"]
    input_name: str


# Dtype mapping for type conversions
_TRITON_DTYPE_MAP: dict[torch.dtype, str] = {
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


class HelionOpsHandler(TritonOverrides):
    """String-based ops handler for Helion epilogue/prologue fusion.

    Inherits most ops from TritonOverrides (sin, cos, exp, log, sqrt, abs,
    floor, ceil, erf, sigmoid, etc.) which generate libdevice/tl_math calls.
    Inductor's codegen adds the necessary imports when it detects these in
    the generated code.

    Overrides only:
    - load/store: Helion-specific accumulator handling
    - constant/to_dtype/index_expr/indirect_indexing: avoid V.kernel/V.graph access
    - relu/maximum/minimum: use tl.* (different NaN handling than triton_helpers)
    - tanh: parent accesses V.kernel.cse for dtype checking
    - reciprocal: parent uses recursive ops.constant call
    - binary ops (add/sub/mul/etc.): add parentheses for correct precedence

    The handler generates Triton code as strings, then parses to AST at the end.
    """

    def __init__(
        self,
        accumulator_name: str | set[str],
        output_var: str,
        subscript_names: list[str],
        register_closure: Callable[[str], str],
    ) -> None:
        """Initialize the ops handler.

        Args:
            accumulator_name: Name(s) of buffer(s) containing kernel output.
            output_var: Variable name for the kernel output (e.g., "_val").
            subscript_names: Helion's index variable names, e.g., ["indices_0", "indices_1"].
            register_closure: Function to register external buffers as closures.
        """
        if isinstance(accumulator_name, set):
            self.accumulator_names = accumulator_name
        else:
            self.accumulator_names = {accumulator_name}
        self.output_var = output_var
        self.subscript_names = subscript_names
        self.register_closure = register_closure
        self.final_value: str = output_var

        # Build index translation map: i0 -> indices_0, etc.
        self._index_map: dict[str, str] = {}
        for i, sub_name in enumerate(subscript_names):
            self._index_map[f"i{i}"] = sub_name
            self._index_map[f"x{i}"] = sub_name

    def _translate_index(self, index: sympy.Expr) -> str:
        """Translate Inductor index to Helion index with broadcasting."""
        index_str = str(index)

        # Determine which dimensions are used
        dims_used = []
        for i, sub_name in enumerate(self.subscript_names):
            if f"i{i}" in index_str or f"x{i}" in index_str:
                dims_used.append(i)

        num_dims = len(self.subscript_names)

        # Multi-dim index needs broadcast slicing
        if len(dims_used) > 1 and num_dims > 1:
            for i, sub_name in enumerate(self.subscript_names):
                for prefix in ["i", "x"]:
                    inductor_var = f"{prefix}{i}"
                    if inductor_var in index_str:
                        slices = [":" if j == i else "None" for j in range(num_dims)]
                        sliced_var = f"{sub_name}[{', '.join(slices)}]"
                        index_str = re.sub(rf'\b{inductor_var}\b', sliced_var, index_str)
        else:
            for inductor_var, helion_var in self._index_map.items():
                index_str = re.sub(rf'\b{inductor_var}\b', helion_var, index_str)

        return index_str

    def _add_broadcast_slice(self, expr: str, translated_index: str) -> str:
        """Add broadcast slicing if the load doesn't use all dimensions."""
        dims_used = [i for i, sub_name in enumerate(self.subscript_names)
                     if re.search(rf'\b{re.escape(sub_name)}\b', translated_index)]

        if len(dims_used) < len(self.subscript_names):
            slices = []
            for i in range(len(self.subscript_names)):
                slices.append(":" if i in dims_used else "None")
            return f"{expr}[{', '.join(slices)}]"
        return expr

    # --- Core overrides (Helion-specific) ---

    def load(self, name: str, index: sympy.Expr) -> str:
        """Handle loads - return output var or load from closure."""
        if name in self.accumulator_names:
            return self.output_var

        # External tensor - register as closure
        closure_name = self.register_closure(name)
        translated_idx = self._translate_index(index)
        load_expr = f"tl.load({closure_name} + {translated_idx})"
        return self._add_broadcast_slice(load_expr, translated_idx)

    def store(self, name: str, index: Any, value: str, mode: Any = None) -> None:
        """Capture final value."""
        self.final_value = value

    def store_reduction(self, name: str, index: Any, value: str) -> None:
        """Capture final value from reduction."""
        self.final_value = value

    # --- Overrides to avoid V.kernel/V.graph access ---

    @staticmethod
    def constant(value: Any, dtype: torch.dtype) -> str:
        """Return constant as string."""
        return repr(value)

    def index_expr(self, expr: sympy.Expr, dtype: Any) -> str:
        """Return index expression as string."""
        return self._translate_index(expr)

    @staticmethod
    def to_dtype(x: str, dtype: torch.dtype, src_dtype: Any = None, use_compute_types: bool = True) -> str:
        """Type conversion without V.kernel access."""
        if dtype == torch.bool:
            return f"({x} != 0)"
        triton_dtype = _TRITON_DTYPE_MAP.get(dtype, "tl.float32")
        return f"({x}).to({triton_dtype})"

    @staticmethod
    def indirect_indexing(*args: Any, **kwargs: Any) -> str:
        """Handle indirect indexing - placeholder."""
        return "0"

    # --- Overrides for ops that need special handling ---
    # Parent TritonOverrides generates tl_math/libdevice/triton_helpers calls.
    # Inductor's codegen adds the necessary imports (tl_math, libdevice) when
    # it detects these in the generated code.
    #
    # We override only:
    # - relu/maximum/minimum: parent uses triton_helpers with different NaN handling
    # - tanh: parent accesses V.kernel.cse for dtype checking
    # - reciprocal: parent uses recursive ops.constant call

    @staticmethod
    def relu(x: str) -> str:
        """ReLU using tl.maximum (parent uses triton_helpers with different NaN handling)."""
        return f"tl.maximum(0, {x})"

    @staticmethod
    def maximum(a: str, b: str) -> str:
        """Maximum using tl.maximum (parent uses triton_helpers with different NaN handling)."""
        return f"tl.maximum({a}, {b})"

    @staticmethod
    def minimum(a: str, b: str) -> str:
        """Minimum using tl.minimum (parent uses triton_helpers with different NaN handling)."""
        return f"tl.minimum({a}, {b})"

    @staticmethod
    def tanh(x: str) -> str:
        """Tanh using libdevice - parent accesses V.kernel.cse for dtype checking."""
        return f"libdevice.tanh({x})"

    @staticmethod
    def reciprocal(x: str) -> str:
        """Reciprocal override - parent uses recursive ops.constant call."""
        return f"1.0 / ({x})"

    # --- Binary operations with parentheses to preserve operator precedence ---
    # BasicMathOpsMixin returns f"{a} + {b}" without parens, which breaks
    # when composed with higher-precedence ops like mul.

    @staticmethod
    def add(a: str, b: str) -> str:
        """Add with parentheses to preserve precedence."""
        return f"({a} + {b})"

    @staticmethod
    def sub(a: str, b: str) -> str:
        """Subtract with parentheses to preserve precedence."""
        return f"({a} - {b})"

    @staticmethod
    def mul(a: str, b: str) -> str:
        """Multiply with parentheses to preserve precedence."""
        return f"({a} * {b})"

    @staticmethod
    def truediv(a: str, b: str) -> str:
        """True division with parentheses to preserve precedence."""
        return f"({a} / {b})"

    @staticmethod
    def floordiv(a: str, b: str) -> str:
        """Floor division with parentheses to preserve precedence."""
        return f"({a} // {b})"

    @staticmethod
    def mod(a: str, b: str) -> str:
        """Modulo with parentheses to preserve precedence."""
        return f"({a} % {b})"

    def to_ast(self) -> python_ast.AST:
        """Parse final value string to AST."""
        return python_ast.parse(self.final_value, mode='eval').body


class _ASTSubstituter(python_ast.NodeTransformer):
    """Substitute a placeholder name with an AST node."""

    def __init__(self, placeholder: str, replacement: python_ast.AST):
        self.placeholder = placeholder
        self.replacement = replacement

    def visit_Name(self, node: python_ast.Name) -> python_ast.AST:
        return self.replacement if node.id == self.placeholder else node


def _apply_fusion_ast(
    value: python_ast.AST,
    subscript_names: list[str],
    specs: list[Union[EpilogueSpec, PrologueSpec]],
    register_closure: Callable[[str], str],
    fusion_type: str,
) -> python_ast.AST:
    """Apply fusion operations (epilogue or prologue) to a value using AST manipulation."""
    if not specs:
        return value

    from torch._inductor.virtualized import OpsValue

    current_value = value
    placeholder = f"_hl_{fusion_type}_val_"

    for spec in specs:
        # Get nodes and accumulator name based on spec type
        if isinstance(spec, EpilogueSpec):
            nodes, acc_name = spec.epilogue_nodes, spec.accumulator_name
        else:
            nodes, acc_name = spec.prologue_nodes, spec.input_name

        if not nodes:
            continue

        handler = HelionOpsHandler(
            accumulator_name=acc_name,
            output_var=placeholder,
            subscript_names=subscript_names,
            register_closure=register_closure,
        )

        try:
            with V.set_ops_handler(handler):
                for s_node in nodes:
                    node = s_node.node
                    if isinstance(node, ComputedBuffer):
                        data = node.data
                        if isinstance(data, Pointwise):
                            result = data.inner_fn(data._index(data.ranges))
                            if isinstance(result, OpsValue):
                                handler.final_value = result.value
                            elif isinstance(result, str):
                                handler.final_value = result
        except Exception as e:
            raise RuntimeError(
                f"{fusion_type.capitalize()} fusion failed for '{acc_name}': {e}"
            ) from e

        substituter = _ASTSubstituter(placeholder, current_value)
        current_value = substituter.visit(handler.to_ast())

    return current_value


def apply_epilogues_ast(
    value: python_ast.AST,
    subscript_names: list[str],
    epilogue_specs: list[EpilogueSpec],
    device_function: Any = None,
    output_dtype: Optional[torch.dtype] = None,
) -> python_ast.AST:
    """Apply epilogue operations to a value using AST manipulation."""
    return _apply_fusion_ast(
        value, subscript_names, epilogue_specs, register_epilogue_closure, "epilogue"
    )


def apply_prologues_ast(
    value: python_ast.AST,
    subscript_names: list[str],
    prologue_specs: list[PrologueSpec],
    input_buffer_name: str,
    device_function: Any = None,
) -> python_ast.AST:
    """Apply prologue operations to a loaded value using AST manipulation."""
    return _apply_fusion_ast(
        value, subscript_names, prologue_specs, register_prologue_closure, "prologue"
    )


def _is_realized_node(node: IRNode) -> bool:
    """Check if a node is already realized (materialized in memory)."""
    return isinstance(
        node,
        (
            ComputedBuffer,
            InputBuffer,
            ReinterpretView,
            TemplateBuffer,
            Buffer,
        ),
    )


def _realize_input(tensor_box: TensorBox) -> IRNode:
    """Realize a tensor input if it's not already materialized.

    This handles Pointwise, Reduction, and other computed nodes by
    materializing them into buffers before use.
    """
    if isinstance(tensor_box, TensorBox):
        data = tensor_box.data
        if isinstance(data, StorageBox):
            inner_data = data.data
            if _is_realized_node(inner_data):
                return inner_data
            # Realize the storage box - this converts Pointwise etc to ComputedBuffer
            data.realize()
            return data.data
        elif _is_realized_node(data):
            return data
        else:
            # Try to realize via the tensor box
            tensor_box.realize()
            if isinstance(tensor_box.data, StorageBox):
                return tensor_box.data.data
            return tensor_box.data
    return tensor_box  # type: ignore[return-value]


class HelionKernelBuffer(TritonTemplateBuffer):
    """Represents a Helion kernel call in Inductor's IR.

    This generates inline Triton code for the Helion kernel.
    Extends TritonTemplateBuffer to support epilogue and prologue fusion.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        kernel: Any,  # Helion Kernel object
        kernel_idx: int,
        constant_args: dict[str, Any],
        tensor_arg_names: list[str],
        bound_kernel: Any,  # BoundKernel object
        make_kernel_render: Optional[Callable[..., Any]] = None,
        allowed_prologue_inps: Optional[OrderedSet[str]] = None,
    ) -> None:
        # Store these before super().__init__ since we need them in make_kernel_render
        self.helion_kernel = kernel
        self.kernel_idx = kernel_idx
        self.constant_args_dict = constant_args
        self.tensor_arg_names = tensor_arg_names
        self.bound_kernel = bound_kernel

        # Expose output iteration ranges for fusion decisions
        # These are the symbolic dimensions of the output tensor
        # e.g., [m, n] from hl.tile([m, n]) in the Helion kernel
        # Range matching: if epilogue.get_size() == output_ranges, fusion is valid
        # Note: MultiOutputLayout doesn't have a size attribute, so we initialize to empty
        # and let it be set later when outputs are created
        if hasattr(layout, 'size'):
            self.output_ranges: list[sympy.Expr] = list(layout.size)
        else:
            self.output_ranges = []

        # Track epilogue specs added by Inductor's scheduler
        # Maps output tensor name -> list of EpilogueSpec
        self.epilogue_specs: dict[str, list[EpilogueSpec]] = {}

        # Track prologue specs added by Inductor's scheduler
        # Maps input tensor name -> list of PrologueSpec
        self.prologue_specs: dict[str, list[PrologueSpec]] = {}

        # Cache for atomic detection
        self._uses_atomics: Optional[bool] = None

        # Create make_kernel_render if not provided
        if make_kernel_render is None:
            make_kernel_render = self._create_kernel_render()

        # By default, allow prologue fusion on all inputs
        # The scheduler uses this to determine if a prologue can be fused
        if allowed_prologue_inps is None:
            # Allow all tensor inputs to have prologues
            allowed_prologue_inps = OrderedSet(
                inp.get_name() for inp in inputs if hasattr(inp, 'get_name')
            )

        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=make_kernel_render,
            mutated_inputs=None,  # Helion handles mutations differently
            allowed_prologue_inps=allowed_prologue_inps,
        )

        # After buffer is registered (has a name), check for atomics and prevent
        # fusion if needed. This must happen before the scheduler runs.
        if self.uses_atomics():
            # Add to no_fuse_buffer_names to prevent epilogue fusion at scheduler level
            V.graph.no_fuse_buffer_names.add(self.get_name())

    def add_epilogue(self, output_name: str, spec: "EpilogueSpec") -> None:
        """Add an epilogue spec for an output tensor.

        Called by Inductor's scheduler when fusing epilogues.
        """
        if output_name not in self.epilogue_specs:
            self.epilogue_specs[output_name] = []
        self.epilogue_specs[output_name].append(spec)

    def add_prologue(self, input_name: str, spec: "PrologueSpec") -> None:
        """Add a prologue spec for an input tensor.

        Called by Inductor's scheduler when fusing prologues.

        Args:
            input_name: The name of the input tensor parameter (e.g., "x", "y").
            spec: The PrologueSpec containing the prologue operations.
        """
        if input_name not in self.prologue_specs:
            self.prologue_specs[input_name] = []
        self.prologue_specs[input_name].append(spec)

    def _create_kernel_render(self) -> Callable[..., Any]:
        """Create the kernel render function for this Helion kernel."""
        from .helion_template import HelionTemplateKernel, PartialHelionRender

        def make_render(
            template_buffer: "HelionKernelBuffer", hint_override: Optional[int] = None
        ) -> tuple["HelionTemplateKernel", "PartialHelionRender"]:
            kernel = HelionTemplateKernel(template_buffer)
            render = PartialHelionRender(template_buffer, kernel)
            return kernel, render

        return make_render

    def get_kernel_name(self) -> str:
        """Return the kernel name to call."""
        return f"helion_kernel_{self.kernel_idx}"

    def get_layout(self) -> Layout:
        """Return the layout for this buffer.

        For MultiOutputLayout, returns the layout of the first MultiOutput node.
        This is needed for proper multi-output codegen.
        """
        if isinstance(self.layout, MultiOutputLayout):
            # For multi-output, get the layout from the first MultiOutput node
            multi_output_nodes = getattr(self, 'multi_output_nodes', None)
            if multi_output_nodes is not None and len(multi_output_nodes) > 0:
                first_output = multi_output_nodes[0]
                if hasattr(first_output, 'layout') and isinstance(first_output.layout, Layout):
                    return first_output.layout
            # Fallback: this shouldn't happen if multi_output_nodes is set correctly
            raise RuntimeError(
                "HelionKernelBuffer with MultiOutputLayout has no multi_output_nodes set. "
                "This is a bug in the lowering."
            )
        return super().get_layout()

    def uses_atomics(self) -> bool:
        """Check if this kernel uses atomic operations.

        Atomic operations (tl.atomic_add, etc.) cannot have epilogue fusion
        because atomics may execute multiple times per element. The epilogue
        must run as a separate kernel after all atomics complete.

        Returns:
            True if the kernel uses atomic operations, False otherwise.
        """
        if self._uses_atomics is not None:
            return self._uses_atomics

        # Generate the base Triton code and check for atomic patterns
        if self.bound_kernel is None:
            self._uses_atomics = False
            return False

        try:
            from .generate_ast import generate_ast
            from .ast_extension import unparse
            from ..runtime.config import Config

            config = self.bound_kernel._require_implicit_config()
            if not isinstance(config, Config):
                config = Config(**config)
            self.bound_kernel.env.config_spec.normalize(config)

            with self.bound_kernel.env:
                with epilogue_fusion_context({}) as _:
                    root = generate_ast(
                        self.bound_kernel.host_function, config, emit_repro_caller=False
                    )
                base_code = unparse(root, output_origin_lines=False)
                self._uses_atomics = "tl.atomic_" in base_code
        except Exception:
            # If we can't generate the code, assume no atomics
            self._uses_atomics = False

        return self._uses_atomics

    def supports_epilogue_fusion(self) -> bool:
        """Check if this kernel supports epilogue fusion.

        Returns False for kernels with atomic operations, which cannot have
        epilogue fusion because atomics may execute multiple times per element.

        Returns:
            True if epilogue fusion is supported, False otherwise.
        """
        return not self.uses_atomics()


def _get_helion_kernel(kernel_idx: int) -> Any:
    """Get a Helion kernel from the Dynamo side table."""
    from torch._higher_order_ops.helion_kernel_wrap import get_helion_kernel
    return get_helion_kernel(kernel_idx)


from torch._higher_order_ops.helion_kernel_wrap import helion_kernel_wrapper_mutation as _helion_hop


@register_lowering(
    _helion_hop,
    type_promotion_kind=None,
)
def lower_helion_kernel_wrapper_mutation(
    *,
    kernel_idx: int,
    constant_args: dict[str, Any],
    tensor_args: dict[str, TensorBox],
    output_spec: dict[str, Any],
) -> tuple[TensorBox, ...]:
    """Lower helion_kernel_wrapper_mutation HOP to Inductor IR.

    This creates a HelionKernelBuffer that represents the Helion kernel call
    in Inductor's scheduling graph.
    """
    # Get the kernel from the side table
    kernel = _get_helion_kernel(kernel_idx)

    # Realize tensor args to ensure they are materialized buffers
    # This handles cases where inputs are Pointwise/Reduction nodes
    tensor_inputs = []
    tensor_arg_names = []
    for name, tensor_box in tensor_args.items():
        if isinstance(tensor_box, TensorBox):
            realized_input = _realize_input(tensor_box)
            tensor_inputs.append(realized_input)
            tensor_arg_names.append(name)

    # Get output info from spec
    output_dtype = output_spec.get("dtype", torch.float32)
    device_str = output_spec.get("device", "cuda")
    device = torch.device(device_str)
    num_outputs = output_spec.get("num_outputs", 1)
    output_shape = output_spec.get("shape", [])

    # Create fake tensors to bind the kernel
    fake_tensors = []
    sig_params = kernel.signature.parameters
    param_names = list(sig_params.keys())
    for name in param_names:
        if name in tensor_arg_names:
            idx = tensor_arg_names.index(name)
            tb = list(tensor_args.values())[idx]
            if hasattr(tb, 'get_size'):
                size = [int(s) if isinstance(s, (int, sympy.Integer)) else 64 for s in tb.get_size()]
                fake_tensors.append(torch.empty(size, dtype=output_dtype, device=device))
            else:
                fake_tensors.append(torch.empty(1, dtype=output_dtype, device=device))
        elif name in constant_args:
            fake_tensors.append(constant_args[name])
        else:
            # Handle parameters with default values
            param = sig_params.get(name)
            if param is not None and param.default is not param.empty:
                fake_tensors.append(param.default)

    # Bind the kernel to get the BoundKernel for code generation
    bound_kernel = None
    try:
        bound_kernel = kernel.bind(tuple(fake_tensors))
    except Exception as e:
        import traceback
        traceback.print_exc()

    # Handle single vs multi-output differently
    if num_outputs == 1:
        # Single output: simple case
        output_layout = FixedLayout(
            device=device,
            dtype=output_dtype,
            size=output_shape,
        )

        helion_buffer = HelionKernelBuffer(
            layout=output_layout,
            inputs=tensor_inputs,
            kernel=kernel,
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_arg_names=tensor_arg_names,
            bound_kernel=bound_kernel,
        )

        return (TensorBox(StorageBox(helion_buffer)),)
    else:
        # Multi-output: use MultiOutputLayout and MultiOutput nodes
        # Create base buffer with MultiOutputLayout
        helion_buffer = HelionKernelBuffer(
            layout=MultiOutputLayout(device=device),
            inputs=tensor_inputs,
            kernel=kernel,
            kernel_idx=kernel_idx,
            constant_args=constant_args,
            tensor_arg_names=tensor_arg_names,
            bound_kernel=bound_kernel,
        )

        # Create a MultiOutput node for each output
        # Each output gets the same layout (since output_spec only has one shape)
        # TODO: Support per-output shapes when output_spec is extended
        output_layout = FixedLayout(
            device=device,
            dtype=output_dtype,
            size=output_shape,
        )

        multi_outputs = []
        for i in range(num_outputs):
            # indices indicate the path to this output in the result structure
            # For a simple tuple[Tensor, Tensor], it's [(tuple, 0)] and [(tuple, 1)]
            multi_out = MultiOutput(
                layout=output_layout,
                input=helion_buffer,
                indices=[(tuple, i)],
            )
            multi_outputs.append(multi_out)

        # Store the MultiOutput nodes for get_layout() to use
        # NOTE: Do NOT add multi_outputs to self.outputs - they're separate scheduler nodes
        # and would cause duplicate names in name_to_buf.
        # self.outputs stays as [self] (default from TritonTemplateBuffer) so buf0 is in name_to_buf
        helion_buffer.multi_output_nodes = multi_outputs

        # Return TensorBox-wrapped MultiOutput nodes
        return tuple(TensorBox.create(mo) for mo in multi_outputs)


def register_helion_lowerings() -> None:
    """Register all Helion lowerings with Inductor.

    This function should be called during initialization to ensure
    Helion kernels can be lowered by Inductor.
    """
    # The @register_lowering decorator already registers the lowering
    # when this module is imported. This function is provided for
    # explicit registration if needed.
    pass
