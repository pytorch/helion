"""Helion template kernel for Inductor integration.

This module provides the template kernel interface that enables epilogue fusion
for Helion kernels when used with torch.compile.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Sequence

import sympy

from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

if TYPE_CHECKING:
    from .helion_inductor import HelionKernelBuffer


def _inject_closure_params_ast(
    root: ast.Module,
    closures: dict[str, str],
    triton_kernel_name: str,
    host_fn_name: str,
) -> None:
    """Inject epilogue closure parameters into AST (not string regex).

    Modifies the AST in-place to add closure parameters to:
    1. The @triton.jit kernel function signature
    2. The host launcher function signature
    3. The _launcher call within the host function

    Args:
        root: The AST module to modify.
        closures: Maps buffer name -> parameter name (e.g., {"arg2_1": "epilogue_closure_0"}).
        triton_kernel_name: Name of the @triton.jit kernel (e.g., "_helion_matmul").
        host_fn_name: Name of the host launcher function (e.g., "matmul").
    """
    if not closures:
        return

    closure_param_names = list(closures.values())

    # Track which functions we've already modified to avoid duplicates
    modified_triton = False
    modified_host = False

    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef):
            if node.name == triton_kernel_name and not modified_triton:
                # This is the @triton.jit kernel function
                _add_closure_params_to_triton_kernel(node, closure_param_names)
                modified_triton = True
            elif node.name == host_fn_name and not modified_host:
                # This is the host launcher function
                _add_closure_params_to_host_fn(node, closure_param_names)
                _add_closure_args_to_launcher_call(node, triton_kernel_name, closure_param_names)
                modified_host = True

    # Validate that we found and modified the expected functions
    # Silent failure here would lead to incorrect runtime behavior
    if not modified_triton:
        raise RuntimeError(
            f"Failed to inject closure params: Triton kernel '{triton_kernel_name}' not found in AST. "
            f"Closures: {list(closures.keys())}"
        )
    if not modified_host:
        raise RuntimeError(
            f"Failed to inject closure params: Host function '{host_fn_name}' not found in AST. "
            f"Closures: {list(closures.keys())}"
        )


def _add_closure_params_to_triton_kernel(func_def: ast.FunctionDef, closure_params: list[str]) -> None:
    """Add closure parameters to @triton.jit kernel before _BLOCK_SIZE params."""
    args = func_def.args

    # Check for existing params to avoid duplicates
    existing_param_names = {arg.arg for arg in args.args}
    params_to_add = [name for name in closure_params if name not in existing_param_names]

    if not params_to_add:
        return

    # Find insertion point: before the first _BLOCK_SIZE parameter
    insert_idx = len(args.args)
    for i, arg in enumerate(args.args):
        if arg.arg.startswith("_BLOCK_SIZE"):
            insert_idx = i
            break

    # Create new arg nodes for closures
    new_args = [ast.arg(arg=name, annotation=None) for name in params_to_add]

    # Insert closure params
    args.args = args.args[:insert_idx] + new_args + args.args[insert_idx:]


def _add_closure_params_to_host_fn(func_def: ast.FunctionDef, closure_params: list[str]) -> None:
    """Add closure parameters to host launcher function with torch.Tensor annotation."""
    args = func_def.args

    # Check for existing params to avoid duplicates
    existing_param_names = {arg.arg for arg in args.args}

    # Create torch.Tensor annotation
    tensor_annotation = ast.Attribute(
        value=ast.Name(id="torch", ctx=ast.Load()),
        attr="Tensor",
        ctx=ast.Load(),
    )

    # Add closure params at the end of args (only if not already present)
    for name in closure_params:
        if name not in existing_param_names:
            new_arg = ast.arg(arg=name, annotation=tensor_annotation)
            args.args.append(new_arg)


def _add_closure_args_to_launcher_call(
    func_def: ast.FunctionDef,
    triton_kernel_name: str,
    closure_params: list[str],
) -> None:
    """Add closure arguments to the _launcher call inside the host function."""

    class LauncherCallVisitor(ast.NodeTransformer):
        """Find and modify the _launcher call."""

        def __init__(self) -> None:
            self.modified = False

        def visit_Call(self, node: ast.Call) -> ast.Call:
            # Only modify once
            if self.modified:
                return node

            # Check if this is a _launcher call with our kernel
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_launcher"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == triton_kernel_name
            ):
                # Check for existing closure args to avoid duplicates
                existing_args = {
                    arg.id for arg in node.args
                    if isinstance(arg, ast.Name)
                }
                args_to_add = [name for name in closure_params if name not in existing_args]

                if not args_to_add:
                    return node

                # Find insertion point: before _BLOCK_SIZE args
                insert_idx = len(node.args)
                for i, arg in enumerate(node.args):
                    if isinstance(arg, ast.Name) and arg.id.startswith("_BLOCK_SIZE"):
                        insert_idx = i
                        break

                # Create closure arg nodes
                closure_args = [ast.Name(id=name, ctx=ast.Load()) for name in args_to_add]

                # Insert closure args
                node.args = node.args[:insert_idx] + closure_args + node.args[insert_idx:]
                self.modified = True

            return node

    visitor = LauncherCallVisitor()
    visitor.visit(func_def)


def _rename_host_function_ast(root: ast.Module, old_name: str, new_name: str) -> None:
    """Rename the host function in the AST."""
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef) and node.name == old_name:
            node.name = new_name
            break


class PartialHelionRender:
    """Partial render for Helion kernel that can be finalized with epilogue fusion."""

    def __init__(
        self,
        template_buffer: "HelionKernelBuffer",
        kernel: "HelionTemplateKernel",
    ) -> None:
        self.template_buffer = template_buffer
        self.kernel = kernel
        self._triton_code: Optional[str] = None
        self._replacement_hooks: dict[str, Any] = {}

    def __call__(self) -> "PartialHelionRender":
        """Return self - finalization happens via finalize_hook."""
        # Generate the initial triton code
        try:
            self._generate_triton_code()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        return self

    @property
    def code(self) -> str:
        """Return the finalized code."""
        return self._triton_code or ""

    def finalize_hook(self, hook_name: str, strict: bool = True) -> None:
        """Finalize hook called after epilogue nodes are added."""
        # Mark hook as finalized
        if hook_name in self._replacement_hooks:
            self._replacement_hooks[hook_name] = None

    def finalize_remaining(self) -> str:
        """Finalize any remaining hooks and return the code."""
        return self._triton_code or ""

    def finalize_all(self) -> str:
        """Finalize all hooks and return the code."""
        return self._triton_code or ""

    def _generate_triton_code(self) -> None:
        """Generate the Triton code for this kernel.

        Uses AST-based manipulation (not regex) for closure injection.
        The generated code uses Placeholder.KERNEL_NAME for the host launcher function,
        which Inductor's define_kernel will replace with the actual kernel name.

        Note: Kernels with atomic operations (tl.atomic_*) cannot have epilogue fusion
        because atomics may be executed multiple times per element. We detect this by
        generating the base code first and checking for atomic patterns.

        Graceful fallback: If epilogue/prologue fusion fails for any reason, we fall back to
        generating code without fusion rather than failing the compilation.
        """
        from torch._inductor.utils import Placeholder
        from .generate_ast import generate_ast
        from .ast_extension import unparse
        from .output_header import get_needed_imports
        from .helion_inductor import epilogue_fusion_context, prologue_fusion_context
        from ..runtime.config import Config
        import logging

        log = logging.getLogger(__name__)

        bound_kernel = self.template_buffer.bound_kernel
        if bound_kernel is None:
            return

        # Get config for code generation
        config = bound_kernel._require_implicit_config()
        if not isinstance(config, Config):
            config = Config(**config)
        bound_kernel.env.config_spec.normalize(config)

        # Get function names for AST manipulation
        host_fn_name = self.template_buffer.helion_kernel.name
        triton_kernel_name = f"_helion_{host_fn_name}"
        placeholder_name = str(Placeholder.KERNEL_NAME)

        # First, generate base code WITHOUT epilogue/prologue to check for atomics
        with bound_kernel.env:
            with epilogue_fusion_context({}) as _:
                with prologue_fusion_context({}) as _:
                    base_root = generate_ast(bound_kernel.host_function, config, emit_repro_caller=False)

            # Check if the kernel uses atomic operations
            base_code = unparse(base_root, output_origin_lines=False)
            uses_atomics = "tl.atomic_" in base_code

        # Get epilogue specs (if any were added by Inductor's scheduler)
        epilogue_specs = getattr(self.template_buffer, 'epilogue_specs', {})

        # Get prologue specs (if any were added by Inductor's scheduler)
        prologue_specs = getattr(self.template_buffer, 'prologue_specs', {})

        # If kernel uses atomics, skip epilogue fusion - atomics may execute
        # multiple times per element, so fusing epilogue would cause incorrect results
        if uses_atomics and epilogue_specs:
            log.debug("Skipping epilogue fusion for kernel with atomic operations: %s", host_fn_name)
            epilogue_specs = {}

        # For multi-output templates, set up mapping from store index to buffer name
        # This allows each store to get only its own epilogues
        from torch._inductor.ir import MultiOutputLayout
        store_index_to_buffer: dict[int, str] = {}
        if isinstance(self.template_buffer.layout, MultiOutputLayout):
            # Use multi_output_nodes (buf1, buf2, ...) not outputs (which is just [buf0])
            # The multi_output_nodes contain the actual IR buffer names that epilogues reference
            multi_output_nodes = getattr(self.template_buffer, 'multi_output_nodes', [])
            for i, mo_node in enumerate(multi_output_nodes):
                if hasattr(mo_node, 'get_name'):
                    store_index_to_buffer[i] = mo_node.get_name()

        # Try to generate with epilogue/prologue fusion, fall back to no fusion on error
        fusion_succeeded = False
        if epilogue_specs or prologue_specs:
            try:
                with bound_kernel.env:
                    with epilogue_fusion_context(epilogue_specs, store_index_to_buffer) as epilogue_closures:
                        with prologue_fusion_context(prologue_specs) as prologue_closures:
                            root = generate_ast(bound_kernel.host_function, config, emit_repro_caller=False)

                    # Merge all closures
                    all_closures = {**epilogue_closures, **prologue_closures}

                    if all_closures:
                        _inject_closure_params_ast(root, all_closures, triton_kernel_name, host_fn_name)

                        for buf_name, param_name in epilogue_closures.items():
                            self.kernel.epilogue_closures[buf_name] = {
                                "param_name": param_name,
                                "tensor": None,
                            }
                        # Track prologue closures separately if needed
                        for buf_name, param_name in prologue_closures.items():
                            if buf_name not in self.kernel.epilogue_closures:
                                self.kernel.epilogue_closures[buf_name] = {
                                    "param_name": param_name,
                                    "tensor": None,
                                }

                    _rename_host_function_ast(root, host_fn_name, placeholder_name)

                    self._triton_code = get_needed_imports(root) + unparse(
                        root, output_origin_lines=bound_kernel.settings.output_origin_lines
                    )
                    fusion_succeeded = True
            except Exception as e:
                log.warning(
                    "Epilogue/Prologue fusion failed for kernel %s, falling back to no fusion: %s",
                    host_fn_name, e
                )
                # Clear any partial state
                self.kernel.epilogue_closures.clear()
                fusion_succeeded = False

        # Generate without fusion if no epilogue/prologue specs or fusion failed
        if not fusion_succeeded:
            with bound_kernel.env:
                with epilogue_fusion_context({}) as _:
                    with prologue_fusion_context({}) as _:
                        root = generate_ast(bound_kernel.host_function, config, emit_repro_caller=False)

                _rename_host_function_ast(root, host_fn_name, placeholder_name)

                self._triton_code = get_needed_imports(root) + unparse(
                    root, output_origin_lines=bound_kernel.settings.output_origin_lines
                )


class HelionCSE:
    """Minimal CSE implementation for Helion templates."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def invalidate(self, names: OrderedSet[str]) -> None:
        """Invalidate cached values."""
        for name in names:
            self._cache.pop(name, None)


class HelionTemplateKernel:
    """Template kernel wrapper for Helion kernels.

    This provides the minimal interface expected by Inductor's codegen_template
    for epilogue fusion support. Does not inherit from TritonKernel to avoid
    complexity.
    """

    def __init__(
        self,
        template_buffer: "HelionKernelBuffer",
    ) -> None:
        self.template_buffer = template_buffer

        # Get output size for numel calculation
        output_size = template_buffer.get_size()
        self._output_size = output_size

        # Track epilogue closures (external tensors referenced in epilogue)
        self.epilogue_closures: dict[str, Any] = OrderedDict()
        self._closure_counter = 0

        # For prologue fusion
        self.prologue_fused_inputs: set[str] = set()
        self.prologue_fused_inputs_preserve_zero: set[str] = set()

        # Named input nodes for prologue - map parameter names to input IR nodes
        # This is used by Inductor's scheduler to identify which inputs have prologues
        self.named_input_nodes: dict[str, Any] = {}
        # Populate from the template buffer's tensor_arg_names and inputs
        for i, name in enumerate(template_buffer.tensor_arg_names):
            if i < len(template_buffer.inputs):
                self.named_input_nodes[name] = template_buffer.inputs[i]

        # Store subgraph bodies for epilogue/prologue
        self._subgraph_bodies: dict[str, list[str]] = {}
        self._current_subgraph: Optional[str] = None

        # Track removed buffers
        self.removed_buffers: set[str] = set()
        self.inplaced_to_remove: set[str] = set()

        # CSE instance
        self.cse = HelionCSE()

        # Kernel name
        self.kernel_name: Optional[str] = None

    def __enter__(self) -> "HelionTemplateKernel":
        """Context manager entry - set up kernel handler."""
        self._prev_kernel_handler = V.set_kernel_handler(self)
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - restore previous kernel handler."""
        V.set_kernel_handler(self._prev_kernel_handler)

    def set_current_node(self, node: Any) -> "CurrentNodeContext":
        """Set the current node being codegen'd."""
        return CurrentNodeContext(self, node)

    def get_store_output_count(self) -> int:
        """Return number of store output subgraphs (1 for single output)."""
        return 1

    def _get_store_output_subgraph_name(self, index: int) -> str:
        """Get the name for a store output subgraph."""
        return f"<STORE_OUTPUT_{index}>"

    def set_subgraph_body(self, name: str) -> "SubgraphContext":
        """Set the current subgraph for code generation."""
        return SubgraphContext(self, name)

    def split_and_set_ranges(
        self, lengths: Sequence[Sequence[sympy.Expr]]
    ) -> list[list[sympy.Expr]]:
        """Split ranges and return index variables for epilogue codegen.

        Args:
            lengths: A sequence of sequences of symbolic expressions representing
                    the sizes of different dimensions for each node.

        Returns:
            A list of lists of symbolic expressions representing the mapped
            iteration variables for each dimension.
        """
        # For epilogue fusion, we need to provide index variables that
        # correspond to the output tile dimensions
        # Return a list of lists, matching the structure of lengths
        result: list[list[sympy.Expr]] = []
        idx_counter = 0
        for range_group in lengths:
            group_vars: list[sympy.Expr] = []
            for _ in range_group:
                # Create a symbolic variable for this dimension
                idx_var = sympy.Symbol(f"idx{idx_counter}")
                group_vars.append(idx_var)
                idx_counter += 1
            result.append(group_vars)
        return result

    def add_epilogue_closure(self, buffer_name: str, tensor: Any) -> str:
        """Add an external tensor as an epilogue closure parameter.

        Returns the parameter name to use in the kernel.
        """
        if buffer_name not in self.epilogue_closures:
            param_name = f"epilogue_closure_{self._closure_counter}"
            self._closure_counter += 1
            self.epilogue_closures[buffer_name] = {
                "param_name": param_name,
                "tensor": tensor,
            }
        return self.epilogue_closures[buffer_name]["param_name"]

    def call_kernel(self, kernel_name: str, template_buffer: Any = None) -> None:
        """Generate the kernel call in the wrapper code.

        The kernel code has already been emitted by define_kernel() - this method
        only emits the actual call to the kernel function.
        """
        wrapper = V.graph.wrapper_code

        # Get output buffer name
        output_name = self.template_buffer.get_name()

        # Get input tensor names
        input_names = [inp.get_name() for inp in self.template_buffer.inputs]

        # Build the argument list in the correct order
        helion_kernel = self.template_buffer.helion_kernel
        param_names = list(helion_kernel.signature.parameters.keys())
        tensor_arg_names = self.template_buffer.tensor_arg_names
        constant_args_dict = self.template_buffer.constant_args_dict

        args = []
        input_idx = 0
        sig_params = helion_kernel.signature.parameters
        for name in param_names:
            if name in tensor_arg_names:
                if input_idx < len(input_names):
                    args.append(input_names[input_idx])
                    input_idx += 1
            elif name in constant_args_dict:
                args.append(repr(constant_args_dict[name]))
            else:
                # Check if the parameter has a default value in the kernel signature
                param = sig_params.get(name)
                if param is not None and param.default is not param.empty:
                    args.append(repr(param.default))

        # Add epilogue closure arguments
        # buf_name is the Inductor buffer name (e.g., "arg2_1") that's in scope
        # param_name is what the kernel function parameter is called
        for buf_name, closure_info in self.epilogue_closures.items():
            args.append(buf_name)

        # Generate the kernel call using the kernel_name from define_kernel()
        # The kernel function was already emitted to wrapper.header by define_kernel()
        wrapper.writeline(
            f"{output_name} = {kernel_name}({', '.join(args)})"
        )

        # For Helion kernels with fused epilogues, add alias lines so that
        # epilogue buffer names point to the same tensor as the kernel output.
        # This is needed because the graph's output references the epilogue buffer.
        #
        # For multi-output templates, the kernel returns a tuple and we need to
        # index into it (e.g., buf1 = buf0[0], buf2 = buf0[1]) instead of just
        # aliasing (buf1 = buf0).
        from torch._inductor.ir import MultiOutputLayout, MultiOutput

        epilogue_aliases = getattr(self, '_helion_epilogue_aliases', [])
        is_multi_output = isinstance(self.template_buffer.layout, MultiOutputLayout)

        if is_multi_output:
            # For multi-output templates, generate unpacking code for each MultiOutput node.
            # The kernel returns a tuple, and we need to extract each element.
            #
            # Get MultiOutput nodes from multi_output_nodes attribute (set during lowering)
            multi_output_nodes = getattr(self.template_buffer, 'multi_output_nodes', [])

            # Generate unpacking code for each MultiOutput node
            # e.g., buf1 = buf0[0], buf2 = buf0[1]
            for mo_node in multi_output_nodes:
                if hasattr(mo_node, 'get_name') and hasattr(mo_node, 'indices'):
                    mo_name = mo_node.get_name()
                    indices = mo_node.indices
                    if indices:
                        # indices is like [(tuple, 0)] for the first output
                        index_str = output_name
                        for itype, idx in indices:
                            index_str = f"{index_str}[{idx}]"
                        wrapper.writeline(f"{mo_name} = {index_str}")
                    else:
                        # No indices, just alias directly
                        wrapper.writeline(f"{mo_name} = {output_name}")

            # Also handle any epilogue aliases that reference the outputs
            outputs = getattr(self.template_buffer, 'outputs', [])
            epilogue_specs = getattr(self.template_buffer, 'epilogue_specs', {})

            # Build mapping from epilogue buffer name -> output buffer name
            epilogue_to_output: dict[str, str] = {}
            for output_buf_name, specs in epilogue_specs.items():
                for spec in specs:
                    for ep_node in spec.epilogue_nodes:
                        if hasattr(ep_node, 'node') and hasattr(ep_node.node, 'get_name'):
                            ep_buf_name = ep_node.node.get_name()
                            epilogue_to_output[ep_buf_name] = output_buf_name
                        elif hasattr(ep_node, 'get_name'):
                            ep_buf_name = ep_node.get_name()
                            epilogue_to_output[ep_buf_name] = output_buf_name

            # Build a set of output names for quick lookup
            output_names = {out.get_name() for out in outputs}
            mo_names = {mo.get_name() for mo in multi_output_nodes if hasattr(mo, 'get_name')}

            for alias_name in epilogue_aliases:
                # Skip if already handled (main output or MultiOutput nodes)
                if alias_name == output_name or alias_name in mo_names:
                    continue

                # Find the MultiOutput node for indexing
                multi_output_node = None

                # Check if this alias is a MultiOutput node itself FIRST
                if alias_name in output_names:
                    for out in outputs:
                        if out.get_name() == alias_name:
                            multi_output_node = out
                            break
                else:
                    # Check if this is an epilogue buffer (reads from a MultiOutput)
                    source_output_name = epilogue_to_output.get(alias_name)
                    if source_output_name:
                        # Look in multi_output_nodes, not outputs, since source_output_name
                        # is the name of a MultiOutput IR node (like buf1, buf2)
                        for mo in multi_output_nodes:
                            if hasattr(mo, 'get_name') and mo.get_name() == source_output_name:
                                multi_output_node = mo
                                break

                if multi_output_node is not None and hasattr(multi_output_node, 'indices'):
                    indices = multi_output_node.indices
                    if indices:
                        index_str = output_name
                        for itype, idx in indices:
                            index_str = f"{index_str}[{idx}]"
                        wrapper.writeline(f"{alias_name} = {index_str}")
                    else:
                        wrapper.writeline(f"{alias_name} = {output_name}")
                else:
                    wrapper.writeline(f"{alias_name} = {output_name}")
        else:
            # Single-output template: simple aliasing
            for alias_name in epilogue_aliases:
                if alias_name != output_name:
                    wrapper.writeline(f"{alias_name} = {output_name}")


class SubgraphContext:
    """Context manager for setting the current subgraph body."""

    def __init__(self, kernel: HelionTemplateKernel, name: str) -> None:
        self.kernel = kernel
        self.name = name
        self._prev_subgraph: Optional[str] = None

    def __enter__(self) -> "SubgraphContext":
        self._prev_subgraph = self.kernel._current_subgraph
        self.kernel._current_subgraph = self.name
        if self.name not in self.kernel._subgraph_bodies:
            self.kernel._subgraph_bodies[self.name] = []
        return self

    def __exit__(self, *args: Any) -> None:
        self.kernel._current_subgraph = self._prev_subgraph


class CurrentNodeContext:
    """Context manager for setting the current node being processed."""

    def __init__(self, kernel: HelionTemplateKernel, node: Any) -> None:
        self.kernel = kernel
        self.node = node
        self._prev_node: Any = None

    def __enter__(self) -> "CurrentNodeContext":
        self._prev_node = getattr(self.kernel, "_current_node", None)
        self.kernel._current_node = self.node
        return self

    def __exit__(self, *args: Any) -> None:
        self.kernel._current_node = self._prev_node
