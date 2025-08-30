"""
RNG Utilities for PyTorch Inductor - Clean and Well-Structured

This module handles Random Number Generation (RNG) in the Inductor lowering pipeline:
1. Tracks RNG seed buffers across the computation graph
2. Resolves dependencies when operations need RNG seeds  
3. Generates proper index expressions for multi-dimensional RNG operations

The module is split into two main components:
- RNGBufferManager: Tracks and manages RNG seed buffer dependencies
- RNGIndexGenerator: Handles generation of index expressions for RNG ops

Example Usage:
    # In inductor_lowering.py
    rng_manager = RNGBufferManager()
    rng_manager.initialize_from_graph(node.graph)
    
    if rng_manager.is_rng_related_buffer(buffer):
        # Handle RNG dependencies
        read_names, mapping, extras = rng_manager.handle_buffer_dependencies(...)
"""
from __future__ import annotations

import sympy as sp
import torch
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from torch._inductor.codegen.common import triton_type
from torch._inductor.ir import (
    ComputedBuffer, ExternKernelOut, InputBuffer, Pointwise, 
    RandomSeeds, Reduction, StorageBox, TensorBox
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .ast_extension import expr_from_string




class RNGBufferManager:
    """
    Manages RNG seed buffers and their dependencies in the PyTorch Inductor pipeline.
    
    The main challenge this solves:
    - RNG operations need access to seed buffers
    - These dependencies aren't always explicit in the graph
    - We need to track and inject these dependencies during lowering
    
    Example flow:
        1. Graph contains: seed_buffer = inductor_seeds(...)
        2. Later operation: random_values = rand_like(tensor)
        3. This class ensures rand_like gets seed_buffer as input
    """
    
    def __init__(self):
        # Core state: track seed buffers by name
        self._seed_buffers: Dict[str, torch.fx.Node] = {}
        self._seed_buffer_objects: List[RandomSeeds] = []
        self._known_buffer_names: set[str] = set()
        
        # Public interface: node -> buffer name mapping
        self.node_to_buf_name_mapping: Dict[torch.fx.Node, str] = {}
        
        # Proactive tracking: map synthetic RNG symbols to dimension indices
        # This avoids complex resolution logic later
        self._rng_symbol_to_dim_index: Dict[sp.Symbol, int] = {}
    
    def initialize_from_graph(self, graph: torch.fx.Graph) -> None:
        """
        Scan the FX graph to find all RNG seed buffers.
        
        Example:
            graph nodes: [placeholder, inductor_seeds, rand_like, output]
            This finds inductor_seeds and registers it as a seed buffer.
        """
        global _current_seed_buffers
        
        for node in graph.nodes:
            if not self._has_buffer(node):
                continue
                
            buffer = node.meta["lowering"].buffer
            buffer_name = buffer.get_name()
            self._known_buffer_names.add(buffer_name)
            
            # Register RandomSeeds buffers for tracking
            if isinstance(buffer, RandomSeeds):
                self._seed_buffers[buffer_name] = node
                self._seed_buffer_objects.append(buffer)
                self.node_to_buf_name_mapping[node] = buffer_name
                # Update global list for seed count determination
                if buffer not in _current_seed_buffers:
                    _current_seed_buffers.append(buffer)
    
    def update_known_buffers(self, names: List[str]) -> None:
        """Add buffer names to our tracking set."""
        self._known_buffer_names.update(names)
    
    def is_rng_related_buffer(self, buffer) -> bool:
        """
        Check if a buffer needs RNG seed handling.
        
        Returns True if:
        - Buffer reads from seed buffers
        - Buffer name contains RNG markers
        - Buffer is Pointwise operation (may need seeds)
        """
        if not isinstance(buffer, ComputedBuffer):
            return False
        
        # Check explicit RNG dependencies
        for name in buffer.get_read_names():
            if (name in self._seed_buffers or 
                'inductor_lookup_seed' in name or 
                'inductor_random' in name):
                return True
        
        # Pointwise ops might need seeds even without explicit deps
        return isinstance(buffer.data, Pointwise) and bool(self._seed_buffers)
    
    def handle_buffer_dependencies(self, buffer, node, current_input_nodes, 
                                 current_input_names, node_to_buf_name_mapping):
        """
        Main entry point: resolve all dependencies for a buffer.
        
        This method:
        1. Gets buffer's read dependencies
        2. Resolves synthetic names to actual buffer names
        3. Finds missing seed buffer dependencies
        4. Updates the node's inputs to include seeds
        
        Returns:
            (resolved_read_names, input_mapping, extra_seed_names)
        """
        # Update our knowledge base
        self.update_known_buffers(list(node_to_buf_name_mapping.values()))
        
        # Get and resolve dependencies
        read_names = buffer.get_read_names()
        read_names = self._resolve_synthetic_names(read_names, current_input_names)
        
        # Build initial mapping
        input_mapping = dict(zip(current_input_nodes, current_input_names, strict=True))
        
        # Find and add missing seed dependencies
        extra_nodes, extra_names = self._find_missing_seeds(
            buffer, read_names, input_mapping, current_input_nodes
        )
        
        if extra_nodes:
            # Update node's inputs to include seeds
            node._input_nodes = list(current_input_nodes) + extra_nodes
            for extra_node, name in zip(extra_nodes, extra_names, strict=True):
                input_mapping[extra_node] = name
        
        return read_names, input_mapping, extra_names
    
    def process_new_buffers(self, new_buffers: List[Any], node: torch.fx.Node, 
                          buffer_name_to_output_index: Dict[str, int]) -> List[Any]:
        """
        Process newly created buffers, handling special RNG types.
        
        Returns list of buffers that need further processing.
        """
        processed = []
        
        for i, buf in enumerate(new_buffers):
            name = buf.get_name()
            
            # Handle terminal buffers (last RandomSeeds/ExternKernelOut)
            if self._is_terminal_buffer(buf, i, len(new_buffers)):
                node.meta["lowering"] = self._create_name_lowering(buf)
            # Track seed buffers
            elif isinstance(buf, RandomSeeds):
                buffer_name_to_output_index[name] = i
                self._register_seed_from_graph(node.graph, buf)
            # Track external kernels
            elif isinstance(buf, ExternKernelOut):
                buffer_name_to_output_index[name] = i
            # Regular computed buffers need processing
            elif isinstance(buf, ComputedBuffer) and isinstance(buf.data, (Pointwise, Reduction)):
                processed.append(buf)
                buffer_name_to_output_index[name] = i
        
        return processed
    
    # Private helper methods
    
    def _has_buffer(self, node: torch.fx.Node) -> bool:
        """Check if node has a buffer in its lowering metadata."""
        return (node.op == "call_function" and 
                hasattr(node, "meta") and 
                "lowering" in node.meta and 
                hasattr(node.meta["lowering"], "buffer"))
    
    def _resolve_synthetic_names(self, names: OrderedSet[str], 
                               current_names: List[str]) -> OrderedSet[str]:
        """
        Map Inductor's synthetic names to actual buffer names.
        
        Example:
            synthetic: ['inductor_lookup_seed_default_input0']
            current: ['buf0']
            result: ['buf0']
        """
        if not current_names or not names:
            return names
        
        resolved = OrderedSet()
        unknown_idx = 0
        
        for name in names:
            if name in self._known_buffer_names:
                resolved.add(name)
            elif unknown_idx < len(current_names):
                resolved.add(current_names[unknown_idx])
                unknown_idx += 1
            else:
                resolved.add(name)
        
        return resolved
    
    def _find_missing_seeds(self, buffer, read_names, input_mapping, current_inputs):
        """Find seed buffers needed but not in current inputs."""
        extra_nodes = []
        extra_names = []
        
        if isinstance(buffer, ComputedBuffer) and isinstance(buffer.data, Pointwise):
            for name in read_names:
                if (name in self._seed_buffers and 
                    name not in input_mapping.values()):
                    node = self._seed_buffers[name]
                    if node not in current_inputs:
                        extra_nodes.append(node)
                        extra_names.append(name)
        
        return extra_nodes, extra_names
    
    def _is_terminal_buffer(self, buffer, index, total):
        """Check if buffer is terminal (last RandomSeeds/ExternKernelOut)."""
        return (isinstance(buffer, (RandomSeeds, ExternKernelOut)) and 
                index == total - 1)
    
    def _create_name_lowering(self, buffer):
        """Create lowering that returns buffer name."""
        class NameLowering:
            def __init__(self, b): self.buffer = b
            def codegen(self, ctx, node): return self.buffer.get_name()
        return NameLowering(buffer)
    
    def _register_seed_from_graph(self, graph, buffer):
        """Find and register the node containing this seed buffer."""
        global _current_seed_buffers
        
        name = buffer.get_name()
        for n in graph.nodes:
            if (self._has_buffer(n) and 
                n.meta["lowering"].buffer is buffer):
                self._seed_buffers[name] = n
                self._seed_buffer_objects.append(buffer)
                self._known_buffer_names.add(name)
                # Update global list for seed count determination
                if isinstance(buffer, RandomSeeds) and buffer not in _current_seed_buffers:
                    _current_seed_buffers.append(buffer)
                break


class RNGIndexGenerator:
    """
    Handles generation of index expressions for N-dimensional RNG operations.
    
    This class is responsible for:
    1. Converting symbolic expressions to Triton code
    2. Handling N-dimensional broadcasting for RNG operations
    3. Managing dimension symbols and their mappings
    
    Example:
        For a 2D tensor with indices i0, i1:
        Input expression: i0 * n + i1
        Output code: indices_0[:, None] * n + indices_1[None, :]
    """
    
    def __init__(self):
        # Track registered symbols and their dimension indices
        self._symbol_dimensions: Dict[sp.Symbol, int] = {}
        # Track the maximum seed offset needed
        self._max_seed_offset: int = 0
        # Proactive tracking: map synthetic RNG symbols to dimension indices
        self._rng_symbol_to_dim_index: Dict[sp.Symbol, int] = {}
    
    def generate_seed_load(self, buffer_name: str, offset: int, 
                          name_lookup: dict, codegen) -> str:
        """
        Generate code to load from a seed buffer at given offset.
        
        Example:
            Input: buffer_name="seed_buf", offset=0
            Output: "tl.load(seed_buf + 0)"
        """
        # Track the maximum offset needed
        self._max_seed_offset = max(self._max_seed_offset, offset)
        
        mapped_name = self._resolve_buffer_name(buffer_name, name_lookup, codegen)
        return codegen.lift(expr_from_string(f"tl.load({mapped_name} + {offset})")).id
    
    def generate_index_expression(self, expr: sp.Expr, dtype, codegen) -> str:
        """
        Generate Triton code for an RNG index expression.
        
        This handles multi-dimensional indexing for RNG operations.
        For example, a 2D index expression i0 * n + i1 becomes proper
        broadcasting code.
        """
        # Register all symbols in the expression
        self._register_expression_symbols(expr, codegen)
        
        # Extract index symbols (i0, i1, i2, etc)
        index_symbols = self._extract_index_symbols(expr)
        
        # Generate appropriate code based on dimensionality
        if len(index_symbols) >= 2:
            # Build the broadcasting expression directly
            expr_str = self._generate_nd_broadcast(index_symbols, expr, codegen)
        else:
            expr_str = codegen.device_function.user_sympy_expr(expr)
        
        # Convert to appropriate type
        name = codegen.lift(expr_from_string(expr_str)).id
        if name in codegen.device_function._constexpr_args:
            return name
        return f"{name}.to({triton_type(dtype)})"
    
    def process_dynamic_size(self, size):
        """
        Convert TensorBox sizes to symbolic variables for RNG operations.
        
        This handles dynamic shapes in RNG operations by creating symbols.
        Example:
            Input: [TensorBox(shape=[32]), TensorBox(scalar)]
            Output: [32, Symbol('tile_size_1')]
        """
        if not isinstance(size, (list, tuple)) or not size:
            return size
        
        processed = []
        for i, s in enumerate(size):
            if isinstance(s, TensorBox):
                # Convert TensorBox to symbol
                if shape := s.get_size():
                    processed.append(shape[0])
                else:
                    # Scalar tensor - create symbol
                    sym = self._create_size_symbol(s, i)
                    processed.append(sym)
                    # Proactively track this symbol's dimension index
                    self._rng_symbol_to_dim_index[sym] = i
            else:
                processed.append(s)
        
        return processed
    
    def register_rng_dimension_symbol(self, sym: sp.Symbol, dim_index: int):
        """Proactively register an RNG dimension symbol with its index."""
        self._rng_symbol_to_dim_index[sym] = dim_index
    
    def get_rng_symbol_dimension(self, sym: sp.Symbol) -> Optional[int]:
        """Get the dimension index for an RNG symbol if registered."""
        return self._rng_symbol_to_dim_index.get(sym)
    
    def _resolve_buffer_name(self, name: str, lookup: dict, codegen) -> str:
        """Resolve buffer name, creating seed buffer arg if needed."""
        if name in lookup:
            m = lookup[name]
            return codegen.lift(m).id if hasattr(m, 'id') else str(m)
        
        # Create seed buffer argument if not present
        from .device_function import TensorArg
        df = codegen.device_function
        if not any(a.name == "seed_buffer_arg" for a in df.arguments):
            # Get the seed count from the global RNG buffer manager if available
            # Otherwise use the maximum offset + 1
            num_seeds = _get_required_seed_count()
            
            df.arguments.append(TensorArg(
                "seed_buffer_arg",
                torch.empty(num_seeds, dtype=torch.int64, device='cuda'),
                f"torch.ops.prims.inductor_seeds({num_seeds}, device='cuda')"
            ))
        return "seed_buffer_arg"
    
    def _register_expression_symbols(self, expr: sp.Expr, codegen):
        """Register all symbols in an expression with the device function."""
        from .device_function import VarInfo
        from .host_function import HostFunction
        device_function = codegen.device_function
        
        for sym in expr.free_symbols:
            if isinstance(sym, sp.Symbol) and sym not in device_function.expr_to_var_info:
                if self._is_index_symbol(sym):
                    # Index symbol like i0, i1, i2
                    idx = int(sym.name[1:])
                    var_name = self._get_loop_var_name(idx, codegen)
                    device_function.expr_to_var_info[sym] = VarInfo(var_name, idx)
                else:
                    # Dimension symbol - use our simplified approach
                    var_name = self._resolve_dimension_symbol(sym, device_function)
                    device_function.expr_to_var_info[sym] = VarInfo(var_name, sym)
    
    def _resolve_dimension_symbol(self, sym: sp.Symbol, device_function) -> str:
        """Resolve a dimension symbol to its variable name using proactive tracking."""
        from .host_function import HostFunction
        from .device_function import SymbolArgument
        
        # First check if it has an origin (normal path)
        expr_to_origin = HostFunction.current().expr_to_origin
        if sym in expr_to_origin:
            return device_function._lift_sympy_arg(sym)
        
        # Check if we proactively tracked this RNG symbol
        if sym in self._rng_symbol_to_dim_index:
            dim_idx = self._rng_symbol_to_dim_index[sym]
            # Get the corresponding symbol argument
            symbol_args = [arg for arg in device_function.arguments 
                         if isinstance(arg, SymbolArgument)]
            assert dim_idx < len(symbol_args), \
                f"RNG symbol {sym} has dim_idx {dim_idx} but only {len(symbol_args)} symbol args"
            return symbol_args[dim_idx].name
        
        # If we get here, it means we failed to proactively track this symbol
        raise AssertionError(
            f"Symbol {sym} was not proactively tracked. "
            f"This should have been registered when the RNG operation was created. "
            f"Available tracked symbols: {list(self._rng_symbol_to_dim_index.keys())}"
        )
    
    def _is_index_symbol(self, sym: sp.Symbol) -> bool:
        """Check if symbol is an index symbol (i0, i1, etc)."""
        return sym.name.startswith('i') and sym.name[1:].isdigit()
    
    def _get_loop_var_name(self, idx: int, codegen) -> str:
        """Get the variable name for a loop index."""
        loops = getattr(codegen, 'active_device_loops', {}).get(idx, [])
        if loops:
            return loops[-1].strategy.index_var(idx)
        return f"indices_{idx}"
    
    def _get_dimension_name(self, idx: Optional[int]) -> str:
        """Get readable name for a dimension."""
        return f"dim{idx}" if idx is not None else "unknown_dim"
    
    def _extract_index_symbols(self, expr: sp.Expr) -> List[sp.Symbol]:
        """Extract and sort index symbols from expression."""
        return sorted(
            [s for s in expr.free_symbols if self._is_index_symbol(s)],
            key=lambda s: int(s.name[1:])
        )
    
    def _generate_nd_broadcast(self, indices: List[sp.Symbol], expr: sp.Expr, codegen) -> str:
        """
        Generate N-dimensional broadcasting expression.
        
        Examples:
            1D: indices_0
            2D: indices_0[:, None] * width + indices_1[None, :]
            3D: indices_0[:, None, None] * height * width + 
                indices_1[None, :, None] * width + 
                indices_2[None, None, :]
        """
        ndim = len(indices)
        if ndim == 0:
            return ""
        
        # Get actual variable names for indices
        idx_vars = []
        for sym in indices:
            # Get the registered variable name
            device_function = codegen.device_function
            if sym in device_function.expr_to_var_info:
                idx_vars.append(device_function.expr_to_var_info[sym].name)
            else:
                # Fallback
                idx_vars.append(self._get_loop_var_name(int(sym.name[1:]), codegen))
        
        if ndim == 1:
            return idx_vars[0]
        
        # Extract stride information from the expression
        stride_info = self._extract_strides_from_expr(expr, indices)
        
        # Build the broadcasting expression
        parts = []
        for i, (idx_sym, idx_var) in enumerate(zip(indices, idx_vars)):
            # Create broadcasting slice for this index
            bc_slice = self._make_broadcast_slice(i, ndim)
            broadcasted_var = f"{idx_var}[{bc_slice}]"
            
            if i == ndim - 1:
                # Last dimension has no stride
                parts.append(broadcasted_var)
            else:
                # Get the stride for this dimension
                stride = stride_info.get(idx_sym)
                if stride:
                    # Get the registered name for each symbol in the stride
                    stride_str = self._convert_stride_symbols(stride, codegen.device_function)
                    parts.append(f"{broadcasted_var} * {stride_str}")
                else:
                    parts.append(broadcasted_var)
        
        return " + ".join(parts)
    
    def _convert_stride_symbols(self, stride: sp.Expr, device_function) -> str:
        """Convert stride symbols to their registered variable names."""
        if isinstance(stride, sp.Symbol):
            # Single symbol - get its registered name
            if stride in device_function.expr_to_var_info:
                return device_function.expr_to_var_info[stride].name
            else:
                return str(stride)
        else:
            # Complex expression - convert each symbol
            result = str(stride)
            for sym in stride.free_symbols:
                if sym in device_function.expr_to_var_info:
                    var_name = device_function.expr_to_var_info[sym].name
                    result = result.replace(str(sym), var_name)
            return result
    
    def _convert_stride_to_code(self, stride: sp.Expr, codegen) -> str:
        """Convert a stride expression to code, handling all symbols properly."""
        # Register all symbols in the stride expression
        self._register_expression_symbols(stride, codegen)
        
        # Convert to code
        return codegen.device_function.user_sympy_expr(stride)
    
    def _extract_strides_from_expr(self, expr: sp.Expr, indices: List[sp.Symbol]) -> Dict[sp.Symbol, sp.Expr]:
        """
        Extract stride multipliers for each index symbol from the expression.
        
        For expression i0 * n + i1, returns {i0: n}
        For expression i0 * m * n + i1 * n + i2, returns {i0: m*n, i1: n}
        """
        stride_info = {}
        
        # Convert expression to expanded form
        expr_expanded = sp.expand(expr)
        
        # For each index symbol, find what it's multiplied by
        for idx in indices[:-1]:  # Skip last index as it has no stride
            # Find the coefficient of this index symbol
            coeff = expr_expanded.coeff(idx)
            if coeff and coeff != 1:
                stride_info[idx] = coeff
        
        return stride_info
    
    def _make_broadcast_slice(self, current_dim: int, total_dims: int) -> str:
        """Create broadcasting slice (e.g., ':, None, None')."""
        slices = ["None"] * total_dims
        slices[current_dim] = ":"
        return ", ".join(slices)
    
    def _create_size_symbol(self, tensor_box, index):
        """Create symbol for dynamic size."""
        data = tensor_box.data
        if (isinstance(data, StorageBox) and hasattr(data, 'data') and
            isinstance(data.data, InputBuffer) and not data.data.get_size()):
            var_name = f"{data.data.get_name()}_val"
        else:
            var_name = f"tile_size_{index}"
        
        sym = sp.Symbol(var_name, integer=True, positive=True)
        self._symbol_dimensions[sym] = index
        return sym
    
    def register_rng_dimension_symbol(self, sym: sp.Symbol, dim_index: int):
        """Proactively register an RNG dimension symbol with its index."""
        self._rng_symbol_to_dim_index[sym] = dim_index
    
    def get_rng_symbol_dimension(self, sym: sp.Symbol) -> Optional[int]:
        """Get the dimension index for an RNG symbol if registered."""
        return self._rng_symbol_to_dim_index.get(sym)


# Global singleton instances for convenience
_rng_index_generator = RNGIndexGenerator()

# Legacy global tracking for backward compatibility with inductor_lowering_extra.py
_symbol_to_index = _rng_index_generator._symbol_dimensions

# Global storage for RandomSeeds buffers from current compilation
_current_seed_buffers: List[RandomSeeds] = []


def register_rng_dimension_symbol(sym: sp.Symbol, dim_index: int):
    """
    Proactively register an RNG dimension symbol with its dimension index.
    
    This allows us to avoid complex resolution logic later by tracking
    the symbol -> dimension mapping when the symbol is created.
    
    Example:
        sym = Symbol("inductor_random_default_input0_val")
        register_rng_dimension_symbol(sym, 0)  # This is dimension 0 (m)
    """
    _rng_index_generator.register_rng_dimension_symbol(sym, dim_index)


# Module-level convenience functions

def _get_required_seed_count() -> int:
    """Get the required seed count from registered RandomSeeds buffers."""
    global _current_seed_buffers
    
    # If we have RandomSeeds buffers, use their size
    if _current_seed_buffers:
        # All seed buffers should have the same size
        return _current_seed_buffers[0].get_size()[0]
    
    # Fallback: use the maximum offset + 1, minimum 10
    return max(_rng_index_generator._max_seed_offset + 1, 10)


def handle_load_seed(name: str, offset: int, input_name_lookup: dict, cg) -> str:
    """
    Generate code to load from a seed buffer.
    
    Example output: "tl.load(seed_buffer_arg + 0)"
    """
    return _rng_index_generator.generate_seed_load(name, offset, input_name_lookup, cg)


def handle_index_expr(expr: sp.Expr, dtype: torch.dtype, cg) -> str:
    """
    Generate code for N-dimensional RNG index expressions.
    
    Handles broadcasting for multi-dimensional tensors:
    - 1D: indices_0
    - 2D: indices_0[:, None] * n + indices_1[None, :]
    - 3D: Complex broadcasting with proper strides
    """
    return _rng_index_generator.generate_index_expression(expr, dtype, cg)


def process_dynamic_size(size):
    """
    Convert TensorBox sizes to symbolic variables for RNG operations.
    
    This handles dynamic shapes in RNG operations by creating symbols.
    """
    return _rng_index_generator.process_dynamic_size(size)


@contextmanager
def inductor_context(fake_mode):
    """Context manager for inductor operations."""
    class Handler:
        def __init__(self, fm): self.fake_mode = fm
        def __getattr__(self, n): return None
    
    with V.set_graph_handler(Handler(fake_mode)), V.set_fake_mode(fake_mode):
        yield


def extract_index_symbols(expr: sp.Expr) -> List[sp.Symbol]:
    """Extract index symbols (i0, i1, etc) from expression."""
    return sorted(
        [s for s in expr.free_symbols if isinstance(s, sp.Symbol) and 
         s.name.startswith('i') and s.name[1:].isdigit()],
        key=lambda s: int(s.name[1:])
    )


def resolve_synthetic_names(read_names: OrderedSet[str], current_input_names: List[str],
                          known_buffer_names: set[str]) -> OrderedSet[str]:
    """Legacy function for resolving synthetic names."""
    manager = RNGBufferManager()
    manager._known_buffer_names = known_buffer_names
    return manager._resolve_synthetic_names(read_names, current_input_names)


def handle_special_buffer(buffer: Any, i: int, total_buffers: int):
    """Legacy function for special buffer handling."""
    if isinstance(buffer, (RandomSeeds, ExternKernelOut)) and i == total_buffers - 1:
        class L:
            def __init__(self, b): self.buffer = b
            def codegen(self, ctx, node): return self.buffer.get_name()
        return L(buffer)
    return None