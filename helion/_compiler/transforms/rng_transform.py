from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
import torch.fx

from ..ast_extension import create
from ..ast_extension import create_arg
from ..ast_extension import expr_from_string
from ..generate_ast import statement_from_string
from .base import TransformPass

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..host_function import HostFunction
    from ..inductor_lowering import GraphInterpreter


class RNGTransformPass(TransformPass):
    def __init__(self, device_function: DeviceFunction) -> None:
        self.rng_seed_buffer_param_name = device_function.new_var("rng_seed_buffer")
        self.seed_index_map: dict[torch.fx.Node, int] = {}

    @staticmethod
    def has_rng_ops(host_function: HostFunction) -> bool:
        """Check if any graph in the device IR contains RNG operations."""
        rng_ops = {
            torch.ops.aten.rand.default,
            torch.ops.aten.randn.default,
        }

        for graph_info in host_function.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.op == "call_function" and node.target in rng_ops:
                    return True

        return False

    def get_or_allocate_seed_index(self, node: torch.fx.Node) -> int:
        """Get the seed index for a given RNG operation node, allocating if necessary."""
        if node not in self.seed_index_map:
            seed_index = len(self.seed_index_map)
            self.seed_index_map[node] = seed_index

        return self.seed_index_map[node]

    def add_kernel_arguments(self, args: list[ast.arg]) -> None:
        """Add RNG seed buffer argument to kernel signature."""
        args.append(create_arg(self.rng_seed_buffer_param_name))

    def add_host_arguments(self, args: list[str]) -> None:
        """Add RNG seed buffer argument to host function call."""
        args.append("_rng_seed_buffer")

    def get_host_preamble_statements(self) -> list[ast.AST]:
        """Get statements to inject into host function preamble."""
        import_stmt = statement_from_string(
            "from torch._inductor import inductor_prims"
        )

        # Create host-side seed buffer with the required number of seeds
        seed_buffer_stmt = statement_from_string(
            f"_rng_seed_buffer = inductor_prims.seeds({len(self.seed_index_map)}, torch.device('cuda'))"
        )

        return [import_stmt, seed_buffer_stmt]

    def codegen_rng_op(
        self, ctx: GraphInterpreter, node: torch.fx.Node, rng_function: str
    ) -> ast.AST:
        """Generate Triton code for an RNG op.

        Args:
            ctx: The graph interpreter context
            node: The FX node for this operation
            rng_function: Either "rand" or "randn"

        Returns:
            AST expression for the RNG operation
        """
        assert rng_function in ["rand", "randn"]

        # Get the seed index for this operation
        seed_index = self.get_or_allocate_seed_index(node)

        # Get dimensionality and dtype
        assert hasattr(node, "meta") and "val" in node.meta
        ndim = node.meta["val"].ndim
        dtype = node.kwargs.get("dtype", None)

        # Get the dimension variable names from the device function's symbol arguments
        device_fn = ctx.cg.device_function
        symbol_args = [
            arg
            for arg in device_fn.arguments
            if hasattr(arg, "__class__") and arg.__class__.__name__ == "SymbolArgument"
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
        seed_expr = expr_from_string(
            "tl.load({buffer} + {index})",
            buffer=expr_from_string(self.rng_seed_buffer_param_name),
            index=create(ast.Constant, value=seed_index),
        )

        # Generate the RNG call
        # Note: tl.rand() and tl.randn() always return float32
        rng_expr = expr_from_string(
            f"tl.{rng_function}({{seed}}, {{offset}})",
            seed=seed_expr,
            offset=offset_expr,
        )

        # Cast to target dtype only if explicitly specified
        if dtype is not None:
            assert isinstance(dtype, torch.dtype)
            from torch._inductor.utils import triton_type

            rng_expr = expr_from_string(
                f"{{val}}.to({triton_type(dtype)})", val=rng_expr
            )

        return rng_expr
