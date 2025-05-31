from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from torch._inductor.codegen.simd import constant_repr
from torch._inductor.codegen.triton import triton_acc_type

from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .tile_strategy import DeviceLoopState
from .tile_strategy import TileStrategy

if TYPE_CHECKING:
    import torch

    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState


class PersistentScanState:
    """State for persistent scan operations."""
    def __init__(self, strategy: "PersistentScanStrategy") -> None:
        self.strategy = strategy
        
    @property
    def block_indices(self) -> list[int]:
        return self.strategy.block_indices


class ScanStrategy(TileStrategy):
    """
    Base class for inclusive prefix-scan (cumsum / cumprod / generic associative scan).
    
    Subclasses implement specific strategies for different scenarios:
    - PersistentScanStrategy: when the entire scan axis fits in a block
    - LoopedScanStrategy: when we need to iterate tile-by-tile with carry
    """

    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        mask_var: str | None,
        block_size_var: str | None,
    ) -> None:
        super().__init__(fn=fn, block_indices=[block_index])
        self._mask_var = mask_var
        if block_size_var is not None:
            fn.block_size_var_cache[(block_index,)] = block_size_var

    @property
    def block_index(self) -> int:
        return self.block_indices[0]

    def mask_var(self, block_idx: int) -> str | None:
        assert block_idx == self.block_index
        return self._mask_var

    def _broadcast(self, base: str, fake_inp: torch.Tensor, dim: int) -> str:
        expand = self.fn.tile_strategy.expand_str([*fake_inp.size()], dim)
        shape = self.fn.tile_strategy.shape_str([*fake_inp.size()])
        return f"tl.broadcast_to({base}{expand}, {shape})"

    def _maybe_mask(
        self,
        state: CodegenState,
        fake_inp: torch.Tensor,
        dim: int,
        expr: str,
        default: float | bool,
    ) -> str:
        # Check if there's already a mask for this dimension
        mask_var = state.codegen.mask_var(self.block_index)
        if mask_var is None:
            return expr
        mask_expr = self._broadcast(mask_var, fake_inp, dim)
        return state.codegen.lift(
            expr_from_string(f"tl.where({mask_expr}, {expr}, {constant_repr(default)})")
        ).id

    def codegen_scan(
        self,
        state: CodegenState,
        input_name: str,
        scan_type: str,
        dim: int,
        fake_input: torch.Tensor,
    ) -> ast.AST:
        """
        Emits Triton AST for an inclusive prefix-scan along *dim*.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class PersistentScanStrategy(ScanStrategy):
    """
    Strategy for when the entire scan axis fits within a block.
    Executes a single Triton `tl.cumsum` / `tl.cumprod` / `tl.associative_scan`.
    """

    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
    ) -> None:
        env = CompileEnvironment.current()
        numel = env.block_sizes[block_index].numel
        
        # For persistent scan, we might have symbolic sizes
        # We need a mask if the size is not known at compile time
        if isinstance(numel, int):
            mask_var = None
            block_size = numel
        else:
            # Symbolic size - we need a mask
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)
            # Use a reasonable default block size
            block_size = 1024
        
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_RDIM_SIZE_{block_index}"),
        )
        self.offset_vars[block_index] = "0"
        self.index_vars[block_index] = fn.new_var(f"sindex_{block_index}", dce=True)
        self.block_size = block_size

    def offset_var(self, block_idx: int) -> str:
        assert block_idx == self.block_index
        return "0"

    def codegen_preamble(self, state: CodegenState) -> None:
        env = CompileEnvironment.current()
        block_idx = self.block_index
        numel = env.block_sizes[block_idx].numel
        
        # Check if there's already an active loop/strategy for this dimension
        # If so, reuse its variables
        if block_idx in state.codegen.active_device_loops and state.codegen.active_device_loops[block_idx]:
            existing_loop = state.codegen.active_device_loops[block_idx][-1]
            if hasattr(existing_loop, 'strategy'):
                # Reuse the index variable from the existing strategy
                self.index_vars[block_idx] = existing_loop.strategy.index_var(block_idx)
                # Don't create new variables or set active loops
                return
        
        index_var = self.index_var(block_idx)
        block_size_var = self.fn.block_size_var_cache.get((block_idx,))
        
        if block_size_var is None:
            # No existing block size var, create one
            block_size_var = self.fn.new_var(f"_SCAN_SIZE_{block_idx}")
            self.fn.block_size_var_cache[(block_idx,)] = block_size_var
        
        if state.device_function.constexpr_arg(block_size_var):
            if isinstance(numel, int):
                state.codegen.host_statements.append(
                    statement_from_string(f"{block_size_var} = {numel!r}")
                )
            else:
                # For symbolic sizes, use a reasonable default
                state.codegen.host_statements.append(
                    statement_from_string(f"{block_size_var} = {self.block_size!r}")
                )
        
        state.add_statement(
            f"{index_var} = tl.arange(0, {block_size_var}).to({env.triton_index_type()})"
        )
        
        # Add mask if needed
        if self._mask_var is not None:
            state.add_statement(
                f"{self._mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
            )
        
        state.codegen.set_active_loops(PersistentScanState(self))

    def codegen_scan(
        self,
        state: CodegenState,
        input_name: str,
        scan_type: str,
        dim: int,
        fake_input: torch.Tensor,
    ) -> ast.AST:
        """
        Emits a single Triton scan operation for the entire axis.
        """
        # For persistent scan, the input is already properly masked by the reduction strategy
        # Just apply the scan operation directly
        if scan_type == "sum":
            call = f"tl.cumsum({input_name}, {dim})"
        elif scan_type == "prod":
            call = f"tl.cumprod({input_name}, {dim})"
        else:  # generic
            call = (
                f"tl.associative_scan({input_name}, {dim}, "
                f'combine_fn=triton_helpers.get_scan_combine_fn("{scan_type}"))'
            )
        return expr_from_string(call)
    
    def codegen_grid(self, state: CodegenState) -> None:
        """Scan strategies don't manage the grid, they work within existing loops."""
        # This shouldn't be called - scan operations happen within tile loops
        raise NotImplementedError("Scan strategies should not be used as grid strategies")


class LoopedScanStrategy(ScanStrategy):
    """
    Strategy for when the scan axis doesn't fit in a single block.
    Iterates tile-by-tile and keeps a carry that propagates the running prefix.
    Works even when the underlying stride is > 1 (non-contiguous axis).
    """

    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        block_size: int,
    ) -> None:
        env = CompileEnvironment.current()
        numel = env.block_sizes[block_index].numel
        
        # Need a mask when numel is not a multiple of block_size
        if env.known_multiple(numel, block_size):
            mask_var = None
        else:
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)
        
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_SCAN_BLKSIZE_{block_index}"),
        )
        self.offset_vars[block_index] = fn.new_var(f"soffset_{block_index}", dce=True)
        self.index_vars[block_index] = fn.new_var(f"sindex_{block_index}", dce=True)
        self.block_size = block_size
        assert block_size > 1
        
    def offset_var(self, block_idx: int) -> str:
        assert block_idx == self.block_index
        return self.offset_vars[block_idx]


    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        """Creates the device loop for iterating over tiles."""
        env = CompileEnvironment.current()
        block_index = self.block_index
        device_function = state.device_function
        numel = env.block_sizes[block_index].numel
        offset_var = self.offset_var(block_index)
        index_var = self.index_var(block_index)
        block_size_var = self.block_size_var(block_index)
        assert block_size_var is not None
        if state.device_function.constexpr_arg(block_size_var):
            state.codegen.host_statements.append(
                statement_from_string(f"{block_size_var} = {self.block_size!r}")
            )
        body: list[ast.AST] = [
            statement_from_string(
                f"{index_var} = {offset_var} + tl.arange(0, ({block_size_var})).to({env.triton_index_type()})"
            ),
        ]
        if (mask_var := self._mask_var) is not None:
            body.append(
                statement_from_string(
                    f"{mask_var} = {index_var} < {device_function.sympy_expr(numel)}"
                )
            )
        for_node = create(
            ast.For,
            target=create(ast.Name, id=offset_var, ctx=ast.Store()),
            iter=expr_from_string(
                f"range(0, ({device_function.sympy_expr(numel)}), {block_size_var})"
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
        )

    def codegen_scan(
        self,
        state: CodegenState,
        input_name: str,
        scan_type: str,
        dim: int,
        fake_input: torch.Tensor,
    ) -> ast.AST:
        """
        Emits Triton AST for a tile-by-tile scan with carry propagation.
        """
        # Get the device loop for this block index
        if self.block_index not in state.codegen.active_device_loops:
            raise RuntimeError(f"No active device loop for block index {self.block_index}")
        device_loops = state.codegen.active_device_loops[self.block_index]
        if not device_loops:
            raise RuntimeError(f"Empty device loop list for block index {self.block_index}")
        device_loop = device_loops[-1]
        assert isinstance(device_loop, DeviceLoopState)
        shape = self.fn.tile_strategy.shape_str([*fake_input.size()])
        zero_or_one = 0 if scan_type == "sum" else 1
        assert state.fx_node is not None
        carry = self.fn.new_var(f"{state.fx_node.name}_carry", dce=True)
        device_loop.outer_prefix.append(
            statement_from_string(
                f"{carry} = tl.full({shape}, {constant_repr(zero_or_one)}, "
                f"{triton_acc_type(fake_input.dtype)})"
            )
        )

        result = self.fn.new_var(state.fx_node.name, dce=True)
        masked = self._maybe_mask(state, fake_input, dim, input_name, zero_or_one)
        
        if scan_type == "sum":
            state.add_statement(f"{result} = {carry} + tl.cumsum({masked}, {dim})")
            # pick last element along dim -> update carry
            idxs = [":" for _ in range(fake_input.dim())]
            idxs[dim] = "-1"
            last = f"{result}[{', '.join(idxs)}]"
            state.add_statement(f"{carry} = {last}")
        elif scan_type == "prod":
            state.add_statement(f"{result} = {carry} * tl.cumprod({masked}, {dim})")
            # pick last element along dim -> update carry
            idxs = [":" for _ in range(fake_input.dim())]
            idxs[dim] = "-1"
            last = f"{result}[{', '.join(idxs)}]"
            state.add_statement(f"{carry} = {last}")
        else:
            tile_scan = (
                f"tl.associative_scan({masked}, {dim}, "
                f'combine_fn=triton_helpers.get_scan_combine_fn("{scan_type}"))'
            )
            op = f"triton_helpers.{scan_type}_combine"
            state.add_statement(f"{result} = {op}({carry}, {tile_scan})")
            # pick last element along dim -> update carry
            idxs = [":" for _ in range(fake_input.dim())]
            idxs[dim] = "-1"
            last = f"{result}[{', '.join(idxs)}]"
            state.add_statement(f"{carry} = {last}")

        device_loop.outer_suffix.append(statement_from_string(f"{result} = {result}"))
        return expr_from_string(result)
    
    def codegen_grid(self, state: CodegenState) -> None:
        """Scan strategies don't manage the grid, they work within existing loops."""
        # This shouldn't be called - scan operations happen within tile loops
        raise NotImplementedError("Scan strategies should not be used as grid strategies")


def create_scan_strategy(
    fn: DeviceFunction,
    block_index: int,
    scan_loop: int | None = None,
) -> ScanStrategy:
    """
    Creates the appropriate scan strategy based on the configuration.
    
    Parameters:
    -----------
    fn : DeviceFunction
        The device function context
    block_index : int
        The index of the axis we are scanning
    scan_loop : int | None
        The scan loop size from config; if None, auto-detect strategy
    
    Returns:
    --------
    ScanStrategy
        Either PersistentScanStrategy or LoopedScanStrategy
    """
    env = CompileEnvironment.current()
    
    # Check if there's already an active device loop for this block index
    # If so, we should use persistent scan to avoid nested loops
    codegen = getattr(fn, '_current_codegen', None)
    if codegen and block_index in codegen.active_device_loops and codegen.active_device_loops[block_index]:
        # There's already a loop for this dimension, use persistent scan
        return PersistentScanStrategy(fn, block_index)
    
    # If scan_loop is not provided, try to get it from config
    if scan_loop is None:
        # Check if we have scan_loops in the current config
        if hasattr(env, 'config') and hasattr(env.config, 'scan_loops') and env.config.scan_loops:
            # For now, use the first scan_loop value (similar to reduction)
            scan_loop = env.config.scan_loops[0] if env.config.scan_loops else None
    
    # If still no scan_loop, use heuristic
    if scan_loop is None:
        numel = env.block_sizes[block_index].numel
        # For scan operations within tiles, always use persistent strategy
        # This avoids conflicts with existing tile loops
        return PersistentScanStrategy(fn, block_index)
    else:
        # scan_loop provided (from config or explicit), but check if we can use it
        # If there's already a device loop, force persistent
        return PersistentScanStrategy(fn, block_index)