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
from .tile_strategy import PersistentReductionState
from .tile_strategy import TileStrategy

if TYPE_CHECKING:
    import torch

    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState


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
        if self._mask_var is None:
            return expr
        mask_expr = self._broadcast(self._mask_var, fake_inp, dim)
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
        
        # For persistent scan, we always know the size at compile time
        # and it must fit in a block
        assert isinstance(numel, int), f"Expected int numel for persistent scan, got {type(numel)}"
        
        # No mask needed for persistent scan since we process the entire axis
        mask_var = None
        
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_SCAN_SIZE_{block_index}"),
        )
        self.offset_vars[block_index] = "0"
        self.index_vars[block_index] = fn.new_var(f"sindex_{block_index}", dce=True)
        self.block_size = numel

    def offset_var(self, block_idx: int) -> str:
        assert block_idx == self.block_index
        return "0"

    def codegen_preamble(self, state: CodegenState) -> None:
        env = CompileEnvironment.current()
        block_idx = self.block_index
        numel = env.block_sizes[block_idx].numel
        index_var = self.index_var(block_idx)
        block_size_var = self.fn.block_size_var_cache[(block_idx,)]
        
        if state.device_function.constexpr_arg(block_size_var):
            state.codegen.host_statements.append(
                statement_from_string(f"{block_size_var} = {numel!r}")
            )
        
        state.add_statement(
            f"{index_var} = tl.arange(0, {block_size_var}).to({env.triton_index_type()})"
        )
        state.codegen.set_active_loops(PersistentReductionState(self))

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
        default = 0 if scan_type == "sum" else 1
        expr = self._maybe_mask(state, fake_input, dim, input_name, default)
        
        if scan_type == "sum":
            call = f"tl.cumsum({expr}, {dim})"
        elif scan_type == "prod":
            call = f"tl.cumprod({expr}, {dim})"
        else:  # generic
            call = (
                f"tl.associative_scan({expr}, {dim}, "
                f'combine_fn=triton_helpers.get_scan_combine_fn("{scan_type}"))'
            )
        return expr_from_string(call)


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

    def _make_device_loop(self, state: CodegenState) -> DeviceLoopState:
        env = CompileEnvironment.current()
        bidx = self.block_index
        numel = env.block_sizes[bidx].numel

        off_var = self.offset_var(bidx)
        idx_var = self.index_var(bidx)
        blksz_var = self.fn.block_size_var_cache[(bidx,)]

        # host constant
        if state.device_function.constexpr_arg(blksz_var):
            state.codegen.host_statements.append(
                statement_from_string(f"{blksz_var} = {self.block_size!r}")
            )

        body = [
            statement_from_string(
                f"{idx_var} = {off_var} + tl.arange(0, {blksz_var}).to({env.triton_index_type()})"
            )
        ]
        if self._mask_var:
            body.append(
                statement_from_string(
                    f"{self._mask_var} = {idx_var} < {self.fn.sympy_expr(numel)}"
                )
            )

        for_node = create(
            ast.For,
            target=create(ast.Name, id=off_var, ctx=ast.Store()),
            iter=expr_from_string(
                f"range(0, ({self.fn.sympy_expr(numel)}), {blksz_var})"
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        return DeviceLoopState(self, for_node, body)

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
        dl = self._make_device_loop(state)
        state.codegen.set_active_loops(dl)  # push
        rank = fake_input.dim()
        shape = self.fn.tile_strategy.shape_str(fake_input.size())

        zero_or_one = 0 if scan_type == "sum" else 1
        carry = self.fn.new_var(f"{state.fx_node.name}_carry", dce=True)
        dl.outer_prefix.append(
            statement_from_string(
                f"{carry} = tl.full({shape}, {constant_repr(zero_or_one)}, "
                f"{triton_acc_type(fake_input.dtype)})"
            )
        )

        masked = self._maybe_mask(state, fake_input, dim, input_name, zero_or_one)
        if scan_type == "sum":
            tile_scan = f"tl.cumsum({masked}, {dim})"
            op = "+"
        elif scan_type == "prod":
            tile_scan = f"tl.cumprod({masked}, {dim})"
            op = "*"
        else:
            tile_scan = (
                f"tl.associative_scan({masked}, {dim}, "
                f'combine_fn=triton_helpers.get_scan_combine_fn("{scan_type}"))'
            )
            # no generic symbol in Python; we call helper after code-gen
            op = f"triton_helpers.{scan_type}_combine"

        result = self.fn.new_var(state.fx_node.name, dce=True)
        state.add_statement(f"{result} = {carry} {op} {tile_scan}")

        # pick last element along dim -> update carry
        idxs = [":" for _ in range(rank)]
        idxs[dim] = "-1"
        last = f"{result}[{', '.join(idxs)}]"
        if scan_type == "sum" or scan_type == "prod":
            state.add_statement(f"{carry} = {last}")
        else:
            state.add_statement(f"{carry} = {last}")  # generic

        dl.outer_suffix.append(statement_from_string(f"{result} = {result}"))
        return expr_from_string(result)


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
    
    # If scan_loop is not provided, try to get it from config
    if scan_loop is None:
        # Check if we have scan_loops in the current config
        if hasattr(env, 'config') and hasattr(env.config, 'scan_loops') and env.config.scan_loops:
            # For now, use the first scan_loop value (similar to reduction)
            scan_loop = env.config.scan_loops[0] if env.config.scan_loops else None
    
    # If still no scan_loop, use heuristic
    if scan_loop is None:
        numel = env.block_sizes[block_index].numel
        # heuristic: next power-of-2 ≤ 1024 (fits a warp on Ada/Lovelace)
        if isinstance(numel, int):
            auto_block_size = min(1024, 1 << (numel - 1).bit_length())
            # If the axis fits entirely within the block size, use persistent strategy
            if numel <= auto_block_size:
                return PersistentScanStrategy(fn, block_index)
            else:
                return LoopedScanStrategy(fn, block_index, auto_block_size)
        else:
            # For symbolic sizes, default to looped with block size 1024
            return LoopedScanStrategy(fn, block_index, 1024)
    else:
        # scan_loop provided (from config or explicit), use looped strategy
        return LoopedScanStrategy(fn, block_index, scan_loop)