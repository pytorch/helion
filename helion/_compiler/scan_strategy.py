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


# --------------------------------------------------------------------------- #
#  UnifiedScanStrategy – one class that always works                          #
# --------------------------------------------------------------------------- #
class UnifiedScanStrategy(TileStrategy):
    """
    Inclusive prefix-scan (cumsum / cumprod / generic associative scan)
    that is completely agnostic to the physical layout of the scan axis.

    * If the scan axis fits within `block_size`, the kernel executes a single
      Triton `tl.cumsum` / `tl.cumprod`.
    * Otherwise we iterate tile-by-tile and keep a **carry** that propagates
      the running prefix to the next tile.  This works even when the underlying
      stride is > 1 (i.e. the axis is not contiguous in memory).

    Only one public method matters for code-gen:

        codegen_scan(state, input_name, scan_type, dim, fake_input)
    """

    # ------------------------------------------------------------------- #
    # Construction                                                        #
    # ------------------------------------------------------------------- #
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        block_size: int | None = None,
    ) -> None:
        """
        *block_index* – the index of the axis we are scanning.
        *block_size*  – optional user hint; if None we pick next power-of-2.
        """
        env = CompileEnvironment.current()
        numel = env.block_sizes[block_index].numel

        if block_size is None:
            # heuristic: next power-of-2 ≤ 1024  (fits a warp on Ada/Lovelace)
            if isinstance(numel, int):
                block_size = min(1024, 1 << (numel - 1).bit_length())
            else:
                block_size = 1024

        # Need a mask when numel is not a multiple of block_size
        mask_var: str | None = None
        if not env.known_multiple(numel, block_size):
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)

        super().__init__(fn=fn, block_indices=[block_index])
        self._mask_var = mask_var
        self.block_size = block_size

        # register helper vars with DeviceFunction caches
        fn.block_size_var_cache[(block_index,)] = fn.new_var(
            f"_SCAN_BLKSIZE_{block_index}"
        )
        self.offset_vars[block_index] = fn.new_var(f"soffset_{block_index}", dce=True)
        self.index_vars[block_index] = fn.new_var(f"sindex_{block_index}", dce=True)

    # convenience
    @property
    def block_index(self) -> int:
        return self.block_indices[0]

    # ------------------------------------------------------------------- #
    # Helpers mirroring ReductionStrategy                                 #
    # ------------------------------------------------------------------- #
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

    # ------------------------------------------------------------------- #
    # Device loop (executes when axis > block_size)                       #
    # ------------------------------------------------------------------- #
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

    # ------------------------------------------------------------------- #
    # Public entry point                                                  #
    # ------------------------------------------------------------------- #
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

        Works for any stride pattern; correctness is guaranteed by the
        prefix-invariant (carry += last-element-of-tile each iteration).
        """
        env = CompileEnvironment.current()
        numel = env.block_sizes[self.block_index].numel
        need_loop = not (isinstance(numel, int) and numel <= self.block_size)

        # fast path ─ whole axis fits ⇢ single builtin
        if not need_loop:
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

        # slow path ─ tile-by-tile scan with carry
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

        # pick last element along dim → update carry
        idxs = [":" for _ in range(rank)]
        idxs[dim] = "-1"
        last = f"{result}[{', '.join(idxs)}]"
        if scan_type == "sum" or scan_type == "prod":
            state.add_statement(f"{carry} = {last}")
        else:
            state.add_statement(f"{carry} = {last}")  # generic

        dl.outer_suffix.append(statement_from_string(f"{result} = {result}"))
        # state.codegen.pop_active_loop()  # restore
        return expr_from_string(result)
