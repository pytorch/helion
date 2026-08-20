"""FlyDSL-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "flydsl")``
registrations; ``_codegen_modules`` imports it so registration keeps the same
eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


def _flydsl_rolled_col_block(
    env: object, state: CodegenState, tensor: torch.Tensor, subscript: object
) -> int | None:
    """Block id of the rolled column loop a ``:`` slice iterates over.

    A whole-column ``:`` slice (e.g. ``x[tile_m, :]``) is wider than one
    wavefront (64 threads), so Helion rolls the ``:`` axis into a runtime loop
    over 64-wide chunks (``for roffset in range(0, N, chunk)``). This maps such
    a ``:`` to that loop's block id so the load/store indexes the column by the
    loop's element index (``roffset + lane``, one element per thread) — correct
    for any N — instead of a bare ``thread_idx.x`` that only covers the first 64
    lanes (which would make every iteration overwrite the same first chunk).
    """
    loop_blocks = [
        bs.block_id
        for bs in env.block_sizes
        if bs.reduction and state.codegen.active_device_loops.get(bs.block_id)
    ]
    if not loop_blocks:
        return None

    for ax, idx in enumerate(subscript):
        if isinstance(idx, slice) and idx == slice(None):
            try:
                ax_hint = env.size_hint(tensor.size(ax))
            except Exception:
                continue
            for bid in loop_blocks:
                if env.block_sizes[bid].size_hint() == ax_hint:
                    return bid
    return None


_FLYDSL_COPY_CLS = {
    8: "BufferCopy8b",
    16: "BufferCopy16b",
    32: "BufferCopy32b",
    64: "BufferCopy64b",
    128: "BufferCopy128b",
}


def _flydsl_copy_cls(bits: int) -> str:
    """BufferCopy class for a ``bits``-wide transaction, else BackendUnsupported.

    A single BufferCopy tops out at 128 bits, so e.g. fp32 V=8 (256-bit) is not
    representable -- reject it cleanly instead of raising a raw KeyError. The
    autotuner only enumerates V that keep ``V * element_bits <= 128``.
    """
    cls = _FLYDSL_COPY_CLS.get(bits)
    if cls is None:
        raise exc.BackendUnsupported(
            "flydsl", f"vector copy width {bits} bits (max 128; reduce vec width V)"
        )
    return cls


def _flydsl_col_tail_pred(
    state: CodegenState, offset_var: str, vec: int, n_expr: str, lane_mod: int = 64
) -> str:
    """Per-element column predicate that masks the rolled column-loop tail.

    Lane ``l`` at loop offset ``offset_var`` owns columns
    ``offset_var + l*vec + j`` for ``j in range(vec)``. Returns a length-``vec``
    bool vector ``col < N`` so out-of-range columns (present when N is not a
    multiple of the chunk ``lane_mod*vec``) can be dropped from loads/stores.
    ``lane_mod`` is the column-loop thread_count (= 64*W); lanes span
    0..lane_mod-1 across the W wavefronts.
    """

    _elems = ", ".join(str(j) for j in range(vec))
    iota = f"fx.Vector.from_elements([{_elems}], fx.Int32)"
    col_base = f"({offset_var} + (fx.thread_idx.x % {lane_mod}) * {vec})"
    return f"(({col_base}) + {iota}) < ({n_expr})"


def _flydsl_buffer_setup(
    env: object,
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_name: str,
    subscript: object,
) -> dict:
    """Resolve indexing and build the (cached) buffer/div/copy-atom setup shared
    by the flydsl buffer load and store codegen.

    Both load and store need the identical loop-invariant prologue: find the
    row/column block ids, detect a rolled ``:`` column loop, derive the per-thread
    vector width and (for a tail) the masking predicate, then build the
    ``make_buffer_tensor`` / ``logical_divide`` / ``make_copy_atom`` trio (memoized
    on ``device_function._flydsl_setup`` per ``(tensor, is_vec, is_rolled_col)``).
    Returns a dict the callers consume; the write/read tail handling is the only
    part that differs between store and load.
    """
    backend = env.backend

    row_block_id = 0
    col_block_id = None
    seen_row = False
    if tensor.ndim == 1:
        # A 1-D tensor loaded alongside 2-D tiles (e.g. weight[tile_n]) is
        # indexed along its column/vector dimension and broadcasts across
        # rows, so its sole block id is the column, not the row. This routes
        # it through the vectorized path with the same 128-bit vec width and
        # column chunk index as the 2-D operands it multiplies against.
        for idx in subscript:
            if isinstance(idx, torch.SymInt):
                bid = env.get_block_id(idx)
                if bid is not None:
                    col_block_id = bid
                    break
    else:
        for idx in subscript:
            if isinstance(idx, torch.SymInt):
                bid = env.get_block_id(idx)
                if bid is not None:
                    if not seen_row:
                        row_block_id = bid
                        seen_row = True
                    else:
                        col_block_id = bid
                        break

    # Rolled column ``:``: a whole-column slice wider than one wavefront is
    # rolled into a loop over 64-wide chunks. Index the column by the loop's
    # element index (roffset + lane), one element per thread (vec_width 1),
    # so any N is covered by looping over the axis. Only when no explicit
    # tile-n block id was found in the subscript.
    is_rolled_col = False
    if col_block_id is None:
        _rc = _flydsl_rolled_col_block(env, state, tensor, subscript)
        if _rc is not None:
            col_block_id = _rc
            is_rolled_col = True
    # Per-thread vec width for a rolled column loop = chunk / threads
    # (V contiguous elems/thread). Read it from the loop strategy so the
    # load tiles V elements per thread and loops over N. rolled_col_offset is
    # the loop's element offset; the chunk index is ``offset // V + lane``.
    rolled_col_vec = 1
    rolled_col_offset: str | None = None
    _tc = 0
    _lb = 0
    if is_rolled_col:
        _rs = state.device_function.tile_strategy.block_id_to_strategy.get(
            (col_block_id,)
        )
        _tc = getattr(_rs, "_thread_count", 0) or 0
        _lb = getattr(_rs, "_loop_block_size", 0) or 0

        if _rs is not None and _tc > 0 and _lb >= _tc:
            rolled_col_vec = max(1, _lb // _tc)
            rolled_col_offset = _rs.offset_var(col_block_id)
    # Column tail: when N is not a multiple of the chunk (64*V), the last
    # loop pass reads/writes past column N. Build a per-element predicate to
    # drop those out-of-range columns from loads and stores.
    rolled_col_pred: str | None = None
    if is_rolled_col and rolled_col_offset is not None:
        _numel = env.block_sizes[col_block_id].numel
        if not env.known_multiple(_numel, _lb):
            rolled_col_pred = _flydsl_col_tail_pred(
                state,
                rolled_col_offset,
                rolled_col_vec,
                state.sympy_expr(_numel),
                lane_mod=_tc if _tc > 0 else 64,
            )
    # Scalar path (col_block_id is None): x[tile_m,:] -- thread t loads column t, M must equal 64.
    # Vectorized path (col_block_id is not None): x[tile_m,tile_n] -- 4 elems/thread, 256/tile.
    # Keep separate div/atom per (tensor, mode) so both paths can coexist for the same tensor.
    is_vec = col_block_id is not None
    if getattr(backend, "_flydsl_warps_per_row", 1) > 1:
        assert col_block_id is not None, (
            "flydsl W>1 regime requires the vectorized (2D-tile) path"
        )

    device_fn = state.device_function
    if not hasattr(device_fn, "_flydsl_setup"):
        device_fn._flydsl_setup = {}

    setup_key = (tensor_name, is_vec, is_rolled_col)
    setup = device_fn._flydsl_setup.get(setup_key)
    if setup is None:
        _hoist = None
        if col_block_id is not None and state.codegen.active_device_loops[col_block_id]:
            _hoist = state.codegen.active_device_loops[col_block_id][-1].outer_prefix
        # Loop-invariant setup (buffer/row/logical_divide/copy_atom). Under a
        # runtime scf.for column loop the loop body is lifted into its own
        # function, so setup emitted inside it is invisible to a sibling
        # loop body (NameError). set_statements redirects these lifts to the
        # column loop's outer_prefix (enclosing scope) so both loop-body
        # functions can close over them; a no-op (None) otherwise.
        with state.codegen.set_statements(_hoist):
            bid_expr = state.codegen.index_var(
                row_block_id
            )  # pid*bm + warp_id (warp-per-row)
            buf = state.codegen.lift(
                expr_from_string(
                    f"fx.rocdl.make_buffer_tensor({tensor_name})",
                ),
                prefix="flydsl_buf",
                dce=True,
            )
            if tensor.ndim == 1:
                # 1-D load (e.g. weight[:] broadcast or weight[tile_n]): the
                # whole rank-1 buffer is the vector; there is no separate row
                # dimension to slice. Slicing a rank-1 memref with a 2-tuple
                # fails the flydsl profile check.
                row = buf
            else:
                row = state.codegen.lift(
                    expr_from_string(
                        f"fx.slice({buf.id}, ({bid_expr}, None))",
                    ),
                    prefix="flydsl_row",
                    dce=True,
                )
            if is_vec:
                # The vectorized path tiles 256 columns/warp across 64 threads =
                # 4 elements/thread (design invariant, dtype-independent). The
                # copy transaction width follows from the dtype so the register
                # (vec_width * element bits) matches the copy. Previously this
                # used vec_width = 128 // bits (= 8 for fp16) with a hardcoded
                # 128-bit copy, which both over-provisioned the 64-thread tiling
                # (32 chunks vs 64 threads -> wrong fp16 results) and mismatched
                # the 64-bit fp16 register (invalid bitcast).

                vec_width = rolled_col_vec if is_rolled_col else 4
            else:
                # Derive vec_width from column size: col // 64 threads, supported widths 1/2/4.
                # M=64 -> 1x32b, M=128 -> 2x64b, M>=256 -> 4x128b.
                _col = tensor.shape[-1]
                vec_width = max(1, min(4, _col // 64))
                if vec_width == 3:
                    vec_width = 2
            # Copy width must match the register size = vec_width * element
            # bits, NOT vec_width * 32 (which assumed 32-bit elements and
            # paired a 128-bit copy with a 64-bit fp16 register, producing an
            # invalid ``bitcast i128 -> vector<4xf16>`` at LLVM translation).
            _elem_bits = tensor.element_size() * 8
            _bits = vec_width * _elem_bits

            _copy_cls = _flydsl_copy_cls(_bits)
            div = state.codegen.lift(
                expr_from_string(
                    f"fx.logical_divide({row.id}, fx.make_layout({vec_width}, 1))",
                ),
                prefix="flydsl_div",
                dce=True,
            )
            atom = state.codegen.lift(
                expr_from_string(f"fx.make_copy_atom(fx.rocdl.{_copy_cls}(), {_bits})"),
                prefix="flydsl_atom",
                dce=True,
            )

        setup = {"div": div.id, "atom": atom.id, "vec": vec_width, "row": row.id}
        device_fn._flydsl_setup[setup_key] = setup

    return {
        "setup": setup,
        "col_block_id": col_block_id,
        "is_vec": is_vec,
        "is_rolled_col": is_rolled_col,
        "rolled_col_offset": rolled_col_offset,
        "rolled_col_vec": rolled_col_vec,
        "rolled_col_pred": rolled_col_pred,
        "tc": _tc,
        "lb": _lb,
    }


@_decorators.codegen(store, "flydsl")
def _(state: CodegenState) -> ast.AST:
    from ..compile_environment import CompileEnvironment

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    value = state.ast_arg(2)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    env = CompileEnvironment.current()
    backend = env.backend
    tensor_name = state.device_function.tensor_arg(tensor).name
    use_buffer = getattr(backend, "_tensor_use_buffer", {}).get(id(tensor), False)

    if use_buffer:
        _info = _flydsl_buffer_setup(env, state, tensor, tensor_name, subscript)
        setup = _info["setup"]
        col_block_id = _info["col_block_id"]
        is_vec = _info["is_vec"]
        is_rolled_col = _info["is_rolled_col"]
        rolled_col_offset = _info["rolled_col_offset"]
        rolled_col_pred = _info["rolled_col_pred"]
        _tc = _info["tc"]

        dtype_str = backend.dtype_str(tensor.dtype)
        vec_width = setup["vec"]
        r_var = f"_r_{tensor_name}_s"
        state.add_statement(
            statement_from_string(
                f"{r_var} = fx.make_rmem_tensor({vec_width}, {dtype_str})"
            )
        )
        state.add_statement(
            statement_from_string(
                f"fx.memref_store_vec({{value}}, {r_var})", value=value
            )
        )

        if is_rolled_col and rolled_col_offset is not None:
            # V contiguous elems/thread at chunk index ``offset // V + lane``.
            # Lane spans the column loop's thread_count (= 64*W waves/row), not 64.
            _lane_mod = _tc if _tc > 0 else 64
            chunk_idx_s = (
                f"({rolled_col_offset}) // {vec_width} + fx.thread_idx.x % {_lane_mod}"
            )
        elif is_vec:
            chunk_idx_s = state.codegen.index_var(col_block_id)
        else:
            chunk_idx_s = "fx.thread_idx.x"

        _atom_name = setup["atom"]
        _div_name = setup["div"]
        # Tail handling, and why store and load are ASYMMETRIC here:
        #
        # In a rolled column loop the last iteration has lanes whose column index
        # runs past N. A STORE handles this by emitting guarded per-element scalar
        # stores (``if col < N: memref_store(...)``) in every tail case below -- it
        # NEVER issues a wide vectorized write into the tail, so it can never write
        # into the next row and never touches memory past the tensor. Because the
        # write is already element-guarded, no page-fault guard is needed.
        #
        # A LOAD (see the load codegen) instead issues an UNCONDITIONAL vectorized
        # BufferCopy for the whole chunk and masks the tail lanes AFTERWARD
        # (``pred.select(loaded, 0)``) -- faster (one copy vs V scalar loads) for
        # the common small tail. But that physically READS the out-of-range columns,
        # so when the tail runs far enough past N to cross into an unmapped page it
        # page-faults before the mask can help. Hence load carries an extra
        # ``_rolled_col_large_oob`` guard that falls back to guarded scalar loads;
        # store needs no such guard because it never reads/writes the tail wide.
        #
        # Explicit hl.tile(n) column tail: N not a multiple of the per-iteration
        # span (4 * 64 * W) -> the last chunk's high lanes address past column N.
        _expl_tail = (
            is_vec
            and not is_rolled_col
            and not env.known_multiple(
                env.block_sizes[col_block_id].numel,
                4 * 64 * (getattr(backend, "_flydsl_warps_per_row", 1) or 1),
            )
        )
        if rolled_col_pred is not None:
            # Column tail: the vectorized BufferCopy is a single transaction and
            # cannot be predicated per element, so (like CuTe's scalar edge
            # fallback) emit one guarded scalar store per element. Element j of
            # this thread is column ``offset + lane*V + j``; store it only when
            # in range so tail columns are never written into the next row.
            _n_expr = state.sympy_expr(env.block_sizes[col_block_id].numel)
            _row = setup["row"]
            _lane_mod = _tc if _tc > 0 else 64
            for _j in range(vec_width):
                _colj = (
                    f"{rolled_col_offset} + (fx.thread_idx.x % {_lane_mod}) "
                    f"* {vec_width} + {_j}"
                )
                state.add_statement(
                    statement_from_string(
                        f"if ({_colj}) < ({_n_expr}):\n"
                        f"    fx.memref_store(fx.memref_load({r_var}, {_j}), {_row}, {_colj})"
                    )
                )
        elif _expl_tail:
            # Explicit-tile column tail: guarded scalar stores (the vectorized
            # BufferCopy can't be per-element predicated). Element j of this thread
            # is column ``chunk_idx * 4 + j``; store only when in range so tail
            # columns are never written into the next row.
            _n_expr = state.sympy_expr(env.block_sizes[col_block_id].numel)
            _row = setup["row"]
            for _j in range(vec_width):
                _colj = f"(({chunk_idx_s}) * {vec_width}) + {_j}"
                state.add_statement(
                    statement_from_string(
                        f"if ({_colj}) < ({_n_expr}):\n"
                        f"    fx.memref_store(fx.memref_load({r_var}, {_j}), {_row}, {_colj})"
                    )
                )
        else:
            state.add_statement(
                statement_from_string(
                    f"fx.copy_atom_call({_atom_name}, {r_var}, fx.slice({_div_name}, (None, {chunk_idx_s})))"
                )
            )
        return None
    # simple element-wise store
    parts = []
    for idx in subscript:
        if idx is None:
            continue
        if isinstance(idx, slice) and idx == slice(None):
            parts.append("fx.thread_idx.x")
        elif isinstance(idx, torch.SymInt):
            bid = env.get_block_id(idx)
            if bid is not None:
                parts.append(state.codegen.index_var(bid))
            else:
                parts.append(state.sympy_expr(idx))
        elif isinstance(idx, int):
            parts.append(str(idx))
    idx_str = ", ".join(parts)
    state.add_statement(
        statement_from_string(f"{tensor_name}[{idx_str}] = {{value}}", value=value)
    )
    return None


@_decorators.codegen(load, "flydsl")
def _(state: CodegenState) -> ast.AST:
    from ..compile_environment import CompileEnvironment

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    env = CompileEnvironment.current()
    backend = env.backend
    tensor_name = state.device_function.tensor_arg(tensor).name
    use_buffer = getattr(backend, "_tensor_use_buffer", {}).get(id(tensor), False)

    if use_buffer:
        # buffer tensor path for both x[tile_n,:] and x[tile_m,tile_n].
        _info = _flydsl_buffer_setup(env, state, tensor, tensor_name, subscript)
        setup = _info["setup"]
        col_block_id = _info["col_block_id"]
        is_vec = _info["is_vec"]
        is_rolled_col = _info["is_rolled_col"]
        rolled_col_offset = _info["rolled_col_offset"]
        rolled_col_pred = _info["rolled_col_pred"]
        _tc = _info["tc"]
        _lb = _info["lb"]

        dtype_str = backend.dtype_str(tensor.dtype)
        vec_width = setup["vec"]
        r_var = f"_r_{tensor_name}"

        if is_rolled_col and rolled_col_offset is not None:
            # V contiguous elems/thread at chunk index ``offset // V + lane``.
            # Lane spans the column loop's thread_count (= 64*W waves/row), not 64.
            _lane_mod = _tc if _tc > 0 else 64
            chunk_idx = (
                f"({rolled_col_offset}) // {vec_width} + fx.thread_idx.x % {_lane_mod}"
            )
        elif is_vec:
            chunk_idx = state.codegen.index_var(col_block_id)
        else:
            chunk_idx = "fx.thread_idx.x"

        # Explicit hl.tile(n) column tail: N not a multiple of the per-iteration
        # span (4 * 64 * W) -> the last chunk's high lanes would read past column N
        # via the vectorized BufferCopy. The AMD buffer descriptor has max_size set
        # to 0xFFFFFFFF which should return 0 for OOB reads, but in practice the
        # hardware still fires a GPU page fault when the byte address exceeds the
        # tensor's actual allocation (e.g. N=640, last tile reads cols 640-767 which
        # falls one page past the end). Use guarded scalar loads instead so no
        # memory access ever goes past the last valid column.
        _expl_tail = (
            is_vec
            and not is_rolled_col
            and not env.known_multiple(
                env.block_sizes[col_block_id].numel,
                4 * 64 * (getattr(backend, "_flydsl_warps_per_row", 1) or 1),
            )
        )

        state.add_statement(
            statement_from_string(
                f"{r_var} = fx.make_rmem_tensor({vec_width}, {dtype_str})"
            )
        )
        _atom_name = setup["atom"]
        _div_name = setup["div"]
        _row = setup["row"]

        if _expl_tail:
            # Explicit-tile column tail: use guarded scalar loads to avoid reading
            # past the tensor allocation. Element j of this thread is column
            # ``chunk_idx * vec_width + j``; load from the row buffer only when
            # in range, otherwise write the zero identity into r_var.
            _n_expr = state.sympy_expr(env.block_sizes[col_block_id].numel)
            _zero = f"{dtype_str}(0)"
            for _j in range(vec_width):
                _colj = f"(({chunk_idx}) * {vec_width}) + {_j}"
                state.add_statement(
                    statement_from_string(
                        f"if ({_colj}) < ({_n_expr}):\n"
                        f"    fx.memref_store(fx.memref_load({_row}, {_colj}), {r_var}, {_j})\n"
                        f"else:\n"
                        f"    fx.memref_store({_zero}, {r_var}, {_j})"
                    )
                )
            return expr_from_string(f"fx.memref_load_vec({r_var})")

        # Rolled column-loop tail with large OOB: when the last loop pass
        # has threads whose chunk_idx exceeds the logical_divide layout by more than
        # AMD's buffer-descriptor padding (~256 fp16 elements = 512 bytes), the
        # unconditional copy_atom_call fires a GPU page fault even though the
        # per-element predicate would zero the result afterward.  Fall back to
        # guarded scalar loads (element-wise if-in-range) for these cases.
        #
        # This guard has NO store counterpart on purpose (see the store codegen's
        # tail comment): the store always writes the tail with guarded per-element
        # scalar stores, so it never issues a wide access that can fault. The load
        # keeps the fast unconditional vectorized read for the common small tail and
        # only falls back to scalar loads here, when the OOB distance is unsafe.
        # OOB distance = chunk_idx_max − (numel // vec_width)
        #              = chunk // V + thread_count − 1 − numel // V
        # When _tc > 0 and _lb is set, compute and compare to the safe threshold.
        _rolled_col_large_oob = False
        if (
            is_rolled_col
            and rolled_col_pred is not None
            and _tc > 0
            and _lb > 0
            and vec_width > 0
        ):
            import contextlib as _cl

            _numel_int: int | None = None
            _numel_sym = env.block_sizes[col_block_id].numel
            with _cl.suppress(TypeError, ValueError):
                _numel_int = int(_numel_sym)
            if _numel_int is not None:
                _chunk_idx_max = _lb // vec_width + _tc - 1
                _div_layout_size = (_numel_int + vec_width - 1) // vec_width
                _oob_elems = _chunk_idx_max - _div_layout_size
                # Convert OOB elements to bytes and compare against the
                # conservative lower-bound for PyTorch's ROCm allocator rounding
                # (512 bytes).  AMD's buffer descriptor hardware returns 0 for OOB
                # reads within allocated physical pages but page-faults once the
                # byte address crosses into an unmapped page.  Using bytes (not
                # elements) makes the threshold dtype-agnostic: fp16 allows
                # 512/2=256 OOB elements, fp32 allows 512/4=128, int64 512/8=64.
                _elem_bytes = tensor.element_size()
                _SAFE_OOB_BYTES = 512
                if _oob_elems * _elem_bytes > _SAFE_OOB_BYTES:
                    _rolled_col_large_oob = True

        if _rolled_col_large_oob:
            # Use per-element guarded scalar loads: column of element j is
            # rolled_col_offset + lane * vec_width + j (same as _colj in the store).
            _n_expr_rl = state.sympy_expr(env.block_sizes[col_block_id].numel)
            _lane_mod_rl = _tc if _tc > 0 else 64
            _zero_rl = f"{dtype_str}(0)"
            for _j in range(vec_width):
                _colj_rl = (
                    f"({rolled_col_offset}) + (fx.thread_idx.x % {_lane_mod_rl}) "
                    f"* {vec_width} + {_j}"
                )
                state.add_statement(
                    statement_from_string(
                        f"if ({_colj_rl}) < ({_n_expr_rl}):\n"
                        f"    fx.memref_store(fx.memref_load({_row}, {_colj_rl}), {r_var}, {_j})\n"
                        f"else:\n"
                        f"    fx.memref_store({_zero_rl}, {r_var}, {_j})"
                    )
                )
            return expr_from_string(f"fx.memref_load_vec({r_var})")

        state.add_statement(
            statement_from_string(
                f"fx.copy_atom_call({_atom_name}, fx.slice({_div_name}, (None, {chunk_idx})), {r_var})"
            )
        )

        if rolled_col_pred is not None:
            # Zero out-of-range tail columns so masked-out lanes contribute the
            # sum/mean identity (0) instead of neighbouring-row data. Use a
            # shape-only zero (``filled_like(_lv, 0)``) NOT ``_lv - _lv``: a tail
            # read past the tensor can return nan/inf garbage, and nan-nan is nan,
            # which ``.select`` would then keep.
            _lv = f"_lv_{r_var}"
            state.add_statement(
                statement_from_string(f"{_lv} = fx.memref_load_vec({r_var})")
            )
            return expr_from_string(
                f"({rolled_col_pred}).select({_lv}, fx.Vector.filled_like({_lv}, 0))"
            )
        return expr_from_string(f"fx.memref_load_vec({r_var})")

    # simple element-wise path: x[tile_m, tile_n]
    parts = []
    for idx in subscript:
        if idx is None:
            continue
        if isinstance(idx, slice) and idx == slice(None):
            parts.append("fx.thread_idx.x")
        elif isinstance(idx, torch.SymInt):
            bid = env.get_block_id(idx)
            if bid is not None:
                parts.append(state.codegen.index_var(bid))
            else:
                parts.append(state.sympy_expr(idx))
        elif isinstance(idx, int):
            parts.append(str(idx))
    return expr_from_string(f"{tensor_name}[{', '.join(parts)}]")
