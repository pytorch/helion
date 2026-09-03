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

    from ..compile_environment import CompileEnvironment
    from ..inductor_lowering import CodegenState


def _flydsl_rolled_col_block(
    env: CompileEnvironment,
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> int | None:
    """Block id of the rolled column loop a ``:`` slice iterates over.

    A whole-column ``:`` (e.g. ``x[tile_m, :]``) is wider than one wavefront (64
    threads), so Helion rolls it into a runtime loop over 64-wide chunks. Map the
    ``:`` to that loop's block id so the column is indexed by the loop element
    (``roffset + lane``, one element/thread, correct for any N) rather than a bare
    ``thread_idx.x`` that only covers the first 64 lanes.
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

    A single BufferCopy tops out at 128 bits (e.g. fp32 V=8 = 256b is not
    representable); reject it cleanly instead of a raw KeyError.
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
    """Per-element column predicate that masks the rolled column-loop tail."""

    _elems = ", ".join(str(j) for j in range(vec))
    iota = f"fx.Vector.from_elements([{_elems}], fx.Int32)"
    col_base = f"({offset_var} + (fx.thread_idx.x % {lane_mod}) * {vec})"
    return f"(({col_base}) + {iota}) < ({n_expr})"


def _flydsl_buffer_setup(
    env: CompileEnvironment,
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_name: str,
    subscript: list[object] | tuple[object, ...],
) -> dict:
    """Build the cached buffer/div/copy-atom setup shared by load and store.

    The loop-invariant prologue both need: find row/column block ids, detect a
    rolled ``:`` loop, derive the per-thread vec width and (for a tail) the mask
    predicate, then build the buffer/logical_divide/copy_atom trio (memoized per
    ``(tensor, is_vec, is_rolled_col)``). Only tail handling differs by caller.
    """
    row_block_id = 0
    col_block_id = None
    seen_row = False
    if tensor.ndim == 1:
        # A 1-D tensor (e.g. weight[tile_n]) is indexed along its column and
        # broadcasts across rows, so its sole block id is the column, not the
        # row -- route it through the vectorized path like the 2-D operands.
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

    # Rolled column ``:``: a whole-column slice rolled into a loop over 64-wide
    # chunks. Only when no explicit tile-n block id was found in the subscript.
    is_rolled_col = False
    if col_block_id is None:
        _rc = _flydsl_rolled_col_block(env, state, tensor, subscript)
        if _rc is not None:
            col_block_id = _rc
            is_rolled_col = True
    # Per-thread vec width for a rolled loop = chunk / threads (V contiguous
    # elems/thread), read from the loop strategy. rolled_col_offset is the loop's
    # element offset; the chunk index is ``offset // V + lane``.
    rolled_col_vec = 1
    rolled_col_offset: str | None = None
    _tc = 0
    _lb = 0
    if is_rolled_col:
        assert col_block_id is not None
        _rs = state.device_function.tile_strategy.block_id_to_strategy.get(
            (col_block_id,)
        )
        _tc = getattr(_rs, "_thread_count", 0) or 0
        _lb = getattr(_rs, "_loop_block_size", 0) or 0

        if _rs is not None and _tc > 0 and _lb >= _tc:
            rolled_col_vec = max(1, _lb // _tc)
            rolled_col_offset = _rs.offset_var(col_block_id)
    # Column tail: when N is not a multiple of the chunk (64*V), the last pass
    # runs past column N. Build a per-element predicate to drop those columns.
    rolled_col_pred: str | None = None
    if is_rolled_col and rolled_col_offset is not None:
        assert col_block_id is not None
        _numel = env.block_sizes[col_block_id].numel
        if not env.known_multiple(_numel, _lb):
            rolled_col_pred = _flydsl_col_tail_pred(
                state,
                rolled_col_offset,
                rolled_col_vec,
                state.sympy_expr(_numel),
                lane_mod=_tc if _tc > 0 else 64,
            )
    # Scalar path (col_block_id None): thread t loads column t, M must equal 64.
    # Vectorized path (not None): 4 elems/thread, 256/tile. Separate div/atom per
    # (tensor, mode) so both paths can coexist for the same tensor.
    is_vec = col_block_id is not None

    device_fn = state.device_function
    if not hasattr(device_fn, "_flydsl_setup"):
        device_fn._flydsl_setup = {}  # pyrefly: ignore[missing-attribute]

    setup_key = (tensor_name, is_vec, is_rolled_col)
    setup = device_fn._flydsl_setup.get(setup_key)  # pyrefly: ignore[missing-attribute]
    if setup is None:
        _hoist = None
        # A rolled ``:`` column that fits in one wavefront (N <= 64) has a loop
        # that is never emitted, so its outer_prefix is dead and hoisting setup
        # there drops it. Keep _hoist None so setup emits inline in the live body.
        _degenerate_rolled_col = is_rolled_col and rolled_col_offset is None
        if (
            col_block_id is not None
            and not _degenerate_rolled_col
            and state.codegen.active_device_loops[col_block_id]
        ):
            _loop = state.codegen.active_device_loops[col_block_id][-1]
            _hoist = _loop.outer_prefix  # pyrefly: ignore[missing-attribute]
        # Under a runtime scf.for the loop body is lifted into its own function,
        # so setup emitted inside it is invisible to a sibling body (NameError).
        # Redirect these lifts to the loop's outer_prefix (enclosing scope) so
        # both bodies can close over them; a no-op (None) otherwise.
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
                # 1-D load: the whole rank-1 buffer is the vector, no row to
                # slice. Slicing a rank-1 memref with a 2-tuple fails flydsl's
                # profile check.
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
                # Vectorized path tiles 256 columns/warp across 64 threads = 4
                # elems/thread (design invariant). Not 8: that would leave half
                # the warp idle and give wrong fp16 results.
                vec_width = rolled_col_vec if is_rolled_col else 4
            else:
                # Scalar path: vec_width = col // 64 threads (1/2/4).
                # M=64 -> 1x32b, M=128 -> 2x64b, M>=256 -> 4x128b.
                _col = tensor.shape[-1]
                vec_width = max(1, min(4, _col // 64))
                if vec_width == 3:
                    vec_width = 2
            # Copy width = vec_width * element bits (dtype-aware), NOT vec_width *
            # 32 -- the latter pairs a 128-bit copy with a 64-bit fp16 register
            # (invalid ``bitcast i128 -> vector<4xf16>``).
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
        device_fn._flydsl_setup[setup_key] = setup  # pyrefly: ignore[missing-attribute]

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
def _(state: CodegenState) -> None:
    from ..compile_environment import CompileEnvironment
    from ..compile_environment import _symint_sympy_expr

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
        # Store vs load are ASYMMETRIC in the tail: a STORE always uses guarded
        # per-element scalar stores (``if col < N``), never a wide write into the
        # tail, so it can't touch the next row or memory past the tensor -- no
        # page-fault guard needed. A LOAD instead issues a wide read and masks
        # after (faster for the common small tail), so it needs the extra
        # ``_rolled_col_large_oob`` guard for reads that cross an unmapped page.
        #
        # Explicit hl.tile(n) tail: N not a multiple of the per-pass span (4*64)
        # -> the last chunk's high lanes address past column N.
        _expl_tail = (
            is_vec
            and not is_rolled_col
            and not env.known_multiple(
                env.block_sizes[col_block_id].numel,
                4 * 64,
            )
        )
        if rolled_col_pred is not None:
            # Rolled tail: the wide BufferCopy can't be predicated per element,
            # so emit one guarded scalar store per element. Element j is column
            # ``offset + lane*V + j``; store only when in range.
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
            # Explicit-tile tail: guarded scalar stores. Element j is column
            # ``chunk_idx * V + j``; store only when in range.
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
                parts.append(state.sympy_expr(_symint_sympy_expr(idx)))
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
    from ..compile_environment import _symint_sympy_expr

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

        # Explicit hl.tile(n) tail: N not a multiple of the per-pass span (4*64)
        # -> the last chunk's high lanes would read past column N. The AMD buffer
        # descriptor should return 0 for OOB reads but in practice still page-
        # faults once the byte address crosses the tensor's allocation. Use
        # guarded scalar loads so no access goes past the last valid column.
        _expl_tail = (
            is_vec
            and not is_rolled_col
            and not env.known_multiple(
                env.block_sizes[col_block_id].numel,
                4 * 64,
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
            # Explicit-tile tail: guarded scalar loads to avoid reading past the
            # allocation. Element j is column ``chunk_idx * V + j``; load when in
            # range, else write the zero identity into r_var.
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

        # Rolled tail with large OOB: when the wide read overshoots N by more than
        # AMD's buffer-descriptor slack (~512 bytes), copy_atom_call page-faults
        # even though the predicate would zero the result after. Fall back to
        # guarded scalar loads. (No store counterpart: store never reads/writes
        # the tail wide.) OOB distance = chunk//V + thread_count - 1 - numel//V.
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
                # Compare in BYTES against ROCm's allocator rounding slack (512).
                # The descriptor returns 0 for OOB reads within mapped pages but
                # faults once the address crosses an unmapped page. Bytes (not
                # elements) makes the threshold dtype-agnostic: fp16 256, fp32
                # 128, int64 64.
                _elem_bytes = tensor.element_size()
                _SAFE_OOB_BYTES = 512
                if _oob_elems * _elem_bytes > _SAFE_OOB_BYTES:
                    _rolled_col_large_oob = True

        if _rolled_col_large_oob:
            # Guarded scalar loads: element j is column
            # rolled_col_offset + lane*V + j (same as _colj in the store).
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
            # Zero out-of-range tail columns so masked lanes contribute the
            # identity (0). Use a shape-only zero (``filled_like(_lv, 0)``) NOT
            # ``_lv - _lv``: a garbage tail read can be nan, and nan-nan is nan,
            # which ``.select`` would keep.
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
                parts.append(state.sympy_expr(_symint_sympy_expr(idx)))
        elif isinstance(idx, int):
            parts.append(str(idx))
    return expr_from_string(f"{tensor_name}[{', '.join(parts)}]")
