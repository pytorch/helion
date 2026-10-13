from __future__ import annotations

import contextlib
import dataclasses
import math
import operator
import re
import textwrap
import types
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast
from unittest.mock import patch

import pytest
import torch

# The renderer's DSL runtime helpers (``flex_score_mod_runtime``) import the
# CuTe DSL at module level; every test here renders through it.
pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import helion
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import flex_score_mod_runtime
from helion._compiler.cute.attention_plan import AttentionScorePlan
from helion._compiler.cute.attention_plan import FlexAuxBuffer
from helion._compiler.cute.attention_plan import FlexKvWalk
from helion._compiler.cute.attention_plan import FlexRegion
from helion._compiler.cute.flash_score_mod import FlexHookCode
from helion._compiler.cute.flash_score_mod import build_flex_region
from helion._compiler.cute.flash_score_mod import flex_aux_smem_candidates
from helion._compiler.cute.flash_score_mod import flex_aux_smem_layout
from helion._compiler.cute.flash_score_mod import flex_aux_smem_staged
from helion._compiler.cute.flash_score_mod import flex_coalesce_smem_bytes
from helion._compiler.cute.flash_score_mod import flex_coalesce_steps
from helion._compiler.cute.flash_score_mod import flex_coalesce_widths
from helion._compiler.cute.flash_score_mod import flex_hook_ops
from helion._compiler.cute.flash_score_mod import flex_hook_params
from helion._compiler.cute.flash_score_mod import flex_prefetch_count
from helion._compiler.cute.flash_score_mod import flex_prefetch_regs
from helion._compiler.cute.flash_score_mod import flex_prefetch_schedule
from helion._compiler.cute.flash_score_mod import render_flex_hook
from helion._compiler.cute.flex_coop_layout import coop_pitch
from helion._compiler.cute.flex_score_mod_runtime import coop_lane_map
from helion._compiler.helper_function import CodegenInterface
from helion._testing import TestCase
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl

if TYPE_CHECKING:
    import ast
    from collections.abc import Iterator
    from collections.abc import Sequence

# Geometry of the emulated kernel.
_HEADS = 4
_BATCH = 2
_CHUNK_COLS = 32
_ROWS = (0, 1, 5, 31, 32, 63, 64, 100, 127)
# Exclusive coordinate bounds covering every emulated geometry (``m_tile`` <= 3,
# ``kv_tile`` <= 7): the kernel's static bounds proof of table indices.
_BOUNDS = (_BATCH, _HEADS, 4 * 128, 8 * 128)


# ---------------------------------------------------------------------------
# Codegen context stand-ins (the live GenerateAST / DeviceFunction of a kernel
# compile; here only the naming and statement-capture surface is needed).
# ---------------------------------------------------------------------------


class _StubCodegen(CodegenInterface):
    def __init__(self) -> None:
        self.statements_stack: list[list[ast.AST]] = [[]]

    def add_statement(self, stmt: ast.AST | str | None) -> None:
        if stmt is None:
            return
        if isinstance(stmt, str):
            stmt = statement_from_string(stmt)
        self.statements_stack[-1].append(stmt)

    @contextlib.contextmanager
    def set_statements(self, new_statements: list[ast.AST] | None) -> Iterator[None]:
        if new_statements is None:
            yield
            return
        self.statements_stack.append(new_statements)
        try:
            yield
        finally:
            self.statements_stack.pop()


class _StubDeviceFunction:
    def __init__(self) -> None:
        self.codegen = _StubCodegen()
        self.codegen.device_function = self  # pyrefly: ignore [bad-assignment]
        self._counts: dict[str, int] = {}
        self.expr_to_var_info: dict[object, object] = {}
        self._constexpr_args: dict[str, object] = {}
        self.config = helion.Config()
        self.tile_strategy = types.SimpleNamespace(
            supports_index_rank_expansion=lambda: False,
            shape_str=lambda shape: str(shape),
        )

    def unique_name(self, prefix: str, dce: bool = False) -> str:
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        return f"{prefix}_{self._counts[prefix]}"

    def new_var(self, name: str, *, dce: bool = False) -> str:
        return self.unique_name(name, dce=dce)

    def user_sympy_expr(self, expr: object) -> str:
        return str(expr)


@contextlib.contextmanager
def _environment(fast_math: bool = False) -> Iterator[CompileEnvironment]:
    settings = helion.Settings(backend="cute", fast_math=fast_math)
    with (
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        CompileEnvironment(torch.device("cpu"), settings) as env,
    ):
        yield env


_WALK = FlexKvWalk("pc", "pi", "fc", "fi", 128, 128, 1, 1)


def _plan(
    score_region: FlexRegion | None,
    mask_region: FlexRegion | None,
    aux: Sequence[FlexAuxBuffer],
    scalars: Sequence[str],
    *,
    scale_arg: str | None = None,
) -> AttentionScorePlan:
    return AttentionScorePlan(
        head_dim=64,
        qk_scale_log2=1.0,
        kv_walk=_WALK,
        score_region=score_region,
        mask_region=mask_region,
        aux_buffers=tuple(aux),
        host_scalars=tuple(scalars),
        scale_arg=scale_arg,
        scale_peeled=score_region is None,
    )


def _render(
    plan: AttentionScorePlan,
    *,
    with_score: bool,
    with_mask: bool,
    tail_mask: bool,
    stage: int = 0,
    bounds: bool = False,
    chunk_cols: int = _CHUNK_COLS,
    mask_form: str = "bitmask",
    aux_smem_budget: int = 0,
    prefetch_regs: int = 0,
    coalesce: int = 0,
    coop_slots: int = 1,
) -> FlexHookCode:
    return render_flex_hook(
        plan,
        stage=stage,
        score_tensor="flash_disc_frg",
        coord_tensor="tLDcS[None, flash_ci, None, None]",
        m_tile_expr=f"flash_m_tile{stage}",
        kv_tile_expr="flash_kv",
        chunk_expr="flash_ci",
        chunk_cols=chunk_cols,
        indent=16,
        with_score=with_score,
        with_mask=with_mask,
        tail_mask=tail_mask,
        row_coord_expr="tLDcS[0][0]",
        device_function=_StubDeviceFunction(),  # pyrefly: ignore [bad-argument-type]
        coord_bounds=_BOUNDS if bounds else None,
        mask_form=mask_form,
        aux_smem_budget=aux_smem_budget,
        prefetch_regs=prefetch_regs,
        coalesce=coalesce,
        coop_slots=coop_slots,
    )


# ---------------------------------------------------------------------------
# DSL emulation: execute the rendered blocks on Python scalars.
# ---------------------------------------------------------------------------


class _Dtype:
    inf = math.inf

    def __init__(self, cast: Callable[[object], object]) -> None:
        self._cast = cast

    def __call__(self, value: object) -> object:
        return self._cast(value)


def _to_int(value: object) -> int:
    if isinstance(value, float):
        return int(value)
    return int(value)  # pyrefly: ignore [bad-argument-type]


_INT = _Dtype(_to_int)
_FLOAT = _Dtype(float)  # pyrefly: ignore [bad-argument-type]
_BOOL = _Dtype(bool)  # pyrefly: ignore [bad-argument-type]


class _Ptr:
    def __init__(self, storage: list[object], offset: int) -> None:
        self.storage = storage
        self.offset = offset

    def __add__(self, other: object) -> _Ptr:
        return _Ptr(self.storage, self.offset + int(other))  # pyrefly: ignore [bad-argument-type]

    def load(self) -> object:
        if self.offset < 0:
            raise IndexError("negative pointer offset")
        return self.storage[self.offset]


class _FakeTensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.iterator = _Ptr(tensor.flatten().tolist(), 0)
        self.layout = types.SimpleNamespace(stride=tuple(tensor.stride()))


class _ChunkCoords:
    def __init__(self, row: int, base: int) -> None:
        self.row = row
        self.base = base

    def __getitem__(self, index: int) -> tuple[int, int]:
        return (self.row, self.base + index)


class _Coords:
    """``tLDcS`` of one thread: ``[None, ci, None, None]`` chunk slices and ``[0]``."""

    def __init__(self, row: int, chunk_cols: int = _CHUNK_COLS) -> None:
        self.row = row
        self.chunk_cols = chunk_cols

    def __getitem__(self, key: object) -> object:
        if isinstance(key, tuple):
            return _ChunkCoords(self.row, key[1] * self.chunk_cols)
        if key == 0:
            return (self.row, 0)
        raise IndexError(key)


def _load_vec(ptr: _Ptr, width: int, dtype: _Dtype) -> list[object]:
    if ptr.offset < 0 or ptr.offset + width > len(ptr.storage):
        raise IndexError("vector load out of bounds")
    return [dtype(ptr.storage[ptr.offset + j]) for j in range(width)]


def _load_run(ptr: _Ptr, count: int, width: int, dtype: _Dtype) -> list[object]:
    values: list[object] = []
    for k in range(count):
        values.extend(_load_vec(ptr + width * k, width, dtype))
    return values


def _load_window(
    ptr: _Ptr, last_ptr: _Ptr, count: int, width: int, dtype: _Dtype
) -> list[object]:
    values: list[object] = []
    for k in range(count - 1):
        values.extend(_load_vec(ptr + width * k, width, dtype))
    values.extend(_load_vec(last_ptr, width, dtype))
    return values


def _coop_issue(
    row_ptr: _Ptr,
    col: object,
    slot: object,
    row_step: int,
    width: int,
    units: int,
    per_vector: int,
    dtype: _Dtype,
    stage_ptr: _Ptr,
    lane: object,
    pitch: int,
    limit: int | None = None,
) -> None:
    """The GPU copies every lane's share of the block into the slot; the
    single-thread emulation writes the owner row's values into the lane's
    staged row directly (the lane's request row minus its own row is the
    renderer's ``delta``, so its own row sits ``delta`` rows below its copy
    base), clamping each piece as the copy does.  Synchronous: the reads
    that follow the group's wait find the data."""
    assert row_ptr.offset >= 0
    assert 0 <= int(slot) < width // 16  # pyrefly: ignore [bad-argument-type]
    row = cast("int", coop_lane_map(int(lane), width)[0])  # pyrefly: ignore [bad-argument-type]
    owner = row_ptr + (int(lane) - row) * row_step  # pyrefly: ignore [bad-argument-type]
    lanes_per_row = width // 16
    unit_elems = width * per_vector // 16
    assert units * unit_elems <= pitch
    target = stage_ptr + int(lane) * pitch  # pyrefly: ignore [bad-argument-type]
    for u in range(units):
        for s in range(lanes_per_row):
            offset = int(col) + s * per_vector + u * unit_elems  # pyrefly: ignore [bad-argument-type]
            if limit is not None:
                offset = min(max(offset, 0), limit)
            values = _load_vec(owner + offset, per_vector, dtype)
            first = u * unit_elems + s * per_vector
            target.storage[
                target.offset + first : target.offset + first + per_vector
            ] = values


def _recast_ptr(ptr: _Ptr, dtype: _Dtype) -> _Ptr:
    """A fresh staging region per recast (one per table region per item):
    the single-thread emulation needs no aliasing between regions."""
    assert isinstance(ptr, _Ptr)
    return _Ptr([0] * (32 * 4096), 0)


def _mask_tail(frg: list[float], col_limit: int) -> None:
    for c in range(len(frg)):
        if c >= col_limit:
            frg[c] = -math.inf


def _namespace() -> dict[str, object]:
    return {
        "operator": operator,
        "cutlass": types.SimpleNamespace(
            Int32=_INT,
            Int64=_INT,
            Int8=_INT,
            Uint8=_INT,
            Int16=_INT,
            Float32=_FLOAT,
            BFloat16=_FLOAT,
            Float16=_FLOAT,
            Boolean=_BOOL,
            select_=lambda c, a, b: a if c else b,
            range_constexpr=range,
            min=min,
            max=max,
        ),
        "cute": types.SimpleNamespace(
            size=len,
            recast_ptr=_recast_ptr,
            math=types.SimpleNamespace(
                max=lambda a, b, propagate_nan=True: max(a, b),
                min=lambda a, b, propagate_nan=True: min(a, b),
                log=lambda x, fastmath=False: math.log(x),
                log2=lambda x, fastmath=False: math.log2(x),
                log1p=lambda x, fastmath=False: math.log1p(x),
                expm1=lambda x, fastmath=False: math.expm1(x),
                erf=lambda x, fastmath=False: math.erf(x),
                erfc=lambda x, fastmath=False: math.erfc(x),
                exp2=lambda x, fastmath=False: 2.0**x,
                exp=lambda x, fastmath=False: math.exp(x),
                rcp=lambda x, approx=False, ftz=False: 1.0 / x,
                div=lambda a, b, approx=False, ftz=False: a / b,
                fma=lambda a, b, c: a * b + c,
                tanh=lambda x, approx=False, fastmath=False: math.tanh(x),
                sqrt=lambda x, fastmath=False: math.sqrt(x),
                rsqrt=lambda x, fastmath=False: 1.0 / math.sqrt(x),
                abs=abs,
                floor=lambda x, fastmath=False: float(math.floor(x)),
                ceil=lambda x, fastmath=False: float(math.ceil(x)),
                trunc=lambda x, fastmath=False: float(math.trunc(x)),
                roundeven=lambda x, fastmath=False: float(round(x)),
            ),
            arch=types.SimpleNamespace(
                exp2=lambda x: 2.0**x,
                fmin=lambda a, b, nan=False: min(a, b),
                rcp_approx=lambda x: 1.0 / x,
                # The stream's copies are synchronous in the emulation.
                cp_async_wait_group=lambda n: None,
                cp_async_commit_group=lambda: None,
                sync_warp=lambda: None,
            ),
        ),
        "_cute_python_mod": operator.mod,
        "_cute_flex_tanh": lambda y, approx=False: math.tanh(y),
        # C fmod (dividend's sign); ints stay ints.
        "_cute_flex_fmod": lambda a, b: (
            int(math.fmod(a, b))
            if isinstance(a, int) and isinstance(b, int)
            else math.fmod(a, b)
        ),
        # torch.remainder on floats is Python's floor-mod.
        "_cute_flex_remainder": operator.mod,
        "_cute_flex_load_vec": _load_vec,
        "_cute_flex_load_run": _load_run,
        "_cute_flex_load_window": _load_window,
        "_cute_flex_coop_lane": coop_lane_map,
        "_cute_flex_coop_issue": _coop_issue,
        # The staging buffers: an address the stubs only carry along.
        "_flash_flex_coop_smem": _Ptr([], 0),
        "_cute_flex_lane_below": lambda limit, width: (
            (1 << max(0, min(int(limit), width))) - 1
        ),
        "_helion_flash_rt": types.SimpleNamespace(mask_r2p_sm100_rank1=_mask_tail),
    }


def _prefetch_fill(
    ns: dict[str, object], prefetch: object, depth: int, tile: int, init: bool
) -> None:
    """The emitter's ring fill for KV tile ``tile``: chunks ``0 .. depth - 1``
    into slots ``0 .. depth - 1``, chunk 0 again into slot ``depth``."""
    ns["flash_flex_pkv"] = tile
    for slot in range(depth + 1):
        ns["flash_flex_pc"] = slot if slot < depth else 0
        ns["flash_flex_ps"] = slot
        ns["flash_flex_pf_init"] = init and slot == 0
        exec(prefetch, ns)  # pyrefly: ignore[bad-argument-type]


def _emulate_tiles(
    code: FlexHookCode,
    *,
    score_tiles: Sequence[torch.Tensor],
    bh: int,
    m_tile: int,
    kv_tiles: Sequence[int],
    kv_tail_cols: int,
    aux_tensors: Sequence[torch.Tensor],
    scalar_values: Sequence[object],
    scale: float | None = None,
    rows: Sequence[int] = _ROWS,
    stage: int = 0,
    chunk_cols: int = _CHUNK_COLS,
    smem_tables: Sequence[int] = (),
    absent: Sequence[int] = (),
) -> list[torch.Tensor]:
    """Run the hook over a sequence of KV tiles per thread row with the
    emitter's prefetch protocol (the ring filled for the first tile before
    the walk, for the next tile after each tile's chunks; ``absent`` tiles
    skip their chunks like a status-0 step but still refill) and return the
    masked/transformed tiles.

    ``smem_tables`` are the aux indices the hook reads from the shared-memory
    copy (``_flash_aux{i}_smem``): the flattened table plus the zero sentinel
    the emitter's fill appends.
    """
    prelude = compile(textwrap.dedent(code.item_prelude), "<item_prelude>", "exec")
    prologue = compile(textwrap.dedent(code.tile_prologue), "<tile_prologue>", "exec")
    body = compile(textwrap.dedent(code.chunk_body), "<chunk_body>", "exec")
    prefetch = compile(textwrap.dedent(code.chunk_prefetch), "<prefetch>", "exec")
    issue = compile(textwrap.dedent(code.chunk_issue), "<issue>", "exec")
    consume = compile(textwrap.dedent(code.chunk_consume), "<consume>", "exec")
    chunks = 128 // chunk_cols
    depth = code.prefetch_depth
    slots = code.coop_slots

    def stream_fill(ns: dict[str, object], tile: int) -> None:
        """The emitter's stream fill: chunks ``0 .. slots - 1`` of ``tile``."""
        ns["flash_flex_pkv"] = tile
        for slot in range(slots):
            ns["flash_flex_pc"] = slot
            ns["flash_flex_ps"] = slot
            exec(issue, ns)  # pyrefly: ignore[bad-argument-type]

    results = [tile.clone().double() for tile in score_tiles]
    fake_aux = [_FakeTensor(t) for t in aux_tensors]
    fake_smem = {
        index: _Ptr([*aux_tensors[index].flatten().tolist(), 0], 0)
        for index in smem_tables
    }
    for row in rows:
        ns = _namespace()
        ns.update(
            {
                "flash_bh": bh,
                "_flash_heads": _HEADS,
                "_flash_kv_tail_cols": kv_tail_cols,
                f"flash_m_tile{stage}": m_tile,
                "tLDcS": _Coords(row, chunk_cols),
            }
        )
        for i, tensor in enumerate(fake_aux):
            ns[f"_flash_aux{i}"] = tensor
        for i, pointer in fake_smem.items():
            ns[f"_flash_aux{i}_smem"] = pointer
        for j, value in enumerate(scalar_values):
            ns[f"_flash_hs{j}"] = value
        if scale is not None:
            ns["_flash_scale"] = scale
        exec(prelude, ns)
        if depth:
            _prefetch_fill(ns, prefetch, depth, kv_tiles[0], True)
        if slots:
            stream_fill(ns, kv_tiles[0])
        for index, kv_tile in enumerate(kv_tiles):
            ns["flash_kv"] = kv_tile
            # The walk's peek: the next tile, 0 past the end (dead loads).
            next_tile = kv_tiles[index + 1] if index + 1 < len(kv_tiles) else 0
            ns["flash_kv_peek"] = next_tile
            exec(prologue, ns)
            if index not in absent:
                ns["flash_flex_pkv"] = kv_tile
                result = results[index]
                for ci in range(chunks):
                    # The emitter's wrap-around: the last iterations prefetch
                    # early chunks into the freed ring slot (dead code on the
                    # GPU, overwritten by the fill below).
                    if depth:
                        ns["flash_flex_pc"] = (ci + depth) % chunks
                        ns["flash_flex_ps"] = (ci + depth) % (depth + 1)
                        ns["flash_flex_pf_init"] = False
                        ns["flash_flex_pkv"] = kv_tile
                        exec(prefetch, ns)
                    ns["flash_ci"] = ci
                    if slots:
                        # The stream: the chunk's rows from its slot, then
                        # chunk ``ci + slots`` (of the next tile past the
                        # chunk count) into the freed slot.
                        exec(consume, ns)
                        ns["flash_flex_pc"] = (ci + slots) % chunks
                        ns["flash_flex_ps"] = ci % slots
                        ns["flash_flex_pkv"] = (
                            kv_tile if ci + slots < chunks else next_tile
                        )
                        exec(issue, ns)
                    frg = result[row, ci * chunk_cols : (ci + 1) * chunk_cols].tolist()
                    ns["flash_disc_frg"] = frg
                    exec(body, ns)
                    result[row, ci * chunk_cols : (ci + 1) * chunk_cols] = torch.tensor(
                        frg, dtype=torch.float64
                    )
            elif slots:
                # An absent step issued nothing for the next tile: the fill.
                stream_fill(ns, next_tile)
            if depth:
                _prefetch_fill(ns, prefetch, depth, next_tile, False)
    return [result.float() for result in results]


def _emulate(
    code: FlexHookCode,
    *,
    score_tile: torch.Tensor,
    bh: int,
    m_tile: int,
    kv_tile: int,
    kv_tail_cols: int,
    aux_tensors: Sequence[torch.Tensor],
    scalar_values: Sequence[object],
    scale: float | None = None,
    rows: Sequence[int] = _ROWS,
    stage: int = 0,
    chunk_cols: int = _CHUNK_COLS,
    smem_tables: Sequence[int] = (),
) -> torch.Tensor:
    """One KV tile of ``_emulate_tiles``."""
    return _emulate_tiles(
        code,
        score_tiles=[score_tile],
        bh=bh,
        m_tile=m_tile,
        kv_tiles=[kv_tile],
        kv_tail_cols=kv_tail_cols,
        aux_tensors=aux_tensors,
        scalar_values=scalar_values,
        scale=scale,
        rows=rows,
        stage=stage,
        chunk_cols=chunk_cols,
        smem_tables=smem_tables,
    )[0]


# ---------------------------------------------------------------------------
# Torch reference of the same region functions.
# ---------------------------------------------------------------------------


def _ref_load(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    """``hl.load`` semantics on real tensors: out-of-range coordinates read 0."""
    tensors = [i for i in index if isinstance(i, torch.Tensor)]
    shape = torch.broadcast_shapes(*(t.shape for t in tensors)) if tensors else ()
    ok = torch.ones(shape, dtype=torch.bool)
    coords: list[torch.Tensor] = []
    for dim, entry in enumerate(index):
        size = tensor.shape[dim]
        if isinstance(entry, torch.Tensor):
            idx = entry.to(torch.int64).expand(shape)
        else:
            idx = torch.full(shape, int(entry), dtype=torch.int64)  # pyrefly: ignore [bad-argument-type]
        ok = ok & (idx >= 0) & (idx < size)
        coords.append(idx.clamp(0, size - 1))
    values = tensor[tuple(coords)]
    return torch.where(ok, values, torch.zeros((), dtype=tensor.dtype))


def _reference(
    score_fn: Callable[..., torch.Tensor] | None,
    mask_fn: Callable[..., torch.Tensor] | None,
    *,
    score_tile: torch.Tensor,
    bh: int,
    m_tile: int,
    kv_tile: int,
    kv_tail_cols: int | None,
    aux_tensors: Sequence[torch.Tensor],
    scalar_values: Sequence[object],
) -> torch.Tensor:
    b = torch.tensor([[bh // _HEADS]], dtype=torch.int32)
    h = torch.tensor([[bh % _HEADS]], dtype=torch.int32)
    q_idx = (m_tile * 128 + torch.arange(128, dtype=torch.int32))[:, None]
    kv_idx = (kv_tile * 128 + torch.arange(128, dtype=torch.int32))[None, :]
    extras = (*aux_tensors, *scalar_values)
    with patch.object(hl, "load", _ref_load):
        out = score_tile
        if score_fn is not None:
            out = score_fn(score_tile, b, h, q_idx, kv_idx, *extras)
        out = out.expand(128, 128).clone()
        if mask_fn is not None:
            keep = mask_fn(score_tile, b, h, q_idx, kv_idx, *extras).expand(128, 128)
            out = torch.where(keep, out, torch.tensor(-math.inf))
    if kv_tail_cols is not None:
        out[:, kv_tail_cols:] = -math.inf
    return out


# ---------------------------------------------------------------------------
# Region bodies (attention-gym MOD_* shapes, FlexAttention signatures plus the
# captured tables and host scalars as trailing arguments).
# ---------------------------------------------------------------------------


def mask_dilated_sw(score, b, h, q_idx, kv_idx):
    diff = torch.abs(q_idx - kv_idx)
    return (diff <= 512) & ((diff % 2) == 0)


def mask_natten2d(score, b, h, q_idx, kv_idx):
    canvas_w, canvas_h, half = 64, 32, 6
    q_x = q_idx // canvas_w
    q_y = q_idx % canvas_w
    kv_x = kv_idx // canvas_w
    kv_y = kv_idx % canvas_w
    center_x = q_x.clamp(half, canvas_w - 1 - half)
    center_y = q_y.clamp(half, canvas_h - 1 - half)
    hori = (center_x - half <= kv_x) & (kv_x <= center_x + half)
    vert = (center_y - half <= kv_y) & (kv_y <= center_y + half)
    return hori & vert


def mask_sta2d(score, b, h, q_idx, kv_idx):
    # canvas_tile_h == 2 gives ``clamp(x, 1, 0)``: torch returns the max.
    tile_numel, canvas_tile_w, canvas_tile_h, left, right = 256, 4, 2, 1, 0
    q_tile = q_idx // tile_numel
    kv_tile = kv_idx // tile_numel
    q_tile_h = q_tile // canvas_tile_w
    q_tile_w = q_tile % canvas_tile_w
    kv_tile_h = kv_tile // canvas_tile_w
    kv_tile_w = kv_tile % canvas_tile_w
    center_h = q_tile_h.clamp(left, canvas_tile_h - 1 - right)
    center_w = q_tile_w.clamp(left, canvas_tile_w - 1 - right)
    h_mask = (kv_tile_h >= center_h - left) & (kv_tile_h <= center_h + right)
    w_mask = (kv_tile_w >= center_w - left) & (kv_tile_w <= center_w + right)
    return h_mask & w_mask


def mask_sta3d(score, b, h, q_idx, kv_idx):
    tile_numel, hw_tiles, w_tiles, t_tiles, h_tiles = 128, 16, 4, 2, 4
    left_t, right_t, left_h, right_h, left_w, right_w = 1, 0, 1, 0, 1, 0
    q_tile = q_idx // tile_numel
    kv_tile = kv_idx // tile_numel
    q_t = q_tile // hw_tiles
    q_h = (q_tile % hw_tiles) // w_tiles
    q_w = q_tile % w_tiles
    kv_t = kv_tile // hw_tiles
    kv_h = (kv_tile % hw_tiles) // w_tiles
    kv_w = kv_tile % w_tiles
    c_t = q_t.clamp(left_t, t_tiles - 1 - right_t)
    c_h = q_h.clamp(left_h, h_tiles - 1 - right_h)
    c_w = q_w.clamp(left_w, w_tiles - 1 - right_w)
    t_mask = (kv_t >= c_t - left_t) & (kv_t <= c_t + right_t)
    h_mask = (kv_h >= c_h - left_h) & (kv_h <= c_h + right_h)
    w_mask = (kv_w >= c_w - left_w) & (kv_w <= c_w + right_w)
    return t_mask & h_mask & w_mask


def mask_batchify(score, b, h, q_idx, kv_idx):
    same_batch = (q_idx // 512) == (kv_idx // 512)
    inner = (q_idx % 512) >= (kv_idx % 512)
    return same_batch & inner


def mask_block_diffusion(score, b, h, q_idx, kv_idx):
    seq_len, block = 512, 128
    q_noised = q_idx < seq_len
    kv_noised = kv_idx < seq_len
    q_block = (q_idx % seq_len) // block
    kv_block = (kv_idx % seq_len) // block
    block_diagonal = (q_block == kv_block) & (q_noised == kv_noised)
    offset_block_causal = (q_block > kv_block) & q_noised & ~kv_noised
    block_causal = (q_block >= kv_block) & ~q_noised & ~kv_noised
    return block_diagonal | offset_block_causal | block_causal


def mask_svg_spatial(score, b, h, q_idx, kv_idx):
    first_row = q_idx < 128
    first_col = kv_idx < 128
    band = torch.abs(q_idx - kv_idx) < 256
    return first_col | first_row | band


def mask_svg_temporal(score, b, h, q_idx, kv_idx):
    # ``(q - 128) % 512`` has a negative dividend on the first 128 rows.
    first_row = q_idx < 128
    first_col = kv_idx < 128
    q_tok = (q_idx - 128) % 512
    kv_tok = (kv_idx - 128) % 512
    band = torch.abs(q_tok - kv_tok) < 64
    return first_col | first_row | band


def mask_jetspec_training(score, b, h, q_idx, kv_idx):
    prefix, block = 640, 16
    q_block = q_idx // block
    q_off = q_idx % block
    kv_rel = kv_idx - prefix
    kv_block = kv_rel // block
    kv_off = kv_rel % block
    same = (kv_idx >= prefix) & (q_block == kv_block) & (kv_off <= q_off)
    return (kv_idx < prefix) | same


def mask_shared_prefix(score, b, h, q_idx, kv_idx, doc_start, prefix_start, prefix_end):
    same_doc_causal = (kv_idx >= hl.load(doc_start, [q_idx])) & (q_idx >= kv_idx)
    prefix = (kv_idx >= hl.load(prefix_start, [q_idx])) & (
        kv_idx < hl.load(prefix_end, [q_idx])
    )
    return same_doc_causal | prefix


def mask_global_sw(score, b, h, q_idx, kv_idx, is_global):
    window = torch.abs(q_idx - kv_idx) <= 128
    return window | hl.load(is_global, [q_idx]) | hl.load(is_global, [kv_idx])


def mask_flamingo(score, b, h, q_idx, kv_idx, start_tok, end_tok):
    return (q_idx >= hl.load(start_tok, [kv_idx])) & (
        q_idx < hl.load(end_tok, [kv_idx])
    )


def mask_jetspec_tree(score, b, h, q_idx, kv_idx, ancestor):
    prefix, num_nodes = 640, 64
    tree_idx = (kv_idx - prefix).clamp(0, num_nodes - 1)
    tree_visible = (kv_idx >= prefix) & (kv_idx < prefix + num_nodes)
    ancestor_bit = hl.load(ancestor, [q_idx, tree_idx])
    return (kv_idx < prefix) | (tree_visible & ancestor_bit)


def mask_document_table(score, b, h, q_idx, kv_idx, doc_start):
    return (kv_idx >= hl.load(doc_start, [b, q_idx])) & (q_idx >= kv_idx)


def mask_host_scalar(score, b, h, q_idx, kv_idx, prefix):
    return (kv_idx < prefix) | (q_idx >= kv_idx)


def mask_python_semantics(score, b, h, q_idx, kv_idx):
    minus_five = kv_idx * 0 - 5
    return (
        ((minus_five % 512) == 507)
        & ((minus_five // 16) == -1)
        & (q_idx.clamp(1, 0) == 0)
    )


def mask_out_of_range_table(score, b, h, q_idx, kv_idx, small):
    # ``small`` has 16 entries; almost every coordinate is out of range.
    return hl.load(small, [kv_idx]) | (hl.load(small, [q_idx - 3]) & (kv_idx < 4))


def mask_causal(score, b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


# Both regions of a plan share one placeholder list (spec §1), so a mask paired
# with a table-reading score_mod takes the tables / scalars as well.
def mask_causal_sandwich(score, b, h, q_idx, kv_idx, rel_bias, head_scale):
    return q_idx >= kv_idx


def mask_causal_scaled(score, b, h, q_idx, kv_idx, bias_scale, scale):
    return q_idx >= kv_idx


def score_sandwich(score, b, h, q_idx, kv_idx, rel_bias, head_scale):
    return score + hl.load(rel_bias, [q_idx - kv_idx + 1023]) * hl.load(head_scale, [h])


def score_sigmoid_act(score, b, h, q_idx, kv_idx):
    return torch.log(torch.sigmoid(score) + 1.0)


def score_softcap(score, b, h, q_idx, kv_idx):
    return 20.0 * torch.tanh(score / 20.0)


def score_alibi_inline(score, b, h, q_idx, kv_idx):
    slope = torch.exp2(-((h + 1) * 8.0 / _HEADS))
    return score + (kv_idx - q_idx) * slope


def score_rel_bias(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)


def score_graphormer_spatial(score, b, h, q_idx, kv_idx, distances, spatial_bias):
    dist = hl.load(distances, [b, q_idx, kv_idx])
    return score + hl.load(spatial_bias, [h, dist])


def score_graphormer_edge(
    score, b, h, q_idx, kv_idx, path_lengths, path_types, edge_bias
):
    length = hl.load(path_lengths, [b, q_idx, kv_idx])
    total = torch.zeros_like(score)
    et0 = hl.load(path_types, [b, q_idx, kv_idx, 0])
    c0 = hl.load(edge_bias, [h, et0 + 0])
    total = total + torch.where(length > 0, c0, 0.0)
    et1 = hl.load(path_types, [b, q_idx, kv_idx, 1])
    c1 = hl.load(edge_bias, [h, et1 + 8])
    total = total + torch.where(length > 1, c1, 0.0)
    et2 = hl.load(path_types, [b, q_idx, kv_idx, 2])
    c2 = hl.load(edge_bias, [h, et2 + 16])
    total = total + torch.where(length > 2, c2, 0.0)
    return score + total / torch.clamp(length, min=1).to(torch.float32)


def score_strided_runs(score, b, h, q_idx, kv_idx, narrow, wide, flags):
    # ``narrow[B, 1024, 64, 3]`` int32: out of range past column 64 (the run's
    # group guard); ``wide[B, 1024, 1024, 3]`` bf16 (eight-lane groups: the
    # int32 run is loaded in six vectors of the common group); ``flags[B,
    # 1024, 1024, 2]`` bool (a one-vector run of bytes).
    a = hl.load(narrow, [b, q_idx, kv_idx, 1]).to(torch.float32)
    c = hl.load(narrow, [b, q_idx, kv_idx, 2]).to(torch.float32)
    w = hl.load(wide, [b, q_idx, kv_idx, 0]).to(torch.float32)
    keep = hl.load(flags, [b, q_idx, kv_idx, 1])
    return torch.where(keep, score + a * 0.5 + c, score * 0.25 + w)


def score_two_gathers(score, b, h, q_idx, kv_idx, idx, small, big):
    # Two data-dependent gathers: ``small`` (16 x 7) and ``big`` (16 x 1024)
    # are staging candidates in plan order.
    i = hl.load(idx, [b, q_idx, kv_idx])
    return score + hl.load(small, [h, i]) + hl.load(big, [h, i * 37])


def score_runtime_scale(score, b, h, q_idx, kv_idx, bias_scale, scale):
    return score * scale + (q_idx - kv_idx) * bias_scale


def score_bf16_table(score, b, h, q_idx, kv_idx, bias):
    return score + hl.load(bias, [b, q_idx, kv_idx])


def score_extra_ops(score, b, h, q_idx, kv_idx):
    # The rest of the matcher's admitted fp32 op set.
    a = torch.pow(score, 2) * 0.01 + torch.pow(torch.abs(score) + 1.0, 0.5)
    a = a + torch.sign(score) + torch.floor(score) + torch.ceil(score)
    a = a + torch.trunc(score) + torch.round(score * 0.5)
    a = a + torch.reciprocal(torch.abs(score) + 10.0)
    a = a + torch.log1p(torch.abs(score)) + torch.log2(torch.abs(score) + 1.0)
    a = a + torch.sqrt(torch.abs(score)) + torch.rsqrt(torch.abs(score) + 1.0)
    a = a + torch.fmod(score, 3.0) + torch.remainder(score, -2.5)
    a = a + torch.minimum(score, 0.5 * kv_idx) + torch.maximum(score, -0.25 * q_idx)
    return a + torch.exp(-torch.abs(score)) + torch.exp2(-torch.abs(score)) + score / 4


def mask_int_fmod(score, b, h, q_idx, kv_idx):
    # ``torch.fmod`` keeps the dividend's sign on negative differences.
    d = q_idx - kv_idx
    return torch.logical_and(
        torch.fmod(d, 3) != 2, torch.logical_not(torch.fmod(d, 5) == -1)
    )


# ---------------------------------------------------------------------------
# Table generators (seeded).
# ---------------------------------------------------------------------------


def _int_table(
    gen: torch.Generator, shape: Sequence[int], low: int, high: int
) -> torch.Tensor:
    return torch.randint(low, high, tuple(shape), generator=gen, dtype=torch.int32)


def _shared_prefix_tables(gen: torch.Generator) -> list[torch.Tensor]:
    S, doc_len = 1024, 256
    doc_start = (torch.arange(S) // doc_len * doc_len).to(torch.int32)
    doc = torch.arange(S) // doc_len
    prefix_start = torch.where(doc % 2 == 1, (doc - 1) * doc_len, 0).to(torch.int32)
    prefix_end = torch.where(doc % 2 == 1, doc * doc_len, 0).to(torch.int32)
    return [doc_start, prefix_start, prefix_end]


def _global_tables(gen: torch.Generator) -> list[torch.Tensor]:
    S = 1024
    is_global = (torch.arange(S) < 8) | (torch.arange(S) % 256 == 0)
    return [is_global]


def _flamingo_tables(gen: torch.Generator) -> list[torch.Tensor]:
    N = 1024
    image = torch.arange(N) // 256
    start = (image * 512).to(torch.int32)
    end = (image * 512 + 384).to(torch.int32)
    return [start, end]


def _jetspec_tables(gen: torch.Generator) -> list[torch.Tensor]:
    return [torch.rand(64, 64, generator=gen) < 0.3]


def _document_tables(gen: torch.Generator) -> list[torch.Tensor]:
    S = 1024
    doc_start = (torch.arange(S) // 300 * 300).to(torch.int32)
    return [torch.stack([doc_start, (doc_start // 2 * 2)]).contiguous()]


def _small_table(gen: torch.Generator) -> list[torch.Tensor]:
    return [torch.rand(16, generator=gen) < 0.5]


def _sandwich_tables(gen: torch.Generator) -> list[torch.Tensor]:
    return [
        torch.randn(2048, generator=gen),
        torch.randn(_HEADS, generator=gen),
    ]


def _graphormer_spatial_tables(gen: torch.Generator) -> list[torch.Tensor]:
    return [
        _int_table(gen, (_BATCH, 1024, 1024), 0, 7),
        torch.randn(_HEADS, 8, generator=gen),
    ]


def _graphormer_edge_tables(gen: torch.Generator) -> list[torch.Tensor]:
    return [
        _int_table(gen, (_BATCH, 1024, 1024), 0, 4),
        _int_table(gen, (_BATCH, 1024, 1024, 3), 0, 8),
        torch.randn(_HEADS, 24, generator=gen),
    ]


def _strided_run_tables(gen: torch.Generator) -> list[torch.Tensor]:
    return [
        _int_table(gen, (_BATCH, 1024, 64, 3), -4, 4),
        torch.randn(_BATCH, 1024, 1024, 3, generator=gen).to(torch.bfloat16),
        torch.rand(_BATCH, 1024, 1024, 2, generator=gen) < 0.5,
    ]


def _two_gather_tables(gen: torch.Generator) -> list[torch.Tensor]:
    return [
        _int_table(gen, (_BATCH, 1024, 1024), -1, 9),
        torch.randn(_HEADS, 7, generator=gen),
        torch.randn(_HEADS, 1024, generator=gen),
    ]


def _bf16_bias_tables(gen: torch.Generator) -> list[torch.Tensor]:
    return [torch.randn(_BATCH, 1024, 1024, generator=gen).to(torch.bfloat16)]


def _specs(
    tables: Sequence[torch.Tensor], names: Sequence[str]
) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
    return [
        (name, t.dtype, tuple(t.shape)) for name, t in zip(names, tables, strict=True)
    ]


@dataclasses.dataclass
class _Case:
    name: str
    score: Callable[..., torch.Tensor] | None = None
    mask: Callable[..., torch.Tensor] | None = None
    tables: Callable[[torch.Generator], list[torch.Tensor]] | None = None
    table_names: Sequence[str] = ()
    scalars: Sequence[tuple[str, type, object]] = ()
    scale_arg: str | None = None
    # (flash_bh, m_tile, kv_tile, kv_tail_cols or None for a full tile)
    geometries: Sequence[tuple[int, int, int, int | None]] = (
        (0, 0, 0, None),
        (5, 3, 2, None),
        (7, 1, 7, 104),
    )
    fast_math: bool = False


_CASES = [
    _Case("dilated_sw", mask=mask_dilated_sw),
    _Case("natten2d", mask=mask_natten2d),
    _Case("sta2d", mask=mask_sta2d),
    _Case("sta3d", mask=mask_sta3d),
    _Case("batchify", mask=mask_batchify),
    _Case("block_diffusion", mask=mask_block_diffusion),
    _Case("svg_spatial", mask=mask_svg_spatial),
    _Case("svg_temporal", mask=mask_svg_temporal),
    _Case("jetspec_training", mask=mask_jetspec_training),
    _Case(
        "shared_prefix",
        mask=mask_shared_prefix,
        tables=_shared_prefix_tables,
        table_names=("doc_start", "prefix_start", "prefix_end"),
    ),
    _Case(
        "global_sw",
        mask=mask_global_sw,
        tables=_global_tables,
        table_names=("is_global",),
    ),
    _Case(
        "flamingo",
        mask=mask_flamingo,
        tables=_flamingo_tables,
        table_names=("start_tok", "end_tok"),
    ),
    _Case(
        "jetspec_tree",
        mask=mask_jetspec_tree,
        tables=_jetspec_tables,
        table_names=("ancestor",),
    ),
    _Case(
        "document_table",
        mask=mask_document_table,
        tables=_document_tables,
        table_names=("doc_start",),
    ),
    _Case(
        "host_scalar_mask",
        mask=mask_host_scalar,
        scalars=(("prefix", int, 300),),
    ),
    _Case("python_semantics", mask=mask_python_semantics),
    _Case(
        "out_of_range_table",
        mask=mask_out_of_range_table,
        tables=_small_table,
        table_names=("small",),
    ),
    _Case(
        "sandwich",
        score=score_sandwich,
        mask=mask_causal_sandwich,
        tables=_sandwich_tables,
        table_names=("rel_bias", "head_scale"),
    ),
    _Case("sigmoid_act", score=score_sigmoid_act),
    _Case("sigmoid_act_fast_math", score=score_sigmoid_act, fast_math=True),
    _Case("softcap", score=score_softcap, mask=mask_causal),
    _Case("softcap_fast_math", score=score_softcap, fast_math=True),
    _Case("alibi_inline", score=score_alibi_inline, mask=mask_causal),
    _Case("rel_bias", score=score_rel_bias, mask=mask_dilated_sw),
    _Case(
        "graphormer_spatial",
        score=score_graphormer_spatial,
        tables=_graphormer_spatial_tables,
        table_names=("distances", "spatial_bias"),
    ),
    _Case(
        "graphormer_edge",
        score=score_graphormer_edge,
        tables=_graphormer_edge_tables,
        table_names=("path_lengths", "path_types", "edge_bias"),
    ),
    _Case(
        "runtime_scale",
        score=score_runtime_scale,
        mask=mask_causal_scaled,
        scalars=(("bias_scale", float, 0.125), ("scale", float, 0.25)),
        scale_arg="scale",
    ),
    _Case(
        "bf16_table",
        score=score_bf16_table,
        tables=_bf16_bias_tables,
        table_names=("bias",),
    ),
    _Case(
        "strided_runs",
        score=score_strided_runs,
        tables=_strided_run_tables,
        table_names=("narrow", "wide", "flags"),
    ),
    _Case(
        "two_gathers",
        score=score_two_gathers,
        tables=_two_gather_tables,
        table_names=("idx", "small", "big"),
    ),
    _Case("extra_ops", score=score_extra_ops),
    _Case("int_fmod", mask=mask_int_fmod),
]


def _case_inputs(case: _Case) -> tuple[list[torch.Tensor], list[object]]:
    """Real tables and scalar values (built OUTSIDE the compile environment,
    whose fake-tensor mode would otherwise capture them)."""
    gen = torch.Generator().manual_seed(0)
    tables = case.tables(gen) if case.tables is not None else []
    return tables, [value for _name, _kind, value in case.scalars]


def _build(case: _Case, tables: Sequence[torch.Tensor]) -> AttentionScorePlan:
    """Trace the case's regions into a plan (inside the compile environment)."""
    aux_specs = _specs(tables, case.table_names)
    scalar_specs = [(name, kind) for name, kind, _value in case.scalars]
    score_region = mask_region = None
    aux: tuple[FlexAuxBuffer, ...] = ()
    names: tuple[str, ...] = ()
    if case.score is not None:
        score_region, aux, names = build_flex_region(
            case.score, aux_specs=aux_specs, host_scalar_specs=scalar_specs
        )
    if case.mask is not None:
        mask_region, mask_aux, names = build_flex_region(
            case.mask, aux_specs=aux_specs, host_scalar_specs=scalar_specs
        )
        if case.score is None:
            aux = mask_aux
    return _plan(score_region, mask_region, aux, names, scale_arg=case.scale_arg)


def _render_case(
    case: _Case,
    *,
    with_tail: bool,
    bounds: bool = False,
    chunk_cols: int = _CHUNK_COLS,
    mask_form: str = "bitmask",
    aux_smem_budget: int = 0,
    prefetch_regs: int = 0,
    coalesce: int = 0,
    coop_slots: int = 1,
    stage: int = 0,
) -> tuple[FlexHookCode, AttentionScorePlan, list[torch.Tensor], list[object]]:
    tables, scalar_values = _case_inputs(case)
    with _environment(fast_math=case.fast_math):
        plan = _build(case, tables)
        code = _render(
            plan,
            with_score=case.score is not None,
            with_mask=case.mask is not None,
            tail_mask=with_tail,
            stage=stage,
            bounds=bounds,
            chunk_cols=chunk_cols,
            mask_form=mask_form,
            aux_smem_budget=aux_smem_budget,
            prefetch_regs=prefetch_regs,
            coalesce=coalesce,
            coop_slots=coop_slots,
        )
    return code, plan, tables, scalar_values


@skipUnlessCuteAvailable("renders CuTe DSL source through the cute backend")
class TestCuteFlexScoreMod(TestCase):
    def _check_case(
        self,
        case: _Case,
        *,
        bounds: bool = False,
        chunk_cols: int = _CHUNK_COLS,
        mask_form: str = "bitmask",
        aux_smem_budget: int = 0,
        prefetch_regs: int = 0,
        coalesce: int = 0,
        coop_slots: int = 1,
    ) -> None:
        gen = torch.Generator().manual_seed(1)
        codes = {
            with_tail: _render_case(
                case,
                with_tail=with_tail,
                bounds=bounds,
                chunk_cols=chunk_cols,
                mask_form=mask_form,
                aux_smem_budget=aux_smem_budget,
                prefetch_regs=prefetch_regs,
                coalesce=coalesce,
                coop_slots=coop_slots,
            )
            for with_tail in (False, True)
        }
        for bh, m_tile, kv_tile, tail in case.geometries:
            code, plan, tables, scalar_values = codes[tail is not None]
            staged = [
                entry.aux_index for entry in flex_aux_smem_layout(plan, aux_smem_budget)
            ]
            score_tile = torch.randn(128, 128, generator=gen) * 4.0
            scale_value = None
            if case.scale_arg is not None:
                index = list(plan.host_scalars).index(case.scale_arg)
                scale_value = float(scalar_values[index])  # pyrefly: ignore [bad-argument-type]
            result = _emulate(
                code,
                score_tile=score_tile,
                bh=bh,
                m_tile=m_tile,
                kv_tile=kv_tile,
                kv_tail_cols=tail if tail is not None else 128,
                aux_tensors=tables,
                scalar_values=scalar_values,
                scale=scale_value,
                chunk_cols=chunk_cols,
                smem_tables=staged,
            )
            expected = _reference(
                case.score,
                case.mask,
                score_tile=score_tile,
                bh=bh,
                m_tile=m_tile,
                kv_tile=kv_tile,
                kv_tail_cols=tail,
                aux_tensors=tables,
                scalar_values=scalar_values,
            )
            rows = list(_ROWS)
            label = f"{case.name} bh={bh} m_tile={m_tile} kv_tile={kv_tile} tail={tail}"
            torch.testing.assert_close(
                result[rows], expected[rows], rtol=1e-5, atol=1e-5, msg=label
            )

    def test_emulation_matrix(self) -> None:
        for case in _CASES:
            with self.subTest(case=case.name):
                self._check_case(case)

    def test_emulation_matrix_with_coordinate_bounds(self) -> None:
        """With the coordinate bounds the proven loads drop their guards and
        the diagonal gathers take the window form; same values."""
        for case in _CASES:
            with self.subTest(case=case.name):
                self._check_case(case, bounds=True)

    def test_emulation_matrix_element_form(self) -> None:
        """The per-element mask form (the ``"element"`` knob value)."""
        for case in _CASES:
            if case.mask is None:
                continue
            with self.subTest(case=case.name):
                self._check_case(case, mask_form="element")

    def test_emulation_matrix_sixteen_column_chunks(self) -> None:
        """``s_load_rep == 16``: 16-bit keep words, 16-column spans."""
        for case in _CASES:
            if case.mask is None:
                continue
            for mask_form in ("bitmask", "element"):
                with self.subTest(case=case.name, mask_form=mask_form):
                    self._check_case(case, chunk_cols=16, mask_form=mask_form)

    def test_emulation_matrix_with_shared_memory_staging(self) -> None:
        """Every candidate table staged (budget 16384): the gathers read the
        copy through the sentinel select; same values, with and without the
        coordinate bounds and at both chunk widths."""
        for case in _CASES:
            if case.tables is None:
                continue
            for bounds, chunk_cols in ((False, 32), (True, 32), (True, 16)):
                with self.subTest(case=case.name, bounds=bounds, chunk=chunk_cols):
                    self._check_case(
                        case,
                        bounds=bounds,
                        chunk_cols=chunk_cols,
                        aux_smem_budget=16384,
                    )

    def test_emulation_matrix_with_next_chunk_prefetch(self) -> None:
        """Every prefetchable load issued one chunk ahead (budget 128) and a
        partial budget (32): the chunk bodies read the parity slots the
        prefetch block filled; same values at both chunk widths, with and
        without the bounds and with the staged copies."""
        for case in _CASES:
            if case.tables is None:
                continue
            for bounds, chunk_cols, budget, smem in (
                (True, 32, 128, 0),
                (True, 16, 128, 16384),
                (False, 32, 32, 0),
                (True, 32, 32, 0),
                (True, 16, 64, 0),
                (False, 16, 32, 4096),
            ):
                with self.subTest(
                    case=case.name, bounds=bounds, chunk=chunk_cols, budget=budget
                ):
                    self._check_case(
                        case,
                        bounds=bounds,
                        chunk_cols=chunk_cols,
                        aux_smem_budget=smem,
                        prefetch_regs=budget,
                    )

    def test_prefetch_ring_carries_across_kv_tiles(self) -> None:
        """The emitter's cross-tile protocol: the ring filled for the first
        tile before the walk and for the next tile after each tile's chunks
        (an absent step skips its chunks but refills); every tile of a
        sequence reads its own chunks at every depth, both chunk widths,
        with and without the staged copies."""
        gen = torch.Generator().manual_seed(7)
        tiles = [2, 5, 1, 7, 0]
        for name in (
            "graphormer_edge",
            "graphormer_spatial",
            "sandwich",
            "strided_runs",
            "flamingo",
            "two_gathers",
        ):
            case = next(c for c in _CASES if c.name == name)
            for chunk_cols, budget, smem in (
                (16, 128, 0),
                (32, 128, 16384),
                (16, 32, 0),
                (32, 64, 0),
            ):
                code, plan, tables, values = _render_case(
                    case,
                    with_tail=False,
                    bounds=True,
                    chunk_cols=chunk_cols,
                    aux_smem_budget=smem,
                    prefetch_regs=budget,
                )
                staged = [e.aux_index for e in flex_aux_smem_layout(plan, smem)]
                scores = [torch.randn(128, 128, generator=gen) * 4.0 for _ in tiles]
                outs = _emulate_tiles(
                    code,
                    score_tiles=scores,
                    bh=5,
                    m_tile=1,
                    kv_tiles=tiles,
                    kv_tail_cols=128,
                    aux_tensors=tables,
                    scalar_values=values,
                    chunk_cols=chunk_cols,
                    smem_tables=staged,
                    absent=(2,),
                )
                rows = list(_ROWS)
                for index, (kv_tile, score, out) in enumerate(
                    zip(tiles, scores, outs, strict=True)
                ):
                    if index == 2:
                        continue
                    expected = _reference(
                        case.score,
                        case.mask,
                        score_tile=score,
                        bh=5,
                        m_tile=1,
                        kv_tile=kv_tile,
                        kv_tail_cols=None,
                        aux_tensors=tables,
                        scalar_values=values,
                    )
                    torch.testing.assert_close(
                        out[rows],
                        expected[rows],
                        rtol=1e-5,
                        atol=1e-5,
                        msg=f"{name} chunk={chunk_cols} budget={budget} depth={code.prefetch_depth} tile#{index}",
                    )

    def test_emulation_matrix_with_cooperative_loads(self) -> None:
        """The warp-cooperative block form (``coalesce`` 128 / 64 / 32): the
        runs with a proven affine row step are fetched through the
        cooperative stubs and exchanged into the owner row; same values at
        both chunk widths, with the ring and the staged copies, and on the
        unproven lane dimension (strided_runs' narrow table: the clamped
        fetch under the group guard).  Without the coordinate bounds no
        row step is proven and the render is byte-identical to the
        per-lane form; so is every mask-only render."""
        for case in _CASES:
            if case.tables is None:
                continue
            for chunk_cols, coalesce, budget, smem, slots in (
                (32, 128, 0, 0, 1),
                (16, 128, 0, 0, 2),
                (32, 64, 128, 4096, 2),
                (16, 64, 64, 0, 4),
                (32, 32, 32, 16384, 4),
                (16, 32, 128, 4096, 1),
            ):
                with self.subTest(
                    case=case.name,
                    chunk=chunk_cols,
                    coalesce=coalesce,
                    budget=budget,
                    slots=slots,
                ):
                    self._check_case(
                        case,
                        bounds=True,
                        chunk_cols=chunk_cols,
                        aux_smem_budget=smem,
                        prefetch_regs=budget,
                        coalesce=coalesce,
                        coop_slots=slots,
                    )
            plain, _plan_, _tables, _values = _render_case(case, with_tail=False)
            same, _plan_, _tables, _values = _render_case(
                case, with_tail=False, coalesce=128
            )
            self.assertEqual(plain, same, case.name)
            self.assertNotIn("coop", "".join(plain[:4]))
        for case in _CASES:
            if case.score is not None:
                continue
            plain, _plan_, _tables, _values = _render_case(
                case, with_tail=True, bounds=True
            )
            same, _plan_, _tables, _values = _render_case(
                case, with_tail=True, bounds=True, coalesce=128
            )
            self.assertEqual(plain, same, case.name)

    def test_cooperative_stream_carries_across_kv_tiles(self) -> None:
        """The stream's issue / consume protocol over a tile sequence with
        an absent step (the emitter's splice: the first tile's chunks
        before the walk, chunk ``ci + slots`` of the current or the peeked
        tile after each consume, the refill at the absent step): every
        tile reads its own block at every width and ring depth, with the
        register prefetch of the other loads alongside.  Streamed runs
        leave the register ring (no ``_pf`` slot of theirs)."""
        gen = torch.Generator().manual_seed(11)
        tiles = [2, 5, 1, 7, 0]
        for name in ("graphormer_edge", "graphormer_spatial", "strided_runs"):
            case = next(c for c in _CASES if c.name == name)
            for chunk_cols, budget, coalesce, slots in (
                (16, 128, 128, 2),
                (32, 128, 64, 1),
                (32, 64, 32, 4),
                (16, 0, 64, 1),
            ):
                code, plan, tables, values = _render_case(
                    case,
                    with_tail=False,
                    bounds=True,
                    chunk_cols=chunk_cols,
                    aux_smem_budget=4096,
                    prefetch_regs=budget,
                    coalesce=coalesce,
                    coop_slots=slots,
                )
                self.assertEqual(code.coop_slots, slots)
                self.assertIn("_cute_flex_coop_issue(", code.chunk_issue)
                self.assertIn("cute.arch.cp_async_commit_group()", code.chunk_issue)
                self.assertIn(
                    f"cute.arch.cp_async_wait_group({slots - 1})", code.chunk_consume
                )
                self.assertNotIn("_cute_flex_coop_issue(", code.chunk_body)
                self.assertNotIn("_cute_flex_coop_issue(", code.chunk_prefetch)
                self.assertNotIn("_run_pf", code.chunk_prefetch)
                staged = [e.aux_index for e in flex_aux_smem_layout(plan, 4096)]
                scores = [torch.randn(128, 128, generator=gen) * 4.0 for _ in tiles]
                outs = _emulate_tiles(
                    code,
                    score_tiles=scores,
                    bh=5,
                    m_tile=1,
                    kv_tiles=tiles,
                    kv_tail_cols=128,
                    aux_tensors=tables,
                    scalar_values=values,
                    chunk_cols=chunk_cols,
                    smem_tables=staged,
                    absent=(2,),
                )
                rows = list(_ROWS)
                for index, (kv_tile, score, out) in enumerate(
                    zip(tiles, scores, outs, strict=True)
                ):
                    if index == 2:
                        continue
                    expected = _reference(
                        case.score,
                        case.mask,
                        score_tile=score,
                        bh=5,
                        m_tile=1,
                        kv_tile=kv_tile,
                        kv_tail_cols=None,
                        aux_tensors=tables,
                        scalar_values=values,
                    )
                    torch.testing.assert_close(
                        out[rows],
                        expected[rows],
                        rtol=1e-5,
                        atol=1e-5,
                        msg=f"{name} chunk={chunk_cols} coalesce={coalesce} tile#{index}",
                    )

    def test_cooperative_block_render_and_facts(self) -> None:
        """graphormer-edge at 128 and 32 columns with a two-slot ring: the
        lane roots once per item, the warp's ring (two chunk blocks of 544
        bytes per row: 34816 bytes; stage 1 at ring 4), each run's slot-0
        region (recast to the table's dtype) and the lane's staged row,
        each run's copy row base (the owner's leading offsets plus the
        lane's row delta times the row step: 1024 / 3072 elements per query
        row); the issue block copies 1 / 3 pieces of 128 bytes per row of
        chunk ``flash_flex_pc`` of tile ``flash_flex_pkv`` into slot
        ``flash_flex_ps`` and commits; the consume block waits for the
        chunk's group, reads the whole-chunk runs from slot ``flash_ci %
        2`` between warp syncs; the element block indexes the chunk run;
        no run is left in the body or the register ring.  At 16 columns
        the 64- and 192-byte spans cap the request at 64 bytes; the knob's
        effective identity and the staging size follow."""
        case = next(c for c in _CASES if c.name == "graphormer_edge")
        code, plan, _tables, _values = _render_case(
            case, with_tail=False, bounds=True, coalesce=128, coop_slots=2, stage=1
        )
        self.assertEqual(code.coop_slots, 2)
        prelude = code.item_prelude
        self.assertIn(
            "flash_flex1_lane = cutlass.Int32(tLDcS[0][0]) % cutlass.Int32(32)", prelude
        )
        self.assertIn(
            "flash_flex1_coopr128, flash_flex1_coops128 = _cute_flex_coop_lane("
            "flash_flex1_lane, 128)",
            prelude,
        )
        self.assertIn(
            "flash_flex1_coopd128 = flash_flex1_coopr128 - flash_flex1_lane", prelude
        )
        self.assertIn(
            "flash_flex1_coopbuf = _flash_flex_coop_smem + (cutlass.Int32(4) + "
            "cutlass.Int32(tLDcS[0][0]) // cutlass.Int32(32)) * cutlass.Int32(34816)",
            prelude,
        )
        self.assertIn(
            "flash_flex1s_load_sp = cute.recast_ptr(flash_flex1_coopbuf, dtype=cutlass.Int32)",
            prelude,
        )
        self.assertIn(
            "flash_flex1s_load_rsp = flash_flex1s_load_sp + flash_flex1_lane * 36",
            prelude,
        )
        self.assertIn(
            "flash_flex1s_load_1_sp = cute.recast_ptr(flash_flex1_coopbuf + 4608, "
            "dtype=cutlass.Int32)",
            prelude,
        )
        self.assertIn(
            "flash_flex1s_load_1_rsp = flash_flex1s_load_1_sp + flash_flex1_lane * 100",
            prelude,
        )
        self.assertIn(
            "flash_flex1s_load_cb = flash_flex1s_load_i0 * cutlass.Int32(1048576) + "
            "flash_flex1s_load_i1 * cutlass.Int32(1024) + flash_flex1_coopd128 * "
            "cutlass.Int32(1024)",
            prelude,
        )
        self.assertIn(
            "flash_flex1s_load_1_cb = flash_flex1s_load_1_i0 * cutlass.Int32(3145728) + "
            "flash_flex1s_load_1_i1 * cutlass.Int32(3072) + flash_flex1_coopd128 * "
            "cutlass.Int32(3072)",
            prelude,
        )
        issue = code.chunk_issue
        self.assertIn(
            "flash_flex1_ikvbase = cutlass.Int32(flash_flex_pkv) * cutlass.Int32(128)",
            issue,
        )
        self.assertIn(
            "flash_flex1_icbase = flash_flex1_ikvbase + cutlass.Int32(tLDcS[None, "
            "flash_flex_pc, None, None][0][1])",
            issue,
        )
        self.assertIn(
            "_cute_flex_coop_issue(_flash_aux0.iterator + flash_flex1s_load_cb, "
            "flash_flex1s_load_lo, flash_flex1_coops128, 1024, 128, 1, 4, cutlass.Int32, "
            "flash_flex1s_load_sp + flash_flex_ps * 4352, flash_flex1_lane, 36)",
            issue,
        )
        self.assertIn(
            "_cute_flex_coop_issue(_flash_aux1.iterator + flash_flex1s_load_1_cb, "
            "(flash_flex1s_load_1_lo) * cutlass.Int32(3), flash_flex1_coops128, 3072, "
            "128, 3, 4, cutlass.Int32, flash_flex1s_load_1_sp + flash_flex_ps * 4352, "
            "flash_flex1_lane, 100)",
            issue,
        )
        self.assertTrue(issue.rstrip().endswith("cute.arch.cp_async_commit_group()"))
        consume = code.chunk_consume.strip().splitlines()
        self.assertEqual(consume[0].strip(), "cute.arch.cp_async_wait_group(1)")
        self.assertEqual(consume[1].strip(), "cute.arch.sync_warp()")
        self.assertEqual(consume[-1].strip(), "cute.arch.sync_warp()")
        self.assertIn(
            "flash_flex1s_load_run = _cute_flex_load_run(flash_flex1s_load_rsp + "
            "(flash_ci % 2) * 4352, 8, 4, cutlass.Int32)",
            code.chunk_consume,
        )
        self.assertIn(
            "flash_flex1s_load_1_run = _cute_flex_load_run(flash_flex1s_load_1_rsp + "
            "(flash_ci % 2) * 4352, 24, 4, cutlass.Int32)",
            code.chunk_consume,
        )
        body = code.chunk_body
        self.assertIn("flash_flex1s_load_1_run[flash_flex_i * 3 + 2]", body)
        self.assertIn("flash_flex1s_load_run[flash_flex_i]", body)
        self.assertNotIn("_cute_flex_load_run(", body)
        self.assertNotIn("_cute_flex_load_vec(", body)
        self.assertNotIn("_flash_aux0.iterator", body)
        self.assertNotIn("_flash_aux1.iterator", body)
        # Both lane dimensions proven: no run guard (the gathers keep theirs).
        self.assertNotIn("flash_flex1s_load_ok", body)
        self.assertNotIn("flash_flex1s_load_1_ok", body)
        self.assertEqual(code.chunk_prefetch, "")
        # The element block is the per-lane form's up to the run indexing.
        plain, _plan_, _tables, _values = _render_case(
            case, with_tail=False, bounds=True, stage=1
        )
        self.assertNotIn("coop", plain.item_prelude + plain.chunk_body)
        self.assertEqual(plain.coop_slots, 0)
        # 16 columns, one slot: 64-byte spans (path_lengths) and 192-byte
        # spans (path_types): both requests 64 bytes wide; 80 + 208 bytes
        # per row, a 9216-byte ring.
        narrow, _plan_, _tables, _values = _render_case(
            case, with_tail=False, bounds=True, coalesce=128, chunk_cols=16
        )
        self.assertEqual(narrow.coop_slots, 1)
        self.assertIn("_cute_flex_coop_lane(flash_flex0_lane, 64)", narrow.item_prelude)
        self.assertIn("* cutlass.Int32(9216)", narrow.item_prelude)
        self.assertIn(
            "flash_flex0_coopbuf + 2560, dtype=cutlass.Int32", narrow.item_prelude
        )
        self.assertIn(
            "flash_flex0_coops64, 1024, 64, 1, 4, cutlass.Int32, flash_flex0s_load_sp + "
            "flash_flex_ps * 2304, flash_flex0_lane, 20)",
            narrow.chunk_issue,
        )
        self.assertIn(
            "flash_flex0_coops64, 3072, 64, 3, 4, cutlass.Int32, flash_flex0s_load_1_sp + "
            "flash_flex_ps * 2304, flash_flex0_lane, 52)",
            narrow.chunk_issue,
        )
        self.assertIn("cute.arch.cp_async_wait_group(0)", narrow.chunk_consume)
        self.assertIn(
            "(flash_ci % 1) * 2304, 4, 4, cutlass.Int32)", narrow.chunk_consume
        )
        # The unproven lane dimension keeps the owner's per-group guard and
        # passes the copy its clamp limit (strided_runs' 64-column table).
        strided = next(c for c in _CASES if c.name == "strided_runs")
        guarded, _plan_, _tables, _values = _render_case(
            strided, with_tail=False, bounds=True, coalesce=128
        )
        self.assertIn(
            "flash_flex0s_load_ok = (flash_flex0s_load_glo >= cutlass.Int32(0)) & "
            "(flash_flex0s_load_glo < cutlass.Int32(64))",
            guarded.chunk_body,
        )
        self.assertIn("flash_flex0_lane, 100, 188)", guarded.chunk_issue)
        self.assertIn(
            "cutlass.select_(flash_flex0s_load_ok, cutlass.Int32("
            "flash_flex0s_load_run[flash_flex_i * 3 + 1])",
            guarded.chunk_body,
        )
        # Facts: bytes per column of the coalescible runs; the per-run
        # request widths and the staging bytes at a knob value and width.
        self.assertEqual(flex_coalesce_steps(plan, _BOUNDS), (4, 12))
        self.assertEqual(flex_coalesce_steps(plan, None), ())
        expected = {
            "graphormer_spatial": (4,),
            "bf16_table": (2,),
            "strided_runs": (12, 6, 2),
            "two_gathers": (4,),
            "flamingo": (),
            "sandwich": (),
            "global_sw": (),
        }
        for name, steps in expected.items():
            other = next(c for c in _CASES if c.name == name)
            _code, other_plan, _tables, _values = _render_case(
                other, with_tail=False, bounds=True
            )
            self.assertEqual(flex_coalesce_steps(other_plan, _BOUNDS), steps, name)
        self.assertEqual(flex_coalesce_widths((4, 12), 128, 32), (128, 128))
        self.assertEqual(flex_coalesce_widths((4, 12), 128, 16), (64, 64))
        self.assertEqual(flex_coalesce_widths((4, 12), 64, 32), (64, 64))
        self.assertEqual(flex_coalesce_widths((4, 12), 32, 16), (32, 32))
        self.assertEqual(flex_coalesce_widths((2,), 128, 16), (32,))
        self.assertEqual(flex_coalesce_widths((1,), 128, 16), (0,))
        self.assertEqual(flex_coalesce_widths((12, 6, 2), 128, 32), (128, 64, 64))
        self.assertEqual(flex_coalesce_widths((4,), 0, 32), (0,))
        self.assertEqual(
            flex_coalesce_smem_bytes((4, 12), 128, 32), 8 * 32 * (144 + 400)
        )
        self.assertEqual(
            flex_coalesce_smem_bytes((4, 12), 128, 16), 8 * 32 * (80 + 208)
        )
        self.assertEqual(flex_coalesce_smem_bytes((4,), 32, 32), 8 * 32 * 144)
        self.assertEqual(flex_coalesce_smem_bytes((4,), 128, 16), 8 * 32 * 80)
        self.assertEqual(flex_coalesce_smem_bytes((1,), 128, 16), 0)
        self.assertEqual(flex_coalesce_smem_bytes((), 128, 32), 0)
        # The score hook's per-element operations (the chunk-loop roll's
        # default fact): graphormer-edge's three gathers, selects, sums and
        # division; nothing for the mask-only plans.
        self.assertEqual(flex_hook_ops(plan, _BOUNDS), 23)
        ops = {
            "graphormer_spatial": 3,
            "sandwich": 5,
            "strided_runs": 13,
            "extra_ops": 54,
            "flamingo": 0,
            "shared_prefix": 0,
        }
        for name, count in ops.items():
            other = next(c for c in _CASES if c.name == name)
            _code, other_plan, _tables, _values = _render_case(
                other, with_tail=False, bounds=True
            )
            self.assertEqual(flex_hook_ops(other_plan, _BOUNDS), count, name)

    def test_coop_runtime_redistributes_rows_and_is_bank_conflict_free(self) -> None:
        """The runtime helpers on a simulated warp: 32 lanes run
        ``coop_issue`` and the row reads in lockstep (threads, the warp sync
        a barrier, the DSL copies / loads replaced by Python ones over a
        fake table and a fake staging slot).  Every lane ends with its own
        row's chunk span; each request of a piece covers whole rows (the
        coalescing); every 8-lane phase of the copies' shared-memory writes
        and of the row loads hits 32 distinct banks at the padded pitch,
        for every width, with and without the clamp."""
        import threading

        class Lane:
            def __init__(self, memory: str, offset: int) -> None:
                self.memory = memory
                self.offset = offset

            def __add__(self, other: object) -> Lane:
                return Lane(self.memory, self.offset + int(other))  # pyrefly: ignore [bad-argument-type]

        class Value(int):
            def ir_value(self) -> Value:
                return self

        class Dtype:
            def __init__(self, width: int) -> None:
                self.width = width
                self.mlir_type = f"i{width}"

            def __call__(self, value: object) -> Value:
                return Value(value)  # pyrefly: ignore [bad-argument-type]

        for width in (128, 64, 32):
            for clamp in (False, True):
                with self.subTest(width=width, clamp=clamp):
                    self._simulate_coop(width, clamp, threading, Lane, Dtype)

    def _simulate_coop(
        self,
        width: int,
        clamp: bool,
        threading: object,
        lane_cls: type,
        dtype_cls: type,
    ) -> None:
        itemsize = 4
        per_vector = 16 // itemsize
        dtype = dtype_cls(32)
        units = 3 if width == 128 else 2
        row_step = 2048  # elements between query rows (8 KB: lines apart)
        rows = 48
        q0 = 9  # the warp's first row
        col = 96  # the chunk's first element along the lane dimension
        unit_elems = width // itemsize
        span = units * unit_elems  # the row's chunk span in elements
        limit = (col + span - per_vector - 8) if clamp else None
        table = [r * 100000 + e for r in range(rows) for e in range(row_step)]
        pitch = coop_pitch(span * itemsize) // itemsize  # elements
        staging: list[object] = [None] * (32 * pitch)
        global_loads: dict[int, list[tuple[int, int]]] = {}
        stores: dict[int, list[tuple[int, int]]] = {}
        loads: dict[int, list[tuple[int, int]]] = {}
        counters: dict[str, list[int]] = {
            "load": [0] * 32,
            "store": [0] * 32,
            "sload": [0] * 32,
        }
        local = threading.local()  # pyrefly: ignore [missing-attribute]
        barrier = threading.Barrier(32)  # pyrefly: ignore [missing-attribute]
        lock = threading.Lock()  # pyrefly: ignore [missing-attribute]

        def fake_load(ptr: object, vector_type: object) -> list[object]:
            lane = local.lane
            count = vector_type[0]
            if ptr.memory == "table":  # pyrefly: ignore [missing-attribute]
                with lock:
                    index = counters["load"][lane]
                    counters["load"][lane] += 1
                    global_loads.setdefault(index, []).append((lane, ptr.offset))  # pyrefly: ignore [missing-attribute]
                self.assertEqual(ptr.offset % per_vector, 0)  # pyrefly: ignore [missing-attribute]
                return table[ptr.offset : ptr.offset + count]  # pyrefly: ignore [missing-attribute]
            with lock:
                index = counters["sload"][lane]
                counters["sload"][lane] += 1
                loads.setdefault(index, []).append((lane, ptr.offset))  # pyrefly: ignore [missing-attribute]
            self.assertEqual(ptr.offset % per_vector, 0)  # pyrefly: ignore [missing-attribute]
            return staging[ptr.offset : ptr.offset + count]  # pyrefly: ignore [missing-attribute]

        def fake_copy(dst: object, src: object, size: int, modifier: str) -> None:
            lane = local.lane
            self.assertEqual((size, modifier), (16, "cg"))
            self.assertEqual(src.memory, "table")  # pyrefly: ignore [missing-attribute]
            self.assertEqual(dst.memory, "stage")  # pyrefly: ignore [missing-attribute]
            with lock:
                index = counters["load"][lane]
                counters["load"][lane] += 1
                global_loads.setdefault(index, []).append((lane, src.offset))  # pyrefly: ignore [missing-attribute]
                stores.setdefault(index, []).append((lane, dst.offset))  # pyrefly: ignore [missing-attribute]
            self.assertEqual(src.offset % per_vector, 0)  # pyrefly: ignore [missing-attribute]
            self.assertEqual(dst.offset % per_vector, 0)  # pyrefly: ignore [missing-attribute]
            staging[dst.offset : dst.offset + per_vector] = table[  # pyrefly: ignore [missing-attribute]
                src.offset : src.offset
                + per_vector  # pyrefly: ignore [missing-attribute]
            ]

        fake_cute = types.SimpleNamespace(
            arch=types.SimpleNamespace(
                load=fake_load,
                cp_async_shared_global=fake_copy,
                sync_warp=lambda: barrier.wait(),
            )
        )
        fake_ir = types.SimpleNamespace(
            VectorType=types.SimpleNamespace(get=lambda shape, t: (shape[0], t))
        )
        fake_cutlass = types.SimpleNamespace(Int32=int, min=min, max=max)
        results: dict[int, list[object]] = {}
        errors: list[BaseException] = []

        def run(lane: int) -> None:
            try:
                local.lane = lane
                row, slot = cast("tuple[int, int]", coop_lane_map(lane, width))
                # The renderer's loader base: the owner's row shifted by the
                # lane's request row minus its own row.
                row_ptr = lane_cls("table", (q0 + lane + row - lane) * row_step)
                stage = lane_cls("stage", 0)
                flex_score_mod_runtime.coop_issue(
                    row_ptr,
                    col,
                    slot,
                    row_step,
                    width,
                    units,
                    per_vector,
                    dtype,
                    stage,
                    lane,
                    pitch,
                    limit,
                )
                barrier.wait()  # the group's wait and the warp sync
                # The consume block's read of the staged row.
                results[lane] = flex_score_mod_runtime.load_run(
                    stage + lane * pitch, span * itemsize // 16, per_vector, dtype
                )
            except BaseException as error:
                errors.append(error)
                barrier.abort()

        with (
            patch.object(flex_score_mod_runtime, "cute", fake_cute),
            patch.object(flex_score_mod_runtime, "ir", fake_ir),
            patch.object(flex_score_mod_runtime, "cutlass", fake_cutlass),
        ):
            threads = [threading.Thread(target=run, args=(lane,)) for lane in range(32)]  # pyrefly: ignore [missing-attribute]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(errors, [])
        lanes_per_row = width // 16
        rows_per_request = 32 // lanes_per_row
        # Every lane holds its own row's span (the clamp folds the pieces
        # past the limit onto the limit, as the GPU fetch does).
        for lane in range(32):
            expected: list[int] = []
            for u in range(units):
                for s_ in range(lanes_per_row):
                    offset = col + s_ * per_vector + u * unit_elems
                    if limit is not None:
                        offset = min(max(offset, 0), limit)
                    base = (q0 + lane) * row_step + offset
                    expected.extend(table[base : base + per_vector])
            self.assertEqual([int(v) for v in results[lane]], expected, lane)
        # Coalescing: request ``k`` of piece ``u`` covers rows ``row + k *
        # rows_per_request`` whole (every slot of each row).
        self.assertEqual(len(global_loads), units * lanes_per_row)
        for index, entries in global_loads.items():
            u, k = divmod(index, lanes_per_row)
            covered = set()
            for lane, offset in entries:
                row, slot = cast("tuple[int, int]", coop_lane_map(lane, width))
                piece = col + slot * per_vector + u * unit_elems
                if limit is not None:
                    piece = min(max(piece, 0), limit)
                self.assertEqual(
                    offset, (q0 + row + k * rows_per_request) * row_step + piece
                )
                covered.add((row + k * rows_per_request, slot))
            self.assertEqual(
                covered,
                {
                    (r, s_)
                    for r in range(k * rows_per_request, (k + 1) * rows_per_request)
                    for s_ in range(lanes_per_row)
                },
            )

        # Bank conflicts: per copy / load instruction and 8-lane phase, the
        # 16-byte shared-memory accesses (element offsets) sit on 32
        # distinct banks.
        def check_phases(accesses: dict[int, list[tuple[int, int]]]) -> None:
            for index, entries in accesses.items():
                by_lane = dict(entries)
                self.assertEqual(len(by_lane), 32, index)
                for phase in range(4):
                    banks: list[int] = []
                    for lane in range(phase * 8, phase * 8 + 8):
                        word = by_lane[lane] * itemsize // 4
                        banks.extend((word + w) % 32 for w in range(4))
                    self.assertEqual(len(set(banks)), 32, (index, phase, banks))

        self.assertEqual(len(stores), units * lanes_per_row)
        self.assertEqual(len(loads), span // per_vector)
        check_phases(stores)
        check_phases(loads)

    def test_prefetch_block_double_buffers_the_next_chunk(self) -> None:
        """graphormer-edge at budget 128: the ``path_lengths`` vector and the
        shared ``path_types`` run of chunk ``flash_flex_pc`` are appended
        per group to the chunk-parity slot in the prefetch block (slots
        initialised per KV step), the group block reads its entry, and no
        table load is left in the chunk body; at budget 32 only the vector
        (32 registers) is prefetched and the run (96) stays in the body."""
        case = next(c for c in _CASES if c.name == "graphormer_edge")
        code, plan, _tables, _values = _render_case(
            case, with_tail=False, bounds=True, prefetch_regs=128
        )
        self.assertEqual(flex_prefetch_regs(plan, _BOUNDS), (32, 96))
        prefetch = code.chunk_prefetch
        self.assertIn(
            "flash_flex0s_load_vec_pf = [None] * 2 if flash_flex_pf_init else "
            "flash_flex0s_load_vec_pf",
            prefetch,
        )
        self.assertIn(
            "flash_flex0s_load_1_run_pf = [None] * 2 if flash_flex_pf_init else "
            "flash_flex0s_load_1_run_pf",
            prefetch,
        )
        self.assertNotIn("_pf", code.tile_prologue)
        self.assertIn("flash_flex0s_load_vec_pf[flash_flex_ps] = []", prefetch)
        self.assertIn("for flash_flex_g in cutlass.range_constexpr(8):", prefetch)
        self.assertIn(
            "flash_flex0_pkvbase = cutlass.Int32(flash_flex_pkv) * cutlass.Int32(128)",
            prefetch,
        )
        self.assertIn(
            "flash_flex0_pgbase = flash_flex0_pkvbase + cutlass.Int32(tLDcS[None, "
            "flash_flex_pc, None, None][flash_flex_g * 4][1])",
            prefetch,
        )
        self.assertIn(
            "flash_flex0s_load_vec_pf[flash_flex_ps].append(_cute_flex_load_vec(",
            prefetch,
        )
        self.assertIn(
            "flash_flex0s_load_1_run_pf[flash_flex_ps].append(_cute_flex_load_run(",
            prefetch,
        )
        self.assertIn(
            "flash_flex0s_load_vec = flash_flex0s_load_vec_pf[flash_ci % 2][flash_flex_g]",
            code.chunk_body,
        )
        self.assertIn(
            "flash_flex0s_load_1_run = flash_flex0s_load_1_run_pf[flash_ci % 2][flash_flex_g]",
            code.chunk_body,
        )
        self.assertNotIn("_cute_flex_load_vec(", code.chunk_body)
        self.assertNotIn("_cute_flex_load_run(", code.chunk_body)
        partial, _plan_, _tables, _values = _render_case(
            case, with_tail=False, bounds=True, prefetch_regs=32
        )
        self.assertIn("_cute_flex_load_vec(", partial.chunk_prefetch)
        self.assertNotIn("_cute_flex_load_run(", partial.chunk_prefetch)
        self.assertIn(
            "flash_flex0s_load_1_run = _cute_flex_load_run(", partial.chunk_body
        )
        self.assertNotIn("run_pf", partial.chunk_prefetch + partial.chunk_body)
        none, _plan_, _tables, _values = _render_case(
            case, with_tail=False, bounds=True
        )
        self.assertEqual(none.chunk_prefetch, "")
        self.assertNotIn("_pf", none.tile_prologue + none.chunk_body)
        # Budget rule: in order while the chunk data fits (a later smaller
        # load still fits); the resolver scales these by the chunk width.
        self.assertEqual(flex_prefetch_count((32, 96), 32), 1)
        self.assertEqual(flex_prefetch_count((32, 96), 128), 2)
        self.assertEqual(flex_prefetch_count((96, 48, 16), 64), 2)
        self.assertEqual(flex_prefetch_count((96, 48, 16), 0), 0)
        # ... then the budget's multiple of the taken set is the depth (at
        # least 1, at most chunks - 1).
        self.assertEqual(flex_prefetch_schedule((32, 96), 32, 4), (1, 1))
        self.assertEqual(flex_prefetch_schedule((32, 96), 64, 4), (1, 2))
        self.assertEqual(flex_prefetch_schedule((32, 96), 128, 4), (2, 1))
        self.assertEqual(flex_prefetch_schedule((16,), 128, 8), (1, 7))
        self.assertEqual(flex_prefetch_schedule((32,), 128, 4), (1, 3))
        self.assertEqual(flex_prefetch_schedule((34,), 32, 4), (0, 0))
        self.assertEqual(code.prefetch_depth, 1)
        self.assertEqual(partial.prefetch_depth, 1)
        self.assertEqual(none.prefetch_depth, 0)
        # graphormer-spatial's 32-register vector at budget 128: three
        # chunks of lead over four-slot rings.
        case = next(c for c in _CASES if c.name == "graphormer_spatial")
        deep, _plan_, _tables, _values = _render_case(
            case, with_tail=False, bounds=True, prefetch_regs=128
        )
        self.assertEqual(deep.prefetch_depth, 3)
        self.assertIn(
            "flash_flex0s_load_vec_pf = [None] * 4 if flash_flex_pf_init else "
            "flash_flex0s_load_vec_pf",
            deep.chunk_prefetch,
        )
        self.assertIn(
            "flash_flex0s_load_vec_pf[flash_flex_ps] = []", deep.chunk_prefetch
        )
        self.assertIn(
            "flash_flex0s_load_vec = flash_flex0s_load_vec_pf[flash_ci % 4][flash_flex_g]",
            deep.chunk_body,
        )

    def test_prefetch_facts_and_window_prefetch(self) -> None:
        """The fact lists the per-chunk loads at 32 columns in order: the
        spatial vector (32), sandwich's fp32 window (17 two-lane vectors:
        34), flamingo's two int32 vectors (32 each), global-sw's bool
        vector (8), the strided case's int32 / bf16 / bool runs (96 / 48 /
        16); unproven windows are gathers (no entry).  A prefetched window
        computes its base, misalignment bit and vectors for the next chunk
        into the slots."""
        expected = {
            "graphormer_spatial": (32,),
            "sandwich": (34,),
            "flamingo": (32, 32),
            "strided_runs": (96, 48, 16),
            "global_sw": (8,),
            "shared_prefix": (),
            "jetspec_tree": (),
        }
        for name, regs in expected.items():
            case = next(c for c in _CASES if c.name == name)
            _code, plan, _tables, _values = _render_case(
                case, with_tail=False, bounds=True
            )
            self.assertEqual(flex_prefetch_regs(plan, _BOUNDS), regs, name)
        case = next(c for c in _CASES if c.name == "sandwich")
        _code, plan, _tables, _values = _render_case(case, with_tail=False)
        self.assertEqual(flex_prefetch_regs(plan, None), ())
        code, _plan_, _tables, _values = _render_case(
            case, with_tail=False, bounds=True, prefetch_regs=64
        )
        self.assertIn(
            "flash_flex0s_load_win_pf = [None] * 2 if flash_flex_pf_init else "
            "flash_flex0s_load_win_pf",
            code.chunk_prefetch,
        )
        self.assertIn(
            "flash_flex0s_load_m0_pf = [None] * 2 if flash_flex_pf_init else "
            "flash_flex0s_load_m0_pf",
            code.chunk_prefetch,
        )
        self.assertIn(
            "flash_flex0_pcbase = flash_flex0_pkvbase + cutlass.Int32(tLDcS[None, "
            "flash_flex_pc, None, None][0][1])",
            code.chunk_prefetch,
        )
        self.assertIn(
            "flash_flex0s_load_win_pf[flash_flex_ps] = _cute_flex_load_window(",
            code.chunk_prefetch,
        )
        self.assertIn(
            "flash_flex0s_load_m0_pf[flash_flex_ps] = (flash_flex0s_load_m & "
            "cutlass.Int32(1)) != cutlass.Int32(0)",
            code.chunk_prefetch,
        )
        self.assertIn(
            "flash_flex0s_load_win = flash_flex0s_load_win_pf[flash_ci % 2]",
            code.chunk_body,
        )
        self.assertIn(
            "flash_flex0s_load_m0 = flash_flex0s_load_m0_pf[flash_ci % 2]",
            code.chunk_body,
        )
        self.assertNotIn("_cute_flex_load_window(", code.chunk_body)
        # Below the window's 34 registers nothing is prefetched.
        code, _plan_, _tables, _values = _render_case(
            case, with_tail=False, bounds=True, prefetch_regs=32
        )
        self.assertEqual(code.chunk_prefetch, "")
        self.assertIn("_cute_flex_load_window(", code.chunk_body)

    def test_strided_runs_share_one_vector_run_per_group(self) -> None:
        """graphormer-edge: the three ``path_types[b, q, kv, n]`` reads (kv at
        stride 3, literal last index) are one 48-byte run per group of four
        lanes, loaded once as three vectors and indexed ``j * 3 + n``; no
        per-element ``kv``, no scalar load of that table."""
        case = next(c for c in _CASES if c.name == "graphormer_edge")
        code, plan, _tables, _values = _render_case(case, with_tail=False, bounds=True)
        self.assertEqual(code.chunk_body.count("_cute_flex_load_run("), 1)
        self.assertIn(
            "flash_flex0s_load_1_run = _cute_flex_load_run(_flash_aux1.iterator + "
            "flash_flex0s_load_1_i0 * cutlass.Int32(3145728) + flash_flex0s_load_1_i1 "
            "* cutlass.Int32(3072) + (flash_flex0s_load_1_lo) * cutlass.Int32(3), 3, 4, "
            "cutlass.Int32)",
            code.chunk_body,
        )
        for n, name in (
            (0, "flash_flex0s_load_1"),
            (1, "flash_flex0s_load_3"),
            (2, "flash_flex0s_load_5"),
        ):
            slot = "flash_flex_j * 3" if n == 0 else f"flash_flex_j * 3 + {n}"
            self.assertIn(
                f"{name} = cutlass.Int32(cutlass.Int32(flash_flex0s_load_1_run[{slot}]))",
                code.chunk_body,
            )
        self.assertNotIn("flash_flex0_kv =", code.chunk_body)
        self.assertNotIn(
            "_flash_aux1.iterator +",
            code.chunk_body.split("_run = ")[1].split("\n", 1)[1],
        )
        self.assertEqual(code.chunk_body.count(".load()"), 3)
        # ``path_lengths`` keeps the plain four-lane vector of the same group.
        self.assertIn(", 4, cutlass.Int32)", code.chunk_body)
        self.assertEqual(
            [a.index_class for a in plan.aux_buffers],
            ["column_affine", "row_col", "value"],
        )
        # Per-dimension proof: ``edge_bias[h, et + 8n]`` keeps the guard and
        # the clamp on the data-dependent column only.
        self.assertIn("flash_flex0s_load_2_i0 * cutlass.Int32(24)", code.chunk_body)
        self.assertNotIn("cutlass.max(flash_flex0s_load_2_i0", code.chunk_body)
        self.assertIn("cutlass.max(flash_flex0s_load_2_i1", code.chunk_body)
        self.assertIn(
            "flash_flex0s_load_2_ok = (flash_flex0s_load_2_i1 >= cutlass.Int32(0)) & "
            "(flash_flex0s_load_2_i1 < cutlass.Int32(24))",
            code.chunk_body,
        )
        # Without the bounds the run keeps the group guard and clamp.
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn(
            "cutlass.min(cutlass.max(flash_flex0s_load_1_lo, cutlass.Int32(0)), "
            "cutlass.Int32(1020))) * cutlass.Int32(3)",
            code.chunk_body,
        )
        self.assertIn("cutlass.select_(flash_flex0s_load_1_ok, ", code.chunk_body)

    def test_mixed_run_groups_take_the_largest_lane_count(self) -> None:
        """int32 stride 3 (four lanes), bf16 stride 3 and bool stride 2 (eight
        lanes): the group loop is eight lanes, the int32 run six vectors."""
        case = next(c for c in _CASES if c.name == "strided_runs")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn(
            "for flash_flex_g in cutlass.range_constexpr(cute.size(flash_disc_frg) // 8):",
            code.chunk_body,
        )
        self.assertIn(", 6, 4, cutlass.Int32)", code.chunk_body)
        self.assertIn(", 3, 8, cutlass.BFloat16)", code.chunk_body)
        self.assertIn("_cute_flex_load_vec(_flash_aux2.iterator", code.chunk_body)
        self.assertIn(", 16, cutlass.Int8)", code.chunk_body)
        self.assertIn("[flash_flex_j * 2 + 1]) != cutlass.Int8(0)", code.chunk_body)

    def test_staged_gathers_read_shared_memory_through_the_sentinel(self) -> None:
        case = next(c for c in _CASES if c.name == "graphormer_spatial")
        code, plan, _tables, _values = _render_case(
            case, with_tail=False, bounds=True, aux_smem_budget=4096
        )
        # 4 x 8 fp32 plus the sentinel, 16-byte rounded.
        self.assertEqual(flex_aux_smem_candidates(plan), (144,))
        self.assertEqual(
            flex_aux_smem_layout(plan, 4096),
            [(1, 0, 32, 144)],
        )
        self.assertEqual(flex_aux_smem_layout(plan, 0), [])
        self.assertIn(
            "flash_flex0s_load_1_off = cutlass.Int32(cutlass.select_(flash_flex0s_load_1_ok,"
            " flash_flex0s_load_1_i0 * cutlass.Int32(8) + flash_flex0s_load_1_i1 * "
            "cutlass.Int32(1), cutlass.Int32(32)))",
            code.chunk_body,
        )
        self.assertIn(
            "flash_flex0s_load_1 = cutlass.Float32((_flash_aux1_smem + "
            "flash_flex0s_load_1_off).load())",
            code.chunk_body,
        )
        self.assertNotIn("cutlass.max(", code.chunk_body)
        self.assertNotIn("_flash_aux1.iterator", code.chunk_body)
        # The vector-form table is not a candidate and stays in global memory.
        self.assertIn("_cute_flex_load_vec(_flash_aux0.iterator", code.chunk_body)
        # Two candidates: staged in plan order while the total fits.
        case = next(c for c in _CASES if c.name == "two_gathers")
        code, plan, _tables, _values = _render_case(
            case, with_tail=False, aux_smem_budget=4096
        )
        self.assertEqual(flex_aux_smem_candidates(plan), (128, 16400))
        self.assertEqual(flex_aux_smem_staged((128, 16400), 4096), 1)
        self.assertEqual(flex_aux_smem_staged((128, 16400), 16384), 1)
        self.assertEqual(flex_aux_smem_staged((128, 16400), 32768), 2)
        self.assertEqual(flex_aux_smem_staged((128, 16400), 0), 0)
        self.assertEqual(flex_aux_smem_layout(plan, 32768)[1][:2], (2, 128))
        self.assertIn("_flash_aux1_smem", code.chunk_body)
        self.assertIn("_flash_aux2.iterator", code.chunk_body)
        # A clamped column index is data-dependent to the proof: a candidate.
        case = next(c for c in _CASES if c.name == "jetspec_tree")
        _code, plan, _tables, _values = _render_case(case, with_tail=False)
        self.assertEqual(flex_aux_smem_candidates(plan), (4112,))
        for name in ("sandwich", "shared_prefix", "flamingo", "bf16_table"):
            case = next(c for c in _CASES if c.name == name)
            _code, plan, _tables, _values = _render_case(case, with_tail=False)
            self.assertEqual(flex_aux_smem_candidates(plan), (), name)

    def test_proven_loads_drop_guards_and_diagonal_gathers_take_windows(self) -> None:
        case = next(c for c in _CASES if c.name == "sandwich")
        guarded, _plan, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn("flash_flex0s_load_ok = ", guarded.chunk_body)
        self.assertIn("flash_flex0s_load_1_ok = ", guarded.item_prelude)
        self.assertNotIn("_cute_flex_load_window", guarded.chunk_body)
        code, plan, tables, values = _render_case(case, with_tail=False, bounds=True)
        # ``rel_bias[q - kv + 1023]`` over q < 512, kv < 1024 lies in [0, 1535)
        # of the 2048-entry table; ``head_scale[h]`` over h < 4.
        self.assertNotIn(
            "_ok", code.item_prelude + code.tile_prologue + code.chunk_body
        )
        self.assertNotIn("cutlass.max(", code.item_prelude + code.chunk_body)
        self.assertNotIn(".load()", code.chunk_body)
        # Per chunk: the first entry at the chunk's last column (falling
        # index), the aligned base, 17 two-lane vectors with the last clamped.
        self.assertIn(
            "flash_flex0s_load_lo = flash_flex0_q + cutlass.Int32(1023)"
            " - (flash_flex0_cbase + cutlass.Int32(31))",
            code.chunk_body,
        )
        self.assertIn(
            "flash_flex0s_load_win = _cute_flex_load_window(_flash_aux0.iterator + "
            "flash_flex0s_load_wb, _flash_aux0.iterator + cutlass.min(flash_flex0s_load_wb"
            " + cutlass.Int32(32), cutlass.Int32(2046)), 17, 2, cutlass.Float32)",
            code.chunk_body,
        )
        self.assertIn(
            "flash_flex0s_load = cutlass.Float32(cutlass.select_(flash_flex0s_load_m0, "
            "flash_flex0s_load_win[31 - flash_flex_i + 1], "
            "flash_flex0s_load_win[31 - flash_flex_i + 0]))",
            code.chunk_body,
        )
        # The window and its selects are the only per-chunk / per-element
        # load statements (no per-element pointer arithmetic).
        self.assertEqual(code.chunk_body.count("_cute_flex_load_window"), 1)

    def test_bounds_proof_declines_small_tables_and_unbounded_indices(self) -> None:
        # A 16-entry table indexed by q: out of range for q >= 16, guarded.
        case = next(c for c in _CASES if c.name == "out_of_range_table")
        code, _plan, _tables, _values = _render_case(case, with_tail=False, bounds=True)
        self.assertIn(
            "_ok = ", code.item_prelude + code.tile_prologue + code.chunk_body
        )
        # A dependent gather (index loaded from another table) has no affine
        # form: guarded and clamped.
        case = next(c for c in _CASES if c.name == "graphormer_spatial")
        code, _plan, _tables, _values = _render_case(case, with_tail=False, bounds=True)
        self.assertIn("flash_flex0s_load_1_ok = ", code.chunk_body)
        self.assertIn("cutlass.max(flash_flex0s_load_1_i1", code.chunk_body)
        # Its ``distances[b, q, kv]`` read is proven (kv < 1024) and keeps the
        # aligned vector form without the per-lane select.
        self.assertIn("_cute_flex_load_vec(_flash_aux0.iterator", code.chunk_body)
        self.assertNotIn("flash_flex0s_load_ok", code.chunk_body)

    def test_row_only_table_loads_render_in_item_prelude(self) -> None:
        case = next(c for c in _CASES if c.name == "shared_prefix")
        code, plan, _tables, _values = _render_case(case, with_tail=False)
        self.assertEqual(code.item_prelude.count(".load()"), 3)
        self.assertNotIn(".load()", code.tile_prologue)
        self.assertNotIn(".load()", code.chunk_body)
        self.assertIn("flash_flex0_q = cutlass.Int32(tLDcS[0][0])", code.item_prelude)
        self.assertEqual([a.index_class for a in plan.aux_buffers], ["row_uniform"] * 3)

    def test_kv_tile_division_renders_in_tile_prologue(self) -> None:
        case = next(c for c in _CASES if c.name == "batchify")
        code, _plan_, _tables, _values = _render_case(
            case, with_tail=False, mask_form="element"
        )
        self.assertIn("// cutlass.Int32(512)", code.tile_prologue)
        self.assertIn("(flash_flex0_kvbase) // cutlass.Int32(512)", code.tile_prologue)
        # ``kv % 512`` is the tile's first-column remainder (a mask: 512 is a
        # power of two) plus the chunk column, plus the lane per element.
        self.assertIn(
            "flash_flex0m_remainder_1_u = flash_flex0_kvbase & cutlass.Int32(511)",
            code.tile_prologue,
        )
        self.assertIn(
            "flash_flex0m_remainder_1_c = flash_flex0m_remainder_1_u + "
            "cutlass.Int32(tLDcS[None, flash_ci, None, None][0][1])",
            code.chunk_body,
        )
        self.assertIn(
            "flash_flex0m_remainder_1 = flash_flex0m_remainder_1_c + "
            "cutlass.Int32(flash_flex_i)",
            code.chunk_body,
        )
        self.assertNotIn("_cute_python_mod(", code.chunk_body)
        self.assertNotIn("// cutlass.Int32(512)", code.chunk_body)
        # The per-row half (``q // 512``, ``q % 512``) is item-invariant.
        self.assertIn("flash_flex0_q // flash_flex0m_div_t0", code.item_prelude)

    def test_sta2d_predicate_collapses_to_one_select_per_element(self) -> None:
        case = next(c for c in _CASES if c.name == "sta2d")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        body_lines = [line for line in code.chunk_body.splitlines() if line.strip()]
        self.assertEqual(len(body_lines), 2)
        self.assertIn("cutlass.select_(flash_flex0m_bitwise_and_2", body_lines[1])

    def test_contiguous_gather_renders_vector_load(self) -> None:
        case = next(c for c in _CASES if c.name == "graphormer_spatial")
        code, plan, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn(
            "for flash_flex_g in cutlass.range_constexpr(cute.size(flash_disc_frg) // 4):",
            code.chunk_body,
        )
        self.assertIn(", 4, cutlass.Int32)", code.chunk_body)
        self.assertIn("_cute_flex_load_vec(_flash_aux0.iterator", code.chunk_body)
        # The dependent ``bias[h, dist]`` read stays a per-lane gather.
        self.assertIn("(_flash_aux1.iterator", code.chunk_body)
        self.assertEqual(
            [a.index_class for a in plan.aux_buffers], ["column_affine", "value"]
        )

    def test_bf16_table_uses_eight_wide_vectors(self) -> None:
        case = next(c for c in _CASES if c.name == "bf16_table")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn(", 8, cutlass.BFloat16)", code.chunk_body)

    def test_hook_params_follow_plan_order(self) -> None:
        case = next(c for c in _CASES if c.name == "graphormer_spatial")
        _code, plan, _tables, _values = _render_case(case, with_tail=False)
        self.assertEqual(
            flex_hook_params(plan),
            [("_flash_aux0", "tensor"), ("_flash_aux1", "tensor")],
        )
        case = next(c for c in _CASES if c.name == "runtime_scale")
        code, plan, _tables, _values = _render_case(case, with_tail=False)
        # The runtime scale is the emitter's ``_flash_scale``, not a hook scalar.
        self.assertEqual(flex_hook_params(plan), [("_flash_hs0", "float32")])
        self.assertIn("_flash_scale", code.chunk_body)
        self.assertIn("_flash_hs0", code.chunk_body)
        case = next(c for c in _CASES if c.name == "host_scalar_mask")
        _code, plan, _tables, _values = _render_case(case, with_tail=False)
        self.assertEqual(flex_hook_params(plan), [("_flash_hs0", "int32")])

    def test_tanh_policy(self) -> None:
        case = next(c for c in _CASES if c.name == "softcap")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn("_cute_flex_tanh(cutlass.Float32(", code.chunk_body)
        self.assertIn("), False)", code.chunk_body)
        self.assertNotIn("cute.math.tanh", code.chunk_body)
        case = next(c for c in _CASES if c.name == "softcap_fast_math")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn("), True)", code.chunk_body)
        # The divide follows the backend override: approximate only under fast_math.
        self.assertIn("cute.math.div(", code.chunk_body)

    def test_sigmoid_and_log_use_backend_overrides(self) -> None:
        case = next(c for c in _CASES if c.name == "sigmoid_act")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn("cute.math.rcp(1.0 + cute.math.exp2(", code.chunk_body)
        self.assertIn("cute.math.log(", code.chunk_body)
        self.assertNotIn(
            "cute.math.log(v_", code.chunk_body.replace("cute.math.log(v_", "LOG(", 0)
        )
        self.assertNotIn(
            "fastmath=True)", code.chunk_body.split("cute.math.log(")[1].split("\n")[0]
        )

    def test_remainder_is_python_mod(self) -> None:
        case = next(c for c in _CASES if c.name == "svg_temporal")
        code, _plan_, _tables, _values = _render_case(
            case, with_tail=False, mask_form="element"
        )
        # ``(q - 128) % 512`` (negative on the first rows) is Python-mod in
        # the prelude; ``(kv - 128) % 512`` is lane-affine over the tile.
        self.assertIn("_cute_python_mod(cutlass.Int32(", code.item_prelude)
        self.assertIn(
            "flash_flex0m_remainder_1_u = (flash_flex0_kvbase + cutlass.Int32(-128))"
            " & cutlass.Int32(511)",
            code.tile_prologue,
        )
        self.assertNotIn(" % ", code.chunk_body)
        self.assertNotIn(" % ", code.tile_prologue)
        self.assertNotIn("_cute_python_mod(", code.chunk_body)
        # ``kv < 128`` is constant over a 128-column tile.
        self.assertIn(
            "flash_flex0m_lt_1 = operator.lt((flash_flex0_kvbase + cutlass.Int32(-128)),"
            " cutlass.Int32(0))",
            code.tile_prologue,
        )

        # A non-power-of-two span modulus keeps the Python-mod helper.
        def mask_mod_384(score, b, h, q_idx, kv_idx):
            return (kv_idx % 384) <= (q_idx % 384)

        with _environment():
            region, aux, scalars = build_flex_region(mask_mod_384)
            plan = _plan(None, region, aux, scalars)
            code = _render(plan, with_score=False, with_mask=True, tail_mask=False)
        self.assertIn(
            "_cute_python_mod(flash_flex0_kvbase, cutlass.Int32(384))",
            code.tile_prologue,
        )

    def test_tail_mask_is_applied_after_the_selects(self) -> None:
        case = next(c for c in _CASES if c.name == "sigmoid_act")
        code, _plan_, _tables, _values = _render_case(case, with_tail=True)
        lines = [line.strip() for line in code.chunk_body.splitlines() if line.strip()]
        self.assertEqual(
            lines[-1],
            "_helion_flash_rt.mask_r2p_sm100_rank1(flash_disc_frg, _flash_kv_tail_cols - flash_ci * 32)",
        )
        self.assertTrue(lines[-2].startswith("flash_disc_frg[flash_flex_i] = "))

    def test_item_invariant_values_are_capped(self) -> None:
        def mask_many_tables(score, b, h, q_idx, kv_idx, *tables):
            keep = kv_idx >= hl.load(tables[0], [q_idx])
            for table in tables[1:]:
                keep = keep | (kv_idx >= hl.load(table, [q_idx]))
            return keep

        gen = torch.Generator().manual_seed(3)
        tables = [_int_table(gen, (1024,), 0, 1024) for _ in range(10)]
        names = [f"t{i}" for i in range(10)]
        with _environment():
            region, aux, scalars = build_flex_region(
                mask_many_tables, aux_specs=_specs(tables, names)
            )
            plan = _plan(None, region, aux, scalars)
            code = _render(plan, with_score=False, with_mask=True, tail_mask=False)
        self.assertEqual(code.item_prelude.count(".load()"), 8)
        self.assertEqual(code.tile_prologue.count(".load()"), 2)
        score_tile = torch.randn(128, 128, generator=gen)
        result = _emulate(
            code,
            score_tile=score_tile,
            bh=2,
            m_tile=1,
            kv_tile=3,
            kv_tail_cols=128,
            aux_tensors=tables,
            scalar_values=[],
        )
        expected = _reference(
            None,
            mask_many_tables,
            score_tile=score_tile,
            bh=2,
            m_tile=1,
            kv_tile=3,
            kv_tail_cols=None,
            aux_tensors=tables,
            scalar_values=[],
        )
        torch.testing.assert_close(result[list(_ROWS)], expected[list(_ROWS)])

    def test_region_index_classes(self) -> None:
        gen = torch.Generator().manual_seed(0)
        sandwich = _specs(_sandwich_tables(gen), ("rel_bias", "head_scale"))
        flamingo = _specs(_flamingo_tables(gen), ("start_tok", "end_tok"))
        jetspec = _specs(_jetspec_tables(gen), ("ancestor",))
        global_sw = _specs(_global_tables(gen), ("is_global",))
        with _environment():
            _region, aux, _scalars = build_flex_region(
                score_sandwich, aux_specs=sandwich
            )
            self.assertEqual([a.index_class for a in aux], ["diagonal", "head"])
            _region, aux, _scalars = build_flex_region(
                mask_flamingo, aux_specs=flamingo
            )
            self.assertEqual([a.index_class for a in aux], ["column_affine"] * 2)
            _region, aux, _scalars = build_flex_region(
                mask_jetspec_tree, aux_specs=jetspec
            )
            self.assertEqual([a.index_class for a in aux], ["row_col"])
            _region, aux, _scalars = build_flex_region(
                mask_global_sw, aux_specs=global_sw
            )
            # Read both by row and by column: no single class.
            self.assertEqual([a.index_class for a in aux], ["general"])

    def test_two_hooks_of_a_stage_share_prelude_text(self) -> None:
        """The emitter deduplicates prelude / prologue lines by exact text."""
        case = next(c for c in _CASES if c.name == "sandwich")
        tables, _values = _case_inputs(case)
        with _environment():
            plan = _build(case, tables)
            masked = _render(plan, with_score=True, with_mask=True, tail_mask=True)
            unmasked = _render(plan, with_score=True, with_mask=False, tail_mask=False)
        self.assertTrue(
            set(unmasked.item_prelude.splitlines())
            <= set(masked.item_prelude.splitlines())
        )
        self.assertTrue(
            set(unmasked.tile_prologue.splitlines())
            <= set(masked.tile_prologue.splitlines())
        )
        for code in (masked, unmasked):
            for block in code[:4]:
                assert isinstance(block, str)
                self.assertIsNone(re.search(r"\bv_\d+\b", block), block)
                self.assertTrue(block == "" or block.endswith("\n"))
        # Lifted temporaries are named after the node they belong to.
        self.assertIn("flash_flex0s_add_t0 = 1023", unmasked.chunk_body)

    def test_default_row_expression_substitutes_the_chunk_index(self) -> None:
        case = next(c for c in _CASES if c.name == "dilated_sw")
        tables, _values = _case_inputs(case)
        with _environment():
            plan = _build(case, tables)
            code = render_flex_hook(
                plan,
                stage=1,
                score_tensor="flash_disc_frg",
                coord_tensor="tLDcS[None, flash_ci, None, None]",
                m_tile_expr="flash_m_tile1",
                kv_tile_expr="flash_kv",
                chunk_expr="flash_ci",
                chunk_cols=16,
                indent=16,
                with_score=False,
                with_mask=True,
                tail_mask=True,
                device_function=_StubDeviceFunction(),  # pyrefly: ignore [bad-argument-type]
            )
        self.assertIn(
            "flash_flex1_q = cutlass.Int32(tLDcS[None, 0, None, None][0][0])",
            code.item_prelude,
        )
        self.assertIn("_flash_kv_tail_cols - flash_ci * 16", code.chunk_body)

    def test_host_scalars_are_cast_at_use(self) -> None:
        case = next(c for c in _CASES if c.name == "host_scalar_mask")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn("cutlass.Int32(_flash_hs0)", code.chunk_body)
        case = next(c for c in _CASES if c.name == "runtime_scale")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn("cutlass.Float32(_flash_scale)", code.chunk_body)
        self.assertIn("cutlass.Float32(_flash_hs0)", code.chunk_body)

    def test_matcher_copied_load_sources(self) -> None:
        """A region may keep the ``_host_tensor`` source node of a copied load
        (with ``meta["flex_source_name"]``) instead of an aux placeholder."""
        from helion._compiler.inductor_lowering import APIFuncLowering
        from helion.language import _tracing_ops

        case = next(c for c in _CASES if c.name == "shared_prefix")
        tables, _values = _case_inputs(case)
        with _environment():
            plan = _build(case, tables)
            graph = plan.mask_region.graph  # pyrefly: ignore [missing-attribute]
            placeholders = [n for n in graph.nodes if n.op == "placeholder"]
            loads = [n for n in graph.nodes if n.target is hl.load]
            self.assertEqual(len(loads), 3)
            for load in loads:
                source = load.args[0]
                index = placeholders.index(source) - 5
                name = plan.aux_buffers[index].name
                with graph.inserting_before(load):
                    host = graph.call_function(_tracing_ops._host_tensor, (name,))
                host.meta["val"] = source.meta["val"]
                host.meta["lowering"] = APIFuncLowering(_tracing_ops._host_tensor)
                host.meta["location"] = load.meta["location"]
                load.meta["flex_source_name"] = name
                load.args = (host, *load.args[1:])
            code = _render(plan, with_score=False, with_mask=True, tail_mask=False)
        self.assertEqual(code.item_prelude.count(".load()"), 3)
        for i in range(3):
            self.assertIn(f"_flash_aux{i}.iterator", code.item_prelude)
        score_tile = torch.randn(128, 128, generator=torch.Generator().manual_seed(9))
        result = _emulate(
            code,
            score_tile=score_tile,
            bh=3,
            m_tile=2,
            kv_tile=1,
            kv_tail_cols=128,
            aux_tensors=tables,
            scalar_values=[],
        )
        expected = _reference(
            None,
            case.mask,
            score_tile=score_tile,
            bh=3,
            m_tile=2,
            kv_tile=1,
            kv_tail_cols=None,
            aux_tensors=tables,
            scalar_values=[],
        )
        torch.testing.assert_close(result[list(_ROWS)], expected[list(_ROWS)])

    def test_stage_names_are_distinct(self) -> None:
        case = next(c for c in _CASES if c.name == "dilated_sw")
        tables, _values = _case_inputs(case)
        with _environment():
            plan = _build(case, tables)
            code0 = _render(
                plan, with_score=False, with_mask=True, tail_mask=False, stage=0
            )
            code1 = _render(
                plan, with_score=False, with_mask=True, tail_mask=False, stage=1
            )
        self.assertIn("flash_flex0_q = ", code0.item_prelude)
        self.assertIn("flash_flex1_q = ", code1.item_prelude)
        self.assertIn("flash_m_tile1", code1.item_prelude)
        self.assertNotIn("flash_flex0", code1.chunk_body)

    def test_empty_hook(self) -> None:
        case = next(c for c in _CASES if c.name == "dilated_sw")
        tables, _values = _case_inputs(case)
        with _environment():
            plan = _build(case, tables)
            code = _render(plan, with_score=True, with_mask=False, tail_mask=False)
            self.assertEqual(code, FlexHookCode("", "", ""))
            code = _render(plan, with_score=False, with_mask=False, tail_mask=True)
        self.assertEqual(code.item_prelude, "")
        self.assertIn("mask_r2p_sm100_rank1", code.chunk_body)

    # -- keep-word peephole -------------------------------------------------

    def _bitmask_lines(self, code: FlexHookCode) -> tuple[list[str], list[str]]:
        """(word lines of the chunk block, lines of the element loop)."""
        lines = [line for line in code.chunk_body.splitlines() if line.strip()]
        loop = next(i for i, line in enumerate(lines) if "for flash_flex_i" in line)
        return lines[:loop], lines[loop + 1 :]

    def test_bitmask_interval_and_parity_words(self) -> None:
        """dilated-sw: ``|q - kv| <= 512`` is a lane interval (two ``below``
        bounds), ``|q - kv| % 2 == 0`` the ``0x55555555`` pattern shifted by
        the row's parity (item-invariant), applied by one lane-bit select."""
        case = next(c for c in _CASES if c.name == "dilated_sw")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn(
            "_cute_flex_lane_below(cutlass.Int32(flash_flex0_q) - flash_flex0_cbase"
            " - cutlass.Int32(512), 32)",
            code.chunk_body,
        )
        self.assertIn(
            "_cute_flex_lane_below(cutlass.Int32(flash_flex0_q) - flash_flex0_cbase"
            " + cutlass.Int32(513), 32)",
            code.chunk_body,
        )
        self.assertIn("* cutlass.Int32(1431655765)", code.item_prelude)
        self.assertIn("& cutlass.Int32(1)", code.item_prelude)
        _words, element = self._bitmask_lines(code)
        self.assertEqual(len(element), 1)
        self.assertIn(
            "cutlass.select_(cutlass.Boolean(flash_flex0m_w6 & (cutlass.Int32(1) << "
            "cutlass.Int32(flash_flex_i))), cutlass.Float32(flash_disc_frg[flash_flex_i]),"
            " -cutlass.Float32.inf)",
            element[0],
        )
        for text in ("abs(", "_cute_python_mod(", "operator."):
            self.assertNotIn(text, code.chunk_body)
        # 16-column chunks: 16-bit words, complements masked to the chunk.
        code16, _plan_, _tables, _values = _render_case(
            case, with_tail=False, chunk_cols=16
        )
        self.assertIn(", 16)", code16.chunk_body)
        self.assertIn("~flash_flex0m_w0 & cutlass.Int32(65535)", code16.chunk_body)
        self.assertIn("* cutlass.Int32(21845)", code16.item_prelude)

    def test_bitmask_lane_affine_remainder_and_uniform_broadcast(self) -> None:
        """svg-temporal: ``(kv - 128) % 512`` is lane-affine over the tile,
        the band ``|q_tok - kv_tok| < 64`` an interval, ``kv < 128`` a
        tile-uniform comparison OR-ed in as a broadcast word."""
        case = next(c for c in _CASES if c.name == "svg_temporal")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn(
            "flash_flex0m_lt_1 = operator.lt((flash_flex0_kvbase + cutlass.Int32(-128)),"
            " cutlass.Int32(0))",
            code.tile_prologue,
        )
        self.assertIn(
            "cutlass.select_(flash_flex0m_bitwise_or, cutlass.Int32(-1), cutlass.Int32(0))",
            code.tile_prologue,
        )
        self.assertIn(
            "_cute_flex_lane_below(cutlass.Int32(flash_flex0m_remainder) - "
            "flash_flex0m_remainder_1_c - cutlass.Int32(63), 32)",
            code.chunk_body,
        )
        self.assertIn(
            "_cute_flex_lane_below(cutlass.Int32(flash_flex0m_remainder) - "
            "flash_flex0m_remainder_1_c + cutlass.Int32(64), 32)",
            code.chunk_body,
        )
        words, element = self._bitmask_lines(code)
        self.assertEqual(len(element), 1)
        # No per-element remainder: the lane-affine form is only read by the
        # word thresholds.
        self.assertNotIn("+ cutlass.Int32(flash_flex_i)", code.chunk_body)
        self.assertEqual(len([w for w in words if "_cute_flex_lane_below" in w]), 2)

    def test_bitmask_block_equality_and_period_pattern(self) -> None:
        """jetspec-training: ``kv_rel // 16 == q_block`` is the half-open
        block ``[q_block * 16, q_block * 16 + 16)``; ``kv_off <= q_off`` with
        period 16 < 32 is the residue pattern ``below(q_off + 1, 16)``
        replicated (``* 0x10001``) with no rotation (``(-640) % 16 == 0``)."""
        case = next(c for c in _CASES if c.name == "jetspec_training")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn(
            "_cute_flex_lane_below(cutlass.Int32(flash_flex0m_div) * cutlass.Int32(16)"
            " - flash_flex0_cbase + cutlass.Int32(640), 32)",
            code.chunk_body,
        )
        self.assertIn("+ cutlass.Int32(656), 32)", code.chunk_body)
        self.assertIn(
            "_cute_flex_lane_below(cutlass.Int32(flash_flex0m_remainder) + "
            "cutlass.Int32(1), 16)",
            code.item_prelude,
        )
        self.assertIn("* cutlass.Int32(65537)", code.item_prelude)
        self.assertNotIn(">>", code.item_prelude)
        _words, element = self._bitmask_lines(code)
        self.assertEqual(len(element), 1)

    def test_bitmask_runtime_thresholds_from_row_tables(self) -> None:
        case = next(c for c in _CASES if c.name == "shared_prefix")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertIn(
            "_cute_flex_lane_below(cutlass.Int32(flash_flex0m_load) - flash_flex0_cbase, 32)",
            code.chunk_body,
        )
        _words, element = self._bitmask_lines(code)
        self.assertEqual(len(element), 1)
        self.assertEqual(code.chunk_body.count("_cute_flex_lane_below"), 4)

    def test_bitmask_mixed_with_per_element_table_term(self) -> None:
        """global-sw: the window interval and the row's flag fold into a word;
        the per-column flag (a vector load) stays per element and is OR-ed
        with the word's lane bit."""
        case = next(c for c in _CASES if c.name == "global_sw")
        code, _plan_, _tables, _values = _render_case(
            case, with_tail=False, bounds=True
        )
        self.assertIn(
            "cutlass.select_(flash_flex0m_load, cutlass.Int32(-1), cutlass.Int32(0))",
            code.item_prelude,
        )
        self.assertIn(
            "flash_flex0m_bitwise_or = cutlass.Boolean(flash_flex0m_w5 & "
            "(cutlass.Int32(1) << cutlass.Int32(flash_flex_j",
            code.chunk_body.replace("flash_flex_i", "flash_flex_j"),
        )
        self.assertIn("_cute_flex_load_vec(", code.chunk_body)
        self.assertIn(
            "flash_flex0m_bitwise_or_1 = flash_flex0m_bitwise_or | flash_flex0m_load_1",
            code.chunk_body,
        )
        self.assertNotIn("abs(", code.chunk_body)

    def test_bitmask_declines_per_column_tables_and_fmod(self) -> None:
        for name in ("flamingo", "int_fmod", "python_semantics"):
            case = next(c for c in _CASES if c.name == name)
            code, _plan_, _tables, _values = _render_case(case, with_tail=False)
            self.assertNotIn("_w0", code.chunk_body, name)
            self.assertNotIn("_cute_flex_lane_below", code.chunk_body, name)
        # A uniform predicate (sta2d) keeps the plain per-element select.
        case = next(c for c in _CASES if c.name == "sta2d")
        code, _plan_, _tables, _values = _render_case(case, with_tail=False)
        self.assertNotIn("_w0", code.chunk_body + code.tile_prologue)
        element = _render_case(case, with_tail=False, mask_form="element")[0]
        for block, other in zip(code[:4], element[:4], strict=True):
            assert isinstance(block, str) and isinstance(other, str)
            self.assertEqual(sorted(block.splitlines()), sorted(other.splitlines()))

    def test_bitmask_with_score_region_and_tail(self) -> None:
        case = next(c for c in _CASES if c.name == "rel_bias")
        code, _plan_, _tables, _values = _render_case(case, with_tail=True)
        lines = [line.strip() for line in code.chunk_body.splitlines() if line.strip()]
        self.assertTrue(
            lines[-2].startswith(
                "flash_disc_frg[flash_flex_i] = cutlass.Float32(cutlass.select_("
                "cutlass.Boolean(flash_flex0m_w6 & "
            )
        )
        self.assertIn("mask_r2p_sm100_rank1", lines[-1])

    def test_uniform_comparison_and_remainder_classes(self) -> None:
        """``kv < 512`` / ``kv // 64`` / ``kv % 64`` over 128- and 32-column
        spans: the hoisted part lands in the tile / chunk block."""
        case = next(c for c in _CASES if c.name == "block_diffusion")
        code, _plan_, _tables, _values = _render_case(
            case, with_tail=False, mask_form="element"
        )
        self.assertIn(
            "operator.lt((flash_flex0_kvbase + cutlass.Int32(-512)), cutlass.Int32(0))",
            code.tile_prologue,
        )
        case = next(c for c in _CASES if c.name == "natten2d")
        code, _plan_, _tables, _values = _render_case(
            case, with_tail=False, mask_form="element"
        )
        self.assertIn(
            "flash_flex0m_remainder_1_u = flash_flex0_cbase & cutlass.Int32(63)",
            code.chunk_body,
        )
        self.assertIn("(flash_flex0_cbase) // cutlass.Int32(64)", code.chunk_body)

        # ``kv <= 1023`` is ``kv < 1024``: tile-uniform; ``kv <= 1024`` is not.
        def mask_le(score, b, h, q_idx, kv_idx):
            return (kv_idx <= 1023) & (kv_idx <= 1024) & (q_idx >= 0)

        with _environment():
            region, aux, scalars = build_flex_region(mask_le)
            plan = _plan(None, region, aux, scalars)
            code = _render(
                plan,
                with_score=False,
                with_mask=True,
                tail_mask=False,
                mask_form="element",
            )
        self.assertIn(
            "flash_flex0m_le = operator.le((flash_flex0_kvbase + cutlass.Int32(-1023)),"
            " cutlass.Int32(0))",
            code.tile_prologue,
        )
        self.assertIn(
            "flash_flex0m_le_1 = operator.le(flash_flex0_kv,", code.chunk_body
        )

    def test_unknown_mask_form_is_rejected(self) -> None:
        case = next(c for c in _CASES if c.name == "dilated_sw")
        tables, _values = _case_inputs(case)
        with _environment():
            plan = _build(case, tables)
            with self.assertRaises(helion.exc.BackendUnsupported):
                _render(
                    plan,
                    with_score=False,
                    with_mask=True,
                    tail_mask=False,
                    mask_form="r2p",
                )
