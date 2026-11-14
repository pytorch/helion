from __future__ import annotations

import unittest

import torch

from helion.runtime.settings import _get_backend

if _get_backend() != "triton":
    raise unittest.SkipTest("the gluon tests run in the triton lane")

from triton.runtime.jit import JITFunction

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessGluon
import helion.language as hl


@helion.kernel(static_shapes=True, backend="gluon")
def row_sums(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc = acc + x[tile_m, tile_n].sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def pick_rows(state: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    b, heads, r, c = idx.size(0), state.size(1), state.size(2), state.size(3)
    out = torch.empty([b * heads, r, c], dtype=state.dtype, device=state.device)
    for tile_bh, tile_r in hl.tile([b * heads, r]):
        s = idx[tile_bh.id // heads].long()
        out[tile_bh.id, tile_r, :] = state[s, tile_bh.id % heads, tile_r, :] * 2
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def dot_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def rewrite_mm(x: torch.Tensor, y: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    """Each step writes its x tile, doubled, to ``scratch`` and reads it back."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            scratch[tile_m, tile_k] = x[tile_m, tile_k] * 2
            acc = torch.addmm(acc, scratch[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def decayed_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc * 0.5, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def jagged_add(x: torch.Tensor, offsets: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(y)
    for tile in hl.tile(y.size(0)):
        starts = offsets[tile]
        nnz = offsets[tile.index + 1] - starts
        for tile_n in hl.tile(0, nnz.amax()):
            mask = tile_n.index[None, :] < nnz[:, None]
            row = hl.load(x, [starts[:, None] + tile_n.index[None, :]], extra_mask=mask)
            out[tile, tile_n] = y[tile, tile_n] + row
    return out


@helion.kernel(static_shapes=False, backend="gluon")
def segment_mm(
    a: torch.Tensor,
    w: torch.Tensor,
    order: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    max_t: int,
) -> torch.Tensor:
    out = torch.zeros([a.size(0), w.size(2)], dtype=torch.float32, device=a.device)
    for e in hl.grid(w.size(0)):
        start, count = starts[e], counts[e]
        for tile_t, tile_n in hl.tile([max_t, w.size(2)]):
            valid = tile_t.index < count
            rows = order[start + torch.where(valid, tile_t.index, 0)]
            acc = hl.zeros([tile_t, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(a.size(1)):
                acc = torch.addmm(acc, a[rows, tile_k], w[e, tile_k, tile_n])
            hl.store(out, [rows, tile_n], acc, extra_mask=valid[:, None])
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def gather_mv(w: torch.Tensor, idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    s = x.size(0)
    out = torch.empty([idx.size(0) * s, 1], dtype=torch.float32, device=x.device)
    for tile in hl.tile(idx.size(0) * s):
        acc = hl.zeros([tile, 1], dtype=torch.float32)
        rows = idx[tile.index // s]
        for tile_k in hl.tile(s):
            acc = hl.dot(w[rows * s + tile.index % s, tile_k], x[tile_k, :], acc=acc)
        out[tile, :] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def split_k_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, 256]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(outer_k.begin, outer_k.end):
            acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n], acc)
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def mm_nt(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n, _ = w.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_n, tile_k].T)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def step_scaled_mm(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a = (x[tile_m, tile_k] * s[tile_k.begin]).to(x.dtype)
            acc = torch.addmm(acc, a, y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def asm_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a = hl.inline_asm_elementwise(
                "mov.b16 $0, $1;", "=h,h", [x[tile_m, tile_k]], torch.bfloat16, True, 1
            )
            acc = torch.addmm(acc, a, y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batch, m, k = a.size()
    n = b.size(2)
    out = torch.empty([batch, m, n], dtype=torch.float32, device=a.device)
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def panel_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        a = x[tile_m, :]
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = hl.dot((a * 2).to(x.dtype), y[:, tile_n])
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def mm_add(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n], acc)
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def double_then_copy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    for tile in hl.tile(n):
        v = x[tile]
        x[tile] = v * 2
        y[tile] = v
    return y


@helion.kernel(static_shapes=True, backend="gluon")
def scatter_twice(
    slots: torch.Tensor, val: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    t, h, d = val.size()
    for tile_t, tile_h in hl.tile([t, h], block_size=[1, 4]):
        slot = slots[tile_t]
        block, offset = (slot // 16)[:, None, None], (slot % 16)[:, None, None]
        index = [
            block,
            offset,
            tile_h.index[None, :, None],
            hl.arange(d)[None, None, :],
        ]
        hl.store(a, index, val[tile_t, tile_h, :])
        hl.store(b, index, val[tile_t, tile_h, :] * 2)
    return b


@helion.kernel(static_shapes=True, backend="gluon")
def two_dots(
    q: torch.Tensor, k: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n, c, d, dv = q.size(0), hl.specialize(q.size(1)), q.size(2), h.size(2)
    out = torch.empty([n, c, dv], dtype=torch.float32, device=q.device)
    attn = torch.empty([n, c, c], dtype=torch.float32, device=q.device)
    for tile_n, tile_v in hl.tile([n, dv]):
        acc = hl.zeros([tile_n, c, tile_v], dtype=torch.float32)
        scores = hl.zeros([tile_n, c, c], dtype=torch.float32)
        for tile_d in hl.tile(d):
            qt, kt = q[tile_n, :, tile_d], k[tile_n, :, tile_d]
            ht = h[tile_n, tile_d, tile_v]
            acc = hl.dot(qt, ht, acc=acc)
            scores = hl.dot(qt, kt.transpose(-2, -1), acc=scores)
        out[tile_n, :, tile_v] = acc
        attn[tile_n, :, :] = scores
    return out, attn


@helion.kernel(static_shapes=True, backend="gluon")
def ring_steps(
    q: torch.Tensor, k: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c, d, dv = q.size(0), hl.specialize(q.size(1)), q.size(2), h.size(2)
    out = torch.empty([n, c, dv], dtype=torch.float32, device=q.device)
    attn = torch.empty([n, c, c], dtype=torch.float32, device=q.device)
    gram = torch.empty([n, c, c], dtype=torch.float32, device=q.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n, c, dv], dtype=torch.float32)
        scores = hl.zeros([tile_n, c, c], dtype=torch.float32)
        grams = hl.zeros([tile_n, c, c], dtype=torch.float32)
        for tile_d in hl.tile(d):
            qt = q[tile_n, :, tile_d]
            ht = h[tile_n, tile_d, :]
            kt = k[tile_n, :, tile_d]
            scores = hl.dot(qt, kt.transpose(-2, -1), acc=scores)
            grams = hl.dot(kt, kt.transpose(-2, -1), acc=grams)
            acc = hl.dot(kt, ht.to(kt.dtype), acc=acc)
        out[tile_n, :, :] = acc
        attn[tile_n, :, :] = scores
        gram[tile_n, :, :] = grams
    return out, attn, gram


@helion.kernel(static_shapes=True, backend="gluon")
def fp8_fp4_mm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """mxfp8 times mxfp4 packed along K."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for kk in hl.grid(k // 128):
            acc = hl.dot_scaled(
                x[tile_m, kk * 128 + hl.arange(128)],
                x_scale[tile_m, kk * 4 + hl.arange(4)],
                "e4m3",
                y[kk * 64 + hl.arange(64), tile_n],
                y_scale[tile_n, kk * 4 + hl.arange(4)],
                "e2m1",
                acc=acc,
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def bf16_fp4_mm(
    x: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """bf16 (no scale) times mxfp4 packed along K."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for kk in hl.grid(k // 128):
            acc = hl.dot_scaled(
                x[tile_m, kk * 128 + hl.arange(128)],
                None,
                "bf16",
                y[kk * 64 + hl.arange(64), tile_n],
                y_scale[tile_n, kk * 4 + hl.arange(4)],
                "e2m1",
                acc=acc,
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, backend="gluon")
def nvfp4_mm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """nvfp4 (e4m3 scales per 16) times nvfp4, both packed along K."""
    m = x.size(0)
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for kk in hl.grid(y.size(0) // 128):
            acc = hl.dot_scaled(
                x[tile_m, kk * 128 + hl.arange(128)],
                x_scale[tile_m, kk * 16 + hl.arange(16)],
                "e2m1",
                y[kk * 128 + hl.arange(128), tile_n],
                y_scale[tile_n, kk * 16 + hl.arange(16)],
                "e2m1",
                acc=acc,
            )
        out[tile_m, tile_n] = acc
    return out


def _e2m1(packed: torch.Tensor) -> torch.Tensor:
    """fp4 bytes packed along dim 0 (low nibble first) as float32 values."""
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device, dtype=torch.float32
    )
    codes = torch.stack([packed & 15, packed >> 4], 1).flatten(0, 1).long()
    return torch.where(codes >= 8, -values[codes % 8], values[codes % 8])


def _last_max(
    a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]
):
    return torch.maximum(a[0], b[0]), b[1]


@helion.kernel(static_shapes=True, backend="gluon")
def scan_pair(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        tags = idx[tile_m].float().unsqueeze(1).expand_as(x[tile_m, tile_n])
        out[tile_m, tile_n], _ = hl.associative_scan(
            _last_max, (x[tile_m, tile_n], tags), dim=0
        )
    return out


def _asm(
    kernel: object, args: tuple[torch.Tensor, ...], **config: object
) -> tuple[dict[str, str], torch.Tensor]:
    """The assembly (by stage) of ``kernel`` compiled for ``args`` under ``config``,
    and its output."""
    fn = kernel.bind(args).compile_config(helion.Config(**config))  # pyrefly: ignore [missing-attribute]
    out = fn(*args)
    jit = next(v for v in fn.__globals__.values() if isinstance(v, JITFunction))
    (compiled,) = jit.device_caches[args[0].device.index][0].values()
    return compiled.asm, out


@skipUnlessGluon()
@skipIfRefEager("gluon codegen")
class TestGluon(TestCase):
    def test_loop_unroll(self) -> None:
        # 7 inner steps unrolled by 3 (two unrolled steps and a remainder); the
        # persistent loop unrolled by 2 leaves some programs no row at all.
        x = torch.randn(40, 224, device=DEVICE)
        for config in (
            {"range_unroll_factors": [0, 3]},
            {"range_unroll_factors": [2, 0], "pid_type": "persistent_blocked"},
        ):
            code, out = code_and_output(row_sums, (x,), block_sizes=[8, 32], **config)
            self.assertIn("loop_unroll_factor=", code)
            torch.testing.assert_close(out, x.sum(-1))

    def test_scalar_index_expand_dims(self) -> None:
        # A loaded scalar index broadcast as s[:, None] (Triton splats it to [1]).
        state = torch.randn(5, 3, 32, 64, device=DEVICE)
        idx = torch.tensor([4, 0, 2], device=DEVICE, dtype=torch.int32)
        code, out = code_and_output(pick_rows, (state, idx), block_sizes=[1, 16])
        self.assertIn("[:, None] * 6144", code)
        torch.testing.assert_close(out, (state[idx.long()] * 2).reshape(9, 32, 64))

    def test_matmul_configs(self) -> None:
        x = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16)
        # A TMA ring on a flat launch.
        code, out = code_and_output(
            mm,
            (x, y),
            block_sizes=[128, 128, 64],
            indexing="tensor_descriptor",
            num_stages=3,
            num_warps=4,
        )
        self.assertIn("ring_acquire(", code)
        torch.testing.assert_close(out, x @ y, rtol=1e-2, atol=1e-1)
        # An 8-row tile: the epilogue splits an accumulator layout holding
        # broadcast registers.
        _, out = code_and_output(
            mm,
            (x, y),
            block_sizes=[8, 64, 64],
            indexing=["pointer", "pointer", "tensor_descriptor"],
            num_warps=2,
            epilogue_subtile=2,
        )
        torch.testing.assert_close(out, x @ y, rtol=1e-2, atol=1e-1)

    def test_atomic_of_repeated_registers(self) -> None:
        # An mma.sync accumulator of 4 rows holds each element in two registers:
        # its atomic issues once.
        x = torch.randn(64, 256, device=DEVICE, dtype=torch.float16)
        y = torch.randn(256, 64, device=DEVICE, dtype=torch.float16)
        out = torch.zeros(64, 64, device=DEVICE)
        code_and_output(mm_add, (x, y, out), block_sizes=[4, 8, 128], num_warps=2)
        torch.testing.assert_close(out, x.float() @ y.float(), rtol=1e-2, atol=1e-1)

    def test_load_read_once(self) -> None:
        # A small tile's load stays one load: its second use does not re-read
        # memory behind the store between them.
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        y = torch.zeros_like(x)
        ref = x.clone()
        code_and_output(double_then_copy, (x, y), block_sizes=[64], num_warps=4)
        torch.testing.assert_close(y, ref)
        torch.testing.assert_close(x, ref * 2)

    def test_load_used_across_stores(self) -> None:
        # A loaded index whose first use folds away indexes two stores: the second
        # store's index is not re-read behind the first.
        slots = torch.tensor([37], device=DEVICE)
        val = torch.randn(1, 8, 64, device=DEVICE)
        a, b = (
            torch.zeros(4, 16, 8, 64, device=DEVICE),
            torch.zeros(4, 16, 8, 64, device=DEVICE),
        )
        code_and_output(scatter_twice, (slots, val, a, b))
        torch.testing.assert_close(a[2, 5], val[0])
        torch.testing.assert_close(b[2, 5], val[0] * 2)

    def test_ring_operand_of_later_dot(self) -> None:
        # k loads ahead of the first dot but feeds the second: its ring slot is
        # the second dot's step (it was the first's: the GPU faulted).
        q, k = (
            torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16) for _ in "qk"
        )
        h = torch.randn(8, 128, 128, device=DEVICE, dtype=torch.bfloat16)
        code, (out, attn) = code_and_output(
            two_dots,
            (q, k, h),
            block_sizes=[1, 64, 64],
            indexing="tensor_descriptor",
            num_stages=3,
            num_warps=4,
        )
        self.assertIn("ring_acquire(", code)
        torch.testing.assert_close(out, q.float() @ h.float(), rtol=1e-2, atol=1e-1)
        torch.testing.assert_close(attn, q.float() @ k.float().mT, rtol=1e-2, atol=1e-1)

    def test_ring_steps_in_order(self) -> None:
        # k feeds three dots, one reading it as both operands (one slot), and the
        # fp32 h is read into registers while q's step is open (a plain load):
        # both took a step out of order (the GPU faulted).
        q, k = (
            torch.randn(8, 64, 128, device=DEVICE, dtype=torch.bfloat16) for _ in "qk"
        )
        h = torch.randn(8, 128, 64, device=DEVICE)
        code, (out, attn, gram) = code_and_output(
            ring_steps,
            (q, k, h),
            block_sizes=[1, 64],
            indexing="tensor_descriptor",
            num_stages=3,
            num_warps=4,
        )
        self.assertIn("ring_acquire(", code)
        kf = k.float()
        torch.testing.assert_close(out, kf @ h.bfloat16().float(), rtol=1e-2, atol=1e-1)
        torch.testing.assert_close(attn, q.float() @ kf.mT, rtol=1e-2, atol=1e-1)
        torch.testing.assert_close(gram, kf @ kf.mT, rtol=1e-2, atol=1e-1)

    def test_dot_scaled_mixed_formats(self) -> None:
        # fp8 x fp4 is a tcgen05 MMA with the fp4 operand padded (128-row tiles)
        # or Triton's decomposition in a loop (32); bf16 x fp4 the decomposition.
        x = (torch.randn(256, 512, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
        x16 = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randint(0, 256, [256, 128], dtype=torch.uint8, device=DEVICE)
        x_scale, y_scale = (
            torch.randint(124, 130, [rows, 16], dtype=torch.uint8, device=DEVICE)
            for rows in (256, 128)
        )
        factor_x, factor_y = (
            torch.exp2(s.float() - 127).repeat_interleave(32, 1)
            for s in (x_scale, y_scale)
        )
        w = _e2m1(y) * factor_y.T
        for block_sizes in ([128, 128], [32, 32]):
            with self.subTest(block_sizes=block_sizes):
                config = {"block_sizes": block_sizes, "num_stages": 2}
                config["indexing"] = "tensor_descriptor"
                _, out = code_and_output(fp8_fp4_mm, (x, x_scale, y, y_scale), **config)
                expected = (x.float() * factor_x) @ w
                torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
                _, out = code_and_output(bf16_fp4_mm, (x16, y, y_scale), **config)
                torch.testing.assert_close(out, x16.float() @ w, rtol=1e-3, atol=1e-3)

    def test_dot_scaled_nvfp4(self) -> None:
        # e4m3 scales: a row-major scale tile reaches tensor memory through
        # registers (tcgen05.cp reads packed blocks), and off tcgen05 Triton's
        # decomposition casts them (the translator's reads e8m0 exponents).
        x = torch.randint(0, 256, [256, 256], dtype=torch.uint8, device=DEVICE)
        y = torch.randint(0, 256, [256, 128], dtype=torch.uint8, device=DEVICE)
        x_scale, y_scale = (
            (torch.rand(rows, 32, device=DEVICE) + 0.5).to(torch.float8_e4m3fn)
            for rows in (256, 128)
        )
        factor_x, factor_y = (
            s.float().repeat_interleave(16, 1) for s in (x_scale, y_scale)
        )
        expected = (_e2m1(x.T).T * factor_x) @ (_e2m1(y) * factor_y.T)
        for config in (
            {
                "block_sizes": [128, 64],
                "indexing": "tensor_descriptor",
                "num_stages": 3,
            },
            {"block_sizes": [32, 32], "num_stages": 1},
        ):
            with self.subTest(**config):
                _, out = code_and_output(nvfp4_mm, (x, x_scale, y, y_scale), **config)
                torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

    def test_pipelined_pointer_loads(self) -> None:
        # Pointer loads feeding a dot stream num_stages - 1 steps ahead (cp.async):
        # a tcgen05 dot reads their slots, mma.sync reads them back, and so does
        # a dot beside a TMA ring operand (in the ring's body partition).
        x = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16)
        ring = ["tensor_descriptor", "pointer", "pointer"]
        for block_sizes, indexing, kind in (
            ([128, 128, 64], "pointer", "pipe_release"),
            ([32, 64, 64], "pointer", "pipe_take"),
            ([128, 128, 64], ring, "pipe_take"),
        ):
            with self.subTest(block_sizes=block_sizes, indexing=indexing):
                code, out = code_and_output(
                    mm, (x, y), block_sizes=block_sizes, indexing=indexing, num_stages=3
                )
                self.assertIn(f"_helion_gluon.{kind}(", code)
                if kind == "pipe_release":
                    # The zero fill is the first MMA's to overwrite (Triton's
                    # OptimizeAccumulatorInit): no tensor-memory store ahead.
                    self.assertNotIn("acc_put(", code)
                torch.testing.assert_close(out, x @ y, rtol=1e-2, atol=1e-1)
        # tcgen05 reads an fp8 operand in either order (Triton's MMAv5 rule): the
        # MN-major y slots are read in place, nothing restaged.
        x8, y8 = x.to(torch.float8_e4m3fn), y.to(torch.float8_e4m3fn)
        asm, out = _asm(dot_mm, (x8, y8), block_sizes=[128, 128, 64], num_stages=3)
        self.assertEqual(asm["ttgir"].count("ttg.local_store"), 0)
        torch.testing.assert_close(out, x8.float() @ y8.float(), rtol=1e-3, atol=1e-2)
        # A load of memory its loop writes stays in program order (only y streams).
        scratch = torch.zeros_like(x)
        code, out = code_and_output(
            rewrite_mm, (x, y[:, :64], scratch), block_sizes=[64, 64, 64], num_stages=3
        )
        self.assertEqual(code.count("pipe_alloc("), 1)
        expected = (2 * x.float()) @ y[:, :64].float()
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)
        # mma.sync reads its operands in place, a transposed one (w.T) too.
        w = y.T.contiguous()
        asm, out = _asm(mm_nt, (x[:16], w), block_sizes=[16, 16, 64], num_stages=3)
        self.assertNotRegex(asm["ttgir"], "convert_layout.*dot_op")
        torch.testing.assert_close(
            out, x[:16].float() @ w.float().T, rtol=1e-2, atol=1e-1
        )
        # A load cp.async cannot stream (K = 101: no 4 contiguous bytes) is a
        # plain load its dot stages (mma.sync through swizzled shared copies,
        # stored before either is read); an asynchronous MMA's goes to staged slots.
        x101, y101 = x[:, :101].contiguous(), y[:101].contiguous()
        for block_sizes in ([32, 32, 32], [128, 64, 32]):
            with self.subTest(block_sizes=block_sizes, k=101):
                asm, out = _asm(mm, (x101, y101), block_sizes=block_sizes, num_stages=3)
                self.assertNotRegex(asm["ttgir"], "convert_layout.*dot_op")
                torch.testing.assert_close(out, x101 @ y101, rtol=1e-2, atol=1e-1)
        # A scalar load feeding the dot (one per step) streams as one element;
        # unrolled, num_stages counts unrolled steps (Triton unrolls first).
        s = torch.rand(512, device=DEVICE)
        a = x * s[::64].repeat_interleave(64).to(x.dtype)
        for unroll, slots in ((0, 2), (2, 4)):
            with self.subTest(unroll=unroll):
                asm, out = _asm(
                    step_scaled_mm,
                    (x, y, s),
                    block_sizes=[64, 64, 64],
                    num_stages=3,
                    range_unroll_factors=[0, unroll],
                )
                self.assertIn(f"memdesc<{slots}x1xf32", asm["ttgir"])
                torch.testing.assert_close(
                    out, a.float() @ y.float(), rtol=1e-2, atol=1e-1
                )
        # A unit batch squeezes: the MMAs read the slots, into a resident accumulator
        # (mma.sync: through the squeezing reshape).
        a3, b3 = x.view(2, 128, 512), y.view(2, 256, 256).repeat(1, 2, 1)
        for indexing, tile in (
            ("pointer", 128),
            ("tensor_descriptor", 128),
            ("pointer", 16),
        ):
            with self.subTest(bmm=indexing, tile=tile):
                code, out = code_and_output(
                    bmm,
                    (a3, b3),
                    block_sizes=[1, tile, tile, 64],
                    indexing=indexing,
                    num_stages=3,
                )
                self.assertNotIn("acc_get(", code)
                torch.testing.assert_close(
                    out, (a3.float() @ b3.float()), rtol=1e-2, atol=1e-1
                )
        # An operand through inline asm (an argument list in its statement) too:
        # restaged beside y's slot, which the MMA reads in place, so that MMA
        # completes before the slot's refill.
        code, out = code_and_output(
            asm_mm, (x, y), block_sizes=[64, 64, 64], num_stages=3
        )
        self.assertRegex(code, r"tmem_dot\([^\n]*, True\)")
        torch.testing.assert_close(out, x.float() @ y.float(), rtol=1e-2, atol=1e-1)

    def test_split_k(self) -> None:
        # Triton's axis info of a comparison: the split's bound (a multiple of the
        # block) keeps each mask run whole, so its loads stream (cp.async); and
        # its float atomics issue as vectors.
        x = torch.randn(64, 1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1024, 64, device=DEVICE, dtype=torch.bfloat16)
        asm, out = _asm(split_k_mm, (x, y), block_sizes=[32, 32, 64], num_stages=3)
        self.assertIn("ttg.async_copy_global_to_local", asm["ttgir"])
        self.assertRegex(asm["ptx"], r"(red|atom)\.global[.\w]*\.v[248]\.f32")
        torch.testing.assert_close(out, x.float() @ y.float(), rtol=1e-2, atol=1e-1)

    def test_rescaled_accumulator_stays_in_tensor_memory(self) -> None:
        # An accumulator rescaled into each MMA (flash attention's) is carried in
        # tensor memory, read just before its rescale (Triton's HoistTMEMAlloc).
        x = torch.randn(128, 256, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(256, 64, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(decayed_mm, (x, y), block_sizes=[128, 64, 64])
        self.assertRegex(code, r"acc_put\((tmem_acc\w*), acc\)\n(\s*#.*\n)*\s*for ")
        expected = torch.zeros(128, 64, device=DEVICE)
        for k in range(0, 256, 64):
            expected = expected * 0.5 + x[:, k : k + 64].float() @ y[k : k + 64].float()
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)

    def test_invariant_copies_ahead_of_the_loop(self) -> None:
        # A small loop-invariant load (fewer elements than threads: it keeps an
        # auto layout) the pipelined gather's address math reads several times
        # is copied ahead of the K loop, not re-loaded in it (Triton's LICM).
        w = torch.randn(4 * 2048, 2048, device=DEVICE, dtype=torch.bfloat16)
        idx = torch.randint(0, 4, [2], device=DEVICE, dtype=torch.int64)
        x = torch.randn(2048, 1, device=DEVICE, dtype=torch.bfloat16)
        asm, out = _asm(
            gather_mv, (w, idx, x), block_sizes=[32, 128], num_warps=2, num_stages=5
        )
        lines = asm["ttgir"].splitlines()
        start = next(i for i, line in enumerate(lines) if "scf.for" in line)
        indent = len(lines[start]) - len(lines[start].lstrip())
        body = []
        for line in lines[start + 1 :]:
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
        self.assertIn("async_copy_global_to_local", "\n".join(body))
        self.assertNotRegex("\n".join(body), r"tt\.load .*tensor<32x!tt\.ptr<i64>")
        expected = w.view(4, 2048, 2048)[idx].reshape(-1, 2048).float() @ x.float()
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def test_copies_of_copies(self) -> None:
        # A loop-invariant value's copy ahead of the loop takes its operands'
        # copies itself (the body's trace takes back only it: the replayed
        # load's mask), and keeps their metadata past the dry run of an inner
        # loop that takes it (rows' copies, replayed in the K loop's).
        lengths = torch.randint(0, 129, [64], device=DEVICE)
        lengths[::16] = 128  # each row tile's loop covers y
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        x = torch.randn(int(offsets[-1]), device=DEVICE)
        y = torch.randn(64, 128, device=DEVICE)
        _, out = code_and_output(jagged_add, (x, offsets, y), block_sizes=[16, 16])
        expected = y.clone()
        for i, n in enumerate(lengths.tolist()):
            expected[i, :n] += x[offsets[i] : offsets[i] + n]
        torch.testing.assert_close(out, expected)
        a = torch.randn(256, 100, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(4, 100, 200, device=DEVICE, dtype=torch.bfloat16)
        order = torch.randperm(256, device=DEVICE, dtype=torch.int32)
        counts = torch.tensor([64, 100, 60, 32], device=DEVICE, dtype=torch.int32)
        starts = (counts.cumsum(0) - counts).to(torch.int32)
        args = (a, w, order, starts, counts, 100)
        _, out = code_and_output(
            segment_mm, args, block_sizes=[32, 128, 64], num_stages=3
        )
        expert = torch.repeat_interleave(torch.arange(4, device=DEVICE), counts.long())
        rows = order.long()
        expected = torch.zeros_like(out)
        expected[rows] = torch.bmm(a[rows].float()[:, None], w[expert].float())[:, 0]
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)

    def test_loop_invariant_code_motion(self) -> None:
        # A value the inner loop does not vary is computed, and staged for its
        # MMA, once ahead of the loop (Triton's LICM).
        x = torch.randn(256, 64, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(64, 256, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(panel_mm, (x, y), block_sizes=[128, 64])
        self.assertIn("_helion_staged", code)
        torch.testing.assert_close(
            out, (x * 2).float() @ y.float(), rtol=1e-2, atol=1e-1
        )

    def test_scan_mixed_layouts(self) -> None:
        # A scan's operands from differently laid out values share one layout.
        x = torch.randn(32, 64, device=DEVICE)
        idx = torch.arange(32, device=DEVICE, dtype=torch.int32)
        _, out = code_and_output(scan_pair, (x, idx), block_sizes=[32, 32])
        torch.testing.assert_close(out, torch.cummax(x, 0).values)


if __name__ == "__main__":
    unittest.main()
