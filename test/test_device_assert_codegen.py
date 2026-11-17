from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from packaging import version
import pytest
import torch

import helion
from helion import exc
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipUnlessBackends
from helion._testing.cute_aux import as_sm100
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl
from helion.runtime.settings import _get_backend

pytestmark = pytest.mark.cpu_only


@helion.kernel(static_shapes=True)
def checked_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    out = torch.empty([idx.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(idx.size(0)):
        i = idx[tile]
        hl.device_assert((i >= 0) & (i < x.size(0)), "index out of bounds")
        out[tile] = hl.load(x, [i])
    return out


@helion.kernel(static_shapes=True)
def checked_columns(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = hl.load(x, [tile, hl.arange(100)])
        hl.device_assert(v != 0, "zero element")
        hl.store(out, [tile, hl.arange(100)], v)
    return out


@helion.kernel(static_shapes=True)
def checked_range(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(hl.arange(100) < 100, "range past its length")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True, device_asserts=False)
def unchecked(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(torch.sin(y[tile]) < 0, "not checked")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True)
def checked_shift(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile, :]
        hl.device_assert(v > 0, "nonpositive")
        out[tile, :] = v - v.amax(-1, keepdim=True)
    return out


@helion.kernel(static_shapes=True)
def tensor_message(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(x[tile] > 0, x[tile])
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True)
def braces(debug: torch.Tensor, sanitize_overflow: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(debug)
    for tile in hl.tile(debug.size(0)):
        hl.device_assert(debug[tile] < sanitize_overflow[tile], "{debug} < {x}")
        out[tile] = debug[tile]
    return out


@helion.kernel(static_shapes=True)
def assert_in_combine(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.reduce(checked_add, x[tile, :], dim=1)
    return out


@helion.kernel(static_shapes=True, device_asserts=False)
def unchecked_in_combine(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.reduce(checked_add, x[tile, :], dim=1)
    return out


def checked_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    hl.device_assert(a >= 0, "negative partial sum")
    return a + b


@helion.kernel(static_shapes=True)
def matmul_checked(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        hl.device_assert(acc == acc, "nan")
        out[tile_m, tile_n] = acc.to(a.dtype)
    return out


@helion.kernel(static_shapes=True)
def reshaped_condition(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows, cols in hl.tile(x.size()):
        hl.device_assert(x[rows, cols].flatten() > 0, "nonpositive")
        out[rows, cols] = x[rows, cols]
    return out


@helion.kernel(static_shapes=True)
def moved_condition(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        hl.device_assert(x[rows, :].reshape(2, rows) > 0, "nonpositive")
        out[rows, :] = x[rows, :]
    return out


@helion.kernel(static_shapes=True)
def outer_reshape(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows, cols in hl.tile(x.size()):
        flat = x[rows, cols].flatten()
        for _ in hl.tile(1):
            hl.device_assert(flat > 0, "nonpositive")
        out[rows, cols] = x[rows, cols]
    return out


@helion.kernel(static_shapes=True)
def kept_reshape(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for rows in hl.tile(x.size(0)):
        v = x[rows, :].view(rows, 2, 64)
        hl.device_assert(v.sum(-1).sum(-1) > 0, "nonpositive")
        out[rows] = x[rows, :].sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carried_reshape(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        cond = hl.full([rows, 2], True, dtype=torch.bool)
        for _ in hl.tile(2, block_size=1):
            hl.device_assert(cond, "carried")
            cond = x[rows, :].transpose(0, 1).reshape(rows, 2) > 0
        out[rows, :] = x[rows, :]
    return out


@helion.kernel(static_shapes=True)
def returned_reshape(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        cond = hl.full([rows, 2], True, dtype=torch.bool)
        for _ in hl.tile(2, block_size=1):
            cond = x[rows, :].transpose(0, 1).reshape(rows, 2) > 0
        hl.device_assert(cond, "after loop")
        out[rows, :] = x[rows, :]
    return out


@helion.kernel(static_shapes=True)
def branched_reshape(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        cond = hl.full([rows, 2], True, dtype=torch.bool)
        if flag[0] > 0:
            cond = x[rows, :].transpose(0, 1).reshape(rows, 2) > 0
        hl.device_assert(cond, "after branch")
        out[rows, :] = x[rows, :]
    return out


@helion.kernel(static_shapes=True)
def other_carry(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for rows in hl.tile(x.size(0)):
        flat = x[rows, :].reshape(-1)
        total = hl.zeros([rows])
        for _ in hl.tile(2, block_size=1):
            flat = flat * 2
            total = total + x[rows, 0]
        hl.device_assert(total >= 0, "total")
        out[rows] = total
    return out


@helion.kernel(static_shapes=True)
def permuted_range(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = hl.arange(100)[:, None] < hl.full([100, 4], 100, dtype=torch.int32)
        hl.device_assert(v.transpose(0, 1), "permuted range")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True)
def stacked_condition(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows, cols in hl.tile(x.size()):
        v = x[rows, cols] > 0
        hl.device_assert(torch.stack([v, v, v]), "stacked")
        hl.device_assert(torch.stack([v, v, v]).all(0), "reduced")
        out[rows, cols] = x[rows, cols]
    return out


@helion.kernel(static_shapes=True)
def split_range(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(hl.arange(100).reshape(2, 50) < 100, "split range")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True)
def empty_condition(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(hl.arange(0) < 0, "empty")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True)
def reshaped_ranges(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(hl.arange(100).unsqueeze(0) < 100, "unsqueezed")
        hl.device_assert(hl.arange(100).view(1, 100) < 100, "viewed")
        hl.device_assert(hl.arange(100).unsqueeze(0).expand(4, 100) < 100, "expanded")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True)
def halved_range(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=32):
        r = hl.arange(tile.block_size // 2)
        hl.device_assert(r < tile.block_size // 2, "range")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True)
def reduced_reshape(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        total = x[rows, :].reshape(-1).sum()
        hl.device_assert(total > -1e30, "sum")
        out[rows, :] = x[rows, :]
    return out


@helion.kernel(static_shapes=True)
def doubled_range(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=32):
        r = hl.arange(2 * tile.block_size)
        hl.device_assert(r < 2 * tile.block_size, "range")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def checked_gathered_matmul(
    a: torch.Tensor, b: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    k, n = b.shape
    m = rows.numel()
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        source_rows = rows[tile_m]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[source_rows, tile_k] * 0.5, b[tile_k, tile_n])
        hl.device_assert(acc == acc, "nan")
        out[tile_m, tile_n] = torch.relu(acc).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def computed_range(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=32):
        r = hl.arange(3 * tile.block_size)
        hl.device_assert(r < 3 * tile.block_size, "range")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True)
def debug_tunable(x: torch.Tensor) -> torch.Tensor:
    hl.register_tunable("_triton_config_debug", EnumFragment(choices=(False,)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(x[tile] >= 0, "negative")
        out[tile] = x[tile]
    return out


def checked_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.device_assert(x[tile] >= 0, "negative")
        out[tile] = x[tile]
    return out


@helion.kernel(static_shapes=True, backend="cute")
def checked_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batch_heads, seq_len, head_dim = q.size()
    out = torch.empty_like(q)
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_tile = q[bh.begin, tile_m, :]
        for tile_n in hl.tile(seq_len):
            qk = hl.dot(q_tile * 0.18033688, k[bh.begin, tile_n, :].transpose(0, 1))
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = hl.dot(
                p.to(v.dtype), v[bh.begin, tile_n, :], acc=acc * alpha[:, None]
            )
            m_i = m_new
        hl.device_assert(l_i > 0, "empty softmax row")
        out[bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def checked_attention_backward(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    o_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # examples/attention.py's attention_backward, with an assert.
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_n_rows = k.size(0)
    dq = torch.zeros((q.size(0), head_dim), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    for tile_n in hl.tile(total_n_rows, block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        dk_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        batch_idx = tile_n.begin // n_dim
        start_m = batch_idx * m_dim
        end_m = start_m + m_dim
        for tile_m in hl.tile(start_m, end_m, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            m_i = lse[tile_m]
            di = delta[tile_m]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            p_t = torch.exp2(qk_t - m_i[None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - di[None, :])).to(q.dtype)
            dq_acc = hl.dot(ds_t.T, k_j, out_dtype=torch.float32)
            hl.atomic_add(dq, [tile_m, slice(None)], dq_acc * 0.6931471824645996)
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
        hl.device_assert(dk_acc == dk_acc, "nan")
        dv[tile_n, :] = dv_acc.to(v.dtype)
        dk[tile_n, :] = (dk_acc * sm_scale).to(k.dtype)
    return dq.reshape(q_in.size()), dk.reshape(k_in.size()), dv.reshape(v_in.size())


@helion.kernel(static_shapes=True, backend="cute")
def checked_kv_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batch_heads, seq_len, head_dim = q.size()
    out = torch.empty_like(q)
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_tile = q[bh.begin, tile_m, :]
        for tile_n in hl.tile(seq_len):
            hl.device_assert(tile_n.begin < seq_len, "kv tile past the end")
            qk = hl.dot(q_tile * 0.18033688, k[bh.begin, tile_n, :].transpose(0, 1))
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = hl.dot(
                p.to(v.dtype), v[bh.begin, tile_n, :], acc=acc * alpha[:, None]
            )
            m_i = m_new
        out[bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


def code(kernel: object, *args: object) -> str:
    bound = cpu_bind(kernel, args)
    return bound.to_code(bound.config_spec.default_config())


@onlyBackends(["triton", "cute"])
class TestDeviceAssertCodegen(TestCase):
    def test_gather(self):
        # The lanes past the end of the last tile load 0 and read as true; the
        # launch compiles the check in (Triton's debug mode, without its
        # overflow sanitizer; CuTe's --enable-assertions).
        source = code(
            checked_gather, torch.zeros(1000), torch.zeros(777, dtype=torch.int64)
        )
        if _get_backend() == "triton":
            lines = [
                "_mask_to = tl.where(mask_0, v_4, tl.full([], True, tl.int1))",
                "tl.device_assert(_mask_to, 'index out of bounds')",
                "debug=True, sanitize_overflow=False",
            ]
        else:
            lines = [
                "_mask_to = cutlass.Boolean(v_4) if mask_0 else cutlass.Boolean(True)",
                "cute.testing.assert_(_mask_to, 'index out of bounds')",
                "cute_compile_options='--enable-assertions'",
            ]
        for line in lines:
            self.assertIn(line, source)

    @skipUnlessBackends(["triton"])  # CuTe's hl.arange needs a tile.
    def test_padded_range(self):
        # hl.arange(100) runs over 128 lanes, which no tile masks: the 28 past
        # its elements read as true.
        source = code(checked_range, torch.zeros(1000))
        self.assertIn(
            "tl.device_assert(tl.where(tl.arange(0, 128) < 100, v_1, True), "
            "'range past its length')",
            source,
        )
        # In the dim of a block of 100 (a load at [tile, hl.arange(100)]),
        # _mask_to masks them already.
        source = code(checked_columns, torch.zeros(1000, 100))
        self.assertIn("tl.device_assert(_mask_to, 'zero element')", source)

    def test_compiled_out(self):
        # Nothing of the check is left: not its condition, which reads y, nor
        # y; the printed decorator keeps the setting.
        bound = cpu_bind(unchecked, (torch.zeros(1000), torch.zeros(1000)))
        config = bound.config_spec.default_config()
        source = bound.to_code(config)
        device, _, host = source.partition("def unchecked(")
        for text in ("tl.device_assert(", "cute.testing.assert_(", "_mask_to", "sin"):
            self.assertNotIn(text, device)
        self.assertNotIn(", y,", device)
        # Nor the launch options of the asserts, in the host's launch.
        for text in ("debug=True", "--enable-assertions"):
            self.assertNotIn(text, host)
        self.assertIn(
            "device_asserts=False",
            bound.format_kernel_decorator(config, bound.settings),
        )

    def test_rolled_reduction(self):
        # A reduction loop checks each chunk of the row, the lanes past its
        # 4000 elements read as true; the loop that reloads it after the
        # check does not keep the reduction persistent.
        bound = cpu_bind(checked_shift, (torch.zeros(64, 4000),))
        config = bound.config_spec.default_config()
        source = bound.to_code(
            helion.Config(**{**config.config, "reduction_loops": [256]})
        )
        self.assertEqual(source.count("for roffset_1 in "), 2)
        call = (
            "tl.device_assert" if _get_backend() == "triton" else "cute.testing.assert_"
        )
        self.assertIn(f"\n        {call}(_mask_to, 'nonpositive')", source)

    def test_message_must_be_a_string(self):
        with self.assertRaises(exc.TypeInferenceError):
            code(tensor_message, torch.zeros(1000))

    def test_names(self):
        # The message is a literal, braces included; arguments named after the
        # compile options a Triton launch gets are renamed.
        source = code(braces, torch.zeros(1000), torch.zeros(1000))
        self.assertIn("'{debug} < {x}')", source)
        if _get_backend() == "triton":
            self.assertIn(
                "def _helion_braces(debug_1, sanitize_overflow_1, out):", source
            )

    def test_other_backends(self):
        # A backend without a device assert (Metal; TileIR, which shares
        # Triton's codegen) compiles a kernel with one only with the checks
        # compiled out.
        x = torch.zeros(1000)
        if _get_backend() == "triton":
            # ROCm's Triton keeps tl.device_assert from 3.7 on.
            with (
                patch("helion._compat.is_hip", return_value=True),
                patch(
                    "helion._compat.get_triton_version",
                    return_value=version.parse("3.6"),
                ),
                self.assertRaisesRegex(exc.BackendUnsupported, "hl.device_assert"),
            ):
                code(checked_gather, x, torch.zeros(777, dtype=torch.int64))
            # A build whose launch lacks an option the asserts add.
            with (
                patch(
                    "helion._compat.supported_launch_options",
                    return_value=frozenset({"num_warps", "num_stages", "debug"}),
                ),
                self.assertRaisesRegex(exc.BackendUnsupported, "hl.device_assert"),
            ):
                code(checked_gather, x, torch.zeros(777, dtype=torch.int64))
        with patch.dict(os.environ, {"ENABLE_TILE": "1"}):
            for backend in ("metal", "tileir"):
                # debug_dtype_asserts=False: Metal rejects those first.
                kernel = helion.kernel(
                    checked_copy,
                    backend=backend,
                    static_shapes=True,
                    debug_dtype_asserts=False,
                )
                with self.assertRaisesRegex(exc.BackendUnsupported, "hl.device_assert"):
                    code(kernel, x)
                kernel = helion.kernel(
                    checked_copy,
                    backend=backend,
                    static_shapes=True,
                    device_asserts=False,
                    debug_dtype_asserts=False,
                )
                self.assertNotIn("negative'", code(kernel, x))

    @skipUnlessBackends(["cute"])
    def test_flash_attention(self):
        # The fused tcgen05 attention body does not emit the assert: the
        # search surface leaves a kernel with one to the generic lowering,
        # which checks it, and a fused body that drops it fails the launch.
        args = tuple(torch.zeros(1, 256, 64, dtype=torch.bfloat16) for _ in range(3))
        with as_sm100():
            bound = cpu_bind(checked_attention, args)
            source = bound.to_code(bound.config_spec.default_config())
            self.assertFalse(bound.config_spec.cute_flash_search_enabled)
            self.assertRegex(
                source, r"cute\.testing\.assert_\(\w+, 'empty softmax row'\)"
            )
            self.assertIn("--enable-assertions", source)
            # Hide the assert from the search surface only: cute/backend.py's
            # own binding of the helper, which _reject_device_asserts calls,
            # still finds it.
            with patch(
                "helion._compiler.cute.analysis.attention_match.device_ir_has_device_assert",
                return_value=False,
            ):
                bound = cpu_bind(checked_attention, args)
                with self.assertRaisesRegex(
                    exc.BackendUnsupported, "hl.device_assert in a fused attention body"
                ):
                    bound.to_code(bound.config_spec.default_config())
            # In the KV loop, the generic lowering's lane schedule of the two
            # matmuls takes assignments only.
            bound = cpu_bind(checked_kv_attention, args)
            with self.assertRaisesRegex(exc.BackendUnsupported, "hl.device_assert"):
                bound.to_code(bound.config_spec.default_config())
            # The backward search surface leaves a kernel with one to the
            # generic lowering too.
            heads = [torch.zeros(1, 2, 256, 64, dtype=torch.bfloat16) for _ in range(4)]
            rows = torch.zeros(1, 2, 256)
            bwd_args = (*heads, rows, heads[0], rows, 0.125)
            bound = cpu_bind(checked_attention_backward, bwd_args)
            source = bound.to_code(bound.config_spec.default_config())
            self.assertIn("'nan')", source)

    def test_reshaped_condition(self):
        # flatten() merges the tiles' dims, whose padded lanes (17 of the 32
        # lanes of a (4, 8) tile over a (3, 5) x) no mask reaches any more;
        # reshape(2, rows) keeps a rows dim, which holds other lanes.
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "reshaped from tiles"):
            code(reshaped_condition, torch.zeros(3, 5))
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "reshaped from tiles"):
            code(moved_condition, torch.zeros(3, 2))
        # Also when an inner loop asserts what its outer one reshaped; a
        # reshape that keeps the tile's lanes ([rows, 128] into [rows, 2, 64])
        # is checked.
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "reshaped from tiles"):
            code(outer_reshape, torch.zeros(3, 5))
        # And when a loop carries it into the next iteration's assert, or a
        # loop or a branch returns it; another value a loop carries is not it.
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "reshaped from tiles"):
            code(carried_reshape, torch.zeros(3, 2))
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "reshaped from tiles"):
            code(returned_reshape, torch.zeros(3, 2))
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "reshaped from tiles"):
            code(branched_reshape, torch.zeros(3, 2), torch.ones(1))
        if _get_backend() == "triton":  # CuTe merges no tile dims.
            self.assertIn("'total')", code(other_carry, torch.zeros(3, 2)))
        if _get_backend() == "triton":  # CuTe splits no tile dim.
            source = code(kept_reshape, torch.zeros(1000, 128))
            self.assertIn("'nonpositive')", source)
        # A reduction of it reads the padding lanes the reshape moved.
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "reshaped from tiles"):
            code(reduced_reshape, torch.zeros(1000, 128))

    def test_computed_range(self):
        # hl.arange(3 * tile.block_size) runs over 128 lanes for a block of 32,
        # of which 96 hold elements: no constant count of padded lanes.
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "computed length"):
            code(computed_range, torch.zeros(1000))
        if _get_backend() == "triton":  # CuTe's needs a tile's extent.
            # 2 * tile.block_size and tile.block_size // 2, powers of two,
            # have none.
            self.assertIn("'range')", code(doubled_range, torch.zeros(1000)))
            self.assertIn("'range')", code(halved_range, torch.zeros(1000)))
        # hl.arange(100)[:, None] is [128, 1]: the mask of its 28 padded
        # lanes does not follow a permute.
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "permute"):
            code(permuted_range, torch.zeros(1000))
        # torch.stack([v, v, v]) of a tile's v pads its 3 to 4 lanes.
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "constant dim of 3"):
            code(stacked_condition, torch.ones(3, 5))
        # hl.arange(100).reshape(2, 50) puts lanes 100 to 113 in row 1.
        with self.assertRaisesRegex(exc.InvalidAPIUsage, "reshape"):
            code(split_range, torch.zeros(1000))
        if _get_backend() == "triton":  # CuTe's hl.arange needs a tile.
            # An empty condition holds: no assert, nor debug mode for it.
            source = code(empty_condition, torch.zeros(1000))
            self.assertNotIn("'empty')", source)
            self.assertNotIn("debug=True", source)
        if _get_backend() == "triton":
            # hl.arange(100).unsqueeze(0) is [1, 100], over 128 lanes too.
            source = code(reshaped_ranges, torch.zeros(1000))
            for message in ("unsqueezed", "viewed", "expanded"):
                (line,) = [
                    line for line in source.splitlines() if f"'{message}')" in line
                ]
                self.assertIn("< 100", line)

    @skipUnlessBackends(["cute"])
    def test_collective_matmul(self):
        # The collective MMA would stage the K loop's loads across an assert,
        # ahead of one that checks their indices: it refuses the region.
        bound = cpu_bind(
            checked_gathered_matmul,
            (
                torch.empty((193, 78), dtype=torch.float16),
                torch.empty((78, 70), dtype=torch.float16),
                torch.empty(130, dtype=torch.int32),
            ),
        )
        with self.assertRaisesRegex(exc.BackendUnsupported, "collective MMA"):
            bound.to_code(
                helion.Config(
                    block_sizes=[64, 32, 64],
                    num_threads=[4, 32, 1],
                    cute_vector_widths=[1, 1, 1],
                    cute_collective_mma=True,
                    cute_collective_copy="scalar",
                    loop_orders=[[0, 1]],
                )
            )

    @skipUnlessBackends(["triton"])
    def test_launch_option_tunable(self):
        # The asserts' debug=True wins over a _triton_config_debug tunable,
        # which would repeat the keyword or turn them off.
        bound = cpu_bind(debug_tunable, (torch.zeros(1000),))
        config = bound.config_spec.default_config()
        source = bound.to_code(config)
        (launch,) = [line for line in source.splitlines() if "_launcher(" in line]
        self.assertEqual(launch.count("debug="), 1)
        self.assertIn("debug=True", launch)

    def test_combine_function(self):
        # Rejected whatever the setting, so that compiling the checks out
        # does not hide it.
        for kernel in (assert_in_combine, unchecked_in_combine):
            with self.assertRaisesRegex(exc.InvalidAPIUsage, "combine function"):
                code(kernel, torch.zeros(64, 100))

    @skipUnlessBackends(["cute"])
    def test_no_split_k(self):
        # tcgen05_split_k runs a tile's body on several CTAs: the assert would
        # check each one's partial sum.
        bound = cpu_bind(
            matmul_checked,
            (torch.zeros(1024, 1024, dtype=torch.bfloat16),) * 2,
        )
        self.assertFalse(bound.config_spec.tcgen05.split_k_plain_body)


if __name__ == "__main__":
    unittest.main()
