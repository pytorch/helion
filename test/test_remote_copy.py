"""Codegen tests for ``hl.start_async_remote_copy`` / ``wait_async_remote_copy``.

These assert on the *generated backend source* (``BoundKernel.to_code``) rather
than executing, so they need no TPU/GPU: they lock in the Pallas
``make_async_remote_copy`` lowering for the asymmetric, runtime-routed remote
copy that fused comms+compute kernels (reduce-scatter / all-to-all) rely on,
plus the trace-time src<->dst contract.
"""

from __future__ import annotations

import re

import torch

import helion
from helion import exc
from helion._testing import TestCase
import helion.language as hl

# --- kernels (module-level so inspect.getsource works during compile) --------


@helion.kernel(backend="pallas", distributed=[8], config=helion.Config())
def _pallas_scatter(src, out, dest, wpos, ws: hl.constexpr):
    """Asymmetric, runtime-routed: push src[i] -> out[wpos[i]] on chip dest[i]."""
    m = src.shape[0]
    for i in hl.grid(m):
        op = hl.start_async_remote_copy(src, [i], dest[i], dst=out, dst_index=[wpos[i]])
        op.wait()
    return out


@helion.kernel(backend="pallas", distributed=[8], config=helion.Config())
def _pallas_symmetric(buf, ws: hl.constexpr):
    """Symmetric ring-all-gather form: push buf[i] -> buf[i] on constant peer 3."""
    m = buf.shape[0]
    for i in hl.grid(m):
        op = hl.start_async_remote_copy(buf, [i], 3)
        op.wait()
    return buf


def _scatter_args(
    src_dtype=torch.float32, dst_dtype=None, h=128, dst_h=None, src_stride0=None
):
    """Build (src, out, dest, wpos, ws) for the scatter kernels."""
    dst_dtype = dst_dtype or src_dtype
    dst_h = dst_h or h
    m, s = 8, 16
    if src_stride0 is not None:
        # Non-contiguous leading dim: allocate wider then slice, so
        # stride(0) == src_stride0 != h while each row stays contiguous.
        src = torch.zeros(m, src_stride0, dtype=src_dtype)[:, :h]
    else:
        src = torch.zeros(m, h, dtype=src_dtype)
    return (
        src,
        torch.zeros(s, dst_h, dtype=dst_dtype),
        torch.zeros(m, dtype=torch.int32),
        torch.zeros(m, dtype=torch.int32),
        hl.constexpr(4),
    )


def _code(kernel, args):
    return kernel.bind(args).to_code(helion.Config())


def _remote_copy_line(code):
    return next(
        ln.strip() for ln in code.splitlines() if "make_async_remote_copy" in ln
    )


class TestRemoteCopyPallasCodegen(TestCase):
    def test_asymmetric_emits_distinct_src_and_dst_refs(self):
        line = _remote_copy_line(_code(_pallas_scatter, _scatter_args()))
        refs = re.findall(r"(\w+)\.at\[", line)
        self.assertEqual(len(refs), 2, line)
        self.assertNotEqual(
            refs[0], refs[1], f"src and dst should be distinct refs: {line}"
        )
        self.assertIn("device_id_type=pl.DeviceIdType.LOGICAL", line)

    def test_symmetric_reuses_same_ref_backward_compat(self):
        # The 3-arg (dst omitted) form must degrade to the old ring-all-gather
        # lowering: the same buffer appears as both src and dst.
        line = _remote_copy_line(
            _code(_pallas_symmetric, (torch.zeros(8, 128), hl.constexpr(4)))
        )
        refs = re.findall(r"(\w+)\.at\[", line)
        self.assertEqual(len(refs), 2, line)
        self.assertEqual(
            refs[0], refs[1], f"symmetric form should reuse one ref: {line}"
        )

    def test_device_id_is_runtime_scalar(self):
        # dest[i] is data-dependent, so device_id must be an SSA name, not a
        # baked-in constant.
        line = _remote_copy_line(_code(_pallas_scatter, _scatter_args()))
        m = re.search(r"device_id=(\w+)", line)
        self.assertIsNotNone(m, line)
        self.assertFalse(m.group(1).isdigit(), f"expected runtime device_id: {line}")

    def test_device_id_constant_is_inlined(self):
        line = _remote_copy_line(
            _code(_pallas_symmetric, (torch.zeros(8, 128), hl.constexpr(4)))
        )
        self.assertRegex(line, r"device_id=3\b")

    def test_start_and_wait_share_op_var(self):
        code = _code(_pallas_scatter, _scatter_args())
        start = re.search(r"(\w+) = pltpu\.make_async_remote_copy", code)
        self.assertIsNotNone(start, code)
        self.assertIn(f"{start.group(1)}.start()", code)
        self.assertIn(f"{start.group(1)}.wait()", code)


class TestRemoteCopyContract(TestCase):
    # The src<->dst contract is validated in prepare_args (before any backend
    # codegen), so it is backend-agnostic; exercise it via the Pallas kernel.
    def test_dtype_mismatch_raises(self):
        with self.assertRaises(exc.TypeInferenceError):
            _code(
                _pallas_scatter,
                _scatter_args(src_dtype=torch.float32, dst_dtype=torch.bfloat16),
            )

    def test_element_count_mismatch_raises(self):
        with self.assertRaises(exc.TypeInferenceError):
            _code(_pallas_scatter, _scatter_args(h=128, dst_h=64))
