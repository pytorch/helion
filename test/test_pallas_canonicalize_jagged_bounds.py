from __future__ import annotations

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import subtest

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipUnlessPallas
from helion._testing import xfailIfPallasInterpret
import helion.language as hl

_LOOP_TYPES = [
    "fori_loop",
    subtest(
        "emit_pipeline",
        name="emit_pipeline",
        decorators=[xfailIfPallasInterpret("BoundedSlice unsupported in interpret")],
    ),
]
_OFFSETS = [0, 13, 61, 192]
_REJECTED_KINDS = (
    "nonadjacent_offsets",
    "different_offsets",
    "shifted_store",
    "bare_tile_load",
    "coordinate_use",
    "aliasing_store",
    "loop_carried",
    "multi_store",
    "nested_store",
    "newaxis_store",
    "non_leading_load",
    "nonzero_col_begin",
    "outer_store",
    "partial_col_store",
    "scalar_col_load",
)


def _inputs(offsets, dtype=torch.bfloat16, n=128, k=128):
    torch.manual_seed(18)
    return (
        torch.randn(offsets[-1], k, dtype=dtype, device=DEVICE),
        torch.randn(k, n, dtype=dtype, device=DEVICE),
        torch.tensor(offsets, dtype=torch.int32, device=DEVICE),
    )


def _config(loop_type="fori_loop", blocks=(32, 128, 128), **kwargs):
    return helion.Config(block_sizes=list(blocks), pallas_loop_type=loop_type, **kwargs)


@helion.kernel(static_shapes=False)
def _jagged_gemm(a, b, offs):
    out = torch.empty(a.size(0), b.size(1), dtype=a.dtype, device=a.device)
    for g in hl.grid(offs.size(0) - 1):
        s, e = offs[g], offs[g + 1]
        m = e - s
        if m != 0:
            for tm, tn in hl.tile([m, b.size(1)]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(a.size(1)):
                    acc = torch.addmm(acc, a[s + tm.index, tk], b[tk, tn])
                out[s + tm.index, tn] = acc.to(out.dtype)
    return out


@helion.kernel(static_shapes=False)
def _rejected_gemm(a, b, offs, ends, kind: str):
    step = 2 if kind == "nonadjacent_offsets" else 1
    extra_rows = 1 if kind == "shifted_store" else 0
    col_begin = 16 if kind == "nonzero_col_begin" else 0
    col_end = b.size(1) // 2 if kind == "partial_col_store" else b.size(1)
    out = torch.zeros(a.size(0) + extra_rows, b.size(1), dtype=a.dtype, device=a.device)
    for g in hl.grid(offs.size(0) - step):
        s = offs[g]
        e = ends[g + 1] if kind == "different_offsets" else offs[g + step]
        carried = hl.zeros([1], dtype=torch.float32)
        if kind == "outer_store":
            out[s, :] = 0
        for tm, tn in hl.tile([0, col_begin], [e - s, col_end]):
            if kind == "loop_carried":
                carried = carried + 1.0
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            rows = s + tm.index + 1 if kind == "shifted_store" else s + tm.index
            for tk in hl.tile(a.size(1)):
                lhs = a[tm, tk] if kind == "bare_tile_load" else a[s + tm.index, tk]
                acc = torch.addmm(acc, lhs, b[tk, tn])
                if kind == "nested_store":
                    out[rows, tn] = acc.to(out.dtype)
            if kind == "coordinate_use":
                acc = acc + tm.begin
            if kind == "multi_store":
                out[rows, tn] = (2 * acc).to(out.dtype)
            if kind == "aliasing_store":
                a[rows, tn] = acc.to(a.dtype)
            elif kind == "newaxis_store":
                out[rows, None, tn] = acc[:, None, :].to(out.dtype)
            elif kind == "non_leading_load":
                out[rows, tn] = a[tn, rows]
            elif kind == "scalar_col_load":
                out[rows, tn] = a[rows, tn.begin][:, None]
            elif kind != "nested_store":
                out[rows, tn] = acc.to(out.dtype)
    return out


@onlyBackends(["pallas"])
@skipUnlessPallas("Pallas not available")
class TestPallasCanonicalizeJaggedBounds(TestCase):
    def setUp(self):
        super().setUp()
        # Clear cached BoundKernels on the module-level kernel so each test
        # re-traces and runs canonicalize_jagged_bounds from scratch even when
        # input shapes and dtypes match an earlier test.
        _jagged_gemm.reset()
        self.addCleanup(_jagged_gemm.reset)

    def _check(self, bound, config, args, expected, *, carry=True):
        (row_bid,) = bound.config_spec.block_sizes[0].block_ids
        self.assertEqual(
            row_bid in bound.host_function.device_ir.noncanonical_task_origin_block_ids,
            carry,
        )
        code = bound.to_code(config)
        self.assertEqual("carry" in code, carry)
        if carry:
            self.assertNotIn("one_hot", code)
        torch.testing.assert_close(bound.compile_config(config)(*args), expected)

    @parametrize("loop_type", _LOOP_TYPES)
    @parametrize("dtype", [torch.bfloat16, torch.float32])
    @parametrize("offsets", [[0, 32, 96, 192], _OFFSETS])
    def test_jagged_gemm_correctness(self, loop_type, dtype, offsets):
        args = _inputs(offsets, dtype=dtype)
        bound = _jagged_gemm.bind(args)
        self._check(bound, _config(loop_type), args, args[0] @ args[1])

    def test_nested_control_flow_in_k_loop(self):
        @helion.kernel(static_shapes=False)
        def fn(a, b, offs):
            out = torch.empty(a.size(0), b.size(1), dtype=a.dtype, device=a.device)
            for g in hl.grid(offs.size(0) - 1):
                s, e = offs[g], offs[1 + g]
                for tm, tn in hl.tile([e - s, b.size(1)]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(a.size(1)):
                        if e - s > 32:
                            acc = torch.addmm(acc, a[s + tm.index, tk], b[tk, tn])
                        else:
                            acc = torch.addmm(acc, a[tm.index + s, tk], 2 * b[tk, tn])
                    out[tm.index + s, tn] = acc.to(out.dtype)
            return out

        args = _inputs(_OFFSETS)
        expected = args[0] @ args[1]
        expected[:13] = args[0][:13] @ (2 * args[1])
        self._check(fn.bind(args), _config(), args, expected)

    def test_only_empty_group_checks_enclose_row_loop(self):
        @helion.kernel(static_shapes=False)
        def fn(a, offs, min_rows: hl.constexpr):
            out = torch.empty_like(a)
            for g in hl.grid(offs.size(0) - 1):
                s, e = offs[g], offs[g + 1]
                if e - s > min_rows:
                    for tm, tn in hl.tile([e - s, a.size(1)]):
                        out[s + tm.index, tn] = a[s + tm.index, tn] * 2
            return out

        a, _, offs = _inputs(_OFFSETS)
        for min_rows, rewritten in ((0, True), (8, False)):
            bound = fn.bind((a, offs, min_rows))
            (row_bid,) = bound.config_spec.block_sizes[0].block_ids
            device_ir = bound.host_function.device_ir
            self.assertEqual(
                row_bid in device_ir.noncanonical_task_origin_block_ids, rewritten
            )

    @parametrize("kind", _REJECTED_KINDS)
    def test_rejected_gemm(self, kind):
        a, b, offs = _inputs(_OFFSETS)
        _rejected_gemm.reset()
        args = (a, b, offs, offs.clone(), kind)
        bound = _rejected_gemm.bind(args)
        (row_bid,) = bound.config_spec.block_sizes[0].block_ids
        self.assertNotIn(
            row_bid,
            bound.host_function.device_ir.noncanonical_task_origin_block_ids,
        )
        if kind == "different_offsets":
            self._check(bound, _config(), args, a @ b, carry=False)

    def test_untiled_store_col(self):
        # A 1-D row loop whose store slices the full column dimension (:) has no
        # column tile block_id, so canonicalize_jagged_bounds rejects it and
        # plan_tiling lowers it via 1-D one_hot gather/scatter (carry=False).
        @helion.kernel(static_shapes=False)
        def untiled_store_col(a, offs):
            out = torch.empty_like(a)
            for g in hl.grid(offs.size(0) - 1):
                s, e = offs[g], offs[g + 1]
                for tm in hl.tile(e - s):
                    out[s + tm.index, :] = a[s + tm.index, :] * 2
            return out

        a, _, offs = _inputs(_OFFSETS)
        self._check(
            untiled_store_col.bind((a, offs)),
            _config(blocks=(32,)),
            (a, offs),
            a * 2,
            carry=False,
        )

    def test_multi_dim_load(self):
        # Indexing two dimensions by s + tm.index is rejected by
        # canonicalize_jagged_bounds; leaving both indirect indices in place
        # causes plan_tiling (called during bind()) to raise BackendUnsupported.
        @helion.kernel(static_shapes=False)
        def multi_dim_load(a, offs):
            out = torch.empty_like(a)
            for g in hl.grid(offs.size(0) - 1):
                s, e = offs[g], offs[g + 1]
                for tm, tn in hl.tile([e - s, a.size(1)]):
                    out[s + tm.index, tn] = a[s + tm.index, s + tm.index]
            return out

        a, _, offs = _inputs(_OFFSETS)
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "indirect access is unsupported"
        ):
            multi_dim_load.bind((a, offs))

    def test_two_row_loops(self):
        @helion.kernel(static_shapes=False)
        def fn(a, b, offs):
            out1 = torch.empty(a.size(0), b.size(1), dtype=a.dtype, device=a.device)
            out2 = torch.empty(a.size(0), b.size(1), dtype=a.dtype, device=a.device)
            for g in hl.grid(offs.size(0) - 1):
                s, e = offs[g], offs[g + 1]
                for tm, tn in hl.tile([e - s, b.size(1)]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(a.size(1)):
                        acc = torch.addmm(acc, a[s + tm.index, tk], b[tk, tn])
                    out1[s + tm.index, tn] = acc.to(out1.dtype)
                for tm, tn in hl.tile([e - s, b.size(1)]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(a.size(1)):
                        acc = torch.addmm(acc, a[s + tm.index, tk], b[tk, tn])
                    out2[s + tm.index, tn] = (2 * acc).to(out2.dtype)
            return out1, out2

        args = _inputs(_OFFSETS)
        bound, config = fn.bind(args), _config(blocks=(32, 128, 128, 32, 128, 128))
        code = bound.to_code(config)
        self.assertIn("carry", code)
        self.assertNotIn("one_hot", code)
        o1, o2 = bound.compile_config(config)(*args)
        ref = args[0] @ args[1]
        torch.testing.assert_close(o1, ref)
        torch.testing.assert_close(o2, (2 * ref.float()).to(ref.dtype))


instantiate_parametrized_tests(TestPallasCanonicalizeJaggedBounds)
