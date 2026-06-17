"""Tests for the ``compact_worklist`` Pallas loop type.

Layered bottom-up so each layer is tested before the next depends on it:

1. Builder unit tests (this file, ``TestCompactWorklistBuilder``) -- pure JAX,
   CPU-only, no Helion compiler.  Assert :func:`flatten_worklist` matches a slow
   Python reference across length distributions and edge cases.

Later layers (resolver generality, codegen goldens, numerics) are added as the
corresponding checkpoints land.
"""

from __future__ import annotations

import ast
import unittest

import torch

import helion
from helion import exc
from helion._compiler.pallas.compact_worklist import Axis
from helion._compiler.pallas.compact_worklist import CompactWorklistPlan
from helion._compiler.pallas.compact_worklist import TensorPolicy
from helion._compiler.pallas.compact_worklist import _classify_tensor_policies
from helion._compiler.pallas.compact_worklist import _validate_host_bounds
from helion._compiler.pallas.compact_worklist import detect_compact_worklist_plan
from helion._compiler.pallas.compact_worklist import metadata_arg_names
from helion._compiler.pallas.compact_worklist import render_build_worklist
from helion._compiler.pallas.compact_worklist import resolve_for_worklist
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfPallasInterpret
from helion._testing import skipUnlessPallas
import helion.language as hl

try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    import numpy as np

    from helion.runtime.compact_worklist import flatten_worklist
    from helion.runtime.compact_worklist import packed_upper_bound

    HAS_JAX = True
except ImportError:  # pragma: no cover - jax optional
    HAS_JAX = False


def _expr(src: str) -> ast.AST:
    return ast.parse(src, mode="eval").body


def _python_flatten_reference(base, length, block, *, dep_base=None, dep_len=None):
    """Slow, obviously-correct Python reference for :func:`flatten_worklist`.

    Returns plain Python lists over the *valid* ``[0, num_work)`` range only
    (no padding); the builder's padded tail is never indexed at runtime.
    """
    owner_ids = []
    tile_starts = []
    tile_extents = []
    range_start = []
    range_len = []
    for p in range(len(base)):
        cnt = -(-int(length[p]) // block)  # cdiv, NOT clamped up
        for t in range(cnt):
            owner_ids.append(p)
            tile_starts.append(int(base[p]) + t * block)
            tile_extents.append(min(block, int(length[p]) - t * block))
            if dep_base is not None:
                range_start.append(int(dep_base[p]))
                range_len.append(int(dep_len[p]))
    return {
        "owner_ids": owner_ids,
        "tile_starts": tile_starts,
        "tile_extents": tile_extents,
        "range_start": range_start if dep_base is not None else None,
        "range_len": range_len if dep_len is not None else None,
        "num_work": len(owner_ids),
    }


@onlyBackends(["pallas"])
@unittest.skipUnless(HAS_JAX, "jax not available")
class TestCompactWorklistBuilder(unittest.TestCase):
    """Unit tests for :func:`flatten_worklist`."""

    def _check(self, lengths, block, *, dep_lengths=None):
        """Build with JAX and assert the valid range matches the reference."""
        lengths = list(lengths)
        # Owners are packed contiguously, so base = exclusive cumsum of lengths.
        base = [0]
        for L in lengths[:-1]:
            base.append(base[-1] + L)
        total = sum(lengths)
        num_owners = len(lengths)
        UPPER = packed_upper_bound(total, num_owners, block)

        dep_base = None
        if dep_lengths is not None:
            dep_lengths = list(dep_lengths)
            dep_base = [0]
            for L in dep_lengths[:-1]:
                dep_base.append(dep_base[-1] + L)

        meta = flatten_worklist(
            jnp.asarray(base, jnp.int32),
            jnp.asarray(lengths, jnp.int32),
            block,
            UPPER,
            dep_base=None if dep_base is None else jnp.asarray(dep_base, jnp.int32),
            dep_len=None
            if dep_lengths is None
            else jnp.asarray(dep_lengths, jnp.int32),
        )
        ref = _python_flatten_reference(
            base, lengths, block, dep_base=dep_base, dep_len=dep_lengths
        )

        num_work = int(meta.num_work)
        self.assertEqual(num_work, ref["num_work"])
        self.assertLessEqual(num_work, UPPER)
        # Every metadata array has the static UPPER shape.
        self.assertEqual(meta.owner_ids.shape, (UPPER,))
        self.assertEqual(meta.tile_starts.shape, (UPPER,))
        self.assertEqual(meta.tile_extents.shape, (UPPER,))

        # The valid [0, num_work) prefix matches the reference exactly.
        np.testing.assert_array_equal(
            np.asarray(meta.owner_ids)[:num_work], ref["owner_ids"]
        )
        np.testing.assert_array_equal(
            np.asarray(meta.tile_starts)[:num_work], ref["tile_starts"]
        )
        np.testing.assert_array_equal(
            np.asarray(meta.tile_extents)[:num_work], ref["tile_extents"]
        )
        if dep_lengths is not None:
            np.testing.assert_array_equal(
                np.asarray(meta.range_start)[:num_work], ref["range_start"]
            )
            np.testing.assert_array_equal(
                np.asarray(meta.range_len)[:num_work], ref["range_len"]
            )
        else:
            self.assertIsNone(meta.range_start)
            self.assertIsNone(meta.range_len)
        return meta, num_work

    def test_length_distributions(self):
        rng = np.random.default_rng(29)
        cases = [
            ("uniform", [256, 256, 256, 256]),
            ("ramp", [1, 128, 256, 384, 512]),
            ("descending", [512, 256, 128, 1]),
            ("random", rng.integers(1, 600, size=16).tolist()),
            ("all_partial", [3, 7, 1, 5]),
            ("single_owner", [500]),
            ("sparse", [0, 0, 300, 0]),
        ]
        for name, lengths in cases:
            with self.subTest(name=name):
                self._check(lengths, block=128)

    def test_num_work_zero(self):
        # total_compact == 0 => num_work == 0; valid range is empty.
        _, num_work = self._check([0, 0, 0, 0], block=128)
        self.assertEqual(num_work, 0)

    def test_with_dep_range(self):
        self._check([300, 100, 256], block=128, dep_lengths=[512, 64, 200])

    def test_empty_dep_stored_raw(self):
        # An empty ordered range (kv_len == 0) must be stored RAW (not clamped
        # up to 1) -- otherwise the inner fori_loop processes phantom KV rows.
        meta, num_work = self._check(
            [256, 128, 256], block=128, dep_lengths=[0, 100, 0]
        )
        rs = np.asarray(meta.range_len)[:num_work]
        # owners 0 and 2 have kv_len == 0; their work items must carry raw 0.
        owners = np.asarray(meta.owner_ids)[:num_work]
        self.assertTrue((rs[owners == 0] == 0).all())
        self.assertTrue((rs[owners == 2] == 0).all())
        self.assertTrue((rs[owners == 1] == 100).all())

    def test_upper_bound_is_megablocks(self):
        self.assertEqual(packed_upper_bound(1000, 4, 128), -(-1000 // 128) + 3)
        self.assertEqual(packed_upper_bound(0, 4, 128), 3)


@onlyBackends(["pallas"])
class TestResolveForWorklist(unittest.TestCase):
    """resolve_for_worklist generality + leak-check.

    Proves "no +1 pattern-match": offsets[g+a] for arbitrary a, distinct
    begin/end tensors, and affine g*S forms all resolve, while a device/
    body-local bound raises exc.InvalidConfig.
    """

    def _resolve(self, src, subs, leak_ok):
        return ast.unparse(resolve_for_worklist(_expr(src), subs, leak_ok=set(leak_ok)))

    def test_supported_expressions(self):
        cases = [
            ("offsets[g]", {"offsets"}, "offsets[work_g]"),
            ("offsets[g + 1]", {"offsets"}, "offsets[work_g + 1]"),
            ("offsets[g + 2]", {"offsets"}, "offsets[work_g + 2]"),
            ("lo[g]", {"lo", "hi"}, "lo[work_g]"),
            ("hi[g]", {"lo", "hi"}, "hi[work_g]"),
            # base-in-index style affine bound; S is a host constant.
            ("g * S", {"S"}, "work_g * S"),
            ("g * S + S", {"S"}, "work_g * S + S"),
        ]
        for src, leak_ok, expected in cases:
            with self.subTest(src=src):
                self.assertEqual(
                    self._resolve(src, {"g": "work_g"}, leak_ok),
                    expected,
                )

    def test_does_not_mutate_input(self):
        e = _expr("offsets[g + 1]")
        resolve_for_worklist(e, {"g": "work_g"}, leak_ok={"offsets"})
        self.assertEqual(ast.unparse(e), "offsets[g + 1]")

    def test_unsupported_leaks_rejected(self):
        cases = [
            "offsets[g] + tmp",  # tmp is neither owner coordinate nor host tensor.
            "dev_val[g]",
        ]
        for src in cases:
            with self.subTest(src=src), self.assertRaises(exc.InvalidConfig):
                resolve_for_worklist(_expr(src), {"g": "work_g"}, leak_ok={"offsets"})


def _axis(kind, loop_var, block_id=0):
    return Axis(
        kind=kind,
        block_id=block_id,
        loop_var=loop_var,
        base=_expr("0"),
        length=_expr("1"),
        block_size_var="1" if kind == "owner_grid" else "",
    )


def _plan(*, ordered, owner_indexed):
    axes = [_axis("owner_grid", "seq", 0), _axis("compact_tile", "tile_q", 1)]
    if ordered:
        axes.append(_axis("ordered", "tile_kv", 2))
    policies = [TensorPolicy(arg_name="q", kind="compact_aligned_load")]
    if owner_indexed:
        policies.append(TensorPolicy(arg_name="k", kind="owner_indexed"))
    return CompactWorklistPlan(
        axes=tuple(axes), tensor_policies=tuple(policies), upper_bound_expr=""
    )


@onlyBackends(["pallas"])
class TestMetadataArgNames(unittest.TestCase):
    """metadata_arg_names is policy-derived."""

    def test_args_by_shape(self):
        cases = [
            (
                "dense_kv",
                _plan(ordered=False, owner_indexed=True),
                ["work_seq", "q_begin", "q_extent"],
            ),
            (
                "fully_jagged",
                _plan(ordered=True, owner_indexed=False),
                ["work_seq", "q_begin", "q_extent", "kv_begin", "kv_len"],
            ),
        ]
        for name, plan, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(metadata_arg_names(plan), expected)
                self.assertEqual(len(metadata_arg_names(plan)), len(expected))


@onlyBackends(["pallas"])
class TestRenderBuilderParams(unittest.TestCase):
    """render_build_worklist param collection (feedback #3)."""

    def test_num_owners_free_name_included(self):
        # num_owners_expr="B" references a host scalar not in base/length; the
        # builder must take B as a parameter (else jnp.arange(B) is undefined).
        plan = CompactWorklistPlan(
            axes=(
                _axis("owner_grid", "seq", 0),
                Axis(
                    kind="compact_tile",
                    block_id=1,
                    loop_var="tile_q",
                    base=_expr("off[seq]"),
                    length=_expr("off[seq + 1] - off[seq]"),
                    block_size_var="",
                ),
            ),
            tensor_policies=(TensorPolicy(arg_name="q", kind="compact_aligned_load"),),
            upper_bound_expr="",
            num_owners_expr="B",
        )
        src, offset_params = render_build_worklist(
            plan, block_expr="8", upper_expr="16"
        )
        self.assertIn("off", offset_params)
        self.assertIn("B", offset_params)
        self.assertIn("jnp.arange(B", src)


# ---------------------------------------------------------------------------
# Detection + autotuner-gating integration (trace real kernels; CPU ok)
# ---------------------------------------------------------------------------


@helion.kernel(backend="pallas", static_shapes=True)
def _dense_kv_kernel(q, k, v, q_offsets):
    num_sequences = q_offsets.size(0) - 1
    out = torch.empty_like(q)
    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]
        k_blk = k[seq_idx, :, :, :].transpose(0, 1)
        v_blk = v[seq_idx, :, :, :].transpose(0, 1)
        for tile_q in hl.tile(q_start, q_end):
            q_blk = q[tile_q, :, :].transpose(0, 1)
            scores = torch.bmm(q_blk, k_blk.transpose(-2, -1))
            acc = torch.bmm(scores.to(v.dtype), v_blk)
            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _fully_jagged_kernel(q, k, v, q_offsets, kv_offsets):
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = q_offsets.size(0) - 1
    out = torch.empty_like(q)
    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]
        k_start = kv_offsets[seq_idx]
        k_end = kv_offsets[seq_idx + 1]
        for tile_q in hl.tile(q_start, q_end):
            q_blk = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([H, tile_q, D], dtype=torch.float32)
            for tile_kv in hl.tile(k_start, k_end):
                k_blk = k[tile_kv, :, :].transpose(0, 1)
                v_blk = v[tile_kv, :, :].transpose(0, 1)
                scores = torch.bmm(q_blk, k_blk.transpose(-2, -1))
                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)
            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)
    return out


@helion.kernel(backend="pallas", static_shapes=True)
def _add_kernel(x, y):
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _offsets(lengths):
    lengths = torch.tensor(lengths, dtype=torch.int32)
    return torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(lengths, 0).to(torch.int32)]
    )


@onlyBackends(["pallas"])
class TestDetectAndGating(unittest.TestCase):
    """detect_compact_worklist_plan + autotuner gating over real traces.

    Runs on CPU: ``kernel.bind`` traces with fake tensors, no device needed.
    """

    def _dense_kv_plan(self):
        qo = _offsets([16, 16, 16, 16])
        lq = int(qo[-1])
        q = torch.randn(lq, 2, 8)
        k = torch.randn(4, 16, 2, 8)
        v = torch.randn(4, 16, 2, 8)
        bk = _dense_kv_kernel.bind((q, k, v, qo))
        with bk.env:
            return detect_compact_worklist_plan(bk.host_function)

    def _fully_jagged_plan(self):
        qo = _offsets([16, 16, 16, 16])
        lq = int(qo[-1])
        q = torch.randn(lq, 2, 8)
        k = torch.randn(lq, 2, 8)
        v = torch.randn(lq, 2, 8)
        bk = _fully_jagged_kernel.bind((q, k, v, qo, qo))
        with bk.env:
            return detect_compact_worklist_plan(bk.host_function)

    def test_dense_kv_axes_and_policies(self):
        plan = self._dense_kv_plan()
        self.assertEqual(
            [(a.kind, a.loop_var) for a in plan.axes],
            [("owner_grid", "seq_idx"), ("compact_tile", "tile_q")],
        )
        self.assertIsNone(plan.ordered_axis)
        policies = {p.arg_name: p.kind for p in plan.tensor_policies}
        self.assertEqual(policies["q"], "compact_aligned_load")
        self.assertEqual(policies["out"], "compact_exact_store")
        self.assertEqual(policies["k"], "owner_indexed")
        self.assertEqual(policies["v"], "owner_indexed")
        # q_offsets is a builder input, not a device tensor policy.
        self.assertNotIn("q_offsets", policies)

    def test_dense_kv_bounds_general(self):
        # Proves capture is general (no +1 pattern-match): length is end - begin.
        plan = self._dense_kv_plan()
        compact = plan.compact_axis
        self.assertEqual(ast.unparse(compact.base), "q_offsets[seq_idx]")
        self.assertEqual(
            ast.unparse(compact.length),
            "q_offsets[seq_idx + 1] - q_offsets[seq_idx]",
        )

    def test_dense_kv_metadata_args(self):
        # Owner_indexed tensors consume owner_ids -> 3 scalar-prefetch args.
        self.assertEqual(
            metadata_arg_names(self._dense_kv_plan()),
            ["work_seq", "q_begin", "q_extent"],
        )

    def test_fully_jagged_ordered_axis(self):
        plan = self._fully_jagged_plan()
        self.assertEqual(
            [(a.kind, a.loop_var) for a in plan.axes],
            [
                ("owner_grid", "seq_idx"),
                ("compact_tile", "tile_q"),
                ("ordered", "tile_kv"),
            ],
        )
        ordered = plan.ordered_axis
        self.assertIsNotNone(ordered)
        self.assertEqual(ast.unparse(ordered.base), "kv_offsets[seq_idx]")
        self.assertEqual(
            ast.unparse(ordered.length),
            "kv_offsets[seq_idx + 1] - kv_offsets[seq_idx]",
        )

    def test_fully_jagged_metadata_args(self):
        # owner_ids is always included (the owner-grid prologue q_offsets[seq]
        # is not DCE'd, so the owner pid must be a valid owner index).
        self.assertEqual(
            metadata_arg_names(self._fully_jagged_plan()),
            ["work_seq", "q_begin", "q_extent", "kv_begin", "kv_len"],
        )

    def test_gating_offers_compact_worklist_for_jagged(self):
        qo = _offsets([16, 16, 16, 16])
        lq = int(qo[-1])
        bk = _dense_kv_kernel.bind(
            (
                torch.randn(lq, 2, 8),
                torch.randn(4, 16, 2, 8),
                torch.randn(4, 16, 2, 8),
                qo,
            )
        )
        fields = bk.env.config_spec._flat_fields()
        choices = fields["pallas_loop_type"].choices
        self.assertIn("compact_worklist", choices)
        self.assertEqual(list(bk.env.config_spec.grid_block_ids), [0])

    def test_gating_absent_for_non_jagged(self):
        bk = _add_kernel.bind((torch.randn(64, 64), torch.randn(64, 64)))
        fields = bk.env.config_spec._flat_fields()
        # No inner Pallas loops -> no pallas_loop_type field at all.
        self.assertNotIn("pallas_loop_type", fields)


@onlyBackends(["pallas"])
@unittest.skipUnless(HAS_JAX, "jax not available")
class TestBuildWorklistRender(unittest.TestCase):
    """The generated jnp _build_worklist.

    Renders the builder from a real plan, execs it under JAX (CPU), and checks
    it matches the library flatten_worklist -- proving the resolver output drops
    straight into a runnable jnp builder with a traced num_work.
    """

    BLOCK = 32

    def _render_and_exec(self, plan, offset_arrays, upper):
        src, offset_params = render_build_worklist(
            plan, block_expr=str(self.BLOCK), upper_expr=str(upper)
        )
        namespace: dict = {}
        exec(compile(src, "<build_worklist>", "exec"), namespace)
        builder = namespace["_build_worklist"]
        return src, offset_params, builder(*offset_arrays)

    def test_dense_kv_builder_matches_flatten(self):
        qlens = [20, 40, 5, 33]
        qo = _offsets(qlens)
        lq = int(qo[-1])
        bk = _dense_kv_kernel.bind(
            (
                torch.randn(lq, 2, 8),
                torch.randn(4, 16, 2, 8),
                torch.randn(4, 16, 2, 8),
                qo,
            )
        )
        with bk.env:
            plan = detect_compact_worklist_plan(bk.host_function)
        upper = packed_upper_bound(lq, 4, self.BLOCK)
        qo_j = jnp.asarray(qo.numpy())
        src, offset_params, meta = self._render_and_exec(plan, (qo_j,), upper)

        self.assertEqual(offset_params, ["q_offsets"])
        self.assertIn("flatten_worklist(", src)
        self.assertIn("jnp.arange(q_offsets.shape[0] - 1", src)
        # No dependent range for dense-KV.
        self.assertNotIn("dep_base", src)
        self.assertIsNone(meta.range_start)

        ref = flatten_worklist(qo_j[:-1], qo_j[1:] - qo_j[:-1], self.BLOCK, upper)
        nw = int(meta.num_work)
        self.assertEqual(nw, int(ref.num_work))
        self.assertEqual(nw, sum(-(-n // self.BLOCK) for n in qlens))
        np.testing.assert_array_equal(
            np.asarray(meta.tile_starts)[:nw], np.asarray(ref.tile_starts)[:nw]
        )
        np.testing.assert_array_equal(
            np.asarray(meta.tile_extents)[:nw], np.asarray(ref.tile_extents)[:nw]
        )

    def test_fully_jagged_builder_has_raw_dep_range(self):
        qlens = [20, 40, 5, 33]
        kvlens = [16, 0, 30, 8]  # owner 1 has an empty KV range
        qo = _offsets(qlens)
        kvo = _offsets(kvlens)
        lq = int(qo[-1])
        lkv = int(kvo[-1])
        bk = _fully_jagged_kernel.bind(
            (
                torch.randn(lq, 2, 8),
                torch.randn(lkv, 2, 8),
                torch.randn(lkv, 2, 8),
                qo,
                kvo,
            )
        )
        with bk.env:
            plan = detect_compact_worklist_plan(bk.host_function)
        upper = packed_upper_bound(lq, 4, self.BLOCK)
        qo_j = jnp.asarray(qo.numpy())
        kvo_j = jnp.asarray(kvo.numpy())
        src, offset_params, meta = self._render_and_exec(plan, (qo_j, kvo_j), upper)

        self.assertEqual(offset_params, ["q_offsets", "kv_offsets"])
        self.assertIn("dep_base=dep_base, dep_len=dep_len", src)
        self.assertIsNotNone(meta.range_start)

        nw = int(meta.num_work)
        owners = np.asarray(meta.owner_ids)[:nw]
        rl = np.asarray(meta.range_len)[:nw]
        # Empty KV stored RAW (not clamped up), nonzero preserved.
        self.assertTrue((rl[owners == 1] == 0).all())
        self.assertTrue((rl[owners == 2] == 30).all())


# ---------------------------------------------------------------------------
# Review-feedback hardening
# ---------------------------------------------------------------------------


@helion.kernel(backend="pallas", static_shapes=True)
def _dual_offset_kernel(q, lo, hi):
    # Distinct begin/end tensors (lo[g]/hi[g]); arrays have B entries (not B+1),
    # so num_owners must be lo.shape[0], NOT lo.shape[0] - 1.
    out = torch.empty_like(q)
    for seq in hl.grid(lo.size(0)):
        for tile in hl.tile(lo[seq], hi[seq]):
            out[tile, :, :] = q[tile, :, :] * 2.0
    return out


@onlyBackends(["pallas"])
@unittest.skipUnless(HAS_JAX, "jax not available")
class TestBuilderDistinctTensors(unittest.TestCase):
    """Builder render with distinct begin/end tensors (feedback #2)."""

    BLOCK = 8

    def test_lo_hi_builder_includes_both_args(self):
        # Render must accept distinct begin/end tensors (lo[g]/hi[g]) -- this is
        # the render-layer generality of feedback #2.  Detection now REJECTS this
        # idiom for store safety (see TestUpperBoundPacking), so the plan is built
        # manually here to keep the render path covered for a future non-packed PR.
        lo = torch.tensor([0, 30, 70], dtype=torch.int32)
        hi = torch.tensor([20, 60, 99], dtype=torch.int32)
        plan = CompactWorklistPlan(
            axes=(
                _axis("owner_grid", "seq", 0),
                Axis(
                    kind="compact_tile",
                    block_id=1,
                    loop_var="tile",
                    base=_expr("lo[seq]"),
                    length=_expr("hi[seq] - lo[seq]"),
                    block_size_var="",
                ),
            ),
            tensor_policies=(TensorPolicy(arg_name="q", kind="compact_aligned_load"),),
            upper_bound_expr="",
            num_owners_expr="lo.shape[0]",
        )
        # num_owners from the hl.grid bound (lo.size(0) -> lo.shape[0]), no -1.
        self.assertEqual(plan.num_owners_expr, "lo.shape[0]")
        upper = packed_upper_bound(120, 3, self.BLOCK)
        src, offset_params = render_build_worklist(
            plan, block_expr=str(self.BLOCK), upper_expr=str(upper)
        )
        # Both lo and hi must be parameters (hi was missing before the fix).
        self.assertEqual(offset_params, ["lo", "hi"])
        self.assertIn("jnp.arange(lo.shape[0]", src)

        namespace: dict = {}
        exec(compile(src, "<bw>", "exec"), namespace)
        meta = namespace["_build_worklist"](
            jnp.asarray(lo.numpy()), jnp.asarray(hi.numpy())
        )
        lengths = (hi - lo).tolist()
        ref = flatten_worklist(
            jnp.asarray(lo.numpy()),
            jnp.asarray((hi - lo).numpy()),
            self.BLOCK,
            upper,
        )
        nw = int(meta.num_work)
        self.assertEqual(nw, int(ref.num_work))
        self.assertEqual(nw, sum(-(-n // self.BLOCK) for n in lengths))
        np.testing.assert_array_equal(
            np.asarray(meta.tile_starts)[:nw], np.asarray(ref.tile_starts)[:nw]
        )


class _StubParams:
    def __init__(self, arguments):
        self.arguments = arguments


class _StubHostFn:
    """Minimal HostFunction stand-in for _validate_host_bounds unit tests."""

    def __init__(self, arg_tensors):
        # arg_tensors: dict[name -> torch.Tensor] (1-D = offsets, >1-D = data).
        self.args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=n) for n in arg_tensors],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        self.local_types = None
        self.params = _StubParams(dict(arg_tensors))


def _bound_axis(base_src, length_src):
    return Axis(
        kind="compact_tile",
        block_id=1,
        loop_var="tile_q",
        base=_expr(base_src),
        length=_expr(length_src),
        block_size_var="",
    )


@onlyBackends(["pallas"])
class TestHostBoundValidation(unittest.TestCase):
    """detect rejects non-host-evaluable / non-offsets bounds (feedback #3, #4)."""

    def _host_fn(self):
        return _StubHostFn(
            {
                "q": torch.empty(64, 2, 8),  # 3-D data tensor
                "q_offsets": torch.empty(5, dtype=torch.int32),  # 1-D offsets
            }
        )

    _OWNERS = "q_offsets.shape[0] - 1"

    def test_host_offsets_tensor_accepted(self):
        host_fn = self._host_fn()
        axes = [_bound_axis("q_offsets[seq]", "q_offsets[seq + 1] - q_offsets[seq]")]
        _validate_host_bounds(host_fn, axes, "seq", self._OWNERS)  # no raise

    def test_invalid_bounds_rejected(self):
        cases = [
            (
                "unknown_device_value",
                self._host_fn(),
                [_bound_axis("q_offsets[seq]", "dev[seq] - q_offsets[seq]")],
                self._OWNERS,
            ),
            (
                "multidim_data_tensor",
                self._host_fn(),
                [_bound_axis("q_offsets[seq]", "q[seq]")],
                self._OWNERS,
            ),
            (
                "scalar_num_owners",
                _StubHostFn({"q_offsets": torch.empty(5, dtype=torch.int32), "B": 4}),
                [_bound_axis("q_offsets[seq]", "q_offsets[seq + 1] - q_offsets[seq]")],
                "B",
            ),
            (
                "scalar_bound",
                _StubHostFn({"q_offsets": torch.empty(5, dtype=torch.int32), "S": 16}),
                [_bound_axis("q_offsets[seq]", "S")],
                self._OWNERS,
            ),
        ]
        for name, host_fn, axes, owners in cases:
            with self.subTest(name=name), self.assertRaises(exc.InvalidConfig):
                _validate_host_bounds(host_fn, axes, "seq", owners)


@onlyBackends(["pallas"])
class TestPolicyClassification(unittest.TestCase):
    """Unsupported compact indexing is rejected, not silently dropped."""

    def _grid_loop(self, src):
        return ast.parse(src).body[0]

    def test_leading_dim_classified(self):
        loop = self._grid_loop(
            "for seq in g:\n"
            "    for tile_q in t:\n"
            "        out[tile_q, :, :] = q[tile_q, :, :]\n"
        )
        policies = {
            p.arg_name: p.kind
            for p in _classify_tensor_policies(loop, "seq", "tile_q", None, set())
        }
        self.assertEqual(policies["q"], "compact_aligned_load")
        self.assertEqual(policies["out"], "compact_exact_store")

    def test_non_leading_compact_dim_rejected(self):
        loop = self._grid_loop(
            "for seq in g:\n"
            "    for tile_q in t:\n"
            "        out[:, tile_q, :] = q[:, tile_q, :]\n"
        )
        with self.assertRaises(exc.InvalidConfig):
            _classify_tensor_policies(loop, "seq", "tile_q", None, set())


@onlyBackends(["pallas"])
class TestNoSilentFallback(unittest.TestCase):
    """compact_worklist routes to the compact launcher, never the default."""

    def _kernel(self, fn, **cfg):
        return helion.kernel(
            fn, config=helion.Config(**cfg), static_shapes=True, backend="pallas"
        )

    def test_matching_kernel_generates_compact(self):
        def fn(q, k, v, q_offsets):
            out = torch.empty_like(q)
            for seq_idx in hl.grid(q_offsets.size(0) - 1):
                k_blk = k[seq_idx, :, :, :].transpose(0, 1)
                v_blk = v[seq_idx, :, :, :].transpose(0, 1)
                for tile_q in hl.tile(q_offsets[seq_idx], q_offsets[seq_idx + 1]):
                    q_blk = q[tile_q, :, :].transpose(0, 1)
                    scores = torch.bmm(q_blk, k_blk.transpose(-2, -1))
                    out[tile_q, :, :] = (
                        torch.bmm(scores.to(v.dtype), v_blk)
                        .transpose(0, 1)
                        .to(out.dtype)
                    )
            return out

        qo = _offsets([12, 20, 5, 30])
        lq = int(qo[-1])
        args = (
            torch.randn(lq, 2, 8),
            torch.randn(4, 16, 2, 8),
            torch.randn(4, 16, 2, 8),
            qo,
        )
        kernel = self._kernel(fn, block_sizes=[8], pallas_loop_type="compact_worklist")
        bound = kernel.bind(args)
        code = bound.to_triton_code(
            helion.Config(block_sizes=[8], pallas_loop_type="compact_worklist")
        )
        # Routes to the compact launcher with the worklist builder + scalar
        # prefetch, NOT the default unroll launcher.
        self.assertIn("_default_pallas_compact_worklist_launcher", code)
        self.assertIn("def _build_worklist(", code)
        # Offsets arg index is non-empty (q_offsets feeds the builder).
        self.assertRegex(code, r"_compact_offset_arg_indices=\[\d")
        self.assertIn("_compact_num_scalar_prefetch=3", code)
        self.assertIn("_wid = pl.program_id(0)", code)
        self.assertIn("work_seq_ref[_wid]", code)
        self.assertNotIn("_default_pallas_launcher", code)

    def test_unsupported_kernel_raises(self):
        def fn(x, y):
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        kernel = self._kernel(
            fn, block_sizes=[16, 16], pallas_loop_type="compact_worklist"
        )
        args = (torch.randn(64, 64), torch.randn(64, 64))
        bound = kernel.bind(args)
        with self.assertRaises(exc.InvalidConfig):
            bound.ensure_config_exists(args)
            bound.to_triton_code(bound._config)


def _eager_dense_kv(q, k, v, qo):
    """Ground-truth jagged-Q dense-KV GDPA in plain torch (per sequence)."""
    out = torch.empty_like(q)
    for s in range(len(qo) - 1):
        a, b = int(qo[s]), int(qo[s + 1])
        kb = k[s].transpose(0, 1)
        vb = v[s].transpose(0, 1)
        qb = q[a:b].transpose(0, 1)
        scores = torch.bmm(qb, kb.transpose(-2, -1))
        out[a:b] = torch.bmm(scores, vb).transpose(0, 1)
    return out


def _eager_fully_jagged(q, k, v, qo, kvo):
    """Ground-truth fully-jagged GDPA in plain torch (empty KV range -> zeros)."""
    out = torch.empty_like(q)
    for s in range(len(qo) - 1):
        a, b = int(qo[s]), int(qo[s + 1])
        ka, kb_ = int(kvo[s]), int(kvo[s + 1])
        qb = q[a:b].transpose(0, 1)
        if kb_ > ka:
            kb = k[ka:kb_].transpose(0, 1)
            vb = v[ka:kb_].transpose(0, 1)
            scores = torch.bmm(qb, kb.transpose(-2, -1))
            acc = torch.bmm(scores, vb)
        else:
            acc = torch.zeros_like(qb)
        out[a:b] = acc.transpose(0, 1)
    return out


@onlyBackends(["pallas"])
class TestUpperBoundPacking(unittest.TestCase):
    """Only packed-offset bounds are accepted; non-packed is rejected.

    The masked full-block store is only correct for monotonic owner offsets, so
    detection accepts only the packed-offsets idiom (begin=T[g], end=T[g+1]).
    """

    def test_packed_consecutive_structural_matcher(self):
        from helion._compiler.pallas.compact_worklist import _packed_consecutive

        cases = [
            ("offsets[seq]", "offsets[seq + 1]", True),
            ("offsets[seq + 2]", "offsets[seq + 3]", True),
            # Same tensor but non-consecutive indices may skip rows.
            ("offsets[seq]", "offsets[seq + 2]", False),
            ("lo[seq]", "hi[seq + 1]", False),
            ("offsets[seq * seq]", "offsets[seq * seq + 1]", False),
        ]
        for begin, end, expected in cases:
            with self.subTest(begin=begin, end=end):
                self.assertIs(
                    _packed_consecutive(_expr(begin), _expr(end), "seq"),
                    expected,
                )

    def test_packed_idiom_accepted(self):
        # offsets[g] / offsets[g+1] (same tensor, consecutive) => packed: accepted.
        qo = _offsets([16, 16, 16, 16])
        bk = _dense_kv_kernel.bind(
            (
                torch.randn(64, 2, 8),
                torch.randn(4, 16, 2, 8),
                torch.randn(4, 16, 2, 8),
                qo,
            )
        )
        with bk.env:
            plan = detect_compact_worklist_plan(bk.host_function)
        self.assertEqual(plan.compact_axis.loop_var, "tile_q")

    def test_distinct_begin_end_rejected(self):
        # lo[g] / hi[g] (distinct tensors) may be non-monotonic -> the spilling
        # store could clobber an earlier owner; rejected for store safety.
        lo = torch.tensor([0, 30, 70], dtype=torch.int32)
        hi = torch.tensor([20, 60, 99], dtype=torch.int32)
        bk = _dual_offset_kernel.bind((torch.randn(120, 2, 8), lo, hi))
        with bk.env, self.assertRaises(exc.InvalidConfig):
            detect_compact_worklist_plan(bk.host_function)


@onlyBackends(["pallas"])
class TestConfigStateIsolation(unittest.TestCase):
    """compact plan must not leak into a later config.

    One CompileEnvironment is reused across all configs of a BoundKernel, and
    many lowering paths gate on ``env.compact_worklist_plan is not None``.  If a
    compact config is codegen'd first, a later fori/emit/unroll config on the
    same kernel must NOT inherit the stale plan and emit compact codegen.
    """

    def test_no_stale_compact_plan_in_later_config(self):
        kernel = helion.kernel(
            _dense_kv_kernel.fn, static_shapes=True, backend="pallas"
        )
        qo = _offsets([12, 20, 5, 30])
        lq = int(qo[-1])
        args = (
            torch.randn(lq, 2, 8),
            torch.randn(4, 16, 2, 8),
            torch.randn(4, 16, 2, 8),
            qo,
        )
        bound = kernel.bind(args)
        # Compact config first -> sets env.compact_worklist_plan.
        compact = bound.to_triton_code(
            helion.Config(block_sizes=[8], pallas_loop_type="compact_worklist")
        )
        self.assertIn("work_seq_ref", compact)  # sanity: compact really compiled
        # Then a fori config on the SAME bound kernel (same env): must be clean.
        fori = bound.to_triton_code(
            helion.Config(block_sizes=[8], pallas_loop_type="fori_loop")
        )
        self.assertNotIn("work_seq_ref", fori)
        self.assertNotIn("_default_pallas_compact_worklist_launcher", fori)
        self.assertNotIn("_build_worklist", fori)


@onlyBackends(["pallas"])
class TestPrologueScoping(unittest.TestCase):
    """Prologue assigns must stop at the selected loop."""

    def test_collect_stops_before_loop(self):
        from helion._compiler.pallas.compact_worklist import _collect_prologue_assigns

        body = ast.parse(
            "x = a[g]\n"
            "for t in loop:\n"
            "    use(x)\n"
            "x = a[g] + 5\n"  # reassignment AFTER the loop must be ignored
        ).body
        loop = body[1]
        scoped = _collect_prologue_assigns(body, before=loop)
        self.assertEqual(ast.unparse(scoped["x"]), "a[g]")
        # Without the cutoff, the later (wrong) binding would win.
        unscoped = _collect_prologue_assigns(body)
        self.assertEqual(ast.unparse(unscoped["x"]), "a[g] + 5")


@skipUnlessPallas("compact_worklist numerics need Pallas TPU or interpret mode")
class TestCompactWorklistNumerics(unittest.TestCase):
    """Device/interpret numerics for compact_worklist vs an eager ground truth.

    These are the device-side guards the host tests can't give: compact_worklist's
    masked full-block ordered-overwrite store and its ordered inner ``fori`` loop
    must reproduce the exact result of a plain per-sequence torch computation.
    Runs on real TPU in CI and, via ``@skipUnlessPallas``, in pallas interpret
    mode on CPU.

    NB: the comparison is against an INDEPENDENT eager reference, not the
    ``fori_loop`` lowering -- ``fori_loop`` miscompiles these jagged patterns in
    interpret mode (verified: ~64.0 abs error vs eager), so it is not a sound
    oracle here.  compact_worklist matches eager to float32 roundoff.
    """

    def test_dense_kv_unaligned_matches_eager(self):
        # Unaligned offsets + partial last tiles => the store-overlap case that
        # ordered-overwrite under "arbitrary" must repair.  block=16, lengths not
        # multiples of 16 and one < block (7).
        B, H, KV, D, block = 4, 2, 16, 16, 16
        qo = _offsets([10, 23, 7, 40])
        lq = int(qo[-1])
        torch.manual_seed(0)
        # bf16 (the kernels' real dtype): matmuls run at the same precision as the
        # eager reference, so a plain assert_close at the usual jagged-attention
        # tolerance holds. fp32 inputs would expose the TPU bf16-matmul gap vs an
        # fp32 reference and need an unreasonably loose tolerance.
        q = torch.randn(lq, H, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, KV, H, D, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(B, KV, H, D, device=DEVICE, dtype=torch.bfloat16)
        _, out = code_and_output(
            _dense_kv_kernel,
            (q, k, v, qo.to(DEVICE)),
            block_sizes=[block],
            pallas_loop_type="compact_worklist",
        )
        ref = _eager_dense_kv(q.cpu(), k.cpu(), v.cpu(), qo)
        torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)

    def test_dense_kv_empty_batch_zero_grid(self):
        # total_q == 0 => num_work == 0 => dynamic grid=(0,).  End-to-end guard
        # that the empty-batch launch returns an empty output (Mosaic tolerates
        # the zero grid; no static skip needed).
        B, H, KV, D, block = 4, 2, 16, 16, 16
        qo = _offsets([0, 0, 0, 0])
        q = torch.zeros(0, H, D, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, KV, H, D, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, KV, H, D, device=DEVICE, dtype=torch.float32)
        _, out = code_and_output(
            _dense_kv_kernel,
            (q, k, v, qo.to(DEVICE)),
            block_sizes=[block],
            pallas_loop_type="compact_worklist",
        )
        self.assertEqual(tuple(out.shape), (0, H, D))

    @skipIfPallasInterpret(
        "fori inner-loop (used by the ordered KV axis) miscompiles in pallas "
        "interpret mode; this path is validated on real TPU"
    )
    def test_fully_jagged_with_empty_kv_matches_eager(self):
        # Ordered inner KV loop + an empty KV range (seq 2: kv_len==0) to exercise
        # the empty-range identity and per-work-item scratch privacy.
        H, D, block = 2, 16, 16
        qo = _offsets([10, 23, 7, 40])
        kvo = _offsets([16, 5, 0, 33])
        lq, lkv = int(qo[-1]), int(kvo[-1])
        torch.manual_seed(0)
        q = torch.randn(lq, H, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(lkv, H, D, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(lkv, H, D, device=DEVICE, dtype=torch.bfloat16)
        _, out = code_and_output(
            _fully_jagged_kernel,
            (q, k, v, qo.to(DEVICE), kvo.to(DEVICE)),
            block_sizes=[block, block],  # compact Q tile + ordered KV tile
            pallas_loop_type="compact_worklist",
        )
        ref = _eager_fully_jagged(q.cpu(), k.cpu(), v.cpu(), qo, kvo)
        torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)


@unittest.skipUnless(HAS_JAX, "jax not available")
@skipUnlessPallas("jax_fn export needs Pallas TPU or interpret mode")
class TestCompactWorklistJaxExport(unittest.TestCase):
    """compact_worklist embeds in an outer ``jax.jit`` via ``Kernel.jax_fn``.

    Guards the pure-JAX export path: ``jax.jit(kernel.jax_fn)`` must match the
    eager result.
    """

    def test_jax_fn_under_jit_matches_eager(self):
        import jax
        import jax.numpy as jnp

        B, H, KV, D, block = 8, 2, 16, 16, 16
        qo = _offsets([10, 23, 7, 40, 0, 16, 33, 5])
        lq = int(qo[-1])
        q = jax.random.normal(jax.random.PRNGKey(0), (lq, H, D), jnp.bfloat16)
        k = jax.random.normal(jax.random.PRNGKey(1), (B, KV, H, D), jnp.bfloat16)
        v = jax.random.normal(jax.random.PRNGKey(2), (B, KV, H, D), jnp.bfloat16)
        qod = jnp.asarray(qo.numpy())
        kernel = helion.kernel(
            _dense_kv_kernel.fn,
            config=helion.Config(
                block_sizes=[block], pallas_loop_type="compact_worklist"
            ),
            static_shapes=True,
            backend="pallas",
        )
        out = jax.block_until_ready(jax.jit(kernel.jax_fn)(q, k, v, qod))
        # jnp reference (dense-KV GDPA) at the kernel's bf16 precision -- stays in
        # JAX, no torch round-trip.
        bounds = qo.tolist()
        per_seq = []
        for s in range(B):
            a0, b0 = bounds[s], bounds[s + 1]
            qb = jnp.swapaxes(q[a0:b0], 0, 1)  # [H, m, D]
            kb = jnp.swapaxes(k[s], 0, 1)  # [H, KV, D]
            vb = jnp.swapaxes(v[s], 0, 1)
            scores = jnp.matmul(qb, jnp.swapaxes(kb, -2, -1))
            per_seq.append(jnp.swapaxes(jnp.matmul(scores.astype(v.dtype), vb), 0, 1))
        ref = jnp.concatenate(per_seq, axis=0)
        np.testing.assert_allclose(
            np.asarray(out).astype(np.float32),
            np.asarray(ref).astype(np.float32),
            rtol=2e-2,
            atol=2e-2,
        )


if __name__ == "__main__":
    unittest.main()
