"""helion.flex_attention: compiled flex_attention on the Helion CuTe kernels.

Every case runs with BACKEND="TRITON" (the lowering that asks Inductor's
choice hooks) twice: with helion.flex_attention enabled, where it must run
the Helion kernels (or keep Triton's, logging why, for the calls they do not
support), and disabled, as the reference: Helion's error against an fp32
eager reference must be within twice Triton's plus the reference's rounding.

The case tables after "test_flex_flash.py's cases" are ported from PyTorch's
test/inductor/test_flex_flash.py (BACKEND="FLASH"), with the same names.
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import io
import math
import os
import tempfile
from typing import Callable
import unittest
from unittest import mock

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE
from torch.nn.attention.flex_attention import AuxRequest
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.testing._internal.common_utils import DeterministicGuard
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import subtest

from helion._compat import inductor_boxes_template_multi_outputs
from helion._compat import supports_inductor_flex_attention_choices
from helion._testing import DEVICE
from helion._testing import _has_cute_dsl
from helion.autotuner import AccuracyCheck
from helion.autotuner import LocalAutotuneCache
from helion.runtime.kernel import Kernel

if not supports_inductor_flex_attention_choices():
    pytest.skip(
        "helion.flex_attention needs Inductor's flex_attention choice hooks "
        "(PyTorch 2.15 and later)",
        allow_module_level=True,
    )

import helion.flex_attention
from helion.flex_attention import _choices
from helion.flex_attention import _kernels
from helion.flex_attention import _lowering

# Collected and skipped, not skipped at import: CI runs this module alone (the
# b200-flex-attention job), where a module-level skip collects nothing and
# pytest exits with status 5.
pytestmark = pytest.mark.skipif(
    not inductor_boxes_template_multi_outputs(),
    reason="helion.flex_attention's kernels need a PyTorch nightly from "
    "2026-11-19 or later (pytorch/pytorch#205348)",
)

HAS_SM100 = (
    _has_cute_dsl()
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
)
TRITON = {"BACKEND": "TRITON"}
HELION_FORWARD = "_helion_flex_attention_forward"
HELION_BACKWARD = "_helion_flex_attention_backward"
LOGGER = "helion.flex_attention"

# Importing helion.flex_attention routes every compiled flex_attention to
# Helion; keep that to this module's tests (setUp enables it), not the other
# modules a pytest worker runs.
helion.flex_attention.disable()


def _causal_block(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _softcap(score, b, h, q_idx, kv_idx):
    return 30.0 * torch.tanh(score / 30.0)


def _tensors(b, hq, hkv, m, n, d, dtype=torch.bfloat16, requires_grad=True):
    q = torch.randn(b, hq, m, d, device=DEVICE, dtype=dtype)
    k = torch.randn(b, hkv, n, d, device=DEVICE, dtype=dtype)
    v = torch.randn(b, hkv, n, d, device=DEVICE, dtype=dtype)
    return tuple(t.requires_grad_(requires_grad) for t in (q, k, v))


def _case(name):
    """(q, k, v, score_mod, block_mask) of a test case."""
    if name == "dense":
        return (*_tensors(2, 4, 4, 256, 256, 64), None, None)
    if name == "causal":
        mask = create_block_mask(_causal_block, None, None, 256, 256, device=DEVICE)
        return (*_tensors(2, 4, 4, 256, 256, 128), None, mask)
    if name == "softcap_causal":
        mask = create_block_mask(_causal_block, None, None, 256, 256, device=DEVICE)
        return (*_tensors(2, 4, 4, 256, 256, 128), _softcap, mask)
    if name in ("alibi_document", "bias_document"):
        bias = torch.randn(2, 4, 256, 256, device=DEVICE)
        slopes = torch.rand(4, device=DEVICE)
        doc = torch.repeat_interleave(torch.arange(4, device=DEVICE), 64)

        def alibi(score, b, h, q_idx, kv_idx):
            return score + slopes[h] * (kv_idx - q_idx)

        def biased(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]

        def document_causal(b, h, q_idx, kv_idx):
            return (doc[q_idx] == doc[kv_idx]) & (q_idx >= kv_idx)

        mask = create_block_mask(document_causal, None, None, 256, 256, device=DEVICE)
        score_mod = alibi if name == "alibi_document" else biased
        return (*_tensors(2, 4, 4, 256, 256, 64), score_mod, mask)
    if name == "gqa":
        mask = create_block_mask(_causal_block, None, None, 256, 256, device=DEVICE)
        return (*_tensors(2, 8, 2, 256, 256, 128), None, mask)
    if name == "rect":

        def offset_causal(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx + 256

        mask = create_block_mask(offset_causal, None, None, 256, 512, device=DEVICE)
        return (*_tensors(2, 4, 4, 256, 512, 64), None, mask)
    raise AssertionError(name)


# A dense walk, a BlockMask, a coordinate score_mod, captured-tensor gathers,
# GQA and a rectangular (q_len != kv_len) mask: each lowers differently.
CASES = [
    "dense",
    "causal",
    "softcap_causal",
    "alibi_document",
    "bias_document",
    "gqa",
    "rect",
]


@contextmanager
def _triton_only():
    """helion.flex_attention disabled (and Dynamo's graphs dropped)."""
    torch._dynamo.reset()
    helion.flex_attention.disable()
    try:
        yield
    finally:
        helion.flex_attention.enable()
        torch._dynamo.reset()


@unittest.skipUnless(HAS_SM100, "requires the CuTe DSL and a Blackwell (sm100) GPU")
class FlexHelionTestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        # A real autotune takes minutes per kernel: run Helion's default
        # configs (test_autotunes covers the autotuning path). The autotune
        # cache is a fresh directory, which the lowering reads before it
        # autotunes: an entry left by another run would skip the autotuning
        # the tests observe.
        super().setUpClass()
        cls._cache_dir = tempfile.TemporaryDirectory()
        cls._effort = mock.patch.dict(
            os.environ,
            {"HELION_AUTOTUNE_EFFORT": "none", "HELION_CACHE_DIR": cls._cache_dir.name},
        )
        cls._effort.start()

    @classmethod
    def tearDownClass(cls):
        cls._effort.stop()
        cls._cache_dir.cleanup()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        counters.clear()
        torch._dynamo.reset()
        helion.flex_attention.enable()
        self.addCleanup(helion.flex_attention.disable)
        self.addCleanup(helion.flex_attention.configure)

    def _assert_within_triton_error(self, name, ref, triton, helion, dtype):
        """Helion's error against the fp32 ``ref`` is within twice Triton's
        plus the reference's rounding to ``dtype``."""
        ref, triton, helion = (t.float() for t in (ref, triton, helion))
        self.assertEqual(helion.shape, ref.shape)
        # A row that sees no key has a logsumexp of -inf.
        no_keys = ref == float("-inf")
        self.assertTrue((helion[no_keys] == float("-inf")).all(), name)
        ref, triton, helion = (t.masked_fill(no_keys, 0) for t in (ref, triton, helion))
        self.assertTrue(helion.isfinite().all(), name)
        rounding = (ref.to(dtype).float() - ref).abs().max().item()
        triton_err = (triton - ref).abs().max().item()
        helion_err = (helion - ref).abs().max().item()
        self.assertLessEqual(
            helion_err,
            2 * triton_err + rounding,
            f"{name}: Helion error {helion_err:.2e} vs Triton {triton_err:.2e}",
        )

    def _assert_kernels(self, code, forward, backward):
        """Whether the generated ``code`` runs the Helion forward / backward."""
        for name, expected in ((HELION_FORWARD, forward), (HELION_BACKWARD, backward)):
            if expected:
                self.assertIn(name, code)
            else:
                self.assertNotIn(name, code)

    def _check_against_triton(
        self,
        q,
        k,
        v,
        score_mod=None,
        block_mask=None,
        max_scores=False,
        dynamic=False,
        declined=None,
        backward_declined=None,
        check_log=True,
    ):
        """Check the Helion kernels against an fp32 reference: within twice
        Triton's error plus the reference's input-dtype rounding. With
        grad-requiring inputs the loss reads both the output and the
        logsumexp, so the backward also folds in the logsumexp's gradient.
        ``max_scores`` (forward only) also compares the returned row max.
        ``declined`` is the reason (a regex) the Helion kernels must log for
        keeping Triton's on the whole call (which then runs Triton's kernels
        only, so there is nothing to compare), ``backward_declined`` on its
        backward only (not checked with ``check_log=False``, for a call that
        may reuse a dynamic graph, which lowers nothing)."""
        backward = q.requires_grad
        b, hq, m, _ = q.shape
        grad_out = torch.randn(b, hq, m, v.size(-1), device=DEVICE)
        grad_lse = torch.randn(b, hq, m, device=DEVICE)

        def run(q, k, v, compiled=True):
            fn = (
                torch.compile(flex_attention, fullgraph=True, dynamic=dynamic)
                if compiled
                else flex_attention
            )
            out, aux = fn(
                q,
                k,
                v,
                score_mod=score_mod,
                block_mask=block_mask,
                enable_gqa=q.size(1) != k.size(1),
                return_aux=AuxRequest(lse=True, max_scores=max_scores),
                kernel_options=TRITON if compiled else None,
            )
            if max_scores:
                return out.float(), aux.lse, aux.max_scores
            if not backward:
                return out.float(), aux.lse
            loss = (out * grad_out).sum() + (aux.lse * grad_lse).sum()
            grads = torch.autograd.grad(loss, (q, k, v))
            return out.float(), aux.lse, *(g.float() for g in grads)

        if declined is None:
            fp32 = [t.detach().float().requires_grad_(backward) for t in (q, k, v)]
            ref = run(*fp32, compiled=False)
            with _triton_only():
                triton = run(q, k, v)
        reasons = [r for r in (declined, backward_declined) if r is not None]
        reasons = reasons if check_log else []
        logs = self.assertLogs(LOGGER, "INFO") if reasons else contextlib.nullcontext()
        with logs as captured:
            helion, code = run_and_get_code(run, q, k, v)
        code = "\n".join(code)
        self._assert_kernels(
            code,
            forward=declined is None,
            backward=backward and declined is None and backward_declined is None,
        )
        if declined is None and backward_declined is None:
            # Only the Helion kernels run: no Triton flex_attention templates.
            self.assertNotIn("triton_tem_fused", code)
        for reason in reasons:
            self.assertRegex("\n".join(captured.output), reason)
        if declined is not None:
            return
        names = ("out", "lse", "max_scores" if max_scores else "dq", "dk", "dv")
        for name, r, t, h in zip(names, ref, triton, helion, strict=False):
            self._assert_within_triton_error(name, r, t, h, q.dtype)


@instantiate_parametrized_tests
class TestFlexHelion(FlexHelionTestCase):
    @parametrize("case", CASES)
    def test_forward(self, case):
        q, k, v, score_mod, block_mask = _case(case)
        q, k, v = (t.detach() for t in (q, k, v))
        self._check_against_triton(q, k, v, score_mod, block_mask)

    # Each backward compiles both kernels (about 19s): dense stays in the pull
    # request tier, the rest are slow.
    @parametrize(
        "case",
        [
            c if c == "dense" else subtest(c, decorators=[pytest.mark.slow])
            for c in CASES
        ],
    )
    def test_backward(self, case):
        self._check_against_triton(*_case(case))

    # A full-block walk, a softcap score_mod, and captured-tensor gathers;
    # test_forward compiles the same calls without the row max, so only dense
    # stays in the pull request tier.
    @parametrize(
        "case",
        [
            "dense",
            subtest("softcap_causal", decorators=[pytest.mark.slow]),
            subtest("bias_document", decorators=[pytest.mark.slow]),
        ],
    )
    def test_max_scores(self, case):
        q, k, v, score_mod, block_mask = _case(case)
        q, k, v = (t.detach() for t in (q, k, v))
        self._check_against_triton(q, k, v, score_mod, block_mask, max_scores=True)

    @pytest.mark.slow
    @config.patch({"emulate_precision_casts": True})
    def test_emulate_precision_casts(self):
        # The forward keeps score_mod's float casts, rounding as eager does; the
        # fused backward does not fit them yet and keeps Triton's.
        self._check_against_triton(
            *_case("softcap_causal"),
            backward_declined="emulate_precision_casts=True keeps",
        )

    # The table's load cast to the input dtype: fp16 (a no-op cast) and bf16.
    @pytest.mark.slow
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_upcast_of_captured_table_kept(self, dtype):
        # bias[h].to(score.dtype) * 100 computes in fp32 (eager and Triton):
        # the cast of the fp16 table's load becomes one to fp32, or the
        # product overflows fp16 (|bias| * 100 past 65504).
        bias = (torch.randn(4, device=DEVICE) * 1000).half()

        def scaled_bias(score, b, h, q_idx, kv_idx):
            return score + bias[h].to(score.dtype) * 100.0 / (1 + kv_idx)

        q, k, v = _tensors(2, 4, 4, 256, 256, 64, dtype, requires_grad=False)
        self._check_against_triton(q, k, v, scaled_bias)

    # q - kv; a narrowing cast (int8 wraps kv_idx 128 to -128).
    @parametrize(
        "case",
        ["difference", subtest("narrowing_cast", decorators=[pytest.mark.slow])],
    )
    def test_negative_gather_index_wraps(self, case):
        # table[q_idx - kv_idx] reads table[255] at (0, 1), as eager does
        # (the kernels' gathers read zero out of range). Checked against eager:
        # Triton's kernel faults on the negative index.
        table = torch.randn(256, device=DEVICE)

        def relative(score, b, h, q_idx, kv_idx):
            if case == "difference":
                index = q_idx - kv_idx
            else:
                index = kv_idx.to(torch.int8).long() + kv_idx % 64
            return score + table[index]

        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        ref = flex_attention(*(t.float() for t in (q, k, v)), score_mod=relative)
        out, code = run_and_get_code(
            torch.compile(flex_attention),
            q,
            k,
            v,
            score_mod=relative,
            kernel_options=TRITON,
        )
        self._assert_kernels("\n".join(code), forward=True, backward=False)
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=0.0)

    @pytest.mark.slow
    def test_integer_cast_to_input_dtype_is_exact(self):
        # (q_idx - kv_idx).to(score.dtype) is exact in fp32, as Inductor
        # computes it; at bf16 it rounds |q - kv| > 256.
        def distance(score, b, h, q_idx, kv_idx):
            return score + (q_idx - kv_idx).to(score.dtype) / 64

        q, k, v = _tensors(1, 2, 2, 1024, 1024, 64, requires_grad=False)
        self._check_against_triton(q, k, v, distance)

    @pytest.mark.slow
    def test_fp8_cast_in_score_mod_kept(self):
        # A score_mod that rounds through fp8 keeps that cast: only casts
        # among fp16, bf16 and fp32 are elided.
        def fp8_round(score, b, h, q_idx, kv_idx):
            return score.to(torch.float8_e4m3fn).to(score.dtype)

        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        self._check_against_triton(q, k, v, fp8_round)

    @pytest.mark.slow
    def test_float16(self):
        mask = create_block_mask(_causal_block, None, None, 256, 256, device=DEVICE)
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, dtype=torch.float16)
        self._check_against_triton(q, k, v, _softcap, mask)

    @pytest.mark.slow
    def test_disable(self):
        # disable() keeps Inductor's kernels for both directions (also on the
        # autograd thread, which builds its own choices handler); enable()
        # brings the Helion kernels back.
        q, k, v, score_mod, block_mask = _case("causal")

        def code():
            torch._dynamo.reset()
            compiled = torch.compile(
                functools.partial(flex_attention, kernel_options=TRITON)
            )

            def step():
                compiled(q, k, v, block_mask=block_mask).sum().backward()

            return "\n".join(run_and_get_code(step)[1])

        self._assert_kernels(code(), forward=True, backward=True)
        helion.flex_attention.disable()
        self.addCleanup(helion.flex_attention.enable)
        self._assert_kernels(code(), forward=False, backward=False)
        helion.flex_attention.enable()
        self._assert_kernels(code(), forward=True, backward=True)

    def test_dynamic_shapes_keep_triton(self):
        # The kernels compile for static shapes. A graph with symbolic sizes
        # (dynamic=True, or a size marked dynamic) keeps Triton's kernels
        # rather than being specialized: guards would recompile it per shape
        # and change the graph Triton's backward lowers.
        doc = torch.repeat_interleave(torch.arange(4, device=DEVICE), 64)

        def document_causal(b, h, q_idx, kv_idx):
            return (doc[q_idx] == doc[kv_idx]) & (q_idx >= kv_idx)

        mask = create_block_mask(document_causal, None, None, 256, 256, device=DEVICE)
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(
            functools.partial(flex_attention, kernel_options=TRITON),
            backend=counter,
            dynamic=True,
        )
        # Batch sizes that no other size equals, so duck sizing adds no guards.
        for b in (3, 5):
            q, k, v = _tensors(b, 4, 4, 256, 256, 64, requires_grad=False)
            # The second batch size reuses the graph and lowers nothing.
            logs = (
                self.assertLogs(LOGGER, "INFO") if b == 3 else contextlib.nullcontext()
            )
            with logs as captured:
                out = compiled(q, k, v, block_mask=mask)
            if b == 3:
                self.assertRegex("\n".join(captured.output), "NYI: dynamic shapes")
            torch.testing.assert_close(
                out, flex_attention(q, k, v, block_mask=mask), atol=2e-2, rtol=2e-2
            )
        self.assertEqual(counter.frame_count, 1)

        torch._dynamo.reset()
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        for t in (q, k, v):
            torch._dynamo.mark_dynamic(t, 2)
        with self.assertLogs(LOGGER, "INFO") as captured:
            _, code = run_and_get_code(
                torch.compile(flex_attention), q, k, v, kernel_options=TRITON
            )
        self.assertRegex("\n".join(captured.output), "NYI: dynamic shapes")
        self._assert_kernels("\n".join(code), forward=False, backward=False)

    def test_autotunes(self):
        # The kernels autotune at Helion's effort on real inputs, with dense
        # BlockMask tables (every block partial, as the causal mask_mod can
        # mask), accept configs within the configured tolerance, and lower
        # the config autotuning finds.

        mask = create_block_mask(_causal_block, None, None, 256, 256, device=DEVICE)
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        found = {}

        def autotune(kernel, args, force):
            self.assertFalse(force)
            self.assertEqual(kernel.settings.autotune_effort, "quick")
            self.assertEqual(
                kernel.settings.autotune_baseline_accuracy_check_fn,
                AccuracyCheck(atol=0.05, rtol=0.0, atol_scale="peak"),
            )
            names = list(kernel.signature.parameters)
            table = dict(zip(names, args, strict=True))
            self.assertNotIsInstance(table["query"], torch._subclasses.FakeTensor)
            ncols = table["block_mask_kv_indices"].size(-1)
            self.assertTrue((table["block_mask_kv_num_blocks"] == ncols).all())
            self.assertTrue((table["block_mask_full_kv_num_blocks"] == 0).all())
            found["config"] = kernel.bind(args).config_spec.default_config()
            return found["config"]

        helion.flex_attention.configure(autotune_tolerance=0.05)
        with (
            mock.patch.dict(os.environ, {"HELION_AUTOTUNE_EFFORT": "quick"}),
            mock.patch.object(Kernel, "autotune", autospec=True) as kernel_autotune,
            mock.patch.object(
                _kernels, "get_kernel", wraps=_kernels.get_kernel
            ) as get_kernel,
        ):
            kernel_autotune.side_effect = autotune
            out = torch.compile(flex_attention)(
                q, k, v, block_mask=mask, kernel_options=TRITON
            )
        self.assertTrue(out.isfinite().all())
        self.assertEqual(kernel_autotune.call_count, 1)
        get_kernel.assert_called_with("forward", config=dict(found["config"].config))

    def test_one_large_block_applies_mask_mod(self):
        # A BlockMask of one _LARGE_SPARSE_BLOCK_SIZE block (the frontend's
        # "no BlockMask" shape) lists that block as partial: the kernels walk
        # dense tables whose blocks are partial, so mask_mod still applies.
        mask = BlockMask.from_kv_blocks(
            torch.ones([1, 1, 1], dtype=torch.int32, device=DEVICE),
            torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=DEVICE),
            BLOCK_SIZE=_LARGE_SPARSE_BLOCK_SIZE,
            mask_mod=_causal_block,
            seq_lengths=(256, 256),
        )
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        self._check_against_triton(q, k, v, block_mask=mask)

    # One block marked full (it skips mask_mod, as in Triton's kernel) and no
    # block at all (no key visible: a zero output and a -inf logsumexp).
    @pytest.mark.slow
    @parametrize("case", ["full", "empty"])
    def test_one_large_block_counts(self, case):
        ones = torch.ones([1, 1, 1], dtype=torch.int32, device=DEVICE)
        zeros = torch.zeros([1, 1, 1], dtype=torch.int32, device=DEVICE)
        indices = torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=DEVICE)
        mask = BlockMask.from_kv_blocks(
            zeros,
            indices,
            ones if case == "full" else None,
            indices if case == "full" else None,
            BLOCK_SIZE=_LARGE_SPARSE_BLOCK_SIZE,
            mask_mod=_causal_block if case == "full" else None,
            seq_lengths=(256, 256),
        )
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)

        def run():
            return torch.compile(flex_attention)(
                q, k, v, block_mask=mask, return_lse=True, kernel_options=TRITON
            )

        with _triton_only():
            triton_out, triton_lse = run()
        (out, lse), code = run_and_get_code(run)
        self._assert_kernels("\n".join(code), forward=True, backward=False)
        torch.testing.assert_close(out, triton_out, atol=2e-2, rtol=0.0)
        torch.testing.assert_close(lse, triton_lse, atol=1e-3, rtol=0.0)
        if case == "empty":
            self.assertTrue((out == 0).all())
            self.assertTrue((lse == float("-inf")).all())

    # Python wrapper code only; on-device autotuning and atomics rule out
    # Inductor's deterministic modes.
    @pytest.mark.slow
    @parametrize(
        "mode,reason",
        [
            ("cpp_wrapper", "NYI: cpp_wrapper and AOTInductor"),
            ("deterministic", "Inductor's deterministic mode is on"),
            ("batch_invariant", "Inductor's deterministic mode is on"),
        ],
    )
    def test_wrapper_and_deterministic_modes_keep_triton(self, mode, reason):
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        with config.patch({mode: True}):
            self._check_against_triton(q, k, v, declined=reason)

    def test_disallowed_autotuning_keeps_triton(self):
        # Helion's autotuner raises where autotuning is disallowed (also in
        # every MAST job), before reading any cache: without a configured
        # config, a kernel that would autotune keeps Triton's.
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        env = {"HELION_AUTOTUNE_EFFORT": "quick", "HELION_DISALLOW_AUTOTUNING": "1"}
        with mock.patch.dict(os.environ, env):
            self._check_against_triton(q, k, v, declined="autotuning is disallowed")

    def test_reference_mode_keeps_triton(self):
        # Under HELION_INTERPRET=1 the Helion kernels run eagerly and bind no
        # host function to lower: the call keeps Triton's kernel.
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        with mock.patch.dict(os.environ, {"HELION_INTERPRET": "1"}):
            self._check_against_triton(q, k, v, declined="reference mode")

    def test_unboxed_template_outputs_keep_triton(self):
        # Before pytorch/pytorch#205348, Inductor left a template's outputs
        # unboxed, and a kernel output the graph returns and reuses failed to
        # lower: the call keeps Triton's kernel.
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        with mock.patch.object(
            _lowering, "inductor_boxes_template_multi_outputs", return_value=False
        ):
            self._check_against_triton(q, k, v, declined="pytorch/pytorch#205348")

    def test_internal_error_keeps_triton_with_a_warning(self):
        # A Helion bug while binding the kernel keeps Triton's kernel, and
        # says so at WARNING (an unsupported call logs at INFO).
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        error = helion.exc.InternalError(RuntimeError("boom"))
        with (
            mock.patch.object(Kernel, "_bind_isolated", side_effect=error),
            self.assertLogs(LOGGER, "WARNING") as captured,
        ):
            out, code = run_and_get_code(
                torch.compile(flex_attention), q, k, v, kernel_options=TRITON
            )
        self._assert_kernels("\n".join(code), forward=False, backward=False)
        self.assertIn("RuntimeError: boom", "\n".join(captured.output))
        self.assertTrue(out.isfinite().all())

    @pytest.mark.slow
    @parametrize("case", ["in_graph", "graph_input"])
    def test_misaligned_operand_view_is_copied(self, case):
        # A query view one element into an (aligned) buffer: TMA needs a
        # 16-byte aligned base, so the operand is copied into a fresh buffer.
        # in_graph: a contiguous view taken in the graph (require_contiguous
        # alone kept the view and its offset). graph_input: a non-dense view
        # taken before the compile, which Inductor gives offset 0 and lists
        # in V.graph.unaligned_buffers: the launcher stages it into an
        # aligned copy with its strides (its clone compacted it under TMA
        # maps built for its row stride).
        b, h, m, d = 2, 4, 256, 64
        width = d if case == "in_graph" else 2 * d
        storage = torch.randn(
            b * h * m * width + 1, device=DEVICE, dtype=torch.bfloat16
        )
        _, k, v = _tensors(b, h, h, m, m, d, requires_grad=False)

        def view(storage):
            return storage[1:].view(b, h, m, width)[..., :d]

        if case == "in_graph":

            def attend(q, k, v):
                return flex_attention(view(q), k, v, kernel_options=TRITON)

            q = storage
        else:

            def attend(q, k, v):
                return flex_attention(q, k, v, kernel_options=TRITON)

            q = view(storage)
        compiled = torch.compile(attend, fullgraph=True)
        with _triton_only():
            triton = compiled(q, k, v)
        helion, code = run_and_get_code(compiled, q, k, v)
        self._assert_kernels("\n".join(code), forward=True, backward=False)
        torch.testing.assert_close(helion, triton, atol=2e-2, rtol=0.0)

    @pytest.mark.slow
    def test_autotunes_with_broadcast_captured_tensor(self):
        # The autotune examples keep a captured tensor's layout: an expanded
        # (zero-stride) table used to fail the compile (normal_() on
        # overlapping memory) at any autotuning effort.
        slopes = torch.rand(4, device=DEVICE)
        table = slopes[None, :, None, None].expand(2, 4, 256, 256)

        def bias(score, b, h, q_idx, kv_idx):
            return score + table[b, h, q_idx, kv_idx]

        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        strides = []

        def autotune(kernel, args, force):
            buffers = dict(zip(kernel.signature.parameters, args, strict=True))
            strides.extend(t.stride() for t in buffers["score_mod_buffers"])
            return kernel.bind(args).config_spec.default_config()

        with (
            mock.patch.dict(os.environ, {"HELION_AUTOTUNE_EFFORT": "quick"}),
            mock.patch.object(Kernel, "autotune", autospec=True) as kernel_autotune,
        ):
            kernel_autotune.side_effect = autotune
            out = torch.compile(flex_attention)(
                q, k, v, score_mod=bias, kernel_options=TRITON
            )
        self.assertEqual(strides, [(0, 1, 0, 0)])
        ref = flex_attention(*(t.float() for t in (q, k, v)), score_mod=bias)
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=0.0)

    @pytest.mark.slow
    def test_autotunes_with_broadcast_block_mask_tables(self):
        # BlockMask tables expanded over batch and heads (zero strides): the
        # autotune example tables are written through distinct elements
        # (copy_ into the expanded layout failed the compile).
        mask = create_block_mask(_causal_block, None, None, 256, 256, device=DEVICE)
        kv_num_blocks, kv_indices = mask.kv_num_blocks, mask.kv_indices
        mask = BlockMask.from_kv_blocks(
            kv_num_blocks.expand(2, 4, -1),
            kv_indices.expand(2, 4, -1, -1),
            mask.full_kv_num_blocks.expand(2, 4, -1),
            mask.full_kv_indices.expand(2, 4, -1, -1),
            BLOCK_SIZE=mask.BLOCK_SIZE,
            mask_mod=_causal_block,
            seq_lengths=(256, 256),
        )
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        tuned = []

        def autotune(kernel, args, force):
            tuned.append(True)
            return kernel.bind(args).config_spec.default_config()

        with (
            mock.patch.dict(os.environ, {"HELION_AUTOTUNE_EFFORT": "quick"}),
            mock.patch.object(Kernel, "autotune", autospec=True) as kernel_autotune,
        ):
            kernel_autotune.side_effect = autotune
            out, code = run_and_get_code(
                torch.compile(flex_attention),
                q,
                k,
                v,
                block_mask=mask,
                kernel_options=TRITON,
            )
        self._assert_kernels("\n".join(code), forward=True, backward=False)
        self.assertEqual(tuned, [True])
        with _triton_only():
            triton = torch.compile(flex_attention)(
                q, k, v, block_mask=mask, kernel_options=TRITON
            )
        torch.testing.assert_close(out, triton, atol=2e-2, rtol=0.0)

    @pytest.mark.slow
    def test_backward_autotunes_on_real_row_terms(self):
        # The backward autotunes on the row terms of its example inputs'
        # forward: with random ones, P = exp2(score - lse2) overflows under a
        # score_mod that adds a large bias, and every config's gradients are
        # inf / NaN alike.

        def bias(score, b, h, q_idx, kv_idx):
            return score + (kv_idx - q_idx)

        q, k, v = _tensors(2, 4, 4, 256, 256, 64)
        checked = []

        def autotune(kernel, args, force):
            config = kernel.bind(args).config_spec.default_config()
            table = dict(zip(kernel.signature.parameters, args, strict=True))
            if "lse2" not in table:
                return config
            q, k = (table[n].float() for n in ("query", "key"))
            positions = torch.arange(q.size(2), device=DEVICE)
            scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
            scores = scores + (positions[None, :] - positions[:, None])
            lse2 = torch.logsumexp(scores, -1) / math.log(2.0)
            torch.testing.assert_close(table["lse2"], lse2, atol=0.05, rtol=0.0)
            self.assertTrue(table["delta"].isfinite().all())
            checked.append(True)
            return config

        with (
            mock.patch.dict(os.environ, {"HELION_AUTOTUNE_EFFORT": "quick"}),
            mock.patch.object(Kernel, "autotune", autospec=True) as kernel_autotune,
        ):
            kernel_autotune.side_effect = autotune
            out = torch.compile(flex_attention)(
                q, k, v, score_mod=bias, kernel_options=TRITON
            )
            out.sum().backward()
        self.assertEqual(checked, [True])
        self.assertTrue(q.grad.isfinite().all())

    @pytest.mark.slow
    def test_autotunes_with_the_mask_applied(self):
        # A score_mod finite only where the mask keeps an entry (1 / 0 at
        # kv == q + 1 under a causal mask): the autotuning examples list
        # their blocks as partial, so they apply the mask_mod as the call
        # does; with every block full the examples' outputs, and the
        # backward's example row terms, were NaN, which passes the accuracy
        # check.

        def over_distance(score, b, h, q_idx, kv_idx):
            return score / (1 + q_idx - kv_idx).to(torch.float32)

        mask = create_block_mask(_causal_block, None, None, 256, 256, device=DEVICE)
        q, k, v = _tensors(2, 4, 4, 256, 256, 64)
        checked = []

        def autotune(kernel, args, force):
            table = dict(zip(kernel.signature.parameters, args, strict=True))
            if "lse2" in table:
                self.assertTrue(table["lse2"].isfinite().all())
                self.assertTrue(table["delta"].isfinite().all())
                checked.append("backward")
            else:
                ncols = table["block_mask_kv_indices"].size(-1)
                self.assertTrue((table["block_mask_kv_num_blocks"] == ncols).all())
                self.assertTrue((table["block_mask_full_kv_num_blocks"] == 0).all())
                checked.append("forward")
            return kernel.bind(args).config_spec.default_config()

        with (
            mock.patch.dict(os.environ, {"HELION_AUTOTUNE_EFFORT": "quick"}),
            mock.patch.object(Kernel, "autotune", autospec=True) as kernel_autotune,
        ):
            kernel_autotune.side_effect = autotune
            out = torch.compile(flex_attention)(
                q, k, v, score_mod=over_distance, block_mask=mask, kernel_options=TRITON
            )
            out.sum().backward()
        self.assertEqual(checked, ["forward", "backward"])
        self.assertTrue(q.grad.isfinite().all())

    @pytest.mark.slow
    def test_configured_configs(self):
        # configure(forward_config=..., backward_config=...) fixes each kernel
        # to that config with no autotuning, even when Helion's effort would
        # autotune; a config that does not apply raises.

        mask = create_block_mask(_causal_block, None, None, 256, 256, device=DEVICE)
        configs = {}

        def default_config(kernel, args, force):
            config = kernel.bind(args).config_spec.default_config()
            params = kernel.signature.parameters
            direction = "backward" if "grad_out" in params else "forward"
            configs[direction] = dict(config.config)
            return config

        def run(autotune):
            torch._dynamo.reset()
            q, k, v = _tensors(2, 4, 4, 256, 256, 64)
            with mock.patch.object(
                Kernel, "autotune", autospec=True
            ) as kernel_autotune:
                kernel_autotune.side_effect = autotune
                out = torch.compile(flex_attention)(
                    q, k, v, block_mask=mask, kernel_options=TRITON
                )
                out.sum().backward()
            return kernel_autotune.call_count

        def no_autotune(kernel, args, force):
            raise AssertionError("autotuned despite a configured config")

        with mock.patch.dict(os.environ, {"HELION_AUTOTUNE_EFFORT": "quick"}):
            self.assertEqual(run(default_config), 2)
            self.assertEqual(set(configs), {"forward", "backward"})
            helion.flex_attention.configure(
                forward_config=configs["forward"], backward_config=configs["backward"]
            )
            with mock.patch.object(
                _kernels, "get_kernel", wraps=_kernels.get_kernel
            ) as get_kernel:
                self.assertEqual(run(no_autotune), 0)
            for direction in ("forward", "backward"):
                self.assertIn(
                    mock.call(direction, config=configs[direction]),
                    get_kernel.call_args_list,
                )
            # A configured config is final: HELION_FORCE_AUTOTUNE does not
            # send the lowering back to autotuning (which, where autotuning
            # is disallowed, failed the compile).
            forced = {"HELION_FORCE_AUTOTUNE": "1", "HELION_DISALLOW_AUTOTUNING": "1"}
            with mock.patch.dict(os.environ, forced):
                self.assertEqual(run(no_autotune), 0)
            helion.flex_attention.configure(forward_config={"not_a_helion_knob": 1})
            with self.assertRaisesRegex(
                Exception, r"configure\(forward_config=\.\.\.\)"
            ):
                run(no_autotune)
            # A valid config outside the fused kernel's envelope raises too,
            # rather than running Helion's generic kernel.
            helion.flex_attention.configure(
                forward_config={**configs["forward"], "pid_type": "persistent_blocked"}
            )
            with self.assertRaisesRegex(Exception, r"fused kernel takes only"):
                run(no_autotune)

    @pytest.mark.slow
    def test_cached_config_skips_the_examples_runs(self):
        # A config in Helion's autotune cache is read before anything
        # compiles: no autotuning, and no default-config run of the examples
        # (the NaN probe, the backward's example forward).
        q, k, v = _tensors(2, 4, 4, 256, 256, 64)

        def cached(cache):
            return cache.kernel.config_spec.default_config()

        def no_autotune(kernel, args, force):
            raise AssertionError("autotuned despite a cached config")

        with (
            mock.patch.dict(os.environ, {"HELION_AUTOTUNE_EFFORT": "quick"}),
            mock.patch.object(
                LocalAutotuneCache, "get", autospec=True, side_effect=cached
            ),
            mock.patch.object(Kernel, "autotune", autospec=True) as kernel_autotune,
            mock.patch.object(
                _kernels, "get_kernel", wraps=_kernels.get_kernel
            ) as get_kernel,
        ):
            kernel_autotune.side_effect = no_autotune
            out = torch.compile(flex_attention)(q, k, v, kernel_options=TRITON)
            out.sum().backward()
        # Per direction: the planning kernel, the tuning kernel and the
        # kernel fixed to the cached config, with no default-config kernel
        # for a probe run.
        fixed = [c for c in get_kernel.call_args_list if c.kwargs.get("config")]
        self.assertEqual([c.args[0] for c in fixed], ["forward", "backward"])
        self.assertTrue(q.grad.isfinite().all())

    def test_fp8_captured_table_keeps_triton(self):
        # The CuTe DSL aborts the process compiling the fused kernels' read
        # of an fp8 table, so the call keeps Triton's kernel.
        table = torch.randn(256, device=DEVICE).to(torch.float8_e4m3fn)

        def bias(score, b, h, q_idx, kv_idx):
            return score + table[kv_idx].float()

        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        self._check_against_triton(q, k, v, bias, declined="captures an fp8 tensor")

    @pytest.mark.slow
    @parametrize("case", ["nan", "all_masked"])
    def test_degenerate_examples_keep_the_default_config(self, case):
        # Autotuning examples the accuracy check cannot tell configs apart
        # on: a captured tensor's random example outside a score_mod's
        # domain (torch.log of a negative bias) makes every row NaN, which
        # counts as a match; a mask_mod reading a captured integer tensor
        # (kv_idx < lengths[b]) on its zero example masks every entry. The
        # kernel keeps Helion's default config rather than autotune.
        bias = torch.rand(256, device=DEVICE) + 0.5
        lengths = torch.tensor([200, 256], device=DEVICE, dtype=torch.int32)

        def log_bias(score, b, h, q_idx, kv_idx):
            return score + torch.log(bias[kv_idx])

        def padding(b, h, q_idx, kv_idx):
            return kv_idx < lengths[b]

        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        if case == "nan":
            mods = {"score_mod": log_bias}
        else:
            mods = {
                "block_mask": _create_block_mask_for_device(
                    padding, 2, 1, 256, 256, device=DEVICE
                )
            }
        ref = flex_attention(*(t.float() for t in (q, k, v)), **mods)

        def no_autotune(kernel, args, force):
            raise AssertionError("autotuned on degenerate examples")

        with (
            mock.patch.dict(os.environ, {"HELION_AUTOTUNE_EFFORT": "quick"}),
            mock.patch.object(Kernel, "autotune", autospec=True) as kernel_autotune,
            self.assertLogs(LOGGER, "INFO") as captured,
        ):
            kernel_autotune.side_effect = no_autotune
            out, code = run_and_get_code(
                torch.compile(flex_attention), q, k, v, kernel_options=TRITON, **mods
            )
        self.assertIn("keeps Helion's default config", "\n".join(captured.output))
        self._assert_kernels("\n".join(code), forward=True, backward=False)
        torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=0.0)

    def test_assert_cache_hit_raises_on_degenerate_examples(self):
        # HELION_ASSERT_CACHE_HIT=1 makes a cache miss raise, also where the
        # examples are degenerate (a padding mask_mod on its zero example
        # masks every entry), which would otherwise keep the default config
        # without reaching the autotuner.
        lengths = torch.tensor([200, 256], device=DEVICE, dtype=torch.int32)

        def padding(b, h, q_idx, kv_idx):
            return kv_idx < lengths[b]

        block_mask = _create_block_mask_for_device(
            padding, 2, 1, 256, 256, device=DEVICE
        )
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        env = {"HELION_AUTOTUNE_EFFORT": "quick", "HELION_ASSERT_CACHE_HIT": "1"}
        with (
            mock.patch.dict(os.environ, env),
            self.assertRaisesRegex(Exception, "CacheAssertionError"),
        ):
            torch.compile(flex_attention)(
                q, k, v, block_mask=block_mask, kernel_options=TRITON
            )

    def test_mods_survive_pickling(self):
        # The Helion autotuner's benchmark subprocess loads the kernel args
        # with torch.load, which re-traces each GraphModule mod: the hl.load
        # calls of the prepared mods must survive that.
        q, k, v, score_mod, block_mask = _case("bias_document")
        prepare_mod = _lowering._prepare_mod
        prepared = []

        def record(*args):
            prepared.append(prepare_mod(*args))
            return prepared[-1]

        with mock.patch.object(_lowering, "_prepare_mod", record):
            torch.compile(flex_attention)(
                q.detach(),
                k.detach(),
                v.detach(),
                score_mod=score_mod,
                block_mask=block_mask,
                kernel_options=TRITON,
            )
        self.assertEqual(len(prepared), 2)
        for gm in prepared:
            buffer = io.BytesIO()
            torch.save(gm, buffer)
            buffer.seek(0)
            loaded = torch.load(buffer, weights_only=False)
            self.assertEqual(
                [(n.op, n.target) for n in loaded.graph.nodes],
                [(n.op, n.target) for n in gm.graph.nodes],
            )

    def test_float32_keeps_triton(self):
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, torch.float32, requires_grad=False)
        self._check_against_triton(q, k, v, declined="float16 or bfloat16")

    def test_unsupported_block_size_keeps_triton(self):
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)
        mask = create_block_mask(
            _causal_block, None, None, 256, 256, device=DEVICE, BLOCK_SIZE=64
        )
        self._check_against_triton(
            q, k, v, block_mask=mask, declined="does not fit Helion's fused CuTe"
        )

    def test_captured_tensor_grad_keeps_triton_backward(self):
        q, k, v = _tensors(2, 4, 4, 256, 256, 64)
        slopes = torch.rand(4, device=DEVICE, requires_grad=True)

        def alibi(score, b, h, q_idx, kv_idx):
            return score + slopes[h] * (kv_idx - q_idx)

        compiled = torch.compile(flex_attention)
        with self.assertLogs(LOGGER, "INFO") as captured:
            out, code = run_and_get_code(
                lambda: compiled(q, k, v, score_mod=alibi, kernel_options=TRITON)
            )
            _, bwd_code = run_and_get_code(lambda: out.sum().backward())
        self._assert_kernels("\n".join(code), forward=True, backward=False)
        self._assert_kernels("\n".join(bwd_code), forward=False, backward=False)
        self.assertRegex(
            "\n".join(captured.output), "gradients of tensors captured by score_mod"
        )
        self.assertIsNotNone(slopes.grad)
        self.assertTrue(slopes.grad.isfinite().all())

    @pytest.mark.slow
    @config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": False})
    def test_fx_graph_cache_keys_on_helion(self):
        # Helion's source, settings and configure() shape the generated code
        # but are not in the graph: a change to them must not reuse a cached
        # graph, nor may enabling or disabling, nor disallowing autotuning
        # (which keeps Triton's kernels for a kernel that would autotune).
        q, k, v = _tensors(2, 4, 4, 256, 256, 64, requires_grad=False)

        def compile_and_run():
            torch._dynamo.reset()
            torch.compile(flex_attention)(q, k, v, kernel_options=TRITON)

        def misses_and_hits():
            return (
                counters["inductor"]["fxgraph_cache_miss"],
                counters["inductor"]["fxgraph_cache_hit"],
            )

        with fresh_cache():
            compile_and_run()
            self.assertEqual(misses_and_hits(), (1, 0))
            compile_and_run()
            self.assertEqual(misses_and_hits(), (1, 1))
            with mock.patch.dict(os.environ, {"HELION_FLEX_TEST_SETTING": "1"}):
                compile_and_run()
            self.assertEqual(misses_and_hits(), (2, 1))
            with mock.patch.object(_choices, "_helion_source_key", return_value="x"):
                compile_and_run()
            self.assertEqual(misses_and_hits(), (3, 1))
            helion.flex_attention.configure(autotune_tolerance=0.05)
            compile_and_run()
            self.assertEqual(misses_and_hits(), (4, 1))
            helion.flex_attention.configure()
            with _triton_only():
                compile_and_run()
            self.assertEqual(misses_and_hits(), (5, 1))
            compile_and_run()
            self.assertEqual(misses_and_hits(), (5, 2))
            with mock.patch.dict(os.environ, {"HELION_DISALLOW_AUTOTUNING": "1"}):
                compile_and_run()
            self.assertEqual(misses_and_hits(), (6, 2))


# -- test_flex_flash.py helpers (ported from PyTorch test/inductor/test_flex_flash.py) --


def _times_two(score, _b, _h, _m, _n):
    return score * 2


def _causal(score, _b, _h, token_q, token_kv):
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _causal_mask(_b, _h, token_q, token_kv):
    return token_q >= token_kv


def _rel_bias(score, _b, _h, token_q, token_kv):
    return score + (token_q - token_kv)


def _score_squared(score, _b, _h, _q, _k):
    return score * score


def _softcap50(score, _b, _h, _q, _k):
    cap = 50.0
    return torch.tanh(score / cap) * cap


def _head_offset(score, _b, h, _q, _k):
    return score + h * 0.1


def _distance_decay(score, _b, _h, q_idx, kv_idx):
    distance_sq = (q_idx - kv_idx) * (q_idx - kv_idx)
    decay = 1.0 / (1.0 + distance_sq * 0.0001)
    return score * decay


def _sm100_bias(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=torch.float16)


# (seq_len, index_dim, bias_factory), from PyTorch's
# test_flash_attention_sm100_aux_score_mod_vec_selection.
SM100_AUX_SCORE_MOD_CASES = (
    (128, "kv", lambda seq_len: (_sm100_bias(seq_len),)),
    (128, "kv_mod_4", lambda seq_len: (_sm100_bias(4),)),
    # A table starting one element into its storage: it reaches the kernel
    # with that offset (a 2-byte base, read in pieces).
    (128, "kv_sliced_storage_offset", lambda seq_len: (_sm100_bias(seq_len + 1)[1:],)),
    (128, "kv_stride_mix", lambda seq_len: (_sm100_bias(2 * seq_len),)),
    (128, "q", lambda seq_len: (_sm100_bias(seq_len),)),
    (128, "q_kv", lambda seq_len: (_sm100_bias(seq_len, seq_len),)),
    (
        128,
        "multi",
        lambda seq_len: (
            _sm100_bias(seq_len),
            _sm100_bias(seq_len),
            _sm100_bias(seq_len, seq_len),
        ),
    ),
)


def sm100_aux_score_mod_case_name(case):
    seq_len, index_dim, _bias_factory = case
    return f"{index_dim}_seq{seq_len}"


def create_alibi_learned(num_heads=4, dtype=torch.float16):
    """ALiBi with learned per-head slopes (tests tensor loading)."""
    slopes = torch.exp2(-torch.linspace(1, 8, num_heads, device=DEVICE, dtype=dtype))

    def alibi_score_mod(score, b, h, q_idx, kv_idx):
        bias = (kv_idx - q_idx) * slopes[h]
        return score + bias

    return alibi_score_mod


def create_pos_bias_table(seq_len=512, dtype=torch.float16):
    """Relative position bias table (tests computed indexing)."""
    max_len = seq_len
    table = torch.randn(2 * max_len - 1, device=DEVICE, dtype=dtype) * 0.1

    def pos_bias_mod(score, b, h, q_idx, kv_idx):
        rel_pos = kv_idx - q_idx + max_len - 1
        bias = table[rel_pos]
        return score + bias

    return pos_bias_mod


def create_head_scale(num_heads=4, dtype=torch.float16):
    """Per-head scaling factors (tests multiplication with tensor loading)."""
    scales = torch.rand(num_heads, device=DEVICE, dtype=dtype) + 0.5

    def head_scale_mod(score, b, h, q_idx, kv_idx):
        return score * scales[h]

    return head_scale_mod


def create_batch_bias(batch_size=2, dtype=torch.float16):
    """Per-batch bias (tests batch indexing)."""
    bias = torch.randn(batch_size, device=DEVICE, dtype=dtype) * 0.1

    def batch_bias_mod(score, b, h, q_idx, kv_idx):
        return score + bias[b]

    return batch_bias_mod


def create_batch_head_bias(batch_size=2, num_heads=4, dtype=torch.float16):
    """Per-batch-head bias matrix (tests 2D indexing with batch + head)."""
    bias_matrix = torch.randn(batch_size, num_heads, device=DEVICE, dtype=dtype) * 0.5

    def batch_head_mod(score, b, h, q_idx, kv_idx):
        bias = bias_matrix[b, h]
        return score + bias

    return batch_head_mod


def create_dual_buffer_bias(num_heads=4, seq_len=512, dtype=torch.float16):
    """Dual buffer loading (tests loading from 2 separate tensors)."""
    head_bias = torch.randn(num_heads, device=DEVICE, dtype=dtype) * 0.2
    pos_scale = torch.arange(seq_len, device=DEVICE, dtype=dtype)

    def dual_buffer_mod(score, b, h, q_idx, kv_idx):
        head_component = head_bias[h]
        pos_component = pos_scale[q_idx] * 0.01
        return score + head_component + pos_component

    return dual_buffer_mod


def create_score_view_mod(num_heads=4, dtype=torch.float16, device=DEVICE):
    base_scales = torch.rand(num_heads, 2, device=device, dtype=dtype) + 0.5
    scales_view = base_scales[:, 0]
    if scales_view.is_contiguous():
        raise AssertionError("Expected non-contiguous view")

    def score_view_mod(score, _b, h, _q_idx, _kv_idx):
        return score + scales_view[h]

    return score_view_mod


def create_complex_score_mod(
    batch_size=2, num_heads=4, seq_len=512, dtype=torch.float16, device=DEVICE
):
    head_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.15
    query_scale = torch.randn(seq_len, device=device, dtype=dtype) * 0.05
    kv_scale = torch.randn(seq_len, device=device, dtype=dtype) * 0.05
    batch_bias = torch.randn(batch_size, device=device, dtype=dtype) * 0.1

    def complex_score(score, b, h, q_idx, kv_idx):
        head_term = head_bias[h]
        query_term = query_scale[q_idx] - kv_scale[kv_idx]
        batch_term = batch_bias[b]
        return score + head_term + query_term + batch_term

    return complex_score


def create_score_mod_buffer(num_heads=4, dtype=torch.float16, device=DEVICE):
    score_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.2

    def score_with_buffer(score, _b, h, _q_idx, _kv_idx):
        return score + score_bias[h]

    return score_with_buffer


def create_mask_mod_scalar_tensor(device=DEVICE):
    """mask_mod that captures a 0-dim (scalar) tensor.

    Regression test: loading a 0-dim tensor in CuTeDSL codegen produces a
    constant index 0, which was incorrectly passed as a bare Python int to
    ssa_to_indexable (expects TensorSSA).  See pytorch/pytorch#177813.
    """
    offset = torch.tensor(5, dtype=torch.int32, device=device)

    def mask_with_scalar_tensor(_b, _h, q_idx, kv_idx):
        return (q_idx + offset) >= kv_idx

    return mask_with_scalar_tensor


def create_mask_mod_buffer(num_heads=4, dtype=torch.float16, device=DEVICE):
    mask_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.1

    def custom_mask(_b, h, q_idx, kv_idx):
        bias_value = mask_bias[h]
        return (q_idx >= kv_idx) | (bias_value > 0)

    return custom_mask


def create_document_mask(lengths_per_batch, device):
    document_ids = []
    for lengths in lengths_per_batch:
        doc_tokens = []
        for doc_id, length in enumerate(lengths):
            doc_tokens.extend([doc_id] * length)
        document_ids.append(doc_tokens)
    document_ids = torch.tensor(document_ids, device=device, dtype=torch.long)

    def document_mask(b, _h, q_idx, kv_idx):
        doc_id_q = document_ids[b, q_idx // 2]
        doc_id_kv = document_ids[b, kv_idx]
        return doc_id_q == doc_id_kv

    return document_mask


def create_mask_mod_view_buffer(num_heads=4, dtype=torch.float16, device=DEVICE):
    base_bias = torch.randn(num_heads, 3, device=device, dtype=dtype)
    mask_bias_view = base_bias[:, 1]
    if mask_bias_view.is_contiguous():
        raise AssertionError("Expected non-contiguous view")

    def mask_with_view_buffer(_b, h, q_idx, kv_idx):
        bias_value = mask_bias_view[h]
        double_bias = bias_value * 2
        return (q_idx >= kv_idx) | (double_bias > 0)

    return mask_with_view_buffer


def create_mask_mod_dual_buffers(
    batch_size=2, num_heads=4, dtype=torch.float16, device=DEVICE
):
    head_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.2
    batch_bias = torch.randn(batch_size, device=device, dtype=dtype) * 0.2

    def dual_buffer_mask(b, h, q_idx, kv_idx):
        head_term = head_bias[h]
        batch_term = batch_bias[b]
        causal = q_idx >= kv_idx
        bias_cond = (head_term + batch_term).to(torch.float32) > 0
        return causal | bias_cond

    return dual_buffer_mask


def create_test_tensors(
    batch_size=2,
    num_heads=4,
    seq_len=512,
    dim=64,
    dtype=torch.float16,
    device=DEVICE,
    requires_grad=False,
    num_heads_kv=None,
):
    num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
    q_shape = (batch_size, num_heads, seq_len, dim)
    kv_shape = (batch_size, num_heads_kv, seq_len, dim)
    q = torch.randn(q_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(kv_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(kv_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def _create_block_mask_for_device(
    mask_mod, batch_size, num_heads, q_len, kv_len, *, device
):
    """Match FlexAttention's block-height expectations per compute capability."""
    q_block = _DEFAULT_SPARSE_BLOCK_SIZE
    kv_block = _DEFAULT_SPARSE_BLOCK_SIZE
    dev = torch.device(device)
    if dev.type == "cuda":
        major, _ = torch.cuda.get_device_capability(dev)
        if major == 8:
            kv_block = 64
        elif major == 10:
            q_block *= 2
    return create_block_mask(
        mask_mod,
        batch_size,
        num_heads,
        q_len,
        kv_len,
        device=device,
        BLOCK_SIZE=(q_block, kv_block),
    )


@contextmanager
def cuda_kernel_profiler():
    """The names of the CUDA kernels the block launches."""
    result = {"kernel_names": []}
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        yield result
    result["kernel_names"] = [
        evt.name
        for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name
    ]


@dataclass
class ScoreModCase:
    name: str
    score_mod_factory: Callable[[torch.dtype, str], Callable | None]
    batch_size: int = 2
    num_heads: int = 4
    num_heads_kv: int | None = None
    seq_len: int = 512
    dim: int = 64
    requires_grad: bool = False


@dataclass
class MaskModCase:
    name: str
    mask_mod_factory: Callable[[torch.dtype, str], Callable]
    batch_size: int = 2
    num_heads: int = 4
    num_heads_kv: int | None = None
    block_mask_num_heads: int | None = None
    seq_len: int = 512
    dim: int = 64
    score_mod_factory: Callable[[torch.dtype, str], Callable | None] | None = None
    requires_grad: bool = False


def score_case_name(case: ScoreModCase):
    return case.name


def mask_case_name(case: MaskModCase):
    return case.name


SCORE_MOD_CASES = [
    ScoreModCase("basic", lambda _dtype, _device: None),
    ScoreModCase("times_two", lambda _dtype, _device: _times_two),
    ScoreModCase("causal", lambda _dtype, _device: _causal),
    ScoreModCase("rel_bias", lambda _dtype, _device: _rel_bias),
    ScoreModCase("causal_unfriendly_127", lambda _dtype, _device: _causal, seq_len=127),
    ScoreModCase("causal_unfriendly_255", lambda _dtype, _device: _causal, seq_len=255),
    ScoreModCase("causal_unfriendly_383", lambda _dtype, _device: _causal, seq_len=383),
    ScoreModCase("causal_unfriendly_511", lambda _dtype, _device: _causal, seq_len=511),
    ScoreModCase(
        "alibi_learned",
        lambda dtype, device: create_alibi_learned(num_heads=4, dtype=dtype),
    ),
    ScoreModCase(
        "pos_bias_table",
        lambda dtype, device: create_pos_bias_table(seq_len=512, dtype=dtype),
    ),
    ScoreModCase(
        "head_scale",
        lambda dtype, device: create_head_scale(num_heads=4, dtype=dtype),
    ),
    ScoreModCase(
        "batch_bias",
        lambda dtype, device: create_batch_bias(batch_size=2, dtype=dtype),
    ),
    ScoreModCase(
        "batch_head_bias",
        lambda dtype, device: create_batch_head_bias(
            batch_size=2, num_heads=4, dtype=dtype
        ),
    ),
    ScoreModCase(
        "dual_buffer_bias",
        lambda dtype, device: create_dual_buffer_bias(
            num_heads=4, seq_len=512, dtype=dtype
        ),
    ),
    ScoreModCase(
        "score_view_buffer",
        lambda dtype, device: create_score_view_mod(
            num_heads=4, dtype=dtype, device=device
        ),
    ),
    ScoreModCase(
        "score_many_buffers",
        lambda dtype, device: create_complex_score_mod(
            batch_size=2, num_heads=4, seq_len=512, dtype=dtype, device=device
        ),
    ),
    ScoreModCase(
        "backward_times_two",
        lambda _dtype, _device: _times_two,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_score_squared",
        lambda _dtype, _device: _score_squared,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_rel_bias",
        lambda _dtype, _device: _rel_bias,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_softcap",
        lambda _dtype, _device: _softcap50,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_head_offset",
        lambda _dtype, _device: _head_offset,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_distance_decay",
        lambda _dtype, _device: _distance_decay,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_alibi_captured",
        lambda dtype, device: create_alibi_learned(num_heads=4, dtype=dtype),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_head_scale_captured",
        lambda dtype, device: create_head_scale(num_heads=4, dtype=dtype),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_batch_head_bias_captured",
        lambda dtype, device: create_batch_head_bias(
            batch_size=2, num_heads=4, dtype=dtype
        ),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_pos_bias_table_captured",
        lambda dtype, device: create_pos_bias_table(seq_len=257, dtype=dtype),
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "gqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=2,
    ),
    ScoreModCase(
        "gqa_causal",
        lambda _dtype, _device: _causal,
        num_heads=8,
        num_heads_kv=2,
    ),
    ScoreModCase(
        "mqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=1,
    ),
    ScoreModCase(
        "mqa_causal",
        lambda _dtype, _device: _causal,
        num_heads=8,
        num_heads_kv=1,
    ),
    ScoreModCase(
        "backward_gqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=2,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_gqa_times_two",
        lambda _dtype, _device: _times_two,
        num_heads=8,
        num_heads_kv=2,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_mqa_basic",
        lambda _dtype, _device: None,
        num_heads=8,
        num_heads_kv=1,
        seq_len=257,
        requires_grad=True,
    ),
    ScoreModCase(
        "backward_mqa_times_two",
        lambda _dtype, _device: _times_two,
        num_heads=8,
        num_heads_kv=1,
        seq_len=257,
        requires_grad=True,
    ),
]


MASK_MOD_CASES = [
    MaskModCase(
        "mask_mod_scalar_tensor",
        lambda _dtype, device: create_mask_mod_scalar_tensor(device=device),
    ),
    MaskModCase("block_mask_causal", lambda _dtype, _device: _causal_mask),
    MaskModCase(
        "block_mask_causal_score_times_two",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _times_two,
    ),
    MaskModCase(
        "mask_mod_buffer",
        lambda dtype, device: create_mask_mod_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
    ),
    MaskModCase(
        "doc_mask",
        lambda _dtype, device: create_document_mask(
            ((16, 31, 25, 56), (40, 9, 23, 56)), device=device
        ),
        block_mask_num_heads=1,
        seq_len=128,
    ),
    MaskModCase(
        "mask_mod_view_buffer",
        lambda dtype, device: create_mask_mod_view_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
        requires_grad=True,
    ),
    MaskModCase(
        "mask_mod_dual_buffers",
        lambda dtype, device: create_mask_mod_dual_buffers(
            batch_size=2, num_heads=4, dtype=dtype, device=device
        ),
    ),
    MaskModCase(
        "score_and_mask_buffers",
        lambda dtype, device: create_mask_mod_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
        score_mod_factory=lambda dtype, device: create_score_mod_buffer(
            num_heads=4, dtype=dtype, device=device
        ),
    ),
    MaskModCase(
        "backward_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_block_mask_causal_score_times_two",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _times_two,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_block_mask_causal_score_squared",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _score_squared,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_block_mask_causal_rel_bias",
        lambda _dtype, _device: _causal_mask,
        score_mod_factory=lambda _dtype, _device: _rel_bias,
        seq_len=257,
        requires_grad=True,
    ),
]


GQA_MQA_BLOCK_MASK_CASES = [
    MaskModCase(
        "gqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=1,
    ),
    MaskModCase(
        "gqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=8,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=1,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=8,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=1,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_gqa_block_mask_causal_per_head_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=2,
        block_mask_num_heads=8,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
    MaskModCase(
        "mqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=1,
    ),
    MaskModCase(
        "mqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=8,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=1,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal_per_head",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=8,
        seq_len=257,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=1,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
    MaskModCase(
        "backward_mqa_block_mask_causal_per_head_dim128",
        lambda _dtype, _device: _causal_mask,
        num_heads=8,
        num_heads_kv=1,
        block_mask_num_heads=8,
        seq_len=257,
        dim=128,
        requires_grad=True,
    ),
]


# -- test_flex_flash.py's cases ---------------------------------------------------

# The test_flex_flash.py table cases the Helion kernels leave to Triton's, and
# the reason they log.
FLASH_CASES_ON_TRITON = {
    # A score_mod that masks (torch.where(q_idx >= kv_idx, score, -inf)): the
    # fused kernels mask only on BlockMask's partial blocks.
    **dict.fromkeys(
        (
            "causal",
            "causal_unfriendly_127",
            "causal_unfriendly_255",
            "causal_unfriendly_383",
            "causal_unfriendly_511",
            "gqa_causal",
            "mqa_causal",
        ),
        "does not fit Helion's fused CuTe flash kernel",
    ),
    # A 0-dim captured tensor, read without indexing.
    "mask_mod_scalar_tensor": "uses a captured tensor other than by indexing it",
}


def _table_gather(index):
    """test_vectorized_group_untraceable_index_op's score_mod: a table gather
    at ``index(q_idx, kv_idx, seq_len)``, scaled per head."""

    def make(seq_len):
        table = torch.randn(2 * seq_len, device=DEVICE)
        head_scale = torch.randn(4, device=DEVICE)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + table[index(q_idx, kv_idx, seq_len)] * head_scale[h]

        return score_mod

    return make


def _chained_kv_gather(seq_len):
    doc_bias = torch.randn(8, device=DEVICE)
    doc_id = torch.arange(seq_len, device=DEVICE) % 8

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + doc_bias[doc_id[kv_idx]]

    return score_mod


def _int64_index(seq_len):
    table = torch.randn(seq_len, device=DEVICE)

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + table[kv_idx.to(torch.int64)]

    return score_mod


def _batch_head_query_bias(seq_len):
    bias = torch.randn(2, 4, seq_len, device=DEVICE, dtype=torch.float16)

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias[b, h, q_idx]

    return score_mod


# score_mods of TestFlexFlash's single tests (named in each comment) that read
# captured tensors at computed indices, on test_flex_flash.py's default inputs.
CAPTURED_SCORE_MODS = {
    # test_vectorized_group_per_lane_gather
    "rel_pos_gather": _table_gather(lambda q, kv, n: q - kv + n - 1),
    # test_vectorized_group_untraceable_index_op
    "abs_index": _table_gather(lambda q, kv, n: (q - kv).abs()),
    "clamp_index": _table_gather(lambda q, kv, n: (q - kv).clamp(0, n - 1)),
    "minimum_index": _table_gather(lambda q, kv, n: torch.minimum(q, kv)),
    "maximum_index": _table_gather(
        lambda q, kv, n: torch.maximum(kv - q, torch.zeros_like(kv))
    ),
    # test_chained_kv_gather
    "chained_kv_gather": _chained_kv_gather,
    # test_captured_table_int64_index
    "int64_index": _int64_index,
    # test_hierarchical_indexing_3d
    "batch_head_query_bias": _batch_head_query_bias,
}


def _fp16(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=torch.float16)


def _score_and_mask_aux(n):
    score_bias, mask_bias = _fp16(n), _fp16(n, n)

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + score_bias[kv_idx]

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) | (mask_bias[q_idx, kv_idx] > 0)

    return score_mod, mask_mod


def _rank1_mask_aux(n):
    col_mask = _fp16(n)

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (col_mask[kv_idx] > 0)

    return None, mask_mod


def _lane_uniform_mask(n):
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx < 64

    return None, mask_mod


def _sliding_window_mask(n):
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= 32)

    return None, mask_mod


def _document_offsets_mask(n):
    positions = torch.arange(n, device=DEVICE, dtype=torch.int32)
    doc_ids = (positions // 32).expand(1, -1).contiguous()
    offsets = torch.arange(0, n + 32, 32, device=DEVICE, dtype=torch.int32)
    score_bias = _fp16(n)

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + score_bias[kv_idx]

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx >= offsets[doc_ids[b, q_idx]]) & (kv_idx <= q_idx)

    return score_mod, mask_mod


def _mixed_gather_tail_mask(n):
    mask_bias, col_mask = _fp16(160, 160), _fp16(n)

    def mask_mod(b, h, q_idx, kv_idx):
        return (mask_bias[q_idx, kv_idx] > 0) & (col_mask[kv_idx] > 0)

    return None, mask_mod


# The (score_mod, mask_mod) pairs and sequence lengths of TestFlexFlash's sm100
# mask_mod tests (named in each comment), whose numerics this checks.
CAPTURED_MASK_MODS = {
    # test_flash_attention_sm100_score_and_mask_mod_vec_selection
    "score_and_mask_aux": (128, _score_and_mask_aux),
    # test_flash_attention_sm100_mask_mod_rank1_aux_stays_scalar
    "rank1_aux": (128, _rank1_mask_aux),
    # test_flash_attention_sm100_mask_mod_vec_lane_uniform_no_aux
    "lane_uniform": (128, _lane_uniform_mask),
    # test_flash_attention_sm100_sliding_window_uses_packed_shift_mask
    "sliding_window": (128, _sliding_window_mask),
    # test_flash_attention_sm100_document_offsets_packed_mask
    "document_offsets": (128, _document_offsets_mask),
    # test_flash_attention_sm100_aux_mask_mod_vec_mixed_gather_tail
    "mixed_gather_tail": (130, _mixed_gather_tail_mask),
}


@instantiate_parametrized_tests
class TestFlexHelionFlashCases(FlexHelionTestCase):
    """test_flex_flash.py's tests that apply to the Helion kernels, each
    checked against Triton's (``_check_against_triton``), for the calls they
    leave to Triton's with the reason they log. Its tables run in bfloat16
    (TestFlexHelion.test_float16 covers float16); tests of BACKEND="FLASH"'s
    generated code, its configuration knobs, or its deterministic modes are
    not ported."""

    def _flash_case(self, case, score_mod, block_mask=None):
        q, k, v = create_test_tensors(
            batch_size=case.batch_size,
            num_heads=case.num_heads,
            num_heads_kv=case.num_heads_kv,
            seq_len=case.seq_len,
            dim=case.dim,
            dtype=torch.bfloat16,
            device=DEVICE,
            requires_grad=case.requires_grad,
        )
        self._check_against_triton(
            q,
            k,
            v,
            score_mod,
            block_mask,
            declined=FLASH_CASES_ON_TRITON.get(case.name),
        )

    @pytest.mark.slow
    # test_flex_flash.py's tables, ported whole (slow tier, 58 cases): each
    # case is a score_mod, mask, GQA / MQA layout or ragged length the FLASH
    # backend is tested on.
    @parametrize("case", SCORE_MOD_CASES, name_fn=score_case_name)
    def test_score_mod_cases(self, case):
        self._flash_case(case, case.score_mod_factory(torch.bfloat16, DEVICE))

    @pytest.mark.slow
    @parametrize(
        "case", MASK_MOD_CASES + GQA_MQA_BLOCK_MASK_CASES, name_fn=mask_case_name
    )
    def test_mask_mod_cases(self, case):
        factory = case.score_mod_factory
        block_mask = _create_block_mask_for_device(
            case.mask_mod_factory(torch.bfloat16, DEVICE),
            case.batch_size,
            case.block_mask_num_heads or case.num_heads,
            case.seq_len,
            case.seq_len,
            device=DEVICE,
        )
        self._flash_case(
            case, factory(torch.bfloat16, DEVICE) if factory else None, block_mask
        )

    @pytest.mark.slow
    @parametrize("name", list(CAPTURED_SCORE_MODS))
    def test_captured_score_mods(self, name):
        q, k, v = create_test_tensors(device=DEVICE)
        self._check_against_triton(q, k, v, CAPTURED_SCORE_MODS[name](q.size(2)))

    @pytest.mark.slow
    # test_flash_attention_sm100_aux_score_mod_vec_selection's numerics (its
    # expected vector sizes, PyTorch's FLASH backend's, do not apply).
    @parametrize(
        "case", SM100_AUX_SCORE_MOD_CASES, name_fn=sm100_aux_score_mod_case_name
    )
    def test_captured_score_mod_layouts(self, case):
        seq_len, index_dim, bias_factory = case
        biases = bias_factory(seq_len)

        def score_mod(score, b, h, q_idx, kv_idx):
            match index_dim:
                case "kv" | "kv_sliced_storage_offset":
                    return score + biases[0][kv_idx]
                case "kv_mod_4":
                    return score + biases[0][kv_idx % 4]
                case "kv_stride_mix":
                    return score + biases[0][2 * kv_idx - kv_idx % 4]
                case "q":
                    return score + biases[0][q_idx]
                case "q_kv":
                    return score + biases[0][q_idx, kv_idx]
                case "multi":
                    return (
                        score
                        + biases[0][kv_idx]
                        + biases[1][q_idx]
                        + biases[2][q_idx, kv_idx]
                    )
                case _:
                    raise AssertionError(index_dim)

        q, k, v = create_test_tensors(1, 1, seq_len, 64, device=DEVICE)
        self._check_against_triton(q, k, v, score_mod)

    @pytest.mark.slow
    @parametrize("name", list(CAPTURED_MASK_MODS))
    def test_captured_mask_mods(self, name):
        seq_len, make = CAPTURED_MASK_MODS[name]
        score_mod, mask_mod = make(seq_len)
        block_mask = _create_block_mask_for_device(
            mask_mod, 1, 1, seq_len, seq_len, device=DEVICE
        )
        q, k, v = create_test_tensors(1, 1, seq_len, 64, device=DEVICE)
        self._check_against_triton(q, k, v, score_mod, block_mask)

    @pytest.mark.slow
    # test_max_autotune_score_mod: the Helion choice replaces max_autotune's
    # Triton candidates.
    @config.patch(max_autotune=True)
    @parametrize("score_mod", [_times_two, _rel_bias], name_fn=lambda f: f.__name__)
    def test_max_autotune_score_mod(self, score_mod):
        q, k, v = create_test_tensors(
            seq_len=257, dtype=torch.bfloat16, device=DEVICE, requires_grad=True
        )
        self._check_against_triton(q, k, v, score_mod)

    @pytest.mark.slow
    def test_gqa_expand_stride_zero_backward(self):
        # expand()ed K/V (a zero head stride), which the lowering copies.
        q, k, v = create_test_tensors(
            1, 4, 512, 128, torch.bfloat16, DEVICE, requires_grad=True, num_heads_kv=1
        )
        k, v = (t.expand(1, 4, 512, 128) for t in (k, v))
        self.assertEqual((k.stride(1), v.stride(1)), (0, 0))
        block_mask = _create_block_mask_for_device(
            _causal_block, 1, 4, 512, 512, device=DEVICE
        )
        self._check_against_triton(q, k, v, None, block_mask)

    def _compare_backends(self, fn, *args, grad_inputs=()):
        """Compiled ``fn(*args)`` with the Helion kernels against Triton's and
        fp32 eager, and the gradients of ``grad_inputs`` (indices into
        ``args``)."""

        def compiled(*args):
            return torch.compile(fn, fullgraph=True, dynamic=False)(*args)

        fp32 = [
            a.detach().float().requires_grad_(a.requires_grad)
            if isinstance(a, torch.Tensor) and a.is_floating_point()
            else a
            for a in args
        ]
        ref = fn(*fp32)
        with _triton_only():
            triton = compiled(*args)
        helion, code = run_and_get_code(compiled, *args)
        self._assert_kernels("\n".join(code), forward=True, backward=False)
        self._assert_within_triton_error("out", ref, triton, helion, helion.dtype)
        if grad_inputs:
            grad = torch.randn_like(ref)
            inputs = [args[i] for i in grad_inputs]
            with _triton_only():
                triton = compiled(*args)
                triton_grads = torch.autograd.grad(triton, inputs, grad)
            helion = compiled(*args)
            helion_grads, code = run_and_get_code(
                torch.autograd.grad, helion, inputs, grad
            )
            self._assert_kernels("\n".join(code), forward=False, backward=True)
            for name, r, t, h in zip(
                ("dq", "dk", "dv"),
                torch.autograd.grad(ref, [fp32[i] for i in grad_inputs], grad),
                triton_grads,
                helion_grads,
                strict=True,
            ):
                self._assert_within_triton_error(name, r, t, h, helion.dtype)

    def test_fused_qkv_reinterpret_view(self):
        # Q, K and V are strided views of one projection.
        B, M, H, D = 2, 256, 4, 64

        def fn(x, weight):
            qkv = (x @ weight).view(B, M, 3, H, D)
            q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
            return flex_attention(q, k, v, kernel_options=TRITON)

        x = torch.randn(B, M, H * D, device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn(H * D, 3 * H * D, device=DEVICE, dtype=torch.bfloat16)
        self._compare_backends(fn, x, weight / 16)

    @pytest.mark.slow
    def test_shared_captured_buffers(self):
        # test_flash_attention_shared_captured_buffers: the mask_mod reads a
        # tensor computed in the graph from the captured ones.
        B, H, Q_LEN, KV_LEN, D = 2, 1, 512, 512, 64

        def fn(q, k, v, rule_masks, prefix):
            masks = rule_masks & prefix.unsqueeze(0)

            def mask_mod(b, h, q_idx, kv_idx):
                return masks[q_idx, b, kv_idx]

            block_mask = _create_block_mask_for_device(
                mask_mod, B, 1, Q_LEN, KV_LEN, device=DEVICE
            )
            return flex_attention(q, k, v, block_mask=block_mask, kernel_options=TRITON)

        q, k, v = create_test_tensors(
            B, H, Q_LEN, D, torch.bfloat16, DEVICE, requires_grad=True
        )
        rule_masks = torch.randint(0, 2, (Q_LEN, B, KV_LEN), device=DEVICE).bool()
        prefix = torch.randint(0, 2, (B, KV_LEN), device=DEVICE).bool()
        self._compare_backends(fn, q, k, v, rule_masks, prefix, grad_inputs=(0, 1, 2))

    def test_viewable_captured_buffer_no_copy(self):
        # A captured view of an input passes as a view, with no copy kernel.
        B, H, Q_LEN, KV_LEN, D = 2, 1, 512, 512, 64

        def model(q, k, v, mask_flat):
            masks = mask_flat.view(Q_LEN, B, KV_LEN)

            def mask_mod(b, h, q_idx, kv_idx):
                return masks[q_idx, b, kv_idx]

            block_mask = _create_block_mask_for_device(
                mask_mod, B, 1, Q_LEN, KV_LEN, device=DEVICE
            )
            return flex_attention(q, k, v, block_mask=block_mask, kernel_options=TRITON)

        q, k, v = create_test_tensors(B, H, Q_LEN, D, torch.bfloat16, DEVICE)
        mask_flat = torch.randint(0, 2, (Q_LEN * B * KV_LEN,), device=DEVICE).bool()
        actual, code = run_and_get_code(
            torch.compile(model, dynamic=False), q, k, v, mask_flat
        )
        self.assertEqual(actual, model(q, k, v, mask_flat), atol=3e-2, rtol=3e-2)
        code = "\n".join(code)
        self._assert_kernels(code, forward=True, backward=False)
        self.assertIn("reinterpret_tensor(", code)
        self.assertNotIn("triton_poi_fused_view", code)

    @parametrize("case", ["offset", "stride"])
    def test_captured_alias_views_keep_distinct_layouts(self, case):
        # Two views of one captured buffer reach the kernel as two views.
        q, k, v = create_test_tensors(1, 2, 128, 64, device=DEVICE)
        if case == "offset":
            base = torch.randn(1, 2, 128, 256, device=DEVICE)
            expected = ["(65536, 32768, 256, 1), 0)", "(65536, 32768, 256, 1), 128)"]
        else:
            base = torch.randn(1, 2, 256, 128, device=DEVICE)
            expected = ["(65536, 32768, 128, 1), 0)", "(65536, 32768, 256, 1), 0)"]

        def fn(q, k, v, base):
            if case == "offset":
                left, right = base[:, :, :, :128], base[:, :, :, 128:]
            else:
                left, right = base[:, :, :128, :], base[:, :, ::2, :]

            def score_mod(score, b, h, m, n):
                return score + left[b, h, m, n] + right[b, h, m, n]

            return flex_attention(q, k, v, score_mod=score_mod, kernel_options=TRITON)

        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True), q, k, v, base
        )
        self.assertEqual(actual, fn(q, k, v, base), atol=3e-2, rtol=3e-2)
        code = "\n".join(code)
        self._assert_kernels(code, forward=True, backward=False)
        for view in expected:
            self.assertIn(view, code)

    def test_return_lse_matches_triton_and_reference(self):
        q, k, v = create_test_tensors(1, 1, 128, 32, torch.bfloat16, DEVICE)

        def lse():
            compiled = torch.compile(
                functools.partial(
                    flex_attention, scale=1.0, return_lse=True, kernel_options=TRITON
                )
            )
            return compiled(q, k, v)[1].float()

        ref = torch.logsumexp(q.float() @ k.float().transpose(-2, -1), dim=-1)
        helion_lse, code = run_and_get_code(lse)
        self._assert_kernels("\n".join(code), forward=True, backward=False)
        with _triton_only():
            triton_lse = lse()
        torch.testing.assert_close(helion_lse, ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(helion_lse, triton_lse, atol=1e-4, rtol=1e-4)

    # test_backward's loss reads the logsumexp too.
    @pytest.mark.slow
    def test_supports_grad_logsumexp(self):
        # test_flash_backend_supports_grad_logsumexp: a unit scale and the
        # logsumexp's gradient.
        q, k, v = create_test_tensors(1, 1, 128, 32, torch.bfloat16, DEVICE)
        grad_out = torch.randn_like(q)
        grad_lse = torch.randn(1, 1, 128, device=DEVICE) * 3

        def grads(fn, q, k, v):
            q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
            out, lse = fn(q, k, v, score_mod=_times_two, scale=1.0, return_lse=True)
            return torch.autograd.grad(
                (out, lse), (q, k, v), (grad_out.to(out.dtype), grad_lse)
            )

        def compiled(*args, **kwargs):
            return torch.compile(flex_attention)(*args, **kwargs, kernel_options=TRITON)

        ref = grads(flex_attention, *(t.float() for t in (q, k, v)))
        with _triton_only():
            triton = grads(compiled, q, k, v)
        helion, code = run_and_get_code(grads, compiled, q, k, v)
        self._assert_kernels("\n".join(code), forward=True, backward=True)
        for name, r, t, h in zip(("dq", "dk", "dv"), ref, triton, helion, strict=True):
            self._assert_within_triton_error(name, r, t, h, q.dtype)

    def test_kernel_called(self):
        # test_flash_attention_kernel_called and
        # test_flash_attention_backward_kernel_called: the profiled run, which
        # also compiles under the profiler, launches the Helion kernels.
        q, k, v = create_test_tensors(dim=128, device=DEVICE, requires_grad=True)
        compiled = torch.compile(flex_attention, fullgraph=True)
        with cuda_kernel_profiler() as prof:
            compiled(q, k, v, kernel_options=TRITON).sum().backward()
        names = prof["kernel_names"]
        for kernel in (HELION_FORWARD, HELION_BACKWARD):
            self.assertTrue(any(kernel in name for name in names), names)

    def test_mixed_dtypes_keep_triton(self):
        q, k, v = create_test_tensors(device=DEVICE)
        with (
            self.assertLogs(LOGGER, "INFO") as captured,
            contextlib.suppress(Exception),
        ):
            # Whether Triton's kernels accept the mix is up to them.
            torch.compile(flex_attention, fullgraph=True)(
                q,
                k.to(torch.float8_e4m3fn),
                v.to(torch.float8_e4m3fn),
                kernel_options=TRITON,
            )
        self.assertRegex("\n".join(captured.output), "must share one dtype")

    def test_inline_asm_keeps_triton(self):
        # test_inline_asm_score_mod, test_inline_asm_score_mod_pack2 and
        # test_inline_asm_mask_mod.
        bias = torch.randn(512, device=DEVICE)

        def fma(score, b, h, q_idx, kv_idx):
            return inline_asm_elementwise(
                score,
                bias[kv_idx],
                asm_str="fma.rn.f32 $0, $1, 0f40000000, $2;",
                constraints="=f,f,f",
                dtype=torch.float32,
            )

        def pack2(score, b, h, q_idx, kv_idx):
            return inline_asm_elementwise(
                score,
                asm_str="add.f32 $0, $2, $2; add.f32 $1, $3, $3;",
                constraints="=f,=f,f,f",
                dtype=torch.float32,
                pack=2,
            )

        def asm_causal(b, h, q_idx, kv_idx):
            return (
                inline_asm_elementwise(
                    q_idx.to(torch.int32),
                    kv_idx.to(torch.int32),
                    asm_str="{\n.reg .pred p;\nsetp.ge.s32 p, $1, $2;\n"
                    "selp.u32 $0, 1, 0, p;\n}",
                    constraints="=r,r,r",
                    dtype=torch.int32,
                )
                != 0
            )

        q, k, v = create_test_tensors(device=DEVICE)
        asm_mask = _create_block_mask_for_device(
            asm_causal, 2, 4, 512, 512, device=DEVICE
        )
        for score_mod, block_mask in ((fma, None), (pack2, None), (None, asm_mask)):
            # pack=2 runs only compiled, so there is no eager reference:
            # compare with Triton's kernels.

            def run(score_mod=score_mod, block_mask=block_mask):
                torch._dynamo.reset()
                return torch.compile(flex_attention)(
                    q,
                    k,
                    v,
                    score_mod=score_mod,
                    block_mask=block_mask,
                    kernel_options=TRITON,
                )

            with _triton_only():
                expected = run()
            with self.assertLogs(LOGGER, "INFO") as captured:
                actual, code = run_and_get_code(run)
            self._assert_kernels("\n".join(code), forward=False, backward=False)
            self.assertRegex("\n".join(captured.output), "NYI: inline_asm_elementwise")
            torch.testing.assert_close(actual, expected)

    @pytest.mark.slow
    def test_deterministic_algorithms_keep_triton(self):
        # torch.use_deterministic_algorithms (also warn_only) turns on
        # Inductor's deterministic mode, which rules out on-device
        # autotuning, and the fused backward accumulates grad_query with
        # atomics: both directions keep Triton's kernels
        # (test_flash_attention_backward_deterministic_* and
        # test_flash_deterministic_block_mask_warn_only_runs_nondeterministic).
        q, k, v = create_test_tensors(
            seq_len=512, dtype=torch.bfloat16, device=DEVICE, requires_grad=True
        )
        block_mask = _create_block_mask_for_device(
            _causal_block, 2, 4, 512, 512, device=DEVICE
        )
        for warn_only in (False, True):
            torch._dynamo.reset()
            with DeterministicGuard(True, warn_only=warn_only):
                self._check_against_triton(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    declined="Inductor's deterministic mode is on",
                )

    # TestFlexHelion.test_captured_tensor_grad_keeps_triton_backward covers the
    # same split in the pull request tier.
    @pytest.mark.slow
    def test_captured_tensor_requires_grad(self):
        # test_flash_impl_error_with_requires_grad: BACKEND="FLASH" raises;
        # here the forward runs the Helion kernel and the backward Triton's,
        # which computes the captured tensor's gradient.
        q, k, v = create_test_tensors(
            dtype=torch.bfloat16, device=DEVICE, requires_grad=True
        )
        bias = torch.randn(4, device=DEVICE, requires_grad=True)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[h]

        self._check_against_triton(
            q,
            k,
            v,
            score_mod,
            backward_declined="gradients of tensors captured by score_mod",
        )


@pytest.mark.slow
class TestFlexHelionDynamicShapes(FlexHelionTestCase):
    """From test_flex_flash.py's TestFlexFlashDynamicShapes. The Helion
    kernels compile for static shapes, so with helion.flex_attention enabled
    dynamic-shape graphs keep Triton's kernels: one forward and one backward
    case check that a declined plan leaves them working (the decline reason
    and values are checked by TestFlexHelion.test_dynamic_shapes_keep_triton),
    and a captured float still runs the Helion kernels with static shapes."""

    def _check_against_triton(self, *args, **kwargs):
        if kwargs.get("dynamic"):
            # TestFlexHelion.test_dynamic_shapes_keep_triton checks the reason.
            kwargs.setdefault("declined", "NYI: dynamic shapes")
            kwargs.setdefault("check_log", False)
        super()._check_against_triton(*args, **kwargs)

    def _seq_lens(self, *, requires_grad: bool) -> None:
        # Two lengths: the second call reuses the dynamic graph.
        for seq_len in (128, 256):
            q, k, v = create_test_tensors(
                seq_len=seq_len, device=DEVICE, requires_grad=requires_grad
            )
            self._check_against_triton(q, k, v, dynamic=True)

    def test_dynamic_seq_len_no_score_mod(self):
        self._seq_lens(requires_grad=False)

    def test_dynamic_backward(self):
        self._seq_lens(requires_grad=True)

    def test_captured_float_works_with_static(self):
        val = 2.0

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * val

        q, k, v = create_test_tensors(seq_len=256, device=DEVICE)
        self._check_against_triton(q, k, v, score_mod)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
