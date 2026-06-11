from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfFn
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfNotTriton
import helion.language as hl
from helion.runtime import _FastLauncher
from helion.runtime.kernel import _tensor_key

triton = pytest.importorskip("triton")
from triton.runtime.driver import driver  # noqa: E402


def _get_jit_function(kernel: helion.Kernel) -> object:
    """Return the underlying ``triton.JITFunction`` behind a primed Helion kernel.

    The generated host wrapper closes over the JITFunction as
    ``_helion_<name>`` in its ``__globals__``. We scan for the unique
    JITFunction instance there so tests can install Triton-side hooks
    (e.g. ``pre_run_hooks``) on a real kernel without patching.
    """
    from triton.runtime.jit import JITFunction

    bound = next(iter(kernel._bound_kernels.values()))  # type: ignore[attr-defined]
    return next(
        v for v in bound._run.__globals__.values() if isinstance(v, JITFunction)
    )


@helion.kernel(config={"block_size": 16})
def _fast_launcher_add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1
    return out


# Large block size + 2 stages makes Triton emit vectorized 16-byte loads,
# which exposes the launcher's per-call spec lookup gap loudly (misaligned
# pointer → CUDA misaligned-address error) rather than as a silent numeric
# drift.
@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[1024], num_warps=4, num_stages=2),
)
def _fast_launcher_add_vectorized(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(out.size(0)):
        out[i] = x[i] + y[i]
    return out


@skipIfNotTriton("fast launcher is Triton-specific")
class TestFastLauncher(RefEagerTestDisabled, TestCase):
    @skipIfNotCUDA()
    @skipIfFn(
        lambda: torch.cuda.device_count() < 2,
        "test requires at least two CUDA devices",
    )
    def test_fast_launcher_uses_current_call_device(self) -> None:
        """The cached launcher must not reuse cuda:0 launch state for cuda:1."""
        _fast_launcher_add_one.reset()
        old_device = torch.cuda.current_device()
        try:
            cuda0 = torch.device("cuda:0")
            cuda1 = torch.device("cuda:1")
            with torch.cuda.device(0):
                x0 = torch.arange(16, device=cuda0, dtype=torch.float32)
                y0 = _fast_launcher_add_one(x0)
                torch.cuda.synchronize(0)
                torch.testing.assert_close(y0, x0 + 1)

            with torch.cuda.device(1):
                x1 = torch.arange(16, device=cuda1, dtype=torch.float32)
                y1 = _fast_launcher_add_one(x1)
                torch.cuda.synchronize(1)
                torch.testing.assert_close(y1, x1 + 1)
                self.assertEqual(y1.device, cuda1)
        finally:
            torch.cuda.set_device(old_device)
            _fast_launcher_add_one.reset()

    @skipIfNotCUDA()
    def test_binder_spec_differs_for_aligned_vs_unaligned_pointers(self) -> None:
        """Pre-condition for the alignment correctness gap.

        Helion's tensor key only tracks dtype/shape/stride, so two
        views with the same shape but different pointer alignment
        share one ``BoundKernel``. Yet Triton's binder reports
        distinct specializations for them. The multi-spec
        ``_FastLauncher`` keys its ``_spec_cache`` on the alignment
        bitmask exactly to handle this: aligned and unaligned land
        on different cache entries (i.e. different compiled binaries).
        This test pins down the underlying Triton-binder distinction
        — if a future Triton change collapses these specs, alignment
        bitmask differentiation would become unnecessary.
        """
        _fast_launcher_add_vectorized.reset()
        n = 1024
        base = torch.randn(n + 4, device=DEVICE, dtype=torch.float32)
        aligned = base[:n]
        unaligned = base[1 : n + 1]
        out_buf = torch.empty_like(aligned)

        self.assertEqual(aligned.data_ptr() % 16, 0)
        self.assertNotEqual(unaligned.data_ptr() % 16, 0)
        self.assertEqual(
            _tensor_key(_fast_launcher_add_vectorized, aligned),
            _tensor_key(_fast_launcher_add_vectorized, unaligned),
        )

        # Run once to ensure ``set_config`` has installed a fast launcher.
        _fast_launcher_add_vectorized(aligned, aligned)
        bound = _fast_launcher_add_vectorized.bind((aligned, aligned))
        compiled = bound._run
        self.assertIsNotNone(compiled)
        launcher = compiled.__kwdefaults__.get("_launcher")
        if not isinstance(launcher, _FastLauncher) or not launcher._primed:
            self.skipTest(
                "fast launcher not installed (non-Triton backend or unsupported Triton)"
            )

        # Fetch the binder from Triton's per-device cache directly —
        # the fast launcher no longer keeps a reference because it
        # computes the alignment bits inline instead of calling the
        # binder on every launch.
        jit_fn = _get_jit_function(_fast_launcher_add_vectorized)
        dev = driver.active.get_current_device()
        binder = jit_fn.device_caches[dev][4]  # pyrefly: ignore[missing-attribute]
        _, spec_aligned, _ = binder(aligned, aligned, out_buf, **launcher._run_kwargs)
        _, spec_unaligned, _ = binder(
            unaligned, unaligned, out_buf, **launcher._run_kwargs
        )
        self.assertNotEqual(spec_aligned, spec_unaligned)

    @skipIfNotCUDA()
    def test_pre_run_hook_installed_after_priming_fires_on_real_kernel(
        self,
    ) -> None:
        """A ``pre_run_hook`` installed after priming must fire on the next call.

        ``JITFunction.run`` iterates ``self.pre_run_hooks`` at the top
        of every launch (jit.py:717-718). The fast launcher's
        C-launcher hot path doesn't go through ``JITFunction.run``, so
        an attached profiler / autotune-timer would silently miss
        every Helion launch unless we mirror that loop ourselves.
        We invoke the hooks inline on the hot path with the same
        ``args`` + ``kwargs`` Triton would, so installing a hook
        after priming keeps the launcher on the fast path AND fires
        the hook. We also check the kwargs include the live
        ``debug``/``instrumentation_mode`` values matching Triton's
        contract.
        """
        _fast_launcher_add_one.reset()
        self.addCleanup(_fast_launcher_add_one.reset)
        x = torch.arange(16, device=DEVICE, dtype=torch.float32)

        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
        jit_fn = _get_jit_function(_fast_launcher_add_one)
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.addCleanup(jit_fn.pre_run_hooks.clear)  # pyrefly: ignore[missing-attribute]
        jit_fn.pre_run_hooks.append(  # pyrefly: ignore[missing-attribute]
            lambda *a, **kw: calls.append((a, kw))
        )
        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
        self.assertEqual(len(calls), 1)
        _, kw = calls[0]
        self.assertIn("debug", kw)
        self.assertIn("instrumentation_mode", kw)
        self.assertIn("num_warps", kw)
        self.assertIn("num_stages", kw)

    @skipIfNotCUDA()
    def test_used_global_vals_mutation_surfaces_to_user_on_real_kernel(
        self,
    ) -> None:
        """Mutating a tracked global must surface as a Triton ``RuntimeError``.

        Helion-generated Triton kernels reference codegen constants
        (e.g. ``_BLOCK_SIZE_0``) as module-level ``constexpr`` globals;
        Triton tracks them in ``used_global_vals`` and raises if the
        value changes between launches — the cached binary has the
        old value baked into codegen, so silently reusing it would
        produce wrong output. The fast launcher's snapshot+equality
        check on every call mirrors that semantic. Without it, the
        Helion kernel would skip Triton's mismatch check entirely.

        This test also verifies the snapshot check does not
        false-positive: ``used_global_vals`` is non-empty on every
        Helion kernel, so a naive non-empty test would defeat the
        fast path. We assert that no recompile happens on a
        non-mutation call (kernel_cache unchanged) and that the fast
        path is restored after the global is reverted.
        """
        _fast_launcher_add_one.reset()
        self.addCleanup(_fast_launcher_add_one.reset)
        x = torch.arange(16, device=DEVICE, dtype=torch.float32)

        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
        jit_fn = _get_jit_function(_fast_launcher_add_one)
        (name, _gid), (compile_value, gdict) = next(
            iter(jit_fn.used_global_vals.items())  # pyrefly: ignore[missing-attribute]
        )
        dev = driver.active.get_current_device()
        kernel_cache = jit_fn.device_caches[dev][0]  # pyrefly: ignore[missing-attribute]

        # No-mutation call: cache unchanged (proves no false-positive
        # fallback / recompile despite ``used_global_vals`` being
        # non-empty).
        cache_size = len(kernel_cache)
        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
        self.assertEqual(len(kernel_cache), cache_size)

        # Mutation: hot path's per-entry equality check detects the
        # change, falls back to ``default_launcher`` -> ``JITFunction.run``
        # which raises rather than silently using the stale binary.
        # Register restore via addCleanup so a mid-test crash cannot
        # leave the global mutated.
        import triton.language as tl

        self.addCleanup(gdict.__setitem__, name, compile_value)
        gdict[name] = tl.constexpr(8)
        with self.assertRaises(RuntimeError):
            _fast_launcher_add_one(x)
        gdict[name] = compile_value

        # Restore: fast path is usable again; still no recompile.
        cache_size = len(kernel_cache)
        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
        self.assertEqual(len(kernel_cache), cache_size)

    @skipIfNotCUDA()
    def test_knobs_runtime_debug_set_after_priming_compiles_new_binary_on_real_kernel(
        self,
    ) -> None:
        """Enabling ``knobs.runtime.debug`` mid-run must trigger Triton recompile.

        ``JITFunction.run`` ORs ``knobs.runtime.debug`` into the per-call
        ``debug`` kwarg, which becomes part of Triton's cache key.
        Setting ``debug=True`` after priming needs a different compiled
        binary than the one we primed with. The fast launcher's hot
        path inspects the knob per call and falls back when set; this
        gives ``JITFunction.run`` the chance to compile and cache the
        debug binary. We assert the kernel_cache grew end-to-end.
        """
        _fast_launcher_add_one.reset()
        self.addCleanup(_fast_launcher_add_one.reset)
        x = torch.arange(16, device=DEVICE, dtype=torch.float32)
        old_debug = triton.knobs.runtime.debug
        self.addCleanup(lambda: setattr(triton.knobs.runtime, "debug", old_debug))
        triton.knobs.runtime.debug = False

        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
        jit_fn = _get_jit_function(_fast_launcher_add_one)
        dev = driver.active.get_current_device()
        kernel_cache = jit_fn.device_caches[dev][0]  # pyrefly: ignore[missing-attribute]
        cache_size = len(kernel_cache)

        triton.knobs.runtime.debug = True
        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
        self.assertGreater(len(kernel_cache), cache_size)

    @skipIfNotCUDA()
    def test_launch_hooks_installed_after_priming_fire_on_real_kernel(
        self,
    ) -> None:
        """``launch_enter_hook`` / ``launch_exit_hook`` installed after priming must fire.

        The Triton C launcher invokes both hooks with a ``launch_metadata``
        argument. The fast launcher used to snapshot the hook references
        at priming, so profilers attaching afterwards were silently
        dropped. The hot path now re-reads them from
        ``triton.knobs.runtime`` per call — we assert the hooks fire on
        a real kernel here.
        """
        _fast_launcher_add_one.reset()
        self.addCleanup(_fast_launcher_add_one.reset)
        x = torch.arange(16, device=DEVICE, dtype=torch.float32)
        old_enter = triton.knobs.runtime.launch_enter_hook
        old_exit = triton.knobs.runtime.launch_exit_hook

        def restore_launch_hooks() -> None:
            triton.knobs.runtime.launch_enter_hook = old_enter
            triton.knobs.runtime.launch_exit_hook = old_exit

        self.addCleanup(restore_launch_hooks)
        triton.knobs.runtime.launch_enter_hook = None
        triton.knobs.runtime.launch_exit_hook = None

        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
        enter_md: list[object] = []
        exit_md: list[object] = []
        triton.knobs.runtime.launch_enter_hook = lambda md: enter_md.append(md)
        triton.knobs.runtime.launch_exit_hook = lambda md: exit_md.append(md)
        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
        self.assertEqual(len(enter_md), 1)
        self.assertEqual(len(exit_md), 1)

    @skipIfNotCUDA()
    def test_unaligned_call_after_aligned_priming_matches_reference(self) -> None:
        """Aligned-then-unaligned call should still produce correct output.

        Before the multi-spec design, the second call launched the
        binary compiled for the aligned spec (vectorized 16-byte loads)
        and tripped ``CUDA error: misaligned address``. The multi-spec
        :class:`_FastLauncher` keys its ``_spec_cache`` on the
        alignment bitmask, so aligned and unaligned land on separate
        cache entries and each gets its own correctly-compiled binary.
        """
        _fast_launcher_add_vectorized.reset()
        n = 1024
        base = torch.randn(n + 4, device=DEVICE, dtype=torch.float32)
        aligned = base[:n]
        unaligned = base[1 : n + 1]

        torch.testing.assert_close(
            _fast_launcher_add_vectorized(aligned, aligned), aligned + aligned
        )
        torch.testing.assert_close(
            _fast_launcher_add_vectorized(unaligned, unaligned), unaligned + unaligned
        )

    @skipIfNotCUDA()
    def test_alternating_specs_both_take_fast_path(self) -> None:
        """Two different alignment specs should each land on the fast path.

        The multi-spec ``_FastLauncher`` caches one compiled binary per
        spec key (alignment + knob state), so a call site that
        alternates between aligned and unaligned tensors has BOTH
        binaries in ``_spec_cache`` after the first call of each. Every
        subsequent call hits the cache and uses the fast path; there is
        no fallback to ``default_launcher`` for either spec.

        This is the user-visible win over the single-spec design: a
        legitimately multi-spec call site no longer pays the
        ``default_launcher`` Python-overhead penalty on every other call.
        """
        _fast_launcher_add_vectorized.reset()
        n = 1024
        base = torch.randn(n + 4, device=DEVICE, dtype=torch.float32)
        aligned = base[:n]
        unaligned = base[1 : n + 1]

        # First call of each spec → cache miss → compile → cache hit
        torch.testing.assert_close(
            _fast_launcher_add_vectorized(aligned, aligned), aligned + aligned
        )
        torch.testing.assert_close(
            _fast_launcher_add_vectorized(unaligned, unaligned),
            unaligned + unaligned,
        )

        bound = _fast_launcher_add_vectorized.bind((aligned, aligned))
        launcher = bound._run.__kwdefaults__.get("_launcher")
        self.assertIsInstance(launcher, _FastLauncher)
        self.assertEqual(len(launcher._spec_cache), 2)

        # Alternating calls — each spec hits its own cache entry; cache
        # size stays at 2.
        for _ in range(3):
            torch.testing.assert_close(
                _fast_launcher_add_vectorized(aligned, aligned), aligned + aligned
            )
            torch.testing.assert_close(
                _fast_launcher_add_vectorized(unaligned, unaligned),
                unaligned + unaligned,
            )
        self.assertEqual(len(launcher._spec_cache), 2)

    @skipIfNotCUDA()
    def test_skip_fast_launcher_env_var_disables_install(self) -> None:
        """``HELION_SKIP_FAST_LAUNCHER=1`` should bypass the fast launcher.

        Useful as a quick escape hatch when debugging: every Helion
        call routes through ``default_launcher`` (Triton's
        ``JITFunction.run``) and the per-spec cache layer is never
        installed. We verify the kwdefault on the generated wrapper
        is the plain ``default_launcher`` function, not a
        ``_FastLauncher`` instance.
        """
        import os

        _fast_launcher_add_one.reset()
        self.addCleanup(_fast_launcher_add_one.reset)
        old = os.environ.get("HELION_SKIP_FAST_LAUNCHER")
        self.addCleanup(
            lambda: (
                os.environ.__setitem__("HELION_SKIP_FAST_LAUNCHER", old)
                if old is not None
                else os.environ.pop("HELION_SKIP_FAST_LAUNCHER", None)
            )
        )
        os.environ["HELION_SKIP_FAST_LAUNCHER"] = "1"

        x = torch.arange(16, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)

        bound = next(iter(_fast_launcher_add_one._bound_kernels.values()))
        launcher = bound._run.__kwdefaults__.get("_launcher")
        self.assertNotIsInstance(launcher, _FastLauncher)


if __name__ == "__main__":
    unittest.main()
