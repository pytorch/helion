from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

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
from helion.runtime import build_fast_launcher
from helion.runtime.kernel import _tensor_key

triton = pytest.importorskip("triton")
from triton.runtime.driver import driver  # noqa: E402


class _FakeActiveDriver:
    def get_current_device(self) -> str:
        return "fake-device"

    def get_current_stream(self, device: object) -> str:
        return f"stream-for-{device}"


class _FakeCompiledKernel:
    def __init__(self) -> None:
        self.compiled_run_published = threading.Event()
        self.packed_metadata = "packed"
        self.launch_metadata = lambda grid, stream, *args: "metadata"

    def run(self, *args: object) -> tuple[str, tuple[object, ...]]:
        return ("run", args)

    @property
    def function(self) -> str:
        # _FastLauncher used to publish _compiled_run immediately before this
        # attribute read. Sleeping here makes that partial initialization window
        # deterministic for the racing thread.
        self.compiled_run_published.set()
        time.sleep(0.2)
        return "function"


class _FakeTritonKernel:
    def __init__(self, compiled: _FakeCompiledKernel) -> None:
        self.compiled = compiled
        self.device_caches = {"fake-device": (None, None, None, None, self.binder)}
        # Match the JITFunction surface that FastLauncher reads per-call.
        self.pre_run_hooks: list[object] = []
        self.used_global_vals: dict[
            tuple[str, int], tuple[object, dict[str, object]]
        ] = {}

    def binder(
        self, *args: object, **kwargs: object
    ) -> tuple[dict[int, object], None, None]:
        return dict(enumerate(args)), None, None

    def run(self, *args: object, **kwargs: object) -> object:
        if kwargs.get("warmup"):
            return self.compiled
        return ("default", args)


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
    def test_first_launch_prime_is_thread_safe(self) -> None:
        """Concurrent first calls must not observe partially primed launch state."""
        compiled = _FakeCompiledKernel()
        triton_kernel = _FakeTritonKernel(compiled)
        launcher = build_fast_launcher(num_warps=4, num_stages=3)
        errors: list[tuple[str, str, str]] = []
        results: list[tuple[str, object]] = []

        old_active = driver._active
        old_default = driver._default
        old_enter = triton.knobs.runtime.launch_enter_hook
        old_exit = triton.knobs.runtime.launch_exit_hook
        driver._active = _FakeActiveDriver()
        driver._default = _FakeActiveDriver()
        triton.knobs.runtime.launch_enter_hook = None
        triton.knobs.runtime.launch_exit_hook = None
        try:

            def first_call() -> None:
                try:
                    results.append(("first", launcher(triton_kernel, (1,), "x")))
                except Exception as e:
                    errors.append(("first", type(e).__name__, str(e)))

            def racing_call() -> None:
                self.assertTrue(compiled.compiled_run_published.wait(timeout=2))
                try:
                    results.append(("racer", launcher(triton_kernel, (1,), "y")))
                except Exception as e:
                    errors.append(("racer", type(e).__name__, str(e)))

            threads = [
                threading.Thread(target=first_call),
                threading.Thread(target=racing_call),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            driver._active = old_active
            driver._default = old_default
            triton.knobs.runtime.launch_enter_hook = old_enter
            triton.knobs.runtime.launch_exit_hook = old_exit

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)

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
            with torch.cuda.device(0):
                x0 = torch.arange(16, device="cuda:0", dtype=torch.float32)
                y0 = _fast_launcher_add_one(x0)
                torch.cuda.synchronize(0)
                torch.testing.assert_close(y0, x0 + 1)

            with torch.cuda.device(1):
                x1 = torch.arange(16, device="cuda:1", dtype=torch.float32)
                y1 = _fast_launcher_add_one(x1)
                torch.cuda.synchronize(1)
                torch.testing.assert_close(y1, x1 + 1)
                self.assertEqual(y1.device, torch.device("cuda:1"))
        finally:
            torch.cuda.set_device(old_device)
            _fast_launcher_add_one.reset()

    @skipIfNotCUDA()
    def test_binder_spec_differs_for_aligned_vs_unaligned_pointers(self) -> None:
        """Pre-condition for the alignment correctness gap.

        ``_FastLauncher.__call__`` recomputes ``self._binder(...)`` on
        every hot call but discards the returned ``_spec``, always
        launching the kernel captured at priming time. Helion's tensor
        key only tracks dtype/shape/stride, so two views with the same
        shape but different pointer alignment share one
        ``BoundKernel`` and one ``_FastLauncher`` — yet Triton would
        normally compile them as distinct specializations. This test
        pins down that the binder still reports distinct specs, so
        the launcher's "ignore ``_spec``" shortcut is observably
        unsafe. If a future Triton change collapses these specs, this
        assertion stops holding and the related correctness test below
        can be retired.
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
        if not isinstance(launcher, _FastLauncher) or launcher._binder is None:
            self.skipTest(
                "fast launcher not installed (non-Triton backend or unsupported Triton)"
            )

        _, spec_aligned, _ = launcher._binder(
            aligned, aligned, out_buf, **launcher._run_kwargs
        )
        _, spec_unaligned, _ = launcher._binder(
            unaligned, unaligned, out_buf, **launcher._run_kwargs
        )
        self.assertNotEqual(spec_aligned, spec_unaligned)

    def _patched_driver_and_hooks(self) -> tuple[object, object, object, object]:
        """Snapshot driver + launch hooks so tests can stub them safely."""
        old_active = driver._active
        old_default = driver._default
        old_enter = triton.knobs.runtime.launch_enter_hook
        old_exit = triton.knobs.runtime.launch_exit_hook
        driver._active = _FakeActiveDriver()
        driver._default = _FakeActiveDriver()
        triton.knobs.runtime.launch_enter_hook = None
        triton.knobs.runtime.launch_exit_hook = None
        return old_active, old_default, old_enter, old_exit

    def _restore_driver_and_hooks(
        self, saved: tuple[object, object, object, object]
    ) -> None:
        old_active, old_default, old_enter, old_exit = saved
        driver._active = old_active
        driver._default = old_default
        triton.knobs.runtime.launch_enter_hook = old_enter
        triton.knobs.runtime.launch_exit_hook = old_exit

    @skipIfNotCUDA()
    def test_pre_run_hook_installed_after_priming_fires_on_real_kernel(
        self,
    ) -> None:
        """A ``pre_run_hook`` installed after priming must fire on the next call.

        ``JITFunction.run`` iterates ``self.pre_run_hooks`` at the top
        of every launch. The fast launcher's cached C-launcher path
        skips them, so an attached profiler / autotune-timer would
        silently miss every Helion launch. The hot path inspects
        ``triton_kernel.pre_run_hooks`` per call and falls back when
        non-empty so the hook actually runs — and we assert that here
        end-to-end on a real kernel.
        """
        _fast_launcher_add_one.reset()
        x = torch.arange(16, device=DEVICE, dtype=torch.float32)
        try:
            torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
            jit_fn = _get_jit_function(_fast_launcher_add_one)
            calls: list[bool] = []
            jit_fn.pre_run_hooks.append(  # pyrefly: ignore[missing-attribute]
                lambda *a, **kw: calls.append(True)
            )
            try:
                torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
                self.assertEqual(len(calls), 1)
            finally:
                jit_fn.pre_run_hooks.clear()  # pyrefly: ignore[missing-attribute]
        finally:
            _fast_launcher_add_one.reset()

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
        x = torch.arange(16, device=DEVICE, dtype=torch.float32)
        try:
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
            import triton.language as tl

            gdict[name] = tl.constexpr(8)
            try:
                with self.assertRaises(RuntimeError):
                    _fast_launcher_add_one(x)
            finally:
                gdict[name] = compile_value

            # Restore: fast path is usable again; still no recompile.
            cache_size = len(kernel_cache)
            torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
            self.assertEqual(len(kernel_cache), cache_size)
        finally:
            _fast_launcher_add_one.reset()

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
        x = torch.arange(16, device=DEVICE, dtype=torch.float32)
        old_debug = triton.knobs.runtime.debug
        triton.knobs.runtime.debug = False
        try:
            torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
            jit_fn = _get_jit_function(_fast_launcher_add_one)
            dev = driver.active.get_current_device()
            kernel_cache = jit_fn.device_caches[dev][0]  # pyrefly: ignore[missing-attribute]
            cache_size = len(kernel_cache)

            triton.knobs.runtime.debug = True
            torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
            self.assertGreater(len(kernel_cache), cache_size)
        finally:
            triton.knobs.runtime.debug = old_debug
            _fast_launcher_add_one.reset()

    def test_knobs_instrumentation_mode_set_after_priming_takes_fallback(
        self,
    ) -> None:
        """Setting ``knobs.compilation.instrumentation_mode`` mid-run skips fast path."""
        compiled = _FakeCompiledKernel()
        triton_kernel = _FakeTritonKernel(compiled)
        launcher = build_fast_launcher(num_warps=4, num_stages=3)
        saved = self._patched_driver_and_hooks()
        old_mode = triton.knobs.compilation.instrumentation_mode
        triton.knobs.compilation.instrumentation_mode = ""
        try:
            first = launcher(triton_kernel, (1,), "x")
            self.assertEqual(first[0], "run")
            triton.knobs.compilation.instrumentation_mode = "profile"
            second = launcher(triton_kernel, (1,), "y")
            self.assertEqual(second[0], "default")
        finally:
            triton.knobs.compilation.instrumentation_mode = old_mode
            self._restore_driver_and_hooks(saved)

    def test_concurrent_first_calls_serialize_priming(self) -> None:
        """Two threads racing through ``_prime`` must not both publish state.

        ``_FastLauncher`` previously gated priming with a bare
        ``if not self._primed`` check, so two threads concurrently
        making their first call could both enter ``_prime``. The fast
        path then reads ``_triton_function``, ``_packed_metadata``,
        ``_priming_spec``, ``_used_global_checks`` and ``_compiled_run``
        without synchronization — if those attribute writes from two
        primings interleave, a later call dispatches with a Frankenstein
        mix from two different priming runs.

        We widen the race window deterministically by wrapping
        ``_FastLauncher._prime`` with a delay so the first thread to
        enter waits for the second to also pass ``if not self._primed``.
        With the prime lock, the second thread is blocked at the lock
        and re-checks ``_primed`` after acquiring it — only ONE thread
        ever runs the priming body. We assert that here.
        """

        class _ConcurrentFakeCompiled:
            def __init__(self, tag: str) -> None:
                self.tag = tag
                self.packed_metadata = ("packed", tag)
                self.launch_metadata = lambda grid, stream, *args: ("md", tag)

            def run(self, *args: object) -> tuple[str, str, tuple[object, ...]]:
                return ("run", self.tag, args)

            @property
            def function(self) -> tuple[str, str]:
                return ("function", self.tag)

        class _RacingTritonKernel:
            def __init__(self) -> None:
                self.device_caches = {
                    "fake-device": (None, None, None, None, self.binder)
                }
                self.pre_run_hooks: list[object] = []
                self.used_global_vals: dict[
                    tuple[str, int], tuple[object, dict[str, object]]
                ] = {}
                self._tag_counter = 0
                self._tag_lock = threading.Lock()

            def binder(
                self, *args: object, **kwargs: object
            ) -> tuple[dict[int, object], None, None]:
                return dict(enumerate(args)), None, None

            def run(self, *args: object, **kwargs: object) -> object:
                if kwargs.get("warmup"):
                    with self._tag_lock:
                        self._tag_counter += 1
                        tag = f"compiled-{self._tag_counter}"
                    return _ConcurrentFakeCompiled(tag)
                return ("default", args)

        triton_kernel = _RacingTritonKernel()
        launcher = build_fast_launcher(num_warps=4, num_stages=3)

        orig_prime = _FastLauncher._prime
        priming_body_entries = [0]
        entries_lock = threading.Lock()
        # The first thread waits to give the second thread a chance to
        # also reach the ``if not self._primed`` check, so both would
        # enter ``_prime`` under the BROKEN (no-lock) implementation.
        both_threads_started = threading.Event()
        threads_started = [0]

        def delayed_prime(self: object, *p_args: object, **p_kwargs: object) -> None:
            with entries_lock:
                priming_body_entries[0] += 1
            orig_prime(self, *p_args, **p_kwargs)

        def synchronized_call(tag: str) -> None:
            # Mark this thread as ready, then wait for the partner
            # before invoking the launcher. This ensures both threads
            # hit ``__call__`` essentially simultaneously.
            with entries_lock:
                threads_started[0] += 1
                if threads_started[0] == 2:
                    both_threads_started.set()
            both_threads_started.wait(timeout=2)
            launcher(triton_kernel, (1,), tag)

        saved = self._patched_driver_and_hooks()
        errors: list[str] = []

        def call_first(tag: str) -> None:
            try:
                synchronized_call(tag)
            except Exception as e:
                errors.append(f"{tag}: {type(e).__name__}: {e}")

        try:
            with mock.patch.object(_FastLauncher, "_prime", delayed_prime):
                threads = [
                    threading.Thread(target=call_first, args=("x",)),
                    threading.Thread(target=call_first, args=("y",)),
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        finally:
            self._restore_driver_and_hooks(saved)

        self.assertEqual(errors, [])
        # Under the lock, exactly one thread enters the priming body.
        # Without the lock, both threads would (since both passed the
        # ``if not self._primed`` check before either ran ``_prime``).
        self.assertEqual(priming_body_entries[0], 1)
        # Sanity-check the published state is internally consistent.
        triton_function = launcher._triton_function
        self.assertIsNotNone(triton_function)
        self.assertEqual(triton_function[0], "function")
        function_tag = triton_function[1]  # type: ignore[index]
        self.assertEqual(launcher._packed_metadata, ("packed", function_tag))
        self.assertEqual(launcher._compiled_kernel.tag, function_tag)  # type: ignore[attr-defined]

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
        x = torch.arange(16, device=DEVICE, dtype=torch.float32)
        old_enter = triton.knobs.runtime.launch_enter_hook
        old_exit = triton.knobs.runtime.launch_exit_hook
        triton.knobs.runtime.launch_enter_hook = None
        triton.knobs.runtime.launch_exit_hook = None
        try:
            torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
            enter_md: list[object] = []
            exit_md: list[object] = []
            triton.knobs.runtime.launch_enter_hook = lambda md: enter_md.append(md)
            triton.knobs.runtime.launch_exit_hook = lambda md: exit_md.append(md)
            torch.testing.assert_close(_fast_launcher_add_one(x), x + 1)
            self.assertEqual(len(enter_md), 1)
            self.assertEqual(len(exit_md), 1)
        finally:
            triton.knobs.runtime.launch_enter_hook = old_enter
            triton.knobs.runtime.launch_exit_hook = old_exit
            _fast_launcher_add_one.reset()

    @skipIfNotCUDA()
    def test_unaligned_call_after_aligned_priming_matches_reference(self) -> None:
        """Aligned-then-unaligned call should still produce correct output.

        Before the spec-aware hot path, the second call launched the
        binary compiled for the aligned spec (vectorized 16-byte loads)
        and tripped ``CUDA error: misaligned address``. With the spec
        check in :class:`_FastLauncher.__call__`, the unaligned call
        observes a different ``_spec`` from the binder and falls back
        to ``default_launcher``, which routes through Triton's
        per-spec ``kernel_cache`` and picks (or compiles) the
        unaligned binary.
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


if __name__ == "__main__":
    unittest.main()
