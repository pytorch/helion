from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import cast

from .._compat import get_triton_find_paths_if
from .._compat import get_triton_iterable_path
from .._compat import get_triton_version
from .triton.launcher import compile_only_launch_args

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from triton.compiler.compiler import CompiledKernel
    from triton.runtime.jit import JITFunction

    from .config import Config
    from .kernel import BoundKernel


def _set_helion_compilation_success(kernel: CompiledKernel, status: bool) -> None:
    """
    Instead of always assuming cached kernel being properly compiled,
    attach a flag to indicate if there is any compilation failure.
    """
    kernel._helion_compilation_success = status  # pyrefly: ignore[missing-attribute]


def _get_helion_compilation_success(kernel: CompiledKernel) -> bool:
    return getattr(kernel, "_helion_compilation_success", True)


def _device_probe_values(
    values: Sequence[object], *, gluon: bool = False
) -> list[object]:
    """Launch arguments as seen by device discovery.

    A launch whose every tensor travels as a Triton (``gluon``: Gluon)
    ``TensorDescriptor`` (all ``tensor_descriptor`` indexing, no bare tensor
    argument) carries its device only inside the descriptors' ``base`` tensors.
    """
    if gluon:
        from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    else:
        from triton.tools.tensor_descriptor import TensorDescriptor

    return [
        value.base if isinstance(value, TensorDescriptor) else value for value in values
    ]


def make_precompiler(
    fn: JITFunction[object],
    config: Config,
    bound_kernel: BoundKernel,
) -> Callable[..., Callable[[], bool]]:
    from .kernel import _find_device

    def _make_precompiler(*args: object, **kwargs: object) -> Callable[[], bool]:
        """
        This is based on the Triton JITFunction.run, but breaks compile into two
        parts so we can wrap it in a subprocess to handle configs that hang in
        Triton compile and never return.
        """
        # pyrefly: ignore [bad-argument-type]
        args, kwargs = compile_only_launch_args(*args, **kwargs)
        # pyrefly: ignore [bad-argument-type]
        device = _find_device(
            _device_probe_values(
                [*args, *kwargs.values()],
                gluon=bound_kernel.env.backend_name == "gluon",
            )
        )
        kwargs["debug"] = (
            kwargs.get("debug", fn.debug) or os.environ.get("TRITON_DEBUG", "0") == "1"
        )
        kernel_cache, kernel_key_cache, target, backend, binder = fn.device_caches[
            device
        ]
        bound_args, specialization, options = binder(*args, **kwargs)
        # Keep this key and the packed compile inputs identical to
        # JITFunction.run(). Triton 3.7 changed both; using the old hand-rolled
        # forms makes the benchmark worker compile every candidate a second time.
        use_runtime_argument_packing = get_triton_version().release >= (3, 7)
        if use_runtime_argument_packing:
            from triton import knobs
            from triton.runtime.jit import compute_cache_key

            # JITFunction.run() also specializes on a stages-inspection hook's
            # pipeline (hl.prefetch_code installs one); without it the parent
            # misses the kernel this precompile compiles.
            hook = knobs.runtime.add_stages_inspection_hook
            if hook is not None:
                # Called with no arguments, the hook returns its cache key and
                # hash (PipelineStagesHook types only the stages call).
                key_hook = cast("Callable[[], tuple[str, str]]", hook)
                _, inspect_stages_hash = key_hook()
                specialization.append(f'("custom_pipeline", {inspect_stages_hash})')
            key = compute_cache_key(kernel_key_cache, specialization, options)
        else:
            key = str(specialization) + str(options)
        kernel = kernel_cache.get(key, None)
        if kernel is not None:
            return (
                already_compiled
                if _get_helion_compilation_success(kernel)
                else already_compiled_fail
            )  # cache hit

        if use_runtime_argument_packing:
            options, signature, constexprs, attrs = fn._pack_args(  # pyrefly: ignore[missing-attribute]
                backend, kwargs, bound_args, specialization, options
            )
        else:
            options = backend.parse_options(kwargs)
            sigkeys = [x.name for x in fn.params]
            sigvals = [x[0] for x in specialization]
            signature = dict(zip(sigkeys, sigvals, strict=False))
            find_paths_if = get_triton_find_paths_if()
            get_iterable_path = get_triton_iterable_path()
            constexpr_paths = cast(
                "list[tuple[int, ...]]",
                find_paths_if(sigvals, lambda _, val: val == "constexpr"),
            )
            constexprs = {
                path: get_iterable_path(list(bound_args.values()), path)
                for path in constexpr_paths
            }
            attrvals = [x[1] for x in specialization]
            attr_paths = cast(
                "list[tuple[int, ...]]",
                find_paths_if(attrvals, lambda _, x: isinstance(x, str)),
            )
            attrs = {
                k: backend.parse_attr(get_iterable_path(attrvals, k))
                for k in attr_paths
            }

        def finish_it(in_child_process: bool = True) -> bool:
            """Compile the kernel into the cache, raising the compile's error:
            it returns True only, as the cache-hit sentinel
            ``already_compiled`` does.  A kernel another thread cached
            meanwhile stays; one marked failed raises as a cached failure
            does (``cached_compile_failure_error``)."""
            compiled_kernel = None
            try:
                src = fn.ASTSource(fn, signature, constexprs, attrs)
                compiled_kernel = fn.compile(
                    src, target=target, options=options.__dict__
                )
                if not in_child_process:
                    # Loaded before it is published: a patched launch that
                    # finds it reads its function handle.
                    compiled_kernel._init_handles()
                _set_helion_compilation_success(compiled_kernel, True)
                # here we update the cache so if this is called in the parent
                # we skip a extra compile.  A kernel another thread cached in
                # the meantime stays, with its status: a launch in clusters,
                # of a resident cohort or of a cooperative grid may have
                # patched it to launch next (helion.runtime.triton.launcher's
                # _patched_launch_lock).
                cached = kernel_cache.setdefault(key, compiled_kernel)
            except Exception:
                if not in_child_process and compiled_kernel is not None:
                    _set_helion_compilation_success(compiled_kernel, False)
                    # Later lookups find it failed (``already_compiled_fail``);
                    # a kernel another thread cached stays.
                    kernel_cache.setdefault(key, compiled_kernel)
                # The precompile child reports the error and its
                # classification to the parent in its result file
                # (precompile_future._run_kernel_in_subprocess_fork); the
                # in-process compile returns it to its caller
                # (benchmark_provider._triton_compile), which counts, dumps
                # and logs it.
                raise
            if cached is not compiled_kernel and not _get_helion_compilation_success(
                cached
            ):
                raise cached_compile_failure_error()
            return True

        return finish_it

    return _make_precompiler


def cached_compile_failure_error() -> RuntimeError:
    """Why a kernel an earlier compile in this process cached but could not
    load (``already_compiled_fail``) fails again: the precompile phase and the
    in-process compile report the same error."""
    return RuntimeError(
        "an earlier compile of this kernel in this process could not load it"
    )


def already_compiled() -> bool:
    return True


def already_compiled_fail() -> bool:
    return False
