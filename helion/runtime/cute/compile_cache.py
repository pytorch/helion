"""The compiled CuTe wrapper and its on-disk cache.

:class:`CompiledCuteLauncher` compiles a generated ``@cute.jit`` wrapper once
(``cute.compile``) and persists the lowered IR under ``CUTE_DSL_CACHE_DIR``
keyed by :func:`disk_cache_key`: a digest of everything that determines the
lowered module -- the generated device source, the helper sources it
compiles (``source_dependencies``), the argument schema, launch shape,
compile options, the IR-affecting ``CUTE_DSL_*`` environment and the cutlass
version.  Leaf module of the launcher: it imports ``source_dependencies`` and
``launch_descriptor`` only.
"""

from __future__ import annotations

import base64
from contextlib import suppress
import hashlib
import importlib
import inspect
import json
import logging
import os
import shlex
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from .launch_descriptor import preferred_smem_carveout
from .launch_descriptor import use_pdl
from .source_dependencies import referenced_helper_sources
from .source_dependencies import sass_check_source_dependencies
from .source_dependencies import wrapper_source_dependencies

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

log: logging.Logger = logging.getLogger(__name__)


_CUTLASS_SHUTDOWN_PATCHED = False


def patch_cutlass_jit_shutdown_unload() -> None:
    """Avoid CUDA library unload hangs during interpreter shutdown.

    On current CUTLASS DSL builds, ``CudaDialectJitModule.__del__`` unconditionally
    calls ``cudaLibraryUnload``. On B200 this can hang during Python finalization
    after a CuTe kernel has already finished executing. Skipping that unload during
    interpreter teardown lets the process exit cleanly while preserving the normal
    unload path during regular runtime GC.
    """

    global _CUTLASS_SHUTDOWN_PATCHED
    if _CUTLASS_SHUTDOWN_PATCHED:
        return

    try:
        import cutlass.cutlass_dsl.cuda_jit_executor as cuda_jit_executor
    except ImportError:
        return

    module_type = cuda_jit_executor.CudaDialectJitModule
    if getattr(module_type, "_helion_shutdown_patch", False):
        _CUTLASS_SHUTDOWN_PATCHED = True
        return

    original_del = cast("Any", module_type.__del__)

    def _helion_del(self: object) -> None:
        module = cast("Any", self)
        if sys.is_finalizing():
            with suppress(Exception):
                module._unloaded = True
            return
        original_del(module)

    module_type.__del__ = _helion_del
    module_type._helion_shutdown_patch = True
    _CUTLASS_SHUTDOWN_PATCHED = True


class CompiledCuteLauncher:
    """Lazily compile a Helion ``@cute.jit`` wrapper via ``cute.compile``.

    The first call compiles ``jit_func`` under the launcher's options
    (``compile_with_options``) to produce a compiled
    callable; subsequent calls invoke the compiled callable directly. This
    bypasses the per-launch ``@cute.jit`` argument-handling/dispatch path,
    matching Quack's pattern (see ``gemm_tvm_ffi_utils.py``). On B200 this
    collapses ~200ms of per-launch host overhead into ~0.1ms.

    When ``cache_key`` is provided, the lowered IR module of the compiled
    kernel is persisted under ``CUTE_DSL_CACHE_DIR`` and reloaded on a later
    process, skipping recompilation.  ``cute.compile`` forces the CuTe DSL's
    own ``no_cache=True`` path, so Helion drives the on-disk cache itself: it
    writes the post-pass ``ir_module`` bytecode (plus a small JSON sidecar
    holding the mangled entry symbol) and, on a hit, reconstructs a runnable
    ``CudaDialectJitCompiledFunction`` by JIT-loading the stored module.
    Any failure in the cache layer falls back to a plain ``cute.compile``.
    """

    __slots__ = (
        "_cache_key",
        "_capi_compiled",
        "_compile_options",
        "_compiled",
        "_jit_func",
        "_kernel_name",
        "_sass_checks",
    )

    def __init__(
        self,
        jit_func: object,
        compile_options: str | None,
        cache_key: str | None = None,
        sass_checks: tuple[str, ...] = (),
        kernel_name: str = "",
    ) -> None:
        self._jit_func = jit_func
        self._compile_options = compile_options
        self._compiled: object = None
        self._capi_compiled: object = None
        self._cache_key = cache_key
        # The kernel's ``_helion_cute_sass_checks``: run on the cubin of a
        # fresh compile before it is persisted or launched (a disk-cache hit
        # was verified by the process that wrote it, under the checker whose
        # source ``disk_cache_key`` folded into the key).
        self._sass_checks = sass_checks
        self._kernel_name = kernel_name

    def capi_function(self) -> object:
        """The compiled function in its C-API (``JitExecutor``) form.

        ``cute.compile`` under ``--enable-tvm-ffi`` returns a TVM-FFI function
        whose launches never create a ``JitExecutor``; only a disk-cache
        reload yields the plain ``CudaDialectJitCompiledFunction``.  The fast
        relaunch patches an executor's packed C arguments, so a fresh-compile
        process would otherwise never build it (and every call would pay the
        FFI marshalling).  JIT the host module once more (~50ms, in memory)
        into the same object a cache hit produces; the TVM-FFI function keeps
        serving the slow path.
        """
        compiled = self._compiled
        if compiled is None or not is_tvm_ffi_compiled_function(compiled):
            return compiled
        twin = self._capi_compiled
        if twin is None:
            twin = self._jit_cuda_dialect_function(
                cast("Any", compiled).ir_module,
                cast("Any", compiled).function_name,
                has_gpu_module=bool(cast("Any", compiled).has_gpu_module),
            )
            self._capi_compiled = twin
        return twin

    def __call__(self, *args: object) -> object:
        compiled = self._compiled
        if compiled is None:
            if self._cache_key is not None:
                compiled = self._reload_from_disk()
            if compiled is None:
                compiled = compile_with_options(
                    self._jit_func, args, self._compile_options
                )
                if self._sass_checks:
                    from .sass_check import verify_compiled

                    verify_compiled(compiled, self._sass_checks, self._kernel_name)
                if self._cache_key is not None:
                    self._persist_to_disk(compiled)
            self._compiled = compiled
        return cast("Any", compiled)(*args)

    def persist_compiled(self) -> None:
        """Persist the already-compiled module into the current on-disk cache dir.

        Used by ``finalize_ephemeral_cache``: the artifact written during
        autotuning died with the ephemeral dir, but the compiled module is
        still in memory and ``_cache_file_paths`` resolves the destination
        from the (now restored) ``CUTE_DSL_CACHE_DIR`` at call time.
        """
        if self._cache_key is not None and self._compiled is not None:
            self._persist_to_disk(self._compiled)

    def _cache_file_paths(self) -> tuple[str, str, str]:
        from cutlass.base_dsl.cache_helpers import get_default_generated_ir_path

        cache_dir = get_default_generated_ir_path("CUTE_DSL")
        mlir = os.path.join(cache_dir, f"cute_dsl_{self._cache_key}.mlir")
        meta = os.path.join(cache_dir, f"cute_dsl_{self._cache_key}.json")
        return cache_dir, mlir, meta

    def _persist_to_disk(self, compiled: object) -> None:
        try:
            from cutlass.base_dsl.cache_helpers import save_ir
            from cutlass.base_dsl.cache_helpers import write_bytecode_with_crc32

            ir_module = getattr(compiled, "ir_module", None)
            function_name = getattr(compiled, "function_name", None)
            if ir_module is None or function_name is None:
                return
            cache_dir, _mlir, meta = self._cache_file_paths()
            os.makedirs(cache_dir, exist_ok=True)
            save_ir(
                "CUTE_DSL",
                ir_module,
                str(self._cache_key),
                output_dir=cache_dir,
                as_bytecode=True,
                bytecode_writer=lambda f: write_bytecode_with_crc32(f, ir_module),
            )
            # Atomic sidecar with the mangled entry symbol (process-dependent,
            # so it cannot be recomputed and must be stored alongside the IR).
            tmp = f"{meta}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "function_name": function_name,
                        "has_gpu_module": bool(
                            getattr(compiled, "has_gpu_module", True)
                        ),
                    },
                    f,
                )
            os.replace(tmp, meta)
        except (ImportError, OSError):
            # Old cutlass or an unwritable cache dir; just recompile next time.
            log.debug(
                "CuTe disk-cache persist failed for key %s",
                self._cache_key,
                exc_info=True,
            )

    def _reload_from_disk(self) -> object:
        try:
            from cutlass.base_dsl.cache_helpers import load_ir
            from cutlass.base_dsl.cache_helpers import read_bytecode_and_check_crc32

            _cache_dir, mlir, meta = self._cache_file_paths()
            if not (os.path.exists(mlir) and os.path.exists(meta)):
                return None
            with open(meta) as f:
                metadata = json.load(f)
            function_name = metadata["function_name"]
            # The parsed Module holds an internal reference to the ir.Context
            # that load_ir opened, so it stays valid after load_ir returns even
            # though its ``with ir.Context()`` block has already exited.
            _, module = load_ir(
                mlir,
                asBytecode=True,
                bytecode_reader=read_bytecode_and_check_crc32,
            )
            return self._jit_cuda_dialect_function(
                module,
                function_name,
                has_gpu_module=bool(metadata.get("has_gpu_module", True)),
            )
        except Exception:
            # Any cutlass-internal change or corrupt artifact -> recompile.
            return None

    def _jit_cuda_dialect_function(
        self,
        module: object,
        function_name: str,
        *,
        has_gpu_module: bool,
    ) -> object:
        """JIT the lowered host module into a plain C-API compiled function."""
        from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction
        from cutlass.cutlass_dsl.cutlass import CuTeDSL

        if (
            self._compile_options
            and TVM_FFI_COMPILE_OPTION in self._compile_options.split()
        ):
            # Compilation imports this runtime through the SDK's FFI
            # provider. A fresh-process disk hit must also load its global
            # symbols before the cached host module is linked.
            importlib.import_module("tvm_ffi")
        dsl = CuTeDSL._get_dsl()
        engine = dsl.compiler_provider.jit(module, shared_libs=dsl.get_shared_libs())
        capi_func = engine.lookup(function_name)
        # The signature is reconstructable from the wrapper, so it does not
        # need to be persisted.
        wrapped = getattr(self._jit_func, "__wrapped__", self._jit_func)
        signature = inspect.signature(cast("Any", wrapped), eval_str=True)
        # Empty kernel_info / default extra-arg state is correct only for the
        # non-experimental ``cute.compile`` path Helion uses here; the
        # experimental DSL would populate these from module attributes.
        return CudaDialectJitCompiledFunction(
            module,
            engine,
            capi_func,
            signature,
            function_name,
            {},
            False,
            None,
            has_gpu_module=has_gpu_module,
        )


def is_tvm_ffi_compiled_function(compiled: object) -> bool:
    """Whether ``compiled`` launches through the DSL's TVM-FFI bridge.

    Looked up rather than imported: a TVM-FFI function can only exist once the
    provider module is loaded, and test doubles must not pull the DSL in.
    """
    provider = sys.modules.get("cutlass.cutlass_dsl.tvm_ffi_provider")
    return provider is not None and isinstance(
        compiled, provider.TVMFFIJitCompiledFunctionBase
    )


TVM_FFI_COMPILE_OPTION = "--enable-tvm-ffi"

# How libNVVM compiles an fp32 ``/`` (and a 16-bit one, which it divides in
# fp32).  ``-prec-div=0`` is ``div.full.f32``: within 2 ulp over the full
# range and without a slow path, what Triton's ``/`` compiles to, and the
# CuTe backend's division too.  ``-prec-div=1`` is the IEEE ``div.rn.f32``
# of ``Settings.cute_exact_division``.  A division the kernel spells as an
# NVVM intrinsic (``cute.math.div(..., rounding=...)``, ``approx=True``) keeps
# its own rounding under either.
FULL_RANGE_DIVISION_COMPILE_OPTION = "nvvm-options=-prec-div=0"
IEEE_DIVISION_COMPILE_OPTION = "nvvm-options=-prec-div=1"
_DIVISION_COMPILE_OPTIONS = (
    FULL_RANGE_DIVISION_COMPILE_OPTION,
    IEEE_DIVISION_COMPILE_OPTION,
)


def merge_default_compile_options(compile_options: str | None) -> str:
    """*compile_options* with the launcher's defaults spliced in.

    ``--enable-tvm-ffi``: the generic launcher always benefits from the FFI
    bridge (it skips CUTLASS-DSL's per-arg cast/pointer work).  The division
    option, unless the kernel chose one (``IEEE_DIVISION_COMPILE_OPTION``).
    Other flags such as ``--generate-line-info`` may already be present
    (e.g. when the autotuner picks ``tcgen05_cubin_lineinfo=True``), so we
    splice rather than replace.
    """
    tokens = compile_options.split() if compile_options is not None else []
    if TVM_FFI_COMPILE_OPTION not in tokens:
        tokens.append(TVM_FFI_COMPILE_OPTION)
    if not any(token in _DIVISION_COMPILE_OPTIONS for token in tokens):
        tokens.append(FULL_RANGE_DIVISION_COMPILE_OPTION)
    return " ".join(tokens)


def compile_with_options(
    jit_func: object, args: Sequence[object], compile_options: str | None
) -> object:
    """``cute.compile`` of *jit_func* under *compile_options*.

    The DSL's option string has no spelling for a libNVVM option, so the
    options go in as the DSL's option objects: its public classes for the
    ones a kernel launches with, and the division option as a raw pass
    option of the compile pipeline (``ExtraCompilerOpts``, what
    ``CUTE_DSL_COMPILER_OPT`` sets for a whole process).  The launcher's
    defaults are merged in here too (``merge_default_compile_options`` is
    idempotent), so no caller compiles a string with the DSL's own division.
    ``None`` compiles as written: a launcher built without options is a
    test double.
    """
    import cutlass.cute as cute

    if compile_options is None:
        return cute.compile(jit_func, *args)
    from cutlass.base_dsl.compiler import ExtraCompilerOpts

    compile_options = merge_default_compile_options(compile_options)

    options: list[object] = []
    division = []
    tokens = iter(shlex.split(compile_options))
    for token in tokens:
        if token in _DIVISION_COMPILE_OPTIONS:
            division.append(token)
        elif token == TVM_FFI_COMPILE_OPTION:
            options.append(cute.EnableTVMFFI(True))
        elif token == "--generate-line-info":
            options.append(cute.GenerateLineInfo(True))
        elif token.startswith("--ptxas-options="):
            options.append(cute.PtxasOptions(token.partition("=")[2]))
        elif token == "--opt-level":
            # ``cute_compiler_pipeline``: the level is the next token.
            level = next(tokens, "")
            if not level.isdecimal():
                raise ValueError(
                    f"CuTe compile option --opt-level needs a level: {compile_options!r}"
                )
            options.append(cute.OptLevel(int(level)))
        else:
            raise ValueError(f"Unsupported CuTe compile option: {token!r}")
    if len(division) > 1:
        raise ValueError(f"More than one division option: {compile_options!r}")
    options.append(ExtraCompilerOpts(" ".join(division)))
    return cute.compile[tuple(options)](jit_func, *args)


def function_globals(function: object) -> Mapping[str, object]:
    """The module namespace a ``@cute.jit`` / ``@cute.kernel`` object's Python
    function runs in (the generated module's globals); empty for a test double."""
    return getattr(getattr(function, "__wrapped__", function), "__globals__", {})


def cache_relevant_env() -> tuple[tuple[str, str], ...]:
    """Return CuTe DSL env vars that can change the compiled IR.

    The CuTe DSL folds *every* one of its ``CUTE_DSL_*`` env vars into its own
    module hash (e.g. ``CUTE_DSL_ENABLE_ASSERTIONS``, ``CUTE_DSL_LINEINFO``,
    ``CUTE_DSL_KEEP``, the tvm-ffi flags), so any of them can alter the
    persisted artifact.  We snapshot the whole set (so future flags are covered
    too) and only exclude the cache *location* ``CUTE_DSL_CACHE_DIR`` — that
    selects where artifacts live (autotuning uses an ephemeral dir) and must not
    affect the key.  Including an env var that does not actually affect codegen
    only costs an occasional missed cache hit, never a wrong-kernel reload.
    """
    return tuple(
        sorted(
            (k, v)
            for k, v in os.environ.items()
            if k.startswith("CUTE_DSL_") and k != "CUTE_DSL_CACHE_DIR"
        )
    )


def disk_cache_key(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    wrapper_plans: tuple[object, ...],
    cluster_shape: object,
    compile_options: str | None,
    num_sm: int | None = None,
    resident_clusters: int | None = None,
    *,
    wrapper_globals: Mapping[str, object] | None = None,
) -> str | None:
    """Compute a stable cross-process key for the on-disk CuTe compile cache.

    Returns ``None`` (disabling the on-disk cache) when the generated-source
    hash is unavailable.  The key must be computable *before* the kernel is
    compiled (so a hit can skip recompilation), so it is derived from the
    inputs that determine the lowered IR rather than from the IR itself:
    generated device-kernel source, external compiled-helper sources, the
    sources of the Helion helper modules the kernel's globals reference
    (``referenced_helper_sources``), full input
    specialization (dtypes, ranks,
    pointer alignments, baked shapes/strides, constexpr values), launch shape
    (block/cluster), CuTe compile options,
    the IR-affecting
    ``CUTE_DSL_*`` env vars (target SM arch among them), and the cutlass version.

    ``num_sm`` is the device SM count the persistent flash wrapper bakes into
    its grid clamp as a literal (``cute.compile`` lowers that literal into the
    persisted ``ir_module``).  The env-var arch capture only distinguishes the
    target *arch*, not the SM *count*, so two same-arch GPUs with different SM
    counts would otherwise collide on one on-disk artifact carrying the wrong
    grid clamp.  It is included unconditionally to match the in-memory cache
    key; for non-persistent kernels num_sm does not affect codegen, so it only
    costs an occasional cross-GPU miss, never a wrong-kernel reload.
    ``resident_clusters`` is the driver's co-resident cluster count a wrapper
    with a cluster above two CTAs bakes into that clamp instead
    (``_cute_resident_clusters``): it varies with the GPC floorsweeping of
    same-SM-count devices, so it joins the key where it is not ``None`` (the
    other kernels' keys are unchanged).  A kernel that declares SASS checks
    (``_helion_cute_sass_checks``) adds them and the checker's source
    fingerprint (``sass_check_source_dependencies``): a cache hit is trusted
    on its writer's verification, so a stricter rule must re-verify.
    """
    source_hash = getattr(cute_kernel, "_helion_cute_source_hash", None)
    if source_hash is None:
        return None
    plans = cast(
        "Sequence[dict[str, object]]",
        getattr(cute_kernel, "_helion_cute_wrapper_plans", ()),
    )
    helper_kinds = [cast("str", plan.get("kind", "")) for plan in plans]
    helper_kinds.extend(
        cast("Sequence[str]", getattr(cute_kernel, "_helion_cute_helper_kinds", ()))
    )
    helper_sources = wrapper_source_dependencies(helper_kinds)
    if helper_sources is None:
        return None
    # The helpers the kernel's globals and the wrapper's namespace (the host
    # entries and tensor-map builders the wrapper body names) reach.
    called_helper_sources = referenced_helper_sources(
        {**function_globals(cute_kernel), **(wrapper_globals or {})}
    )
    if called_helper_sources is None:
        return None
    sass_checks = tuple(
        getattr(cast("Any", cute_kernel), "_helion_cute_sass_checks", ())
    )
    checker: tuple[object, ...] = ()
    if sass_checks:
        checker_sources = sass_check_source_dependencies()
        if checker_sources is None:
            return None
        checker = ((sass_checks, checker_sources),)
    try:
        import cutlass

        cutlass_version = getattr(cutlass, "__version__", "")
    except Exception:
        cutlass_version = ""
    payload = repr(
        (
            "helion-cute-cache-v3",
            source_hash,
            helper_sources,
            called_helper_sources,
            schema_key,
            block,
            wrapper_plans,
            repr(cluster_shape),
            preferred_smem_carveout(cute_kernel, plans),
            compile_options or "",
            cache_relevant_env(),
            cutlass_version,
            num_sm,
            use_pdl(cute_kernel),
            *(() if resident_clusters is None else (resident_clusters,)),
            *checker,
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return base64.b32encode(digest).decode().rstrip("=")
