from __future__ import annotations

import linecache

from ... import exc

# cache_key -> @flyc.jit-wrapped launcher / flyc.CompiledFunction. Both grow
# unbounded across distinct (kernel, nargs, grid, threads) keys; acceptable for
# the experimental backend, but revisit with eviction if a long-lived process
# autotunes many shapes.
_flydsl_jit_cache: dict = {}
_flydsl_compiled_cache: dict = {}
_flydsl_stream: object = None  # the null stream, reused across launches


def default_flydsl_launcher(
    flydsl_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _num_threads: int = 64,
    **kwargs: object,
) -> None:
    """Default launcher for FlyDSL kernels on ROCm devices.

    @flyc.kernel can only run inside @flyc.jit, so build a launcher wrapping it in
    @flyc.jit with explicit fx.Tensor params. flyc's ASTRewriter re-parses the
    wrapper via ``inspect.getsource``, so its source must be retrievable -- we
    register it in ``linecache`` under a synthetic name rather than writing a temp
    file (no disk I/O, nothing to leak if the process dies mid-compile).
    """
    kwargs.pop("num_warps", None)
    kwargs.pop("num_stages", None)
    if kwargs:
        raise exc.BackendUnsupported(
            "flydsl", f"unexpected launcher kwargs: {sorted(kwargs)}"
        )

    n = len(args)
    gx = grid[0] if len(grid) > 0 else 1
    gy = grid[1] if len(grid) > 1 else 1
    gz = grid[2] if len(grid) > 2 else 1

    # _num_threads = 64*bm from launcher_keyword_args; default 64 = bm=1.
    # id(flydsl_kernel) is safe as a key component only because the cached closure
    # (_make(flydsl_kernel)) keeps the kernel object alive, so its id can't be
    # recycled for a different kernel while the entry lives.
    cache_key = (id(flydsl_kernel), n, gx, gy, gz, _num_threads)
    if cache_key not in _flydsl_jit_cache:
        params = ", ".join(f"_a{i}: fx.Tensor" for i in range(n))
        call = ", ".join(f"_a{i}" for i in range(n))
        src = f"""import flydsl.expr as fx
import flydsl.compiler as flyc

def _make(kernel):
    @flyc.jit
    def _launch({params}, _s: fx.Stream):
        kernel({call}).launch(
            grid=({gx}, {gy}, {gz}),
            block=({_num_threads}, 1, 1),
            stream=_s,
        )
    return _launch
"""
        fname = f"<flydsl_jit_{abs(hash(cache_key))}>"
        # Register the source so inspect.getsource (used by flyc's ASTRewriter)
        # can retrieve it from the in-memory string.
        linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
        mod_ns: dict = {}
        exec(compile(src, fname, "exec"), mod_ns)
        _flydsl_jit_cache[cache_key] = mod_ns["_make"](flydsl_kernel)

    # Persistent null stream (fx.Stream(None) -> stream 0), reused across launches.
    global _flydsl_stream
    if _flydsl_stream is None:
        import flydsl.expr as _fx  # pyrefly: ignore[missing-import]

        _flydsl_stream = _fx.Stream(None)

    # Cache a directly-callable flyc.CompiledFunction instead of re-entering the
    # @flyc.jit wrapper on every launch.
    compiled = _flydsl_compiled_cache.get(cache_key)
    if compiled is None:
        import flydsl.compiler as _flyc  # pyrefly: ignore[missing-import]

        compiled = _flyc.compile(_flydsl_jit_cache[cache_key], *args, _flydsl_stream)
        if compiled is None:
            # COMPILE_ONLY: flyc compiled the kernel but built nothing to run.
            # Not cached: a cached None would be called on the next launch.
            return
        _flydsl_compiled_cache[cache_key] = compiled
    compiled(*args, _flydsl_stream)
