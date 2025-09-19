from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from ..autotuner.logger import classify_triton_exception
from ..autotuner.logger import format_triton_compile_failure

if TYPE_CHECKING:
    from collections.abc import Callable

    from triton.runtime.jit import JITFunction

    from .config import Config


def make_precompiler(
    fn: JITFunction[object], config: Config
) -> Callable[..., Callable[[], None]]:
    from triton.runtime.jit import find_paths_if
    from triton.runtime.jit import get_iterable_path

    from .kernel import _find_device

    def _make_precompiler(*args: object, **kwargs: object) -> Callable[[], None]:
        """
        This is based on the Triton JITFunction.run, but breaks compile into two
        parts so we can wrap it in a subprocess to handle configs that hang in
        Triton compile and never return.
        """
        device = _find_device([*args, *kwargs.values()])  # pyright: ignore[reportArgumentType]
        kwargs["debug"] = (
            kwargs.get("debug", fn.debug) or os.environ.get("TRITON_DEBUG", "0") == "1"
        )
        kernel_cache, *_, target, backend, binder = fn.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)
        key = str(specialization) + str(options)
        kernel = kernel_cache.get(key, None)
        if kernel is not None:
            return already_compiled  # cache hit

        options = backend.parse_options(kwargs)
        sigkeys = [x.name for x in fn.params]
        sigvals = [x[0] for x in specialization]
        signature = dict(zip(sigkeys, sigvals, strict=False))
        constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
        constexprs = {
            path: get_iterable_path(list(bound_args.values()), path)
            for path in constexprs
        }
        attrvals = [x[1] for x in specialization]
        attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
        attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

        def finish_it() -> None:
            src = fn.ASTSource(fn, signature, constexprs, attrs)
            # here we update the cache so if this is called in the parent we skip a extra compile

            try:
                kernel_cache[key] = fn.compile(
                    src, target=target, options=options.__dict__
                )
            except Exception as e:
                action = classify_triton_exception(e)
                if action != "debug":
                    print(format_triton_compile_failure(config, e), file=sys.stderr)
                sys.exit(1)

        return finish_it

    return _make_precompiler


def already_compiled() -> None:
    return None
