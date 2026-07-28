"""Triton-specific precompilation.

Inlines the dependency-free Triton launcher (:mod:`helion.runtime.triton.launcher`)
and rewrites the ``helion.runtime.<fn>(`` calls the generated code emits to the
inlined bare functions. The generic assembly lives in the backend-neutral
orchestrator (:mod:`helion.runtime.precompile`).
"""

from __future__ import annotations

from ..precompile import BackendPrecompiler


class TritonPrecompiler(BackendPrecompiler):
    """Precompiler for the Triton backend (also used by ``tileir``)."""

    launcher_module = "helion.runtime.triton.launcher"
    launcher_symbol = "default_launcher"
    launcher_alias = "_default_launcher"
    deps = "torch + triton"
    # get_num_sm / get_num_xcd / set_triton_allocator are launcher functions the
    # generated host wrapper calls as ``helion.runtime.<fn>(``; the standalone
    # inlines them from the launcher, so those qualified calls are rewritten to
    # the bare inlined function.
    helion_call_rewrites = ("get_num_sm", "get_num_xcd", "set_triton_allocator")
