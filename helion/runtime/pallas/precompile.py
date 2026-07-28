"""Pallas-specific precompilation: the torch-tensor standalone.

Inlines the dependency-free Pallas launcher
(:mod:`helion.runtime.pallas.launcher`) via the backend-neutral default flow. The
generic assembly helpers live in the orchestrator
(:mod:`helion.runtime.precompile`).
"""

from __future__ import annotations

from ..precompile import BackendPrecompiler


class PallasPrecompiler(BackendPrecompiler):
    """Precompiler for the Pallas backend (torch-tensor standalone)."""

    launcher_module = "helion.runtime.pallas.launcher"
    launcher_symbol = "default_pallas_launcher"
    launcher_alias = "_default_pallas_launcher"
    deps = "torch + jax"
    helion_call_rewrites = ()
