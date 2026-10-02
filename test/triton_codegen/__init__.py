"""Triton-specific tests.

Every class here carries ``@onlyBackends(["triton"])``, ``@skipIfNotTriton``,
and ``@skipIfRefEager``: the assertions check emitted Triton source, which
tileir (aliased to triton by ``matchesBackends``) may diverge from, and which
ref-eager replaces with Python source.
Portable numerics for the same kernels live in ``test/portable/``; a file here
mirrors its portable counterpart (``test_views_codegen.py`` ↔
``portable/test_views.py``). Kernels are redefined inline rather than imported
to avoid ``@helion.kernel`` running at module import time under other backends.
"""
