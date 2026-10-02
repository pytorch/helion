"""Tests of core Helion kernel language operations.

These tests cover the operations available inside ``@helion.kernel`` bodies:
views, reshapes, expands, loops, reductions, broadcasting, indexing, and
control flow.  Every test here must pass on all backends and in eager mode;
see ``test/portable/__init__.py`` for the gap-declaration convention.
"""
