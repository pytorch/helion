"""Pinned golden files for CuTe codegen byte-identity tests.

Each ``*.py.expected`` file in this directory is the captured
``to_triton_code()`` output for a specific tcgen05 strategy /
seed-config combination. The corresponding ``_*_kernel.py`` file
hosts the matmul kernel definition at a stable file path so the
embedded ``src[<file>:<line>]`` comments do not drift when the
test file changes.

See ``test_cute_lowerings.py``'s
``test_tcgen05_role_local_monolithic_byte_identical_golden`` for
the read site and update protocol (``EXPECTTEST_ACCEPT=1``).
"""
