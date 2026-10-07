"""Backend-portable tests.

Every test in this package is part of Helion's backend conformance surface and
must run on every backend and in eager reference mode. Portable tests use
``unittest.TestCase`` methods and keep backend gates out of the test file.

Known backend gaps live in ``test/backends/<backend>/registry.py``::

    from test.backends import backend_gap

    GAPS = [
        backend_gap(
            "test_feature",
            "TestFeature::test_foo",
            reason="not yet implemented",
        )
    ]

The registry's parent directory names its registered backend. An empty
``inner_key`` gaps a whole file; an exact ``ClassName::test_method`` gap takes
precedence over a whole-file gap regardless of declaration order. A lazy
``condition=lambda: ...`` disables a gap when it returns false; conditions are
evaluated only for the active backend, at collection time, so they must read
only environment variables or settings. Probing a device (for example CUDA)
would initialize it in every pytest-xdist worker during collection. Use
``skip=True`` only when executing
the test is unsafe, such as a compiler crash, device fault, or timeout;
ordinary missing behavior must remain an expected failure.

Expected-failure gaps are strict: failure reports XFAIL and an unexpected pass
fails the run. They are not applied with ``--runxfail`` or in eager reference
mode because eager execution performs no backend lowering. Registry keys and
the TestCase-only contract are validated against classes in the loaded
portable modules.
"""
