"""Backend-portable tests.

Every test in this directory tree is expected to pass on **every** Helion
backend and in eager reference mode (``HELION_INTERPRET=1``).  This is the
conformance surface of the Helion kernel language.

Subdirectories group tests by the language concept under test:

``language/``
    Core kernel language operations — views, reshapes, loops, reductions,
    broadcasting, indexing, control flow, etc.

Adding a test here
------------------
If a test genuinely cannot pass on a backend yet, do not add
``@xfailIfPallas`` or ``@onlyBackends`` to the test.  Instead, record the gap
in ``test/backends/<backend>.py`` using ``xfail_test()``::

    from test.backends import register_backend, xfail_test

    _backend = register_backend("mybackend")
    xfail_test(
        _backend,
        "language/test_views",
        "TestViews::test_foo",
        reason="not yet implemented on mybackend",
    )

The conftest applies the marker automatically at collection time, keeping the
test visible as a known gap rather than hiding it behind a decorator.

The only in-file backend annotations that belong here are:

``@skipUnlessTensorDescriptor(...)``
    A hardware skip for tests that require CUDA sm90+ tensor-descriptor
    hardware.  This is a genuine hardware prerequisite, not a feature gap.

``@skipIfRefEager(...)``
    Skips tests that are meaningless in eager reference mode because they test
    code-generation properties (e.g. checking that a specific intrinsic appears
    in the emitted Triton source).
"""
