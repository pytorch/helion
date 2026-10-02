"""Backend-portable tests.

Every test in this tree must pass on **every** Helion backend and in eager
reference mode (``HELION_INTERPRET=1``).  This is the conformance surface of
the Helion kernel language.

Files live directly in this directory; add a subdirectory only once a group of
related files exists (gap subpaths support nesting either way).

Backend gaps are declared in ``test/backends/<backend>.py`` with
``xfail_test()`` rather than with ``@xfailIfPallas`` / ``@onlyBackends``
decorators::

    from test.backends import register_backend
    from test.backends import xfail_test

    _backend = register_backend("mybackend")
    xfail_test(
        _backend,
        "test_views",
        "TestViews::test_foo",
        reason="not yet implemented on mybackend",
    )

The conftest applies the marker at collection time, keeping the gap visible
instead of hiding it behind a decorator.

The only in-file backend annotations allowed here are genuine prerequisites,
not feature gaps:

``@skipUnlessTensorDescriptor(...)``
    Requires CUDA sm90+ tensor-descriptor hardware.

``@skipIfRefEager(...)``
    Test relies on compiled output or hardware state unavailable in eager mode
    (e.g. a lifted variable, a dtype view that eager cannot trace).

Assertions on *emitted code* are not portable: they describe one backend's
lowering.  Put them in a backend-owned file (e.g. ``test/triton_codegen/``, gated with
``@onlyBackends``, ``@skipIfNotTriton``, and ``@skipIfRefEager``), redefining
the kernel inline inside each test method.  A test whose only numeric assertion
is a shape check belongs in that backend file outright.
"""
