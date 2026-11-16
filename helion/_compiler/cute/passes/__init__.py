"""CuTe AST rewrite passes over the rendered device body and their pipeline.

``pipeline`` lists the ordered pass call sequence ``CuteBackend`` runs from
``Backend.finalize_device_body`` / ``finalize_kernel_def``; the pass bodies
themselves still live in their flat ``helion._compiler.cute`` modules.
``pdl`` places the programmatic-dependent-launch wait and release around
the body before and after those passes, and ``launch_attributes`` builds
the module statements that follow the kernel definition.

Role: device-body pass (section 4 of ``../README.md``).
"""

from __future__ import annotations
