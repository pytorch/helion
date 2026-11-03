"""The CuTe backend: special-purpose and general-purpose code emitters that compose.

``README.md`` in this directory is the map; in short:

* ``backend.py`` -- ``CuteBackend``, the implementation of the ``Backend``
  hooks core calls (``helion/_compiler/backend.py``); core names this package
  nowhere else (``scripts/check_backend_isolation.py``).
* ``emitters/`` -- ``ROOT_EMITTERS``, the special-purpose root emitters, beside
  the general-purpose helpers (memory access, lane packets, item loops);
  ``kernel_generators.py`` -- the whole-kernel generators and the retry policy.
* the families -- tcgen05 GEMM (``cute_mma.py``, ``tcgen05/``, ``tcgen05_config/``),
  flash attention (``cute_flash*.py``), ``cute_mma_dag*.py``, the recurrences,
  ``selection/``, the fragments: each a planner recording a plan in
  ``cute_state.root_plans`` plus an emitter reading it.
* the passes -- ``ir_passes.py`` (device IR), ``passes/pipeline.py`` (device
  body), ``lane_loops/root_passes.py`` (the root-exit lane pipeline),
  ``lane_loops/nest_check.py`` (the per-thread strategies' nest check and
  barrier pass).
* ``state.py`` / ``device_state.py`` -- the objects behind ``backend_state``.
* ``autotune/`` -- the search surface, the heuristics, the knob tables and
  the LLM autotuner's prompt profile.
* ``device/`` and the ``*_runtime`` / ``*_helpers`` modules -- device helpers
  traced into kernels; generated imports pin their names.
* ``testing.py``, ``target_facts.py`` -- the test seams and the target
  model with its per-CTA SMEM probes.

This file stays import-free: ``backend_registry`` imports ``cute.backend``
while ``compile_environment`` initializes (which is why ``backend.py`` imports
its leaf modules at call time), and ``backend_registry._reload_backend_codegen``
reloads the op-handler modules listed in ``_codegen_modules.py``.
"""

from __future__ import annotations
