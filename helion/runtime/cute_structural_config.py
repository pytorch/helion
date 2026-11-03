"""Import path of :mod:`helion.runtime.cute.structural_config` before the CuTe
structural modules moved under ``helion.runtime.cute``.

Kept for the AOT heuristic model files that
``helion.autotuner.aot_structural_policy.attach_model_policy`` generated before
the move: they are persisted in users' repositories and import
``CuteStructuralConfig`` from here.  In-tree code and newly generated models
import ``helion.runtime.cute.structural_config``.
"""

from __future__ import annotations

from .cute.structural_config import CuteStructuralConfig as CuteStructuralConfig
