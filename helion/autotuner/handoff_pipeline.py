"""Connect ordinary autotuning to the opt-in native-source workflow."""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
from uuid import uuid4

from .base_cache import AutotuneCacheBase
from .base_search import BaseSearch
from .handoff import HandoffPolicy
from .handoff import _active_handoff
from .handoff import find_handoff
from .handoff_bundle import build_handoff
from .handoff_cli import CLISourceAgent

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .base_search import BaseAutotuner

_running: ContextVar[bool] = ContextVar("automatic_handoff_running", default=False)


def autotune_with_handoff(
    autotuner: BaseAutotuner, *, skip_cache: bool = False
) -> Config:
    """Opt-in dispatch: keep the normal config and save the native winner separately."""
    if not isinstance(autotuner, (BaseSearch, AutotuneCacheBase)):
        raise TypeError("Automatic handoff requires a BaseSearch or autotune cache")
    search = (
        autotuner.autotuner if isinstance(autotuner, AutotuneCacheBase) else autotuner
    )
    settings = search.settings
    if (
        not settings.autotune_handoff
        or _running.get()
        or _active_handoff.get() is not None
    ):
        return autotuner.autotune(skip_cache=skip_cache)
    CLISourceAgent.validate_available()

    root = (
        Path(settings.autotune_log).with_suffix(".handoff")
        if settings.autotune_log
        else Path(tempfile.gettempdir()) / "helion-handoff"
    )
    directory = (root / uuid4().hex).resolve()
    policy = HandoffPolicy(
        after_seconds=settings.autotune_budget_seconds,
        automatic=settings.autotune_budget_seconds is None,
    )
    token = _running.set(True)
    try:
        with search.log.autotune_tracing("autotune_handoff"):
            point = find_handoff(autotuner, policy, skip_cache=skip_cache)
            # The search's own log sink has closed. Reattach the same log path
            # for source rounds without manufacturing another config dataset.
            with search.log.autotune_logging():
                search.log(f"HANDOFF: exporting native source to {directory}")
                bundle = build_handoff(autotuner, point, directory)
                search.log.record_handoff_event(
                    "handoff_ready",
                    directory=str(directory),
                    native_baseline=bundle.manifest["baseline_evaluation"],
                )
                result = bundle.run_agent_rounds(
                    budget_seconds=settings.autotune_handoff_budget_seconds,
                    log=search.log,
                )
                search.log(
                    f"Native-source handoff finished ({result.stop_reason}). "
                    f"Standalone kernels: {directory}; records: {result.directory}"
                )
            return point.config
    finally:
        _running.reset(token)
