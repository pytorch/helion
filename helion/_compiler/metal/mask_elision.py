"""Which tile bounds masks a Metal kernel drops, and the pass that refutes it.

A tile whose block divides its extent needs no bounds mask, unless the
launch runs more threads on its axis than the tile spans: those surplus
threads would index past it.  The shared rule (``_setup_mask``) judges that
by the dispatch's planned thread axes, and Metal launches each axis at the
largest extent its emitted code gives it, which can differ from the plan: a
reduction placed on the lowest axis (``reduction_axis_first``) moves the
tiles above it, and sibling passes of one loop share their axes.

So a pass drops a mask on the plan's word and records the axis and thread
count it relied on (``MetalBackend.allows_elided_tile_mask``).  The launch
checks each record (``MetalBackend.launcher_keyword_args``); a record the
launch contradicts raises :class:`MaskElisionRefuted`, and :class:`MetalRetry`
generates the kernel again under the earlier rule, which drops masks only
when every thread axis of every strategy has one extent, so no axis can
have surplus threads whichever one the code lands on.

The per-axis rule matters for any launch that isn't square: a GEMV that
lays 32 word columns of its weights beside 8 threads of its K split kept
every mask under the uniform rule, and its masked loads ran it 1.4-1.7x
slower on an M4 Pro.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from ... import exc
from ...exc import BackendUnsupported
from ..emitter_protocols import RetryPolicy

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ...runtime.config import Config


class MaskElisionRefuted(exc.Base):
    """A pass dropped a tile mask for a thread count its launch doesn't
    run on that axis.  An ``exc.Base``, so codegen's visitors pass it to
    ``generate_ast`` unwrapped."""

    def report(self) -> str:
        return str(self)


class MetalRetry(RetryPolicy):
    """Metal's regeneration: a pass whose launch refutes its masks is
    generated again under the uniform rule; ``MatmulFallback`` handles the
    rest."""

    signals = (MaskElisionRefuted, BackendUnsupported)

    def __init__(self) -> None:
        # Imported here: search_surface imports the compiler, which imports
        # the backend that imports this module.
        from .search_surface import MatmulFallback

        self._matmul = MatmulFallback()

    @contextlib.contextmanager
    def scope(self) -> Iterator[None]:
        from .search_surface import _metal_backend

        backend = _metal_backend()
        backend.uniform_masks = False
        try:
            with self._matmul.scope():
                yield
        finally:
            backend.uniform_masks = False

    def repair(
        self, signal: BaseException, config: Config, *, caller_graphs: bool
    ) -> Config | None:
        if isinstance(signal, MaskElisionRefuted):
            from .search_surface import _metal_backend

            backend = _metal_backend()
            if backend.uniform_masks:
                return None
            backend.uniform_masks = True
            return config
        return self._matmul.repair(signal, config, caller_graphs=caller_graphs)
