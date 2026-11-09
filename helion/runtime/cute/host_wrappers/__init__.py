"""Host wrapper emitters of the CuTe backend's special-purpose kernels.

A wrapper plan (one ``dict`` of ``_helion_cute_wrapper_plans``) tells the
launcher how a generated ``@cute.kernel`` wants its ``@cute.jit`` host wrapper
set up: the tiled MMAs and TMA atoms to build over the argument tensors, the
grid to launch, the extra kernel arguments to pass, the host entry that
replaces the launch for whole-root kernels.  Each module here owns the host
half of one emitter family and declares one :class:`HostWrapperEmitter` per
plan kind; :data:`HOST_WRAPPER_EMITTERS` is populated eagerly from these
runtime-side modules (never by a compiler module's import side effects), and
the launcher dispatches ``_append_cute_wrapper_plan`` through it.  Unknown
kinds raise ``BackendUnsupported("cute", "wrapper plan kind: ...")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import block_scaled
from . import flash
from . import gathered
from . import mma_dag
from . import recurrences
from . import tcgen05
from .context import HostWrapperContext as HostWrapperContext
from .context import wrapped_grouped_descriptors as wrapped_grouped_descriptors

if TYPE_CHECKING:
    from .context import HostWrapperEmitter as HostWrapperEmitter

HOST_WRAPPER_EMITTERS: dict[str, HostWrapperEmitter] = {}
for _module in (block_scaled, flash, gathered, mma_dag, recurrences, tcgen05):
    for _emitter in _module.EMITTERS:
        assert _emitter.kind not in HOST_WRAPPER_EMITTERS, _emitter.kind
        HOST_WRAPPER_EMITTERS[_emitter.kind] = _emitter
