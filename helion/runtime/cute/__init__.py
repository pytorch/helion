"""The CuTe backend's runtime: the launcher, the host wrappers and their caches.

``launcher.default_cute_launcher`` is the entry ``CuteBackend.default_launcher_name``
names (through the ``_default_cute_launcher`` alias ``library_imports`` binds
to ``helion.runtime.default_cute_launcher``); ``host_wrappers/`` holds the host
half of each special-purpose emitter family (one ``HostWrapperEmitter`` per
wrapper-plan kind); the leaf modules ``compile_cache``, ``source_dependencies``,
``launch_descriptor``, ``launch_caches``, ``fast_relaunch``, ``arg_signature``,
``tensor_maps``, ``occupancy`` and ``sass_check`` are what the launcher is built
from, and ``structural_policy`` / ``structural_config`` are the structural
settings a bound kernel records beside its config.  The CuTe backend
README's "The runtime" section is the map.  Role: host wrapper.
"""

from __future__ import annotations

from .launcher import cute_cuda_graph as cute_cuda_graph
from .launcher import default_cute_launcher as default_cute_launcher
