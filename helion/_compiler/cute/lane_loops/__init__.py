"""CuTe lane loops: the per-thread element loops of a lane-distributed block.

``markers`` holds the loop forms and the ``_helion_lane_reduce`` marker
protocol, ``owner_guards`` the thread-axis ownership proofs, ``split`` the
lane-split / interchange / sink passes and ``root_passes`` the ordered
root-exit pipeline ``CuteBackend.lower_owned_lane_loop`` runs.  The lane-loop
distribution of a body is ``statements`` (the per-statement analysis),
``placement`` (``distribute_lane_loops`` and the exactness checks) and
``barriers`` (``add_thread_barriers``); ``nest_check`` runs them on a device
loop's nest and on a root body for the per-thread strategies
(``CuteBackend.check_lane_loop_nest`` / ``add_lane_nest_barriers``).  The
modules are deliberately absent from
``cute/_codegen_modules.py``: reloading them would fork the
``_LaneReduceMarker`` identity and the lane stash counter.

Role: general-purpose emitter and device-body pass (section 3.1 of
``../README.md``).
"""

from __future__ import annotations
