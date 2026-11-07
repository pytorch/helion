"""``cute_fragment_scan_exports``: the per-config proof of completed short-scan exports.

The knob's roots and coverage are the ``SCAN_EXPORTS`` row of
``cute.autotune.fragment_knobs``; ``generate_ast`` and the coverage probe
share this validator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import sympy

from .....exc import InvalidConfig
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_warp_scan import active_warp_scan_roots

if TYPE_CHECKING:
    import torch

    from .....runtime.config import Config
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR

KEY = "cute_fragment_scan_exports"


def validate_scan_exports(
    env: CompileEnvironment, device_ir: DeviceIR, config: Config
) -> None:
    from ...fragment_storage import configured_fragment_expr
    from ...scan_analysis import completed_scan_plans

    def shape(sizes: object) -> tuple[int, ...]:
        result = []
        for size in cast("tuple[int | torch.SymInt, ...]", sizes):
            value = env.specialize_expr(
                cast(
                    "sympy.Expr",
                    configured_fragment_expr(
                        env,
                        sympy.sympify(size),
                        lambda bid: env.block_sizes[bid].from_config(config),
                    ),
                )
            )
            if not value.is_number:
                raise InvalidConfig(
                    "completed scan export extent is not statically configured"
                )
            result.append(int(value))
        return tuple(result)

    for root, graphs in fragment_root_regions(device_ir):
        if root in active_warp_scan_roots(env, config) and completed_scan_plans(
            next(g.graph for g in graphs if g.graph_id == root), shape
        ):
            return
    raise InvalidConfig(
        "completed scan exports require a short one-row warp scan and closed terminal integer extrema"
    )
