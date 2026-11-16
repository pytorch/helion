"""``cute_fragment_warp_producer_regions``: the per-config proof of a warp producer prefix.

The knob's roots and coverage are the ``WARP_PRODUCER_REGIONS`` row of
``cute.autotune.fragment_knobs``; ``generate_ast`` and the coverage probe
share this validator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import sympy

from .....exc import InvalidConfig
from .cute_fragment_common import fragment_root_regions

if TYPE_CHECKING:
    import torch

    from .....runtime.config import Config
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR

KEY = "cute_fragment_warp_producer_regions"


def validate_warp_producer_regions(
    env: CompileEnvironment, device_ir: DeviceIR, config: Config
) -> None:
    from ...fragment_storage import configured_fragment_expr
    from ...warp_producer_regions import producer_prefix

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
                raise InvalidConfig("warp producer extent is not statically configured")
            result.append(int(value))
        return tuple(result)

    for root, graphs in fragment_root_regions(device_ir):
        if (
            producer_prefix(
                next(g.graph for g in graphs if g.graph_id == root), env, graphs, shape
            )
            is not None
        ):
            return
    raise InvalidConfig("warp producers require a one-row scalar-frontier prefix")
