"""``cute_fragment_mma_rows``: the tf32 fragment mma lanes walk whole tile rows.

The tf32 ``mma.sync`` phase of the fragment lowering (``cute_fp32_dot``)
walks 16 x 8 output tiles round-robin over the CTA's warps and evaluates
both operands' words for every tile and K step.  The A words of a tile
depend on its 16-row band and the K step only, so a warp that walks a whole
tile row re-derives the same A words once per column tile: for the chunked
cumsum's ``L @ g`` (a coordinate-only lower-triangular operand evaluated in
place) that is most of the lane loop's instructions.  With the knob on, each
warp owns whole (batch, 16-row) rows of the product, forms the row's A
words for every K step once, keeps them in registers and walks the row's
column tiles against them -- the same mma per output tile, so the results
are bitwise identical.  The register cost is four words per K step; the
autotuner weighs it against the saved instructions.  Only roots with a
tf32-eligible fp32 dot carry the knob, and a config pins it on only together
with ``cute_fp32_dot='tf32_mma'``; a K split (``cute_fragment_split_k``)
keeps its unit walk.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import cast

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fp32_dot import fragment_fp32_dot_roots
from .cute_fragment_common import fragment_coverage_carrier
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_mma_rows"


class CuteFragmentMmaRowsHeuristic(AutotunerHeuristic):
    """Register every complete fragment root with a tf32-eligible dot."""

    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_mma_rows_root_ids = fragment_fp32_dot_roots(
                env, device_ir
            )
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_mma_rows_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_mma_rows_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_mma_rows_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    # The witness rides the tf32 mode and the exact tile-transport knobs the
    # row walk is designed to sit on (packet windows, an unrolled load walk,
    # conflict-free padded rows).  Each of those is another coverage group's
    # non-legacy mode, so the carrier declares them as dependencies (a
    # carrier that merely sets them is refused as "not ordinary legacy modes"
    # and never reaches the population); a root without the tf32 group has
    # no witness, and a transport group that is absent is simply not pinned.
    groups = {group.key: group for group in spec.compiler_coverage_groups}
    tf32_group = groups.get("cute_fp32_dot")
    if tf32_group is None:
        return
    dependencies = [
        CoverageDependency(tf32_group.mechanism, "cute_fp32_dot", "tf32_mma")
    ]
    pins: dict[str, object] = {"cute_fp32_dot": "tf32_mma"}
    for key in (
        "cute_fragment_vector_io",
        "cute_fragment_unrolled_loops",
        "cute_fragment_padded_rows",
    ):
        group = groups.get(key)
        if group is None:
            continue
        dependencies.append(CoverageDependency(group.mechanism, key, True))
        pins[key] = True
    previous_enabled = spec.cute_fragment_mma_rows_search_enabled
    spec.cute_fragment_mma_rows_search_enabled = True
    # The generation must see the knob's field: create it after enabling.
    generation = spec.create_config_generation()
    try:
        _, tf32_carrier = generation.strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | pins)
        )
        if any(tf32_carrier.config.get(key) != value for key, value in pins.items()):
            raise InvalidConfig("row-walk carrier lost a pinned mode")
        # A supplemental witness widens the root's innermost tile as far as it
        # stays legal: the walk's saving grows with the column tiles of a row,
        # and a narrow default tile would show the search a walk with nothing
        # to reuse.
        wide = _widest_inner_tile(spec, tf32_carrier)
        for witness in (tf32_carrier, wide):
            if witness is not None:
                generation.strict_config_pair(
                    Config.from_dict(deepcopy(witness.config) | {KEY: True})
                )
    except InvalidConfig:
        spec.cute_fragment_mma_rows_search_enabled = previous_enabled
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism=KEY.replace("cute_", "cute.", 1),
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(tf32_carrier, True),),
            supplemental_witnesses=(CoverageWitness(wide, True),)
            if wide is not None
            else (),
            dependencies=tuple(dependencies),
            deferred=True,
        )
    )


def _widest_inner_tile(spec: ConfigSpec, carrier: Config) -> Config | None:
    """*carrier* with its last block size doubled while strict normalization
    keeps every other key; None when the tile cannot widen."""
    sizes = carrier.config.get("block_sizes")
    if not isinstance(sizes, list) or not sizes or not spec.block_sizes:
        return None
    block = spec.block_sizes[-1]
    current = sizes[-1]
    if not isinstance(current, int) or current < 1:
        return None
    generation = spec.create_config_generation()
    best: Config | None = None
    size = current * 2
    while size <= block.dim_max_size:
        candidate = deepcopy(carrier.config)
        cast("list[int]", candidate["block_sizes"])[-1] = size
        try:
            _, normalized = generation.strict_config_pair(Config.from_dict(candidate))
        except InvalidConfig:
            break
        if (
            any(
                normalized.config.get(key) != carrier.config.get(key)
                for key in set(normalized.config) | set(carrier.config)
                if key != "block_sizes"
            )
            or cast("list[int]", normalized.config["block_sizes"])[-1] != size
        ):
            break
        best = normalized
        size *= 2
    return best
