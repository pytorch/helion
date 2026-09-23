from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES

if TYPE_CHECKING:
    from collections.abc import Mapping


CONFIG_KEY = "tcgen05_grouped_dense_row_union"
RESIDENT_CTAS_KEY = "tcgen05_grouped_ctas_per_sm"
SCHEDULE_KEY = "tcgen05_grouped_row_union_schedule"
LEGACY_SCHEDULE = "legacy"
TRANSPOSED_SCHEDULE = "nm_m256_n80_k64"
PAIRED_CLC_SCHEDULE = "nm_m256_n192_k64_clc"
STARTUP_PREFILL_KEY = "tcgen05_ab_startup_prefill"


@dataclass(frozen=True)
class PairedRowUnionProtocol:
    """Owned launch and lifetime facts for the paired full-rectangle recipe."""

    coverage_warp: int = 7
    output_issuer_warps: tuple[int, ...] = (0, 2)
    static_registers: bool = True
    rolled_ab_fill: bool = True
    late_c_acquire: bool = True
    scheduler_stages: int = 2


@dataclass(frozen=True)
class RowUnionSchedule:
    """Physical geometry shared by source, producer descriptors and cache keys.

    The public block-size coordinates remain a power-of-two search carrier.
    This profile owns the complete contraction and its output schedule; its
    source tile is therefore a separate, explicit code-generation geometry.
    """

    name: str
    mma_m: int
    mma_n: int
    block_k: int
    cluster_m: int
    cluster_n: int
    ab_stages: int
    a_partition_n: int
    epi_m: int
    epi_n: int
    paired_protocol: PairedRowUnionProtocol | None = None

    @property
    def linear_record_clc(self) -> bool:
        return self.paired_protocol is not None

    @property
    def source_tile(self) -> tuple[int, int, int]:
        return self.mma_n, self.mma_m, self.block_k

    @property
    def shared_upper_bound(self) -> int:
        ab = (self.mma_m + self.mma_n) * self.block_k * 2 // self.cluster_m
        # The paired profile's 2KiB allowance includes AB/ACC/scheduler
        # barriers, two transformed 16B mailboxes, a private 16B CLC reply,
        # its 8B barrier, TMEM publication, at most 63 Int32 coverage/interval
        # words, alignment padding, and the SDK's 1KiB reservation. This is
        # a storage bound, not an occupancy or register-allocation claim.
        return self.ab_stages * ab + 2 * self.epi_m * self.epi_n * 2 + 2048

    @property
    def producer_cluster(self) -> tuple[int, int, int]:
        return self.cluster_m, self.a_partition_n, 1

    def index_domain(self, groups: int, m: int, n: int, k: int) -> bool:
        limit = (1 << 31) - 1
        sm, sn, sk = self.source_tile
        if self.linear_record_clc:
            # Preserve a valid legacy coverage carrier. The extra 32-row
            # condition documents the whole-packet TMA tail proof: no partial
            # store descriptor is admitted, and every empty packet commits.
            return (
                index_domain(groups, m, n, k)
                and groups + 1 <= 32
                and m % self.epi_n == n % sn == k % sk == 0
                and self.ab_stages <= k // sk <= TCGEN05_TWO_CTA_MAX_K_TILES
                and max(m * k, m * n, k * n, m + sm - 1, n + sn - 1) <= limit
                and groups * ((m + sm - 1) // sm) * (n // sn) <= limit
            )
        return (
            all(type(v) is int and 0 < v <= limit for v in (groups, m, n, k))
            # Keep a valid legacy counterpart for dependent search coverage.
            and index_domain(groups, m, n, k)
            and m % (sm * self.cluster_n) == n % sn == k % sk == 0
            and k // sk <= TCGEN05_TWO_CTA_MAX_K_TILES
            and max(m * k, m * n, k * n, m + sm - 1, n + sn - 1) <= limit
            and groups * (m // sm) * (n // sn) <= limit
        )

    def record_coordinates(self, record: str, m: int) -> tuple[str, str]:
        """Source row/N tile coordinates for both initial and CLC records."""
        assert self.linear_record_clc
        rows = (m + self.source_tile[0] - 1) // self.source_tile[0]
        return (
            f"({record}) % cutlass.Int32({rows})",
            f"({record}) // cutlass.Int32({rows})",
        )

    def codegen_values(self) -> dict[str, object]:
        values: dict[str, object] = {
            "tcgen05_cluster_m": self.cluster_m,
            "tcgen05_cluster_n": self.cluster_n,
            "tcgen05_ab_stages": self.ab_stages,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_layout_overrides_epi_tile_m": self.epi_m,
            "tcgen05_layout_overrides_epi_tile_n": self.epi_n,
            "tcgen05_layout_overrides_d_store_box_n": self.epi_n,
        }
        if self.paired_protocol is not None:
            values.update(
                tcgen05_strategy="role_local_with_scheduler",
                tcgen05_persistence_model="clc_persistent",
                tcgen05_warp_spec_scheduler_warps=1,
                tcgen05_sched_stage_count=self.paired_protocol.scheduler_stages,
            )
        return values


TRANSPOSED = RowUnionSchedule(TRANSPOSED_SCHEDULE, 256, 80, 64, 2, 2, 10, 1, 128, 16)

PAIRED_CLC = RowUnionSchedule(
    PAIRED_CLC_SCHEDULE, 256, 192, 64, 2, 1, 7, 1, 128, 32, PairedRowUnionProtocol()
)


def schedule_by_name(name: object) -> RowUnionSchedule | None:
    for profile in (TRANSPOSED, PAIRED_CLC):
        if name == profile.name:
            return profile
    return None


def physical_schedule(config: Mapping[str, object]) -> RowUnionSchedule | None:
    return schedule_by_name(config.get(SCHEDULE_KEY, LEGACY_SCHEDULE))


# Two BF16 AB rings plus a conservative allowance for their four barriers,
# four accumulator barriers, TMEM publication and the SDK's 1KiB reservation.
ROW_UNION_SHARED_UPPER_BOUND = 2 * (128 + 64) * 128 * 2 + 2 * 1024
ROW_UNION_TMEM_COLUMNS = 128


def resident_ctas_supported(value: object, shared_capacity: int) -> bool:
    """Bound residency for the existing ONE/AB2/ACC2/SIMT-store schedule.

    This bounds shared and TMEM storage, not compiler register allocation or
    measured occupancy. An actual native object must establish the latter.
    """
    return (
        type(value) is int
        and value in (1, 2)
        and type(shared_capacity) is int
        and shared_capacity >= value * ROW_UNION_SHARED_UPPER_BOUND
        and value * ROW_UNION_TMEM_COLUMNS <= 512
    )


def index_domain(groups: int, m: int, n: int, k: int) -> bool:
    """Bound every dense tile/address expression; offsets retain Int32 wrap."""
    limit = (1 << 31) - 1
    return (
        all(type(value) is int and 0 < value <= limit for value in (groups, m, n, k))
        and m % 128 == n % 64 == k % 128 == 0
        and m * k <= limit
        and m * n <= limit
        and k * n <= limit
        and m + 127 <= limit
        and n + 63 <= limit
        # PID decomposition still names the original group axis, fixed to zero.
        and groups * (m // 128) * (n // 64) <= limit
    )


def schedule_supported(
    config: Mapping[str, object], blocks: tuple[int, int, int]
) -> bool:
    """The ordinary dense pipeline used by the initial row-union carrier.

    These are tile/protocol restrictions, independent of any problem shape or
    kernel name. Ineligible choices retain the existing grouped lowering.
    """
    fixed: dict[str, object] = {
        "pid_type": "persistent_interleaved",
        "num_sm_multiplier": 1,
        "tcgen05_cta_group": "auto",
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_l2_swizzle_size": 1,
        "tcgen05_strategy": "role_local_monolithic",
        "tcgen05_persistence_model": "static_persistent",
        "tcgen05_warp_spec_ab_load_warps": 1,
        "tcgen05_warp_spec_mma_warps": 1,
        "tcgen05_warp_spec_scheduler_warps": 0,
        "tcgen05_warp_spec_c_input_warps": 0,
        "tcgen05_warp_spec_epi_load_warps": 0,
        "tcgen05_warp_spec_store_warps": 0,
        "tcgen05_warp_spec_register_decrease": 120,
        "tcgen05_warp_spec_register_increase": 256,
        "tcgen05_layout_strategy": "default",
        "tcgen05_grouped_full_coverage": "off",
        "tcgen05_flat_role_coordinates": False,
        "tcgen05_diagnostic_invalid_output": False,
    }
    profile = physical_schedule(config)
    expected_blocks = (128, 64, 128)
    allowed_layout_keys: tuple[str, ...] = ()
    if profile is not None:
        expected_blocks = (128, 64, 128)
        if blocks == profile.source_tile:
            expected_blocks = profile.source_tile
            fixed.update(profile.codegen_values())
            allowed_layout_keys = (
                "tcgen05_layout_overrides_epi_tile_m",
                "tcgen05_layout_overrides_epi_tile_n",
                "tcgen05_layout_overrides_d_store_box_n",
            )
    if profile is not None and profile.paired_protocol is not None:
        fixed.update(
            tcgen05_c_acquire_placement="pre_loop",
            tcgen05_acc_wait_placement="subtile_loop",
        )
    l2_groupings = config.get("l2_groupings", [1])
    return (
        blocks == expected_blocks
        and (
            profile is None
            or (
                all(type(value) is int for value in blocks)
                and all(
                    type(config.get(key, value)) is type(value)
                    for key, value in fixed.items()
                )
            )
        )
        and config.get(SCHEDULE_KEY, LEGACY_SCHEDULE)
        in (LEGACY_SCHEDULE, TRANSPOSED_SCHEDULE, PAIRED_CLC_SCHEDULE)
        and type(config.get(STARTUP_PREFILL_KEY, False)) is bool
        and (
            not config.get(STARTUP_PREFILL_KEY, False)
            or (profile is not None and profile.linear_record_clc)
        )
        and (profile is None or config.get(RESIDENT_CTAS_KEY, 1) == 1)
        and type(config.get(RESIDENT_CTAS_KEY, 1)) is int
        and config.get(RESIDENT_CTAS_KEY, 1) in (1, 2)
        and isinstance(l2_groupings, (list, tuple))
        and all(type(value) is int and value == 1 for value in l2_groupings)
        and all(config.get(key, value) == value for key, value in fixed.items())
        and config.get("tcgen05_grouped_mode") is None
        and all(
            config.get(key) is None
            for key in (
                "tcgen05_grouped_worklist_source_m_tile",
                "tcgen05_grouped_external_direct_pointers",
                "tcgen05_grouped_external_direct_strides",
                "tcgen05_grouped_static_problem_signature",
                "tcgen05_layout_overrides_epi_tile_m",
                "tcgen05_layout_overrides_epi_tile_n",
                "tcgen05_layout_overrides_d_store_box_n",
                "tcgen05_layout_overrides_smem_swizzle_a",
                "tcgen05_layout_overrides_smem_swizzle_b",
            )
            if key not in allowed_layout_keys
        )
        and config.get("tcgen05_grouped_runtime_direct", False) is False
        and config.get("tcgen05_grouped_static_reserved_sms", 0) == 0
        and all(
            config.get(key, "normal") == "normal"
            for key in (
                "tcgen05_ab_consumer_phase_mode",
                "tcgen05_ab_consumer_wait_mode",
                "tcgen05_ab_initial_producer_acquire_mode",
                "tcgen05_ab_producer_acquire_mode",
                "tcgen05_ab_producer_advance_mode",
                "tcgen05_acc_producer_advance_mode",
                "tcgen05_acc_producer_mode",
                "tcgen05_c_store_mode",
                "tcgen05_epilogue_layout",
                "tcgen05_sched_consumer_wait_mode",
            )
        )
    )


@dataclass(frozen=True)
class GroupedRowUnionPlan:
    """Guarded allocation/axis facts and invocation-local interval storage."""

    groups: int
    m: int
    n: int
    k: int
    group_block_id: int
    m_block_id: int
    n_block_id: int
    offsets: str
    prefix: str
    resident_ctas: int = 1
    schedule: RowUnionSchedule | None = None

    @property
    def paired_protocol(self) -> PairedRowUnionProtocol | None:
        return self.schedule.paired_protocol if self.schedule is not None else None

    @property
    def linear_record_clc(self) -> bool:
        return self.paired_protocol is not None

    @property
    def all_rows(self) -> str:
        return f"{self.prefix}_all_rows"

    def full_coverage(self) -> str:
        """A sufficient normalized coverage test, never a partition assumption."""
        p = self.prefix
        return f"""{self.all_rows} = {self.starts}[0] == cutlass.Int32(0) and {self.ends}[{self.groups - 1}] == cutlass.Int32({self.m})
for {p}_coverage_group in cutlass.range({self.groups - 1}, unroll_full=True):
    {self.all_rows} = {self.all_rows} and {self.ends}[{p}_coverage_group] == {self.starts}[{p}_coverage_group + 1]
"""

    @property
    def starts(self) -> str:
        return f"{self.prefix}_starts"

    @property
    def ends(self) -> str:
        return f"{self.prefix}_ends"

    @property
    def coverage_flag(self) -> str:
        return f"{self.prefix}_coverage_flag"

    def cooperative_setup(self, *, warp: str, lane: str) -> str:
        """Immutable CTA-local tables, published by the existing TMEM join.

        The linear-record profile reserves warp 7 and proves G+1<=32. All
        lanes execute both shuffles and the vote; only bounded unique lanes
        publish the flag and the normalized fallback intervals.
        """
        assert self.paired_protocol is not None and 1 <= self.groups <= 31
        p = self.prefix
        return f"""{p}_coverage_ptr = cute.arch.alloc_smem(cutlass.Int32, 1)
{self.coverage_flag} = cute.make_tensor({p}_coverage_ptr, cute.make_layout((1,)))
{p}_starts_ptr = cute.arch.alloc_smem(cutlass.Int32, {self.groups})
{p}_ends_ptr = cute.arch.alloc_smem(cutlass.Int32, {self.groups})
{self.starts} = cute.make_tensor({p}_starts_ptr, cute.make_layout(({self.groups},)))
{self.ends} = cute.make_tensor({p}_ends_ptr, cute.make_layout(({self.groups},)))
if {warp} == cutlass.Int32({self.paired_protocol.coverage_warp}):
    {p}_coop_raw_start = cutlass.Int32(0)
    if {lane} < cutlass.Int32({self.groups + 1}):
        {p}_coop_raw_start = cutlass.Int32(({self.offsets}.iterator + cutlass.Int64({lane}) * cutlass.Int64({self.offsets}.layout.stride[0])).load())
    {p}_coop_next_lane = min({lane} + cutlass.Int32(1), cutlass.Int32({self.groups}))
    {p}_coop_raw_end = cute.arch.shuffle_sync({p}_coop_raw_start, {p}_coop_next_lane)
    {p}_coop_extent = max({p}_coop_raw_end - {p}_coop_raw_start, cutlass.Int32(0))
    {p}_coop_start = min(max({p}_coop_raw_start, cutlass.Int32(0)), cutlass.Int32({self.m}))
    {p}_coop_length = min(max({p}_coop_extent + min({p}_coop_raw_start, cutlass.Int32(0)), cutlass.Int32(0)), cutlass.Int32({self.m}) - {p}_coop_start)
    {p}_coop_end = {p}_coop_start + {p}_coop_length
    {p}_coop_next_start = cute.arch.shuffle_sync({p}_coop_start, {p}_coop_next_lane)
    {p}_coop_valid = {lane} >= cutlass.Int32({self.groups}) or (({lane} != cutlass.Int32(0) or {p}_coop_start == cutlass.Int32(0)) and ({lane} != cutlass.Int32({self.groups - 1}) or {p}_coop_end == cutlass.Int32({self.m})) and ({lane} >= cutlass.Int32({self.groups - 1}) or {p}_coop_end == {p}_coop_next_start))
    {p}_coverage_vote = cute.arch.vote_all_sync({p}_coop_valid)
    if {lane} == cutlass.Int32(0):
        {self.coverage_flag}[0] = cutlass.Int32({p}_coverage_vote)
    if not {p}_coverage_vote and {lane} < cutlass.Int32({self.groups}):
        {self.starts}[{lane}] = {p}_coop_start
        {self.ends}[{lane}] = {p}_coop_end
"""

    def interval_setup(self) -> str:
        """Match the packed-split proof's typed subtraction/addition order."""
        p = self.prefix
        return f"""{self.starts} = cute.make_rmem_tensor(cute.make_layout(({self.groups},)), cutlass.Int32)
{self.ends} = cute.make_rmem_tensor(cute.make_layout(({self.groups},)), cutlass.Int32)
for {p}_group in cutlass.range({self.groups}, unroll_full=True):
    {p}_raw_start = cutlass.Int32(({self.offsets}.iterator + cutlass.Int64({p}_group) * cutlass.Int64({self.offsets}.layout.stride[0])).load())
    {p}_raw_end = cutlass.Int32(({self.offsets}.iterator + cutlass.Int64({p}_group + 1) * cutlass.Int64({self.offsets}.layout.stride[0])).load())
    {p}_extent = max({p}_raw_end - {p}_raw_start, cutlass.Int32(0))
    {p}_start = min(max({p}_raw_start, cutlass.Int32(0)), cutlass.Int32({self.m}))
    {p}_length = min(max({p}_extent + min({p}_raw_start, cutlass.Int32(0)), cutlass.Int32(0)), cutlass.Int32({self.m}) - {p}_start)
    {self.starts}[{p}_group] = {p}_start
    {self.ends}[{p}_group] = {p}_start + {p}_length
"""

    def masked_copy(
        self, *, source: str, destination: str, coordinates: str, bits: str, atom: str
    ) -> str:
        """Vectorize only a complete in-bounds row packet; otherwise scalarize.

        The zero-stride predicate alias has the exact vector-copy shape. It
        does not allocate duplicate predicates or relax CuTe copy shape checks.
        """
        p = self.prefix
        source_code = f"""{p}_vector = cutlass.const_expr({bits} // cutlass.BFloat16.width)
{p}_store_src = cute.logical_divide({source}, cute.make_layout({p}_vector))
{p}_store_dst = cute.logical_divide({destination}, cute.make_layout({p}_vector))
{p}_store_coord = cute.logical_divide({coordinates}, cute.make_layout({p}_vector))
{p}_store_pred = cute.make_rmem_tensor((1, {p}_store_src.shape[1]), cutlass.Boolean)
{p}_store_pred_copy = cute.make_tensor({p}_store_pred.iterator, cute.make_layout(({p}_vector, {p}_store_src.shape[1]), stride=(0, {p}_store_pred.layout.stride[1])))
for {p}_copy_i in cutlass.range(cute.size({p}_store_src.shape[1]), unroll_full=True):
    {p}_first_coord = {p}_store_coord[0, {p}_copy_i]
    {p}_same_row = cutlass.Boolean(True)
    {p}_all_in_bounds = cutlass.Boolean(True)
    for {p}_value_i in cutlass.range({p}_vector, unroll_full=True):
        {p}_coord = {p}_store_coord[{p}_value_i, {p}_copy_i]
        {p}_same_row = {p}_same_row and ({p}_coord[0] == {p}_first_coord[0])
        {p}_all_in_bounds = {p}_all_in_bounds and cute.elem_less({p}_coord, ({self.m}, {self.n})) and ({p}_coord[0] >= 0) and ({p}_coord[1] >= 0)
    {p}_first_selected = cutlass.Boolean(False)
    for {p}_mask_group in cutlass.range({self.groups}, unroll_full=True):
        {p}_first_selected = {p}_first_selected or ({p}_first_coord[0] >= {self.starts}[{p}_mask_group] and {p}_first_coord[0] < {self.ends}[{p}_mask_group])
    {p}_can_vector = {p}_same_row and {p}_all_in_bounds
    {p}_store_pred[0, {p}_copy_i] = {p}_can_vector and {p}_first_selected
    if not {p}_can_vector:
        for {p}_value_i in cutlass.range({p}_vector, unroll_full=True):
            {p}_coord = {p}_store_coord[{p}_value_i, {p}_copy_i]
            {p}_selected = cutlass.Boolean(False)
            for {p}_mask_group in cutlass.range({self.groups}, unroll_full=True):
                {p}_selected = {p}_selected or ({p}_coord[0] >= {self.starts}[{p}_mask_group] and {p}_coord[0] < {self.ends}[{p}_mask_group])
            if {p}_selected and cute.elem_less({p}_coord, ({self.m}, {self.n})) and ({p}_coord[0] >= 0) and ({p}_coord[1] >= 0):
                {p}_store_dst[{p}_value_i, {p}_copy_i] = {p}_store_src[{p}_value_i, {p}_copy_i]
cute.copy({atom}, {p}_store_src, {p}_store_dst, pred={p}_store_pred_copy)
"""
        if self.schedule is not None:
            for name in ("first_coord", "coord"):
                assignment = f"{p}_{name} = {p}_store_coord["
                lines = []
                for line in source_code.splitlines():
                    lines.append(line)
                    if assignment in line:
                        indent = line[: len(line) - len(line.lstrip())]
                        lines.append(
                            f"{indent}{p}_{name} = ({p}_{name}[1], {p}_{name}[0])"
                        )
                source_code = "\n".join(lines) + "\n"
        if self.linear_record_clc:
            # The profile fixes the BF16 scalar copy and the 128x32 TMEM
            # packet. Its coordinate projection is (2,2,4,2) with row strides
            # (1,0,2,0): eight rows, each repeated across four columns.
            # Keep all original scalar bounds/address/value expressions.
            marker = f"for {p}_copy_i in cutlass.range("
            at = source_code.index(marker)
            source_code = (
                source_code[:at]
                + f"""assert cutlass.const_expr({p}_vector == 1)
assert cutlass.const_expr({p}_store_coord.shape[1] == (2, 2, 4, 2))
{p}_membership = cute.make_rmem_tensor((2, 4), cutlass.Boolean)
{p}_membership_copy = cute.make_tensor({p}_membership.iterator, cute.make_layout({p}_store_coord.shape[1], stride=(1, 0, 2, 0)))
for {p}_row_i in cutlass.range(8, unroll_full=True):
    {p}_representative = ({p}_row_i % 2) + 4 * ({p}_row_i // 2)
    {p}_representative_coord = {p}_store_coord[0, {p}_representative][1]
    {p}_row_selected = cutlass.Boolean(False)
    for {p}_mask_group in cutlass.range({self.groups}, unroll=1):
        {p}_row_selected = {p}_row_selected or ({p}_representative_coord >= {self.starts}[{p}_mask_group] and {p}_representative_coord < {self.ends}[{p}_mask_group])
    {p}_membership[{p}_row_i] = {p}_row_selected
"""
                + source_code[at:]
            )
            replacements = (
                (
                    (
                        f"    {p}_first_selected = cutlass.Boolean(False)\n"
                        f"    for {p}_mask_group in cutlass.range({self.groups}, unroll_full=True):\n"
                        f"        {p}_first_selected = {p}_first_selected or ({p}_first_coord[0] >= {self.starts}[{p}_mask_group] and {p}_first_coord[0] < {self.ends}[{p}_mask_group])"
                    ),
                    f"    {p}_first_selected = cutlass.Boolean({p}_membership_copy[{p}_copy_i])",
                ),
                (
                    (
                        f"            {p}_selected = cutlass.Boolean(False)\n"
                        f"            for {p}_mask_group in cutlass.range({self.groups}, unroll_full=True):\n"
                        f"                {p}_selected = {p}_selected or ({p}_coord[0] >= {self.starts}[{p}_mask_group] and {p}_coord[0] < {self.ends}[{p}_mask_group])"
                    ),
                    f"            {p}_selected = cutlass.Boolean({p}_membership_copy[{p}_copy_i])",
                ),
            )
            for original, replacement in replacements:
                assert source_code.count(original) == 1
                source_code = source_code.replace(original, replacement)
        return source_code
