"""Codegen pins for ``cute_pdl`` on the tcgen05 GEMM kernels.

CPU-only (fake CUDA target: the ``test_cute_tcgen05_fixed_chain`` binders and
the module's ``cpu_render`` marker): the knob launches the kernel as a
programmatic dependent of the previous kernel in the stream, waits on that
kernel at the top of the body (every warp, ahead of every global read of the
prologue, the root body and the pipelines, on every kernel shape), releases
the dependents right after each of those waits (``cute_pdl_release``
``entry``, the ``auto`` resolution of a one-wave launch) and again once the
MMA warp has issued its last MMA (the family's own release, the only one
under ``exit`` / ``none``).
"""

from __future__ import annotations

import ast
import re
from unittest.mock import patch

import pytest
import torch

from test.cute_population_contracts import tcgen05_mma_support
from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _kernel_body
from test.test_cute_tcgen05_fixed_chain import _source

import helion
from helion._compiler.cute.tcgen05_config import seeds
from helion._compiler.cute.tcgen05_config import surface
import helion.language as hl

# The B200 model with tcgen05 MMA support, CUDA hidden, the inputs traced
# as ``cuda:0`` tensors of a 148-SM device, and the reduction SMEM budget.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(
        mma_support=tcgen05_mma_support(),
        forbid_native=False,
        fake_device_sm_count=148,
        reduction_smem_budget_bytes=232448,
    ),
]

WAIT = "cute.arch.griddepcontrol_wait()"
RELEASE = "cute.arch.griddepcontrol_launch_dependents()"
# Every form a global read takes in these kernels: the wait must precede
# the first of them on every warp, so it is the body's first statement.
GLOBAL_READS = (".load()", "cute.copy(", "prefetch_descriptor(", "ldg_", "tma_atom")


def _body_statements(body: str) -> list[str]:
    lines = [line for line in body.splitlines()[1:] if line.strip()]
    signature_end = next(
        i for i, line in enumerate(lines) if line.rstrip().endswith(":")
    )
    return lines[signature_end + 1 :]


def _assert_no_entry_wait(body: str) -> None:
    assert _body_statements(body)[0].strip() != WAIT


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _assert_release_follows_every_wait(body: str) -> None:
    """``entry`` on the tcgen05 kernels: a release right after each wait of
    the body, in the wait's own block."""
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if WAIT in line:
            assert RELEASE in lines[index + 1], lines[index : index + 2]
            assert _indent(lines[index + 1]) == _indent(line)


def _assert_entry_wait(body: str) -> None:
    lines = [line for line in body.splitlines()[1:] if line.strip()]
    # (the signature may wrap over several lines)
    signature_end = next(
        i for i, line in enumerate(lines) if line.rstrip().endswith(":")
    )
    statements = "\n".join(lines[signature_end + 1 :])
    assert statements.splitlines()[0].strip() == WAIT, statements.splitlines()[:3]
    first_wait = statements.index(WAIT)
    for marker in GLOBAL_READS:
        position = statements.find(marker)
        assert position == -1 or position > first_wait, marker


def _assert_tma_role_waits_before_its_first_stage(body: str) -> None:
    """The TMA-load role (warp 5) waits inside its persistent loop, after the
    tile's address math (the TMA partitions) and right ahead of its first
    stage, so the loads issue as soon as the dependency resolves; the
    epilogue role (warps 0-3) waits at the top of its block."""
    lines = body.splitlines()
    waits = [i for i, line in enumerate(lines) if WAIT in line]
    assert len(waits) == 2, waits
    tma_role = next(
        i
        for i, line in enumerate(lines)
        if line.strip()
        == "if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5):"
    )
    # (a persistent loop when the grid is smaller than the work; the one-shot
    # scheduler lays the single tile out in the role prelude)
    partition = max(
        i for i in range(tma_role, waits[0]) if "tma_partition(" in lines[i]
    )
    assert tma_role < partition < waits[0]
    first_stage = next(
        i for i in range(waits[0], len(lines)) if "cute.copy(tma_atom_a" in lines[i]
    )
    # The stage-0 gate and the stage-0 acquire (shared-memory only) precede
    # the wait; the stage's loads follow it.
    gate = max(
        i
        for i in range(tma_role, waits[0])
        if "tcgen05_tma_initial_full_tile =" in lines[i]
    )
    acquire = max(i for i in range(gate, waits[0]) if "producer_acquire(" in lines[i])
    assert partition < gate < acquire < waits[0] < first_stage
    assert not any("cute.copy(" in lines[i] for i in range(tma_role, waits[0]))
    assert (
        lines[waits[1] - 1].strip()
        == "if cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4):"
    )


def _flat_config(**overrides: object) -> helion.Config:
    values: dict[str, object] = {
        "block_sizes": [64, 16, 128],
        "pid_type": "flat",
        "indexing": ["pointer"] * 3,
        "tcgen05_ab_stages": 2,
    }
    values.update(overrides)
    return helion.Config(**values)


def _persistent_config(**overrides: object) -> helion.Config:
    values: dict[str, object] = {
        "block_sizes": [128, 128, 64],
        "pid_type": "persistent_blocked",
        "indexing": ["tensor_descriptor"] * 3,
        "tcgen05_ab_stages": 4,
    }
    values.update(overrides)
    return helion.Config(**values)


def test_knob_defaults_on_and_stays_on_for_sm90_plus_matmul_kernels() -> None:
    # The autotuner's per-call-flushed regime cannot rank the knob (a memset
    # predecessor gives every kernel a full dependency), and a tcgen05 kernel
    # only gains from it in a chain: the fragment defaults to True, the
    # search keeps it there, and both values stay valid for pinned configs.
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    with bound.env:
        spec = bound.config_spec
        fields = spec.flat_config(lambda x: x.default()).config
        assert fields["cute_pdl"] is True
        fragment = surface.flat_fields(spec._cute_tcgen05_config)["cute_pdl"]
        assert spec.default_config().config["cute_pdl"] is True
        pinned_off = spec.normalized_config(_flat_config(cute_pdl=False))
    assert tuple(fragment.choices) == (True, False)  # pyrefly: ignore[missing-attribute]
    assert fragment.search_choices == (True,)  # pyrefly: ignore[missing-attribute]
    assert pinned_off.config["cute_pdl"] is False


def test_pre_hopper_targets_keep_the_knob_off() -> None:
    # ``griddepcontrol`` needs sm_90: below it the default stays False and
    # the search offers only the off value.
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    with bound.env, patch.object(bound.config_spec, "target_device_capability", (8, 0)):
        spec = bound.config_spec
        state = spec._cute_tcgen05_config
        assert not surface._pdl_searched(state)
        fragment = surface.flat_fields(state)["cute_pdl"]
        assert tuple(fragment.choices) == (False, True)  # pyrefly: ignore[missing-attribute]
        assert fragment.search_choices == (False,)  # pyrefly: ignore[missing-attribute]
        assert spec.default_config().config["cute_pdl"] is False


def test_seeds_carry_the_knob_without_twins_and_normalization_carries_it() -> None:
    # The knob is a searched field, not a seed family: no seed list grows a
    # twin.  Where it is searched every seed carries it (a seed without the
    # dependent launch is screened out of the final selection by that
    # handicap alone), and a normalized config carries ``cute_pdl`` like
    # every other optional knob (so a flat transfer round-trips exactly).
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    with bound.env:
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        seed_configs = seeds.autotune_seed_configs(tcgen05)
        assert seed_configs and all(
            s.config.get("cute_pdl") is True for s in seed_configs
        )
        schedules = [
            {k: v for k, v in s.config.items() if k != "cute_pdl"} for s in seed_configs
        ]
        assert len(schedules) == len(seed_configs)
        assert all(schedules.count(schedule) == 1 for schedule in schedules), (
            "a seed list must not grow a cute_pdl twin"
        )
        with patch.object(surface, "_pdl_searched", return_value=False):
            unsearched = seeds.autotune_seed_configs(tcgen05)
        assert [
            {k: v for k, v in s.config.items() if k != "cute_pdl"} for s in unsearched
        ] == schedules
        assert all("cute_pdl" not in s.config for s in unsearched)
        normalized = spec.normalized_config(
            _persistent_config(loop_orders=[[0, 1]], l2_groupings=[1])
        )
        assert normalized.config["cute_pdl"] is True
        generation = spec.create_config_generation()
        roundtrip = generation.unflatten(generation.flatten(normalized))
    # (``cute_min_blocks_per_mp`` is the launch-bounds heuristic's field; this
    # binding does not register it, so the flat layout alone materializes it.)
    roundtrip.config.pop("cute_min_blocks_per_mp", None)
    assert roundtrip == normalized


def test_flat_kernel_waits_before_the_first_load_and_releases_after_the_mma() -> None:
    bound = _bind_gemm((256, 256, 256), torch.float16)
    plain = _source(bound, _flat_config(cute_pdl=False))
    assert "griddepcontrol" not in plain
    assert "'use_pdl': True" not in plain
    source = _source(bound, _flat_config(cute_pdl=True))
    body = _kernel_body(source)
    assert "'use_pdl': True" in source
    assert body.count(WAIT) == 1 and body.count(RELEASE) == 2
    # Every warp passes the wait after the prologue (TMEM allocation,
    # barrier init, descriptor prefetch) and before the first TMA load; the
    # body has no global read ahead of it, so no entry wait is added.
    assert body.index("prefetch_descriptor(") < body.index(WAIT)
    assert body.index(WAIT) < body.index("cute.copy(tma_atom_a")
    _assert_no_entry_wait(body)
    lines = body.splitlines()
    wait_line = next(i for i, line in enumerate(lines) if WAIT in line)
    releases = [i for i, line in enumerate(lines) if RELEASE in line]
    # The one-wave grid (16 CTAs) resolves ``auto`` to ``entry``: the
    # dependents are released right after the wait, in the same block, so
    # the next kernel's launch overlaps this whole kernel.
    assert releases[0] == wait_line + 1
    assert _indent(lines[releases[0]]) == _indent(lines[wait_line])
    # The MMA warp releases again right after committing its accumulator,
    # ahead of the epilogue's CTA-wide syncs (a no-op after the first).
    commit = body.index("producer_commit(tcgen05_acc_producer_state)")
    assert commit < body.index(RELEASE, body.index(RELEASE) + 1)
    assert body.index(RELEASE, body.index(RELEASE) + 1) < body.index(
        "tcgen05_kernel_desc"
    )
    assert lines[releases[1] - 1].strip() == "if tcgen05_exec_active:"
    # The knob-off body is otherwise unchanged.
    stripped = [
        line
        for index, line in enumerate(lines)
        if "griddepcontrol" not in line and index != releases[1] - 1
    ]
    assert "\n".join(stripped) == _kernel_body(plain)


def test_persistent_kernel_waits_in_the_tma_and_epilogue_roles() -> None:
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    source = _source(bound, _persistent_config(cute_pdl=True))
    body = _kernel_body(source)
    assert "'use_pdl': True" in source
    assert body.count(WAIT) == 2 and body.count(RELEASE) == 3
    # The TMA-load warp (warp 5) waits inside its persistent loop, after the
    # tile's address math and ahead of its first stage; the epilogue warps
    # (0-3) wait at the top of their role block; the MMA warp never touches
    # global memory, and the prologue reads nothing global, so no entry wait
    # is added.
    _assert_tma_role_waits_before_its_first_stage(body)
    _assert_no_entry_wait(body)
    # The persistent launch is one wave: each role releases the dependents
    # right after its wait, and the MMA role once more after its tile loop,
    # in the teardown.
    _assert_release_follows_every_wait(body)
    assert body.rindex(RELEASE) > body.index("tcgen05_role_local_2_tile_sched")
    plain = _kernel_body(_source(bound, _persistent_config(cute_pdl=False)))
    assert "griddepcontrol" not in plain


def test_persistent_kernel_with_a_scheduler_warp_waits_at_entry() -> None:
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    config = _persistent_config(
        cute_pdl=True,
        tcgen05_strategy="role_local_with_scheduler",
        tcgen05_warp_spec_scheduler_warps=1,
    )
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    if normalized.config.get("tcgen05_strategy") != "role_local_with_scheduler":
        pytest.skip("scheduler-warp strategy not admitted on this binding")
    source = _source(bound, normalized)
    assert "'use_pdl': True" in source
    body = _kernel_body(source)
    # this builder has no role-local waits of its own: the TMA-load role's
    # first load would run unwaited, so the entry wait is added
    assert body.count(WAIT) == 1
    _assert_entry_wait(body)


def test_the_entry_wait_path_takes_the_release_placement() -> None:
    # A tcgen05 kernel that gets the generic entry wait takes the release
    # placement (``cute_pdl_release``) like every other kernel: ``auto``
    # (this one-wave persistent grid), ``entry`` and ``last_wave`` (plain
    # entry on one wave) release right after that wait, ahead of the
    # family's release after the MMA warp's last MMA; ``exit`` and ``none``
    # leave the family's release as the only trigger.
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    sources: dict[str, str] = {}
    for placement in (None, "auto", "entry", "exit", "none", "last_wave"):
        config = _persistent_config(
            cute_pdl=True,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            **({} if placement is None else {"cute_pdl_release": placement}),
        )
        with bound.env:
            normalized = bound.config_spec.normalized_config(config)
        if normalized.config.get("tcgen05_strategy") != "role_local_with_scheduler":
            pytest.skip("scheduler-warp strategy not admitted on this binding")
        source = _source(bound, normalized)
        body = _kernel_body(source)
        assert body.count(WAIT) == 1
        _assert_entry_wait(body)
        statements = _body_statements(body)
        wait_index = next(i for i, line in enumerate(statements) if WAIT in line)
        releases = [i for i, line in enumerate(statements) if RELEASE in line]
        # The family's release: under the MMA warp's role predicate, never a
        # top-level last statement (the generic ``exit``).
        family = releases[-1]
        assert _indent(statements[family]) > _indent(statements[wait_index])
        assert statements[family - 1].strip().startswith("if ")
        assert family != len(statements) - 1
        if placement in ("exit", "none"):
            assert releases == [family]
        else:
            assert releases == [wait_index + 1, family]
        sources[placement or "default"] = source
    assert sources["default"] == sources["auto"] == sources["entry"]
    assert sources["auto"] == sources["last_wave"]
    assert sources["exit"] == sources["none"] != sources["auto"]


def test_two_cta_kernels_add_the_launch_attribute_and_the_epilogue_wait() -> None:
    # The CtaGroup.TWO path already waits in its TMA role and releases after
    # the MMAs (griddepcontrol is a no-op without the launch attribute): the
    # knob adds ``use_pdl`` to its launch and the epilogue-role wait.
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    config = helion.Config(
        block_sizes=[256, 128, 64],
        pid_type="persistent_blocked",
        indexing=["tensor_descriptor"] * 3,
        tcgen05_ab_stages=4,
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        cute_pdl=False,
    )
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    if normalized.config.get("tcgen05_cta_group") != "two":
        pytest.skip("two-CTA tile not admitted on this binding")
    plain = _source(bound, normalized)
    assert "'use_pdl': True" not in plain
    plain_body = _kernel_body(plain)
    assert plain_body.count(WAIT) == 1 and plain_body.count(RELEASE) == 1
    with bound.env:
        with_knob = bound.config_spec.normalized_config(
            helion.Config(**{**normalized.config, "cute_pdl": True})
        )
    source = _source(bound, with_knob)
    assert "'use_pdl': True" in source
    body = _kernel_body(source)
    # the TMA role's own wait plus the epilogue role's: every global read
    # sits behind a wait on its warp, so no entry wait is added; each wait
    # is followed by the release (one wave of CTA pairs)
    assert body.count(WAIT) == 2 and body.count(RELEASE) == 3
    _assert_no_entry_wait(body)
    _assert_release_follows_every_wait(body)
    assert _without_pdl_placement(body) == _without_pdl_placement(plain_body)


def _without_pdl_placement(body: str) -> list[str]:
    """The kernel body without what ``cute_pdl`` adds: the griddepcontrol
    lines and the split of stage 0's acquire ahead of the wait (the acquire
    line and the duplicated gate header)."""
    lines: list[str] = []
    for line in body.splitlines():
        if "griddepcontrol" in line:
            continue
        if "producer_acquire(tcgen05_ab_producer_state" in line:
            continue
        if lines and line.strip().startswith("if ") and lines[-1] == line:
            continue
        lines.append(line)
    return lines


def _device_scaled_gemm(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        acc = acc * scale_a[()] * scale_b[()]
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def test_persistent_kernel_binds_device_scalars_per_tile_and_waits_in_the_roles() -> (
    None
):
    # ``acc * scale[()]`` epilogues: the rank-0 scalars are read once per
    # output tile in the epilogue role, ahead of the accumulator wait, not
    # inside every subtile iteration after the TMEM load (that read sat on
    # the critical path: 7% of a 1024^3 fp8 GEMM). Their hoisted prologue
    # loads are dead after the splice and no longer force an entry wait.
    args = (
        torch.empty((1024, 1024), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((1024, 1024), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((), dtype=torch.float32, device=CPU_DEVICE),
        torch.empty((), dtype=torch.float32, device=CPU_DEVICE),
    )
    bound = helion.kernel(_device_scaled_gemm, backend="cute", static_shapes=True).bind(
        args
    )
    source = _source(bound, _persistent_config(cute_pdl=True))
    body = _kernel_body(source)
    assert "'use_pdl': True" in source
    assert body.count(WAIT) == 2 and body.count(RELEASE) == 3
    _assert_no_entry_wait(body)
    _assert_release_follows_every_wait(body)
    _assert_tma_role_waits_before_its_first_stage(body)
    lines = body.splitlines()
    waits = [i for i, line in enumerate(lines) if WAIT in line]
    for name in ("scale_a", "scale_b"):
        reads = [i for i, line in enumerate(lines) if f"{name}.iterator.load()" in line]
        assert len(reads) == 1, reads
        (read,) = reads
        assert lines[read].strip().startswith("tcgen05_scalar")
        assert (
            lines[read].strip().endswith(f"= cutlass.Float32({name}.iterator.load())")
        )
        # bound in the epilogue role, after its wait and ahead of the
        # per-subtile loop (and so ahead of the accumulator wait inside it)
        subtile_loop = next(
            i for i, line in enumerate(lines) if "for _tcgen05_subtile in" in line
        )
        assert waits[1] < read < subtile_loop
    chain = [line for line in lines if line.strip().startswith("tcgen05_chain_step")]
    assert len(chain) == 2
    assert chain[0].strip().endswith("* tcgen05_scalar")
    assert chain[1].strip().endswith("* tcgen05_scalar_1")
    # the plain scalar lowering of the loads is gone from the prologue
    assert not any(
        line.strip().startswith("load") and ".iterator.load()" in line for line in lines
    )
    # (``cute_pdl`` defaults to True on the sm_90+ tcgen05 surface; the
    # knob-off render pins it off.)
    plain = _kernel_body(_source(bound, _persistent_config(cute_pdl=False)))
    assert "griddepcontrol" not in plain
    assert plain.count("scale_a.iterator.load()") == 1


def test_simt_store_binds_device_scalars_ahead_of_the_accumulator_wait() -> None:
    # The SIMT store body (direct stores here; edge tiles, partial TMA
    # stores and column-major outputs take it too) binds the rank-0 scalars
    # in its per-tile setup ahead of the accumulator wait, like the
    # TMA-store body.
    args = (
        torch.empty((1024, 1024), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((1024, 1024), dtype=torch.float16, device=CPU_DEVICE),
        torch.empty((), dtype=torch.float32, device=CPU_DEVICE),
        torch.empty((), dtype=torch.float32, device=CPU_DEVICE),
    )
    bound = helion.kernel(_device_scaled_gemm, backend="cute", static_shapes=True).bind(
        args
    )
    body = _kernel_body(
        _source(bound, _persistent_config(tcgen05_c_store_mode="direct"))
    )
    assert "tcgen05_tma_store_atom" not in body
    lines = body.splitlines()
    bindings = [
        i
        for i, line in enumerate(lines)
        if line.strip().startswith("tcgen05_scalar") and ".iterator.load())" in line
    ]
    assert len(bindings) == 2
    wait = next(
        i
        for i, line in enumerate(lines)
        if "tcgen05_acc_pipeline.consumer_wait(" in line and i > bindings[0]
    )
    assert bindings[0] < bindings[1] < wait


def test_wait_coverage_analysis() -> None:
    from helion._compiler.cute.pdl_coverage import waits_cover_global_reads

    def covered(source: str) -> bool:
        return waits_cover_global_reads(ast.parse(source).body)

    # a top-level wait covers everything after it, role blocks included
    assert covered("""
x = 1
cute.arch.griddepcontrol_wait()
if tma_warp:
    cute.copy(tma_atom_a, tma_gA[None, 0], tma_sA[None, 0], tma_bar_ptr=bar)
""")
    # a wait inside one role block does not cover another role block
    assert not covered("""
if warp == 5:
    cute.arch.griddepcontrol_wait()
    cute.copy(tma_atom_a, tma_gA[None, 0], tma_sA[None, 0], tma_bar_ptr=bar)
if warp < 4:
    v = (bias.iterator + col).load()
""")
    assert covered("""
if warp == 5:
    cute.arch.griddepcontrol_wait()
    cute.copy(tma_atom_a, tma_gA[None, 0], tma_sA[None, 0], tma_bar_ptr=bar)
if warp < 4:
    cute.arch.griddepcontrol_wait()
    v = (bias.iterator + col).load()
""")
    # a prologue read ahead of every wait is uncovered; on-chip copies,
    # fragment loads and the descriptor prefetch are not global reads
    assert not covered("""
start = (offsets.iterator + b).load()
if warp == 5:
    cute.arch.griddepcontrol_wait()
""")
    assert covered("""
cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
acc = tRS_rAcc.load()
tRS_rD.store(acc)
cute.arch.prefetch_descriptor(tma_atom_a)
""")
    assert not covered("cute.arch.cp_async_16(slot, wm_a_g)")
    assert not covered(
        "cute.arch.atomic_add((c.iterator + 0).llvm_ptr, val=v, sem='relaxed')"
    )
    # a loop body: a wait inside covers only what follows it inside
    assert not covered("""
for k in cutlass.range(4):
    cute.copy(tma_atom_a, tma_gA[None, k], tma_sA[None, 0], tma_bar_ptr=bar)
    cute.arch.griddepcontrol_wait()
""")


def test_a_multi_wave_flat_grid_releases_from_its_last_wave() -> None:
    # ``auto`` past one wave: the flat kernel's 64x16 tiles of a 4096x4096
    # output are 16384 CTAs, one per SM at a time (the operand ring fills
    # the shared memory), so the release after the wait is guarded to the
    # CTAs of the last wave (``last_wave``); an explicit ``entry`` releases
    # from every CTA.
    bound = _bind_gemm((4096, 4096, 256), torch.float16)
    source = _source(bound, _flat_config(cute_pdl=True))
    body = _kernel_body(source)
    assert body.count(WAIT) == 1 and body.count(RELEASE) == 2
    _assert_no_entry_wait(body)
    with bound.env:
        num_sm = bound.config_spec.num_sm
    assert num_sm > 1
    lines = body.splitlines()
    wait_line = next(i for i, line in enumerate(lines) if WAIT in line)
    # the CTA index is the body's first statement, the guard follows the wait
    assert _body_statements(body)[0].strip().startswith("pdl_cta = ")
    guard = re.fullmatch(r"if pdl_cta >= (\d+):", lines[wait_line + 1].strip())
    assert guard is not None, lines[wait_line + 1]
    assert RELEASE in lines[wait_line + 2]
    assert _indent(lines[wait_line + 1]) == _indent(lines[wait_line])
    # the capacity is one CTA per SM: the grid minus the SM count
    grid = int(guard.group(1)) + num_sm
    assert grid > num_sm and grid % 64 == 0
    entry = _kernel_body(
        _source(bound, _flat_config(cute_pdl=True, cute_pdl_release="entry"))
    )
    assert "pdl_cta" not in entry
    _assert_release_follows_every_wait(entry)
    # ``cute_min_blocks_per_mp`` is the occupancy the config asked for
    paired = _kernel_body(
        _source(bound, _flat_config(cute_pdl=True, cute_min_blocks_per_mp=2))
    )
    assert f"if pdl_cta >= {grid - 2 * num_sm}:" in paired
