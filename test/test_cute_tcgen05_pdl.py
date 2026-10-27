"""Codegen pins for ``cute_pdl`` on the tcgen05 GEMM kernels.

CPU-only (fake CUDA target, the ``test_cute_tcgen05_fixed_chain`` binders and
fixture): the knob launches the kernel as a programmatic dependent of the
previous kernel in the stream, waits on that kernel at the top of the body
(every warp, ahead of every global read of the prologue, the root body and
the pipelines, on every kernel shape) and releases the dependents once the
MMA warp has issued its last MMA.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401
from test.test_cute_tcgen05_fixed_chain import _kernel_body
from test.test_cute_tcgen05_fixed_chain import _source

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

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


def test_knob_is_searched_on_sm100_matmul_kernels() -> None:
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    with bound.env:
        spec = bound.config_spec
        fields = spec.flat_config(lambda x: x.default()).config
        assert fields["cute_pdl"] is False
        fragment = spec._cute_tcgen05_config.flat_fields()["cute_pdl"]
    assert tuple(fragment.choices) == (False, True)  # pyrefly: ignore[missing-attribute]


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
        seeds = tcgen05.autotune_seed_configs()
        assert seeds and all(s.config.get("cute_pdl") is True for s in seeds)
        schedules = [
            {k: v for k, v in s.config.items() if k != "cute_pdl"} for s in seeds
        ]
        assert len(schedules) == len(seeds)
        assert all(schedules.count(schedule) == 1 for schedule in schedules), (
            "a seed list must not grow a cute_pdl twin"
        )
        with patch.object(tcgen05, "_pdl_searched", return_value=False):
            unsearched = tcgen05.autotune_seed_configs()
        assert [
            {k: v for k, v in s.config.items() if k != "cute_pdl"} for s in unsearched
        ] == schedules
        assert all("cute_pdl" not in s.config for s in unsearched)
        normalized = spec.normalized_config(
            _persistent_config(loop_orders=[[0, 1]], l2_groupings=[1])
        )
        assert normalized.config["cute_pdl"] is False
        generation = spec.create_config_generation()
        roundtrip = generation.unflatten(generation.flatten(normalized))
    # (``cute_min_blocks_per_mp`` is the launch-bounds heuristic's field; this
    # binding does not register it, so the flat layout alone materializes it.)
    roundtrip.config.pop("cute_min_blocks_per_mp", None)
    assert roundtrip == normalized


def test_flat_kernel_waits_before_the_first_load_and_releases_after_the_mma() -> None:
    bound = _bind_gemm((256, 256, 256), torch.float16)
    plain = _source(bound, _flat_config())
    assert "griddepcontrol" not in plain
    assert "'use_pdl': True" not in plain
    source = _source(bound, _flat_config(cute_pdl=True))
    body = _kernel_body(source)
    assert "'use_pdl': True" in source
    assert body.count(WAIT) == 1 and body.count(RELEASE) == 1
    # Every warp passes the wait after the prologue (TMEM allocation,
    # barrier init, descriptor prefetch) and before the first TMA load; the
    # body has no global read ahead of it, so no entry wait is added.
    assert body.index("prefetch_descriptor(") < body.index(WAIT)
    assert body.index(WAIT) < body.index("cute.copy(tma_atom_a")
    _assert_no_entry_wait(body)
    # The MMA warp releases the dependents right after committing its
    # accumulator, ahead of the epilogue's CTA-wide syncs.
    commit = body.index("producer_commit(tcgen05_acc_producer_state)")
    assert commit < body.index(RELEASE) < body.index("tcgen05_kernel_desc")
    lines = body.splitlines()
    release_line = next(i for i, line in enumerate(lines) if RELEASE in line)
    assert lines[release_line - 1].strip() == "if tcgen05_exec_active:"
    # The knob-off body is otherwise unchanged.
    stripped = [
        line
        for index, line in enumerate(lines)
        if "griddepcontrol" not in line and index != release_line - 1
    ]
    assert "\n".join(stripped) == _kernel_body(plain)


def test_persistent_kernel_waits_in_the_tma_and_epilogue_roles() -> None:
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    source = _source(bound, _persistent_config(cute_pdl=True))
    body = _kernel_body(source)
    assert "'use_pdl': True" in source
    assert body.count(WAIT) == 2 and body.count(RELEASE) == 1
    lines = body.splitlines()
    waits = [i for i, line in enumerate(lines) if WAIT in line]
    roles = [lines[i - 1].strip() for i in waits]
    # The TMA-load warp (warp 5) and the epilogue warps (0-3) wait at the
    # top of their role blocks; the MMA warp never touches global memory,
    # and the prologue reads nothing global, so no entry wait is added.
    assert roles == [
        "if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5):",
        "if cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4):",
    ]
    assert waits[0] < body.index("cute.copy(tma_atom_a")
    _assert_no_entry_wait(body)
    # The release follows the MMA role's tile loop, in the teardown.
    assert body.index(RELEASE) > body.index("tcgen05_role_local_2_tile_sched")
    plain = _kernel_body(_source(bound, _persistent_config()))
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


def test_the_entry_wait_path_leaves_the_release_to_the_mma_warp() -> None:
    # The generic ``cute_pdl`` path owns a release placement
    # (``cute_pdl_release``); a tcgen05 kernel that gets the generic entry
    # wait must not also get that release: its MMA warp releases the
    # dependents after its last MMA, and an earlier one (``auto`` resolving
    # to ``entry`` on this one-wave persistent grid, or an explicit
    # placement) would make the dependents' CTAs resident during the GEMM,
    # which the late placement exists to avoid.  Every placement renders
    # the same kernel: one wait at the top, one release, the family's.
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    sources: list[str] = []
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
        assert body.count(RELEASE) == 1
        statements = _body_statements(body)
        wait_index = next(i for i, line in enumerate(statements) if WAIT in line)
        release_index = next(i for i, line in enumerate(statements) if RELEASE in line)
        # Not the generic entry release (the statement right after the wait)
        # nor the generic exit release (a top-level last statement): the
        # family's, under the MMA warp's role predicate.
        assert release_index != wait_index + 1
        release_line = statements[release_index]
        assert len(release_line) - len(release_line.lstrip()) > len(
            statements[wait_index]
        ) - len(statements[wait_index].lstrip())
        assert statements[release_index - 1].strip().startswith("if ")
        sources.append(source)
    assert len(set(sources)) == 1


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
    # sits behind a wait on its warp, so no entry wait is added
    assert body.count(WAIT) == 2 and body.count(RELEASE) == 1
    _assert_no_entry_wait(body)
    assert [line for line in body.splitlines() if WAIT not in line] == [
        line for line in plain_body.splitlines() if WAIT not in line
    ]


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
    assert body.count(WAIT) == 2 and body.count(RELEASE) == 1
    _assert_no_entry_wait(body)
    lines = body.splitlines()
    waits = [i for i, line in enumerate(lines) if WAIT in line]
    assert [lines[i - 1].strip() for i in waits] == [
        "if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5):",
        "if cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4):",
    ]
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
    plain = _kernel_body(_source(bound, _persistent_config()))
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
