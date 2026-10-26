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

from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401
from test.test_cute_tcgen05_fixed_chain import _kernel_body
from test.test_cute_tcgen05_fixed_chain import _source

import helion
from helion._testing import skipUnlessBackends

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
        fragment = spec._cute_tcgen05_config.flat_fields()["cute_pdl"]
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
        assert not state._pdl_searched()
        fragment = state.flat_fields()["cute_pdl"]
        assert tuple(fragment.choices) == (False, True)  # pyrefly: ignore[missing-attribute]
        assert fragment.search_choices == (False,)  # pyrefly: ignore[missing-attribute]
        assert spec.default_config().config["cute_pdl"] is False


def test_seeds_are_not_doubled_and_normalization_carries_the_knob() -> None:
    # The knob is a searched field, not a seed family: no seed list grows a
    # twin, and a normalized config carries ``cute_pdl`` like every other
    # optional knob (so a flat transfer round-trips exactly).
    bound = _bind_gemm((1024, 1024, 1024), torch.float16)
    with bound.env:
        spec = bound.config_spec
        seeds = spec._cute_tcgen05_config.autotune_seed_configs()
        assert seeds and all(s.config.get("cute_pdl") is not True for s in seeds)
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
    # sits behind a wait on its warp, so no entry wait is added
    assert body.count(WAIT) == 2 and body.count(RELEASE) == 1
    _assert_no_entry_wait(body)
    assert [line for line in body.splitlines() if WAIT not in line] == [
        line for line in plain_body.splitlines() if WAIT not in line
    ]


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
