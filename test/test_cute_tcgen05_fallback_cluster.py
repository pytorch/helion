"""The 4-CTA pair cluster launched with its pair as the fallback
(``tcgen05_fallback_cluster``).

GB300 co-schedules 36 4-CTA clusters (144 of its 152 SMs); with a fallback
cluster the GPU forms pairs on the SMs left over.  The kernel outlines its
body into a ``@cute.jit`` function with a ``cutlass.Constexpr`` flag and calls
it once per cluster shape; the statements that depend on the physical cluster
select on the flag, both bodies walk the 2x2 schedule, and the two share their
shared memory through ``SmemAllocator`` allocations.  The source checks render
on the GB300 target; the CUDA checks compare against ``torch.mm``, on grids
with more 4-CTA slots than co-resident clusters and with every cluster forced
onto the fallback body.
"""

from __future__ import annotations

import functools
import linecache
import random
import re
import tempfile

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream
from cutlass.cute.runtime import make_ptr
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion._compiler.cute.tcgen05_config import repair
from helion._compiler.cute.tcgen05_config import surface
from helion._compiler.cute.tcgen05_constants import tcgen05_fallback_cluster_shape
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion._testing.cute_dense_tcgen05 import wrapper_plans
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
from helion.runtime.cute.launcher import _create_cute_wrapper

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

_PAIR_2X2 = {
    "block_sizes": [256, 256, 64],
    "l2_groupings": [8],
    "pid_type": "persistent_interleaved",
    "tcgen05_cluster_m": 2,
    "tcgen05_cluster_n": 2,
    "tcgen05_ab_stages": 6,
}
_SCHEDULES = {
    "static": {
        "tcgen05_persistence_model": "static_persistent",
        "tcgen05_strategy": "role_local_monolithic",
        "tcgen05_warp_spec_ab_load_warps": 2,
    },
    "clc": {
        "tcgen05_persistence_model": "clc_persistent",
        "tcgen05_strategy": "role_local_with_scheduler",
        "tcgen05_warp_spec_scheduler_warps": 1,
    },
}
_FALLBACK = {"tcgen05_fallback_cluster": True}


def _config(schedule: str, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(_PAIR_2X2 | _SCHEDULES[schedule] | overrides)


def _rr_args(
    m: int, n: int, k: int, device: object = "cpu"
) -> tuple[torch.Tensor, ...]:
    # (A K-major, B N-major: the RR layout of the 4096x50264x2048 target)
    return bf16_mm_args(m, n, k, rhs_k_major=False, device=device)


@functools.cache
def _code(schedule: str, fallback: bool = True, shape=(4096, 8192, 2048)) -> str:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_rr_args(*shape))
        return bound.to_code(_config(schedule, **(_FALLBACK if fallback else {})))


def test_the_knob_names_the_pair_of_a_4_cta_cluster() -> None:
    pair = {"tcgen05_cluster_m": 2, "tcgen05_cluster_n": 2}
    assert tcgen05_fallback_cluster_shape(pair | _FALLBACK) == (2, 1, 1)
    assert tcgen05_fallback_cluster_shape(pair) is None
    # a pair cluster has no smaller cluster to fall back to, and a K-split
    # cluster's pairs exchange partials through each other's shared memory
    assert tcgen05_fallback_cluster_shape({"tcgen05_cluster_m": 2} | _FALLBACK) is None
    assert (
        tcgen05_fallback_cluster_shape(pair | _FALLBACK | {"tcgen05_split_k": 2})
        is None
    )


@pytest.mark.parametrize("schedule", ["static", "clc"])
def test_one_body_per_cluster_shape(schedule: str) -> None:
    code = _code(schedule)
    params = (
        "a, b, out, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, "
        "tcgen05_tma_store_atom, tcgen05_tma_store_tensor"
    )
    fallback_params = (
        "a, b, out, tma_atom_a_fb, tma_tensor_a_fb, tma_atom_b_fb, tma_tensor_b_fb, "
        "tcgen05_tma_store_atom, tcgen05_tma_store_tensor"
    )
    # The kernel's body, outlined with the physical cluster as a Constexpr,
    # and the kernel dispatching on the cluster the GPU formed.
    assert (
        "@cute.jit\ndef _helion_plain_mm_into_body("
        f"{params}, tcgen05_fallback_cluster: cutlass.Constexpr):"
    ) in code
    assert (
        "    if cute.experimental.is_preferred_cluster((2, 2, 1)):\n"
        f"        _helion_plain_mm_into_body({params}, False)\n"
        "    else:\n"
        f"        _helion_plain_mm_into_body({fallback_params}, True)\n"
    ) in code
    assert re.search(
        r"@cute.kernel\ndef _helion_plain_mm_into\(a, b, out, tma_atom_a, tma_tensor_a, "
        r"tma_atom_b, tma_tensor_b, tma_atom_a_fb, tma_tensor_a_fb, tma_atom_b_fb, "
        r"tma_tensor_b_fb, ",
        code,
    )
    # The physical cluster behind the pipelines and multicast masks, the CTA
    # rank mask and the A/B ring's release count: a fallback pair's MMA alone
    # releases its stages (no A multicast from a second N lane).
    for line in (
        (
            "tcgen05_cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1) "
            "if tcgen05_fallback_cluster else (2, 2, 1)), (tiled_mma.thr_id.shape,))"
        ),
        (
            "tcgen05_cta_rank_in_cluster = cute.arch.make_warp_uniform("
            "cute.arch.block_idx_in_cluster()) & cutlass.Int32(1 if "
            "tcgen05_fallback_cluster else 3)"
        ),
        (
            "tcgen05_ab_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(1 if tcgen05_fallback_cluster "
            "else 2))"
        ),
    ):
        assert line in code
    # Both bodies keep the 2x2 schedule.
    assert "(2, 2, 1))" in code.split("PersistentTileSchedulerParams(")[1]
    # One shared memory for both bodies: allocator allocations only, the
    # static Int32 counts unwrapped.
    assert "cute.arch.alloc_smem(" not in code
    assert (
        "tcgen05_ab_pipeline_mbars = cutlass.utils.SmemAllocator()."
        "allocate_array(cutlass.Int64, 12)"
    ) in code
    assert (
        "smem_a = cutlass.utils.SmemAllocator().allocate_array(cutlass.BFloat16, "
        "cute.cosize(sA_layout.outer), byte_alignment=128)"
    ) in code
    assert "_helion_plain_mm_into._helion_cute_cluster_shape = (2, 2, 1)" in code
    assert (
        "_helion_plain_mm_into._helion_cute_fallback_cluster_shape = (2, 1, 1)" in code
    )
    # The fallback body's A/B atoms: no A multicast across the N lanes.
    ab_plans = [
        plan for plan in wrapper_plans(code) if plan["kind"] == "tcgen05_ab_tma"
    ]
    assert [(plan["cluster_n"], plan["kernel_args"]) for plan in ab_plans] == [
        (2, ["tma_atom_a", "tma_tensor_a", "tma_atom_b", "tma_tensor_b"]),
        (1, ["tma_atom_a_fb", "tma_tensor_a_fb", "tma_atom_b_fb", "tma_tensor_b_fb"]),
    ]


def test_the_static_grid_fills_every_sm() -> None:
    # 36 co-resident 4-CTA clusters cover 144 SMs; a slot per four SMs (38 on
    # GB300) puts pairs on the rest.
    assert "max(1, _NUM_SM // 4)" in _code("static")
    # (without the fallback: a slot per SM pair, two hardware waves of 4-CTA
    # clusters)
    assert "max(1, _NUM_SM // 2)" in _code("static", fallback=False)


def test_a_fallback_clc_leader_feeds_its_pair() -> None:
    code = _code("clc")
    # the scheduler mailbox's releases and publishes span the physical cluster
    assert (
        "tcgen05_sched_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup("
        "cutlass.pipeline.Agent.Thread, cutlass.Int32(384 if tcgen05_fallback_cluster "
        "else 768))"
    ) in code
    assert (
        code.count(
            "if tcgen05_clc_sched_peer_rank < cutlass.Int32(2 if "
            "tcgen05_fallback_cluster else 4):"
        )
        == 2
    )
    # a cancelled pair-sized unit decodes its own N lane of the 2x2 slot
    assert (
        "tcgen05_clc_initial_sched.cta_id_in_cluster = (cutlass.Int32(0), "
        "tcgen05_clc_bidy % cutlass.Int32(2), cutlass.Int32(0))"
    ) in code


def test_a_fallback_pair_splits_the_l2_prefetch_by_its_n_lane() -> None:
    # A one-wave grid's operand L2 prefetch (``cute_pdl``; a K-major A and an
    # N-major B: 1024x1024 on 32 CTAs): a 4-CTA cluster's two pairs split
    # the A rows that the multicast shares at the pair's rank in the cluster,
    # and a fallback body's pair at the N lane of its 2x2 slot, so the slot's
    # two pairs, each loading the A tile itself, still prefetch each row once.
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_rr_args(1024, 1024, 256))
        code = bound.to_code(
            _config("static", cute_pdl=True, tcgen05_ab_stages=4, **_FALLBACK)
        )
    assert (
        "tcgen05_l2_pf_rank_n = cutlass.Int32(cute.arch.block_idx()[1]) & "
        "cutlass.Int32(1) if tcgen05_fallback_cluster else "
        "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) >> "
        "cutlass.Int32(1) & cutlass.Int32(1)"
    ) in code
    assert "tcgen05_l2_pf_rank_n * cutlass.Int32(64)" in code


@pytest.mark.parametrize("schedule", ["static", "clc"])
def test_off_by_default(schedule: str) -> None:
    code = _code(schedule, fallback=False)
    assert "_body(" not in code
    assert "tcgen05_fallback_cluster" not in code
    assert "cute.arch.alloc_smem(" in code
    assert "SmemAllocator" not in code


@pytest.mark.parametrize(
    "schedule,overrides",
    [
        ("clc", {"tcgen05_cluster_n": 1}),
        (
            "static",
            {
                "tcgen05_split_k": 2,
                "block_sizes": [128, 128, 64],
                "tcgen05_warp_spec_ab_load_warps": 1,
            },
        ),
        # two 256-row A subtiles per stage (the M-paired 512-row tile)
        ("static", {"block_sizes": [512, 256, 64], "tcgen05_ab_stages": 3}),
    ],
    ids=["pair-cluster", "k-split-cluster", "m-pair-tile"],
)
def test_refused_off_the_4_cta_pair_cluster(
    schedule: str, overrides: dict[str, object]
) -> None:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_rr_args(4096, 8192, 2048))
        spec = bound.config_spec
        config = dict(_config(schedule, **_FALLBACK, **overrides).config)
        with pytest.raises(InvalidConfig, match="tcgen05_fallback_cluster=True needs"):
            spec.normalize(dict(config))
        # the search's repair turns a sampled knob off instead
        spec.normalize(config, _fix_invalid=True)
        assert config.get("tcgen05_fallback_cluster", False) is False


def test_the_search_offers_the_knob_beside_the_2x2_cluster() -> None:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_rr_args(4096, 8192, 2048))
        spec = bound.config_spec
        assert surface.fallback_cluster_search_enabled(spec._cute_tcgen05_config)
        gen = ConfigGeneration(spec)
        # About one sample in 75 turns the knob on. The sampler draws from the
        # process-wide ``random``: a fixed seed keeps the sample set from
        # depending on what earlier tests of this worker drew, and the state
        # is handed back afterwards.
        state = random.getstate()
        try:
            random.seed(0)
            samples = [
                dict(gen.unflatten(gen.random_flat()).config) for _ in range(400)
            ]
        finally:
            random.setstate(state)
        on = [s for s in samples if s.get("tcgen05_fallback_cluster") is True]
        assert on
        assert all(tcgen05_fallback_cluster_shape(s) == (2, 1, 1) for s in on)


def test_the_grid_cap_is_inert_under_the_fallback() -> None:
    # A fallback launch takes a slot per four SMs whatever
    # tcgen05_persistent_grid_cap: the search repair keeps one of the two
    # identical kernels, and an explicit cap beside the fallback still
    # validates.
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_rr_args(4096, 8192, 2048))
        spec = bound.config_spec
        state = spec._cute_tcgen05_config
        explicit = dict(
            _config(
                "static", **_FALLBACK, tcgen05_persistent_grid_cap="resident_clusters"
            ).config
        )
        spec.normalize(dict(explicit))
        sampled = dict(explicit)
        repair.fix_search_config(state, sampled)
        assert sampled["tcgen05_fallback_cluster"] is True
        assert sampled.get("tcgen05_persistent_grid_cap", "sm_pairs") == "sm_pairs"
        # (without the fallback the cap stays a live choice)
        live = dict(explicit, tcgen05_fallback_cluster=False)
        repair.fix_search_config(state, live)
        assert live["tcgen05_persistent_grid_cap"] == "resident_clusters"


def _seeds(m: int, n: int, k: int) -> list[dict[str, object]]:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_rr_args(m, n, k))
        with bound.env:
            gen = ConfigGeneration(bound.config_spec)
            return [dict(config.config) for _, config in gen.seed_flat_config_pairs()]


def test_the_clc_2x2_seeds_take_the_fallback_on_a_grid_past_the_clusters() -> None:
    # 50264x4096 (the transposed lowering of the 4096x50264x2048 RR target)
    # on 256x256 pairs: 197 x 8 = 1576 2x2 slots, past the 38 a slot per four
    # of GB300's 152 SMs gives: each CLC 2x2 seed gains a twin with the
    # fallback.
    seeds = _seeds(50264, 4096, 2048)
    clc_2x2 = [
        seed
        for seed in seeds
        if seed.get("tcgen05_persistence_model") == "clc_persistent"
        and (seed.get("tcgen05_cluster_m"), seed.get("tcgen05_cluster_n")) == (2, 2)
    ]
    fallback = [seed for seed in clc_2x2 if seed.get("tcgen05_fallback_cluster")]
    assert fallback
    for seed in fallback:
        twin = {**seed}
        twin.pop("tcgen05_fallback_cluster")
        assert twin in clc_2x2 or {**twin, "tcgen05_fallback_cluster": False} in clc_2x2
    # (a static schedule hands the pairs a slot's share: not seeded)
    assert not any(
        seed.get("tcgen05_fallback_cluster")
        and seed.get("tcgen05_persistence_model") != "clc_persistent"
        for seed in seeds
    )
    # 2048x2048: 8 x 4 = 32 slots, which the co-resident 4-CTA clusters run
    assert not any(
        seed.get("tcgen05_fallback_cluster") for seed in _seeds(2048, 2048, 4096)
    )


def _offline_ptx(code: str) -> str:
    """PTX of a rendered static 2x2 kernel compiled offline through the launcher."""
    namespace: dict[str, object] = {}
    path = f"<fallback_cluster_{hash(code)}>"
    linecache.cache[path] = (len(code), None, code.splitlines(keepends=True), path)
    exec(compile(code, path, "exec"), namespace)
    kernel = namespace["_helion_plain_mm_into"]
    m, n, k = 4096, 8192, 2048
    schema = (
        ("tensor", torch.bfloat16, 2, (m, k), (k, 1)),
        ("tensor", torch.bfloat16, 2, (k, n), (n, 1)),
        ("tensor", torch.bfloat16, 2, (m, n), (n, 1)),
    )
    with cpu_target(forbid_native=False):
        launch = _create_cute_wrapper(kernel, schema, (32, 7, 1), num_sm=152)
        pointers = [
            make_ptr(
                cutlass.BFloat16,
                0x100000 * (i + 1),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory(prefix="fallback_cluster_") as directory:
            ptx = cute.compile(
                launch,
                *pointers,
                cutlass.Int32(2),
                cutlass.Int32(2),
                cutlass.Int32(38),
                make_fake_stream(),
                options=f"--dump-dir {directory} --keep-ptx",
            ).__ptx__
    assert ptx is not None
    return ptx


_TMA_LOAD = re.compile(r"cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global\.[^\s]*")


def test_the_two_bodies_compile_into_one_shared_memory() -> None:
    # Two static copies of the 6-stage 256x256 ring (2 x 192 KiB) do not fit
    # (libNVVM fails); the allocator's allocations overlap.
    ptx = _offline_ptx(_code("static"))
    single = _offline_ptx(_code("static", fallback=False))
    loads = _TMA_LOAD.findall(ptx)
    single_loads = _TMA_LOAD.findall(single)
    multicast = [form for form in loads if "multicast::cluster" in form]
    single_multicast = [form for form in single_loads if "multicast::cluster" in form]
    # The preferred body keeps the A multicast of the one-body kernel; the
    # fallback body adds its loads, A included, without a multicast.
    assert multicast and len(multicast) == len(single_multicast)
    assert len(loads) - len(multicast) > len(single_loads) - len(single_multicast)
    assert not torch.cuda.is_initialized()


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("schedule", ["static", "clc"])
def test_matches_torch_mm_with_more_slots_than_4_cta_clusters(schedule: str) -> None:
    if not _sm100():
        pytest.skip("tcgen05 needs an SM100-family GPU")
    # 4096x8192 on 256x256 pairs: 256 2x2 slots, more than the co-resident
    # 4-CTA clusters (36 on GB300), so pairs run some of them.
    num_sm = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    if helion.runtime.get_max_active_clusters(torch.device(DEVICE), 4) * 4 >= num_sm:
        pytest.skip("the co-resident 4-CTA clusters cover every SM: no pair forms")
    a, b, out = _rr_args(4096, 8192, 1024, device=DEVICE)
    bound = plain_mm_into.bind((a, b, out))
    bound.set_config(_config(schedule, **_FALLBACK))
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=2e-2, rtol=2e-2)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("schedule", ["static", "clc"])
def test_the_fallback_body_alone_matches_torch_mm(schedule: str) -> None:
    if not _sm100():
        pytest.skip("tcgen05 needs an SM100-family GPU")
    # Launched with the pair as its only cluster shape, every CTA runs the
    # fallback body: each 2x2 slot's two lanes as two pair clusters.  The
    # pair-only launch runs a module of its own (the rendered source with the
    # pair as the kernel's cluster): PyCodeCache keys modules by source text,
    # and the rendered source's kernel, which every binding of this config
    # shares, keeps its 2x2 launch and its launch caches.
    a, b, out = _rr_args(2048, 4096, 1024, device=DEVICE)
    code = plain_mm_into._bind_isolated((a, b, out)).to_code(
        _config(schedule, **_FALLBACK)
    )
    preferred = "_helion_plain_mm_into._helion_cute_cluster_shape = (2, 2, 1)\n"
    assert code.count(preferred) == 1
    assert (
        "_helion_plain_mm_into._helion_cute_fallback_cluster_shape = (2, 1, 1)\n"
    ) in code
    module = PyCodeCache.load(
        code.replace(preferred, preferred.replace("(2, 2, 1)", "(2, 1, 1)"))
    )
    out.fill_(float("nan"))
    module.plain_mm_into(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=2e-2, rtol=2e-2)
